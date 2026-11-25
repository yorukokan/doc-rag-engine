import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient, models

from chunker import chunk_file, load_and_chunk_folder
from retriever import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    _qdrant_lock,
    get_embedder,
    scroll_all_points,
)

DATA_DIR = "data"
ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILENAME_LENGTH = 120


def validate_filename(filename: str) -> str:
    """Dosya adını path traversal, kontrol karakteri ve uzantı açısından doğrular."""
    if not isinstance(filename, str):
        raise ValueError("Geçersiz dosya adı.")

    cleaned = filename.strip()
    if not cleaned:
        raise ValueError("Dosya adı boş olamaz.")
    if len(cleaned) > MAX_FILENAME_LENGTH:
        raise ValueError(
            f"Dosya adı en fazla {MAX_FILENAME_LENGTH} karakter olabilir."
        )
    if (
        "/" in cleaned
        or "\\" in cleaned
        or ".." in cleaned
        or "\x00" in cleaned
        or any(ord(c) < 32 for c in cleaned)
    ):
        raise ValueError("Dosya adı geçersiz karakter veya yol ifadesi içeriyor.")
    if cleaned.startswith("."):
        raise ValueError("Gizli dosya adlarına izin verilmez.")

    ext = os.path.splitext(cleaned)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Desteklenmeyen dosya uzantısı: '{ext or '(yok)'}'. İzin verilenler: {allowed_str}"
        )

    return cleaned


def make_point_id(source: str, chunk_id: int | str) -> str:
    """Kaynak adı ve chunk kimliğinden deterministik UUID üretir."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_id}"))


def _resolve_safe_path(filename: str, data_dir: str = DATA_DIR) -> Path:
    """Dosya adını doğrular ve hedef yolun data_dir içinde kaldığını garanti eder."""
    safe_name = validate_filename(filename)
    base = Path(data_dir).resolve()
    target = (base / safe_name).resolve()
    if target.parent != base:
        raise ValueError("Geçersiz dosya yolu.")
    return target


def _to_vector_list(vectors) -> list[list[float]]:
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def list_documents(
    data_dir: str = DATA_DIR,
    qdrant_path: str = QDRANT_PATH,
) -> list[dict]:
    """data_dir içindeki desteklenen dokümanları ve Qdrant chunk sayılarını listeler."""
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        return []

    doc_files = [
        p
        for p in sorted(data_path.iterdir(), key=lambda item: item.name)
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    counts: Counter[str] = Counter()
    if Path(qdrant_path).exists():
        with _qdrant_lock:
            client = QdrantClient(path=qdrant_path)
            try:
                if client.collection_exists(COLLECTION_NAME):
                    points = scroll_all_points(client, COLLECTION_NAME)
                    for point in points:
                        payload = point.payload or {}
                        src = payload.get("source")
                        if src:
                            counts[src] += 1
            finally:
                client.close()

    return [
        {
            "filename": doc_file.name,
            "size_bytes": doc_file.stat().st_size,
            "extension": doc_file.suffix.lower(),
            "chunk_count": counts.get(doc_file.name, 0),
        }
        for doc_file in doc_files
    ]


def index_single_file(
    file_path: str,
    source_name: str,
    qdrant_path: str = QDRANT_PATH,
) -> int:
    """Tek bir dosyayı parçalar, vektörleştirir ve Qdrant'a upsert eder."""
    safe_name = validate_filename(source_name)
    chunks = chunk_file(file_path, source_name=safe_name)
    if not chunks:
        raise ValueError("Dokümandan çıkarılabilir metin bulunamadı.")

    embedder = get_embedder()
    vectors = _to_vector_list(embedder.embed([c["text"] for c in chunks]))

    with _qdrant_lock:
        client = QdrantClient(path=qdrant_path)
        try:
            if not client.collection_exists(COLLECTION_NAME):
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=len(vectors[0]),
                        distance=models.Distance.COSINE,
                    ),
                )
            else:
                client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source",
                                    match=models.MatchValue(value=safe_name),
                                )
                            ]
                        )
                    ),
                )

            points = [
                models.PointStruct(
                    id=make_point_id(safe_name, chunk["chunk_id"]),
                    vector=vector,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "source": safe_name,
                        "section": chunk["metadata"]["section"],
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            client.upsert(collection_name=COLLECTION_NAME, points=points)
        finally:
            client.close()

    return len(chunks)


def save_and_index_document(
    filename: str,
    content: bytes,
    overwrite: bool = False,
    data_dir: str = DATA_DIR,
    qdrant_path: str = QDRANT_PATH,
) -> dict:
    """Dokümanı doğrular, diske kaydeder ve Qdrant'a indeksler; hata olursa diski geri alır."""
    safe_name = validate_filename(filename)
    if not content:
        raise ValueError("Dosya içeriği boş olamaz.")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise OverflowError(
            f"Dosya boyutu en fazla {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB olabilir."
        )

    target_path = _resolve_safe_path(safe_name, data_dir)
    existed = target_path.exists()
    if existed and not overwrite:
        raise FileExistsError(f"'{safe_name}' zaten mevcut.")

    ext = os.path.splitext(safe_name)[1].lower()
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = chunk_file(tmp_path, source_name=safe_name)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Dosya okunamadı veya bozuk formatta.") from exc

        if not chunks:
            raise ValueError("Dokümandan çıkarılabilir metin bulunamadı.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    backup_bytes = target_path.read_bytes() if existed else None
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)

    try:
        chunk_count = index_single_file(
            str(target_path),
            source_name=safe_name,
            qdrant_path=qdrant_path,
        )
    except Exception:
        if backup_bytes is not None:
            target_path.write_bytes(backup_bytes)
        else:
            target_path.unlink(missing_ok=True)
        raise

    return {
        "filename": safe_name,
        "size_bytes": len(content),
        "chunk_count": chunk_count,
        "overwritten": existed,
    }


def delete_document(
    filename: str,
    data_dir: str = DATA_DIR,
    qdrant_path: str = QDRANT_PATH,
) -> dict:
    """Dokümanı hem Qdrant indeksinden hem de diskten siler."""
    target_path = _resolve_safe_path(filename, data_dir)
    safe_name = target_path.name
    if not target_path.exists():
        raise FileNotFoundError(f"'{safe_name}' bulunamadı.")

    deleted_chunks = 0
    if Path(qdrant_path).exists():
        with _qdrant_lock:
            client = QdrantClient(path=qdrant_path)
            try:
                if client.collection_exists(COLLECTION_NAME):
                    source_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=safe_name),
                            )
                        ]
                    )
                    count_result = client.count(
                        collection_name=COLLECTION_NAME,
                        count_filter=source_filter,
                        exact=True,
                    )
                    deleted_chunks = int(count_result.count)
                    if deleted_chunks > 0:
                        client.delete(
                            collection_name=COLLECTION_NAME,
                            points_selector=models.FilterSelector(
                                filter=source_filter
                            ),
                        )
            finally:
                client.close()

    target_path.unlink()
    return {
        "filename": safe_name,
        "deleted_chunks": deleted_chunks,
        "deleted": True,
    }


def index_documents(
    data_dir: str = DATA_DIR,
    qdrant_path: str = QDRANT_PATH,
) -> int:
    """Tüm data_dir klasörünü sıfırdan indeksler (CLI kullanımına yöneliktir)."""
    chunks = load_and_chunk_folder(data_dir)
    if not chunks:
        print(f"{data_dir}/ klasöründe hiç doküman bulunamadı!")
        return 0

    print(f"{len(chunks)} adet chunk bulundu. Vektöre çevriliyor...")

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    vectors = _to_vector_list(embedder.embed(texts))

    with _qdrant_lock:
        client = QdrantClient(path=qdrant_path)
        try:
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)

            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),
                    distance=models.Distance.COSINE,
                ),
            )

            points = [
                models.PointStruct(
                    id=make_point_id(chunk["metadata"]["source"], chunk["chunk_id"]),
                    vector=vector,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "source": chunk["metadata"]["source"],
                        "section": chunk["metadata"]["section"],
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]

            client.upsert(collection_name=COLLECTION_NAME, points=points)
        finally:
            client.close()

    print(
        f"Başarılı! {len(points)} kayıt Qdrant veritabanına ({qdrant_path}/) yazıldı."
    )
    return len(points)


if __name__ == "__main__":
    index_documents()
