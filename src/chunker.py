import os
import re
from pathlib import Path


def _tail_overlap(text: str, overlap_chars: int) -> str:
    """Metnin sonundan kelime ortasını bölmeden yaklaşık overlap_chars kadar parça alır."""
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text.strip()
    tail = text[-overlap_chars:]
    # İlk boşluktan sonrasını al ki kelime yarım kesilmesin
    first_space = tail.find(" ")
    if first_space != -1 and first_space < len(tail) - 1:
        tail = tail[first_space + 1 :]
    return tail.strip()


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Tek bir uzun paragrafı kelime sınırlarından bölerek overlap ile parçalar."""
    parts = []
    start = 0
    step = max(1, max_chars - overlap_chars)

    while start < len(text):
        end = start + max_chars
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
        # Kelime ortasından başlamamak için bir sonraki boşluğa kay
        next_space = text.find(" ", start, end)
        if next_space != -1:
            start = next_space + 1
    return parts


def split_body_with_overlap(
    body: str,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[str]:
    """Metni paragraf yapısını koruyarak max_chars sınırında ve overlap ile böler."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        return []

    pieces: list[str] = []
    current_text = ""

    for para in paragraphs:
        # Tek bir paragraf bile max_chars'tan uzunsa (ör. PDF metinleri) güvenli böl
        if len(para) > max_chars:
            if current_text:
                pieces.append(current_text)
                current_text = ""
            pieces.extend(_split_long_text(para, max_chars, overlap_chars))
            continue

        if not current_text:
            current_text = para
        elif len(current_text) + len(para) + 2 <= max_chars:
            current_text = f"{current_text}\n\n{para}"
        else:
            pieces.append(current_text)
            overlap = _tail_overlap(current_text, overlap_chars)
            if overlap and len(overlap) + len(para) + 2 <= max_chars:
                current_text = f"{overlap}\n\n{para}"
            else:
                current_text = para

    if current_text:
        pieces.append(current_text)

    return pieces


def chunk_markdown(
    file_path: Path,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[dict]:
    # 1. Dosyayı oku
    text = file_path.read_text(encoding="utf-8")

    # 2. Metni #, ## veya ### başlıklarının başladığı yerlerden böl
    sections = re.split(r"(?m)^(?=#{1,3}\s+)", text)

    chunks = []
    chunk_index = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # İlk satırı başlık (heading), kalanını gövde (body) olarak ayır
        lines = section.splitlines()
        heading = lines[0].lstrip("#").strip() if lines[0].startswith("#") else "Genel"
        body = "\n".join(lines[1:]).strip() if lines[0].startswith("#") else section

        if not body:
            continue

        for piece in split_body_with_overlap(body, max_chars=max_chars, overlap_chars=overlap_chars):
            chunks.append({
                "chunk_id": f"{file_path.stem}_{chunk_index}",
                "text": f"{heading}\n{piece}",
                "metadata": {
                    "source": file_path.name,
                    "section": heading,
                },
            })
            chunk_index += 1

    return chunks


def chunk_text(
    file_path: Path,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[dict]:
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    chunks = []
    for idx, piece in enumerate(split_body_with_overlap(text, max_chars=max_chars, overlap_chars=overlap_chars)):
        chunks.append({
            "chunk_id": f"{file_path.stem}_{idx}",
            "text": piece,
            "metadata": {
                "source": file_path.name,
                "section": f"Bölüm {idx + 1}",
            },
        })
    return chunks


def chunk_pdf(
    file_path: Path,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    chunks = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        section_name = f"Sayfa {page_num}"
        for piece in split_body_with_overlap(page_text, max_chars=max_chars, overlap_chars=overlap_chars):
            chunks.append({
                "chunk_id": f"{file_path.stem}_{chunk_index}",
                "text": f"{section_name}\n{piece}",
                "metadata": {
                    "source": file_path.name,
                    "section": section_name,
                },
            })
            chunk_index += 1

    return chunks


def chunk_file(
    file_path: str | Path,
    source_name: str | None = None,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[dict]:
    """Tek bir .md, .txt veya .pdf dosyasını uzantısına göre parçalar."""
    ext = os.path.splitext(source_name or str(file_path))[1].lower()
    path = Path(file_path)

    if ext == ".md":
        chunks = chunk_markdown(path, max_chars=max_chars, overlap_chars=overlap_chars)
    elif ext == ".txt":
        chunks = chunk_text(path, max_chars=max_chars, overlap_chars=overlap_chars)
    elif ext == ".pdf":
        chunks = chunk_pdf(path, max_chars=max_chars, overlap_chars=overlap_chars)
    else:
        raise ValueError(f"Desteklenmeyen dosya uzantısı: {ext}")

    if source_name:
        stem = Path(source_name).stem
        for idx, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{stem}_{idx}"
            chunk["metadata"]["source"] = source_name

    return chunks


def load_and_chunk_folder(
    data_dir: str = "data",
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[dict]:
    # data/ klasöründeki tüm .md, .txt ve .pdf dosyalarını gez ve parçala
    all_chunks = []
    for file_path in sorted(Path(data_dir).iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".md":
            all_chunks.extend(chunk_markdown(file_path, max_chars=max_chars, overlap_chars=overlap_chars))
        elif suffix == ".txt":
            all_chunks.extend(chunk_text(file_path, max_chars=max_chars, overlap_chars=overlap_chars))
        elif suffix == ".pdf":
            all_chunks.extend(chunk_pdf(file_path, max_chars=max_chars, overlap_chars=overlap_chars))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_folder("data")
    print(f"Toplam {len(chunks)} chunk üretildi.\n")
    for c in chunks:
        print(f"ID: {c['chunk_id']} | Kaynak: {c['metadata']['source']} | Bölüm: {c['metadata']['section']}")
        print(f"Önizleme: {c['text'][:100]}...\n" + "-" * 50)
