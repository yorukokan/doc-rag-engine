import logging
import re
import threading
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

COLLECTION_NAME = "dokumanlar"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_PATH = "qdrant_data"

_embedder = None
_embedder_lock = threading.Lock()
_qdrant_lock = threading.Lock()


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def get_qdrant_health() -> dict:
    with _qdrant_lock:
        client = None
        try:
            client = QdrantClient(path=QDRANT_PATH)
            collection_exists = bool(client.collection_exists(COLLECTION_NAME))
            points_count = None
            if collection_exists:
                info = client.get_collection(COLLECTION_NAME)
                points_count = info.points_count
            return {
                "qdrant_connected": True,
                "collection_name": COLLECTION_NAME,
                "collection_exists": collection_exists,
                "points_count": points_count,
            }
        except Exception as exc:
            logger.warning("Qdrant sağlık kontrolü başarısız: %s", exc)
            return {
                "qdrant_connected": False,
                "collection_name": COLLECTION_NAME,
                "collection_exists": False,
                "points_count": None,
            }
        finally:
            if client is not None:
                client.close()


def scroll_all_points(
    client,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 1000,
    scroll_filter: models.Filter | None = None,
) -> list:
    """Koleksiyondaki tüm point'leri sayfalama ile eksiksiz çeker."""
    all_points = []
    offset = None
    while True:
        scroll_kwargs = {
            "collection_name": collection_name,
            "limit": batch_size,
            "offset": offset,
            "with_payload": True,
        }
        if scroll_filter is not None:
            scroll_kwargs["scroll_filter"] = scroll_filter
        batch, next_offset = client.scroll(**scroll_kwargs)
        all_points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return all_points


def tokenize(text: str) -> list[str]:
    # Noktalama işaretlerini at, küçük harfe çevirip kelimelere ayır
    return re.findall(r"\w+", text.lower())


def search(
    query: str,
    top_k: int = 2,
    rrf_k: int = 10,
    use_rerank: bool = False,
    candidate_k: int = 6,
    source_filter: list[str] | None = None,
) -> list[dict]:
    qdrant_filter = None
    if source_filter is not None:
        cleaned = [s.strip() for s in source_filter if isinstance(s, str) and s.strip()]
        if not cleaned:
            return []
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchAny(any=cleaned),
                )
            ]
        )

    embedder = get_embedder()
    query_vector = list(embedder.embed([query]))[0].tolist()

    with _qdrant_lock:
        client = QdrantClient(path=QDRANT_PATH)
        try:
            if not client.collection_exists(COLLECTION_NAME):
                return []
            # 1. Vektör Araması (Anlamsal)
            query_kwargs = {
                "collection_name": COLLECTION_NAME,
                "query": query_vector,
                "limit": 20,
            }
            if qdrant_filter is not None:
                query_kwargs["query_filter"] = qdrant_filter
            dense_response = client.query_points(**query_kwargs)
            # 2. BM25 için koleksiyondaki tüm kartları çek
            if qdrant_filter is not None:
                all_points = scroll_all_points(
                    client,
                    COLLECTION_NAME,
                    scroll_filter=qdrant_filter,
                )
            else:
                all_points = scroll_all_points(client, COLLECTION_NAME)
        finally:
            client.close()

    if not all_points:
        return []

    # Her kartı ID'sine göre sakla
    docs_by_id = {p.payload["chunk_id"]: p.payload for p in all_points}

    # --- Vektör Sıralaması ve Ham Skorları (1, 2, 3...) ---
    dense_ranks = {
        hit.payload["chunk_id"]: rank
        for rank, hit in enumerate(dense_response.points, start=1)
    }
    dense_score_map = {
        hit.payload["chunk_id"]: hit.score
        for hit in dense_response.points
    }

    # --- BM25 Sıralaması ve Ham Skorları (Anahtar Kelime) ---
    chunk_ids = [p.payload["chunk_id"] for p in all_points]
    tokenized_corpus = [tokenize(p.payload["text"]) for p in all_points]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_score_map = dict(zip(chunk_ids, bm25_scores))

    bm25_sorted = sorted(
        zip(chunk_ids, bm25_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    bm25_ranks = {
        cid: rank for rank, (cid, _) in enumerate(bm25_sorted, start=1)
    }

    # --- RRF Birleştirme (Vektör + BM25) ---
    rrf_scores = {}
    for cid in docs_by_id:
        d_rank = dense_ranks.get(cid, len(docs_by_id) + 1)
        b_rank = bm25_ranks.get(cid, len(docs_by_id) + 1)
        rrf_scores[cid] = (1 / (rrf_k + d_rank)) + (1 / (rrf_k + b_rank))

    # En yüksek RRF skoruna göre adayları seç (Rerank açıksa önce daha geniş havuz al)
    fetch_k = max(top_k, candidate_k) if use_rerank else top_k
    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:fetch_k]

    results = []
    for cid in top_ids:
        payload = docs_by_id[cid]
        results.append({
            "score": round(rrf_scores[cid], 4),
            "dense_rank": dense_ranks.get(cid),
            "dense_score": round(float(dense_score_map.get(cid, 0.0)), 4),
            "bm25_rank": bm25_ranks.get(cid),
            "bm25_score": round(float(bm25_score_map.get(cid, 0.0)), 4),
            "chunk_id": cid,
            "text": payload["text"],
            "source": payload["source"],
            "section": payload["section"],
        })

    if use_rerank:
        from reranker import rerank_chunks
        return rerank_chunks(query, results, top_k=top_k)

    return results


if __name__ == "__main__":
    soru = "DNS önbellek süresi düşük olursa ne olur?"
    print(f"Soru: {soru}\n" + "=" * 50)
    for i, item in enumerate(search(soru, top_k=2), start=1):
        print(f"{i}. Sonuç | RRF Skoru: {item['score']} | Bölüm: {item['section']}")
