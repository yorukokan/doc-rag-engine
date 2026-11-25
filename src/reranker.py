import threading
from fastembed.rerank.cross_encoder import TextCrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-v2-m3-int8"

_reranker = None
_reranker_lock = threading.Lock()


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Hibrit aramadan gelen aday kartları Cross-Encoder ile soruyla yan yana okuyup yeniden sıralar."""
    if not chunks:
        return []

    reranker = get_reranker()
    documents = [c["text"] for c in chunks]
    scores = list(reranker.rerank(query, documents))

    reranked = []
    for rrf_rank, (chunk, score) in enumerate(zip(chunks, scores), start=1):
        updated = dict(chunk)
        updated["rrf_rank"] = rrf_rank
        updated["rerank_score"] = round(float(score), 4)
        reranked.append(updated)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    if top_k is not None:
        return reranked[:top_k]
    return reranked
