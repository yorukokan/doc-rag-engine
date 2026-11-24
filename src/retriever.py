import re
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

COLLECTION_NAME = "dokumanlar"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def tokenize(text: str) -> list[str]:
    # Noktalama işaretlerini at, küçük harfe çevirip kelimelere ayır
    return re.findall(r"\w+", text.lower())


def search(
    query: str,
    top_k: int = 2,
    rrf_k: int = 10,
    use_rerank: bool = False,
    candidate_k: int = 6,
) -> list[dict]:
    embedder = get_embedder()
    query_vector = list(embedder.embed([query]))[0].tolist()

    client = QdrantClient(path="qdrant_data")
    try:
        # 1. Vektör Araması (Anlamsal)
        dense_response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=20,
        )
        # 2. BM25 için koleksiyondaki tüm kartları çek
        all_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
        )
    finally:
        client.close()

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
