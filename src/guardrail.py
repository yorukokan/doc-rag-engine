REFUSAL_MESSAGE = "Verilen dokümanlarda bu bilgi bulunmamaktadır."

# 20 soruluk eval/dataset.json kalibrasyon ölçümüne göre varsayılan eşikler
DEFAULT_MIN_RERANK_SCORE = -3.0
DEFAULT_MIN_DENSE_SCORE = 0.45
DEFAULT_MIN_BM25_SCORE = 7.0


def check_retrieval_guardrail(
    chunks: list[dict],
    use_rerank: bool = False,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE,
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
) -> tuple[bool, str | None]:
    """Retrieval skorlarını eşiklerle karşılaştırarak erken reddetme kararı verir.

    Returns:
        (True, None) geçerse, (False, reason) erken reddetme tetiklenirse.
    """
    if not chunks:
        return False, "İlgili doküman parçası bulunamadı (no_chunks)"

    if use_rerank and any("rerank_score" in c for c in chunks):
        max_rerank = max(c["rerank_score"] for c in chunks if "rerank_score" in c)
        if max_rerank < min_rerank_score:
            return (
                False,
                f"Rerank skoru eşiğin altında (max={max_rerank:.4f} < {min_rerank_score})",
            )
        return True, None

    max_dense = max(c.get("dense_score", 0.0) for c in chunks)
    max_bm25 = max(c.get("bm25_score", 0.0) for c in chunks)
    if max_dense < min_dense_score and max_bm25 < min_bm25_score:
        return (
            False,
            f"Arama skorları eşiğin altında "
            f"(dense={max_dense:.4f} < {min_dense_score}, bm25={max_bm25:.4f} < {min_bm25_score})",
        )

    return True, None
