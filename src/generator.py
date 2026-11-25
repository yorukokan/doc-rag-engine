import time

from agent import run_agent_rag
from guardrail import (
    DEFAULT_MIN_BM25_SCORE,
    DEFAULT_MIN_DENSE_SCORE,
    DEFAULT_MIN_RERANK_SCORE,
    REFUSAL_MESSAGE,
    check_retrieval_guardrail,
)
from llm_client import MODEL_NAME, client
from retriever import search
from rewriter import rewrite_query


def ask_rag_with_trace(
    question: str,
    top_k: int = 3,
    use_rerank: bool = False,
    history: list[dict] | None = None,
    use_rewrite: bool = True,
    use_guardrail: bool = True,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE,
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
    use_agent: bool = False,
    max_agent_steps: int = 3,
) -> dict:
    if use_agent:
        return run_agent_rag(
            question=question,
            top_k=top_k,
            use_rerank=use_rerank,
            history=history,
            use_rewrite=use_rewrite,
            use_guardrail=use_guardrail,
            min_rerank_score=min_rerank_score,
            min_dense_score=min_dense_score,
            min_bm25_score=min_bm25_score,
            max_steps=max_agent_steps,
            llm_client=client,
            model_name=MODEL_NAME,
            search_fn=search,
        )

    # 1. Sohbet geçmişi varsa ve rewrite açıksa takip sorusunu bağımsız arama sorgusuna çevir
    if use_rewrite and history:
        rewritten_query, rewrite_ms = rewrite_query(
            question,
            history,
            llm_client=client,
            model_name=MODEL_NAME,
        )
    else:
        rewritten_query = question
        rewrite_ms = 0.0

    # 2. Qdrant + BM25 (ve isteğe bağlı Reranker) ile en alakalı kartları bul ve süresini ölç
    t0 = time.perf_counter()
    retrieved_chunks = search(rewritten_query, top_k=top_k, use_rerank=use_rerank)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

    # 2.5. Erken Reddetme Kapısı (Guardrail): Skorlar eşiğin altındaysa LLM'i hiç çağırma
    if use_guardrail:
        passed, rejection_reason = check_retrieval_guardrail(
            retrieved_chunks,
            use_rerank=use_rerank,
            min_rerank_score=min_rerank_score,
            min_dense_score=min_dense_score,
            min_bm25_score=min_bm25_score,
        )
        if not passed:
            return {
                "answer": REFUSAL_MESSAGE,
                "chunks": retrieved_chunks,
                "rewritten_query": rewritten_query,
                "rewrite_ms": rewrite_ms,
                "retrieval_ms": retrieval_ms,
                "generation_ms": 0.0,
                "early_rejected": True,
                "rejection_reason": rejection_reason,
                "agent_steps": [],
            }

    # 3. Bulunan kartları LLM'in okuyabileceği düzenli bir metne çevir
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Kaynak: {chunk['source']} - {chunk['section']}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    # 4. LLM'e kuralları (System Prompt) ve soruyu ver
    system_prompt = (
        "Sen sadece sana verilen doküman parçalarına dayanarak cevap veren bir teknik asistansın. "
        "Verilen metinlerde olmayan hiçbir bilgiyi kendi ezberinden ekleme. "
        f"Eğer cevap metinlerde yoksa açıkça '{REFUSAL_MESSAGE}' de. "
        "Sohbet geçmişi yalnızca konuşmanın akışını anlamak içindir; tüm olgusal bilgiler mutlaka "
        "bu turda verilen DOKÜMANLAR bölümünden gelmeli ve verdiğin her bilginin sonuna mutlaka "
        "[dosya_adi - Bölüm] formatında kaynak eklenmelidir."
    )

    user_prompt = f"""Aşağıdaki doküman parçalarını kullanarak soruyu yanıtla:

DOKÜMANLAR:
{context_text}

SORU: {question}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        *(history or []),
        {"role": "user", "content": user_prompt},
    ]

    # 5. LLM'e isteği at ve yanıt süresini ölç
    t1 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,  # 0.1 = Yaratıcılığı kıs, sadece metne sadık kal
    )
    generation_ms = round((time.perf_counter() - t1) * 1000, 1)
    answer = (response.choices[0].message.content or "") if response.choices else ""

    return {
        "answer": answer,
        "chunks": retrieved_chunks,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "rewritten_query": rewritten_query,
        "rewrite_ms": rewrite_ms,
        "early_rejected": False,
        "rejection_reason": None,
        "agent_steps": [],
    }


def ask_rag(
    question: str,
    top_k: int = 3,
    use_rerank: bool = False,
    history: list[dict] | None = None,
    use_rewrite: bool = True,
    use_guardrail: bool = True,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE,
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
    use_agent: bool = False,
    max_agent_steps: int = 3,
) -> str:
    return ask_rag_with_trace(
        question,
        top_k=top_k,
        use_rerank=use_rerank,
        history=history,
        use_rewrite=use_rewrite,
        use_guardrail=use_guardrail,
        min_rerank_score=min_rerank_score,
        min_dense_score=min_dense_score,
        min_bm25_score=min_bm25_score,
        use_agent=use_agent,
        max_agent_steps=max_agent_steps,
    )["answer"]


if __name__ == "__main__":
    soru = "Wildcard DNS nedir ve ne gibi riskleri vardır?"
    print(f"Soru: {soru}\n" + "=" * 50)
    cevap = ask_rag(soru)
    print(cevap)
