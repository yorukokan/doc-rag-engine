import time
from collections.abc import Iterator

from agent import run_agent_rag, run_agent_rag_stream
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


def _prepare_standard_rag(
    question: str,
    top_k: int = 3,
    use_rerank: bool = False,
    history: list[dict] | None = None,
    use_rewrite: bool = True,
    use_guardrail: bool = True,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE,
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
) -> dict:
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
                "chunks": retrieved_chunks,
                "rewritten_query": rewritten_query,
                "rewrite_ms": rewrite_ms,
                "retrieval_ms": retrieval_ms,
                "early_rejected": True,
                "rejection_reason": rejection_reason,
                "agent_steps": [],
                "messages": None,
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

    return {
        "chunks": retrieved_chunks,
        "rewritten_query": rewritten_query,
        "rewrite_ms": rewrite_ms,
        "retrieval_ms": retrieval_ms,
        "early_rejected": False,
        "rejection_reason": None,
        "agent_steps": [],
        "messages": messages,
    }


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

    prep = _prepare_standard_rag(
        question=question,
        top_k=top_k,
        use_rerank=use_rerank,
        history=history,
        use_rewrite=use_rewrite,
        use_guardrail=use_guardrail,
        min_rerank_score=min_rerank_score,
        min_dense_score=min_dense_score,
        min_bm25_score=min_bm25_score,
    )

    if prep["early_rejected"]:
        return {
            "answer": REFUSAL_MESSAGE,
            "chunks": prep["chunks"],
            "rewritten_query": prep["rewritten_query"],
            "rewrite_ms": prep["rewrite_ms"],
            "retrieval_ms": prep["retrieval_ms"],
            "generation_ms": 0.0,
            "early_rejected": True,
            "rejection_reason": prep["rejection_reason"],
            "agent_steps": [],
        }

    # 5. LLM'e isteği at ve yanıt süresini ölç
    t1 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prep["messages"],
        temperature=0.1,  # 0.1 = Yaratıcılığı kıs, sadece metne sadık kal
    )
    generation_ms = round((time.perf_counter() - t1) * 1000, 1)
    answer = (response.choices[0].message.content or "") if response.choices else ""

    return {
        "answer": answer,
        "chunks": prep["chunks"],
        "retrieval_ms": prep["retrieval_ms"],
        "generation_ms": generation_ms,
        "rewritten_query": prep["rewritten_query"],
        "rewrite_ms": prep["rewrite_ms"],
        "early_rejected": False,
        "rejection_reason": None,
        "agent_steps": [],
    }


def ask_rag_stream(
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
) -> Iterator[dict]:
    if use_agent:
        yield from run_agent_rag_stream(
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
        return

    prep = _prepare_standard_rag(
        question=question,
        top_k=top_k,
        use_rerank=use_rerank,
        history=history,
        use_rewrite=use_rewrite,
        use_guardrail=use_guardrail,
        min_rerank_score=min_rerank_score,
        min_dense_score=min_dense_score,
        min_bm25_score=min_bm25_score,
    )

    yield {
        "type": "meta",
        "chunks": prep["chunks"],
        "rewritten_query": prep["rewritten_query"],
        "rewrite_ms": prep["rewrite_ms"],
        "retrieval_ms": prep["retrieval_ms"],
        "early_rejected": prep["early_rejected"],
        "rejection_reason": prep["rejection_reason"],
        "agent_steps": [],
    }

    if prep["early_rejected"]:
        yield {"type": "token", "content": REFUSAL_MESSAGE}
        yield {
            "type": "done",
            "answer": REFUSAL_MESSAGE,
            "generation_ms": 0.0,
        }
        return

    t1 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=prep["messages"],
        temperature=0.1,
        stream=True,
    )
    answer_parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if content:
            answer_parts.append(content)
            yield {"type": "token", "content": content}
    generation_ms = round((time.perf_counter() - t1) * 1000, 1)

    yield {
        "type": "done",
        "answer": "".join(answer_parts),
        "generation_ms": generation_ms,
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
