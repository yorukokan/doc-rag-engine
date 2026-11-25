import time

from llm_client import MODEL_NAME, client
from retriever import search
from rewriter import rewrite_query


def ask_rag_with_trace(
    question: str,
    top_k: int = 3,
    use_rerank: bool = False,
    history: list[dict] | None = None,
    use_rewrite: bool = True,
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
        "Eğer cevap metinlerde yoksa açıkça 'Verilen dokümanlarda bu bilgi bulunmamaktadır.' de. "
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
    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "chunks": retrieved_chunks,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "rewritten_query": rewritten_query,
        "rewrite_ms": rewrite_ms,
    }


def ask_rag(
    question: str,
    top_k: int = 3,
    use_rerank: bool = False,
    history: list[dict] | None = None,
    use_rewrite: bool = True,
) -> str:
    return ask_rag_with_trace(
        question,
        top_k=top_k,
        use_rerank=use_rerank,
        history=history,
        use_rewrite=use_rewrite,
    )["answer"]


if __name__ == "__main__":
    soru = "Wildcard DNS nedir ve ne gibi riskleri vardır?"
    print(f"Soru: {soru}\n" + "=" * 50)
    cevap = ask_rag(soru)
    print(cevap)
