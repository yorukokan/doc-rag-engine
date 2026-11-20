import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from retriever import search

# .env dosyasındaki LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME bilgilerini yükle
load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
MODEL_NAME = os.getenv("LLM_MODEL_NAME")


def ask_rag_with_trace(question: str, top_k: int = 3) -> dict:
    # 1. Önce Qdrant + BM25 ile en alakalı bilgi kartlarını bul ve süresini ölç
    t0 = time.perf_counter()
    retrieved_chunks = search(question, top_k=top_k)
    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

    # 2. Bulunan kartları LLM'in okuyabileceği düzenli bir metne çevir
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Kaynak: {chunk['source']} - {chunk['section']}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    # 3. LLM'e kuralları (System Prompt) ve soruyu ver
    system_prompt = (
        "Sen sadece sana verilen doküman parçalarına dayanarak cevap veren bir teknik asistansın. "
        "Verilen metinlerde olmayan hiçbir bilgiyi kendi ezberinden ekleme. "
        "Eğer cevap metinlerde yoksa açıkça 'Verilen dokümanlarda bu bilgi bulunmamaktadır.' de. "
        "Verdiğin her bilginin sonuna mutlaka [dosya_adi - Bölüm] formatında kaynak ekle."
    )

    user_prompt = f"""Aşağıdaki doküman parçalarını kullanarak soruyu yanıtla:

DOKÜMANLAR:
{context_text}

SORU: {question}
"""

    # 4. LLM'e isteği at ve yanıt süresini ölç
    t1 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # 0.1 = Yaratıcılığı kıs, sadece metne sadık kal
    )
    generation_ms = round((time.perf_counter() - t1) * 1000, 1)

    return {
        "answer": response.choices[0].message.content,
        "chunks": retrieved_chunks,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }


def ask_rag(question: str, top_k: int = 3) -> str:
    return ask_rag_with_trace(question, top_k=top_k)["answer"]


if __name__ == "__main__":
    soru = "Wildcard DNS nedir ve ne gibi riskleri vardır?"
    print(f"Soru: {soru}\n" + "=" * 50)
    cevap = ask_rag(soru)
    print(cevap)
