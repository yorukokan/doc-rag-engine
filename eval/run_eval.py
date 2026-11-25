import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from generator import MODEL_NAME, ask_rag_with_trace, client


def _parse_json_response(raw_text: str) -> dict:
    """LLM çıktısındaki olası ```json ... ``` sarmalayıcılarını temizleyip JSON'a çevirir."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def judge_answer(
    question: str,
    chunks: list[dict],
    answer: str,
    answerable: bool,
) -> dict:
    """LLM-as-a-Judge: Cevabın sadakatini, alakasını veya tuzak soru reddini puanlar."""
    if answerable:
        context = "\n---\n".join(c["text"] for c in chunks)
        prompt = f"""Sen titiz bir RAG değerlendirme hakemisin.
Aşağıdaki DOKÜMANLAR, SORU ve ASİSTAN CEVABI'nı incele ve SADECE geçerli bir JSON döndür:
- faithfulness (1-5): Cevaptaki tüm bilgiler gerçekten verilen DOKÜMANLAR'da var mı? (5 = tamamen sadık, 1 = uydurma var)
- relevance (1-5): Cevap kullanıcının sorusunu net ve doğru şekilde yanıtlıyor mu? (5 = tam yanıt, 1 = alakasız)
- reason: 1 kısa Türkçe cümle açıklama.

DOKÜMANLAR:
{context}

SORU: {question}
ASİSTAN CEVABI: {answer}

JSON formatı: {{"faithfulness": 5, "relevance": 5, "reason": "..."}}"""
    else:
        prompt = f"""Sen titiz bir RAG halüsinasyon hakemisin.
Bu bir TUZAK SORU'dur; cevabı dokümanlarda YOKTUR.
Asistanın kendi bilgisinden uydurmadan soruyu nazikçe reddedip reddetmediğini değerlendir ve SADECE geçerli bir JSON döndür:
- refused_properly (true/false): Bilgiyi dokümanlarda bulamadığını açıkça söyleyip uydurmaktan kaçındıysa true, aksi halde false.
- reason: 1 kısa Türkçe cümle açıklama.

SORU: {question}
ASİSTAN CEVABI: {answer}

JSON formatı: {{"refused_properly": true, "reason": "..."}}"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        return _parse_json_response(resp.choices[0].message.content)
    except Exception:
        return {"error": "JSON ayrıştırılamadı"}


def run_evaluation(
    use_rerank: bool = False,
    use_judge: bool = False,
    top_k: int = 2,
    limit: int | None = None,
    use_guardrail: bool = True,
):
    dataset = json.loads(Path("eval/dataset.json").read_text(encoding="utf-8"))
    if limit:
        dataset = dataset[:limit]

    answerable_count = 0
    recall_at_1_hits = 0
    recall_at_k_hits = 0
    citation_hits = 0
    hallucination_passed = 0
    unanswerable_count = 0
    early_rejected_traps = 0
    early_rejected_answerable = 0

    retrieval_times = []
    generation_times = []
    faithfulness_scores = []
    relevance_scores = []

    mode_label = (
        f"Rerank={'AÇIK' if use_rerank else 'KAPALI'} | "
        f"Guardrail={'AÇIK' if use_guardrail else 'KAPALI'} | "
        f"Judge={'AÇIK' if use_judge else 'KAPALI'} | top_k={top_k}"
    )
    print(f"{len(dataset)} SORULUK DEĞERLENDİRME BAŞLIYOR ({mode_label})...\n" + "=" * 72)

    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        expected_source = item["expected_source"]
        expected_section = item["expected_section"]
        answerable = item["answerable"]

        trace = ask_rag_with_trace(
            q,
            top_k=top_k,
            use_rerank=use_rerank,
            use_guardrail=use_guardrail,
        )
        results = trace["chunks"]
        answer = trace["answer"]
        early_rejected = bool(trace.get("early_rejected", False))
        found_sections = [r["section"] for r in results]

        retrieval_times.append(trace["retrieval_ms"])
        generation_times.append(trace["generation_ms"])

        if answerable:
            answerable_count += 1
            if early_rejected:
                early_rejected_answerable += 1
            hit_at_1 = bool(found_sections) and (found_sections[0] == expected_section)
            hit_at_k = expected_section in found_sections
            if hit_at_1:
                recall_at_1_hits += 1
            if hit_at_k:
                recall_at_k_hits += 1

            has_citation = f"[{expected_source} -" in answer
            if has_citation:
                citation_hits += 1

            judge_str = ""
            if use_judge:
                j = judge_answer(q, results, answer, answerable=True)
                f_score = j.get("faithfulness")
                r_score = j.get("relevance")
                if isinstance(f_score, (int, float)):
                    faithfulness_scores.append(f_score)
                if isinstance(r_score, (int, float)):
                    relevance_scores.append(r_score)
                judge_str = f" | Judge(Sadakat:{f_score}/5, Alaka:{r_score}/5)"

            status = "GEÇTİ" if (hit_at_k and has_citation) else "KALDI"
            r1_tag = "#1" if hit_at_1 else f"#{found_sections.index(expected_section) + 1}" if hit_at_k else "YOK"
            print(
                f"[{i:02d}/{len(dataset):02d}] {status} | Sıra:{r1_tag} | "
                f"Citation:{'EVET' if has_citation else 'HAYIR'} | "
                f"{trace['retrieval_ms']}ms+{trace['generation_ms']}ms{judge_str}"
            )
            if not hit_at_k:
                print(f"        -> HATA: Beklenen '{expected_section}', Bulunanlar: {found_sections}")
        else:
            unanswerable_count += 1
            if early_rejected:
                early_rejected_traps += 1
            if use_judge:
                j = judge_answer(q, results, answer, answerable=False)
                refused = bool(j.get("refused_properly", False))
            else:
                refused = "bulunmamaktadır" in answer.lower() or "yer almamaktadır" in answer.lower()

            if refused:
                hallucination_passed += 1
            status = "GEÇTİ" if refused else "KALDI (Salladı!)"
            guardrail_tag = " [Guardrail]" if early_rejected else ""
            print(
                f"[{i:02d}/{len(dataset):02d}] {status}{guardrail_tag} | Tuzak: {q[:40]}... | "
                f"{trace['retrieval_ms']}ms+{trace['generation_ms']}ms"
            )

    print("\n" + "=" * 72)
    print(f"=== SİSTEM KARNESİ ({mode_label}) ===")
    if answerable_count:
        print(f"Recall@1 (Tam 1. Sırada Bulma) : %{(recall_at_1_hits / answerable_count) * 100:.1f} ({recall_at_1_hits}/{answerable_count})")
        print(f"Recall@{top_k} (İlk {top_k}'de Bulma)       : %{(recall_at_k_hits / answerable_count) * 100:.1f} ({recall_at_k_hits}/{answerable_count})")
        print(f"Citation Uyumu                 : %{(citation_hits / answerable_count) * 100:.1f} ({citation_hits}/{answerable_count})")
    if unanswerable_count:
        print(f"Tuzak Soru / Halüsinasyon Testi: %{(hallucination_passed / unanswerable_count) * 100:.1f} ({hallucination_passed}/{unanswerable_count})")
    print(
        f"Erken Reddetme (Guardrail)     : {early_rejected_traps}/{unanswerable_count} tuzak yakalandı "
        f"(LLM atlandı), {early_rejected_answerable} yanlış pozitif"
    )
    if faithfulness_scores:
        print(f"Judge Ortalama Sadakat         : {sum(faithfulness_scores) / len(faithfulness_scores):.2f} / 5.00")
    if relevance_scores:
        print(f"Judge Ortalama Cevap Alakası   : {sum(relevance_scores) / len(relevance_scores):.2f} / 5.00")

    avg_ret = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
    avg_gen = sum(generation_times) / len(generation_times) if generation_times else 0.0
    print(f"Ortalama Süre (Latency)        : Arama {avg_ret:.1f} ms | LLM {avg_gen:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="Doc-RAG Otomatik Değerlendirme & LLM-as-a-Judge")
    parser.add_argument("--rerank", action="store_true", help="Cross-Encoder Reranker'ı aktif ederek test et")
    parser.add_argument("--judge", action="store_true", help="LLM-as-a-Judge ile Sadakat ve Alaka puanla")
    parser.add_argument("--no-guardrail", action="store_true", help="Erken reddetme kapısını (guardrail) devre dışı bırak")
    parser.add_argument("--top-k", type=int, default=2, help="Arama sonucu sayısı (varsayılan: 2)")
    parser.add_argument("--limit", type=int, default=None, help="Sadece ilk N soruyu çalıştır (hızlı test için)")
    args = parser.parse_args()

    run_evaluation(
        use_rerank=args.rerank,
        use_judge=args.judge,
        top_k=args.top_k,
        limit=args.limit,
        use_guardrail=not args.no_guardrail,
    )


if __name__ == "__main__":
    main()
