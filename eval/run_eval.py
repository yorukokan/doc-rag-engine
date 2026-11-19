import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retriever import search
from generator import ask_rag


def run_evaluation():
    dataset = json.loads(Path("eval/dataset.json").read_text(encoding="utf-8"))

    answerable_count = 0
    recall_hits = 0
    citation_hits = 0
    hallucination_passed = 0
    unanswerable_count = 0

    print(f"20 SORULUK DEĞERLENDİRME BAŞLIYOR...\n" + "=" * 60)

    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        expected_source = item["expected_source"]
        expected_section = item["expected_section"]
        answerable = item["answerable"]

        results = search(q, top_k=2)
        found_sections = [r["section"] for r in results]
        answer = ask_rag(q, top_k=2)

        if answerable:
            answerable_count += 1
            hit = expected_section in found_sections
            if hit:
                recall_hits += 1

            has_citation = f"[{expected_source} -" in answer
            if has_citation:
                citation_hits += 1

            status = "GEÇTİ" if (hit and has_citation) else "KALDI"
            print(f"[{i:02d}/20] {status} | Bölüm: {expected_section} | Citation: {'EVET' if has_citation else 'HAYIR'}")
            if not hit:
                print(f"        -> HATA: Bulunanlar: {found_sections}")
        else:
            unanswerable_count += 1
            refused = "bulunmamaktadır" in answer.lower() or "yer almamaktadır" in answer.lower()
            if refused:
                hallucination_passed += 1
            status = "GEÇTİ" if refused else "KALDI (Salladı!)"
            print(f"[{i:02d}/20] {status} | Tuzak Soru: {q[:45]}...")

    recall_rate = (recall_hits / answerable_count) * 100
    citation_rate = (citation_hits / answerable_count) * 100
    refusal_rate = (hallucination_passed / unanswerable_count) * 100

    print("\n=== SİSTEM KARNESİ (16 Chunk / 20 Soru) ===")
    print(f"Recall@2 (Arama Başarısı)      : %{recall_rate:.1f} ({recall_hits}/{answerable_count})")
    print(f"Citation Uyumu                 : %{citation_rate:.1f} ({citation_hits}/{answerable_count})")
    print(f"Tuzak Soru / Halüsinasyon Testi: %{refusal_rate:.1f} ({hallucination_passed}/{unanswerable_count})")


if __name__ == "__main__":
    run_evaluation()
