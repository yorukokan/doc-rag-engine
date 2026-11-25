import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
EVAL_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from memory import ConversationMemory
from metrics import (
    check_refusal,
    compute_coverage,
    compute_keyword_match,
    compute_mode_delta,
    compute_multi_source_citation,
    parse_json_response,
    summarize_multi_hop_mode,
    summarize_multi_turn,
    validate_advanced_dataset,
)


def _get_default_rag_fn() -> Callable[..., dict]:
    from generator import ask_rag_with_trace

    return ask_rag_with_trace


def _get_default_llm() -> tuple[Any, str | None]:
    from llm_client import MODEL_NAME, client

    return client, MODEL_NAME


def judge_advanced_answer(
    question: str,
    answer: str,
    chunks: list[dict],
    answerable: bool,
    expected_sources: list[str],
    client_obj: Any = None,
    model_name: str | None = None,
) -> dict:
    """LLM-as-a-Judge: Çok turlu ve multi-hop cevapları sadakat, alaka ve eksiksizlik açısından puanlar."""
    if client_obj is None or model_name is None:
        default_client, default_model = _get_default_llm()
        active_client = default_client if client_obj is None else client_obj
        active_model = default_model if model_name is None else model_name
    else:
        active_client = client_obj
        active_model = model_name

    if answerable:
        context_parts = [
            f"[Kaynak: {c.get('source', '')} - {c.get('section', '')}]\n{c.get('text', '')}"
            for c in chunks
        ]
        context = "\n---\n".join(context_parts) if context_parts else "(doküman yok)"
        sources_str = ", ".join(expected_sources) if expected_sources else "(belirtilmedi)"
        prompt = f"""Sen titiz bir RAG değerlendirme hakemisin.
Aşağıdaki DOKÜMANLAR, BEKLENEN KAYNAKLAR, SORU ve ASİSTAN CEVABI'nı incele ve SADECE geçerli bir JSON döndür:
- faithfulness (1-5): Cevaptaki tüm bilgiler gerçekten verilen DOKÜMANLAR'da var mı? (5 = tamamen sadık, 1 = uydurma var)
- relevance (1-5): Cevap kullanıcının sorusunu net ve doğru şekilde yanıtlıyor mu? (5 = tam yanıt, 1 = alakasız)
- completeness (1-5): Çok parçalı veya takip sorusunun tüm alt kısımları ve beklenen kaynak bilgileri eksiksiz kapsanmış mı? (5 = tam kapsam, 1 = ciddi eksik)
- reason: 1 kısa Türkçe cümle açıklama.

BEKLENEN KAYNAKLAR: {sources_str}
DOKÜMANLAR:
{context}

SORU: {question}
ASİSTAN CEVABI: {answer}

JSON formatı: {{"faithfulness": 5, "relevance": 5, "completeness": 5, "reason": "..."}}"""
    else:
        prompt = f"""Sen titiz bir RAG halüsinasyon hakemisin.
Bu bir TUZAK SORU'dur; cevabı dokümanlarda YOKTUR.
Asistanın kendi bilgisinden uydurmadan soruyu nazikçe reddedip reddetmediğini değerlendir ve SADECE geçerli bir JSON döndür:
- refused_properly (true/false): Bilgiyi dokümanlarda bulamadığını açıkça söyleyip uydurmaktan kaçındıysa true, aksi halde false.
- reason: 1 kısa Türkçe cümle açıklama.

SORU: {question}
ASİSTAN CEVABI: {answer}

JSON formatı: {{"refused_properly": true, "reason": "..."}}"""

    try:
        resp = active_client.chat.completions.create(
            model=active_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_content = (resp.choices[0].message.content or "") if resp.choices else ""
    except Exception as exc:
        return {"error": f"Judge API hatası ({type(exc).__name__})"}

    parsed = parse_json_response(raw_content)
    if parsed is None:
        return {"error": "JSON ayrıştırılamadı"}

    if answerable:
        required = ("faithfulness", "relevance", "completeness")
        if not all(k in parsed for k in required):
            return {"error": "Judge JSON yanıtında eksik puan alanları var"}
    else:
        if "refused_properly" not in parsed:
            return {"error": "Judge JSON yanıtında 'refused_properly' alanı eksik"}

    return parsed


def evaluate_multi_turn(
    dialogues: list[dict],
    *,
    top_k: int = 3,
    use_rerank: bool = False,
    use_guardrail: bool = True,
    use_judge: bool = False,
    rag_fn: Callable[..., dict] | None = None,
    judge_fn: Callable[..., dict] | None = None,
) -> dict:
    """Çok turlu sohbet senaryolarını hafıza ve rewrite akışıyla değerlendirir."""
    active_rag = _get_default_rag_fn() if rag_fn is None else rag_fn
    active_judge = judge_advanced_answer if judge_fn is None else judge_fn

    turn_records: list[dict] = []

    for dlg in dialogues:
        dialogue_id = dlg["id"]
        scenario_name = dlg["scenario_name"]
        memory = ConversationMemory(max_turns=5)

        for turn_index, turn in enumerate(dlg["turns"], start=1):
            question = turn["question"]
            answerable = bool(turn["answerable"])
            expected_sources = list(turn.get("expected_sources", []))
            expected_sections = list(turn.get("expected_sections", []))
            expected_rewrite_keywords = list(turn.get("expected_rewrite_keywords", []))
            expected_answer_keywords = list(turn.get("expected_answer_keywords", []))
            expect_early_reject = bool(turn.get("expect_early_reject", False))

            history = memory.get_messages()
            trace = active_rag(
                question,
                top_k=top_k,
                use_rerank=use_rerank,
                history=history,
                use_rewrite=True,
                use_guardrail=use_guardrail,
                use_agent=False,
            )

            chunks = trace.get("chunks") or []
            answer = trace.get("answer") or ""
            rewritten_query = trace.get("rewritten_query") or question
            early_rejected = bool(trace.get("early_rejected", False))
            rewrite_ms = float(trace.get("rewrite_ms", 0.0))
            retrieval_ms = float(trace.get("retrieval_ms", 0.0))
            generation_ms = float(trace.get("generation_ms", 0.0))

            # cli.py ile birebir aynı kural: erken reddedilen turlar hafızaya eklenmez
            if not early_rejected:
                memory.add_turn(question, answer)

            found_sources = [c.get("source", "") for c in chunks]
            found_sections = [c.get("section", "") for c in chunks]

            source_cov_ratio, full_source_cov = compute_coverage(expected_sources, found_sources)
            section_cov_ratio, full_section_cov = compute_coverage(expected_sections, found_sections)
            citation_ratio, citation_full = compute_multi_source_citation(answer, expected_sources)
            answer_kw_ratio, answer_kw_full = compute_keyword_match(answer, expected_answer_keywords)
            rewrite_kw_ratio, rewrite_kw_full = compute_keyword_match(
                rewritten_query, expected_rewrite_keywords
            )

            judge_result = None
            if use_judge:
                judge_result = active_judge(
                    question,
                    answer,
                    chunks,
                    answerable,
                    expected_sources,
                )

            refused = check_refusal(answer, early_rejected, judge_result)

            turn_records.append(
                {
                    "dialogue_id": dialogue_id,
                    "scenario_name": scenario_name,
                    "turn_index": turn_index,
                    "history_turn_count": len(history) // 2,
                    "question": question,
                    "rewritten_query": rewritten_query,
                    "answer": answer,
                    "answerable": answerable,
                    "expect_early_reject": expect_early_reject,
                    "early_rejected": early_rejected,
                    "refused": refused,
                    "expected_sources": expected_sources,
                    "found_sources": found_sources,
                    "source_coverage_ratio": round(source_cov_ratio, 4),
                    "full_source_coverage": full_source_cov,
                    "expected_sections": expected_sections,
                    "found_sections": found_sections,
                    "section_coverage_ratio": round(section_cov_ratio, 4),
                    "full_section_coverage": full_section_cov,
                    "citation_ratio": round(citation_ratio, 4),
                    "citation_full": citation_full,
                    "expected_rewrite_keywords": expected_rewrite_keywords,
                    "rewrite_keyword_ratio": round(rewrite_kw_ratio, 4),
                    "rewrite_keyword_full": rewrite_kw_full,
                    "expected_answer_keywords": expected_answer_keywords,
                    "answer_keyword_ratio": round(answer_kw_ratio, 4),
                    "answer_keyword_full": answer_kw_full,
                    "rewrite_ms": rewrite_ms,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": generation_ms,
                    "judge": judge_result,
                }
            )

    return {
        "summary": summarize_multi_turn(turn_records),
        "details": turn_records,
    }


def _build_hop_record(
    item: dict,
    trace: dict,
    *,
    mode: str,
    use_judge: bool,
    active_judge: Callable[..., dict],
) -> dict:
    question = item["question"]
    answerable = bool(item["answerable"])
    expected_sources = list(item.get("expected_sources", []))
    expected_sections = list(item.get("expected_sections", []))
    expected_answer_keywords = list(item.get("expected_answer_keywords", []))
    expect_early_reject = bool(item.get("expect_early_reject", False))

    chunks = trace.get("chunks") or []
    answer = trace.get("answer") or ""
    early_rejected = bool(trace.get("early_rejected", False))
    agent_steps = list(trace.get("agent_steps") or [])
    agent_step_count = len(agent_steps)
    agent_search_steps = sum(
        1
        for step in agent_steps
        if step.get("action") == "search"
        or (step.get("action") == "fallback" and bool(step.get("query")))
    )

    found_sources = [c.get("source", "") for c in chunks]
    found_sections = [c.get("section", "") for c in chunks]

    source_cov_ratio, full_source_cov = compute_coverage(expected_sources, found_sources)
    section_cov_ratio, full_section_cov = compute_coverage(expected_sections, found_sections)
    citation_ratio, citation_full = compute_multi_source_citation(answer, expected_sources)
    answer_kw_ratio, answer_kw_full = compute_keyword_match(answer, expected_answer_keywords)

    judge_result = None
    if use_judge:
        judge_result = active_judge(
            question,
            answer,
            chunks,
            answerable,
            expected_sources,
        )

    refused = check_refusal(answer, early_rejected, judge_result)

    return {
        "id": item["id"],
        "mode": mode,
        "category": item.get("category"),
        "question": question,
        "answer": answer,
        "answerable": answerable,
        "expect_early_reject": expect_early_reject,
        "early_rejected": early_rejected,
        "refused": refused,
        "expected_sources": expected_sources,
        "found_sources": found_sources,
        "source_coverage_ratio": round(source_cov_ratio, 4),
        "full_source_coverage": full_source_cov,
        "expected_sections": expected_sections,
        "found_sections": found_sections,
        "section_coverage_ratio": round(section_cov_ratio, 4),
        "full_section_coverage": full_section_cov,
        "citation_ratio": round(citation_ratio, 4),
        "citation_full": citation_full,
        "expected_answer_keywords": expected_answer_keywords,
        "answer_keyword_ratio": round(answer_kw_ratio, 4),
        "answer_keyword_full": answer_kw_full,
        "agent_step_count": agent_step_count,
        "agent_search_steps": agent_search_steps,
        "agent_steps": agent_steps,
        "rewrite_ms": float(trace.get("rewrite_ms", 0.0)),
        "retrieval_ms": float(trace.get("retrieval_ms", 0.0)),
        "generation_ms": float(trace.get("generation_ms", 0.0)),
        "judge": judge_result,
    }


def evaluate_multi_hop(
    questions: list[dict],
    *,
    top_k: int = 3,
    max_agent_steps: int = 3,
    use_rerank: bool = False,
    use_guardrail: bool = True,
    use_judge: bool = False,
    rag_fn: Callable[..., dict] | None = None,
    judge_fn: Callable[..., dict] | None = None,
) -> dict:
    """Multi-hop soruları Classic RAG ve Agentic RAG modlarında karşılaştırmalı değerlendirir."""
    active_rag = _get_default_rag_fn() if rag_fn is None else rag_fn
    active_judge = judge_advanced_answer if judge_fn is None else judge_fn

    classic_records: list[dict] = []
    agentic_records: list[dict] = []
    details: list[dict] = []

    for item in questions:
        question = item["question"]

        classic_trace = active_rag(
            question,
            top_k=top_k,
            use_rerank=use_rerank,
            history=None,
            use_rewrite=True,
            use_guardrail=use_guardrail,
            use_agent=False,
        )
        classic_rec = _build_hop_record(
            item,
            classic_trace,
            mode="classic",
            use_judge=use_judge,
            active_judge=active_judge,
        )

        agentic_trace = active_rag(
            question,
            top_k=top_k,
            use_rerank=use_rerank,
            history=None,
            use_rewrite=True,
            use_guardrail=use_guardrail,
            use_agent=True,
            max_agent_steps=max_agent_steps,
        )
        agentic_rec = _build_hop_record(
            item,
            agentic_trace,
            mode="agentic",
            use_judge=use_judge,
            active_judge=active_judge,
        )

        classic_records.append(classic_rec)
        agentic_records.append(agentic_rec)
        details.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "question": question,
                "answerable": bool(item["answerable"]),
                "classic": classic_rec,
                "agentic": agentic_rec,
            }
        )

    classic_summary = summarize_multi_hop_mode(classic_records)
    agentic_summary = summarize_multi_hop_mode(agentic_records)
    delta = compute_mode_delta(classic_summary, agentic_summary)

    return {
        "classic": classic_summary,
        "agentic": agentic_summary,
        "delta": delta,
        "details": details,
    }


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "-"
    return f"%{val * 100:.1f}"


def _fmt_delta_pct(val: float | None) -> str:
    if val is None:
        return "-"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f} puan"


def _fmt_score(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val:.2f}/5"


def print_report(report: dict) -> None:
    """Değerlendirme raporunu okunaklı tablo düzeninde konsola yazdırır."""
    meta = report.get("meta", {})
    print("\n" + "=" * 78)
    print("DOC-RAG V4 — ÇOK TURLU & AGENTIC DEĞERLENDİRME KARNESİ")
    if meta:
        print(
            f"Mod={meta.get('mode', 'all')} | top_k={meta.get('top_k')} | "
            f"max_agent_steps={meta.get('max_agent_steps')} | "
            f"Rerank={'AÇIK' if meta.get('use_rerank') else 'KAPALI'} | "
            f"Guardrail={'AÇIK' if meta.get('use_guardrail') else 'KAPALI'} | "
            f"Judge={'AÇIK' if meta.get('use_judge') else 'KAPALI'}"
        )
    print("=" * 78)

    if "multi_turn" in report and report["multi_turn"]:
        mt = report["multi_turn"]["summary"]
        g = mt["guardrail"]
        lat = mt["latency_ms"]
        j = mt["judge"]
        print("\n[1] ÇOK TURLU SOHBET KARNESİ (Multi-Turn + Rewrite + Hafıza)")
        print("-" * 78)
        print(
            f"  Diyalog / Tur Sayısı       : {mt['total_dialogues']} diyalog / {mt['total_turns']} tur "
            f"({mt['answerable_turns']} cevaplanabilir, {mt['unanswerable_turns']} tuzak)"
        )
        print(f"  Tur 1 Sıfır Rewrite Uyumu  : {_fmt_pct(mt['turn1_zero_rewrite_ok'])}")
        print(f"  Takip Rewrite Keyword Oranı: {_fmt_pct(mt['rewrite_keyword_rate'])} ({mt['followup_turns']} takip turu)")
        print(
            f"  Bölüm Kapsama (Ort / Tam)  : {_fmt_pct(mt['section_coverage_avg'])} / "
            f"{_fmt_pct(mt['full_section_coverage_rate'])}"
        )
        print(f"  Kaynak Kapsama (Ort)       : {_fmt_pct(mt['source_coverage_avg'])}")
        print(f"  Citation Tam Uyum          : {_fmt_pct(mt['citation_full_rate'])}")
        print(f"  Cevap Keyword Kapsama      : {_fmt_pct(mt['answer_keyword_avg'])}")
        print(
            f"  Guardrail Tuzak Yakalama   : {_fmt_pct(g['trap_catch_rate'])} "
            f"({g['trap_refused_total']}/{g['trap_total']} red, {g['trap_early_rejected']} erken red) | "
            f"FP: {g['answerable_false_positives']} ({_fmt_pct(g['false_positive_rate'])})"
        )
        if j["evaluated"] or j["errors"]:
            print(
                f"  Judge (N={j['evaluated']}, Hata={j['errors']})    : "
                f"Sadakat {_fmt_score(j['faithfulness_avg'])} | Alaka {_fmt_score(j['relevance_avg'])}"
            )
        print(
            f"  Ortalama Süre (ms)         : Rewrite {lat['rewrite_avg']} | "
            f"Arama {lat['retrieval_avg']} | LLM {lat['generation_avg']} | Toplam {lat['total_avg']}"
        )

    if "multi_hop" in report and report["multi_hop"]:
        mh = report["multi_hop"]
        c = mh["classic"]
        a = mh["agentic"]
        d = mh["delta"]
        print("\n[2] MULTI-HOP KARŞILAŞTIRMA KARNESİ (Classic vs Agentic RAG)")
        print("-" * 78)
        print(
            f"  Soru Sayısı: {c['total_questions']} "
            f"({c['answerable_count']} çoklu-parça, {c['unanswerable_count']} tuzak)\n"
        )
        header = f"  {'Metrik':<32} {'Classic':>12} {'Agentic':>12} {'Delta (A-C)':>16}"
        print(header)
        print("  " + "-" * 74)
        rows = [
            (
                "Tam Bölüm Kapsama",
                _fmt_pct(c["full_section_coverage_rate"]),
                _fmt_pct(a["full_section_coverage_rate"]),
                _fmt_delta_pct(d["full_section_coverage_delta"]),
            ),
            (
                "Ortalama Bölüm Kapsama",
                _fmt_pct(c["section_coverage_avg"]),
                _fmt_pct(a["section_coverage_avg"]),
                _fmt_delta_pct(d["section_coverage_delta"]),
            ),
            (
                "Tam Kaynak Kapsama",
                _fmt_pct(c["full_source_coverage_rate"]),
                _fmt_pct(a["full_source_coverage_rate"]),
                _fmt_delta_pct(d["full_source_coverage_delta"]),
            ),
            (
                "Çoklu Kaynak Citation Tam",
                _fmt_pct(c["citation_full_rate"]),
                _fmt_pct(a["citation_full_rate"]),
                _fmt_delta_pct(d["citation_full_delta"]),
            ),
            (
                "Cevap Keyword Kapsama",
                _fmt_pct(c["answer_keyword_avg"]),
                _fmt_pct(a["answer_keyword_avg"]),
                _fmt_delta_pct(d["answer_keyword_delta"]),
            ),
            (
                "Guardrail Tuzak Yakalama",
                _fmt_pct(c["guardrail"]["trap_catch_rate"]),
                _fmt_pct(a["guardrail"]["trap_catch_rate"]),
                _fmt_delta_pct(d["trap_catch_rate_delta"]),
            ),
            (
                "Guardrail Yanlış Pozitif (FP)",
                _fmt_pct(c["guardrail"]["false_positive_rate"]),
                _fmt_pct(a["guardrail"]["false_positive_rate"]),
                _fmt_delta_pct(d["false_positive_rate_delta"]),
            ),
            (
                "Ortalama Ajan Adımı",
                f"{c['agent']['avg_steps']:.2f}",
                f"{a['agent']['avg_steps']:.2f}",
                f"{d['avg_steps_delta']:+.2f}",
            ),
            (
                "Ortalama Arama Adımı",
                f"{c['agent']['avg_search_steps']:.2f}",
                f"{a['agent']['avg_search_steps']:.2f}",
                f"{a['agent']['avg_search_steps'] - c['agent']['avg_search_steps']:+.2f}",
            ),
            (
                "Toplam Ortalama Süre (ms)",
                f"{c['latency_ms']['total_avg']:.1f}",
                f"{a['latency_ms']['total_avg']:.1f}",
                f"{d['total_latency_ms_delta']:+.1f} ms",
            ),
        ]
        if c["judge"]["evaluated"] or a["judge"]["evaluated"] or c["judge"]["errors"] or a["judge"]["errors"]:
            f_delta = f"{d['faithfulness_delta']:+.2f}" if d["faithfulness_delta"] is not None else "-"
            comp_delta = (
                f"{d['completeness_delta']:+.2f}" if d["completeness_delta"] is not None else "-"
            )
            rows.extend(
                [
                    (
                        "Judge Sadakat",
                        _fmt_score(c["judge"]["faithfulness_avg"]),
                        _fmt_score(a["judge"]["faithfulness_avg"]),
                        f_delta,
                    ),
                    (
                        "Judge Eksiksizlik",
                        _fmt_score(c["judge"]["completeness_avg"]),
                        _fmt_score(a["judge"]["completeness_avg"]),
                        comp_delta,
                    ),
                ]
            )
        for label, c_val, a_val, d_val in rows:
            print(f"  {label:<32} {c_val:>12} {a_val:>12} {d_val:>16}")

    print("\n" + "=" * 78)


def save_report(report: dict, output_path: str) -> Path:
    """Raporu UTF-8 JSON olarak kaydeder; üst dizin yoksa oluşturur."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Doc-RAG V4 Çok Turlu & Agentic Değerlendirme Karnesi"
    )
    parser.add_argument(
        "--mode",
        choices=("all", "multi-turn", "multi-hop"),
        default="all",
        help="Çalıştırılacak değerlendirme modu (varsayılan: all)",
    )
    parser.add_argument(
        "--dataset",
        default="eval/advanced_dataset.json",
        help="Değerlendirme veri seti JSON yolu (varsayılan: eval/advanced_dataset.json)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Cross-Encoder Reranker'ı aktif et",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="LLM-as-a-Judge puanlamasını aktif et",
    )
    parser.add_argument(
        "--no-guardrail",
        action="store_true",
        help="Erken reddetme kapısını (guardrail) devre dışı bırak",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Arama sonucu sayısı (varsayılan: 3)",
    )
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=3,
        help="Agentic RAG maksimum adım sayısı (varsayılan: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Her bölüm için maksimum kayıt sınırı (0 = sınır yok)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Opsiyonel JSON rapor çıktı dosya yolu",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.top_k < 1:
        parser.error("--top-k en az 1 olmalıdır.")
    if args.max_agent_steps < 1:
        parser.error("--max-agent-steps en az 1 olmalıdır.")
    if args.limit < 0:
        parser.error("--limit negatif olamaz.")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"HATA: Veri seti bulunamadı: {dataset_path}", file=sys.stderr)
        return 1

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    errors = validate_advanced_dataset(data)
    if errors:
        print("HATA: Veri seti doğrulaması başarısız:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    use_guardrail = not args.no_guardrail
    report: dict[str, Any] = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "dataset": str(dataset_path),
            "top_k": args.top_k,
            "max_agent_steps": args.max_agent_steps,
            "use_rerank": args.rerank,
            "use_guardrail": use_guardrail,
            "use_judge": args.judge,
            "limit": args.limit,
        }
    }

    if args.mode in ("all", "multi-turn"):
        dialogues = data["multi_turn"]
        if args.limit > 0:
            dialogues = dialogues[: args.limit]
        report["multi_turn"] = evaluate_multi_turn(
            dialogues,
            top_k=args.top_k,
            use_rerank=args.rerank,
            use_guardrail=use_guardrail,
            use_judge=args.judge,
        )

    if args.mode in ("all", "multi-hop"):
        questions = data["multi_hop"]
        if args.limit > 0:
            questions = questions[: args.limit]
        report["multi_hop"] = evaluate_multi_hop(
            questions,
            top_k=args.top_k,
            max_agent_steps=args.max_agent_steps,
            use_rerank=args.rerank,
            use_guardrail=use_guardrail,
            use_judge=args.judge,
        )

    print_report(report)

    if args.output:
        saved = save_report(report, args.output)
        print(f"Rapor kaydedildi: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
