import json
import re
from typing import Any

REFUSAL_MESSAGE = "Verilen dokümanlarda bu bilgi bulunmamaktadır."


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round(value: float, ndigits: int = 4) -> float:
    return round(float(value), ndigits)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_json_response(text: str) -> dict | None:
    """Metin veya ```json ... ``` bloğu içindeki ilk geçerli JSON nesnesini döndürür."""
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = text.strip()
    candidates: list[str] = []

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    candidates.append(cleaned)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate, idx)
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                continue

    return None


def _validate_str_list(value: Any, field_name: str, ctx: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
        errors.append(f"{ctx}: '{field_name}' boş olmayan string listesi olmalıdır.")


def _validate_question_item(item: Any, ctx: str, errors: list[str]) -> None:
    if not isinstance(item, dict):
        errors.append(f"{ctx}: öğe bir dict olmalıdır.")
        return

    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append(f"{ctx}: 'question' boş olmayan bir string olmalıdır.")

    if not isinstance(item.get("answerable"), bool):
        errors.append(f"{ctx}: 'answerable' bool olmalıdır.")

    if not isinstance(item.get("expect_early_reject"), bool):
        errors.append(f"{ctx}: 'expect_early_reject' bool olmalıdır.")

    _validate_str_list(item.get("expected_sources"), "expected_sources", ctx, errors)
    _validate_str_list(item.get("expected_sections"), "expected_sections", ctx, errors)

    if item.get("answerable") is True:
        if isinstance(item.get("expected_sources"), list) and not item["expected_sources"]:
            errors.append(f"{ctx}: 'answerable=True' için 'expected_sources' boş olamaz.")
        if isinstance(item.get("expected_sections"), list) and not item["expected_sections"]:
            errors.append(f"{ctx}: 'answerable=True' için 'expected_sections' boş olamaz.")

    if "expected_rewrite_keywords" in item:
        _validate_str_list(
            item["expected_rewrite_keywords"], "expected_rewrite_keywords", ctx, errors
        )

    if "expected_answer_keywords" in item:
        _validate_str_list(
            item["expected_answer_keywords"], "expected_answer_keywords", ctx, errors
        )


def validate_advanced_dataset(data: dict) -> list[str]:
    """Gelişmiş eval veri setinin şemasını doğrular; hata listesi döner."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["Veri seti üst düzeyde bir JSON nesnesi (dict) olmalıdır."]

    if "multi_turn" not in data:
        errors.append("Eksik üst düzey anahtar: 'multi_turn'")
    elif not isinstance(data["multi_turn"], list):
        errors.append("'multi_turn' bir liste olmalıdır.")
    else:
        seen_dialogue_ids: set[str] = set()
        for idx, dlg in enumerate(data["multi_turn"]):
            ctx = f"multi_turn[{idx}]"
            if not isinstance(dlg, dict):
                errors.append(f"{ctx}: diyalog bir dict olmalıdır.")
                continue

            dlg_id = dlg.get("id")
            if not isinstance(dlg_id, str) or not dlg_id.strip():
                errors.append(f"{ctx}: 'id' boş olmayan bir string olmalıdır.")
            elif dlg_id in seen_dialogue_ids:
                errors.append(f"{ctx}: yinelenen diyalog id '{dlg_id}'.")
            else:
                seen_dialogue_ids.add(dlg_id)

            scenario_name = dlg.get("scenario_name")
            if not isinstance(scenario_name, str) or not scenario_name.strip():
                errors.append(f"{ctx}: 'scenario_name' boş olmayan bir string olmalıdır.")

            turns = dlg.get("turns")
            if not isinstance(turns, list) or len(turns) < 2:
                errors.append(f"{ctx}: 'turns' en az 2 tur içeren bir liste olmalıdır.")
                continue

            for t_idx, turn in enumerate(turns):
                _validate_question_item(turn, f"{ctx}.turns[{t_idx}]", errors)

    if "multi_hop" not in data:
        errors.append("Eksik üst düzey anahtar: 'multi_hop'")
    elif not isinstance(data["multi_hop"], list):
        errors.append("'multi_hop' bir liste olmalıdır.")
    else:
        seen_q_ids: set[str] = set()
        for idx, item in enumerate(data["multi_hop"]):
            ctx = f"multi_hop[{idx}]"
            if isinstance(item, dict):
                q_id = item.get("id")
                if not isinstance(q_id, str) or not q_id.strip():
                    errors.append(f"{ctx}: 'id' boş olmayan bir string olmalıdır.")
                elif q_id in seen_q_ids:
                    errors.append(f"{ctx}: yinelenen soru id '{q_id}'.")
                else:
                    seen_q_ids.add(q_id)
            _validate_question_item(item, ctx, errors)

    return errors


def compute_coverage(expected: list[str], found: list[str]) -> tuple[float, bool]:
    """Beklenen öğelerin bulunanlar içindeki kapsama oranını ve tam kapsama durumunu döner."""
    if not expected:
        return 1.0, True
    expected_set = set(expected)
    found_set = set(found)
    ratio = len(expected_set & found_set) / len(expected_set)
    full = expected_set.issubset(found_set)
    return ratio, full


def compute_multi_source_citation(
    answer: str,
    expected_sources: list[str],
) -> tuple[float, bool]:
    """Cevapta her beklenen kaynak için `[kaynak -` formatında atıf bulunup bulunmadığını ölçer."""
    if not expected_sources:
        return 1.0, True
    unique_sources = list(dict.fromkeys(expected_sources))
    text = answer or ""
    matched = sum(1 for src in unique_sources if f"[{src} -" in text)
    total = len(unique_sources)
    return matched / total, matched == total


def compute_keyword_match(text: str, keywords: list[str]) -> tuple[float, bool]:
    """Case-insensitive anahtar kelime eşleşme oranını ve tam eşleşme durumunu döner."""
    if not keywords:
        return 1.0, True
    lowered = (text or "").lower()
    matched = sum(1 for kw in keywords if kw.lower() in lowered)
    total = len(keywords)
    return matched / total, matched == total


def check_refusal(
    answer: str,
    early_rejected: bool,
    judge_result: dict | None = None,
) -> bool:
    """Tuzak sorunun güvenli biçimde reddedilip reddedilmediğini belirler."""
    if early_rejected:
        return True
    if isinstance(judge_result, dict) and "error" not in judge_result:
        return bool(judge_result.get("refused_properly", False))
    lowered = (answer or "").lower()
    return (
        REFUSAL_MESSAGE.lower() in lowered
        or "bulunmamaktadır" in lowered
        or "yer almamaktadır" in lowered
    )


def _summarize_guardrail(records: list[dict]) -> dict:
    answerable = [r for r in records if r.get("answerable", False)]
    traps = [r for r in records if not r.get("answerable", False)]

    trap_total = len(traps)
    trap_early_rejected = sum(1 for r in traps if r.get("early_rejected", False))
    trap_refused_total = sum(1 for r in traps if r.get("refused", False))
    answerable_fp = sum(1 for r in answerable if r.get("early_rejected", False))

    return {
        "trap_total": trap_total,
        "trap_early_rejected": trap_early_rejected,
        "trap_refused_total": trap_refused_total,
        "trap_catch_rate": _round(trap_refused_total / trap_total) if trap_total else 0.0,
        "answerable_false_positives": answerable_fp,
        "false_positive_rate": _round(answerable_fp / len(answerable)) if answerable else 0.0,
    }


def _summarize_judge(records: list[dict], include_completeness: bool = False) -> dict:
    evaluated = 0
    errors = 0
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []
    completeness_scores: list[float] = []

    for r in records:
        judge = r.get("judge")
        if not isinstance(judge, dict):
            continue
        if "error" in judge:
            errors += 1
            continue
        evaluated += 1
        if _is_number(judge.get("faithfulness")):
            faithfulness_scores.append(float(judge["faithfulness"]))
        if _is_number(judge.get("relevance")):
            relevance_scores.append(float(judge["relevance"]))
        if include_completeness and _is_number(judge.get("completeness")):
            completeness_scores.append(float(judge["completeness"]))

    result: dict[str, Any] = {
        "evaluated": evaluated,
        "errors": errors,
        "faithfulness_avg": _round(_mean(faithfulness_scores), 2) if faithfulness_scores else None,
        "relevance_avg": _round(_mean(relevance_scores), 2) if relevance_scores else None,
    }
    if include_completeness:
        result["completeness_avg"] = (
            _round(_mean(completeness_scores), 2) if completeness_scores else None
        )
    return result


def _summarize_latency(records: list[dict]) -> dict:
    rewrite_vals = [float(r.get("rewrite_ms", 0.0)) for r in records]
    retrieval_vals = [float(r.get("retrieval_ms", 0.0)) for r in records]
    generation_vals = [float(r.get("generation_ms", 0.0)) for r in records]
    total_vals = [
        rw + rt + gn
        for rw, rt, gn in zip(rewrite_vals, retrieval_vals, generation_vals)
    ]
    return {
        "rewrite_avg": _round(_mean(rewrite_vals), 1),
        "retrieval_avg": _round(_mean(retrieval_vals), 1),
        "generation_avg": _round(_mean(generation_vals), 1),
        "total_avg": _round(_mean(total_vals), 1),
    }


def summarize_multi_turn(turn_records: list[dict]) -> dict:
    """Çok turlu değerlendirme kayıtlarından özet karne üretir."""
    dialogue_ids = {r["dialogue_id"] for r in turn_records if "dialogue_id" in r}
    answerable = [r for r in turn_records if r.get("answerable", False)]
    unanswerable = [r for r in turn_records if not r.get("answerable", False)]
    turn1_records = [r for r in turn_records if r.get("turn_index") == 1]
    followup_records = [r for r in turn_records if r.get("turn_index", 0) > 1]

    followup_with_kw = [
        r for r in followup_records if r.get("expected_rewrite_keywords")
    ]
    rewrite_kw_rate = (
        _mean([float(r.get("rewrite_keyword_ratio", 0.0)) for r in followup_with_kw])
        if followup_with_kw
        else 0.0
    )
    turn1_zero_rewrite_ok = (
        sum(1 for r in turn1_records if float(r.get("rewrite_ms", 0.0)) == 0.0) / len(turn1_records)
        if turn1_records
        else 0.0
    )

    return {
        "total_dialogues": len(dialogue_ids),
        "total_turns": len(turn_records),
        "answerable_turns": len(answerable),
        "unanswerable_turns": len(unanswerable),
        "followup_turns": len(followup_records),
        "rewrite_keyword_rate": _round(rewrite_kw_rate),
        "turn1_zero_rewrite_ok": _round(turn1_zero_rewrite_ok),
        "section_coverage_avg": _round(
            _mean([float(r.get("section_coverage_ratio", 0.0)) for r in answerable])
        ),
        "full_section_coverage_rate": _round(
            _mean([1.0 if r.get("full_section_coverage") else 0.0 for r in answerable])
        ),
        "source_coverage_avg": _round(
            _mean([float(r.get("source_coverage_ratio", 0.0)) for r in answerable])
        ),
        "citation_full_rate": _round(
            _mean([1.0 if r.get("citation_full") else 0.0 for r in answerable])
        ),
        "answer_keyword_avg": _round(
            _mean([float(r.get("answer_keyword_ratio", 0.0)) for r in answerable])
        ),
        "guardrail": _summarize_guardrail(turn_records),
        "judge": _summarize_judge(turn_records, include_completeness=False),
        "latency_ms": _summarize_latency(turn_records),
    }


def summarize_multi_hop_mode(records: list[dict]) -> dict:
    """Tek bir multi-hop modu (classic veya agentic) için özet karne üretir."""
    answerable = [r for r in records if r.get("answerable", False)]
    unanswerable = [r for r in records if not r.get("answerable", False)]

    avg_steps = _mean([float(r.get("agent_step_count", 0)) for r in records])
    avg_search_steps = _mean([float(r.get("agent_search_steps", 0)) for r in records])

    return {
        "total_questions": len(records),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "source_coverage_avg": _round(
            _mean([float(r.get("source_coverage_ratio", 0.0)) for r in answerable])
        ),
        "full_source_coverage_rate": _round(
            _mean([1.0 if r.get("full_source_coverage") else 0.0 for r in answerable])
        ),
        "section_coverage_avg": _round(
            _mean([float(r.get("section_coverage_ratio", 0.0)) for r in answerable])
        ),
        "full_section_coverage_rate": _round(
            _mean([1.0 if r.get("full_section_coverage") else 0.0 for r in answerable])
        ),
        "citation_full_rate": _round(
            _mean([1.0 if r.get("citation_full") else 0.0 for r in answerable])
        ),
        "answer_keyword_avg": _round(
            _mean([float(r.get("answer_keyword_ratio", 0.0)) for r in answerable])
        ),
        "guardrail": _summarize_guardrail(records),
        "agent": {
            "avg_steps": _round(avg_steps, 2),
            "avg_search_steps": _round(avg_search_steps, 2),
        },
        "judge": _summarize_judge(records, include_completeness=True),
        "latency_ms": _summarize_latency(records),
    }


def _optional_delta(classic_val: Any, agentic_val: Any, ndigits: int = 2) -> float | None:
    if _is_number(classic_val) and _is_number(agentic_val):
        return _round(float(agentic_val) - float(classic_val), ndigits)
    return None


def compute_mode_delta(classic_summary: dict, agentic_summary: dict) -> dict:
    """Agentic - Classic performans farklarını hesaplar."""
    c_guard = classic_summary.get("guardrail", {})
    a_guard = agentic_summary.get("guardrail", {})
    c_lat = classic_summary.get("latency_ms", {})
    a_lat = agentic_summary.get("latency_ms", {})
    c_agent = classic_summary.get("agent", {})
    a_agent = agentic_summary.get("agent", {})
    c_judge = classic_summary.get("judge", {})
    a_judge = agentic_summary.get("judge", {})

    return {
        "full_section_coverage_delta": _round(
            agentic_summary.get("full_section_coverage_rate", 0.0)
            - classic_summary.get("full_section_coverage_rate", 0.0)
        ),
        "section_coverage_delta": _round(
            agentic_summary.get("section_coverage_avg", 0.0)
            - classic_summary.get("section_coverage_avg", 0.0)
        ),
        "full_source_coverage_delta": _round(
            agentic_summary.get("full_source_coverage_rate", 0.0)
            - classic_summary.get("full_source_coverage_rate", 0.0)
        ),
        "citation_full_delta": _round(
            agentic_summary.get("citation_full_rate", 0.0)
            - classic_summary.get("citation_full_rate", 0.0)
        ),
        "answer_keyword_delta": _round(
            agentic_summary.get("answer_keyword_avg", 0.0)
            - classic_summary.get("answer_keyword_avg", 0.0)
        ),
        "trap_catch_rate_delta": _round(
            a_guard.get("trap_catch_rate", 0.0) - c_guard.get("trap_catch_rate", 0.0)
        ),
        "false_positive_rate_delta": _round(
            a_guard.get("false_positive_rate", 0.0) - c_guard.get("false_positive_rate", 0.0)
        ),
        "total_latency_ms_delta": _round(
            a_lat.get("total_avg", 0.0) - c_lat.get("total_avg", 0.0), 1
        ),
        "avg_steps_delta": _round(
            a_agent.get("avg_steps", 0.0) - c_agent.get("avg_steps", 0.0), 2
        ),
        "faithfulness_delta": _optional_delta(
            c_judge.get("faithfulness_avg"), a_judge.get("faithfulness_avg"), 2
        ),
        "completeness_delta": _optional_delta(
            c_judge.get("completeness_avg"), a_judge.get("completeness_avg"), 2
        ),
    }
