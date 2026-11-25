import json
import re
import time
import warnings

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

VALID_AGENT_ACTIONS = {"search", "answer", "direct_chat"}

AGENT_PLANNER_SYSTEM_PROMPT = (
    "Sen bir Agentic RAG (Retrieval-Augmented Generation) planlama ve karar ajanısın. "
    "Görevin, kullanıcının sorusunu eksiksiz yanıtlayabilmek için doküman deposunda hangi alt sorgularla "
    "arama yapılacağına veya mevcut toplanan bilgilerin yanıt üretmek için yeterli olup olmadığına karar vermektir.\n"
    "Kullanabileceğin aksiyonlar:\n"
    '- "search": Eksik bilgi varsa dokümanlarda aramak için kısa, odaklanmış ve daha önce denenmemiş bir alt sorgu üret.\n'
    '- "answer": Toplanan doküman parçaları soruyu yanıtlamak için yeterliyse sentez aşamasına geç (query: null).\n'
    '- "direct_chat": YALNIZCA 1. adımda, henüz hiç arama yapılmamışken ve kullanıcı sadece selamlaşma, '
    "teşekkür veya gündelik sohbet mesajı yazdıysa kullan (teknik veya doküman gerektiren sorularda KULLANMA; query: null).\n"
    "Kurallar:\n"
    "- Çok parçalı veya karşılaştırmalı sorularda her konuyu ayrı alt sorgularla ara.\n"
    "- Daha önce denenmiş sorguları tekrar etme.\n"
    "- Yanıtını YALNIZCA aşağıdaki şemaya uygun geçerli bir JSON nesnesi olarak döndür, başka metin ekleme:\n"
    '{"thought": "kısa gerekçe", "action": "search|answer|direct_chat", "query": "alt sorgu veya null"}'
)

DIRECT_CHAT_SYSTEM_PROMPT = (
    "Sen teknik dokümantasyon asistanısın. Kullanıcının selamlaşma veya gündelik sohbet mesajına "
    "kısa, nazik ve yardımcı bir yanıt ver. Doküman içi teknik bilgi uydurma."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "Sen sadece sana verilen doküman parçalarına dayanarak cevap veren bir teknik asistansın. "
    "Verilen metinlerde olmayan hiçbir bilgiyi kendi ezberinden ekleme. "
    f"Eğer cevap metinlerde yoksa açıkça '{REFUSAL_MESSAGE}' de. "
    "Sohbet geçmişi yalnızca konuşmanın akışını anlamak içindir; tüm olgusal bilgiler mutlaka "
    "bu turda verilen DOKÜMANLAR bölümünden gelmeli ve verdiğin her bilginin sonuna mutlaka "
    "[dosya_adi - Bölüm] formatında kaynak eklenmelidir."
)


def _parse_agent_action(raw_text: str) -> dict | None:
    """Planlayıcı LLM çıktısından geçerli ajan aksiyon JSON nesnesini ayıklar."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(cleaned[idx:])
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            except ValueError:
                continue

    if not isinstance(parsed, dict):
        return None

    action = parsed.get("action")
    if not isinstance(action, str):
        return None
    action = action.strip()
    if action not in VALID_AGENT_ACTIONS:
        return None

    thought_raw = parsed.get("thought")
    thought = thought_raw.strip() if isinstance(thought_raw, str) else ""

    query_raw = parsed.get("query")
    if action == "search":
        query = query_raw.strip() if isinstance(query_raw, str) else ""
    else:
        query = None

    return {
        "thought": thought,
        "action": action,
        "query": query,
    }


def _chunk_dedup_key(chunk: dict) -> str:
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{chunk.get('source', '')}::{chunk.get('section', '')}::{chunk.get('text', '')[:64]}"


def _chunk_sort_score(chunk: dict, use_rerank: bool) -> float:
    if use_rerank and chunk.get("rerank_score") is not None:
        return float(chunk["rerank_score"])
    return float(chunk.get("score", 0.0))


def merge_and_deduplicate_chunks(
    existing: list[dict],
    new_chunks: list[dict],
    use_rerank: bool = False,
) -> list[dict]:
    """Chunk listelerini tekilleştirir, çakışmada yüksek skorlu kaydı korur ve azalan sıralar."""
    merged: dict[str, dict] = {}
    for chunk in [*existing, *new_chunks]:
        key = _chunk_dedup_key(chunk)
        prev = merged.get(key)
        if prev is None or _chunk_sort_score(chunk, use_rerank) > _chunk_sort_score(prev, use_rerank):
            merged[key] = chunk

    return sorted(
        merged.values(),
        key=lambda c: _chunk_sort_score(c, use_rerank),
        reverse=True,
    )


def _format_planner_context(
    question: str,
    rewritten_query: str,
    step_idx: int,
    max_steps: int,
    seen_queries: list[str],
    collected_chunks: list[dict],
) -> str:
    lines = [
        f"Adım: {step_idx}/{max_steps}",
        f"Kullanıcı Sorusu: {question}",
    ]
    if rewritten_query.strip() != question.strip():
        lines.append(f"Yeniden Yazılmış Sorgu: {rewritten_query}")

    if seen_queries:
        lines.append("Daha Önce Denenen Sorgular:")
        for q in seen_queries:
            lines.append(f"- {q}")
    else:
        lines.append("Daha Önce Denenen Sorgular: (yok)")

    if collected_chunks:
        lines.append(f"Toplanan Doküman Parçaları ({len(collected_chunks)} adet):")
        for idx, chunk in enumerate(collected_chunks, start=1):
            snippet = " ".join(str(chunk.get("text", "")).split())[:220]
            lines.append(
                f"{idx}. [{chunk.get('source', '')} - {chunk.get('section', '')}] {snippet}"
            )
    else:
        lines.append("Toplanan Doküman Parçaları: (henüz arama yapılmadı)")

    lines.append("Sonraki kararını yalnızca JSON formatında döndür.")
    return "\n".join(lines)


def run_agent_rag(
    question: str,
    top_k: int = 3,
    use_rerank: bool = False,
    history: list[dict] | None = None,
    use_rewrite: bool = True,
    use_guardrail: bool = True,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE,
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
    max_steps: int = 3,
    llm_client=None,
    model_name: str | None = None,
    search_fn=None,
) -> dict:
    """Saf Python sınırlı döngülü Agentic RAG yürütücüsü."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps en az 1 olan bir tam sayı olmalıdır.")

    active_client = client if llm_client is None else llm_client
    active_model = MODEL_NAME if model_name is None else model_name
    active_search = search if search_fn is None else search_fn

    if use_rewrite and history:
        rewritten_query, rewrite_ms = rewrite_query(
            question,
            history,
            llm_client=active_client,
            model_name=active_model,
        )
    else:
        rewritten_query = question
        rewrite_ms = 0.0

    merged_chunks: list[dict] = []
    seen_queries_norm: set[str] = set()
    seen_queries_ordered: list[str] = []
    agent_steps: list[dict] = []
    total_retrieval_ms = 0.0
    total_generation_ms = 0.0
    search_count = 0
    direct_chat_triggered = False

    def _run_fallback_search(step_idx: int, step_t0: float, thought: str) -> None:
        nonlocal merged_chunks, total_retrieval_ms, search_count
        s_t0 = time.perf_counter()
        found = active_search(rewritten_query, top_k=top_k, use_rerank=use_rerank)
        total_retrieval_ms += (time.perf_counter() - s_t0) * 1000
        merged_chunks = merge_and_deduplicate_chunks(
            merged_chunks,
            found,
            use_rerank=use_rerank,
        )
        search_count += 1
        agent_steps.append(
            {
                "step": step_idx,
                "thought": thought,
                "action": "fallback",
                "query": rewritten_query,
                "chunks_found": len(found),
                "duration_ms": round((time.perf_counter() - step_t0) * 1000, 1),
            }
        )

    for step_idx in range(1, max_steps + 1):
        step_t0 = time.perf_counter()
        planner_user_prompt = _format_planner_context(
            question=question,
            rewritten_query=rewritten_query,
            step_idx=step_idx,
            max_steps=max_steps,
            seen_queries=seen_queries_ordered,
            collected_chunks=merged_chunks,
        )
        planner_messages = [
            {"role": "system", "content": AGENT_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": planner_user_prompt},
        ]

        parsed: dict | None = None
        planner_failed_exc: Exception | None = None
        try:
            plan_t0 = time.perf_counter()
            plan_resp = active_client.chat.completions.create(
                model=active_model,
                messages=planner_messages,
                temperature=0.0,
            )
            total_generation_ms += (time.perf_counter() - plan_t0) * 1000
            raw_plan = (plan_resp.choices[0].message.content or "") if plan_resp.choices else ""
            parsed = _parse_agent_action(raw_plan)
        except Exception as exc:
            planner_failed_exc = exc

        if parsed is None:
            if planner_failed_exc is not None:
                warnings.warn(
                    f"Agent planlama adımı başarısız oldu, fallback uygulanıyor: {planner_failed_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Agent planlama yanıtı geçerli JSON değil, fallback uygulanıyor.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            if not merged_chunks:
                _run_fallback_search(
                    step_idx,
                    step_t0,
                    "Planlayıcı yanıtı ayrıştırılamadı; varsayılan arama yapıldı.",
                )
            else:
                agent_steps.append(
                    {
                        "step": step_idx,
                        "thought": "Planlayıcı yanıtı ayrıştırılamadı; mevcut parçalarla senteze geçiliyor.",
                        "action": "fallback",
                        "query": None,
                        "chunks_found": 0,
                        "duration_ms": round((time.perf_counter() - step_t0) * 1000, 1),
                    }
                )
            break

        action = parsed["action"]
        thought = parsed["thought"]
        sub_query = parsed["query"]

        if action == "direct_chat":
            if step_idx == 1 and not merged_chunks:
                agent_steps.append(
                    {
                        "step": step_idx,
                        "thought": thought,
                        "action": "direct_chat",
                        "query": None,
                        "chunks_found": 0,
                        "duration_ms": round((time.perf_counter() - step_t0) * 1000, 1),
                    }
                )
                direct_chat_triggered = True
                break
            agent_steps.append(
                {
                    "step": step_idx,
                    "thought": thought,
                    "action": "answer",
                    "query": None,
                    "chunks_found": 0,
                    "duration_ms": round((time.perf_counter() - step_t0) * 1000, 1),
                }
            )
            break

        if action == "answer":
            agent_steps.append(
                {
                    "step": step_idx,
                    "thought": thought,
                    "action": "answer",
                    "query": None,
                    "chunks_found": 0,
                    "duration_ms": round((time.perf_counter() - step_t0) * 1000, 1),
                }
            )
            break

        # action == "search"
        normalized = " ".join((sub_query or "").strip().lower().split())
        if not normalized or normalized in seen_queries_norm:
            if not merged_chunks:
                _run_fallback_search(
                    step_idx,
                    step_t0,
                    thought or "Geçerli yeni alt sorgu üretilemedi; ana sorgu arandı.",
                )
            break

        clean_query = (sub_query or "").strip()
        seen_queries_norm.add(normalized)
        seen_queries_ordered.append(clean_query)

        s_t0 = time.perf_counter()
        found = active_search(clean_query, top_k=top_k, use_rerank=use_rerank)
        total_retrieval_ms += (time.perf_counter() - s_t0) * 1000
        merged_chunks = merge_and_deduplicate_chunks(
            merged_chunks,
            found,
            use_rerank=use_rerank,
        )
        search_count += 1
        agent_steps.append(
            {
                "step": step_idx,
                "thought": thought,
                "action": "search",
                "query": clean_query,
                "chunks_found": len(found),
                "duration_ms": round((time.perf_counter() - step_t0) * 1000, 1),
            }
        )

    if direct_chat_triggered:
        chat_messages = [
            {"role": "system", "content": DIRECT_CHAT_SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": question},
        ]
        c_t0 = time.perf_counter()
        chat_resp = active_client.chat.completions.create(
            model=active_model,
            messages=chat_messages,
            temperature=0.3,
        )
        total_generation_ms += (time.perf_counter() - c_t0) * 1000
        answer = (chat_resp.choices[0].message.content or "") if chat_resp.choices else ""
        return {
            "answer": answer,
            "chunks": [],
            "rewritten_query": rewritten_query,
            "rewrite_ms": round(rewrite_ms, 1),
            "retrieval_ms": 0.0,
            "generation_ms": round(total_generation_ms, 1),
            "early_rejected": False,
            "rejection_reason": None,
            "agent_steps": agent_steps,
        }

    final_limit = top_k * 2 if search_count > 1 else top_k
    final_chunks = merged_chunks[:final_limit]

    if use_guardrail:
        passed, rejection_reason = check_retrieval_guardrail(
            final_chunks,
            use_rerank=use_rerank,
            min_rerank_score=min_rerank_score,
            min_dense_score=min_dense_score,
            min_bm25_score=min_bm25_score,
        )
        if not passed:
            return {
                "answer": REFUSAL_MESSAGE,
                "chunks": final_chunks,
                "rewritten_query": rewritten_query,
                "rewrite_ms": round(rewrite_ms, 1),
                "retrieval_ms": round(total_retrieval_ms, 1),
                "generation_ms": round(total_generation_ms, 1),
                "early_rejected": True,
                "rejection_reason": rejection_reason,
                "agent_steps": agent_steps,
            }

    context_parts = [
        f"[Kaynak: {chunk['source']} - {chunk['section']}]\n{chunk['text']}"
        for chunk in final_chunks
    ]
    context_text = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Aşağıdaki doküman parçalarını kullanarak soruyu yanıtla:

DOKÜMANLAR:
{context_text}

SORU: {question}
"""
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        *(history or []),
        {"role": "user", "content": user_prompt},
    ]

    syn_t0 = time.perf_counter()
    response = active_client.chat.completions.create(
        model=active_model,
        messages=messages,
        temperature=0.1,
    )
    total_generation_ms += (time.perf_counter() - syn_t0) * 1000
    answer = (response.choices[0].message.content or "") if response.choices else ""

    return {
        "answer": answer,
        "chunks": final_chunks,
        "rewritten_query": rewritten_query,
        "rewrite_ms": round(rewrite_ms, 1),
        "retrieval_ms": round(total_retrieval_ms, 1),
        "generation_ms": round(total_generation_ms, 1),
        "early_rejected": False,
        "rejection_reason": None,
        "agent_steps": agent_steps,
    }
