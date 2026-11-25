import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import html
from typing import Any

import streamlit as st

from ui_client import (
    ApiClientError,
    DEFAULT_API_URL,
    ask_question,
    check_health,
    delete_session,
    search_docs,
)

st.set_page_config(
    page_title="Doc-RAG Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_CHAT = "Tam RAG Sohbet"
MODE_SEARCH = "Sadece Arama (Röntgen)"

SAMPLE_QUESTIONS: list[dict[str, str]] = [
    {
        "label": "🔐 SSH Parola Girişini Kapatma",
        "question": "SSH bağlantılarında parola girişini kapatmak için ne yapılır?",
        "hint": "Güvenlik yapılandırması",
    },
    {
        "label": "🌐 HTTP 401 vs 403 Farkı",
        "question": "HTTP 401 ve 403 farkı nedir?",
        "hint": "Web & API durum kodları",
    },
    {
        "label": "🤖 Çok Parçalı: SSH & Log Analizi",
        "question": "SSH parola girişi nasıl kapatılır ve sonrasında SSH servisinin hata logları nasıl incelenir?",
        "hint": "Agentic RAG çok adımlı sentez testi",
    },
    {
        "label": "🛡️ Tuzak Soru: Docker Container",
        "question": "Docker container nedir?",
        "hint": "Guardrail erken reddetme testi",
    },
]


def _inject_styles() -> None:
    """Koyu ve açık tema ile uyumlu hafif, okunabilir UI stilleri ekler."""
    st.markdown(
        """
        <style>
        .rag-header {
            padding: 0.25rem 0 0.75rem 0;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(128, 128, 128, 0.22);
        }
        .rag-header h1 {
            margin: 0 0 0.25rem 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .rag-header p {
            margin: 0;
            opacity: 0.78;
            font-size: 0.95rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.2;
            margin: 0.25rem 0 0.5rem 0;
        }
        .status-ok {
            background: rgba(16, 185, 129, 0.14);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .status-warn {
            background: rgba(245, 158, 11, 0.14);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .status-err {
            background: rgba(239, 68, 68, 0.14);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .health-meta {
            font-size: 0.82rem;
            line-height: 1.55;
            padding: 0.55rem 0.7rem;
            border-radius: 0.5rem;
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            margin-bottom: 0.5rem;
        }
        .empty-hero {
            padding: 1.25rem 1.4rem;
            border-radius: 0.75rem;
            background: rgba(128, 128, 128, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.18);
            margin: 0.75rem 0 1.25rem 0;
        }
        .empty-hero h3 {
            margin: 0 0 0.35rem 0;
            font-size: 1.15rem;
        }
        .empty-hero p {
            margin: 0;
            opacity: 0.8;
            font-size: 0.92rem;
        }
        .chunk-card {
            padding: 0.85rem 1rem;
            border-radius: 0.6rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-left: 3px solid rgba(59, 130, 246, 0.75);
            background: rgba(128, 128, 128, 0.05);
            margin-bottom: 0.7rem;
        }
        .chunk-title {
            font-weight: 600;
            font-size: 0.92rem;
            margin-bottom: 0.45rem;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.55rem;
        }
        .score-badge {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 0.35rem;
            font-size: 0.76rem;
            font-weight: 600;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: rgba(128, 128, 128, 0.12);
            border: 1px solid rgba(128, 128, 128, 0.22);
        }
        .score-badge-primary {
            background: rgba(59, 130, 246, 0.14);
            border-color: rgba(59, 130, 246, 0.35);
        }
        .chunk-preview {
            font-size: 0.86rem;
            line-height: 1.5;
            opacity: 0.9;
            white-space: pre-wrap;
            margin: 0;
        }
        .guardrail-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.2rem 0.6rem;
            border-radius: 0.4rem;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(245, 158, 11, 0.15);
            color: #d97706;
            border: 1px solid rgba(245, 158, 11, 0.35);
            margin-bottom: 0.5rem;
        }
        .agent-step-card {
            padding: 0.75rem 0.95rem;
            border-radius: 0.6rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-left: 3px solid rgba(139, 92, 246, 0.75);
            background: rgba(128, 128, 128, 0.05);
            margin-bottom: 0.6rem;
        }
        .agent-step-card.step-search {
            border-left-color: rgba(59, 130, 246, 0.8);
        }
        .agent-step-card.step-answer {
            border-left-color: rgba(16, 185, 129, 0.8);
        }
        .agent-step-card.step-direct_chat {
            border-left-color: rgba(139, 92, 246, 0.8);
        }
        .agent-step-card.step-fallback {
            border-left-color: rgba(245, 158, 11, 0.85);
        }
        .agent-step-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .agent-step-title {
            display: inline-flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.45rem;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .agent-step-meta {
            display: inline-flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .agent-action-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.14rem 0.5rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 600;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            line-height: 1.3;
        }
        .action-search {
            background: rgba(59, 130, 246, 0.14);
            color: #3b82f6;
            border: 1px solid rgba(59, 130, 246, 0.35);
        }
        .action-answer {
            background: rgba(16, 185, 129, 0.14);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .action-direct_chat {
            background: rgba(139, 92, 246, 0.14);
            color: #8b5cf6;
            border: 1px solid rgba(139, 92, 246, 0.35);
        }
        .action-fallback {
            background: rgba(245, 158, 11, 0.14);
            color: #d97706;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .action-default {
            background: rgba(128, 128, 128, 0.14);
            border: 1px solid rgba(128, 128, 128, 0.3);
        }
        .agent-thought {
            font-size: 0.86rem;
            line-height: 1.5;
            opacity: 0.92;
            margin: 0 0 0.4rem 0;
            white-space: pre-wrap;
        }
        .agent-query-box {
            font-size: 0.81rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            padding: 0.35rem 0.55rem;
            border-radius: 0.4rem;
            background: rgba(128, 128, 128, 0.09);
            border: 1px solid rgba(128, 128, 128, 0.2);
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "session_id": None,
        "api_url": DEFAULT_API_URL,
        "last_checked_url": None,
        "health_data": None,
        "health_error": None,
        "health_checked": False,
        "pending_prompt": None,
        "clear_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _refresh_health(base_url: str) -> None:
    try:
        st.session_state.health_data = check_health(base_url=base_url)
        st.session_state.health_error = None
    except ApiClientError as exc:
        st.session_state.health_data = None
        st.session_state.health_error = str(exc)
    except Exception as exc:
        st.session_state.health_data = None
        st.session_state.health_error = f"Beklenmeyen bağlantı hatası: {exc}"
    st.session_state.health_checked = True
    st.session_state.last_checked_url = base_url


def _shorten_session_id(session_id: str | None) -> str:
    if not session_id:
        return "Henüz başlatılmadı"
    if len(session_id) <= 14:
        return session_id
    return f"{session_id[:8]}…{session_id[-4:]}"


def _format_preview(text: str, max_chars: int = 420) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "…"


def _render_chunk_cards(chunks: list[dict[str, Any]]) -> None:
    st.markdown(f"**Kaynak Parçaları ({len(chunks)})**")
    if not chunks:
        st.caption("Bu sorgu için eşleşen kaynak parçası bulunamadı.")
        return

    for idx, chunk in enumerate(chunks, start=1):
        source = html.escape(str(chunk.get("source") or "Bilinmeyen kaynak"))
        section = html.escape(str(chunk.get("section") or "Genel"))
        preview = html.escape(_format_preview(str(chunk.get("text") or "")))

        badges: list[str] = []
        rerank_score = chunk.get("rerank_score")
        if rerank_score is not None:
            badges.append(
                f'<span class="score-badge score-badge-primary">Rerank: {float(rerank_score):.3f}</span>'
            )

        rrf_score = chunk.get("score")
        rrf_rank = chunk.get("rrf_rank")
        if rrf_score is not None:
            rank_suffix = f" (#{rrf_rank})" if rrf_rank is not None else ""
            badges.append(
                f'<span class="score-badge">RRF: {float(rrf_score):.4f}{rank_suffix}</span>'
            )

        dense_score = chunk.get("dense_score")
        dense_rank = chunk.get("dense_rank")
        if dense_score is not None:
            rank_suffix = f" (#{dense_rank})" if dense_rank is not None else ""
            badges.append(
                f'<span class="score-badge">Vektör: {float(dense_score):.3f}{rank_suffix}</span>'
            )

        bm25_score = chunk.get("bm25_score")
        bm25_rank = chunk.get("bm25_rank")
        if bm25_score is not None:
            rank_suffix = f" (#{bm25_rank})" if bm25_rank is not None else ""
            badges.append(
                f'<span class="score-badge">BM25: {float(bm25_score):.2f}{rank_suffix}</span>'
            )

        badges_html = "".join(badges)
        st.markdown(
            f"""
            <div class="chunk-card">
                <div class="chunk-title">#{idx} &nbsp;{source} — {section}</div>
                <div class="badge-row">{badges_html}</div>
                <p class="chunk-preview">{preview}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


_AGENT_ACTION_META: dict[str, tuple[str, str, str]] = {
    "search": ("action-search", "step-search", "🔍 search"),
    "answer": ("action-answer", "step-answer", "✅ answer"),
    "direct_chat": ("action-direct_chat", "step-direct_chat", "💬 direct_chat"),
    "fallback": ("action-fallback", "step-fallback", "⚠️ fallback"),
}


def _render_agent_steps(agent_steps: list[dict[str, Any]]) -> None:
    if not agent_steps:
        return

    st.markdown(f"**🤖 Ajan Karar Adımları ({len(agent_steps)})**")
    for idx, step in enumerate(agent_steps, start=1):
        step_num = html.escape(str(step.get("step", idx)))
        raw_action = str(step.get("action") or "unknown").strip().lower()
        badge_cls, card_cls, action_label = _AGENT_ACTION_META.get(
            raw_action,
            ("action-default", "", html.escape(raw_action)),
        )

        thought_raw = str(step.get("thought") or "").strip()
        thought_html = (
            f'<p class="agent-thought">{html.escape(thought_raw)}</p>'
            if thought_raw
            else ""
        )

        query_raw = step.get("query")
        query_str = str(query_raw).strip() if query_raw is not None else ""
        query_html = (
            f'<div class="agent-query-box"><strong>Alt Sorgu:</strong> {html.escape(query_str)}</div>'
            if query_str
            else ""
        )

        meta_badges: list[str] = []
        chunks_found = step.get("chunks_found")
        if chunks_found is not None:
            try:
                chunks_int = int(chunks_found)
                meta_badges.append(
                    f'<span class="score-badge">📄 {chunks_int} parça</span>'
                )
            except (TypeError, ValueError):
                pass

        duration_ms = step.get("duration_ms")
        if duration_ms is not None:
            try:
                duration_val = float(duration_ms)
                meta_badges.append(
                    f'<span class="score-badge">⏱️ {duration_val:.1f} ms</span>'
                )
            except (TypeError, ValueError):
                pass

        meta_html = "".join(meta_badges)
        card_class_attr = f"agent-step-card {card_cls}".strip()

        st.markdown(
            f"""
            <div class="{card_class_attr}">
                <div class="agent-step-header">
                    <div class="agent-step-title">
                        <span>Adım #{step_num}</span>
                        <span class="agent-action-badge {badge_cls}">{action_label}</span>
                    </div>
                    <div class="agent-step-meta">{meta_html}</div>
                </div>
                {thought_html}
                {query_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_chat_observability(msg: dict[str, Any]) -> None:
    timings = msg.get("timings") or {}
    rewrite_ms = float(timings.get("rewrite_ms", 0.0))
    retrieval_ms = float(timings.get("retrieval_ms", 0.0))
    generation_ms = float(timings.get("generation_ms", 0.0))
    early_rejected = bool(msg.get("early_rejected", False))
    rejection_reason = msg.get("rejection_reason")
    original_query = str(msg.get("original_query") or "").strip()
    rewritten_query = str(msg.get("rewritten_query") or "").strip()
    agent_steps = msg.get("agent_steps") or []
    chunks = msg.get("chunks") or []

    with st.expander("🔬 Röntgen (Observability)", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Rewrite", f"{rewrite_ms:.1f} ms")
        col2.metric("Arama (Retrieval)", f"{retrieval_ms:.1f} ms")
        if early_rejected:
            col3.metric("LLM Üretim", f"{generation_ms:.1f} ms", delta="Atlandı", delta_color="off")
        else:
            col3.metric("LLM Üretim", f"{generation_ms:.1f} ms")

        if agent_steps:
            _render_agent_steps(agent_steps)

        if rewritten_query and original_query and rewritten_query != original_query:
            st.info(
                f"**Sorgu Yeniden Yazıldı (Query Rewrite):**\n\n"
                f"`{original_query}` → `{rewritten_query}`"
            )

        if early_rejected:
            reason_line = (
                f"\n\n**Gerekçe:** {rejection_reason}"
                if rejection_reason
                else ""
            )
            st.warning(
                f"🛡️ **Erken Reddetme Kapısı Tetiklendi — LLM Atlandı**{reason_line}"
            )

        _render_chunk_cards(chunks)


def _render_search_observability(msg: dict[str, Any]) -> None:
    retrieval_ms = float(msg.get("retrieval_ms", 0.0))
    chunks = msg.get("chunks") or []

    with st.expander("🔬 Röntgen (Observability)", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Arama (Retrieval)", f"{retrieval_ms:.1f} ms")
        col2.metric("Bulunan Parça", str(len(chunks)))
        col3.metric("LLM Çağrısı", "Atlandı", delta="Arama Modu", delta_color="off")
        _render_chunk_cards(chunks)


def _render_message(msg: dict[str, Any]) -> None:
    role = msg.get("role", "assistant")
    mode = msg.get("mode", "chat")

    with st.chat_message(role):
        if role == "user":
            st.markdown(msg.get("content", ""))
            return

        if mode == "error":
            st.error(msg.get("error", "Bilinmeyen bir hata oluştu."))
            return

        if mode == "search":
            st.markdown(msg.get("content", ""))
            _render_search_observability(msg)
            return

        if msg.get("early_rejected"):
            st.markdown(
                '<div class="guardrail-tag">🛡️ Erken Reddetme Kapısı Tetiklendi — LLM Atlandı</div>',
                unsafe_allow_html=True,
            )
        st.markdown(msg.get("content", ""))
        _render_chat_observability(msg)


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### ⚙️ Kontrol Paneli")

        # 1. API Bağlantısı & Canlı Sağlık Kartı
        st.markdown("#### 🔌 API Bağlantısı")
        api_url_input = st.text_input(
            "API Adresi",
            value=st.session_state.api_url,
            placeholder=DEFAULT_API_URL,
            help="FastAPI sunucusunun temel adresi (örn. http://127.0.0.1:8000).",
        ).strip()
        api_url = api_url_input or DEFAULT_API_URL
        st.session_state.api_url = api_url

        if st.button("🔄 Bağlantıyı Yenile", use_container_width=True):
            _refresh_health(api_url)
        elif not st.session_state.health_checked or st.session_state.last_checked_url != api_url:
            _refresh_health(api_url)

        health_data = st.session_state.health_data
        health_error = st.session_state.health_error

        if health_error:
            st.markdown(
                '<span class="status-pill status-err">🔴 Bağlantı Yok</span>',
                unsafe_allow_html=True,
            )
            st.error(
                "FastAPI sunucusuna ulaşılamıyor. API'yi başlatmak için terminalde şu komutu çalıştırın:\n\n"
                "```bash\n"
                "uvicorn api:app --app-dir src --reload\n"
                "```\n\n"
                f"**Detay:** {health_error}"
            )
        elif health_data:
            status = health_data.get("status", "degraded")
            qdrant_ok = bool(health_data.get("qdrant_connected", False))
            collection_exists = bool(health_data.get("collection_exists", False))
            points_count = health_data.get("points_count")
            llm_configured = bool(health_data.get("llm_configured", False))
            llm_model = health_data.get("llm_model") or "Yapılandırılmamış"
            collection_name = health_data.get("collection_name") or "—"

            if status == "ok" and llm_configured:
                badge_cls, badge_text = "status-ok", "🟢 Çevrimiçi (Hazır)"
            elif status == "ok":
                badge_cls, badge_text = "status-warn", "🟠 Kısmi Hazır (LLM Eksik)"
            else:
                badge_cls, badge_text = "status-warn", "🟠 Kısmi / Degraded"

            st.markdown(
                f'<span class="status-pill {badge_cls}">{badge_text}</span>',
                unsafe_allow_html=True,
            )

            points_label = f"{points_count:,}" if isinstance(points_count, int) else "—"
            qdrant_label = (
                f"Bağlı ({html.escape(str(collection_name))})"
                if qdrant_ok and collection_exists
                else ("Bağlı (Koleksiyon yok)" if qdrant_ok else "Bağlı değil")
            )
            llm_label = html.escape(str(llm_model))

            st.markdown(
                f"""
                <div class="health-meta">
                    <div><strong>Qdrant:</strong> {qdrant_label}</div>
                    <div><strong>İndekslenen Parça:</strong> {points_label}</div>
                    <div><strong>LLM Modeli:</strong> {llm_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # 2. Çalışma Modu
        st.markdown("#### 🧭 Çalışma Modu")
        mode = st.radio(
            "Çalışma Modu Seçin",
            options=[MODE_CHAT, MODE_SEARCH],
            index=0,
            label_visibility="collapsed",
        )
        if mode == MODE_CHAT:
            st.caption("Çok turlu sohbet, sorgu yeniden yazma, guardrail ve LLM yanıtı.")
        else:
            st.caption("Yalnızca hibrit arama + rerank skorlarını inceleyin (LLM çağrılmaz).")

        is_search_only = mode == MODE_SEARCH

        st.divider()

        # 3. Arama & Guardrail Ayarları
        st.markdown("#### 🎛️ Arama & Guardrail")
        top_k = st.slider(
            "Kaynak Sayısı (top_k)",
            min_value=1,
            max_value=10,
            value=3,
            help="Arama sonucunda döndürülecek en iyi parça sayısı.",
        )
        use_rerank = st.toggle(
            "Cross-Encoder Rerank",
            value=True,
            help="Hibrit arama adaylarını Cross-Encoder ile yeniden sıralar.",
        )
        use_rewrite = st.toggle(
            "Sorgu Yeniden Yazma (Rewrite)",
            value=True,
            disabled=is_search_only,
            help="Çok turlu sohbetlerde takip sorularını bağlama göre bağımsız hale getirir.",
        )
        use_guardrail = st.toggle(
            "Erken Reddetme (Guardrail)",
            value=True,
            disabled=is_search_only,
            help="Alakasız sorularda LLM çağrısını atlayarak halüsinasyonu ve maliyeti önler.",
        )
        use_agent = st.toggle(
            "🤖 Agentic RAG (Çok Adımlı Ajan)",
            value=False,
            disabled=is_search_only,
            help="Karmaşık ve çok parçalı soruları alt sorgulara bölerek adım adım araştırır ve sentezler.",
        )
        max_agent_steps = 3
        if use_agent and not is_search_only:
            max_agent_steps = st.slider(
                "Maksimum Ajan Adımı",
                min_value=1,
                max_value=5,
                value=3,
                help="Ajanın yanıt üretmeden önce atabileceği en fazla araştırma/karar adımı.",
            )

        with st.expander("Gelişmiş Eşik Ayarları", expanded=False):
            thresholds_disabled = is_search_only or not use_guardrail
            min_rerank_score = st.number_input(
                "Min Rerank Skoru",
                value=-3.0,
                step=0.5,
                format="%.2f",
                disabled=thresholds_disabled,
                help="Cross-Encoder aktifken kabul için gereken en düşük rerank skoru.",
            )
            min_dense_score = st.number_input(
                "Min Vektör (Dense) Skoru",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                format="%.2f",
                disabled=thresholds_disabled,
                help="Rerank kapalıyken kabul için gereken en düşük kosinüs benzerliği.",
            )
            min_bm25_score = st.number_input(
                "Min BM25 Skoru",
                min_value=0.0,
                value=7.0,
                step=0.5,
                format="%.2f",
                disabled=thresholds_disabled,
                help="Rerank kapalıyken kabul için gereken en düşük anahtar kelime (BM25) skoru.",
            )

        st.divider()

        # 4. Oturum Yönetimi
        st.markdown("#### 💬 Oturum Yönetimi")
        active_sid = st.session_state.session_id
        st.caption(f"Aktif Oturum: `{_shorten_session_id(active_sid)}`")

        if st.session_state.clear_notice:
            st.caption(st.session_state.clear_notice)
            st.session_state.clear_notice = None

        if st.button(
            "🗑️ Sohbeti Temizle",
            use_container_width=True,
            disabled=(not st.session_state.messages and not active_sid),
        ):
            notice = "Sohbet geçmişi temizlendi."
            if active_sid:
                try:
                    delete_session(active_sid, base_url=api_url)
                except ApiClientError as exc:
                    notice = f"Yerel sohbet temizlendi (sunucu uyarısı: {exc})."
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.pending_prompt = None
            st.session_state.clear_notice = notice
            st.rerun()

    return {
        "api_url": api_url,
        "mode": mode,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "use_rewrite": use_rewrite,
        "use_guardrail": use_guardrail,
        "use_agent": bool(use_agent and not is_search_only),
        "max_agent_steps": int(max_agent_steps),
        "min_rerank_score": float(min_rerank_score),
        "min_dense_score": float(min_dense_score),
        "min_bm25_score": float(min_bm25_score),
    }


def _render_empty_state(mode: str) -> None:
    mode_desc = (
        "Sorularınızı sorun; sistem ilgili doküman parçalarını bulup kaynaklarıyla yanıtlasın."
        if mode == MODE_CHAT
        else "Doğrudan hibrit arama (Vektör + BM25 + Rerank) sonuçlarını ve skorlarını inceleyin."
    )
    st.markdown(
        f"""
        <div class="empty-hero">
            <h3>👋 Başlamak için bir soru seçin veya aşağıya yazın</h3>
            <p>{html.escape(mode_desc)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for idx, sample in enumerate(SAMPLE_QUESTIONS):
        with cols[idx % 2]:
            if st.button(
                f"{sample['label']}\n\n_{sample['hint']}_",
                key=f"sample_q_{idx}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = sample["question"]
                st.rerun()


def _handle_prompt(prompt: str, config: dict[str, Any]) -> None:
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        return

    mode = config["mode"]
    user_msg = {
        "role": "user",
        "mode": "chat" if mode == MODE_CHAT else "search",
        "content": cleaned_prompt,
    }
    st.session_state.messages.append(user_msg)
    _render_message(user_msg)

    with st.chat_message("assistant"):
        if mode == MODE_CHAT and config.get("use_agent"):
            spinner_text = "Ajan soruyu analiz ediyor ve adım adım araştırıyor..."
        elif mode == MODE_CHAT:
            spinner_text = "Dokümanlar taranıyor ve yanıt hazırlanıyor..."
        else:
            spinner_text = "Hibrit arama ve skorlama çalıştırılıyor..."
        with st.spinner(spinner_text):
            try:
                if mode == MODE_CHAT:
                    result = ask_question(
                        question=cleaned_prompt,
                        session_id=st.session_state.session_id,
                        top_k=config["top_k"],
                        use_rerank=config["use_rerank"],
                        use_rewrite=config["use_rewrite"],
                        use_guardrail=config["use_guardrail"],
                        use_agent=config["use_agent"],
                        max_agent_steps=config["max_agent_steps"],
                        min_rerank_score=config["min_rerank_score"],
                        min_dense_score=config["min_dense_score"],
                        min_bm25_score=config["min_bm25_score"],
                        base_url=config["api_url"],
                    )
                    if result.get("session_id"):
                        st.session_state.session_id = result["session_id"]

                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "mode": "chat",
                        "content": result.get("answer", ""),
                        "original_query": cleaned_prompt,
                        "rewritten_query": result.get("rewritten_query", cleaned_prompt),
                        "early_rejected": bool(result.get("early_rejected", False)),
                        "rejection_reason": result.get("rejection_reason"),
                        "agent_steps": result.get("agent_steps") or [],
                        "chunks": result.get("chunks", []),
                        "timings": result.get("timings", {}),
                    }
                else:
                    result = search_docs(
                        query=cleaned_prompt,
                        top_k=config["top_k"],
                        use_rerank=config["use_rerank"],
                        base_url=config["api_url"],
                    )
                    chunks = result.get("chunks", [])
                    retrieval_ms = float(result.get("retrieval_ms", 0.0))
                    summary = (
                        f"`{cleaned_prompt}` için **{len(chunks)}** kaynak parçası bulundu "
                        f"(`{retrieval_ms:.1f} ms`)."
                    )
                    assistant_msg = {
                        "role": "assistant",
                        "mode": "search",
                        "content": summary,
                        "original_query": cleaned_prompt,
                        "chunks": chunks,
                        "retrieval_ms": retrieval_ms,
                    }
            except ApiClientError as exc:
                assistant_msg = {
                    "role": "assistant",
                    "mode": "error",
                    "error": f"API isteği başarısız oldu: {exc}",
                }
            except Exception as exc:
                assistant_msg = {
                    "role": "assistant",
                    "mode": "error",
                    "error": f"Beklenmeyen bir hata oluştu: {exc}",
                }

        st.session_state.messages.append(assistant_msg)

        if assistant_msg["mode"] == "error":
            st.error(assistant_msg["error"])
        elif assistant_msg["mode"] == "search":
            st.markdown(assistant_msg["content"])
            _render_search_observability(assistant_msg)
        else:
            if assistant_msg.get("early_rejected"):
                st.markdown(
                    '<div class="guardrail-tag">🛡️ Erken Reddetme Kapısı Tetiklendi — LLM Atlandı</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(assistant_msg["content"])
            _render_chat_observability(assistant_msg)


def main() -> None:
    _inject_styles()
    _init_session_state()
    config = _render_sidebar()

    st.markdown(
        """
        <div class="rag-header">
            <h1>🔎 Doc-RAG Engine</h1>
            <p>Hibrit arama, Cross-Encoder reranking, erken reddetme kapısı ve canlı röntgen gözlemlenebilirliği ile doküman asistanı.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages and not st.session_state.pending_prompt:
        _render_empty_state(config["mode"])
    else:
        for msg in st.session_state.messages:
            _render_message(msg)

    placeholder = (
        "Dokümanlar hakkında bir soru sorun..."
        if config["mode"] == MODE_CHAT
        else "Dokümanlarda doğrudan arama yapın (LLM çağrılmaz)..."
    )
    typed_prompt = st.chat_input(placeholder)

    active_prompt: str | None = None
    if st.session_state.pending_prompt:
        active_prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif typed_prompt:
        active_prompt = typed_prompt

    if active_prompt:
        _handle_prompt(active_prompt, config)
        st.rerun()


if __name__ == "__main__":
    main()
