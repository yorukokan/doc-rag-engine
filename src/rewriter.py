import time
import warnings

from llm_client import MODEL_NAME, client

REWRITE_SYSTEM_PROMPT = (
    "Sen bir arama sorgusu yeniden yazma (query rewriting) asistanısın. "
    "Görevin, verilen sohbet geçmişini kullanarak kullanıcının son sorusundaki zamirleri "
    "('o', 'bunu', 'oradaki', 'peki bu' vb.) ve eksik bağlamı çözüp, soruyu tek başına anlaşılır "
    "(standalone) bir arama sorgusuna dönüştürmektir. "
    "Kurallar:\n"
    "- Soruya kesinlikle cevap verme.\n"
    "- Açıklama, ön ek veya tırnak işareti ekleme; yalnızca tek satırlık arama sorgusunu döndür.\n"
    "- Son soru zaten tek başına anlaşılırsa anlamını değiştirmeden aynen koru."
)


def rewrite_query(
    question: str,
    history: list[dict] | None,
    llm_client=None,
    model_name: str | None = None,
) -> tuple[str, float]:
    """Takip sorularını sohbet geçmişine göre tek başına anlaşılır arama sorgusuna çevirir."""
    if not history:
        return question, 0.0

    active_client = client if llm_client is None else llm_client
    active_model = MODEL_NAME if model_name is None else model_name

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": question},
    ]

    t0 = time.perf_counter()
    try:
        response = active_client.chat.completions.create(
            model=active_model,
            messages=messages,
            temperature=0.0,
        )
        rewrite_ms = round((time.perf_counter() - t0) * 1000, 1)
        raw_content = response.choices[0].message.content if response.choices else None
        cleaned = (raw_content or "").strip()
        if not cleaned:
            return question, rewrite_ms
        first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
        return first_line or question, rewrite_ms
    except Exception as exc:
        rewrite_ms = round((time.perf_counter() - t0) * 1000, 1)
        warnings.warn(
            f"Query rewriting başarısız oldu, orijinal sorgu kullanılıyor: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return question, rewrite_ms
