from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


class ApiClientError(Exception):
    """FastAPI backend çağrılarında oluşan hataları temsil eder."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _extract_error_detail(response: httpx.Response) -> str:
    fallback = f"API isteği başarısız oldu (HTTP {response.status_code})."
    try:
        payload = response.json()
    except ValueError:
        return fallback

    if not isinstance(payload, dict) or "detail" not in payload:
        return fallback

    detail = payload["detail"]
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, list) and detail:
        messages: list[str] = []
        for item in detail:
            if isinstance(item, dict) and item.get("msg"):
                messages.append(str(item["msg"]))
            elif item is not None:
                messages.append(str(item))
        if messages:
            return "; ".join(messages)
    if detail is not None and str(detail).strip():
        return str(detail).strip()
    return fallback


def _send_request(
    method: str,
    url: str,
    *,
    timeout: float,
    json_payload: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
) -> httpx.Response:
    try:
        if client is not None:
            return client.request(method, url, json=json_payload, timeout=timeout)
        with httpx.Client(timeout=timeout) as http_client:
            return http_client.request(method, url, json=json_payload)
    except httpx.ConnectError as exc:
        raise ApiClientError(
            "FastAPI sunucusuna bağlanılamadı. Sunucunun çalıştığından emin olun."
        ) from exc
    except httpx.TimeoutException as exc:
        raise ApiClientError("API isteği zaman aşımına uğradı.") from exc
    except httpx.HTTPError as exc:
        raise ApiClientError(f"API isteği sırasında ağ hatası oluştu: {exc}") from exc


def _parse_json_object(response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise ApiClientError(
            _extract_error_detail(response),
            status_code=response.status_code,
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ApiClientError(
            "API yanıtı geçerli JSON formatında değil.",
            status_code=response.status_code,
        ) from exc

    if not isinstance(data, dict):
        raise ApiClientError(
            "API yanıtı beklenen JSON nesnesi formatında değil.",
            status_code=response.status_code,
        )

    return data


def check_health(
    base_url: str = DEFAULT_API_URL,
    timeout: float = 5.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}/health"
    response = _send_request("GET", url, timeout=timeout, client=client)
    return _parse_json_object(response)


def ask_question(
    question: str,
    session_id: str | None = None,
    top_k: int = 3,
    use_rerank: bool = True,
    use_rewrite: bool = True,
    use_guardrail: bool = True,
    min_rerank_score: float = -3.0,
    min_dense_score: float = 0.45,
    min_bm25_score: float = 7.0,
    base_url: str = DEFAULT_API_URL,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
    use_agent: bool = False,
    max_agent_steps: int = 3,
) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}/ask"
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "use_rewrite": use_rewrite,
        "use_guardrail": use_guardrail,
        "min_rerank_score": min_rerank_score,
        "min_dense_score": min_dense_score,
        "min_bm25_score": min_bm25_score,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if use_agent:
        payload["use_agent"] = True
        payload["max_agent_steps"] = max_agent_steps

    response = _send_request(
        "POST",
        url,
        timeout=timeout,
        json_payload=payload,
        client=client,
    )
    return _parse_json_object(response)


def search_docs(
    query: str,
    top_k: int = 3,
    use_rerank: bool = True,
    base_url: str = DEFAULT_API_URL,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}/search"
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "use_rerank": use_rerank,
    }
    response = _send_request(
        "POST",
        url,
        timeout=timeout,
        json_payload=payload,
        client=client,
    )
    return _parse_json_object(response)


def delete_session(
    session_id: str,
    base_url: str = DEFAULT_API_URL,
    timeout: float = 5.0,
    client: httpx.Client | None = None,
) -> bool:
    safe_session_id = quote(session_id, safe="")
    url = f"{_normalize_base_url(base_url)}/sessions/{safe_session_id}"
    response = _send_request("DELETE", url, timeout=timeout, client=client)

    if response.status_code == 200:
        return True
    if response.status_code == 404:
        return False

    raise ApiClientError(
        _extract_error_detail(response),
        status_code=response.status_code,
    )
