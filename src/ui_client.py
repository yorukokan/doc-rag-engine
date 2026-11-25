from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

import httpx

DEFAULT_API_URL = (
    os.getenv("RAG_API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
    or "http://127.0.0.1:8000"
)
DEFAULT_API_BASE_URL = DEFAULT_API_URL


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
    params: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    client: httpx.Client | None = None,
) -> httpx.Response:
    request_kwargs: dict[str, Any] = {}
    if json_payload is not None:
        request_kwargs["json"] = json_payload
    if params is not None:
        request_kwargs["params"] = params
    if files is not None:
        request_kwargs["files"] = files

    try:
        if client is not None:
            return client.request(method, url, timeout=timeout, **request_kwargs)
        with httpx.Client(timeout=timeout) as http_client:
            return http_client.request(method, url, **request_kwargs)
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


def _build_ask_payload(
    question: str,
    session_id: str | None = None,
    top_k: int = 3,
    use_rerank: bool = True,
    use_rewrite: bool = True,
    use_guardrail: bool = True,
    min_rerank_score: float = -3.0,
    min_dense_score: float = 0.45,
    min_bm25_score: float = 7.0,
    use_agent: bool = False,
    max_agent_steps: int = 3,
) -> dict[str, Any]:
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
    return payload


def _parse_sse_block(
    event_name: str | None,
    data_lines: list[str],
    status_code: int,
) -> dict[str, Any]:
    raw_data = "\n".join(data_lines)
    try:
        parsed = json.loads(raw_data)
    except ValueError as exc:
        raise ApiClientError(
            "API akış yanıtı geçerli JSON formatında değil.",
            status_code=status_code,
        ) from exc

    if not isinstance(parsed, dict):
        raise ApiClientError(
            "API akış yanıtı beklenen JSON nesnesi formatında değil.",
            status_code=status_code,
        )

    resolved_event = event_name or "message"
    if resolved_event == "error":
        detail = parsed.get("detail")
        message = (
            detail.strip()
            if isinstance(detail, str) and detail.strip()
            else "Yanıt üretilirken bir sunucu hatası oluştu."
        )
        raise ApiClientError(message, status_code=502)

    return {"event": resolved_event, "data": parsed}


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
    payload = _build_ask_payload(
        question=question,
        session_id=session_id,
        top_k=top_k,
        use_rerank=use_rerank,
        use_rewrite=use_rewrite,
        use_guardrail=use_guardrail,
        min_rerank_score=min_rerank_score,
        min_dense_score=min_dense_score,
        min_bm25_score=min_bm25_score,
        use_agent=use_agent,
        max_agent_steps=max_agent_steps,
    )

    response = _send_request(
        "POST",
        url,
        timeout=timeout,
        json_payload=payload,
        client=client,
    )
    return _parse_json_object(response)


def ask_question_stream(
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
) -> Iterator[dict[str, Any]]:
    url = f"{_normalize_base_url(base_url)}/ask/stream"
    payload = _build_ask_payload(
        question=question,
        session_id=session_id,
        top_k=top_k,
        use_rerank=use_rerank,
        use_rewrite=use_rewrite,
        use_guardrail=use_guardrail,
        min_rerank_score=min_rerank_score,
        min_dense_score=min_dense_score,
        min_bm25_score=min_bm25_score,
        use_agent=use_agent,
        max_agent_steps=max_agent_steps,
    )

    client_ctx = (
        contextlib.nullcontext(client)
        if client is not None
        else httpx.Client(timeout=timeout)
    )

    try:
        with client_ctx as http_client:
            with http_client.stream(
                "POST",
                url,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiClientError(
                        _extract_error_detail(response),
                        status_code=response.status_code,
                    )

                current_event: str | None = None
                data_lines: list[str] = []

                for line in response.iter_lines():
                    if line == "":
                        if data_lines:
                            yield _parse_sse_block(
                                current_event,
                                data_lines,
                                response.status_code,
                            )
                        current_event = None
                        data_lines = []
                        continue

                    if line.startswith(":"):
                        continue

                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                    elif line.startswith("data:"):
                        value = line[5:]
                        if value.startswith(" "):
                            value = value[1:]
                        data_lines.append(value)

                if data_lines:
                    yield _parse_sse_block(
                        current_event,
                        data_lines,
                        response.status_code,
                    )
    except httpx.ConnectError as exc:
        raise ApiClientError(
            "FastAPI sunucusuna bağlanılamadı. Sunucunun çalıştığından emin olun."
        ) from exc
    except httpx.TimeoutException as exc:
        raise ApiClientError("API isteği zaman aşımına uğradı.") from exc
    except httpx.HTTPError as exc:
        raise ApiClientError(f"API isteği sırasında ağ hatası oluştu: {exc}") from exc


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


def list_documents(
    base_url: str = DEFAULT_API_BASE_URL,
    timeout: float = 15.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}/documents"
    response = _send_request("GET", url, timeout=timeout, client=client)
    return _parse_json_object(response)


def upload_document(
    filename: str,
    content: bytes,
    overwrite: bool = False,
    base_url: str = DEFAULT_API_BASE_URL,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    url = f"{_normalize_base_url(base_url)}/documents"
    response = _send_request(
        "POST",
        url,
        timeout=timeout,
        params={"overwrite": str(overwrite).lower()},
        files={"file": (filename, content)},
        client=client,
    )
    return _parse_json_object(response)


def delete_document(
    filename: str,
    base_url: str = DEFAULT_API_BASE_URL,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    safe_filename = quote(filename, safe="")
    url = f"{_normalize_base_url(base_url)}/documents/{safe_filename}"
    response = _send_request("DELETE", url, timeout=timeout, client=client)
    return _parse_json_object(response)
