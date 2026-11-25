import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ui_client import (
    ApiClientError,
    DEFAULT_API_URL,
    ask_question,
    check_health,
    delete_session,
    search_docs,
)


def _make_mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestCheckHealth(unittest.TestCase):
    def test_check_health_success_and_url_normalization(self):
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "qdrant_connected": True,
                    "collection_name": "dokumanlar",
                    "collection_exists": True,
                    "points_count": 42,
                    "embedding_model": "BAAI/bge-m3",
                    "rerank_model": "BAAI/bge-reranker-v2-m3",
                    "llm_configured": True,
                    "llm_model": "gpt-4o-mini",
                },
            )

        with _make_mock_client(handler) as client:
            result = check_health(base_url="http://127.0.0.1:8000///", client=client)

        self.assertEqual(captured["method"], "GET")
        self.assertEqual(captured["url"], f"{DEFAULT_API_URL}/health")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["points_count"], 42)

    def test_check_health_connect_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        with _make_mock_client(handler) as client:
            with self.assertRaises(ApiClientError) as ctx:
                check_health(client=client)

        self.assertIsNone(ctx.exception.status_code)
        self.assertIn("FastAPI sunucusuna bağlanılamadı", str(ctx.exception))

    def test_check_health_invalid_json_raises_api_client_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        with _make_mock_client(handler) as client:
            with self.assertRaises(ApiClientError) as ctx:
                check_health(client=client)

        self.assertEqual(ctx.exception.status_code, 200)
        self.assertIn("geçerli JSON", str(ctx.exception))


class TestAskQuestion(unittest.TestCase):
    def test_ask_question_payload_with_session_id(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "session_id": "sess-42",
                    "answer": "Wildcard DNS alt alan adlarını yönlendirir.",
                    "rewritten_query": "Wildcard DNS nedir?",
                    "early_rejected": False,
                    "rejection_reason": None,
                    "chunks": [],
                    "timings": {
                        "rewrite_ms": 12.5,
                        "retrieval_ms": 34.0,
                        "generation_ms": 420.0,
                    },
                },
            )

        with _make_mock_client(handler) as client:
            result = ask_question(
                question="Wildcard DNS nedir?",
                session_id="sess-42",
                top_k=5,
                use_rerank=False,
                use_rewrite=False,
                use_guardrail=True,
                min_rerank_score=-2.5,
                min_dense_score=0.5,
                min_bm25_score=8.5,
                base_url="http://localhost:9000/",
                client=client,
            )

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "http://localhost:9000/ask")
        self.assertEqual(
            captured["payload"],
            {
                "question": "Wildcard DNS nedir?",
                "session_id": "sess-42",
                "top_k": 5,
                "use_rerank": False,
                "use_rewrite": False,
                "use_guardrail": True,
                "min_rerank_score": -2.5,
                "min_dense_score": 0.5,
                "min_bm25_score": 8.5,
            },
        )
        self.assertEqual(result["session_id"], "sess-42")
        self.assertFalse(result["early_rejected"])

    def test_ask_question_omits_session_id_when_none(self):
        captured_payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "session_id": "generated-id",
                    "answer": "Yanıt",
                    "rewritten_query": "Soru",
                    "early_rejected": False,
                    "rejection_reason": None,
                    "chunks": [],
                    "timings": {"rewrite_ms": 0.0, "retrieval_ms": 1.0, "generation_ms": 2.0},
                },
            )

        with _make_mock_client(handler) as client:
            ask_question(question="Soru", client=client)

        self.assertNotIn("session_id", captured_payload)
        self.assertEqual(captured_payload["top_k"], 3)
        self.assertTrue(captured_payload["use_rerank"])
        self.assertTrue(captured_payload["use_rewrite"])
        self.assertTrue(captured_payload["use_guardrail"])
        self.assertEqual(captured_payload["min_rerank_score"], -3.0)
        self.assertEqual(captured_payload["min_dense_score"], 0.45)
        self.assertEqual(captured_payload["min_bm25_score"], 7.0)

    def test_ask_question_502_error_extracts_detail_and_status_code(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                502,
                json={"detail": "Yanıt üretilirken bir sunucu hatası oluştu."},
            )

        with _make_mock_client(handler) as client:
            with self.assertRaises(ApiClientError) as ctx:
                ask_question(question="Test", client=client)

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(str(ctx.exception), "Yanıt üretilirken bir sunucu hatası oluştu.")

    def test_ask_question_422_validation_error_extracts_messages(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422,
                json={
                    "detail": [
                        {
                            "type": "value_error",
                            "loc": ["body", "question"],
                            "msg": "Value error, Soru boş olamaz.",
                        }
                    ]
                },
            )

        with _make_mock_client(handler) as client:
            with self.assertRaises(ApiClientError) as ctx:
                ask_question(question="   ", client=client)

        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Soru boş olamaz.", str(ctx.exception))


class TestSearchDocs(unittest.TestCase):
    def test_search_docs_success(self):
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "query": "docker compose",
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "source": "docker.md",
                            "section": "Compose",
                            "text": "Docker Compose çoklu konteyner yönetir.",
                            "score": 0.91,
                        }
                    ],
                    "retrieval_ms": 18.4,
                },
            )

        with _make_mock_client(handler) as client:
            result = search_docs(
                query="docker compose",
                top_k=4,
                use_rerank=False,
                client=client,
            )

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], f"{DEFAULT_API_URL}/search")
        self.assertEqual(
            captured["payload"],
            {"query": "docker compose", "top_k": 4, "use_rerank": False},
        )
        self.assertEqual(result["retrieval_ms"], 18.4)
        self.assertEqual(len(result["chunks"]), 1)

    def test_search_docs_timeout_raises_api_client_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Timed out", request=request)

        with _make_mock_client(handler) as client:
            with self.assertRaises(ApiClientError) as ctx:
                search_docs(query="yavaş sorgu", client=client)

        self.assertIsNone(ctx.exception.status_code)
        self.assertIn("zaman aşımına uğradı", str(ctx.exception))


class TestDeleteSession(unittest.TestCase):
    def test_delete_session_200_returns_true(self):
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"session_id": "abc_123", "deleted": True})

        with _make_mock_client(handler) as client:
            deleted = delete_session("abc_123", client=client)

        self.assertTrue(deleted)
        self.assertEqual(captured["method"], "DELETE")
        self.assertEqual(captured["url"], f"{DEFAULT_API_URL}/sessions/abc_123")

    def test_delete_session_404_returns_false(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "Oturum bulunamadı."})

        with _make_mock_client(handler) as client:
            deleted = delete_session("nonexistent", client=client)

        self.assertFalse(deleted)

    def test_delete_session_500_raises_api_client_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "Beklenmeyen sunucu hatası."})

        with _make_mock_client(handler) as client:
            with self.assertRaises(ApiClientError) as ctx:
                delete_session("abc_123", client=client)

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(str(ctx.exception), "Beklenmeyen sunucu hatası.")

    def test_delete_session_connect_error_raises_api_client_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        with _make_mock_client(handler) as client:
            with self.assertRaises(ApiClientError) as ctx:
                delete_session("abc_123", client=client)

        self.assertIsNone(ctx.exception.status_code)
        self.assertIn("FastAPI sunucusuna bağlanılamadı", str(ctx.exception))


class TestDefaultApiUrlEnv(unittest.TestCase):
    @staticmethod
    def _load_ui_client_isolated():
        module_path = Path(SRC_DIR) / "ui_client.py"
        spec = importlib.util.spec_from_file_location("_ui_client_env_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_api_url_uses_fallback_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_API_URL", None)
            module = self._load_ui_client_isolated()

        self.assertEqual(module.DEFAULT_API_URL, "http://127.0.0.1:8000")
        self.assertEqual(module.DEFAULT_API_BASE_URL, "http://127.0.0.1:8000")

    def test_default_api_url_reads_and_normalizes_env_var(self):
        with patch.dict(os.environ, {"RAG_API_URL": "  http://api:8000///  "}, clear=False):
            module = self._load_ui_client_isolated()

        self.assertEqual(module.DEFAULT_API_URL, "http://api:8000")
        self.assertEqual(module.DEFAULT_API_BASE_URL, "http://api:8000")

    def test_default_api_url_falls_back_when_env_is_blank_or_slashes(self):
        for raw_value in ("", "   ", "///"):
            with self.subTest(raw_value=raw_value):
                with patch.dict(os.environ, {"RAG_API_URL": raw_value}, clear=False):
                    module = self._load_ui_client_isolated()

                self.assertEqual(module.DEFAULT_API_URL, "http://127.0.0.1:8000")
                self.assertEqual(module.DEFAULT_API_BASE_URL, "http://127.0.0.1:8000")


if __name__ == "__main__":
    unittest.main()
