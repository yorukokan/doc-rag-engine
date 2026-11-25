import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import api
import retriever
from guardrail import REFUSAL_MESSAGE


SAMPLE_CHUNK = {
    "chunk_id": "dns_chunk_1",
    "source": "dns.md",
    "section": "Wildcard DNS",
    "text": "Wildcard DNS tüm eşleşmeyen alt alan adlarını tek kayda yönlendirir.",
    "score": 0.045,
    "dense_rank": 1,
    "dense_score": 0.84,
    "bm25_rank": 1,
    "bm25_score": 11.2,
}


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        api.session_store.clear_all()
        self.client = TestClient(api.app)

    def test_health_ok_and_does_not_leak_secrets(self):
        mock_health = {
            "qdrant_connected": True,
            "collection_name": "dokumanlar",
            "collection_exists": True,
            "points_count": 42,
        }
        secret_key = "sk-super-secret-api-key-do-not-leak"
        secret_url = "https://private-llm-gateway.internal/v1"

        with patch.object(api, "get_qdrant_health", return_value=mock_health), \
             patch.object(api, "MODEL_NAME", "test-llm-model"), \
             patch.dict(
                 os.environ,
                 {"LLM_API_KEY": secret_key, "LLM_BASE_URL": secret_url},
                 clear=False,
             ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["qdrant_connected"])
        self.assertTrue(body["collection_exists"])
        self.assertEqual(body["points_count"], 42)
        self.assertEqual(body["embedding_model"], api.EMBEDDING_MODEL)
        self.assertEqual(body["rerank_model"], api.RERANK_MODEL)
        self.assertTrue(body["llm_configured"])
        self.assertEqual(body["llm_model"], "test-llm-model")
        # Sırlar asla yanıtta yer almamalı
        self.assertNotIn(secret_key, response.text)
        self.assertNotIn(secret_url, response.text)
        self.assertNotIn("api_key", body)
        self.assertNotIn("base_url", body)

    def test_health_degraded_when_qdrant_down(self):
        mock_health = {
            "qdrant_connected": False,
            "collection_name": "dokumanlar",
            "collection_exists": False,
            "points_count": None,
        }

        with patch.object(api, "get_qdrant_health", return_value=mock_health), \
             patch.object(api, "MODEL_NAME", None), \
             patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_MODEL_NAME": ""}, clear=False):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "degraded")
        self.assertFalse(body["qdrant_connected"])
        self.assertFalse(body["collection_exists"])
        self.assertIsNone(body["points_count"])
        self.assertFalse(body["llm_configured"])


class TestSearchEndpoint(unittest.TestCase):
    def setUp(self):
        api.session_store.clear_all()
        self.client = TestClient(api.app)

    def test_search_success_returns_chunks_and_retrieval_ms(self):
        with patch.object(api, "search", return_value=[SAMPLE_CHUNK]) as mock_search:
            response = self.client.post(
                "/search",
                json={"query": "  Wildcard DNS  ", "top_k": 3, "use_rerank": False},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["query"], "Wildcard DNS")
        self.assertEqual(len(body["chunks"]), 1)
        self.assertEqual(body["chunks"][0]["chunk_id"], "dns_chunk_1")
        self.assertIsInstance(body["retrieval_ms"], float)
        self.assertGreaterEqual(body["retrieval_ms"], 0.0)
        mock_search.assert_called_once_with(
            query="Wildcard DNS",
            top_k=3,
            rrf_k=10,
            use_rerank=False,
            candidate_k=6,
        )

    def test_search_validation_errors(self):
        # Boş / sadece boşluk sorgu -> 422
        resp_empty = self.client.post("/search", json={"query": "   "})
        self.assertEqual(resp_empty.status_code, 422)

        # top_k = 0 -> 422
        resp_zero = self.client.post("/search", json={"query": "DNS", "top_k": 0})
        self.assertEqual(resp_zero.status_code, 422)

        # top_k = 25 (> 20) -> 422
        resp_high = self.client.post("/search", json={"query": "DNS", "top_k": 25})
        self.assertEqual(resp_high.status_code, 422)

    def test_search_exception_returns_503_without_leaking_details(self):
        secret_detail = "internal-qdrant-path=/srv/secret/token-abc"
        with patch.object(api, "search", side_effect=RuntimeError(secret_detail)), \
             patch.object(api.logger, "exception"):
            response = self.client.post("/search", json={"query": "DNS nedir?"})

        self.assertEqual(response.status_code, 503)
        self.assertNotIn(secret_detail, response.text)
        self.assertEqual(response.json()["detail"], "Arama servisi şu anda kullanılamıyor.")


class TestAskAndSessionEndpoints(unittest.TestCase):
    def setUp(self):
        api.session_store.clear_all()
        self.client = TestClient(api.app)

    def test_multi_turn_ask_passes_history_and_persists_session(self):
        captured_histories = []

        def fake_ask_rag_with_trace(**kwargs):
            captured_histories.append(list(kwargs["history"]))
            turn_num = len(captured_histories)
            return {
                "answer": f"Yanıt #{turn_num} [dns.md - Wildcard DNS]",
                "chunks": [SAMPLE_CHUNK],
                "rewritten_query": kwargs["question"],
                "rewrite_ms": 0.0 if turn_num == 1 else 15.2,
                "retrieval_ms": 12.4,
                "generation_ms": 140.5,
                "early_rejected": False,
                "rejection_reason": None,
            }

        with patch.object(api, "ask_rag_with_trace", side_effect=fake_ask_rag_with_trace):
            # 1. Tur: session_id verilmiyor -> sunucu üretir, history boş gider
            resp1 = self.client.post("/ask", json={"question": "Wildcard DNS nedir?"})
            self.assertEqual(resp1.status_code, 200)
            body1 = resp1.json()
            session_id = body1["session_id"]
            self.assertTrue(len(session_id) > 0)
            self.assertFalse(body1["early_rejected"])
            self.assertEqual(captured_histories[0], [])

            # 2. Tur: aynı session_id ile -> 1. turun geçmişi iletilir
            resp2 = self.client.post(
                "/ask",
                json={"question": "Peki riskleri nelerdir?", "session_id": session_id},
            )
            self.assertEqual(resp2.status_code, 200)
            body2 = resp2.json()
            self.assertEqual(body2["session_id"], session_id)
            self.assertEqual(
                captured_histories[1],
                [
                    {"role": "user", "content": "Wildcard DNS nedir?"},
                    {"role": "assistant", "content": "Yanıt #1 [dns.md - Wildcard DNS]"},
                ],
            )
            self.assertEqual(body2["timings"]["rewrite_ms"], 15.2)

        # GET /sessions/{session_id} -> 2 tur kayıtlı
        get_resp = self.client.get(f"/sessions/{session_id}")
        self.assertEqual(get_resp.status_code, 200)
        session_body = get_resp.json()
        self.assertEqual(session_body["session_id"], session_id)
        self.assertEqual(session_body["turn_count"], 2)
        self.assertEqual(len(session_body["messages"]), 4)

        # DELETE /sessions/{session_id} -> 200
        del_resp = self.client.delete(f"/sessions/{session_id}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.json(), {"session_id": session_id, "deleted": True})

        # Silme sonrası GET ve DELETE -> 404
        self.assertEqual(self.client.get(f"/sessions/{session_id}").status_code, 404)
        self.assertEqual(self.client.delete(f"/sessions/{session_id}").status_code, 404)

    def test_early_rejected_turn_is_not_recorded_in_session(self):
        rejected_trace = {
            "answer": REFUSAL_MESSAGE,
            "chunks": [SAMPLE_CHUNK],
            "rewritten_query": "Kubernetes nasıl kurulur?",
            "rewrite_ms": 0.0,
            "retrieval_ms": 8.0,
            "generation_ms": 0.0,
            "early_rejected": True,
            "rejection_reason": "Arama skorları eşiğin altında",
        }

        with patch.object(api, "ask_rag_with_trace", return_value=rejected_trace):
            resp = self.client.post(
                "/ask",
                json={"question": "Kubernetes nasıl kurulur?", "session_id": "sess_rejected_1"},
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["early_rejected"])
        self.assertEqual(body["answer"], REFUSAL_MESSAGE)
        self.assertEqual(body["rejection_reason"], "Arama skorları eşiğin altında")

        # Oturuma kaydedilmediği için GET 404 dönmeli
        self.assertEqual(self.client.get("/sessions/sess_rejected_1").status_code, 404)

    def test_ask_validation_errors(self):
        self.assertEqual(self.client.post("/ask", json={"question": "   "}).status_code, 422)
        self.assertEqual(
            self.client.post("/ask", json={"question": "DNS?", "top_k": 0}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/ask", json={"question": "DNS?", "top_k": 21}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/ask", json={"question": "DNS?", "session_id": "bad/session id"}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/ask", json={"question": "DNS?", "session_id": "a" * 129}
            ).status_code,
            422,
        )

    def test_ask_exception_returns_502_and_never_leaks_secret_token(self):
        leaked_token = "sk-live-secret-token-987654321"

        with patch.object(
            api,
            "ask_rag_with_trace",
            side_effect=RuntimeError(f"Upstream auth failed with bearer {leaked_token}"),
        ), patch.object(api.logger, "exception"):
            response = self.client.post("/ask", json={"question": "Wildcard DNS nedir?"})

        self.assertEqual(response.status_code, 502)
        self.assertNotIn(leaked_token, response.text)
        self.assertEqual(
            response.json()["detail"],
            "Yanıt üretilirken bir sunucu hatası oluştu.",
        )

    def test_nonexistent_session_returns_404(self):
        self.assertEqual(self.client.get("/sessions/olmayan_oturum").status_code, 404)
        self.assertEqual(self.client.delete("/sessions/olmayan_oturum").status_code, 404)


class TestSessionStoreAndQdrantHealth(unittest.TestCase):
    def test_session_store_lru_eviction(self):
        store = api.SessionStore(max_sessions=2, default_max_turns=3)
        store.record_turn("s1", "q1", "a1")
        store.record_turn("s2", "q2", "a2")
        # s1'e erişerek onu en güncel hale getir -> s2 en eski olur
        self.assertEqual(len(store.get_history("s1")), 2)
        store.record_turn("s3", "q3", "a3")

        self.assertIsNotNone(store.get_session("s1"))
        self.assertIsNone(store.get_session("s2"))
        self.assertIsNotNone(store.get_session("s3"))

    def test_get_qdrant_health_success_and_error(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = SimpleNamespace(points_count=17)

        with patch.object(retriever, "QdrantClient", return_value=mock_client):
            health = retriever.get_qdrant_health()

        self.assertEqual(
            health,
            {
                "qdrant_connected": True,
                "collection_name": retriever.COLLECTION_NAME,
                "collection_exists": True,
                "points_count": 17,
            },
        )
        mock_client.close.assert_called_once()

        with patch.object(retriever, "QdrantClient", side_effect=OSError("lock error")), \
             patch.object(retriever.logger, "warning"):
            err_health = retriever.get_qdrant_health()

        self.assertEqual(
            err_health,
            {
                "qdrant_connected": False,
                "collection_name": retriever.COLLECTION_NAME,
                "collection_exists": False,
                "points_count": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
