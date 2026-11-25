import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from qdrant_client import models

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import agent
import api
import generator
import retriever
import ui_client
from session_store import SqliteSessionStore


SAMPLE_CHUNK = {
    "chunk_id": "dns_chunk_1",
    "source": "dns.md",
    "section": "Wildcard DNS",
    "text": "Wildcard DNS tüm eşleşmeyen alt alan adlarını tek kayda yönlendirir.",
    "score": 0.045,
    "dense_rank": 1,
    "dense_score": 0.85,
    "bm25_rank": 1,
    "bm25_score": 11.2,
}


def _make_completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_fake_embedder():
    class _FakeVec:
        def tolist(self):
            return [0.1, 0.2, 0.3]

    embedder = MagicMock()
    embedder.embed.return_value = [_FakeVec()]
    return embedder


class TestSqliteSessionStore(unittest.TestCase):
    def test_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "nested" / "sessions.db"

            store1 = SqliteSessionStore(db_path=db_file, max_sessions=10, default_max_turns=5)
            # Lazy init: ilk DB erişimine kadar dosya ve üst dizin oluşturulmamalı
            self.assertFalse(db_file.exists())
            self.assertFalse(db_file.parent.exists())
            store1.close()  # Açılmamış bağlantıda close() güvenli no-op olmalı
            self.assertFalse(db_file.exists())

            store1.record_turn("sess-persist", "DNS TTL nedir?", "TTL önbellek süresidir.")
            store1.record_turn("sess-persist", "Düşük olursa?", "Sorgu yükü artar.")
            store1.close()

            self.assertTrue(db_file.exists())

            store2 = SqliteSessionStore(db_path=db_file, max_sessions=10, default_max_turns=5)
            try:
                sess = store2.get_session("sess-persist")
                self.assertIsNotNone(sess)
                self.assertEqual(sess["session_id"], "sess-persist")
                self.assertEqual(sess["turn_count"], 2)
                self.assertEqual(sess["max_turns"], 5)
                self.assertEqual(
                    sess["messages"],
                    [
                        {"role": "user", "content": "DNS TTL nedir?"},
                        {"role": "assistant", "content": "TTL önbellek süresidir."},
                        {"role": "user", "content": "Düşük olursa?"},
                        {"role": "assistant", "content": "Sorgu yükü artar."},
                    ],
                )
            finally:
                store2.close()

    def test_lru_access_seq_eviction(self):
        store = SqliteSessionStore(db_path=":memory:", max_sessions=2, default_max_turns=3)
        try:
            store.record_turn("s1", "q1", "a1")
            store.record_turn("s2", "q2", "a2")
            # s1'e erişerek access_seq'ini artır -> s2 tahliye adayı olur
            self.assertEqual(len(store.get_history("s1")), 2)
            store.record_turn("s3", "q3", "a3")

            self.assertIsNotNone(store.get_session("s1"))
            self.assertIsNone(store.get_session("s2"))
            self.assertIsNotNone(store.get_session("s3"))

            # ON DELETE CASCADE: tahliye edilen s2'nin mesaj satırları da silinmiş olmalı
            s2_msg_count = store._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                ("s2",),
            ).fetchone()[0]
            self.assertEqual(s2_msg_count, 0)
        finally:
            store.close()

    def test_max_turns_sliding_window(self):
        store = SqliteSessionStore(db_path=":memory:", max_sessions=10, default_max_turns=2)
        try:
            store.record_turn("s1", "q1", "a1")
            store.record_turn("s1", "q2", "a2")
            store.record_turn("s1", "q3", "a3")

            sess = store.get_session("s1")
            self.assertEqual(sess["turn_count"], 2)
            self.assertEqual(
                sess["messages"],
                [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                    {"role": "user", "content": "q3"},
                    {"role": "assistant", "content": "a3"},
                ],
            )
        finally:
            store.close()

    def test_list_sessions_order_preview_delete_and_clear(self):
        store = SqliteSessionStore(db_path=":memory:", max_sessions=10, default_max_turns=5)
        try:
            long_q = "   Satır1\n\nSatır2   " + ("x" * 100)
            store.record_turn("s1", long_q, "cevap 1")
            store.record_turn("s2", "Kısa soru", "cevap 2")

            listed = store.list_sessions(limit=10)
            self.assertEqual([item["session_id"] for item in listed], ["s2", "s1"])
            self.assertEqual(listed[0]["preview"], "Kısa soru")
            self.assertEqual(listed[0]["turn_count"], 1)
            self.assertLessEqual(len(listed[1]["preview"]), 80)
            self.assertNotIn("\n", listed[1]["preview"])
            self.assertTrue(listed[1]["preview"].startswith("Satır1 Satır2"))

            # s1'e erişince listenin başına geçmeli
            store.get_history("s1")
            listed_after_touch = store.list_sessions(limit=1)
            self.assertEqual([item["session_id"] for item in listed_after_touch], ["s1"])

            # delete_session + ON DELETE CASCADE doğrulaması
            self.assertTrue(store.delete_session("s1"))
            self.assertFalse(store.delete_session("s1"))
            self.assertIsNone(store.get_session("s1"))
            s1_msg_count = store._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                ("s1",),
            ).fetchone()[0]
            self.assertEqual(s1_msg_count, 0)

            # clear_all
            store.clear_all()
            self.assertEqual(store.list_sessions(), [])
            self.assertEqual(store.get_history("s2"), [])
        finally:
            store.close()

    def test_invalid_init_arguments_raise_value_error(self):
        with self.assertRaises(ValueError):
            SqliteSessionStore(max_sessions=0)
        with self.assertRaises(ValueError):
            SqliteSessionStore(max_sessions=True)
        with self.assertRaises(ValueError):
            SqliteSessionStore(default_max_turns=0)
        with self.assertRaises(ValueError):
            SqliteSessionStore(default_max_turns=False)


class TestRetrieverSourceFilter(unittest.TestCase):
    def test_search_passes_match_any_filter_to_query_and_scroll(self):
        point = SimpleNamespace(
            payload={
                "chunk_id": "dns_1",
                "text": "Wildcard DNS kaydı açıklaması",
                "source": "dns.md",
                "section": "Wildcard DNS",
            },
            score=0.88,
        )
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.query_points.return_value = SimpleNamespace(points=[point])
        mock_client.scroll.return_value = ([point], None)

        with patch.object(retriever, "get_embedder", return_value=_make_fake_embedder()), \
             patch.object(retriever, "QdrantClient", return_value=mock_client):
            results = retriever.search(
                "Wildcard DNS",
                top_k=2,
                source_filter=["  dns.md ", "", "ssl.md"],
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "dns.md")

        # query_points çağrısında query_filter kontrolü
        qp_kwargs = mock_client.query_points.call_args.kwargs
        self.assertIn("query_filter", qp_kwargs)
        q_filter = qp_kwargs["query_filter"]
        self.assertIsInstance(q_filter, models.Filter)
        self.assertEqual(len(q_filter.must), 1)
        self.assertEqual(q_filter.must[0].key, "source")
        self.assertEqual(q_filter.must[0].match, models.MatchAny(any=["dns.md", "ssl.md"]))

        # scroll çağrısında scroll_filter kontrolü
        sc_kwargs = mock_client.scroll.call_args.kwargs
        self.assertIn("scroll_filter", sc_kwargs)
        self.assertEqual(sc_kwargs["scroll_filter"], q_filter)

    def test_search_empty_source_filter_returns_empty_without_qdrant(self):
        with patch.object(retriever, "get_embedder") as mock_embed, \
             patch.object(retriever, "QdrantClient") as mock_qdrant:
            self.assertEqual(retriever.search("DNS", source_filter=[]), [])
            self.assertEqual(retriever.search("DNS", source_filter=["   ", ""]), [])

        mock_embed.assert_not_called()
        mock_qdrant.assert_not_called()

    def test_search_none_source_filter_passes_no_filter(self):
        point = SimpleNamespace(
            payload={
                "chunk_id": "dns_1",
                "text": "Wildcard DNS kaydı",
                "source": "dns.md",
                "section": "DNS",
            },
            score=0.80,
        )
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.query_points.return_value = SimpleNamespace(points=[point])
        mock_client.scroll.return_value = ([point], None)

        with patch.object(retriever, "get_embedder", return_value=_make_fake_embedder()), \
             patch.object(retriever, "QdrantClient", return_value=mock_client):
            results = retriever.search("DNS", top_k=1, source_filter=None)

        self.assertEqual(len(results), 1)
        self.assertNotIn("query_filter", mock_client.query_points.call_args.kwargs)
        self.assertNotIn("scroll_filter", mock_client.scroll.call_args.kwargs)


class TestGeneratorAndAgentSourceFilter(unittest.TestCase):
    def test_generator_forwards_source_filter_only_when_not_none(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_completion("Yanıt [dns.md - Wildcard DNS]")

        with patch.object(generator, "client", mock_client), \
             patch.object(generator, "search", return_value=[SAMPLE_CHUNK]) as mock_search:
            generator.ask_rag_with_trace(
                "Wildcard DNS nedir?",
                top_k=2,
                use_rerank=False,
                source_filter=["dns.md"],
            )
            mock_search.assert_called_once_with(
                "Wildcard DNS nedir?",
                top_k=2,
                use_rerank=False,
                source_filter=["dns.md"],
            )

        with patch.object(generator, "client", mock_client), \
             patch.object(generator, "search", return_value=[SAMPLE_CHUNK]) as mock_search:
            generator.ask_rag_with_trace(
                "Wildcard DNS nedir?",
                top_k=2,
                use_rerank=False,
                source_filter=None,
            )
            mock_search.assert_called_once_with(
                "Wildcard DNS nedir?",
                top_k=2,
                use_rerank=False,
            )

    def test_agent_forwards_source_filter_in_search_and_fallback(self):
        # 1) Normal search aksiyonu
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_completion('{"thought": "DNS ara", "action": "search", "query": "Wildcard DNS"}'),
            _make_completion('{"thought": "Yeterli", "action": "answer", "query": null}'),
            _make_completion("Cevap [dns.md - Wildcard DNS]"),
        ]
        captured_calls = []

        def fake_search(query, **kwargs):
            captured_calls.append((query, kwargs))
            return [SAMPLE_CHUNK]

        res = agent.run_agent_rag(
            question="Wildcard DNS nedir?",
            top_k=2,
            use_rerank=False,
            llm_client=mock_client,
            model_name="test-model",
            search_fn=fake_search,
            source_filter=["dns.md"],
        )
        self.assertEqual(res["answer"], "Cevap [dns.md - Wildcard DNS]")
        self.assertEqual(
            captured_calls,
            [("Wildcard DNS", {"top_k": 2, "use_rerank": False, "source_filter": ["dns.md"]})],
        )

        # 2) Fallback search (geçersiz JSON plan)
        fallback_client = MagicMock()
        fallback_client.chat.completions.create.side_effect = [
            _make_completion("geçersiz json çıktı"),
            _make_completion("Fallback cevap [dns.md - Wildcard DNS]"),
        ]
        fallback_calls = []

        def fake_fallback_search(query, **kwargs):
            fallback_calls.append((query, kwargs))
            return [SAMPLE_CHUNK]

        with self.assertWarns(RuntimeWarning):
            res2 = agent.run_agent_rag(
                question="Wildcard DNS nedir?",
                top_k=2,
                use_rerank=False,
                llm_client=fallback_client,
                model_name="test-model",
                search_fn=fake_fallback_search,
                source_filter=["dns.md"],
            )
        self.assertEqual(res2["answer"], "Fallback cevap [dns.md - Wildcard DNS]")
        self.assertEqual(
            fallback_calls,
            [("Wildcard DNS nedir?", {"top_k": 2, "use_rerank": False, "source_filter": ["dns.md"]})],
        )


class TestApiV5Endpoints(unittest.TestCase):
    def setUp(self):
        self._orig_store = api.session_store
        api.session_store = SqliteSessionStore(db_path=":memory:")
        self.client = TestClient(api.app)

    def tearDown(self):
        api.session_store.close()
        api.session_store = self._orig_store

    def test_get_sessions_list_and_detail(self):
        api.session_store.record_turn("sess-1", "İlk soru nedir?", "İlk cevap")
        api.session_store.record_turn("sess-2", "İkinci soru nedir?", "İkinci cevap")

        resp = self.client.get("/sessions?limit=10")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["total_sessions"], 2)
        self.assertEqual([s["session_id"] for s in body["sessions"]], ["sess-2", "sess-1"])
        self.assertEqual(body["sessions"][0]["preview"], "İkinci soru nedir?")
        self.assertEqual(body["sessions"][0]["turn_count"], 1)

        detail = self.client.get("/sessions/sess-1")
        self.assertEqual(detail.status_code, 200)
        detail_body = detail.json()
        self.assertEqual(detail_body["session_id"], "sess-1")
        self.assertEqual(len(detail_body["messages"]), 2)

    def test_post_ask_and_search_forward_source_filter(self):
        fake_trace = {
            "answer": "DNS yanıtı",
            "chunks": [SAMPLE_CHUNK],
            "rewritten_query": "Wildcard DNS",
            "rewrite_ms": 0.0,
            "retrieval_ms": 12.0,
            "generation_ms": 40.0,
            "early_rejected": False,
            "rejection_reason": None,
            "agent_steps": [],
        }

        with patch.object(api, "ask_rag_with_trace", return_value=fake_trace) as mock_ask:
            resp = self.client.post(
                "/ask",
                json={
                    "question": "Wildcard DNS nedir?",
                    "source_filter": [" dns.md ", "", "ssl.md"],
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_ask.call_args.kwargs["source_filter"], ["dns.md", "ssl.md"])

        with patch.object(api, "search", return_value=[SAMPLE_CHUNK]) as mock_search:
            s_resp = self.client.post(
                "/search",
                json={
                    "query": "Wildcard DNS",
                    "source_filter": ["dns.md", "  "],
                },
            )

        self.assertEqual(s_resp.status_code, 200)
        self.assertEqual(mock_search.call_args.kwargs["source_filter"], ["dns.md"])


class TestUiClientV5(unittest.TestCase):
    def test_list_sessions_and_get_session(self):
        captured = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.method, str(request.url)))
            if request.url.path == "/sessions":
                return httpx.Response(
                    200,
                    json={
                        "sessions": [
                            {
                                "session_id": "s1",
                                "turn_count": 1,
                                "max_turns": 5,
                                "created_at": 1.0,
                                "updated_at": 2.0,
                                "preview": "Merhaba",
                            }
                        ],
                        "total_sessions": 1,
                    },
                )
            if request.url.path == "/sessions/s1":
                return httpx.Response(
                    200,
                    json={
                        "session_id": "s1",
                        "turn_count": 1,
                        "max_turns": 5,
                        "messages": [
                            {"role": "user", "content": "Merhaba"},
                            {"role": "assistant", "content": "Selam"},
                        ],
                    },
                )
            return httpx.Response(404, json={"detail": "Not found"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            listed = ui_client.list_sessions(limit=25, base_url="http://testserver", client=client)
            detail = ui_client.get_session("s1", base_url="http://testserver", client=client)

        self.assertEqual(listed["total_sessions"], 1)
        self.assertEqual(detail["session_id"], "s1")
        self.assertEqual(
            captured,
            [
                ("GET", "http://testserver/sessions?limit=25"),
                ("GET", "http://testserver/sessions/s1"),
            ],
        )

    def test_source_filter_payload_inclusion_and_omission(self):
        payload_none = ui_client._build_ask_payload("Soru", source_filter=None)
        self.assertNotIn("source_filter", payload_none)

        payload_empty = ui_client._build_ask_payload("Soru", source_filter=[])
        self.assertNotIn("source_filter", payload_empty)

        payload_filled = ui_client._build_ask_payload("Soru", source_filter=["dns.md", "ssl.md"])
        self.assertEqual(payload_filled["source_filter"], ["dns.md", "ssl.md"])

        received_bodies = []

        def handler(request: httpx.Request) -> httpx.Response:
            received_bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"query": "DNS", "chunks": [], "retrieval_ms": 1.0})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            ui_client.search_docs("DNS", source_filter=None, base_url="http://testserver", client=client)
            ui_client.search_docs("DNS", source_filter=[], base_url="http://testserver", client=client)
            ui_client.search_docs("DNS", source_filter=["dns.md"], base_url="http://testserver", client=client)

        self.assertNotIn("source_filter", received_bodies[0])
        self.assertNotIn("source_filter", received_bodies[1])
        self.assertEqual(received_bodies[2]["source_filter"], ["dns.md"])


if __name__ == "__main__":
    unittest.main()
