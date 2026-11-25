import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import agent
import api
import generator
from guardrail import REFUSAL_MESSAGE
from ui_client import ApiClientError, ask_question_stream


SAMPLE_CHUNK = {
    "chunk_id": "dns_chunk_1",
    "source": "dns.md",
    "section": "Wildcard DNS",
    "text": "Wildcard DNS tüm eşleşmeyen alt alan adlarını tek kayda yönlendirir.",
    "score": 0.045,
    "dense_rank": 1,
    "dense_score": 0.86,
    "bm25_rank": 1,
    "bm25_score": 11.4,
}

SECOND_CHUNK = {
    "chunk_id": "ssl_chunk_1",
    "source": "ssl.md",
    "section": "Wildcard Sertifika",
    "text": "Wildcard SSL sertifikası tek seviyeli alt alan adlarını korur.",
    "score": 0.041,
    "dense_rank": 1,
    "dense_score": 0.82,
    "bm25_rank": 1,
    "bm25_score": 10.5,
}

LOW_SCORE_CHUNK = {
    "chunk_id": "low_1",
    "source": "misc.md",
    "section": "Alakasız",
    "text": "Alakasız metin parçası.",
    "score": 0.005,
    "dense_rank": 1,
    "dense_score": 0.12,
    "bm25_rank": 1,
    "bm25_score": 1.2,
}


def _make_completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_stream_chunk(content: str | None = None, *, empty_choices: bool = False, no_delta: bool = False):
    if empty_choices:
        return SimpleNamespace(choices=[])
    if no_delta:
        return SimpleNamespace(choices=[SimpleNamespace(delta=None)])
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


def _parse_sse_text(raw_text: str) -> list[dict]:
    events: list[dict] = []
    current_event: str | None = None
    data_lines: list[str] = []

    for line in raw_text.splitlines():
        if line == "":
            if data_lines:
                events.append(
                    {
                        "event": current_event,
                        "data": json.loads("\n".join(data_lines)),
                    }
                )
            current_event = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            val = line[5:]
            if val.startswith(" "):
                val = val[1:]
            data_lines.append(val)

    if data_lines:
        events.append(
            {
                "event": current_event,
                "data": json.loads("\n".join(data_lines)),
            }
        )
    return events


class TestGeneratorAskRagStream(unittest.TestCase):
    def test_normal_stream_emits_meta_tokens_and_done_skipping_empty_deltas(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = iter(
            [
                _make_stream_chunk(empty_choices=True),
                _make_stream_chunk(no_delta=True),
                _make_stream_chunk(None),
                _make_stream_chunk(""),
                _make_stream_chunk("Wildcard "),
                _make_stream_chunk("DNS yönlendirir."),
            ]
        )

        with patch.object(generator, "client", fake_client), \
             patch.object(generator, "MODEL_NAME", "test-model"), \
             patch.object(generator, "search", return_value=[SAMPLE_CHUNK]) as mock_search:
            events = list(generator.ask_rag_stream("Wildcard DNS nedir?", top_k=3))

        mock_search.assert_called_once_with("Wildcard DNS nedir?", top_k=3, use_rerank=False)
        fake_client.chat.completions.create.assert_called_once()
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertTrue(call_kwargs["stream"])
        self.assertEqual(call_kwargs["model"], "test-model")
        self.assertEqual(call_kwargs["temperature"], 0.1)

        self.assertEqual([e["type"] for e in events], ["meta", "token", "token", "done"])
        meta = events[0]
        self.assertFalse(meta["early_rejected"])
        self.assertIsNone(meta["rejection_reason"])
        self.assertEqual(meta["rewritten_query"], "Wildcard DNS nedir?")
        self.assertEqual(len(meta["chunks"]), 1)
        self.assertEqual(meta["agent_steps"], [])

        self.assertEqual(events[1]["content"], "Wildcard ")
        self.assertEqual(events[2]["content"], "DNS yönlendirir.")

        done = events[3]
        self.assertEqual(done["answer"], "Wildcard DNS yönlendirir.")
        self.assertGreaterEqual(done["generation_ms"], 0.0)

    def test_guardrail_early_reject_skips_llm_and_streams_refusal(self):
        fake_client = MagicMock()

        with patch.object(generator, "client", fake_client), \
             patch.object(generator, "MODEL_NAME", "test-model"), \
             patch.object(generator, "search", return_value=[LOW_SCORE_CHUNK]):
            events = list(generator.ask_rag_stream("Bilinmeyen konu nedir?", use_guardrail=True))

        fake_client.chat.completions.create.assert_not_called()
        self.assertEqual([e["type"] for e in events], ["meta", "token", "done"])
        self.assertTrue(events[0]["early_rejected"])
        self.assertIsNotNone(events[0]["rejection_reason"])
        self.assertEqual(events[1]["content"], REFUSAL_MESSAGE)
        self.assertEqual(events[2]["answer"], REFUSAL_MESSAGE)
        self.assertEqual(events[2]["generation_ms"], 0.0)

    def test_use_agent_true_delegates_to_run_agent_rag_stream(self):
        fake_events = [
            {
                "type": "meta",
                "chunks": [SAMPLE_CHUNK],
                "rewritten_query": "Wildcard DNS",
                "rewrite_ms": 0.0,
                "retrieval_ms": 12.0,
                "early_rejected": False,
                "rejection_reason": None,
                "agent_steps": [],
            },
            {"type": "token", "content": "Ajan yanıtı"},
            {"type": "done", "answer": "Ajan yanıtı", "generation_ms": 25.0},
        ]

        with patch.object(generator, "run_agent_rag_stream", return_value=iter(fake_events)) as mock_agent_stream:
            events = list(
                generator.ask_rag_stream(
                    "Wildcard DNS",
                    use_agent=True,
                    max_agent_steps=4,
                )
            )

        self.assertEqual(events, fake_events)
        mock_agent_stream.assert_called_once()
        self.assertEqual(mock_agent_stream.call_args.kwargs["max_steps"], 4)


class TestAgentRunAgentRagStream(unittest.TestCase):
    def test_multi_step_search_then_stream_synthesis(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _make_completion(
                json.dumps({"thought": "DNS ara", "action": "search", "query": "Wildcard DNS"})
            ),
            _make_completion(
                json.dumps({"thought": "SSL ara", "action": "search", "query": "Wildcard SSL"})
            ),
            _make_completion(
                json.dumps({"thought": "Yeterli", "action": "answer", "query": None})
            ),
            iter(
                [
                    _make_stream_chunk(empty_choices=True),
                    _make_stream_chunk("DNS ve "),
                    _make_stream_chunk(None),
                    _make_stream_chunk("SSL birlikte çalışır."),
                ]
            ),
        ]

        def fake_search(query: str, top_k: int = 3, use_rerank: bool = False):
            if "DNS" in query:
                return [SAMPLE_CHUNK]
            return [SECOND_CHUNK]

        events = list(
            agent.run_agent_rag_stream(
                question="Wildcard DNS ve SSL farkı nedir?",
                top_k=3,
                max_steps=3,
                llm_client=fake_client,
                model_name="test-agent-model",
                search_fn=fake_search,
            )
        )

        self.assertEqual([e["type"] for e in events], ["meta", "token", "token", "done"])
        meta = events[0]
        self.assertFalse(meta["early_rejected"])
        self.assertEqual(len(meta["chunks"]), 2)
        self.assertEqual([s["action"] for s in meta["agent_steps"]], ["search", "search", "answer"])

        self.assertEqual(events[1]["content"], "DNS ve ")
        self.assertEqual(events[2]["content"], "SSL birlikte çalışır.")
        self.assertEqual(events[3]["answer"], "DNS ve SSL birlikte çalışır.")
        self.assertGreaterEqual(events[3]["generation_ms"], 0.0)

        # 4. çağrı sentez akışı olmalı ve stream=True ile çağrılmalı
        self.assertEqual(fake_client.chat.completions.create.call_count, 4)
        synth_kwargs = fake_client.chat.completions.create.call_args_list[3].kwargs
        self.assertTrue(synth_kwargs["stream"])
        self.assertEqual(synth_kwargs["temperature"], 0.1)

    def test_direct_chat_stream_skips_search(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _make_completion(
                json.dumps({"thought": "Selamlaşma", "action": "direct_chat", "query": None})
            ),
            iter([_make_stream_chunk("Merhaba, "), _make_stream_chunk("nasıl yardımcı olabilirim?")]),
        ]
        mock_search = MagicMock()

        events = list(
            agent.run_agent_rag_stream(
                question="Merhaba!",
                llm_client=fake_client,
                model_name="test-agent-model",
                search_fn=mock_search,
            )
        )

        mock_search.assert_not_called()
        self.assertEqual([e["type"] for e in events], ["meta", "token", "token", "done"])
        self.assertEqual(events[0]["chunks"], [])
        self.assertEqual(events[0]["retrieval_ms"], 0.0)
        self.assertEqual(events[0]["agent_steps"][0]["action"], "direct_chat")
        self.assertEqual(events[3]["answer"], "Merhaba, nasıl yardımcı olabilirim?")

        chat_kwargs = fake_client.chat.completions.create.call_args_list[1].kwargs
        self.assertTrue(chat_kwargs["stream"])
        self.assertEqual(chat_kwargs["temperature"], 0.3)

    def test_guardrail_early_reject_stream_skips_synthesis_llm(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            _make_completion(
                json.dumps({"thought": "Ara", "action": "search", "query": "alakasız sorgu"})
            ),
            _make_completion(
                json.dumps({"thought": "Bitir", "action": "answer", "query": None})
            ),
        ]

        events = list(
            agent.run_agent_rag_stream(
                question="Alakasız soru",
                use_guardrail=True,
                llm_client=fake_client,
                model_name="test-agent-model",
                search_fn=lambda q, top_k=3, use_rerank=False: [LOW_SCORE_CHUNK],
            )
        )

        # Sadece 2 planlama çağrısı yapılmış olmalı, sentez çağrılmamalı
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)
        self.assertEqual([e["type"] for e in events], ["meta", "token", "done"])
        self.assertTrue(events[0]["early_rejected"])
        self.assertIsNotNone(events[0]["rejection_reason"])
        self.assertEqual(events[1]["content"], REFUSAL_MESSAGE)
        self.assertEqual(events[2]["answer"], REFUSAL_MESSAGE)

    def test_invalid_max_steps_raises_value_error(self):
        with self.assertRaises(ValueError):
            agent.run_agent_rag_stream("Soru", max_steps=0)


class TestApiAskStreamEndpoint(unittest.TestCase):
    def setUp(self):
        self._orig_store = api.session_store
        api.session_store = api.SessionStore(db_path=":memory:")
        self.client = TestClient(api.app)

    def tearDown(self):
        api.session_store.close()
        api.session_store = self._orig_store

    def test_ask_stream_sse_format_and_session_recording(self):
        agent_step = {
            "step": 1,
            "thought": "DNS dokümanını ara",
            "action": "search",
            "query": "Wildcard DNS",
            "chunks_found": 1,
            "duration_ms": 14.2,
        }

        def fake_stream(**kwargs):
            yield {
                "type": "meta",
                "chunks": [SAMPLE_CHUNK],
                "rewritten_query": "Wildcard DNS nedir?",
                "rewrite_ms": 5.5,
                "retrieval_ms": 18.0,
                "early_rejected": False,
                "rejection_reason": None,
                "agent_steps": [agent_step],
            }
            yield {"type": "token", "content": "Wildcard DNS "}
            yield {"type": "token", "content": "çalışır."}
            yield {"type": "done", "answer": "Wildcard DNS çalışır.", "generation_ms": 42.3}

        with patch.object(api, "ask_rag_stream", side_effect=fake_stream):
            response = self.client.post(
                "/ask/stream",
                json={"question": "Wildcard DNS nedir?", "session_id": "stream-sess-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.headers.get("cache-control"), "no-cache")
        self.assertEqual(response.headers.get("x-accel-buffering"), "no")
        # ensure_ascii=False: Türkçe karakterler kaçış dizisine dönmemeli
        self.assertIn("çalışır.", response.text)

        events = _parse_sse_text(response.text)
        self.assertEqual([e["event"] for e in events], ["meta", "token", "token", "done"])

        meta = events[0]["data"]
        self.assertEqual(meta["session_id"], "stream-sess-1")
        self.assertEqual(meta["rewritten_query"], "Wildcard DNS nedir?")
        self.assertFalse(meta["early_rejected"])
        self.assertIsNone(meta["rejection_reason"])
        self.assertEqual(len(meta["chunks"]), 1)
        self.assertEqual(meta["chunks"][0]["chunk_id"], "dns_chunk_1")
        self.assertEqual(len(meta["agent_steps"]), 1)
        self.assertEqual(meta["agent_steps"][0]["action"], "search")
        self.assertEqual(
            meta["timings"],
            {"rewrite_ms": 5.5, "retrieval_ms": 18.0, "generation_ms": 0.0},
        )

        self.assertEqual(events[1]["data"], {"content": "Wildcard DNS "})
        self.assertEqual(events[2]["data"], {"content": "çalışır."})

        done = events[3]["data"]
        self.assertEqual(done["answer"], "Wildcard DNS çalışır.")
        self.assertEqual(
            done["timings"],
            {"rewrite_ms": 5.5, "retrieval_ms": 18.0, "generation_ms": 42.3},
        )

        # Akış başarıyla tamamlandığı için oturuma kaydedilmiş olmalı
        session = api.session_store.get_session("stream-sess-1")
        self.assertIsNotNone(session)
        self.assertEqual(session["turn_count"], 1)
        self.assertEqual(session["messages"][0]["content"], "Wildcard DNS nedir?")
        self.assertEqual(session["messages"][1]["content"], "Wildcard DNS çalışır.")

    def test_ask_stream_early_rejected_is_not_recorded_in_session(self):
        def fake_stream(**kwargs):
            yield {
                "type": "meta",
                "chunks": [LOW_SCORE_CHUNK],
                "rewritten_query": "Alakasız soru",
                "rewrite_ms": 0.0,
                "retrieval_ms": 9.0,
                "early_rejected": True,
                "rejection_reason": "Skor eşiğin altında",
                "agent_steps": [],
            }
            yield {"type": "token", "content": REFUSAL_MESSAGE}
            yield {"type": "done", "answer": REFUSAL_MESSAGE, "generation_ms": 0.0}

        with patch.object(api, "ask_rag_stream", side_effect=fake_stream):
            response = self.client.post(
                "/ask/stream",
                json={"question": "Alakasız soru", "session_id": "reject-sess"},
            )

        self.assertEqual(response.status_code, 200)
        events = _parse_sse_text(response.text)
        self.assertEqual([e["event"] for e in events], ["meta", "token", "done"])
        self.assertTrue(events[0]["data"]["early_rejected"])
        self.assertIsNone(api.session_store.get_session("reject-sess"))

    def test_ask_stream_midstream_exception_emits_error_without_leaking_secret_or_recording(self):
        secret_token = "sk-live-super-secret-token-999"

        def failing_stream(**kwargs):
            yield {
                "type": "meta",
                "chunks": [SAMPLE_CHUNK],
                "rewritten_query": "Soru",
                "rewrite_ms": 0.0,
                "retrieval_ms": 10.0,
                "early_rejected": False,
                "rejection_reason": None,
                "agent_steps": [],
            }
            yield {"type": "token", "content": "Kısmi "}
            raise RuntimeError(f"Upstream auth failed with key {secret_token}")

        with patch.object(api, "ask_rag_stream", side_effect=failing_stream), \
             patch.object(api.logger, "exception") as mock_log:
            response = self.client.post(
                "/ask/stream",
                json={"question": "Soru", "session_id": "err-sess"},
            )

        self.assertEqual(response.status_code, 200)
        mock_log.assert_called_once()
        self.assertNotIn(secret_token, response.text)
        self.assertNotIn("Upstream auth failed", response.text)

        events = _parse_sse_text(response.text)
        self.assertEqual([e["event"] for e in events], ["meta", "token", "error"])
        self.assertEqual(
            events[2]["data"],
            {"detail": "Yanıt üretilirken bir sunucu hatası oluştu."},
        )
        # Hata durumunda kısmi cevap oturuma kaydedilmemeli
        self.assertIsNone(api.session_store.get_session("err-sess"))

    def test_ask_stream_validation_error_returns_422_before_streaming(self):
        response = self.client.post("/ask/stream", json={"question": "   "})
        self.assertEqual(response.status_code, 422)


class TestUiClientAskQuestionStream(unittest.TestCase):
    def test_parses_meta_token_done_and_ignores_comments(self):
        captured: dict[str, object] = {}
        sse_body = (
            ": keep-alive comment\n"
            "event: meta\n"
            'data: {"session_id": "s-1", "rewritten_query": "Wildcard DNS", '
            '"early_rejected": false, "rejection_reason": null, "chunks": [], '
            '"agent_steps": [], "timings": {"rewrite_ms": 1.0, "retrieval_ms": 2.0, "generation_ms": 0.0}}\n\n'
            "event: token\n"
            'data: {"content": "Merhaba "}\n\n'
            "event: token\n"
            'data: {"content": "dünya"}\n\n'
            "event: done\n"
            'data: {"answer": "Merhaba dünya", "timings": {"rewrite_ms": 1.0, "retrieval_ms": 2.0, "generation_ms": 15.0}}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            events = list(
                ask_question_stream(
                    question="Wildcard DNS",
                    session_id="s-1",
                    use_agent=True,
                    max_agent_steps=4,
                    base_url="http://localhost:8000/",
                    client=client,
                )
            )

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "http://localhost:8000/ask/stream")
        self.assertEqual(captured["payload"]["session_id"], "s-1")
        self.assertTrue(captured["payload"]["use_agent"])
        self.assertEqual(captured["payload"]["max_agent_steps"], 4)

        self.assertEqual([e["event"] for e in events], ["meta", "token", "token", "done"])
        self.assertEqual(events[0]["data"]["session_id"], "s-1")
        self.assertEqual(events[1]["data"]["content"], "Merhaba ")
        self.assertEqual(events[2]["data"]["content"], "dünya")
        self.assertEqual(events[3]["data"]["answer"], "Merhaba dünya")

    def test_sse_error_event_raises_api_client_error_502(self):
        sse_body = (
            "event: meta\n"
            'data: {"session_id": "s-err", "rewritten_query": "q", "early_rejected": false, '
            '"rejection_reason": null, "chunks": [], "agent_steps": [], '
            '"timings": {"rewrite_ms": 0.0, "retrieval_ms": 0.0, "generation_ms": 0.0}}\n\n'
            "event: error\n"
            'data: {"detail": "Yanıt üretilirken bir sunucu hatası oluştu."}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            stream = ask_question_stream(question="q", client=client)
            first = next(stream)
            self.assertEqual(first["event"], "meta")
            with self.assertRaises(ApiClientError) as ctx:
                next(stream)

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertEqual(str(ctx.exception), "Yanıt üretilirken bir sunucu hatası oluştu.")

    def test_http_422_reads_body_and_raises_api_client_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422,
                json={"detail": [{"msg": "Value error, Soru boş olamaz."}]},
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ApiClientError) as ctx:
                list(ask_question_stream(question="   ", client=client))

        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Soru boş olamaz.", str(ctx.exception))

    def test_connect_error_raises_api_client_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ApiClientError) as ctx:
                list(ask_question_stream(question="Soru", client=client))

        self.assertIsNone(ctx.exception.status_code)
        self.assertIn("FastAPI sunucusuna bağlanılamadı", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
