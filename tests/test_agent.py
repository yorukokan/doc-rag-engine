import io
import json
import sys
import unittest
import warnings
from contextlib import redirect_stdout
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
import cli
import generator
from guardrail import REFUSAL_MESSAGE
from ui_client import ask_question


def _make_llm_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_chunk(
    chunk_id: str | None,
    source: str,
    section: str,
    text: str,
    score: float = 0.05,
    dense_score: float = 0.85,
    bm25_score: float = 12.0,
    rerank_score: float | None = None,
) -> dict:
    chunk = {
        "chunk_id": chunk_id,
        "source": source,
        "section": section,
        "text": text,
        "score": score,
        "dense_rank": 1,
        "dense_score": dense_score,
        "bm25_rank": 1,
        "bm25_score": bm25_score,
    }
    if rerank_score is not None:
        chunk["rerank_score"] = rerank_score
    if chunk_id is None:
        chunk.pop("chunk_id")
    return chunk


class TestParseAgentAction(unittest.TestCase):
    def test_parses_plain_json_actions(self):
        raw_search = json.dumps(
            {"thought": "DNS bilgisi lazım", "action": "search", "query": "Wildcard DNS"}
        )
        self.assertEqual(
            agent._parse_agent_action(raw_search),
            {"thought": "DNS bilgisi lazım", "action": "search", "query": "Wildcard DNS"},
        )

        raw_answer = json.dumps(
            {"thought": "Yeterli bilgi var", "action": "answer", "query": "ignored"}
        )
        self.assertEqual(
            agent._parse_agent_action(raw_answer),
            {"thought": "Yeterli bilgi var", "action": "answer", "query": None},
        )

        raw_chat = json.dumps(
            {"thought": "Selamlaşma", "action": "direct_chat", "query": None}
        )
        self.assertEqual(
            agent._parse_agent_action(raw_chat),
            {"thought": "Selamlaşma", "action": "direct_chat", "query": None},
        )

    def test_parses_markdown_fenced_json_and_surrounding_text(self):
        fenced = (
            "```json\n"
            '{"thought": "Docker ağını ara", "action": "search", "query": "Docker bridge network"}\n'
            "```"
        )
        parsed = agent._parse_agent_action(fenced)
        self.assertEqual(parsed["action"], "search")
        self.assertEqual(parsed["query"], "Docker bridge network")

        wrapped = 'Kararım şu: {"thought": "Tamam", "action": "answer", "query": null} Teşekkürler.'
        parsed_wrapped = agent._parse_agent_action(wrapped)
        self.assertEqual(
            parsed_wrapped,
            {"thought": "Tamam", "action": "answer", "query": None},
        )

    def test_rejects_invalid_action_or_malformed_text(self):
        self.assertIsNone(
            agent._parse_agent_action('{"thought": "x", "action": "delete_all", "query": null}')
        )
        self.assertIsNone(agent._parse_agent_action("bu geçerli bir json değil"))
        self.assertIsNone(agent._parse_agent_action(""))
        self.assertIsNone(agent._parse_agent_action("[]"))


class TestMergeAndDeduplicateChunks(unittest.TestCase):
    def test_keeps_higher_score_on_duplicate_chunk_id_and_sorts_descending(self):
        c1_low = _make_chunk("c1", "dns.md", "Wildcard", "Metin 1", score=0.02)
        c1_high = _make_chunk("c1", "dns.md", "Wildcard", "Metin 1", score=0.09)
        c2 = _make_chunk("c2", "docker.md", "Network", "Metin 2", score=0.05)

        merged = agent.merge_and_deduplicate_chunks([c1_low, c2], [c1_high], use_rerank=False)
        self.assertEqual([c["chunk_id"] for c in merged], ["c1", "c2"])
        self.assertEqual(merged[0]["score"], 0.09)

    def test_uses_rerank_score_when_use_rerank_true(self):
        c1 = _make_chunk("c1", "dns.md", "A", "Metin A", score=0.09, rerank_score=-1.5)
        c1_better = _make_chunk("c1", "dns.md", "A", "Metin A", score=0.01, rerank_score=2.4)
        c2 = _make_chunk("c2", "docker.md", "B", "Metin B", score=0.08, rerank_score=4.1)

        merged = agent.merge_and_deduplicate_chunks([c1], [c1_better, c2], use_rerank=True)
        self.assertEqual([c["chunk_id"] for c in merged], ["c2", "c1"])
        self.assertEqual(merged[1]["rerank_score"], 2.4)

    def test_fallback_key_without_chunk_id(self):
        no_id_low = _make_chunk(None, "dns.md", "TTL", "TTL değeri önbellek süresini belirler.", score=0.03)
        no_id_high = _make_chunk(None, "dns.md", "TTL", "TTL değeri önbellek süresini belirler.", score=0.07)
        other = _make_chunk(None, "dns.md", "SOA", "SOA kaydı yetkili sunucuyu belirtir.", score=0.05)

        merged = agent.merge_and_deduplicate_chunks([no_id_low], [no_id_high, other])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["section"], "TTL")
        self.assertEqual(merged[0]["score"], 0.07)
        self.assertEqual(merged[1]["section"], "SOA")


class TestRunAgentRag(unittest.TestCase):
    def test_invalid_max_steps_raises_value_error(self):
        with self.assertRaises(ValueError):
            agent.run_agent_rag("test", max_steps=0)
        with self.assertRaises(ValueError):
            agent.run_agent_rag("test", max_steps=-2)
        with self.assertRaises(ValueError):
            agent.run_agent_rag("test", max_steps=True)

    def test_multi_step_search_merges_two_documents_and_synthesizes(self):
        dns_chunk = _make_chunk(
            "dns_1", "dns.md", "Wildcard DNS", "Wildcard DNS alt alan adlarını yönlendirir.", score=0.08
        )
        docker_chunk = _make_chunk(
            "docker_1", "docker.md", "Compose", "Docker Compose çoklu servis çalıştırır.", score=0.07
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response(
                '{"thought": "Önce DNS bilgisini ara", "action": "search", "query": "Wildcard DNS"}'
            ),
            _make_llm_response(
                '{"thought": "Şimdi Docker Compose ara", "action": "search", "query": "Docker Compose"}'
            ),
            _make_llm_response(
                '{"thought": "Her iki konu da toplandı", "action": "answer", "query": null}'
            ),
            _make_llm_response(
                "Wildcard DNS yönlendirme yapar [dns.md - Wildcard DNS] ve Docker Compose servis yönetir [docker.md - Compose]."
            ),
        ]

        search_queries = []

        def fake_search(query, top_k=3, use_rerank=False):
            search_queries.append(query)
            if "DNS" in query:
                return [dns_chunk]
            return [docker_chunk]

        result = agent.run_agent_rag(
            question="Wildcard DNS ve Docker Compose nedir?",
            top_k=1,
            max_steps=3,
            llm_client=mock_client,
            model_name="test-model",
            search_fn=fake_search,
        )

        self.assertEqual(search_queries, ["Wildcard DNS", "Docker Compose"])
        self.assertFalse(result["early_rejected"])
        self.assertIsNone(result["rejection_reason"])
        # Çoklu arama olduğu için top_k * 2 = 2 chunk korunmalı
        self.assertEqual(len(result["chunks"]), 2)
        self.assertEqual({c["source"] for c in result["chunks"]}, {"dns.md", "docker.md"})
        self.assertEqual(len(result["agent_steps"]), 3)
        self.assertEqual([s["action"] for s in result["agent_steps"]], ["search", "search", "answer"])
        self.assertEqual([s["step"] for s in result["agent_steps"]], [1, 2, 3])
        self.assertIn("[dns.md - Wildcard DNS]", result["answer"])
        self.assertIn("[docker.md - Compose]", result["answer"])

        # Sentez çağrısında her iki doküman bağlamının yer aldığını doğrula
        synth_call_messages = mock_client.chat.completions.create.call_args_list[-1].kwargs["messages"]
        user_msg = synth_call_messages[-1]["content"]
        self.assertIn("dns.md - Wildcard DNS", user_msg)
        self.assertIn("docker.md - Compose", user_msg)

    def test_max_steps_limit_forces_synthesis_without_infinite_loop(self):
        c1 = _make_chunk("c1", "dns.md", "B1", "İçerik 1", score=0.06)
        c2 = _make_chunk("c2", "dns.md", "B2", "İçerik 2", score=0.05)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response('{"thought": "Ara 1", "action": "search", "query": "sorgu 1"}'),
            _make_llm_response('{"thought": "Ara 2", "action": "search", "query": "sorgu 2"}'),
            _make_llm_response("Zorunlu sentez yanıtı [dns.md - B1]"),
        ]
        mock_search = MagicMock(side_effect=[[c1], [c2]])

        result = agent.run_agent_rag(
            question="Karmaşık soru",
            top_k=3,
            max_steps=2,
            llm_client=mock_client,
            model_name="test-model",
            search_fn=mock_search,
        )

        self.assertEqual(mock_search.call_count, 2)
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        self.assertEqual(len(result["agent_steps"]), 2)
        self.assertEqual(result["answer"], "Zorunlu sentez yanıtı [dns.md - B1]")

    def test_duplicate_query_breaks_loop_stagnation_protection(self):
        c1 = _make_chunk("c1", "dns.md", "B1", "İçerik 1", score=0.06)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response('{"thought": "Ara", "action": "search", "query": "Wildcard DNS"}'),
            _make_llm_response('{"thought": "Tekrar ara", "action": "search", "query": "  wildcard   dns "}'),
            _make_llm_response("Sentez yanıtı [dns.md - B1]"),
        ]
        mock_search = MagicMock(return_value=[c1])

        result = agent.run_agent_rag(
            question="Wildcard DNS nedir?",
            top_k=3,
            max_steps=4,
            llm_client=mock_client,
            model_name="test-model",
            search_fn=mock_search,
        )

        # Aynı sorgu tekrarlandığı için search yalnızca 1 kez çağrılmalı
        self.assertEqual(mock_search.call_count, 1)
        self.assertEqual(len(result["agent_steps"]), 1)
        self.assertEqual(result["answer"], "Sentez yanıtı [dns.md - B1]")

    def test_malformed_json_warns_and_runs_fallback_search(self):
        c1 = _make_chunk("c1", "dns.md", "B1", "Fallback içeriği", score=0.06)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response("Üzgünüm JSON üretemedim!"),
            _make_llm_response("Fallback sonrası yanıt [dns.md - B1]"),
        ]
        mock_search = MagicMock(return_value=[c1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = agent.run_agent_rag(
                question="Wildcard DNS nedir?",
                top_k=3,
                max_steps=3,
                llm_client=mock_client,
                model_name="test-model",
                search_fn=mock_search,
            )

        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        mock_search.assert_called_once_with("Wildcard DNS nedir?", top_k=3, use_rerank=False)
        self.assertEqual(len(result["agent_steps"]), 1)
        self.assertEqual(result["agent_steps"][0]["action"], "fallback")
        self.assertEqual(result["agent_steps"][0]["query"], "Wildcard DNS nedir?")
        self.assertEqual(result["answer"], "Fallback sonrası yanıt [dns.md - B1]")

    def test_direct_chat_skips_search_and_returns_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response(
                '{"thought": "Kullanıcı selam veriyor", "action": "direct_chat", "query": null}'
            ),
            _make_llm_response("Merhaba! Size dokümanlar hakkında nasıl yardımcı olabilirim?"),
        ]
        mock_search = MagicMock()

        result = agent.run_agent_rag(
            question="Merhaba!",
            top_k=3,
            max_steps=3,
            llm_client=mock_client,
            model_name="test-model",
            search_fn=mock_search,
        )

        mock_search.assert_not_called()
        self.assertFalse(result["early_rejected"])
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["retrieval_ms"], 0.0)
        self.assertEqual(len(result["agent_steps"]), 1)
        self.assertEqual(result["agent_steps"][0]["action"], "direct_chat")
        self.assertIn("Merhaba", result["answer"])

    def test_guardrail_skips_synthesis_llm_and_returns_agent_steps(self):
        low_chunk = _make_chunk(
            "low_1", "dns.md", "Alakasız", "Alakasız metin", score=0.01, dense_score=0.12, bm25_score=1.0
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response(
                '{"thought": "Kuantum ara", "action": "search", "query": "kuantum bilgisayar"}'
            ),
            _make_llm_response(
                '{"thought": "Bitir", "action": "answer", "query": null}'
            ),
        ]
        mock_search = MagicMock(return_value=[low_chunk])

        result = agent.run_agent_rag(
            question="Kuantum bilgisayar nasıl çalışır?",
            top_k=3,
            use_guardrail=True,
            max_steps=3,
            llm_client=mock_client,
            model_name="test-model",
            search_fn=mock_search,
        )

        # Yalnızca 2 planlayıcı çağrısı yapılmış olmalı, 3. (sentez) çağrısı atlanmalı
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        self.assertTrue(result["early_rejected"])
        self.assertEqual(result["answer"], REFUSAL_MESSAGE)
        self.assertIsNotNone(result["rejection_reason"])
        self.assertEqual(len(result["agent_steps"]), 2)
        self.assertEqual([s["action"] for s in result["agent_steps"]], ["search", "answer"])


class TestAgentIntegrations(unittest.TestCase):
    def setUp(self):
        self._orig_store = api.session_store
        api.session_store = api.SessionStore(db_path=":memory:")
        self.client = TestClient(api.app)

    def tearDown(self):
        api.session_store.close()
        api.session_store = self._orig_store

    def test_generator_delegates_to_run_agent_rag_when_use_agent_true(self):
        fake_trace = {
            "answer": "Ajan yanıtı",
            "chunks": [],
            "rewritten_query": "Soru",
            "rewrite_ms": 0.0,
            "retrieval_ms": 10.0,
            "generation_ms": 50.0,
            "early_rejected": False,
            "rejection_reason": None,
            "agent_steps": [
                {
                    "step": 1,
                    "thought": "Selamlaşma",
                    "action": "direct_chat",
                    "query": None,
                    "chunks_found": 0,
                    "duration_ms": 12.3,
                }
            ],
        }
        with patch.object(generator, "run_agent_rag", return_value=fake_trace) as mock_agent:
            trace = generator.ask_rag_with_trace("Merhaba", use_agent=True, max_agent_steps=4)

        mock_agent.assert_called_once()
        self.assertEqual(mock_agent.call_args.kwargs["max_steps"], 4)
        self.assertEqual(trace["agent_steps"][0]["action"], "direct_chat")

    def test_api_ask_with_use_agent_true_returns_agent_steps_and_validates_max_steps(self):
        sample_chunk = _make_chunk("c1", "dns.md", "Wildcard", "Metin", score=0.07)
        captured_kwargs = {}

        def fake_ask(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "answer": "Ajan cevabı [dns.md - Wildcard]",
                "chunks": [sample_chunk],
                "rewritten_query": kwargs["question"],
                "rewrite_ms": 0.0,
                "retrieval_ms": 14.5,
                "generation_ms": 88.0,
                "early_rejected": False,
                "rejection_reason": None,
                "agent_steps": [
                    {
                        "step": 1,
                        "thought": "Wildcard DNS ara",
                        "action": "search",
                        "query": "Wildcard DNS",
                        "chunks_found": 1,
                        "duration_ms": 15.2,
                    },
                    {
                        "step": 2,
                        "thought": "Yeterli bilgi toplandı",
                        "action": "answer",
                        "query": None,
                        "chunks_found": 0,
                        "duration_ms": 10.1,
                    },
                ],
            }

        with patch.object(api, "ask_rag_with_trace", side_effect=fake_ask):
            resp = self.client.post(
                "/ask",
                json={"question": "Wildcard DNS nedir?", "use_agent": True, "max_agent_steps": 4},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(captured_kwargs["use_agent"])
        self.assertEqual(captured_kwargs["max_agent_steps"], 4)
        body = resp.json()
        self.assertEqual(len(body["agent_steps"]), 2)
        self.assertEqual(body["agent_steps"][0]["action"], "search")
        self.assertEqual(body["agent_steps"][0]["query"], "Wildcard DNS")
        self.assertEqual(body["agent_steps"][1]["action"], "answer")

        # max_agent_steps sınır kontrolleri (ge=1, le=10)
        self.assertEqual(
            self.client.post("/ask", json={"question": "DNS?", "max_agent_steps": 0}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/ask", json={"question": "DNS?", "max_agent_steps": 11}).status_code,
            422,
        )

    def test_ui_client_ask_question_with_use_agent_true(self):
        captured_payload = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "session_id": "s-agent",
                    "answer": "Ajan yanıtı",
                    "rewritten_query": "Soru",
                    "early_rejected": False,
                    "rejection_reason": None,
                    "chunks": [],
                    "timings": {"rewrite_ms": 0.0, "retrieval_ms": 5.0, "generation_ms": 20.0},
                    "agent_steps": [
                        {
                            "step": 1,
                            "thought": "Ara",
                            "action": "search",
                            "query": "Soru",
                            "chunks_found": 1,
                            "duration_ms": 6.0,
                        }
                    ],
                },
            )

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            result = ask_question(
                question="Soru",
                client=http_client,
                use_agent=True,
                max_agent_steps=5,
            )

        self.assertTrue(captured_payload["use_agent"])
        self.assertEqual(captured_payload["max_agent_steps"], 5)
        self.assertEqual(len(result["agent_steps"]), 1)

    def test_cli_print_xray_displays_agent_steps(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.print_xray(
                chunks=[],
                retrieval_ms=12.0,
                generation_ms=45.0,
                agent_steps=[
                    {
                        "step": 1,
                        "thought": "DNS dokümanını ara",
                        "action": "search",
                        "query": "Wildcard DNS",
                        "chunks_found": 2,
                        "duration_ms": 18.4,
                    }
                ],
            )
        output = buf.getvalue()
        self.assertIn("Ajan Adımları:", output)
        self.assertIn('Adım #1 [search] | sorgu="Wildcard DNS"', output)
        self.assertIn("DNS dokümanını ara", output)


if __name__ == "__main__":
    unittest.main()
