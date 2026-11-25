import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cli
import generator
from guardrail import (
    DEFAULT_MIN_BM25_SCORE,
    DEFAULT_MIN_DENSE_SCORE,
    DEFAULT_MIN_RERANK_SCORE,
    REFUSAL_MESSAGE,
    check_retrieval_guardrail,
)
from memory import ConversationMemory


def _make_llm_response(content: str | None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestCheckRetrievalGuardrail(unittest.TestCase):
    def test_empty_chunks_rejected(self):
        passed, reason = check_retrieval_guardrail([], use_rerank=False)
        self.assertFalse(passed)
        self.assertIsNotNone(reason)
        self.assertIn("no_chunks", reason)

        passed_rerank, reason_rerank = check_retrieval_guardrail([], use_rerank=True)
        self.assertFalse(passed_rerank)
        self.assertIn("no_chunks", reason_rerank)

    def test_rerank_passes_when_above_or_equal_threshold(self):
        chunks = [
            {"rerank_score": -5.2, "dense_score": 0.2, "bm25_score": 1.0},
            {"rerank_score": DEFAULT_MIN_RERANK_SCORE, "dense_score": 0.2, "bm25_score": 1.0},
        ]
        passed, reason = check_retrieval_guardrail(chunks, use_rerank=True)
        self.assertTrue(passed)
        self.assertIsNone(reason)

    def test_rerank_rejects_when_below_threshold(self):
        chunks = [
            {"rerank_score": -6.5080, "dense_score": 0.9, "bm25_score": 12.0},
            {"rerank_score": -4.1, "dense_score": 0.8, "bm25_score": 10.0},
        ]
        passed, reason = check_retrieval_guardrail(chunks, use_rerank=True)
        self.assertFalse(passed)
        self.assertIsNotNone(reason)
        self.assertIn("-4.1000", reason)

    def test_rerank_enabled_without_rerank_score_falls_back_to_dense_and_bm25(self):
        passing_chunks = [{"dense_score": 0.60, "bm25_score": 2.0}]
        passed, reason = check_retrieval_guardrail(passing_chunks, use_rerank=True)
        self.assertTrue(passed)
        self.assertIsNone(reason)

        failing_chunks = [{"dense_score": 0.30, "bm25_score": 2.0}]
        passed_fail, reason_fail = check_retrieval_guardrail(failing_chunks, use_rerank=True)
        self.assertFalse(passed_fail)
        self.assertIsNotNone(reason_fail)

    def test_dense_high_and_bm25_low_passes(self):
        chunks = [
            {"dense_score": DEFAULT_MIN_DENSE_SCORE + 0.1, "bm25_score": 1.5},
        ]
        passed, reason = check_retrieval_guardrail(chunks, use_rerank=False)
        self.assertTrue(passed)
        self.assertIsNone(reason)

    def test_dense_low_and_bm25_high_passes(self):
        # Exact keyword eşleşmesi senaryosu (dense düşük, bm25 yüksek)
        chunks = [
            {"dense_score": 0.3679, "bm25_score": DEFAULT_MIN_BM25_SCORE + 1.74},
        ]
        passed, reason = check_retrieval_guardrail(chunks, use_rerank=False)
        self.assertTrue(passed)
        self.assertIsNone(reason)

    def test_both_dense_and_bm25_low_rejected(self):
        chunks = [
            {"dense_score": 0.4005, "bm25_score": 6.25},
            {"dense_score": 0.25, "bm25_score": 3.10},
        ]
        passed, reason = check_retrieval_guardrail(chunks, use_rerank=False)
        self.assertFalse(passed)
        self.assertIsNotNone(reason)
        self.assertIn("0.4005", reason)
        self.assertIn("6.2500", reason)


class TestGeneratorGuardrail(unittest.TestCase):
    LOW_SCORE_CHUNKS = [
        {
            "source": "dns.md",
            "section": "Temel DNS",
            "text": "DNS alan adlarını IP adreslerine çözer.",
            "score": 0.015,
            "dense_rank": 1,
            "dense_score": 0.31,
            "bm25_rank": 1,
            "bm25_score": 2.4,
        }
    ]

    HIGH_SCORE_CHUNKS = [
        {
            "source": "dns.md",
            "section": "Wildcard DNS",
            "text": "Wildcard DNS tüm alt alan adlarını tek kayda yönlendirir.",
            "score": 0.032,
            "dense_rank": 1,
            "dense_score": 0.82,
            "bm25_rank": 1,
            "bm25_score": 9.1,
        }
    ]

    def test_low_score_skips_llm_and_returns_refusal(self):
        mock_client = MagicMock()

        with patch.object(generator, "search", return_value=self.LOW_SCORE_CHUNKS), \
             patch.object(generator, "client", mock_client), \
             patch.object(generator, "MODEL_NAME", "test-model"):
            trace = generator.ask_rag_with_trace(
                "Kubernetes Ingress nasıl kurulur?",
                top_k=3,
                use_rerank=False,
                use_guardrail=True,
            )

        self.assertEqual(mock_client.chat.completions.create.call_count, 0)
        self.assertEqual(trace["answer"], REFUSAL_MESSAGE)
        self.assertEqual(trace["generation_ms"], 0.0)
        self.assertTrue(trace["early_rejected"])
        self.assertIsNotNone(trace["rejection_reason"])
        self.assertEqual(trace["chunks"], self.LOW_SCORE_CHUNKS)
        self.assertEqual(trace["rewritten_query"], "Kubernetes Ingress nasıl kurulur?")
        self.assertEqual(trace["rewrite_ms"], 0.0)

    def test_guardrail_disabled_calls_llm_even_with_low_scores(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "Dokümanlarda bu bilgi yer almıyor."
        )

        with patch.object(generator, "search", return_value=self.LOW_SCORE_CHUNKS), \
             patch.object(generator, "client", mock_client), \
             patch.object(generator, "MODEL_NAME", "test-model"):
            trace = generator.ask_rag_with_trace(
                "Kubernetes Ingress nasıl kurulur?",
                top_k=3,
                use_rerank=False,
                use_guardrail=False,
            )

        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        self.assertFalse(trace["early_rejected"])
        self.assertIsNone(trace["rejection_reason"])
        self.assertEqual(trace["answer"], "Dokümanlarda bu bilgi yer almıyor.")

    def test_empty_llm_choices_returns_empty_string_safely(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[])

        with patch.object(generator, "search", return_value=self.HIGH_SCORE_CHUNKS), \
             patch.object(generator, "client", mock_client), \
             patch.object(generator, "MODEL_NAME", "test-model"):
            trace = generator.ask_rag_with_trace(
                "Wildcard DNS nedir?",
                top_k=2,
                use_rerank=False,
                use_guardrail=True,
            )

        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        self.assertEqual(trace["answer"], "")
        self.assertFalse(trace["early_rejected"])
        self.assertIsNone(trace["rejection_reason"])


class TestCliGuardrailMemory(unittest.TestCase):
    def test_early_rejected_question_not_added_to_memory(self):
        memory = ConversationMemory(max_turns=3)
        rejected_trace = {
            "answer": REFUSAL_MESSAGE,
            "chunks": [],
            "rewritten_query": "Tuzak soru?",
            "rewrite_ms": 0.0,
            "retrieval_ms": 12.5,
            "generation_ms": 0.0,
            "early_rejected": True,
            "rejection_reason": "Arama skorları eşiğin altında",
        }

        buf = io.StringIO()
        with patch.object(cli, "ask_rag_with_trace", return_value=rejected_trace), \
             redirect_stdout(buf):
            cli.handle_question(
                "Tuzak soru?",
                top_k=3,
                debug=True,
                search_only=False,
                use_rerank=False,
                memory=memory,
                use_rewrite=True,
                use_guardrail=True,
            )

        self.assertTrue(memory.is_empty())
        self.assertEqual(memory.turn_count, 0)
        output = buf.getvalue()
        self.assertIn("Guardrail", output)
        self.assertIn("Arama skorları eşiğin altında", output)

    def test_normal_question_added_to_memory(self):
        memory = ConversationMemory(max_turns=3)
        normal_trace = {
            "answer": "Wildcard DNS alt alan adlarını yakalar [dns.md - Wildcard DNS].",
            "chunks": [],
            "rewritten_query": "Wildcard DNS nedir?",
            "rewrite_ms": 0.0,
            "retrieval_ms": 10.0,
            "generation_ms": 120.0,
            "early_rejected": False,
            "rejection_reason": None,
        }

        buf = io.StringIO()
        with patch.object(cli, "ask_rag_with_trace", return_value=normal_trace), \
             redirect_stdout(buf):
            cli.handle_question(
                "Wildcard DNS nedir?",
                top_k=3,
                debug=False,
                search_only=False,
                use_rerank=False,
                memory=memory,
                use_rewrite=True,
                use_guardrail=True,
            )

        self.assertEqual(memory.turn_count, 1)
        messages = memory.get_messages()
        self.assertEqual(messages[0]["content"], "Wildcard DNS nedir?")
        self.assertEqual(messages[1]["content"], normal_trace["answer"])


if __name__ == "__main__":
    unittest.main()
