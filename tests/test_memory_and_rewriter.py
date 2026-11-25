import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import generator
from memory import ConversationMemory
from rewriter import rewrite_query


def _make_llm_response(content: str | None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestConversationMemory(unittest.TestCase):
    def test_initial_state_is_empty(self):
        mem = ConversationMemory(max_turns=3)
        self.assertTrue(mem.is_empty())
        self.assertEqual(mem.turn_count, 0)
        self.assertEqual(len(mem), 0)
        self.assertEqual(mem.get_messages(), [])

    def test_invalid_max_turns_raises_value_error(self):
        for bad_value in (0, -1, -10, False):
            with self.subTest(max_turns=bad_value):
                with self.assertRaises(ValueError):
                    ConversationMemory(max_turns=bad_value)

    def test_add_turn_and_sliding_window(self):
        mem = ConversationMemory(max_turns=2)
        mem.add_turn("Soru 1", "Cevap 1")
        mem.add_turn("Soru 2", "Cevap 2")
        self.assertFalse(mem.is_empty())
        self.assertEqual(mem.turn_count, 2)
        self.assertEqual(len(mem), 2)

        # 3. tur eklendiğinde 1. tur pencereden düşmeli
        mem.add_turn("Soru 3", "Cevap 3")
        self.assertEqual(mem.turn_count, 2)
        self.assertEqual(
            mem.get_messages(),
            [
                {"role": "user", "content": "Soru 2"},
                {"role": "assistant", "content": "Cevap 2"},
                {"role": "user", "content": "Soru 3"},
                {"role": "assistant", "content": "Cevap 3"},
            ],
        )

    def test_clear_resets_memory(self):
        mem = ConversationMemory(max_turns=3)
        mem.add_turn("Soru 1", "Cevap 1")
        mem.clear()
        self.assertTrue(mem.is_empty())
        self.assertEqual(mem.turn_count, 0)
        self.assertEqual(mem.get_messages(), [])

    def test_get_messages_returns_isolated_copy(self):
        mem = ConversationMemory(max_turns=3)
        mem.add_turn("Orijinal Soru", "Orijinal Cevap")

        snapshot = mem.get_messages()
        snapshot[0]["content"] = "DEĞİŞTİRİLDİ"
        snapshot.append({"role": "user", "content": "Sahte Mesaj"})

        fresh = mem.get_messages()
        self.assertEqual(len(fresh), 2)
        self.assertEqual(fresh[0]["content"], "Orijinal Soru")
        self.assertEqual(fresh[1]["content"], "Orijinal Cevap")


class TestQueryRewriter(unittest.TestCase):
    def test_empty_or_none_history_skips_llm(self):
        mock_client = MagicMock()
        for empty_history in (None, []):
            with self.subTest(history=empty_history):
                rewritten, elapsed_ms = rewrite_query(
                    "Wildcard DNS nedir?",
                    empty_history,
                    llm_client=mock_client,
                    model_name="test-model",
                )
                self.assertEqual(rewritten, "Wildcard DNS nedir?")
                self.assertEqual(elapsed_ms, 0.0)
                mock_client.chat.completions.create.assert_not_called()

    def test_non_empty_history_rewrites_query(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "Wildcard DNS'in güvenlik riskleri nelerdir?"
        )
        history = [
            {"role": "user", "content": "Wildcard DNS nedir?"},
            {"role": "assistant", "content": "Tüm alt alan adlarını tek kayda yönlendirir."},
        ]

        rewritten, elapsed_ms = rewrite_query(
            "Peki bunun riskleri neler?",
            history,
            llm_client=mock_client,
            model_name="test-model",
        )

        self.assertEqual(rewritten, "Wildcard DNS'in güvenlik riskleri nelerdir?")
        self.assertGreaterEqual(elapsed_ms, 0.0)

        mock_client.chat.completions.create.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["temperature"], 0.0)
        sent_messages = kwargs["messages"]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertEqual(sent_messages[1:3], history)
        self.assertEqual(
            sent_messages[-1],
            {"role": "user", "content": "Peki bunun riskleri neler?"},
        )

    def test_empty_llm_output_falls_back_to_original_question(self):
        for empty_output in ("   ", "", None):
            with self.subTest(output=empty_output):
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = _make_llm_response(
                    empty_output
                )
                history = [{"role": "user", "content": "Önceki soru"}]
                rewritten, elapsed_ms = rewrite_query(
                    "Bunun süresi ne kadar?",
                    history,
                    llm_client=mock_client,
                    model_name="test-model",
                )
                self.assertEqual(rewritten, "Bunun süresi ne kadar?")
                self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_exception_warns_and_falls_back_to_original_question(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API hatası")
        history = [{"role": "user", "content": "Önceki soru"}]

        with self.assertWarns(RuntimeWarning) as ctx:
            rewritten, elapsed_ms = rewrite_query(
                "Bunun riskleri neler?",
                history,
                llm_client=mock_client,
                model_name="test-model",
            )

        self.assertEqual(rewritten, "Bunun riskleri neler?")
        self.assertGreaterEqual(elapsed_ms, 0.0)
        self.assertIn("API hatası", str(ctx.warning))


class TestGeneratorWithMemoryAndRewriting(unittest.TestCase):
    SAMPLE_CHUNKS = [
        {
            "source": "dns.md",
            "section": "Wildcard DNS",
            "text": "Wildcard DNS yanlış yapılandırılırsa alt alan adı ele geçirme riski doğurur.",
            "score": 0.9,
            "dense_rank": 1,
            "dense_score": 0.88,
            "bm25_rank": 1,
            "bm25_score": 5.4,
        }
    ]

    def test_generator_exposes_client_and_model_name(self):
        self.assertTrue(hasattr(generator, "client"))
        self.assertTrue(hasattr(generator, "MODEL_NAME"))

    def test_first_turn_skips_rewrite_and_searches_original_question(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "Wildcard DNS tüm alt alan adlarını yakalar [dns.md - Wildcard DNS]."
        )

        with patch.object(generator, "search", return_value=self.SAMPLE_CHUNKS) as mock_search, \
             patch.object(generator, "client", mock_client), \
             patch.object(generator, "MODEL_NAME", "test-model"):
            trace = generator.ask_rag_with_trace("Wildcard DNS nedir?", top_k=2, use_rerank=False)

        mock_search.assert_called_once_with("Wildcard DNS nedir?", top_k=2, use_rerank=False)
        self.assertEqual(trace["rewritten_query"], "Wildcard DNS nedir?")
        self.assertEqual(trace["rewrite_ms"], 0.0)
        self.assertIn("Wildcard DNS", trace["answer"])
        self.assertEqual(trace["chunks"], self.SAMPLE_CHUNKS)
        self.assertIn("retrieval_ms", trace)
        self.assertIn("generation_ms", trace)
        # Sadece generation için 1 kez çağrılmalı
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    def test_second_turn_uses_rewritten_query_and_passes_history_to_generation(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response("Wildcard DNS güvenlik riskleri nelerdir?"),
            _make_llm_response(
                "Alt alan adı ele geçirme riski taşır [dns.md - Wildcard DNS]."
            ),
        ]
        history = [
            {"role": "user", "content": "Wildcard DNS nedir?"},
            {"role": "assistant", "content": "Tüm alt alan adlarını yakalar [dns.md - Wildcard DNS]."},
        ]

        with patch.object(generator, "search", return_value=self.SAMPLE_CHUNKS) as mock_search, \
             patch.object(generator, "client", mock_client), \
             patch.object(generator, "MODEL_NAME", "test-model"):
            trace = generator.ask_rag_with_trace(
                "Peki riskleri neler?",
                top_k=3,
                use_rerank=True,
                history=history,
                use_rewrite=True,
            )

        mock_search.assert_called_once_with(
            "Wildcard DNS güvenlik riskleri nelerdir?",
            top_k=3,
            use_rerank=True,
        )
        self.assertEqual(trace["rewritten_query"], "Wildcard DNS güvenlik riskleri nelerdir?")
        self.assertGreaterEqual(trace["rewrite_ms"], 0.0)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

        # İkinci çağrı (generation) messages listesinde history bulunmalı
        gen_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(gen_messages[0]["role"], "system")
        self.assertEqual(gen_messages[1:3], history)
        self.assertEqual(gen_messages[-1]["role"], "user")
        self.assertIn("Peki riskleri neler?", gen_messages[-1]["content"])

    def test_use_rewrite_false_skips_rewrite_even_with_history(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "Riskler dokümanda belirtilmiştir [dns.md - Wildcard DNS]."
        )
        history = [
            {"role": "user", "content": "Wildcard DNS nedir?"},
            {"role": "assistant", "content": "Tüm alt alan adlarını yakalar."},
        ]

        with patch.object(generator, "search", return_value=self.SAMPLE_CHUNKS) as mock_search, \
             patch.object(generator, "client", mock_client), \
             patch.object(generator, "MODEL_NAME", "test-model"):
            trace = generator.ask_rag_with_trace(
                "Peki riskleri neler?",
                top_k=2,
                use_rerank=False,
                history=history,
                use_rewrite=False,
            )

        mock_search.assert_called_once_with("Peki riskleri neler?", top_k=2, use_rerank=False)
        self.assertEqual(trace["rewritten_query"], "Peki riskleri neler?")
        self.assertEqual(trace["rewrite_ms"], 0.0)
        # Rewrite yapılmadığı için LLM yalnızca generation için 1 kez çağrılmalı
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        gen_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(gen_messages[1:3], history)

    def test_backward_compatible_three_arg_call(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "Cevap [dns.md - Wildcard DNS]."
        )

        with patch.object(generator, "search", return_value=self.SAMPLE_CHUNKS) as mock_search, \
             patch.object(generator, "client", mock_client), \
             patch.object(generator, "MODEL_NAME", "test-model"):
            trace = generator.ask_rag_with_trace("Test sorusu", top_k=2, use_rerank=False)

        mock_search.assert_called_once_with("Test sorusu", top_k=2, use_rerank=False)
        self.assertEqual(trace["answer"], "Cevap [dns.md - Wildcard DNS].")
        self.assertEqual(trace["rewrite_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
