import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
EVAL_DIR = PROJECT_ROOT / "eval"
DATA_DIR = PROJECT_ROOT / "data"

for path_str in (str(SRC_DIR), str(EVAL_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from chunker import chunk_markdown, chunk_text
from metrics import (
    REFUSAL_MESSAGE,
    check_refusal,
    compute_coverage,
    compute_keyword_match,
    compute_mode_delta,
    compute_multi_source_citation,
    parse_json_response,
    summarize_multi_hop_mode,
    summarize_multi_turn,
    validate_advanced_dataset,
)
from run_advanced_eval import (
    evaluate_multi_hop,
    evaluate_multi_turn,
    judge_advanced_answer,
    print_report,
    save_report,
)


def _make_llm_response(content: str | None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestAdvancedDatasetIntegrity(unittest.TestCase):
    def test_real_advanced_dataset_schema_and_document_consistency(self):
        dataset_path = EVAL_DIR / "advanced_dataset.json"
        self.assertTrue(dataset_path.exists(), "eval/advanced_dataset.json bulunamadı")

        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        errors = validate_advanced_dataset(data)
        self.assertEqual(errors, [], f"Veri seti doğrulama hataları: {errors}")

        self.assertGreaterEqual(len(data["multi_turn"]), 4)
        self.assertGreaterEqual(len(data["multi_hop"]), 8)

        # Her data dosyasının gerçek bölüm başlıklarını çıkar
        real_sections_by_file: dict[str, set[str]] = {}
        for file_path in DATA_DIR.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() == ".md":
                chunks = chunk_markdown(file_path)
            elif file_path.suffix.lower() == ".txt":
                chunks = chunk_text(file_path)
            else:
                continue
            real_sections_by_file[file_path.name] = {
                c["metadata"]["section"] for c in chunks
            }

        all_items: list[tuple[str, dict]] = []
        for dlg in data["multi_turn"]:
            self.assertGreaterEqual(len(dlg["turns"]), 3)
            for idx, turn in enumerate(dlg["turns"], start=1):
                all_items.append((f"{dlg['id']}#turn{idx}", turn))
        for item in data["multi_hop"]:
            all_items.append((item["id"], item))

        for label, item in all_items:
            expected_sources = item["expected_sources"]
            expected_sections = item["expected_sections"]

            # Her expected_source data/ altında var olmalı
            available_sections: set[str] = set()
            for src in expected_sources:
                src_path = DATA_DIR / src
                self.assertTrue(
                    src_path.exists(),
                    f"{label}: expected_source '{src}' data/ altında yok",
                )
                self.assertIn(src, real_sections_by_file)
                available_sections.update(real_sections_by_file[src])

            # Her expected_section, expected_sources içindeki gerçek chunker section'larıyla birebir eşleşmeli
            for sec in expected_sections:
                self.assertIn(
                    sec,
                    available_sections,
                    f"{label}: expected_section '{sec}' gerçek bölüm başlıklarında yok: {sorted(available_sections)}",
                )

    def test_validate_advanced_dataset_catches_invalid_schemas(self):
        self.assertTrue(validate_advanced_dataset({}))

        bad_data = {
            "multi_turn": [
                {
                    "id": "mt_bad",
                    "scenario_name": "Tek turlu",
                    "turns": [
                        {
                            "question": "Soru?",
                            "answerable": "evet",
                            "expected_sources": [],
                            "expected_sections": [],
                            "expect_early_reject": False,
                        }
                    ],
                }
            ],
            "multi_hop": [
                {
                    "id": "",
                    "question": "",
                    "answerable": True,
                    "expected_sources": "dosya.md",
                    "expected_sections": [],
                    "expect_early_reject": False,
                }
            ],
        }
        errors = validate_advanced_dataset(bad_data)
        self.assertGreaterEqual(len(errors), 4)


class TestEvalMetrics(unittest.TestCase):
    def test_parse_json_response_variants(self):
        self.assertEqual(parse_json_response('{"a": 1, "b": "ok"}'), {"a": 1, "b": "ok"})

        fenced = 'İşte değerlendirme:\n```json\n{"faithfulness": 5, "reason": "tam"}\n```\nTeşekkürler.'
        self.assertEqual(
            parse_json_response(fenced),
            {"faithfulness": 5, "reason": "tam"},
        )

        with_prefix = 'Sonuç: {"refused_properly": true, "reason": "reddetti"} bitti.'
        self.assertEqual(
            parse_json_response(with_prefix),
            {"refused_properly": True, "reason": "reddetti"},
        )

        self.assertIsNone(parse_json_response("geçersiz metin {bozuk"))
        self.assertIsNone(parse_json_response("[1, 2, 3]"))
        self.assertIsNone(parse_json_response(""))

    def test_compute_coverage(self):
        self.assertEqual(compute_coverage([], ["a"]), (1.0, True))
        self.assertEqual(compute_coverage(["a", "b"], ["b", "a", "c"]), (1.0, True))
        self.assertEqual(compute_coverage(["a", "b"], ["a", "c"]), (0.5, False))
        self.assertEqual(compute_coverage(["a", "b"], ["c"]), (0.0, False))

    def test_compute_multi_source_citation(self):
        self.assertEqual(compute_multi_source_citation("herhangi", []), (1.0, True))

        ans_full = "Bilgi 1 [a.md - Bölüm 1] ve Bilgi 2 [b.md - Bölüm 2]."
        self.assertEqual(compute_multi_source_citation(ans_full, ["a.md", "b.md"]), (1.0, True))

        ans_partial = "Bilgi 1 [a.md - Bölüm 1] ama ikinci kaynak yok [b.md]."
        self.assertEqual(compute_multi_source_citation(ans_partial, ["a.md", "b.md"]), (0.5, False))

        self.assertEqual(compute_multi_source_citation("atıf yok", ["a.md"]), (0.0, False))

    def test_compute_keyword_match(self):
        self.assertEqual(compute_keyword_match("metin", []), (1.0, True))
        self.assertEqual(
            compute_keyword_match("WireGuard ve IPsec kullanılır.", ["wireguard", "IPSEC"]),
            (1.0, True),
        )
        self.assertEqual(
            compute_keyword_match("Sadece WireGuard var.", ["wireguard", "ipsec"]),
            (0.5, False),
        )

    def test_check_refusal_early_judge_and_error_fallback(self):
        self.assertTrue(check_refusal("herhangi", early_rejected=True, judge_result=None))
        self.assertTrue(
            check_refusal("kısa cevap", early_rejected=False, judge_result={"refused_properly": True})
        )
        self.assertFalse(
            check_refusal(
                REFUSAL_MESSAGE,
                early_rejected=False,
                judge_result={"refused_properly": False},
            )
        )
        # Judge hata döndüğünde sessiz False düşmemeli, metin fallback'ine geçmeli
        self.assertTrue(
            check_refusal(
                REFUSAL_MESSAGE,
                early_rejected=False,
                judge_result={"error": "JSON ayrıştırılamadı"},
            )
        )
        self.assertTrue(
            check_refusal(
                "Bu bilgi dokümanlarda yer almamaktadır.",
                early_rejected=False,
                judge_result=None,
            )
        )
        self.assertFalse(
            check_refusal(
                "Port 2222 olarak ayarlanmalıdır.",
                early_rejected=False,
                judge_result={"error": "API timeout"},
            )
        )

    def test_summarize_and_delta_with_and_without_judge(self):
        classic_recs = [
            {
                "answerable": True,
                "source_coverage_ratio": 0.5,
                "full_source_coverage": False,
                "section_coverage_ratio": 0.5,
                "full_section_coverage": False,
                "citation_full": False,
                "answer_keyword_ratio": 0.5,
                "early_rejected": False,
                "refused": False,
                "agent_step_count": 0,
                "agent_search_steps": 0,
                "rewrite_ms": 0.0,
                "retrieval_ms": 20.0,
                "generation_ms": 200.0,
                "judge": None,
            },
            {
                "answerable": False,
                "early_rejected": True,
                "refused": True,
                "agent_step_count": 0,
                "agent_search_steps": 0,
                "rewrite_ms": 0.0,
                "retrieval_ms": 15.0,
                "generation_ms": 0.0,
                "judge": None,
            },
        ]
        agentic_recs = [
            {
                "answerable": True,
                "source_coverage_ratio": 1.0,
                "full_source_coverage": True,
                "section_coverage_ratio": 1.0,
                "full_section_coverage": True,
                "citation_full": True,
                "answer_keyword_ratio": 1.0,
                "early_rejected": False,
                "refused": False,
                "agent_step_count": 3,
                "agent_search_steps": 2,
                "rewrite_ms": 0.0,
                "retrieval_ms": 50.0,
                "generation_ms": 400.0,
                "judge": None,
            },
            {
                "answerable": False,
                "early_rejected": True,
                "refused": True,
                "agent_step_count": 1,
                "agent_search_steps": 1,
                "rewrite_ms": 0.0,
                "retrieval_ms": 20.0,
                "generation_ms": 0.0,
                "judge": None,
            },
        ]

        c_sum = summarize_multi_hop_mode(classic_recs)
        a_sum = summarize_multi_hop_mode(agentic_recs)
        delta = compute_mode_delta(c_sum, a_sum)

        self.assertEqual(delta["full_section_coverage_delta"], 1.0)
        self.assertEqual(delta["full_source_coverage_delta"], 1.0)
        self.assertEqual(delta["citation_full_delta"], 1.0)
        self.assertEqual(delta["avg_steps_delta"], 2.0)
        self.assertIsNone(delta["faithfulness_delta"])
        self.assertIsNone(delta["completeness_delta"])

        # Judge varken delta sayısal olmalı
        classic_recs[0]["judge"] = {"faithfulness": 4, "relevance": 4, "completeness": 3}
        agentic_recs[0]["judge"] = {"faithfulness": 5, "relevance": 5, "completeness": 5}
        delta_j = compute_mode_delta(
            summarize_multi_hop_mode(classic_recs),
            summarize_multi_hop_mode(agentic_recs),
        )
        self.assertEqual(delta_j["faithfulness_delta"], 1.0)
        self.assertEqual(delta_j["completeness_delta"], 2.0)


class TestJudgeAdvancedAnswer(unittest.TestCase):
    def test_judge_answerable_success_and_errors(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            '```json\n{"faithfulness": 5, "relevance": 5, "completeness": 4, "reason": "İyi"}\n```'
        )
        res = judge_advanced_answer(
            "Soru?",
            "Cevap",
            [{"source": "a.md", "section": "B1", "text": "metin"}],
            True,
            ["a.md"],
            client_obj=mock_client,
            model_name="test-model",
        )
        self.assertEqual(res["completeness"], 4)
        self.assertNotIn("error", res)

        # Bozuk JSON
        mock_client.chat.completions.create.return_value = _make_llm_response("JSON yok")
        err_res = judge_advanced_answer(
            "Soru?", "Cevap", [], True, [], client_obj=mock_client, model_name="test-model"
        )
        self.assertIn("error", err_res)

        # API hatası — gizli token/anahtar içeren ham hata metni asla sızmamalı
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "Authorization: Bearer sk-secret-123"
        )
        api_err = judge_advanced_answer(
            "Soru?", "Cevap", [], False, [], client_obj=mock_client, model_name="test-model"
        )
        self.assertIn("error", api_err)
        self.assertIn("RuntimeError", api_err["error"])
        self.assertNotIn("sk-secret-123", api_err["error"])


class TestEvaluateMultiTurn(unittest.TestCase):
    def test_history_growth_and_early_rejected_turn_excluded_from_memory(self):
        dialogue = [
            {
                "id": "mt_test",
                "scenario_name": "Hafıza ve Guardrail Testi",
                "turns": [
                    {
                        "question": "JWT hangi bölümlerden oluşur?",
                        "answerable": True,
                        "expected_sources": ["http_ve_api_notlari.md"],
                        "expected_sections": ["JWT (JSON Web Token) ile Kimlik Doğrulama"],
                        "expected_answer_keywords": ["Header", "Payload"],
                        "expect_early_reject": False,
                    },
                    {
                        "question": "Onu hangi başlıkla göndeririz?",
                        "answerable": True,
                        "expected_rewrite_keywords": ["jwt"],
                        "expected_sources": ["http_ve_api_notlari.md"],
                        "expected_sections": ["JWT (JSON Web Token) ile Kimlik Doğrulama"],
                        "expected_answer_keywords": ["Authorization", "Bearer"],
                        "expect_early_reject": False,
                    },
                    {
                        "question": "Onun EdDSA exp süresi kaç dakikadır?",
                        "answerable": False,
                        "expected_rewrite_keywords": ["jwt"],
                        "expected_sources": [],
                        "expected_sections": [],
                        "expect_early_reject": True,
                    },
                    {
                        "question": "Peki sunucu onu nasıl doğrular?",
                        "answerable": True,
                        "expected_rewrite_keywords": ["jwt"],
                        "expected_sources": ["http_ve_api_notlari.md"],
                        "expected_sections": ["JWT (JSON Web Token) ile Kimlik Doğrulama"],
                        "expected_answer_keywords": ["imza"],
                        "expect_early_reject": False,
                    },
                ],
            }
        ]

        observed_histories: list[list[dict]] = []

        def fake_rag(question, **kwargs):
            history = kwargs.get("history") or []
            observed_histories.append(list(history))
            call_idx = len(observed_histories)

            if call_idx == 1:
                return {
                    "answer": "JWT Header, Payload ve Signature bölümlerinden oluşur [http_ve_api_notlari.md - JWT (JSON Web Token) ile Kimlik Doğrulama].",
                    "chunks": [
                        {
                            "source": "http_ve_api_notlari.md",
                            "section": "JWT (JSON Web Token) ile Kimlik Doğrulama",
                            "text": "JWT Header Payload Signature",
                        }
                    ],
                    "rewritten_query": question,
                    "rewrite_ms": 0.0,
                    "retrieval_ms": 12.0,
                    "generation_ms": 150.0,
                    "early_rejected": False,
                }
            if call_idx == 2:
                return {
                    "answer": "Authorization: Bearer <token> başlığıyla gönderilir [http_ve_api_notlari.md - JWT (JSON Web Token) ile Kimlik Doğrulama].",
                    "chunks": [
                        {
                            "source": "http_ve_api_notlari.md",
                            "section": "JWT (JSON Web Token) ile Kimlik Doğrulama",
                            "text": "Authorization Bearer",
                        }
                    ],
                    "rewritten_query": "JWT hangi HTTP başlığıyla gönderilir?",
                    "rewrite_ms": 45.0,
                    "retrieval_ms": 14.0,
                    "generation_ms": 140.0,
                    "early_rejected": False,
                }
            if call_idx == 3:
                return {
                    "answer": REFUSAL_MESSAGE,
                    "chunks": [],
                    "rewritten_query": "JWT EdDSA exp süresi kaç dakikadır?",
                    "rewrite_ms": 40.0,
                    "retrieval_ms": 11.0,
                    "generation_ms": 0.0,
                    "early_rejected": True,
                }
            return {
                "answer": "Sunucu imzayı doğrulayarak kullanıcıyı tanır [http_ve_api_notlari.md - JWT (JSON Web Token) ile Kimlik Doğrulama].",
                "chunks": [
                    {
                        "source": "http_ve_api_notlari.md",
                        "section": "JWT (JSON Web Token) ile Kimlik Doğrulama",
                        "text": "imzayı doğrulayarak",
                    }
                ],
                "rewritten_query": "Sunucu JWT imzasını nasıl doğrular?",
                "rewrite_ms": 42.0,
                "retrieval_ms": 13.0,
                "generation_ms": 130.0,
                "early_rejected": False,
            }

        result = evaluate_multi_turn(dialogue, rag_fn=fake_rag)
        summary = result["summary"]

        # Geçmiş uzunlukları: Tur 1 -> 0 mesaj, Tur 2 -> 2 mesaj, Tur 3 -> 4 mesaj,
        # Tur 3 early_rejected olduğu için Tur 4 -> HÂLÂ 4 mesaj!
        self.assertEqual([len(h) for h in observed_histories], [0, 2, 4, 4])
        for msg in observed_histories[3]:
            self.assertNotIn("EdDSA", msg["content"])

        self.assertEqual(summary["total_dialogues"], 1)
        self.assertEqual(summary["total_turns"], 4)
        self.assertEqual(summary["answerable_turns"], 3)
        self.assertEqual(summary["unanswerable_turns"], 1)
        self.assertEqual(summary["turn1_zero_rewrite_ok"], 1.0)
        self.assertEqual(summary["rewrite_keyword_rate"], 1.0)
        self.assertEqual(summary["full_section_coverage_rate"], 1.0)
        self.assertEqual(summary["citation_full_rate"], 1.0)
        self.assertEqual(summary["guardrail"]["trap_early_rejected"], 1)
        self.assertEqual(summary["guardrail"]["trap_catch_rate"], 1.0)
        self.assertEqual(summary["guardrail"]["answerable_false_positives"], 0)


class TestEvaluateMultiHopAndReport(unittest.TestCase):
    def test_classic_vs_agentic_delta_and_save_report(self):
        questions = [
            {
                "id": "mh_t1",
                "category": "cross_source",
                "question": "Rate Limiting ve DDoS ilişkisi nedir?",
                "answerable": True,
                "expected_sources": ["http_ve_api_notlari.md", "ag_guvenligi.md"],
                "expected_sections": [
                    "Rate Limiting ve Önbellekleme (Caching)",
                    "DDoS Saldırıları ve Korunma Yöntemleri",
                ],
                "expected_answer_keywords": ["429", "WAF"],
                "expect_early_reject": False,
            },
            {
                "id": "mh_t2",
                "category": "cross_source",
                "question": "HTTP 401 ve chmod 750 ne işe yarar?",
                "answerable": True,
                "expected_sources": ["http_ve_api_notlari.md", "linux_temelleri.md"],
                "expected_sections": [
                    "HTTP Durum Kodları",
                    "Dosya İzinleri (chmod ve chown)",
                ],
                "expected_answer_keywords": ["401", "chmod 750"],
                "expect_early_reject": False,
            },
            {
                "id": "mh_trap",
                "category": "trap",
                "question": "Kubernetes Ingress ve pgBouncer ayarı?",
                "answerable": False,
                "expected_sources": [],
                "expected_sections": [],
                "expect_early_reject": True,
            },
        ]

        def fake_rag(question, *, use_agent=False, **kwargs):
            if "Kubernetes" in question:
                return {
                    "answer": REFUSAL_MESSAGE,
                    "chunks": [],
                    "rewrite_ms": 0.0,
                    "retrieval_ms": 10.0,
                    "generation_ms": 0.0,
                    "early_rejected": True,
                    "agent_steps": (
                        [{"step": 1, "action": "search", "query": question}] if use_agent else []
                    ),
                }

            if "Rate Limiting" in question:
                if not use_agent:
                    # Classic sadece 1 kaynağı bulur
                    return {
                        "answer": "429 kodu döner [http_ve_api_notlari.md - Rate Limiting ve Önbellekleme (Caching)].",
                        "chunks": [
                            {
                                "source": "http_ve_api_notlari.md",
                                "section": "Rate Limiting ve Önbellekleme (Caching)",
                                "text": "429 Too Many Requests",
                            }
                        ],
                        "rewrite_ms": 0.0,
                        "retrieval_ms": 15.0,
                        "generation_ms": 200.0,
                        "early_rejected": False,
                        "agent_steps": [],
                    }
                # Agentic 2 arama + 1 answer adımıyla her iki kaynağı bulur
                return {
                    "answer": (
                        "Sınır aşılınca 429 döner [http_ve_api_notlari.md - Rate Limiting ve Önbellekleme (Caching)], "
                        "DDoS'ta WAF kullanılır [ag_guvenligi.md - DDoS Saldırıları ve Korunma Yöntemleri]."
                    ),
                    "chunks": [
                        {
                            "source": "http_ve_api_notlari.md",
                            "section": "Rate Limiting ve Önbellekleme (Caching)",
                            "text": "429",
                        },
                        {
                            "source": "ag_guvenligi.md",
                            "section": "DDoS Saldırıları ve Korunma Yöntemleri",
                            "text": "WAF",
                        },
                    ],
                    "rewrite_ms": 0.0,
                    "retrieval_ms": 35.0,
                    "generation_ms": 350.0,
                    "early_rejected": False,
                    "agent_steps": [
                        {"step": 1, "action": "search", "query": "HTTP Rate Limiting 429"},
                        {"step": 2, "action": "search", "query": "DDoS korunma WAF"},
                        {"step": 3, "action": "answer", "query": None},
                    ],
                }

            # 2. soru: Classic yanlış pozitif (early_rejected=True), Agentic tam kapsam
            if not use_agent:
                return {
                    "answer": REFUSAL_MESSAGE,
                    "chunks": [],
                    "rewrite_ms": 0.0,
                    "retrieval_ms": 12.0,
                    "generation_ms": 0.0,
                    "early_rejected": True,
                    "agent_steps": [],
                }
            return {
                "answer": (
                    "401 kimlik doğrulama gerektirir [http_ve_api_notlari.md - HTTP Durum Kodları] ve "
                    "chmod 750 sahibine tam yetki verir [linux_temelleri.md - Dosya İzinleri (chmod ve chown)]."
                ),
                "chunks": [
                    {"source": "http_ve_api_notlari.md", "section": "HTTP Durum Kodları", "text": "401"},
                    {
                        "source": "linux_temelleri.md",
                        "section": "Dosya İzinleri (chmod ve chown)",
                        "text": "chmod 750",
                    },
                ],
                "rewrite_ms": 0.0,
                "retrieval_ms": 30.0,
                "generation_ms": 320.0,
                "early_rejected": False,
                "agent_steps": [
                    {"step": 1, "action": "search", "query": "HTTP 401 durum kodu"},
                    {"step": 2, "action": "search", "query": "Linux chmod 750"},
                    {"step": 3, "action": "answer", "query": None},
                ],
            }

        mh = evaluate_multi_hop(questions, rag_fn=fake_rag)
        self.assertEqual(mh["classic"]["full_section_coverage_rate"], 0.0)
        self.assertEqual(mh["agentic"]["full_section_coverage_rate"], 1.0)
        self.assertGreater(mh["delta"]["full_section_coverage_delta"], 0.0)
        self.assertEqual(mh["delta"]["full_source_coverage_delta"], 1.0)
        self.assertEqual(mh["classic"]["guardrail"]["answerable_false_positives"], 1)
        self.assertEqual(mh["agentic"]["guardrail"]["answerable_false_positives"], 0)
        self.assertLess(mh["delta"]["false_positive_rate_delta"], 0.0)
        self.assertEqual(mh["classic"]["guardrail"]["trap_catch_rate"], 1.0)
        self.assertEqual(mh["agentic"]["guardrail"]["trap_catch_rate"], 1.0)
        self.assertEqual(mh["classic"]["agent"]["avg_search_steps"], 0.0)
        # 2 + 2 + 1 = 5 arama adımı / 3 soru = 1.67
        self.assertAlmostEqual(mh["agentic"]["agent"]["avg_search_steps"], 1.67, places=2)

        report = {
            "meta": {"mode": "multi-hop", "top_k": 3, "max_agent_steps": 3},
            "multi_hop": mh,
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_report(report)
        rendered = buf.getvalue()
        self.assertIn("MULTI-HOP KARŞILAŞTIRMA KARNESİ", rendered)
        self.assertIn("Tam Bölüm Kapsama", rendered)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "nested" / "advanced_report.json"
            saved_path = save_report(report, str(out_path))
            self.assertTrue(saved_path.exists())
            loaded = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded["multi_hop"]["delta"]["full_section_coverage_delta"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
