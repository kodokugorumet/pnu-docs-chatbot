from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_service_retrieval_ab",
    ROOT / "scripts" / "analyze_service_retrieval_ab.py",
)
assert SPEC is not None and SPEC.loader is not None
analysis_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis_module)


class AnalyzeServiceRetrievalABTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_linear_quantile_interpolates_between_observations(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(analysis_module.linear_quantile(values, 0.5), 2.5)
        self.assertAlmostEqual(
            analysis_module.linear_quantile(values, 0.95),
            3.85,
        )

    def test_load_complete_run_is_judge_free_and_rejects_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_jsonl(
                path,
                [{"id": "q1", "latency_ms": 10, "sources": []}],
            )

            loaded = analysis_module.load_complete_run(path, {"q1"})
            self.assertEqual(set(loaded), {"q1"})

            with self.assertRaisesRegex(ValueError, "incomplete ID set"):
                analysis_module.load_complete_run(path, {"q1", "q2"})

    def test_document_values_fold_chunks_from_the_same_document(self) -> None:
        case = {
            "expected": {
                "source_title_contains": "정답 안내",
                "source_host": "official.example",
            }
        }
        record = {
            "sources": [
                {
                    "document_id": "doc-a",
                    "chunk_id": "doc-a:1",
                    "source_title": "다른 안내",
                    "source_host": "other.example",
                },
                {
                    "document_id": "doc-a",
                    "chunk_id": "doc-a:2",
                    "source_title": "다른 안내",
                    "source_host": "other.example",
                },
                {
                    "document_id": "doc-b",
                    "chunk_id": "doc-b:1",
                    "source_title": "2026 정답 안내",
                    "source_host": "mirror.example",
                },
            ]
        }

        values = analysis_module.document_values(record, case)

        self.assertEqual(values["rank"], 2)
        self.assertFalse(values["hit_at_k"]["1"])
        self.assertTrue(values["hit_at_k"]["3"])
        self.assertTrue(values["hit_at_k"]["5"])
        self.assertEqual(values["reciprocal_rank_at_5"], 0.5)
        self.assertFalse(values["host_match"])

    def test_evidence_values_separate_any_all_and_recall(self) -> None:
        case = {
            "evidence": [
                {"chunk_id": "gold:1"},
                {"chunk_id": "gold:2"},
                {
                    "chunk_id": "optional:3",
                    "required_for_answer": False,
                },
            ]
        }
        record = {
            "sources": [
                {"chunk_id": "other"},
                {"chunk_id": "gold:1"},
                {"chunk_id": "other:2"},
                {"chunk_id": "other:3"},
                {"chunk_id": "other:4"},
                {"chunk_id": "gold:2"},
            ]
        }

        at_five = analysis_module.evidence_values(record, case, 5)
        at_eight = analysis_module.evidence_values(record, case, 8)

        self.assertTrue(at_five["any_matched"])
        self.assertFalse(at_five["all_matched"])
        self.assertEqual(at_five["recall"], 0.5)
        self.assertTrue(at_eight["all_matched"])
        self.assertEqual(at_eight["recall"], 1.0)
        self.assertEqual(at_eight["wanted"], 2)

    def test_evidence_content_values_accepts_duplicate_document_chunk_by_quote(
        self,
    ) -> None:
        case = {
            "evidence": [
                {
                    "chunk_id": "gold-document:0001",
                    "quote": (
                        "대학원 석사 또는 박사과정 수료자로서 "
                        "학위취득에 필요한 연구를 희망하는 자"
                    ),
                },
                {
                    "chunk_id": "optional:0001",
                    "quote": "질문에 필요하지 않은 참고 문구입니다.",
                    "required_for_answer": False,
                },
            ]
        }
        record = {
            "sources": [
                {
                    "chunk_id": "duplicate-document:0001",
                    "preview": (
                        "대학원 석사 또는 박사과정 수료자 로서\n"
                        "학위취득에 필요한 연구를 희망하는 자"
                    ),
                }
            ]
        }

        exact = analysis_module.evidence_values(record, case, 1)
        content = analysis_module.evidence_content_values(record, case, 1)

        self.assertFalse(exact["any_matched"])
        self.assertTrue(content["all_matched"])
        self.assertEqual(content["recall"], 1.0)
        self.assertEqual(content["wanted"], 1)
        self.assertEqual(content["matches"][0]["method"], "normalized_quote")

    def test_evidence_content_values_accepts_reordered_exact_quote_lines(
        self,
    ) -> None:
        case = {
            "evidence": [
                {
                    "chunk_id": "gold-document:0008",
                    "quote": (
                        "납부금액 : 수업료Ⅱ의 10%\n\n"
                        "학생지원시스템 - 등록 - 등록금 책정표 참고"
                    ),
                }
            ]
        }
        record = {
            "sources": [
                {
                    "chunk_id": "duplicate-document:0007",
                    "preview": (
                        "학생지원시스템 - 등록 - 등록금 책정표 참고\n\n"
                        "납부금액 : 수업료Ⅱ의 10%"
                    ),
                }
            ]
        }

        values = analysis_module.evidence_content_values(record, case, 1)

        self.assertTrue(values["all_matched"])
        self.assertEqual(
            values["matches"][0]["method"],
            "normalized_quote_lines",
        )

    def test_analyze_reports_metrics_transitions_and_changes(self) -> None:
        cases = {
            "q1": {
                "id": "q1",
                "category": "registration",
                "role": None,
                "expected": {"source_title_contains": "정답 문서"},
                "evidence": [
                    {"chunk_id": "target:1"},
                    {"chunk_id": "target:2"},
                ],
            },
            "q2": {
                "id": "q2",
                "category": "academic",
                "role": "학생",
                "expected": {"source_title_contains": "두 번째 정답"},
                "evidence": [{"chunk_id": "q2:gold"}],
            },
        }
        run_a = {
            "q1": {
                "id": "q1",
                "latency_ms": 100,
                "answer": "A 답변",
                "sources": [
                    {
                        "document_id": "target",
                        "chunk_id": "target:1",
                        "source_title": "정답 문서",
                    },
                    {
                        "document_id": "other",
                        "chunk_id": "other:1",
                        "source_title": "기타",
                    },
                ],
            },
            "q2": {
                "id": "q2",
                "latency_ms": 200,
                "answer": "같은 답변",
                "sources": [
                    {
                        "document_id": "wrong",
                        "chunk_id": "wrong:1",
                        "source_title": "오답",
                    }
                ],
            },
        }
        run_b = {
            "q1": {
                "id": "q1",
                "latency_ms": 120,
                "answer": "B 답변",
                "sources": [
                    {
                        "document_id": "target",
                        "chunk_id": "target:1",
                        "source_title": "정답 문서",
                    },
                    {
                        "document_id": "target",
                        "chunk_id": "target:2",
                        "source_title": "정답 문서",
                    },
                ],
            },
            "q2": {
                "id": "q2",
                "latency_ms": 220,
                "answer": "같은 답변",
                "sources": [
                    {
                        "document_id": "q2",
                        "chunk_id": "q2:gold",
                        "source_title": "두 번째 정답",
                    }
                ],
            },
        }

        report, csv_rows = analysis_module.analyze_runs(
            ["q1", "q2"],
            cases,
            run_a,
            run_b,
            label_a="cap2",
            label_b="cap4",
        )

        self.assertEqual(report["arms"]["cap2"]["document"]["hit_at_5"]["count"], 1)
        self.assertEqual(report["arms"]["cap4"]["document"]["hit_at_5"]["count"], 2)
        self.assertEqual(report["arms"]["cap2"]["document"]["mrr_at_5"], 0.5)
        self.assertEqual(report["arms"]["cap4"]["document"]["mrr_at_5"], 1.0)
        self.assertEqual(report["arms"]["cap2"]["latency_ms"]["p50"], 150.0)
        self.assertEqual(report["arms"]["cap2"]["latency_ms"]["p95"], 195.0)
        self.assertEqual(
            report["comparison"]["case_changes"]["document_hit_at_5"]["gained"],
            ["q2"],
        )
        self.assertEqual(
            report["comparison"]["case_changes"]["all_gold_chunk_at_8"]["gained"],
            ["q1", "q2"],
        )
        self.assertEqual(
            report["comparison"]["source_changed"],
            ["q1", "q2"],
        )
        self.assertEqual(report["comparison"]["answer_changed"], ["q1"])
        self.assertTrue(report["cases"][0]["changes"]["answer_changed"])
        self.assertEqual(report["cases"][0]["query"], "")
        self.assertEqual(report["cases"][0]["cap2"]["source_titles"], ["정답 문서", "기타"])
        self.assertEqual(len(csv_rows), 2)
        self.assertIn("gold_chunk_recall_at_8_b", csv_rows[0])
        self.assertIn("chunk_rank_changes", csv_rows[0])

        html = analysis_module.render_html(
            report,
            ["q1", "q2"],
            cases,
            run_a,
            run_b,
            label_a="cap2",
            label_b="cap4",
        )
        self.assertIn("신규/상실/변경 필터", html)
        self.assertIn("A 답변", html)
        self.assertIn("target:2", html)
        self.assertIn("정답 문서", html)

    def test_render_html_escapes_service_content(self) -> None:
        cases = {
            "q1": {
                "id": "q1",
                "query": "<script>alert(1)</script>",
                "expected": {"source_title_contains": "정답"},
                "evidence": [],
            }
        }
        run = {
            "q1": {
                "id": "q1",
                "latency_ms": 10,
                "answer": "<b>답변</b>",
                "sources": [
                    {
                        "document_id": "doc",
                        "chunk_id": "doc:1",
                        "source_title": "<정답>",
                    }
                ],
            }
        }
        report, _ = analysis_module.analyze_runs(
            ["q1"],
            cases,
            run,
            run,
            label_a="a",
            label_b="b",
        )

        html = analysis_module.render_html(
            report,
            ["q1"],
            cases,
            run,
            run,
            label_a="a",
            label_b="b",
        )

        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("&lt;b&gt;답변&lt;/b&gt;", html)


if __name__ == "__main__":
    unittest.main()
