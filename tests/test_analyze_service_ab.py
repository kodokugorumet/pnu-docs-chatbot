from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_service_ab", ROOT / "scripts" / "analyze_service_ab.py"
)
assert SPEC is not None and SPEC.loader is not None
analysis_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis_module)


class AnalyzeServiceABTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def test_load_complete_run_rejects_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_jsonl(path, [{"id": "q1", "judge": {"score": 2}}])

            with self.assertRaisesRegex(ValueError, "incomplete ID set"):
                analysis_module.load_complete_run(path, {"q1", "q2"})

    def test_load_complete_run_joins_separate_judgments(self) -> None:
        answer = analysis_module.build_answer_identity(
            {
                "schema_version": analysis_module.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "exp",
                "condition_id": "c0",
                "generation_run_id": "g1",
                "case_id": "q1",
                "id": "q1",
                "answer": "답변",
                "sources": [],
            }
        )
        judgment = analysis_module.build_judgment_identity(
            {
                "schema_version": analysis_module.JUDGMENT_SCHEMA_VERSION,
                "record_type": "judgment",
                "experiment_id": "exp",
                "judge_run_id": "j1",
                "case_id": "q1",
                "answer_id": answer["answer_id"],
                "answer_sha256": answer["answer_sha256"],
                "judge": {"score": 2, "reason": "ok"},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            answers_path = Path(directory) / "answers.jsonl"
            judgments_path = Path(directory) / "judgments.jsonl"
            self.write_jsonl(answers_path, [answer])
            self.write_jsonl(judgments_path, [judgment])

            run = analysis_module.load_complete_run(
                answers_path,
                {"q1"},
                judgment_path=judgments_path,
                allow_legacy_inline=False,
            )

        self.assertEqual(run["q1"]["judge"]["score"], 2)
        self.assertEqual(run["q1"]["judgment"]["judge_run_id"], "j1")

    def test_load_complete_run_requires_explicit_legacy_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_jsonl(path, [{"id": "q1", "judge": {"score": 2}}])

            with self.assertRaisesRegex(ValueError, "separate judgment"):
                analysis_module.load_complete_run(
                    path, {"q1"}, allow_legacy_inline=False
                )

    def test_retrieval_values_backfills_evidence_from_sources(self) -> None:
        case = {
            "expected": {"source_title_contains": "등록금 납부 계획"},
            "evidence": [{"chunk_id": "doc:2"}],
        }
        record = {
            "retrieval_hit": {"matched": True, "rank": 2},
            "sources": [{"source_title": "안내", "chunk_id": "doc:2"}],
        }

        values = analysis_module.retrieval_values(record, case)

        self.assertTrue(values["hit"])
        self.assertEqual(values["rank"], 2)
        self.assertTrue(values["evidence_hit"])

    def test_retrieval_values_separates_evidence_at_five_and_eight(self) -> None:
        case = {
            "expected": {"source_title_contains": "등록금 납부 계획"},
            "evidence": [
                {"chunk_id": "doc:6"},
                {"chunk_id": "doc:7"},
                {
                    "chunk_id": "doc:99",
                    "required_for_answer": False,
                },
            ],
        }
        record = {
            "retrieval_hit": {"matched": True, "rank": 1},
            "sources": [
                {"source_title": "안내", "chunk_id": f"doc:{rank}"}
                for rank in range(1, 9)
            ],
        }

        values = analysis_module.retrieval_values(record, case)

        self.assertFalse(values["evidence_at_k"]["5"]["matched"])
        self.assertEqual(values["evidence_at_k"]["5"]["recall"], 0.0)
        self.assertTrue(values["evidence_at_k"]["8"]["all_matched"])
        self.assertEqual(values["evidence_at_k"]["8"]["recall"], 1.0)
        self.assertEqual(values["evidence_at_k"]["8"]["wanted"], 2)

    def test_cluster_key_groups_role_variant_with_base(self) -> None:
        self.assertEqual(
            analysis_module.cluster_key("svc_reg_01_role_student"), "svc_reg_01"
        )
        self.assertEqual(analysis_module.cluster_key("svc_reg_01"), "svc_reg_01")


if __name__ == "__main__":
    unittest.main()
