from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_service_ab_review", ROOT / "scripts" / "build_service_ab_review.py"
)
assert SPEC is not None and SPEC.loader is not None
review_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_module)


class BuildServiceABReviewTests(unittest.TestCase):
    def test_build_payload_uses_summary_labels(self) -> None:
        summary = {
            "labels": {"a": "C1", "b": "C2-v3"},
            "a": {"runs": 1},
            "b": {"runs": 1},
            "comparison": {},
            "questions": [{"id": "q1"}],
        }
        case = {"id": "q1", "query": "질문", "reference": "정답"}
        answer = {"id": "q1", "answer": "답변", "sources": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            cases_path = root / "cases.jsonl"
            run_a = root / "a.jsonl"
            run_b = root / "b.jsonl"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            cases_path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            run_a.write_text(json.dumps(answer) + "\n", encoding="utf-8")
            run_b.write_text(json.dumps(answer) + "\n", encoding="utf-8")

            payload = review_module.build_payload(
                summary_path, cases_path, [run_a], [run_b]
            )

        self.assertEqual(payload["summary"]["labels"], {"a": "C1", "b": "C2-v3"})

    def test_build_payload_rejects_empty_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            cases_path = root / "cases.jsonl"
            run_a = root / "a.jsonl"
            run_b = root / "b.jsonl"
            summary_path.write_text(
                json.dumps(
                    {
                        "labels": {"a": "", "b": "B"},
                        "a": {},
                        "b": {},
                        "comparison": {},
                        "questions": [],
                    }
                ),
                encoding="utf-8",
            )
            cases_path.write_text("", encoding="utf-8")
            run_a.write_text("", encoding="utf-8")
            run_b.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "labels.a"):
                review_module.build_payload(
                    summary_path, cases_path, [run_a], [run_b]
                )


if __name__ == "__main__":
    unittest.main()
