from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_generation_failures import (
    FailureAnalysisError,
    analyze,
    failure_causes,
    join_artifacts,
    main,
)


class GenerationFailureAnalysisTests(unittest.TestCase):
    def answer(self, answer_id: str, case_id: str, condition: str = "c1") -> dict:
        return {
            "answer_id": answer_id,
            "answer_sha256": answer_id.rjust(64, "0"),
            "case_id": case_id,
            "condition_id": condition,
            "generation_run_id": "run1",
            "retrieval_hit": {"matched": True},
            "evidence_at_k": {"8": {"all_matched": True}},
        }

    def judgment(
        self,
        answer: dict,
        *,
        score: int,
        gfc: bool,
        statuses: list[str],
        abstention: str = "not_applicable",
        citation: str = "full",
        unsupported: list[str] | None = None,
        contradictions: list[str] | None = None,
        guard_rules: list[str] | None = None,
    ) -> dict:
        return {
            "answer_id": answer["answer_id"],
            "answer_sha256": answer["answer_sha256"],
            "case_id": answer["case_id"],
            "condition_id": answer["condition_id"],
            "generation_run_id": answer["generation_run_id"],
            "error": None,
            "judge": {
                "score": score,
                "grounded_fully_correct": gfc,
                "claim_checks": [
                    {"status": status, "claim_id": f"evidence_{index}"}
                    for index, status in enumerate(statuses, 1)
                ],
                "abstention": abstention,
                "citation_support": citation,
                "unsupported_facts": unsupported or [],
                "contradictions": contradictions or [],
                "reason": "fixture",
            },
            "deterministic_guard": {
                "applied": bool(guard_rules),
                "rules": guard_rules or [],
            },
        }

    def fixture(self) -> tuple[list[dict], list[dict]]:
        answers = [self.answer(str(index), f"case-{index}") for index in range(1, 5)]
        judgments = [
            self.judgment(answers[0], score=2, gfc=True, statuses=["supported"]),
            self.judgment(
                answers[1],
                score=0,
                gfc=False,
                statuses=["missing"],
                abstention="inappropriate",
                citation="not_applicable",
                guard_rules=["answerable_clear_refusal_forces_score_zero"],
            ),
            self.judgment(
                answers[2],
                score=0,
                gfc=False,
                statuses=["contradicted"],
                citation="partial",
                contradictions=["wrong date"],
            ),
            self.judgment(
                answers[3],
                score=1,
                gfc=False,
                statuses=["unsupported"],
                unsupported=["unsupported detail"],
            ),
        ]
        return answers, judgments

    def test_multilabel_and_primary_partition(self) -> None:
        answers, judgments = self.fixture()
        payload = analyze(join_artifacts(answers, judgments))
        summary = payload["conditions"]["c1"]
        self.assertEqual(summary["mean_score"], 0.75)
        self.assertEqual(summary["gfc_count"], 1)
        self.assertEqual(summary["non_gfc_count"], 3)
        self.assertEqual(sum(summary["failure_primary_partition"].values()), 3)
        self.assertEqual(summary["runs"]["run1"]["non_gfc_count"], 3)
        self.assertEqual(summary["failure_causes_multilabel"]["partial_citation"], 1)
        self.assertEqual(
            summary["guard_rule_counts"]["answerable_clear_refusal_forces_score_zero"],
            1,
        )

    def test_failure_causes_preserve_overlap(self) -> None:
        answers, judgments = self.fixture()
        causes = failure_causes(judgments[1])
        self.assertEqual(
            causes,
            [
                "explicit_refusal_guard",
                "inappropriate_abstention",
                "required_claim_missing",
            ],
        )

    def test_join_rejects_missing_judgment(self) -> None:
        answers, judgments = self.fixture()
        with self.assertRaises(FailureAnalysisError):
            join_artifacts(answers, judgments[:-1])

    def test_cli_validates_summary_and_refuses_overwrite(self) -> None:
        answers, judgments = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            judgment_path = root / "judgments.jsonl"
            summary_path = root / "summary.json"
            output_json = root / "analysis.json"
            output_csv = root / "analysis.csv"
            answer_path.write_text(
                "\n".join(json.dumps(row) for row in answers) + "\n",
                encoding="utf-8",
            )
            judgment_path.write_text(
                "\n".join(json.dumps(row) for row in judgments) + "\n",
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "labels": {"b": "C1"},
                        "b": {
                            "mean": 0.75,
                            "score_counts": {"0": 2, "1": 1, "2": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = [
                "--answers",
                str(answer_path),
                "--judgments",
                str(judgment_path),
                "--expected-summary",
                str(summary_path),
                "--out-json",
                str(output_json),
                "--out-csv",
                str(output_csv),
            ]
            self.assertEqual(main(args), 0)
            result = json.loads(output_json.read_text())
            self.assertTrue(result["expected_summary_validation"]["all_match"])
            with self.assertRaises(FailureAnalysisError):
                main(args)


if __name__ == "__main__":
    unittest.main()
