from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_generation_gfc_repeats",
    ROOT / "scripts" / "analyze_generation_gfc_repeats.py",
)
assert SPEC is not None and SPEC.loader is not None
analysis_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis_module)


def run_with_votes(run_id: str, votes: dict[str, bool]) -> dict[str, dict]:
    return {
        case_id: {
            "experiment_id": "exp",
            "condition_id": "c1",
            "generation_run_id": run_id,
            "judge": {"grounded_fully_correct": value},
        }
        for case_id, value in votes.items()
    }


class AnalyzeGenerationGFCRepeatsTests(unittest.TestCase):
    def test_strict_majority(self) -> None:
        self.assertTrue(analysis_module.strict_majority([True, False, True]))
        self.assertFalse(analysis_module.strict_majority([False, True, False]))
        with self.assertRaisesRegex(ValueError, "odd number"):
            analysis_module.strict_majority([True, False])

    def test_summarize_condition_keeps_question_as_sample(self) -> None:
        order = ["q1", "q2"]
        cases = {"q1": {"category": "a"}, "q2": {"category": "a"}}
        runs = [
            run_with_votes("run1", {"q1": True, "q2": False}),
            run_with_votes("run2", {"q1": False, "q2": False}),
            run_with_votes("run3", {"q1": True, "q2": True}),
        ]

        summary, per_question = analysis_module.summarize_condition(
            order, cases, runs
        )

        self.assertEqual(summary["effective_sample_n"], 2)
        self.assertEqual(summary["run_gfc_counts"], [1, 0, 2])
        self.assertEqual(summary["majority_gfc_count"], 1)
        self.assertTrue(per_question["q1"]["majority"])
        self.assertFalse(per_question["q2"]["majority"])

    def test_duplicate_generation_run_id_is_rejected(self) -> None:
        order = ["q1"]
        runs = [
            run_with_votes("run1", {"q1": True}),
            run_with_votes("run1", {"q1": False}),
            run_with_votes("run3", {"q1": True}),
        ]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            analysis_module.validate_run_bindings(runs, order)

    def test_exact_mcnemar_matches_known_values(self) -> None:
        self.assertEqual(analysis_module.exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(analysis_module.exact_mcnemar_p(11, 1), 0.00634765625)
        self.assertEqual(analysis_module.exact_mcnemar_p(1, 1), 1.0)

    def test_compare_conditions_counts_gains_and_losses(self) -> None:
        order = ["q1", "q2", "q3", "q4"]
        cases = {case_id: {} for case_id in order}
        per_a = {
            "q1": {"votes": [False] * 3, "true_count": 0, "majority": False},
            "q2": {"votes": [True] * 3, "true_count": 3, "majority": True},
            "q3": {"votes": [True] * 3, "true_count": 3, "majority": True},
            "q4": {"votes": [False] * 3, "true_count": 0, "majority": False},
        }
        per_b = {
            "q1": {"votes": [True] * 3, "true_count": 3, "majority": True},
            "q2": {"votes": [False] * 3, "true_count": 0, "majority": False},
            "q3": {"votes": [True] * 3, "true_count": 3, "majority": True},
            "q4": {"votes": [False] * 3, "true_count": 0, "majority": False},
        }

        comparison, rows = analysis_module.compare_conditions(
            order, cases, per_a, per_b, 100, 7
        )

        self.assertEqual(comparison["gains_b"], 1)
        self.assertEqual(comparison["losses_b"], 1)
        self.assertEqual(comparison["ties_true"], 1)
        self.assertEqual(comparison["ties_false"], 1)
        self.assertEqual(comparison["majority_gfc_rate_delta_b_minus_a"], 0.0)
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
