from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_grounding_guard import (  # noqa: E402
    SCHEMA_VERSION,
    build_report,
    load_cases,
    main,
)


CASES = ROOT / "config" / "pnu-grounding-adversarial-eval.jsonl"


class GroundingGuardEvaluationTest(unittest.TestCase):
    def test_default_config_has_required_adversarial_coverage(self) -> None:
        cases = load_cases(CASES)
        self.assertGreaterEqual(len(cases), 16)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

        ids = {case["id"] for case in cases}
        required_ids = {
            "permission_reversed_denied",
            "permission_matching_allowed",
            "comparator_ge_reversed",
            "comparator_ge_matching",
            "comparator_le_reversed",
            "comparator_le_matching",
            "comparator_gt_reversed",
            "comparator_gt_matching",
            "comparator_lt_reversed",
            "comparator_lt_matching",
            "comparator_money_equivalent_reversed",
            "comparator_percent_spelling_reversed",
            "comparator_negated_same_operator",
            "direction_rows_correct",
            "direction_rows_swapped",
            "substring_no_abnormality",
            "substring_unavoidable",
            "permission_double_negation",
            "substring_no_possibility",
            "substring_impressive",
            "substring_possibility_false_support",
            "substring_impressive_false_support",
            "later_compatible_source",
            "permission_year_scope_conflict",
            "comparator_year_scope_conflict",
            "direction_subject_value_conflict",
            "direction_negated_increase",
            "critical_date_mismatch",
            "table_row_value_mixing",
            "pnu_abbreviated_date_matching",
            "permission_relation_unverified",
            "direction_value_after_relation_conflict",
            "direction_value_after_relation_positive",
            "tab_separated_relation_rows",
            "permission_contrastive_clause",
            "title_scope_permission_positive",
            "title_scope_permission_reversed",
            "unscoped_permission_ambiguous_years",
            "negated_direction_not_frozen",
            "permission_copular_double_negation",
            "comparator_conditional_modality",
            "compound_man_currency_reversal",
            "quantity_unit_collision",
            "decimal_quantity_positive",
            "minimum_maximum_reversal",
            "temporal_boundary_reversal",
            "temporal_range_order_reversal",
            "permission_under_review",
            "direction_planned_not_completed",
            "metadata_year_conflict",
            "title_scope_comparator_positive",
            "title_scope_direction_positive",
            "permission_compound_false_support",
            "permission_compound_second_positive",
            "permission_exclusive_subject",
            "temporal_subject_binding",
            "calendar_year_scope_conflict",
            "comparator_postfix_subject",
            "direction_possible_not_asserted",
            "comparator_not_required",
            "permission_allowed_negated",
            "direction_plan_cancelled",
            "temporal_boundary_negated",
            "temporal_boundary_planned",
            "direction_slash_rows",
            "direction_and_rows",
            "time_only_boundary_reversal",
            "comparator_shared_subject_positive",
            "title_round_scope_positive",
            "temporal_omitted_endpoint_year_positive",
            "temporal_specific_time_positive",
            "direction_decrease_synonym_positive",
            "permission_shared_subject_positive",
            "comparator_complement_positive",
        }
        self.assertEqual(required_ids - ids, set())

    def test_ablation_is_complete_and_tuned_condition_matches_oracle(self) -> None:
        report = build_report(CASES)

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertTrue(report["benchmark"]["synthetic"])
        self.assertEqual(
            report["benchmark"]["kind"],
            "synthetic_adversarial_microbenchmark",
        )
        self.assertEqual(
            report["config"]["sha256"],
            hashlib.sha256(CASES.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            [row["case_id"] for row in report["cases"]],
            report["config"]["case_ids"],
        )

        legacy = report["conditions"]["legacy"]["metrics"]
        tuned = report["conditions"]["tuned"]["metrics"]
        tuned_errors = [
            row["case_id"]
            for row in report["cases"]
            if not row["tuned"]["joint_correct"]
        ]
        self.assertEqual(tuned_errors, [])
        self.assertEqual(tuned["label_accuracy"], 1.0)
        self.assertEqual(tuned["reason_accuracy"], 1.0)
        self.assertEqual(tuned["source_ids_accuracy"], 1.0)
        self.assertEqual(tuned["joint_accuracy"], 1.0)
        self.assertEqual(tuned["error_case_ids"], [])
        self.assertGreater(tuned["joint_accuracy"], legacy["joint_accuracy"])
        self.assertGreater(legacy["confusion"]["false_positive"], 0)
        self.assertEqual(report["comparison"]["regression_case_ids"], [])

    def test_output_is_byte_deterministic_and_csv_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "report.json"
            csv_path = root / "report.csv"
            args = [
                "--cases",
                str(CASES),
                "--output-json",
                str(json_path),
                "--output-csv",
                str(csv_path),
            ]
            self.assertEqual(main(args), 0)
            first_json = json_path.read_bytes()
            first_csv = csv_path.read_bytes()
            self.assertEqual(main(args), 0)
            self.assertEqual(first_json, json_path.read_bytes())
            self.assertEqual(first_csv, csv_path.read_bytes())

            report = json.loads(first_json)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), report["config"]["case_count"])
            self.assertEqual(
                [row["case_id"] for row in rows], report["config"]["case_ids"]
            )

    def test_loader_rejects_duplicate_and_invalid_cases(self) -> None:
        valid = {
            "id": "duplicate",
            "claim": "등록금을 납부합니다.",
            "sources": [{"chunk_id": "doc#1", "text": "등록금을 납부합니다."}],
            "expected_supported": True,
            "expected_reason": "supported",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_path = root / "duplicate.jsonl"
            duplicate_path.write_text(
                json.dumps(valid, ensure_ascii=False)
                + "\n"
                + json.dumps(valid, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                load_cases(duplicate_path)

            invalid_path = root / "invalid.jsonl"
            invalid = dict(valid)
            invalid.pop("sources")
            invalid_path.write_text(
                json.dumps(invalid, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sources required"):
                load_cases(invalid_path)


if __name__ == "__main__":
    unittest.main()
