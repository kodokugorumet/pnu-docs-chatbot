from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_rule_activation import (
    ActivationError,
    analyze_records,
    load_inventory,
    main,
)


class RuleActivationTests(unittest.TestCase):
    def make_inventory(self, root: Path) -> Path:
        path = root / "inventory.md"
        path.write_text(
            "\n".join(
                (
                    "| Rule ID | name | class | target |",
                    "|---|---|---|---|",
                    "| `EXP-003` | d2 | 표적 | `svc_intl_03` |",
                    "| `FACET-002` | sibling | 일반 | — |",
                    "| `FACET-011` | d2 facet | 표적 | `svc_intl_03` |",
                    "| `GEN-012` | d2 prompt | 표적 | `svc_intl_03` |",
                    "| `POST-002` | abstention | 일반 | — |",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def make_records(self) -> list[dict[str, object]]:
        return [
            {
                "case_id": "svc_intl_03",
                "condition_id": "c1",
                "generation_run_id": "run1",
                "query": "D-2 비자 연장 단체접수는 어떻게 이용하나요?",
                "retrieval": {
                    "retrieval_query": "단체접수 체류기간 사전 예약 제출서류 수수료",
                    "neighbor_expansion": {
                        "enabled": True,
                        "completed_count": 2,
                        "group_visa_application": True,
                    },
                },
                "evaluation_trace": {
                    "raw_draft": "제공된 근거에서 확인할 수 없습니다.",
                    "generation_input": {
                        "user_prompt": "수수료 금액과 현금·권종 등 납부방식"
                    },
                },
                "claims": [
                    {
                        "text": "확인할 수 없습니다.",
                        "supported": False,
                        "validation_reason": "model_abstention",
                    }
                ],
            },
            {
                "case_id": "svc_other",
                "condition_id": "c1",
                "generation_run_id": "run1",
                "query": "도서관 운영시간은 언제인가요?",
                "retrieval": {
                    "retrieval_query": "도서관 운영시간 언제인가요",
                    "neighbor_expansion": {
                        "enabled": True,
                        "completed_count": 0,
                        "group_visa_application": False,
                    },
                },
                "evaluation_trace": {
                    "raw_draft": "도서관은 9시에 엽니다.",
                    "generation_input": {"user_prompt": "일반 prompt"},
                },
                "claims": [],
            },
        ]

    def test_inventory_and_observed_possible_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory = load_inventory(self.make_inventory(Path(directory)))
        payload = analyze_records(self.make_records(), inventory)
        self.assertEqual(payload["inventory"]["rule_count"], 5)
        self.assertEqual(payload["inventory"]["target_case_count"], 1)
        self.assertEqual(
            payload["coverage"]["possible_targeted_intended_case_count"], 1
        )
        self.assertEqual(
            payload["coverage"]["possible_targeted_spillover_case_ids"], []
        )
        d2 = payload["records"][0]["rule_activations"]
        by_id = {item["rule_id"]: item for item in d2}
        self.assertEqual(by_id["EXP-003"]["trace_status"], "observed_active")
        self.assertIs(by_id["EXP-003"]["trigger_possible"], True)
        self.assertEqual(by_id["FACET-011"]["trace_status"], "observed_active")
        self.assertEqual(by_id["GEN-012"]["trace_status"], "observed_active")
        self.assertEqual(by_id["POST-002"]["trace_status"], "observed_active")

        other = payload["records"][1]["rule_activations"]
        by_id = {item["rule_id"]: item for item in other}
        self.assertEqual(by_id["EXP-003"]["trace_status"], "observed_inactive")
        self.assertIs(by_id["EXP-003"]["trigger_possible"], False)
        self.assertEqual(by_id["FACET-002"]["trace_status"], "observed_inactive")

    def test_missing_trace_is_not_converted_to_observed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory = load_inventory(self.make_inventory(Path(directory)))
        record = {
            "case_id": "svc_intl_03",
            "condition_id": "c0",
            "generation_run_id": "run1",
            "query": "D-2 비자 연장 단체접수는 어떻게 이용하나요?",
        }
        payload = analyze_records([record], inventory)
        by_id = {
            item["rule_id"]: item
            for item in payload["records"][0]["rule_activations"]
        }
        self.assertEqual(by_id["FACET-011"]["trace_status"], "trace_absent")
        self.assertIs(by_id["FACET-011"]["trigger_possible"], True)

    def test_cli_writes_json_and_csv_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self.make_inventory(root)
            answers = root / "answers.jsonl"
            answers.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in self.make_records())
                + "\n",
                encoding="utf-8",
            )
            output_json = root / "analysis.json"
            output_csv = root / "analysis.csv"
            args = [
                "--answers",
                str(answers),
                "--inventory",
                str(inventory),
                "--out-json",
                str(output_json),
                "--out-csv",
                str(output_csv),
            ]
            self.assertEqual(main(args), 0)
            self.assertTrue(output_json.is_file())
            self.assertIn("observed_active_rules", output_csv.read_text())
            with self.assertRaises(ActivationError):
                main(args)

    def test_duplicate_inventory_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = self.make_inventory(root)
            with inventory.open("a", encoding="utf-8") as target:
                target.write("| `EXP-003` | duplicate | 표적 | `svc_x` |\n")
            with self.assertRaises(ActivationError):
                load_inventory(inventory)


if __name__ == "__main__":
    unittest.main()
