from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_judge_repeat_selection.py"
SPEC = importlib.util.spec_from_file_location(
    "service_eval_artifacts_for_repeat_selection_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
assert SPEC is not None and SPEC.loader is not None
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


class BuildJudgeRepeatSelectionTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def answer(self, case_id: str) -> dict:
        return artifacts.build_answer_identity(
            {
                "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "selection-fixture",
                "condition_id": "c0",
                "generation_run_id": "run1",
                "case_id": case_id,
                "answer": f"answer {case_id}",
                "error": None,
            }
        )

    def run_cli(self, answers: Path, out: Path, case_ids: str):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT),
                "--answers",
                str(answers),
                "--case-ids",
                case_ids,
                "--out",
                str(out),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_builds_ordered_hash_bound_selection_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers_path = root / "answers.jsonl"
            answers = [self.answer(f"q{index}") for index in range(1, 11)]
            self.write_jsonl(answers_path, answers)
            output = root / "selection.json"

            selection = "q9,q1,q2,q3,q4,q5,q6,q7,q8"
            result = self.run_cli(answers_path, output, selection)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["selected_answer_ids"],
                [answer["answer_id"] for answer in answers[:9]],
            )
            self.assertTrue(payload["selection_id"].startswith("judge_repeat_selection_"))
            self.assertEqual(len(payload["selection_sha256"]), 64)

            second = self.run_cli(answers_path, output, selection)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_rejects_unknown_duplicate_and_output_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers_path = root / "answers.jsonl"
            answers = [self.answer(f"q{index}") for index in range(1, 10)]
            self.write_jsonl(answers_path, answers)
            unknown = self.run_cli(
                answers_path,
                root / "unknown.json",
                "q1,q2,q3,q4,q5,q6,q7,q8,q-missing",
            )
            self.assertNotEqual(unknown.returncode, 0)
            self.assertIn("unknown selected case ids", unknown.stderr)
            wrong_count = self.run_cli(
                answers_path,
                root / "wrong-count.json",
                "q1,q2,q3,q4,q5,q6,q7,q8",
            )
            self.assertNotEqual(wrong_count.returncode, 0)
            self.assertIn("fixed expected 9", wrong_count.stderr)
            duplicate = self.run_cli(
                answers_path,
                root / "duplicate.json",
                "q1,q1,q2,q3,q4,q5,q6,q7,q8",
            )
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("duplicate selected case ids", duplicate.stderr)
            alias = self.run_cli(
                answers_path,
                answers_path,
                "q1,q2,q3,q4,q5,q6,q7,q8,q9",
            )
            self.assertNotEqual(alias.returncode, 0)
            self.assertIn("must not overwrite", alias.stderr)

    def test_rejects_terminal_selected_slot_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers_path = root / "answers.jsonl"
            successes = []
            for case_id in ["q1", *[f"q{index}" for index in range(3, 10)]]:
                success = self.answer(case_id)
                success["slot_outcome"] = "answer"
                success["answer_eligible_for_judge"] = True
                success["collector_config"] = {"max_attempts": 3}
                success["collector_config_sha256"] = artifacts.sha256_json(
                    success["collector_config"]
                )
                success["request_attempts"] = [
                    {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
                ]
                success["collection_attempt_number"] = 1
                successes.append(artifacts.build_answer_identity(success))
            terminal = self.answer("q2")
            attempts = [
                {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
            ]
            terminal.update(
                {
                    "slot_outcome": "service_error",
                    "answer_eligible_for_judge": False,
                    "answer": "[SERVICE_ERROR]",
                    "collector_config": {"max_attempts": 3},
                    "collector_config_sha256": artifacts.sha256_json(
                        {"max_attempts": 3}
                    ),
                    "request_attempts": attempts,
                    "collection_attempt_number": 1,
                    "service_error": {
                        "stage": "response_validation",
                        "type": "AnswerPayloadError",
                        "message": "response answer must be a non-empty string",
                        "retryable": False,
                        "http_status": None,
                        "request_attempts": attempts,
                    },
                }
            )
            terminal = artifacts.build_answer_identity(terminal)
            self.write_jsonl(answers_path, [successes[0], terminal, *successes[1:]])

            result = self.run_cli(
                answers_path,
                root / "selection.json",
                "q1,q2,q3,q4,q5,q6,q7,q8,q9",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stability diagnosis is incomplete", result.stderr)
            self.assertIn("no replacement is allowed", result.stderr)

    def test_rejects_symlink_input_and_broken_symlink_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers_path = root / "answers.jsonl"
            answers = [self.answer(f"q{index}") for index in range(1, 10)]
            self.write_jsonl(answers_path, answers)
            answers_link = root / "answers-link.jsonl"
            answers_link.symlink_to(answers_path)

            result = self.run_cli(
                answers_link,
                root / "selection.json",
                "q1,q2,q3,q4,q5,q6,q7,q8,q9",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not be symlinks", result.stderr)
            self.assertFalse((root / "selection.json").exists())

            broken_output = root / "broken-selection.json"
            broken_output.symlink_to(root / "missing-target.json")
            result = self.run_cli(
                answers_path,
                broken_output,
                "q1,q2,q3,q4,q5,q6,q7,q8,q9",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertTrue(broken_output.is_symlink())
            self.assertFalse((root / "selection.json").exists())


if __name__ == "__main__":
    unittest.main()
