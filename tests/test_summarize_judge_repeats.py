from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_judge_repeats.py"
ARTIFACTS_SPEC = importlib.util.spec_from_file_location(
    "service_eval_artifacts_for_repeat_summary_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
assert ARTIFACTS_SPEC is not None and ARTIFACTS_SPEC.loader is not None
artifacts = importlib.util.module_from_spec(ARTIFACTS_SPEC)
ARTIFACTS_SPEC.loader.exec_module(artifacts)


class SummarizeJudgeRepeatsTests(unittest.TestCase):
    maxDiff = None

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def file_sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def answer_record(self, case_id: str) -> dict:
        return artifacts.build_answer_identity(
            {
                "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "repeat-summary-fixture",
                "condition_id": "c1",
                "generation_run_id": "generation-1",
                "case_id": case_id,
                "id": case_id,
                "answer": f"{case_id} answer",
                "error": None,
            }
        )

    def terminal_answer_record(self, case_id: str) -> dict:
        attempts = [
            {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
        ]
        return artifacts.build_answer_identity(
            {
                "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "repeat-summary-fixture",
                "condition_id": "c1",
                "generation_run_id": "generation-1",
                "case_id": case_id,
                "id": case_id,
                "answer": "[SERVICE_ERROR]",
                "slot_outcome": "service_error",
                "answer_eligible_for_judge": False,
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
                "error": None,
            }
        )

    def final_answer_records(
        self, *, generation_run_id: str = "run1", condition_id: str = "c0"
    ) -> list[dict]:
        case_ids = [f"q{index:02d}" for index in range(1, 28)]
        authorization = {
            "schema_version": "pnu.final-generation-authorization.v1",
            "collection_purpose": "final_generation",
            "artifact_id": f"{condition_id}-core-{generation_run_id}",
            "split": "holdout-core",
            "expected_case_count": 27,
            "expected_case_ids_sha256": artifacts.sha256_json(case_ids),
        }
        collector_config = {
            "max_attempts": 3,
            "final_authorization": authorization,
        }
        records: list[dict] = []
        for case_id in case_ids:
            answer = self.answer_record(case_id)
            answer.pop("answer_id", None)
            answer.pop("record_sha256", None)
            answer.update(
                condition_id=condition_id,
                generation_run_id=generation_run_id,
                slot_outcome="answer",
                answer_eligible_for_judge=True,
                collector_config=collector_config,
                collector_config_sha256=artifacts.sha256_json(collector_config),
                request_attempts=[
                    {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
                ],
                collection_attempt_number=1,
            )
            records.append(artifacts.build_answer_identity(answer))
        return records

    def judgment_record(
        self,
        answer: dict,
        *,
        run_id: str,
        score: int,
        gfc: bool | None,
        answers_artifact_sha256: str,
        error: str | None = None,
        config_label: str = "same-config",
        selection_binding: dict | None = None,
    ) -> dict:
        judge_config = {"provider": "fixture", "model": config_label}
        record = {
            "schema_version": artifacts.JUDGMENT_SCHEMA_VERSION,
            "record_type": "judgment",
            "experiment_id": answer["experiment_id"],
            "condition_id": answer["condition_id"],
            "generation_run_id": answer["generation_run_id"],
            "judge_run_id": run_id,
            "case_id": answer["case_id"],
            "answer_id": answer["answer_id"],
            "answer_sha256": answer["answer_sha256"],
            "answer_record_sha256": artifacts.sha256_json(answer),
            "answers_artifact_sha256": answers_artifact_sha256,
            "judge_config": judge_config,
            "judge_config_sha256": artifacts.sha256_json(judge_config),
            "judge": {"score": score},
            "error": error,
        }
        if selection_binding is not None:
            record["judge_repeat_selection"] = selection_binding
        if gfc is not None:
            record["judge"]["grounded_fully_correct"] = gfc
        return artifacts.build_judgment_identity(record)

    def run_cli(
        self,
        *,
        answers: Path,
        judgments: list[Path],
        json_out: Path,
        csv_out: Path | None = None,
        selection_manifest: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--answers",
            str(answers),
            "--judgments",
            *(str(path) for path in judgments),
            "--json-out",
            str(json_out),
        ]
        if csv_out is not None:
            command.extend(["--csv-out", str(csv_out)])
        if selection_manifest is not None:
            command.extend(["--selection-manifest", str(selection_manifest)])
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def write_selection(self, path: Path, answers_path: Path, answers: list[dict]) -> None:
        selected_ids = [str(answer["answer_id"]) for answer in answers]
        payload = {
            "schema_version": "pnu.judge-repeat-selection.v1",
            "answers_artifact_sha256": self.file_sha256(answers_path),
            "selected_answer_ids": selected_ids,
            "selected_answer_ids_sha256": artifacts.sha256_json(selected_ids),
        }
        payload["selection_sha256"] = artifacts.sha256_json(payload)
        payload["selection_id"] = (
            "judge_repeat_selection_" + payload["selection_sha256"][:24]
        )
        path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def selection_binding(self, path: Path) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "schema_version": payload["schema_version"],
            "selection_id": payload["selection_id"],
            "selection_sha256": payload["selection_sha256"],
            "selection_artifact_sha256": self.file_sha256(path),
            "answers_artifact_sha256": payload["answers_artifact_sha256"],
            "selected_answer_count": len(payload["selected_answer_ids"]),
            "selected_answer_ids_sha256": payload[
                "selected_answer_ids_sha256"
            ],
        }

    def test_cli_aggregates_three_repeats_at_question_level_and_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "partial-answers.jsonl"
            answers = [self.answer_record(case_id) for case_id in ("q1", "q2", "q3")]
            self.write_jsonl(answer_path, answers)
            answer_artifact_sha = self.file_sha256(answer_path)

            run_values = [
                ("judge-1", [(2, True), (0, False), (0, False)]),
                ("judge-2", [(2, True), (1, False), (0, False)]),
                ("judge-3", [(1, False), (2, True), (0, False)]),
            ]
            judgment_paths: list[Path] = []
            for run_id, values in run_values:
                path = root / f"{run_id}.jsonl"
                records = [
                    self.judgment_record(
                        answer,
                        run_id=run_id,
                        score=score,
                        gfc=gfc,
                        answers_artifact_sha256=answer_artifact_sha,
                    )
                    for answer, (score, gfc) in zip(answers, values)
                ]
                self.write_jsonl(path, records)
                judgment_paths.append(path)

            json_out = root / "summary.json"
            csv_out = root / "summary.csv"
            result = self.run_cli(
                answers=answer_path,
                judgments=judgment_paths,
                json_out=json_out,
                csv_out=csv_out,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "pnu.judge-repeat-summary.v1")
            self.assertEqual(
                payload["output_publication"][
                    "authoritative_completion_artifact"
                ],
                "json",
            )
            companion = payload["output_publication"][
                "required_companion_artifacts"
            ][0]
            self.assertEqual(companion["kind"], "case_csv")
            self.assertEqual(companion["path"], str(csv_out.resolve()))
            self.assertEqual(companion["sha256"], self.file_sha256(csv_out))
            self.assertEqual(companion["bytes"], len(csv_out.read_bytes()))
            self.assertEqual(payload["methodology"]["sample_unit"], "question")
            self.assertFalse(
                payload["methodology"]["judge_repeats_count_as_additional_samples"]
            )
            self.assertEqual(payload["summary"]["question_count"], 3)
            self.assertEqual(payload["summary"]["effective_sample_n"], 3)
            self.assertEqual(payload["summary"]["judge_repeat_count"], 3)
            self.assertEqual(payload["summary"]["judgment_record_count"], 9)
            self.assertAlmostEqual(payload["summary"]["mean_score"], 8 / 9)
            self.assertEqual(
                payload["summary"]["question_level_majority_gfc_true_count"], 1
            )
            self.assertAlmostEqual(
                payload["summary"]["question_level_majority_gfc_rate"], 1 / 3
            )
            self.assertEqual(
                payload["summary"]["gfc_agreement_counts"], {"3/3": 1, "2/3": 2}
            )

            q1, q2, q3 = payload["cases"]
            self.assertEqual(q1["scores"], [2, 2, 1])
            self.assertEqual(q1["score_mode"], 2)
            self.assertIs(q1["gfc_majority"], True)
            self.assertEqual(q1["gfc_agreement_label"], "2/3")
            self.assertEqual(q2["scores"], [0, 1, 2])
            self.assertIsNone(q2["score_mode"])
            self.assertIs(q2["gfc_majority"], False)
            self.assertEqual(q3["gfc_agreement_label"], "3/3")

            with csv_out.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 3)
            self.assertEqual(json.loads(csv_rows[0]["scores"]), [2, 2, 1])
            self.assertEqual(
                json.loads(csv_rows[0]["grounded_fully_correct_values"]),
                [True, True, False],
            )

    def test_missing_case_in_any_judgment_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answers = [self.answer_record("q1"), self.answer_record("q2")]
            self.write_jsonl(answer_path, answers)
            artifact_sha = self.file_sha256(answer_path)
            judgment_path = root / "judge.jsonl"
            self.write_jsonl(
                judgment_path,
                [
                    self.judgment_record(
                        answers[0],
                        run_id="judge-1",
                        score=2,
                        gfc=True,
                        answers_artifact_sha256=artifact_sha,
                    )
                ],
            )

            output = root / "summary.json"
            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                json_out=output,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing judgment", result.stderr)
            self.assertFalse(output.exists())

    def test_full_mode_skips_terminal_slot_and_reports_reduced_stability_n(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            success = self.answer_record("q1")
            success.update(
                slot_outcome="answer",
                answer_eligible_for_judge=True,
                collector_config={"max_attempts": 3},
                collector_config_sha256=artifacts.sha256_json(
                    {"max_attempts": 3}
                ),
                request_attempts=[
                    {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
                ],
                collection_attempt_number=1,
            )
            success = artifacts.build_answer_identity(success)
            terminal = self.terminal_answer_record("q2")
            self.write_jsonl(answer_path, [success, terminal])
            artifact_sha = self.file_sha256(answer_path)
            judgment_paths: list[Path] = []
            for run_id in ("judge-1", "judge-2"):
                path = root / f"{run_id}.jsonl"
                self.write_jsonl(
                    path,
                    [
                        self.judgment_record(
                            success,
                            run_id=run_id,
                            score=2,
                            gfc=True,
                            answers_artifact_sha256=artifact_sha,
                        )
                    ],
                )
                judgment_paths.append(path)

            output = root / "summary.json"
            result = self.run_cli(
                answers=answer_path,
                judgments=judgment_paths,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["question_count"], 1)
            self.assertEqual(payload["summary"]["effective_sample_n"], 1)
            self.assertEqual(
                payload["summary"]["terminal_service_error_count"], 1
            )
            self.assertEqual(
                payload["summary"]["terminal_service_error_case_ids"], ["q2"]
            )
            self.assertEqual(payload["methodology"]["selected_logical_slot_count"], 2)

            invalid_path = root / "judged-terminal.jsonl"
            self.write_jsonl(
                invalid_path,
                [
                    self.judgment_record(
                        success,
                        run_id="judge-bad",
                        score=2,
                        gfc=True,
                        answers_artifact_sha256=artifact_sha,
                    ),
                    self.judgment_record(
                        terminal,
                        run_id="judge-bad",
                        score=0,
                        gfc=False,
                        answers_artifact_sha256=artifact_sha,
                    ),
                ],
            )
            invalid = self.run_cli(
                answers=answer_path,
                judgments=[invalid_path],
                json_out=root / "invalid-summary.json",
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("unknown answer_id", invalid.stderr)

            selection_path = root / "terminal-selection.json"
            self.write_selection(
                selection_path, answer_path, [success, terminal]
            )
            bound_path = root / "selection-bound.jsonl"
            self.write_jsonl(
                bound_path,
                [
                    self.judgment_record(
                        success,
                        run_id="judge-selection",
                        score=2,
                        gfc=True,
                        answers_artifact_sha256=artifact_sha,
                        selection_binding=self.selection_binding(selection_path),
                    )
                ],
            )
            incomplete = self.run_cli(
                answers=answer_path,
                judgments=[bound_path],
                selection_manifest=selection_path,
                json_out=root / "incomplete-selection.json",
            )
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("stability diagnosis is incomplete", incomplete.stderr)

    def test_duplicate_judgment_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answer = self.answer_record("q1")
            self.write_jsonl(answer_path, [answer])
            record = self.judgment_record(
                answer,
                run_id="judge-1",
                score=2,
                gfc=True,
                answers_artifact_sha256=self.file_sha256(answer_path),
            )
            judgment_path = root / "duplicate.jsonl"
            self.write_jsonl(judgment_path, [record, record])

            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                json_out=root / "summary.json",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate judgment_id", result.stderr)

    def test_judgment_error_is_rejected_even_when_score_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answer = self.answer_record("q1")
            self.write_jsonl(answer_path, [answer])
            judgment_path = root / "errored.jsonl"
            self.write_jsonl(
                judgment_path,
                [
                    self.judgment_record(
                        answer,
                        run_id="judge-1",
                        score=2,
                        gfc=True,
                        answers_artifact_sha256=self.file_sha256(answer_path),
                        error="timeout",
                    )
                ],
            )

            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                json_out=root / "summary.json",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has error: timeout", result.stderr)

    def test_duplicate_judge_run_id_across_repeat_files_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answer = self.answer_record("q1")
            self.write_jsonl(answer_path, [answer])
            artifact_sha = self.file_sha256(answer_path)
            paths = [root / "judge-a.jsonl", root / "judge-b.jsonl"]
            for path in paths:
                self.write_jsonl(
                    path,
                    [
                        self.judgment_record(
                            answer,
                            run_id="same-run",
                            score=2,
                            gfc=True,
                            answers_artifact_sha256=artifact_sha,
                        )
                    ],
                )

            result = self.run_cli(
                answers=answer_path,
                judgments=paths,
                json_out=root / "summary.json",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate judge_run_id", result.stderr)

    def test_even_gfc_split_has_no_question_level_majority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answer = self.answer_record("q1")
            self.write_jsonl(answer_path, [answer])
            artifact_sha = self.file_sha256(answer_path)
            paths: list[Path] = []
            for run_id, score, gfc in (
                ("judge-1", 2, True),
                ("judge-2", 1, False),
            ):
                path = root / f"{run_id}.jsonl"
                self.write_jsonl(
                    path,
                    [
                        self.judgment_record(
                            answer,
                            run_id=run_id,
                            score=score,
                            gfc=gfc,
                            answers_artifact_sha256=artifact_sha,
                        )
                    ],
                )
                paths.append(path)

            output = root / "summary.json"
            result = self.run_cli(
                answers=answer_path,
                judgments=paths,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(payload["cases"][0]["gfc_majority"])
            self.assertEqual(
                payload["summary"]["question_level_gfc_no_majority_count"], 1
            )
            self.assertEqual(
                payload["summary"]["question_level_majority_gfc_rate"], 0.0
            )
            self.assertEqual(payload["summary"]["gfc_agreement_counts"], {"1/2": 1})

    def test_hash_tampering_and_missing_gfc_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answer = self.answer_record("q1")
            tampered = dict(answer, answer="changed after hashing")
            self.write_jsonl(answer_path, [tampered])
            judgment_path = root / "judge.jsonl"
            self.write_jsonl(
                judgment_path,
                [
                    self.judgment_record(
                        answer,
                        run_id="judge-1",
                        score=2,
                        gfc=True,
                        answers_artifact_sha256=self.file_sha256(answer_path),
                    )
                ],
            )
            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                json_out=root / "tampered-summary.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("answer_sha256 mismatch", result.stderr)

            self.write_jsonl(answer_path, [answer])
            artifact_sha = self.file_sha256(answer_path)
            missing_gfc = self.judgment_record(
                answer,
                run_id="judge-2",
                score=1,
                gfc=None,
                answers_artifact_sha256=artifact_sha,
            )
            self.write_jsonl(judgment_path, [missing_gfc])
            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                json_out=root / "missing-gfc-summary.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing grounded_fully_correct", result.stderr)

    def test_partial_repeat_requires_and_honors_bound_selection_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answers = [self.answer_record(case_id) for case_id in ("q1", "q2", "q3")]
            self.write_jsonl(answer_path, answers)
            artifact_sha = self.file_sha256(answer_path)
            selected = [answers[0], answers[2]]
            selection_path = root / "selection.json"
            self.write_selection(selection_path, answer_path, selected)
            selection_binding = self.selection_binding(selection_path)

            judgment_paths: list[Path] = []
            for run_id, values in (
                ("judge-2", [(2, True), (0, False)]),
                ("judge-3", [(2, True), (2, True)]),
            ):
                path = root / f"{run_id}.jsonl"
                self.write_jsonl(
                    path,
                    [
                        self.judgment_record(
                            answer,
                            run_id=run_id,
                            score=score,
                            gfc=gfc,
                            answers_artifact_sha256=artifact_sha,
                            selection_binding=selection_binding,
                        )
                        for answer, (score, gfc) in zip(selected, values)
                    ],
                )
                judgment_paths.append(path)

            output = root / "summary.json"
            result = self.run_cli(
                answers=answer_path,
                judgments=judgment_paths,
                selection_manifest=selection_path,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["question_count"], 2)
            self.assertEqual([row["case_id"] for row in payload["cases"]], ["q1", "q3"])
            self.assertEqual(
                payload["inputs"]["selection"]["selected_answer_ids_sha256"],
                artifacts.sha256_json([row["answer_id"] for row in selected]),
            )
            self.assertEqual(
                payload["methodology"]["full_answer_artifact_question_count"], 3
            )

            without_selection = self.run_cli(
                answers=answer_path,
                judgments=judgment_paths,
                json_out=root / "unsafe-summary.json",
            )
            self.assertNotEqual(without_selection.returncode, 0)
            self.assertIn("selection-bound judgment requires", without_selection.stderr)

    def test_final_partial_repeat_is_run1_core_and_exactly_two_judge_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answers = self.final_answer_records()
            self.write_jsonl(answer_path, answers)
            selected = answers[:9]
            selection_path = root / "selection.json"
            self.write_selection(selection_path, answer_path, selected)
            selection_binding = self.selection_binding(selection_path)
            artifact_sha = self.file_sha256(answer_path)
            judgment_paths: list[Path] = []
            for run_id in ("judge-v8-r2", "judge-v8-r3"):
                path = root / f"{run_id}.jsonl"
                self.write_jsonl(
                    path,
                    [
                        self.judgment_record(
                            answer,
                            run_id=run_id,
                            score=2,
                            gfc=True,
                            answers_artifact_sha256=artifact_sha,
                            selection_binding=selection_binding,
                        )
                        for answer in selected
                    ],
                )
                judgment_paths.append(path)

            incomplete = self.run_cli(
                answers=answer_path,
                judgments=judgment_paths[:1],
                selection_manifest=selection_path,
                json_out=root / "one-repeat.json",
            )
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("exactly 2 additional", incomplete.stderr)

            complete = self.run_cli(
                answers=answer_path,
                judgments=judgment_paths,
                selection_manifest=selection_path,
                json_out=root / "two-repeats.json",
            )
            self.assertEqual(complete.returncode, 0, complete.stderr)
            payload = json.loads(
                (root / "two-repeats.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["methodology"]["final_protocol_repeat_judge_run_count"],
                2,
            )

            run2_path = root / "run2-answers.jsonl"
            run2_answers = self.final_answer_records(generation_run_id="run2")
            self.write_jsonl(run2_path, run2_answers)
            run2_selection = root / "run2-selection.json"
            self.write_selection(run2_selection, run2_path, run2_answers[:9])
            rejected = self.run_cli(
                answers=run2_path,
                judgments=judgment_paths,
                selection_manifest=run2_selection,
                json_out=root / "run2-summary.json",
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("must use generation run1", rejected.stderr)

    def test_partial_selection_hash_order_and_completeness_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answers = [self.answer_record(case_id) for case_id in ("q1", "q2")]
            self.write_jsonl(answer_path, answers)
            artifact_sha = self.file_sha256(answer_path)
            judgment_path = root / "judge.jsonl"
            self.write_jsonl(
                judgment_path,
                [
                    self.judgment_record(
                        answers[0],
                        run_id="judge-2",
                        score=2,
                        gfc=True,
                        answers_artifact_sha256=artifact_sha,
                    )
                ],
            )
            selection_path = root / "selection.json"
            self.write_selection(selection_path, answer_path, [answers[0]])
            selection_binding = self.selection_binding(selection_path)
            first = json.loads(judgment_path.read_text(encoding="utf-8"))
            first["judge_repeat_selection"] = selection_binding
            first = artifacts.build_judgment_identity(first)
            self.write_jsonl(judgment_path, [first])
            payload = json.loads(selection_path.read_text(encoding="utf-8"))
            payload["selected_answer_ids_sha256"] = "0" * 64
            selection_path.write_text(json.dumps(payload), encoding="utf-8")

            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                selection_manifest=selection_path,
                json_out=root / "bad-hash.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("selected_answer_ids_sha256 mismatch", result.stderr)

            self.write_selection(selection_path, answer_path, [answers[0]])
            selection_binding = self.selection_binding(selection_path)
            first = json.loads(judgment_path.read_text(encoding="utf-8"))
            first["judge_repeat_selection"] = selection_binding
            first = artifacts.build_judgment_identity(first)
            self.write_jsonl(judgment_path, [first])
            extra = self.judgment_record(
                answers[1],
                run_id="judge-2",
                score=1,
                gfc=False,
                answers_artifact_sha256=artifact_sha,
                selection_binding=selection_binding,
            )
            records = [json.loads(line) for line in judgment_path.read_text().splitlines()]
            self.write_jsonl(judgment_path, [*records, extra])
            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                selection_manifest=selection_path,
                json_out=root / "extra-judgment.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown answer_id", result.stderr)

    def test_cli_outputs_are_immutable_and_inputs_cannot_be_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer_path = root / "answers.jsonl"
            answer = self.answer_record("q1")
            self.write_jsonl(answer_path, [answer])
            artifact_sha = self.file_sha256(answer_path)
            judgment_path = root / "judge.jsonl"
            self.write_jsonl(
                judgment_path,
                [
                    self.judgment_record(
                        answer,
                        run_id="judge-1",
                        score=2,
                        gfc=True,
                        answers_artifact_sha256=artifact_sha,
                    )
                ],
            )
            existing = root / "existing.json"
            existing.write_text("owner data\n", encoding="utf-8")

            result = self.run_cli(
                answers=answer_path,
                judgments=[judgment_path],
                json_out=existing,
                csv_out=root / "must-not-appear.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(existing.read_text(encoding="utf-8"), "owner data\n")
            self.assertFalse((root / "must-not-appear.csv").exists())

            answer_link = root / "answers-link.jsonl"
            answer_link.symlink_to(answer_path)
            result = self.run_cli(
                answers=answer_link,
                judgments=[judgment_path],
                json_out=root / "linked-input.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not be symlinks", result.stderr)
            self.assertFalse((root / "linked-input.json").exists())


if __name__ == "__main__":
    unittest.main()
