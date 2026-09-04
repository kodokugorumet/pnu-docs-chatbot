from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_final_generation_gfc.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


artifacts = load_module(
    "service_eval_artifacts_for_final_gfc_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
calibration = load_module(
    "human_calibration_for_final_gfc_test",
    ROOT / "scripts" / "analyze_judge_human_calibration.py",
)


class AnalyzeFinalGenerationGfcTests(unittest.TestCase):
    maxDiff = None

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def setUpFixture(
        self,
        root: Path,
        *,
        calibration_passes: bool,
        terminal_slots: set[tuple[str, str, str]] | None = None,
        empty_terminal_slots: set[tuple[str, str, str]] | None = None,
    ) -> dict:
        root.mkdir(parents=True, exist_ok=True)
        terminal_slots = terminal_slots or set()
        empty_terminal_slots = empty_terminal_slots or set()
        terminal_slots = terminal_slots | empty_terminal_slots
        cases = [
            {
                "id": f"q{index}",
                "split": "holdout-core",
                "family_id": f"family-{index}",
                "query": f"question {index}",
            }
            for index in range(1, 5)
        ]
        cases.append(
            {
                "id": "challenge-1",
                "split": "holdout-challenge",
                "family_id": "challenge-family",
                "query": "challenge",
            }
        )
        cases_path = root / "cases.jsonl"
        self.write_jsonl(cases_path, cases)
        core = cases[:4]
        core_ids = [case["id"] for case in core]
        cases_sha = artifacts.sha256_text(cases_path.read_text(encoding="utf-8"))
        cases_canonical_sha = artifacts.sha256_json(cases)
        selected_sha = artifacts.sha256_json(core_ids)

        gfc_matrix = {
            "c0": {
                "run1": [False, False, True, True],
                "run2": [False, True, True, False],
                "run3": [False, False, True, False],
            },
            "c1": {
                "run1": [False, True, True, True],
                "run2": [True, True, True, False],
                "run3": [False, True, True, True],
            },
        }
        answer_paths: dict[str, list[Path]] = {"c0": [], "c1": []}
        judgment_paths: dict[str, list[Path]] = {"c0": [], "c1": []}
        run1_answers: list[dict] = []
        run1_judgments: list[dict] = []
        judge_config_base = {"provider": "fixture", "model": "judge-v8"}
        judge_config_sha = artifacts.sha256_json(judge_config_base)
        judge_config = {
            **judge_config_base,
            "judge_config_sha256": judge_config_sha,
        }
        for condition in ("c0", "c1"):
            for run_id in ("run1", "run2", "run3"):
                collector = {
                    "cases_sha256": cases_sha,
                    "cases_canonical_sha256": cases_canonical_sha,
                    "selected_case_ids_sha256": selected_sha,
                    "provider": "frontier",
                    "model": "gemini-fixture",
                    "expected_generation_max_output_tokens": 900,
                    "max_attempts": 3,
                }
                answers: list[dict] = []
                for case in core:
                    terminal = (condition, run_id, case["id"]) in terminal_slots
                    record = {
                        "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                        "record_type": "answer",
                        "experiment_id": "final-gfc-fixture",
                        "condition_id": condition,
                        "generation_run_id": run_id,
                        "case_id": case["id"],
                        "id": case["id"],
                        "case_sha256": artifacts.sha256_json(case),
                        "collector_config": collector,
                        "collector_config_sha256": artifacts.sha256_json(collector),
                        "slot_outcome": (
                            "service_error" if terminal else "answer"
                        ),
                        "answer_eligible_for_judge": not terminal,
                        "answer": (
                            "[SERVICE_ERROR]"
                            if terminal
                            else f"{condition} {run_id} {case['id']} answer"
                        ),
                        "error": None,
                    }
                    if terminal:
                        if (condition, run_id, case["id"]) in empty_terminal_slots:
                            attempts = [
                                {
                                    "attempt_number": 1,
                                    "status": "ok",
                                    "elapsed_ms": 1.0,
                                }
                            ]
                            record["service_error"] = {
                                "stage": "response_validation",
                                "type": "AnswerPayloadError",
                                "message": (
                                    "response answer must be a non-empty string"
                                ),
                                "retryable": False,
                                "http_status": None,
                                "request_attempts": attempts,
                            }
                        else:
                            attempts = [
                                {
                                    "attempt_number": attempt,
                                    "status": "error",
                                    "elapsed_ms": 1.0,
                                    "error_type": "TimeoutError",
                                    "error": "retry budget exhausted",
                                    "retryable": True,
                                    "http_status": None,
                                }
                                for attempt in (1, 2, 3)
                            ]
                            record["service_error"] = {
                                "stage": "chat_transport",
                                "type": "TimeoutError",
                                "message": "retry budget exhausted",
                                "retryable": True,
                                "http_status": None,
                                "request_attempts": attempts,
                            }
                        record["request_attempts"] = attempts
                        record["collection_attempt_number"] = 1
                    else:
                        record["request_attempts"] = [
                            {
                                "attempt_number": 1,
                                "status": "ok",
                                "elapsed_ms": 1.0,
                            }
                        ]
                        record["collection_attempt_number"] = 1
                        user_prompt = (
                            f"fixture prompt {condition} {run_id} {case['id']}"
                        )
                        system_instruction = "fixture system instruction"
                        prompt_sha = artifacts.sha256_text(user_prompt)
                        system_sha = artifacts.sha256_text(system_instruction)
                        request_config = {
                            "provider": "gemini",
                            "model_requested": "gemini-fixture",
                            "api_style": "generateContent",
                            "generation_config": {"maxOutputTokens": 900},
                            "prompt_used": True,
                            "system_instruction_sha256": system_sha,
                        }
                        record["generation"] = {
                            "requested": "frontier",
                            "used": "frontier",
                            "model": "gemini-fixture",
                            "request_config": request_config,
                            "request_config_sha256": artifacts.sha256_json(
                                request_config
                            ),
                            "prompt_sha256": prompt_sha,
                            "system_instruction_sha256": system_sha,
                        }
                        record["evaluation_trace"] = {
                            "schema_version": 1,
                            "generation_input": {
                                "user_prompt": user_prompt,
                                "system_instruction": system_instruction,
                                "prompt_sha256": prompt_sha,
                                "system_instruction_sha256": system_sha,
                            },
                            "raw_draft": record["answer"],
                            "sanitized_draft": record["answer"],
                            "retrieval_stages": {"final_contexts": []},
                        }
                    answers.append(artifacts.build_answer_identity(record))
                answer_path = root / f"{condition}-{run_id}.answers.jsonl"
                self.write_jsonl(answer_path, answers)
                answer_artifact_sha = artifacts.sha256_text(
                    answer_path.read_text(encoding="utf-8")
                )
                judgments = []
                for answer, gfc in zip(answers, gfc_matrix[condition][run_id]):
                    if answer["slot_outcome"] == "service_error":
                        continue
                    judgment = artifacts.build_judgment_identity(
                        {
                            "schema_version": artifacts.JUDGMENT_SCHEMA_VERSION,
                            "record_type": "judgment",
                            "experiment_id": answer["experiment_id"],
                            "condition_id": condition,
                            "generation_run_id": run_id,
                            "judge_run_id": f"judge-v8-{condition}-{run_id}-r1",
                            "case_id": answer["case_id"],
                            "answer_id": answer["answer_id"],
                            "answer_sha256": answer["answer_sha256"],
                            "answer_record_sha256": artifacts.sha256_json(answer),
                            "answers_artifact_sha256": answer_artifact_sha,
                            "judge_config": judge_config,
                            "judge_config_sha256": judge_config_sha,
                            "judge": {
                                "score": 2 if gfc else 1,
                                "grounded_fully_correct": gfc,
                            },
                            "error": None,
                        }
                    )
                    judgments.append(judgment)
                judgment_path = root / f"{condition}-{run_id}.judgments.jsonl"
                self.write_jsonl(judgment_path, judgments)
                answer_paths[condition].append(answer_path)
                judgment_paths[condition].append(judgment_path)
                if run_id == "run1":
                    run1_answers.extend(answers)
                    run1_judgments.extend(judgments)

        labels: list[dict] = []
        run1_judgment_by_answer = {
            row["answer_id"]: row for row in run1_judgments
        }
        for answer in run1_answers:
            if answer["slot_outcome"] == "service_error":
                continue
            judge_gfc = bool(
                run1_judgment_by_answer[answer["answer_id"]]["judge"][
                    "grounded_fully_correct"
                ]
            )
            human_gfc = judge_gfc if calibration_passes else not judge_gfc
            score = 2 if human_gfc else 1
            for reviewer_id, kind in (
                ("human-a", "independent"),
                ("human-b", "independent"),
                ("consensus", "adjudicated"),
            ):
                labels.append(
                    {
                        "schema_version": "pnu.human-answer-label.v1",
                        "record_type": "human_label",
                        "answer_id": answer["answer_id"],
                        "case_id": answer["case_id"],
                        "blind_item_id": f"blind-{answer['answer_id']}",
                        "reviewer_id": reviewer_id,
                        "label_kind": kind,
                        "score": score,
                        "grounded_fully_correct": human_gfc,
                        "uncertain": False,
                        "split": "holdout-core",
                        "notes": None,
                    }
                )
        labels_path = root / "human-labels.jsonl"
        self.write_jsonl(labels_path, labels)
        calibration_payload, _ = calibration.analyze_calibration(
            judgment_paths=[judgment_paths["c0"][0], judgment_paths["c1"][0]],
            human_label_paths=[labels_path],
        )
        calibration_path = root / "calibration.json"
        calibration_path.write_text(
            json.dumps(calibration_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.assertIs(
            calibration_payload["summary"]["protocol_gate"]["passed"],
            calibration_passes,
        )
        return {
            "cases": cases_path,
            "calibration": calibration_path,
            "answers": answer_paths,
            "judgments": judgment_paths,
            "human_labels": labels_path,
        }

    def run_cli(
        self,
        fixture: dict,
        *,
        json_out: Path,
        csv_out: Path,
        include_humans: bool = False,
        answer_override: dict[str, list[Path]] | None = None,
        judgment_override: dict[str, list[Path]] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        answer_paths = answer_override or fixture["answers"]
        judgment_paths = judgment_override or fixture["judgments"]
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--cases",
            str(fixture["cases"]),
            "--calibration",
            str(fixture["calibration"]),
            "--c0-answers",
            *(str(path) for path in answer_paths["c0"]),
            "--c0-judgments",
            *(str(path) for path in judgment_paths["c0"]),
            "--c1-answers",
            *(str(path) for path in answer_paths["c1"]),
            "--c1-judgments",
            *(str(path) for path in judgment_paths["c1"]),
        ]
        if include_humans:
            command.extend(["--human-labels", str(fixture["human_labels"])])
        command.extend(
            [
                "--json-out",
                str(json_out),
                "--csv-out",
                str(csv_out),
                "--bootstrap",
                "250",
                "--sign-flip-iterations",
                "500",
                "--seed",
                "41",
                "--expected-core-count",
                "4",
            ]
        )
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_gate_pass_uses_three_generation_runs_and_reports_run_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.setUpFixture(root, calibration_passes=True)
            output = root / "analysis.json"
            csv_out = root / "analysis.csv"
            result = self.run_cli(fixture, json_out=output, csv_out=csv_out)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["calibration_gate_passed"])
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
            self.assertEqual(
                companion["sha256"], hashlib.sha256(csv_out.read_bytes()).hexdigest()
            )
            self.assertEqual(companion["bytes"], len(csv_out.read_bytes()))
            self.assertEqual(
                payload["headline"]["source"],
                "llm_judge_three_independent_generation_runs",
            )
            self.assertEqual(payload["headline"]["effective_sample_n"], 4)
            self.assertEqual(
                payload["judge_three_generation_runs"]["total_logical_generation_slots"],
                24,
            )
            self.assertEqual(
                payload["judge_three_generation_runs"]["total_definitive_judgments"],
                24,
            )
            self.assertEqual(
                payload["judge_three_generation_runs"][
                    "total_terminal_service_error_count"
                ],
                0,
            )
            run2 = payload["judge_three_generation_runs"]["by_generation_run"]["run2"]
            self.assertEqual(run2["c0"]["gfc_count"], 2)
            self.assertEqual(run2["c1"]["gfc_count"], 3)
            self.assertEqual(run2["c0"]["service_error_count"], 0)
            self.assertEqual(run2["paired"]["gained_case_ids"], ["q1"])
            self.assertEqual(len(payload["cases"]), 4)
            self.assertEqual(payload["cases"][0]["c0"]["gfc_success_count"], 0)
            self.assertTrue(csv_out.is_file())

            output2 = root / "analysis-2.json"
            csv2 = root / "analysis-2.csv"
            result2 = self.run_cli(fixture, json_out=output2, csv_out=csv2)
            self.assertEqual(result2.returncode, 0, result2.stderr)
            payload2 = json.loads(output2.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["headline"]["paired_family_bootstrap_95ci"],
                payload2["headline"]["paired_family_bootstrap_95ci"],
            )
            self.assertEqual(
                payload["headline"]["paired_sign_flip_two_sided"],
                payload2["headline"]["paired_sign_flip_two_sided"],
            )

    def test_gate_fail_requires_and_uses_adjudicated_run1_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.setUpFixture(root, calibration_passes=False)
            rejected = self.run_cli(
                fixture,
                json_out=root / "rejected.json",
                csv_out=root / "rejected.csv",
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--human-labels", rejected.stderr)
            self.assertFalse((root / "rejected.json").exists())

            output = root / "fallback.json"
            csv_out = root / "fallback.csv"
            accepted = self.run_cli(
                fixture,
                json_out=output,
                csv_out=csv_out,
                include_humans=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["calibration_gate_passed"])
            self.assertEqual(
                payload["headline"]["source"],
                "adjudicated_human_generation_run1_fallback",
            )
            self.assertEqual(
                payload["human_run1_fallback"][
                    "terminal_service_error_case_ids_by_condition"
                ],
                {"c0": [], "c1": []},
            )
            self.assertIs(
                payload["human_run1_fallback"][
                    "terminal_service_error_automatic_gfc_value"
                ],
                False,
            )
            self.assertEqual(
                payload["judge_three_generation_runs"]["reporting_status"],
                "exploratory",
            )
            self.assertEqual(payload["human_run1_fallback"]["generation_run_id"], "run1")
            self.assertEqual(payload["headline"]["effective_sample_n"], 4)

    def test_duplicate_generation_artifact_and_output_alias_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.setUpFixture(root, calibration_passes=True)
            duplicate = {
                "c0": [
                    fixture["answers"]["c0"][0],
                    fixture["answers"]["c0"][0],
                    fixture["answers"]["c0"][2],
                ],
                "c1": fixture["answers"]["c1"],
            }
            result = self.run_cli(
                fixture,
                json_out=root / "duplicate.json",
                csv_out=root / "duplicate.csv",
                answer_override=duplicate,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("same input path", result.stderr)

            alias = self.run_cli(
                fixture,
                json_out=fixture["answers"]["c0"][0],
                csv_out=root / "alias.csv",
            )
            self.assertNotEqual(alias.returncode, 0)
            self.assertIn("must not overwrite", alias.stderr)

    def test_existing_and_symlink_outputs_are_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.setUpFixture(root, calibration_passes=True)
            output = root / "existing.json"
            output.write_text("owner data\n", encoding="utf-8")
            csv_out = root / "must-not-appear.csv"

            result = self.run_cli(
                fixture,
                json_out=output,
                csv_out=csv_out,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "owner data\n")
            self.assertFalse(csv_out.exists())

            broken_target = root / "missing.json"
            symlink_output = root / "symlink.json"
            symlink_output.symlink_to(broken_target)
            result = self.run_cli(
                fixture,
                json_out=symlink_output,
                csv_out=root / "symlink.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertTrue(symlink_output.is_symlink())

    def test_symlink_input_and_invalid_generation_trace_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.setUpFixture(root, calibration_passes=True)
            cases_link = root / "cases-link.jsonl"
            cases_link.symlink_to(fixture["cases"])
            fixture_with_link = {**fixture, "cases": cases_link}
            result = self.run_cli(
                fixture_with_link,
                json_out=root / "link.json",
                csv_out=root / "link.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not be symlinks", result.stderr)

            fixture = self.setUpFixture(
                root / "missing-trace", calibration_passes=True
            )
            answer_path = fixture["answers"]["c0"][1]
            answers = [
                json.loads(line) for line in answer_path.read_text().splitlines()
            ]
            answers[0].pop("evaluation_trace")
            answers[0] = artifacts.build_answer_identity(answers[0])
            self.write_jsonl(answer_path, answers)
            result = self.run_cli(
                fixture,
                json_out=root / "missing-trace.json",
                csv_out=root / "missing-trace.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("omitted evaluation trace", result.stderr)

            fixture = self.setUpFixture(
                root / "forged-trace", calibration_passes=True
            )
            answer_path = fixture["answers"]["c1"][2]
            answers = [
                json.loads(line) for line in answer_path.read_text().splitlines()
            ]
            answers[0]["evaluation_trace"]["generation_input"][
                "user_prompt"
            ] = "forged prompt content"
            answers[0] = artifacts.build_answer_identity(answers[0])
            self.write_jsonl(answer_path, answers)
            result = self.run_cli(
                fixture,
                json_out=root / "forged-trace.json",
                csv_out=root / "forged-trace.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt content SHA mismatch", result.stderr)

            fixture = self.setUpFixture(
                root / "forged-attempts", calibration_passes=True
            )
            answer_path = fixture["answers"]["c0"][2]
            answers = [
                json.loads(line) for line in answer_path.read_text().splitlines()
            ]
            answers[0]["request_attempts"] = [
                {
                    "attempt_number": index,
                    "status": "ok",
                    "elapsed_ms": 1.0,
                }
                for index in range(1, 5)
            ]
            answers[0]["collection_attempt_number"] = 99
            answers[0] = artifacts.build_answer_identity(answers[0])
            self.write_jsonl(answer_path, answers)
            result = self.run_cli(
                fixture,
                json_out=root / "forged-attempts.json",
                csv_out=root / "forged-attempts.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("preserve 1..3 request_attempts", result.stderr)

    def test_terminal_service_errors_are_gfc_zero_and_force_human_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.setUpFixture(
                root,
                calibration_passes=True,
                terminal_slots={
                    ("c0", "run1", "q1"),
                },
                empty_terminal_slots={("c1", "run2", "q4")},
            )
            output = root / "terminal-analysis.json"
            result = self.run_cli(
                fixture,
                json_out=output,
                csv_out=root / "terminal-analysis.csv",
                include_humans=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["calibration_threshold_gate_passed"])
            self.assertFalse(payload["required_run1_complete"])
            self.assertFalse(payload["calibration_gate_passed"])
            self.assertEqual(
                payload["headline"]["source"],
                "adjudicated_human_generation_run1_fallback",
            )
            analysis = payload["judge_three_generation_runs"]
            self.assertEqual(analysis["total_logical_generation_slots"], 24)
            self.assertEqual(analysis["total_successful_answers"], 22)
            self.assertEqual(analysis["total_definitive_judgments"], 22)
            self.assertEqual(analysis["total_terminal_service_error_count"], 2)
            self.assertEqual(
                analysis["by_generation_run"]["run1"]["c0"][
                    "service_error_case_ids"
                ],
                ["q1"],
            )
            q1 = payload["cases"][0]
            self.assertIsNone(q1["c0"]["judge_scores_by_run"]["run1"])
            self.assertFalse(q1["c0"]["gfc_by_run"]["run1"])
            self.assertTrue(q1["c0"]["service_errors_by_run"]["run1"])

            terminal_answer = json.loads(
                fixture["answers"]["c0"][0].read_text().splitlines()[0]
            )
            config = {"provider": "fixture", "model": "judge-v8"}
            invalid_judgment = artifacts.build_judgment_identity(
                {
                    "schema_version": artifacts.JUDGMENT_SCHEMA_VERSION,
                    "record_type": "judgment",
                    "experiment_id": terminal_answer["experiment_id"],
                    "condition_id": "c0",
                    "generation_run_id": "run1",
                    "judge_run_id": "invalid-terminal-judge",
                    "case_id": "q1",
                    "answer_id": terminal_answer["answer_id"],
                    "answer_sha256": terminal_answer["answer_sha256"],
                    "answer_record_sha256": artifacts.sha256_json(terminal_answer),
                    "answers_artifact_sha256": artifacts.sha256_text(
                        fixture["answers"]["c0"][0].read_text()
                    ),
                    "judge_config": config,
                    "judge_config_sha256": artifacts.sha256_json(config),
                    "judge": {"score": 0, "grounded_fully_correct": False},
                    "error": None,
                }
            )
            judgment_path = fixture["judgments"]["c0"][0]
            existing = [
                json.loads(line) for line in judgment_path.read_text().splitlines()
            ]
            self.write_jsonl(judgment_path, [invalid_judgment, *existing])
            rejected = self.run_cli(
                fixture,
                json_out=root / "judged-terminal.json",
                csv_out=root / "judged-terminal.csv",
                include_humans=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("terminal service-error slot(s) were judged", rejected.stderr)

    def test_case_hash_missing_judgment_and_error_judgment_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.setUpFixture(root, calibration_passes=True)

            answer_path = fixture["answers"]["c0"][1]
            answers = [json.loads(line) for line in answer_path.read_text().splitlines()]
            answers[0]["case_sha256"] = "0" * 64
            answers[0]["record_sha256"] = artifacts.record_sha256(answers[0])
            self.write_jsonl(answer_path, answers)
            result = self.run_cli(
                fixture,
                json_out=root / "case-hash.json",
                csv_out=root / "case-hash.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("case_sha256 mismatch", result.stderr)

            fixture = self.setUpFixture(root / "missing", calibration_passes=True)
            judgment_path = fixture["judgments"]["c1"][2]
            judgments = [json.loads(line) for line in judgment_path.read_text().splitlines()]
            self.write_jsonl(judgment_path, judgments[:-1])
            result = self.run_cli(
                fixture,
                json_out=root / "missing.json",
                csv_out=root / "missing.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly cover successful Core", result.stderr)

            fixture = self.setUpFixture(root / "error", calibration_passes=True)
            judgment_path = fixture["judgments"]["c1"][2]
            judgments = [json.loads(line) for line in judgment_path.read_text().splitlines()]
            judgments[0]["error"] = "timeout"
            judgments[0]["record_sha256"] = artifacts.record_sha256(judgments[0])
            self.write_jsonl(judgment_path, judgments)
            result = self.run_cli(
                fixture,
                json_out=root / "error.json",
                csv_out=root / "error.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has error: timeout", result.stderr)

            fixture = self.setUpFixture(root / "repeat", calibration_passes=True)
            judgment_path = fixture["judgments"]["c1"][2]
            judgments = [
                json.loads(line) for line in judgment_path.read_text().splitlines()
            ]
            judgments[0]["judge_repeat_selection"] = {
                "selection_id": "forbidden-stability-repeat"
            }
            judgments[0] = artifacts.build_judgment_identity(judgments[0])
            self.write_jsonl(judgment_path, judgments)
            result = self.run_cli(
                fixture,
                json_out=root / "repeat.json",
                csv_out=root / "repeat.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stability-repeat judgment", result.stderr)

            fixture = self.setUpFixture(root / "config", calibration_passes=True)
            answer_path = fixture["answers"]["c0"][1]
            answers = [
                json.loads(line) for line in answer_path.read_text().splitlines()
            ]
            answers[0]["collector_config"] = {
                **answers[0]["collector_config"],
                "unexpected_drift": True,
            }
            answers[0]["collector_config_sha256"] = artifacts.sha256_json(
                answers[0]["collector_config"]
            )
            answers[0] = artifacts.build_answer_identity(answers[0])
            self.write_jsonl(answer_path, answers)
            result = self.run_cli(
                fixture,
                json_out=root / "config.json",
                csv_out=root / "config.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mixes collector configs", result.stderr)


if __name__ == "__main__":
    unittest.main()
