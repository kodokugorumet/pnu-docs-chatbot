from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_answer_review_packet.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review_packet = load_module("build_answer_review_packet_for_test", SCRIPT)
artifacts = load_module(
    "service_eval_artifacts_for_answer_review_packet_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
calibration = load_module(
    "analyze_judge_human_calibration_for_answer_review_packet_test",
    ROOT / "scripts" / "analyze_judge_human_calibration.py",
)


class BuildAnswerReviewPacketTests(unittest.TestCase):
    maxDiff = None

    def cases(self) -> list[dict]:
        return [
            {
                "id": "q-core-1",
                "split": "holdout-core",
                "query": "핵심 질문 1",
                "role": "pnu-student",
                "answerable": True,
                "expected_behavior": "answer",
                "required_claims": [
                    {
                        "claim_id": "claim-1",
                        "description": "필수 사실",
                        "critical_values": ["값 1"],
                        "evidence_options": [
                            {
                                "document_id": "doc-gold-1",
                                "source_title": "공식 문서 1",
                                "quote": "값 1을 뒷받침하는 원문",
                            }
                        ],
                    }
                ],
            },
            {
                "id": "q-challenge-1",
                "split": "holdout-challenge",
                "query": "답할 수 없는 개인 질문",
                "role": None,
                "answerable": False,
                "expected_behavior": "abstain",
                "challenge_type": "unanswerable",
                "challenge_oracle": {
                    "must_do": ["근거가 없다고 설명"],
                    "must_not_do": ["개인 결과를 추측"],
                },
                "required_claims": [],
            },
            {
                "id": "q-core-2",
                "split": "holdout-core",
                "query": "핵심 질문 2",
                "role": "pnu-student",
                "answerable": True,
                "expected_behavior": "answer",
                "required_claims": [],
            },
        ]

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def write_cases(self, root: Path, cases: list[dict]) -> Path:
        path = root / "cases.jsonl"
        self.write_jsonl(path, cases)
        return path

    def answer_records(
        self,
        cases: list[dict],
        *,
        selected_ids: list[str],
        condition_id: str,
        run_id: str = "run-1",
        experiment_id: str = "experiment-fixture",
        declared_selected_ids: list[str] | None = None,
        declared_cases_sha256: str | None = None,
        error: object | None = None,
        empty_answer: bool = False,
        explicit_slots: bool = False,
        terminal_ids: set[str] | None = None,
    ) -> list[dict]:
        by_id = {case["id"]: case for case in cases}
        terminal_ids = terminal_ids or set()
        model_secret = f"MODEL_SECRET_{condition_id}"
        provider_secret = f"PROVIDER_SECRET_{condition_id}"
        raw_cases_sha = artifacts.sha256_text(
            "".join(
                json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
                for case in cases
            )
        )
        collector_config = {
            "cases_path": "private/cases.jsonl",
            "cases_sha256": declared_cases_sha256
            or (raw_cases_sha if explicit_slots else artifacts.sha256_json(cases)),
            "selected_case_ids_sha256": artifacts.sha256_json(
                declared_selected_ids
                if declared_selected_ids is not None
                else selected_ids
            ),
            "provider": provider_secret,
            "model": model_secret,
        }
        if explicit_slots:
            collector_config["cases_canonical_sha256"] = artifacts.sha256_json(cases)
            collector_config["max_attempts"] = 3
        records: list[dict] = []
        for case_id in selected_ids:
            case = by_id[case_id]
            terminal = case_id in terminal_ids
            answer_text = (
                "[SERVICE_ERROR]"
                if terminal
                else "" if empty_answer
                else f"{case_id}의 검토 대상 답변"
            )
            record = {
                "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": experiment_id,
                "condition_id": condition_id,
                "generation_run_id": run_id,
                "case_id": case_id,
                "id": case_id,
                "case_sha256": artifacts.sha256_json(case),
                "collector_config": collector_config,
                "collector_config_sha256": artifacts.sha256_json(
                    collector_config
                ),
                "answer": answer_text,
                "cited_answer": f"{answer_text} [1]",
                "generator": f"GENERATOR_SECRET_{condition_id}",
                "generation": {
                    "requested": provider_secret,
                    "used": provider_secret,
                    "model": model_secret,
                    "request_config": {
                        "provider": provider_secret,
                        "model_requested": model_secret,
                    },
                },
                "claims": [
                    {
                        "text": answer_text,
                        "supported": True,
                        "confidence": "AUTO_CONFIDENCE_SECRET",
                        "validation_reason": "AUTO_REASON_SECRET",
                        "citations": [
                            {
                                "source_number": 1,
                                "chunk_id": f"chunk-{case_id}",
                                "source_title": "검색 문서",
                                "excerpt": "검색 근거 원문",
                            }
                        ],
                    }
                ],
                "evaluation_trace": {
                    "retrieval_stages": {
                        "final_contexts": [
                            {
                                "rank": 1,
                                "chunk_id": f"chunk-{case_id}",
                                "source_title": "검색 문서",
                                "section_path": ["절"],
                                "text": "검색 근거 원문",
                            }
                        ]
                    }
                },
                "error": error,
            }
            if explicit_slots:
                record["slot_outcome"] = "service_error" if terminal else "answer"
                record["answer_eligible_for_judge"] = not terminal
                if not terminal:
                    record["request_attempts"] = [
                        {
                            "attempt_number": 1,
                            "status": "ok",
                            "elapsed_ms": 1.0,
                        }
                    ]
                    record["collection_attempt_number"] = 1
            if terminal:
                attempts = [
                    {
                        "attempt_number": 1,
                        "status": "ok",
                        "elapsed_ms": 1.0,
                    }
                ]
                record["request_attempts"] = attempts
                record["collection_attempt_number"] = 1
                record["service_error"] = {
                    "stage": "response_validation",
                    "type": "AnswerPayloadError",
                    "message": "response answer must be a non-empty string",
                    "retryable": False,
                    "http_status": None,
                    "request_attempts": attempts,
                }
            records.append(artifacts.build_answer_identity(record))
        return records

    def write_answers(
        self,
        root: Path,
        name: str,
        records: list[dict],
    ) -> Path:
        path = root / name
        self.write_jsonl(path, records)
        return path

    def output_paths(self, root: Path, prefix: str = "review") -> dict[str, Path]:
        return {
            "packet": root / f"{prefix}.md",
            "mapping": root / f"{prefix}.mapping.json",
            "labels_a": root / f"{prefix}.reviewer-a.jsonl",
            "labels_b": root / f"{prefix}.reviewer-b.jsonl",
            "adjudication": root / f"{prefix}.adjudication.jsonl",
        }

    def run_cli(
        self,
        *,
        cases_path: Path,
        answer_paths: list[Path],
        outputs: dict[str, Path],
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--cases",
            str(cases_path),
        ]
        for path in answer_paths:
            command.extend(["--answers", str(path)])
        command.extend(
            [
                "--out",
                str(outputs["packet"]),
                "--mapping-out",
                str(outputs["mapping"]),
                "--labels-a-out",
                str(outputs["labels_a"]),
                "--labels-b-out",
                str(outputs["labels_b"]),
                "--adjudication-out",
                str(outputs["adjudication"]),
                "--seed",
                "20260902",
            ]
        )
        if check:
            command.append("--check")
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_partial_artifacts_build_blind_deterministic_compatible_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.cases()
            cases_path = self.write_cases(root, cases)
            selected = ["q-core-1", "q-challenge-1"]
            c0 = self.write_answers(
                root,
                "c0.jsonl",
                self.answer_records(cases, selected_ids=selected, condition_id="C0"),
            )
            c1 = self.write_answers(
                root,
                "c1.jsonl",
                self.answer_records(cases, selected_ids=selected, condition_id="C1"),
            )
            outputs = self.output_paths(root)

            result = self.run_cli(
                cases_path=cases_path,
                answer_paths=[c0, c1],
                outputs=outputs,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("4 blind items", result.stdout)
            # Only the selected subset is required; q-core-2 is intentionally absent.
            packet = outputs["packet"].read_text(encoding="utf-8")
            self.assertEqual(packet.count("### 검토 대상 cited answer"), 4)
            self.assertNotIn("q-core-2", packet)

            public_text = packet + "".join(
                outputs[name].read_text(encoding="utf-8")
                for name in ("labels_a", "labels_b", "adjudication")
            )
            for secret in (
                "MODEL_SECRET_C0",
                "MODEL_SECRET_C1",
                "PROVIDER_SECRET_C0",
                "PROVIDER_SECRET_C1",
                "GENERATOR_SECRET_C0",
                "GENERATOR_SECRET_C1",
                "AUTO_CONFIDENCE_SECRET",
                "AUTO_REASON_SECRET",
            ):
                self.assertNotIn(secret, public_text)
            for forbidden_key in (
                '"condition_id"',
                '"experiment_id"',
                '"generation_run_id"',
                '"provider"',
                '"model"',
            ):
                self.assertNotIn(forbidden_key, public_text)

            mapping_text = outputs["mapping"].read_text(encoding="utf-8")
            for secret in (
                "MODEL_SECRET_C0",
                "MODEL_SECRET_C1",
                "PROVIDER_SECRET_C0",
                "PROVIDER_SECRET_C1",
                "GENERATOR_SECRET_C0",
                "GENERATOR_SECRET_C1",
            ):
                self.assertIn(secret, mapping_text)
            mapping = json.loads(mapping_text)
            self.assertEqual(mapping["item_count"], 4)
            self.assertEqual(
                mapping["output_publication"][
                    "authoritative_completion_artifact"
                ],
                "mapping",
            )
            self.assertEqual(
                len(
                    mapping["output_publication"][
                        "required_companion_artifacts"
                    ]
                ),
                4,
            )
            for companion in mapping["output_publication"][
                "required_companion_artifacts"
            ]:
                companion_path = Path(companion["path"])
                self.assertEqual(
                    companion["sha256"],
                    artifacts.sha256_text(
                        companion_path.read_text(encoding="utf-8")
                    ),
                )
                self.assertEqual(companion["bytes"], len(companion_path.read_bytes()))
            self.assertEqual(
                {item["condition_id"] for item in mapping["items"]},
                {"C0", "C1"},
            )

            labels_by_name: dict[str, list[dict]] = {}
            for name in ("labels_a", "labels_b", "adjudication"):
                labels_by_name[name] = [
                    json.loads(line)
                    for line in outputs[name].read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(len(labels_by_name[name]), 4)
            self.assertEqual(
                {row["reviewer_id"] for row in labels_by_name["labels_a"]},
                {"REPLACE_WITH_REVIEWER_A_ID"},
            )
            self.assertEqual(
                {row["reviewer_id"] for row in labels_by_name["labels_b"]},
                {"REPLACE_WITH_REVIEWER_B_ID"},
            )
            self.assertEqual(
                {row["reviewer_id"] for row in labels_by_name["adjudication"]},
                {"REPLACE_WITH_ADJUDICATOR_ID"},
            )
            for name, rows in labels_by_name.items():
                expected_kind = (
                    "adjudicated" if name == "adjudication" else "independent"
                )
                for row in rows:
                    self.assertEqual(
                        row["schema_version"], "pnu.human-answer-label.v1"
                    )
                    self.assertEqual(row["record_type"], "human_label")
                    self.assertEqual(row["label_kind"], expected_kind)
                    self.assertIn(row["split"], {"holdout-core", "holdout-challenge"})
                    if row["split"] == "holdout-core":
                        self.assertNotIn("challenge_type", row)
                    else:
                        self.assertEqual(row["challenge_type"], "unanswerable")

                    completed = dict(row)
                    completed.update(
                        score=2,
                        grounded_fully_correct=True,
                        uncertain=False,
                    )
                    validated = calibration._validate_human_record(
                        completed,
                        path=outputs[name],
                        line_number=1,
                    )
                    self.assertEqual(validated["answer_id"], row["answer_id"])

            forward = review_packet.build_outputs(
                cases_path=cases_path,
                answer_paths=[c0, c1],
                seed=20260902,
            )
            reversed_inputs = review_packet.build_outputs(
                cases_path=cases_path,
                answer_paths=[c1, c0],
                seed=20260902,
            )
            self.assertEqual(forward, reversed_inputs)

    def test_check_passes_then_detects_stale_output_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.cases()
            cases_path = self.write_cases(root, cases)
            answer_path = self.write_answers(
                root,
                "answers.jsonl",
                self.answer_records(
                    cases, selected_ids=["q-core-1"], condition_id="C0"
                ),
            )
            outputs = self.output_paths(root)
            created = self.run_cli(
                cases_path=cases_path,
                answer_paths=[answer_path],
                outputs=outputs,
            )
            self.assertEqual(created.returncode, 0, created.stderr)

            current = self.run_cli(
                cases_path=cases_path,
                answer_paths=[answer_path],
                outputs=outputs,
                check=True,
            )
            self.assertEqual(current.returncode, 0, current.stderr)
            labels_before = outputs["labels_a"].read_bytes()

            outputs["packet"].write_text("stale\n", encoding="utf-8")
            stale = self.run_cli(
                cases_path=cases_path,
                answer_paths=[answer_path],
                outputs=outputs,
                check=True,
            )
            self.assertEqual(stale.returncode, 1)
            self.assertIn("content mismatch", stale.stderr)
            self.assertEqual(outputs["packet"].read_text(encoding="utf-8"), "stale\n")
            self.assertEqual(outputs["labels_a"].read_bytes(), labels_before)

    def test_terminal_service_error_is_private_auto_gfc_zero_not_human_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.cases()
            cases_path = self.write_cases(root, cases)
            answer_path = self.write_answers(
                root,
                "final-run1.answers.jsonl",
                self.answer_records(
                    cases,
                    selected_ids=["q-core-1", "q-core-2"],
                    condition_id="c0",
                    run_id="run1",
                    explicit_slots=True,
                    terminal_ids={"q-core-2"},
                ),
            )
            outputs = self.output_paths(root, "terminal")

            result = self.run_cli(
                cases_path=cases_path,
                answer_paths=[answer_path],
                outputs=outputs,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            packet = outputs["packet"].read_text(encoding="utf-8")
            self.assertEqual(packet.count("### 검토 대상 cited answer"), 1)
            self.assertNotIn("[SERVICE_ERROR]", packet)
            for label_name in ("labels_a", "labels_b", "adjudication"):
                labels = [
                    json.loads(line)
                    for line in outputs[label_name].read_text().splitlines()
                ]
                self.assertEqual(len(labels), 1)
                self.assertEqual(labels[0]["case_id"], "q-core-1")

            mapping = json.loads(outputs["mapping"].read_text(encoding="utf-8"))
            self.assertEqual(mapping["logical_slot_count"], 2)
            self.assertEqual(mapping["item_count"], 1)
            self.assertEqual(mapping["terminal_service_error_count"], 1)
            terminal = mapping["terminal_service_errors"][0]
            self.assertEqual(terminal["case_id"], "q-core-2")
            self.assertIs(terminal["automatic_gfc"], False)
            self.assertIs(terminal["human_label_required"], False)
            self.assertEqual(
                mapping["inputs"]["cases"]["sha256"],
                artifacts.sha256_text(cases_path.read_text(encoding="utf-8")),
            )
            self.assertEqual(
                mapping["inputs"]["cases"]["canonical_sha256"],
                artifacts.sha256_json(cases),
            )

    def test_same_run_can_be_supplied_as_disjoint_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.cases()
            cases_path = self.write_cases(root, cases)
            first = self.write_answers(
                root,
                "first.jsonl",
                self.answer_records(
                    cases,
                    selected_ids=["q-core-1"],
                    condition_id="C1",
                    run_id="same-run",
                ),
            )
            second = self.write_answers(
                root,
                "second.jsonl",
                self.answer_records(
                    cases,
                    selected_ids=["q-challenge-1"],
                    condition_id="C1",
                    run_id="same-run",
                ),
            )

            outputs = review_packet.build_outputs(
                cases_path=cases_path,
                answer_paths=[first, second],
                seed=7,
            )

            self.assertEqual(len(outputs["labels_a"].splitlines()), 2)

    def assert_cli_rejects(
        self,
        *,
        cases: list[dict],
        records_by_file: list[list[dict]],
        expected_error: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = self.write_cases(root, cases)
            answer_paths = [
                self.write_answers(root, f"answers-{index}.jsonl", records)
                for index, records in enumerate(records_by_file, start=1)
            ]
            outputs = self.output_paths(root)
            result = self.run_cli(
                cases_path=cases_path,
                answer_paths=answer_paths,
                outputs=outputs,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(expected_error, result.stderr)
            self.assertFalse(any(path.exists() for path in outputs.values()))

    def test_rejects_declared_selection_with_missing_answer(self) -> None:
        cases = self.cases()
        records = self.answer_records(
            cases,
            selected_ids=["q-core-1"],
            declared_selected_ids=["q-core-1", "q-core-2"],
            condition_id="C0",
        )
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[records],
            expected_error="selected_case_ids_sha256 mismatch",
        )

    def test_rejects_wrong_full_cases_binding(self) -> None:
        cases = self.cases()
        records = self.answer_records(
            cases,
            selected_ids=["q-core-1"],
            declared_cases_sha256="0" * 64,
            condition_id="C0",
        )
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[records],
            expected_error="collector cases_sha256 mismatch",
        )

    def test_rejects_case_hash_mismatch(self) -> None:
        cases = self.cases()
        records = self.answer_records(
            cases, selected_ids=["q-core-1"], condition_id="C0"
        )
        records[0]["case_sha256"] = "0" * 64
        records[0] = artifacts.build_answer_identity(records[0])
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[records],
            expected_error="case_sha256 mismatch",
        )

    def test_rejects_answer_integrity_hash_mismatch(self) -> None:
        cases = self.cases()
        records = self.answer_records(
            cases, selected_ids=["q-core-1"], condition_id="C0"
        )
        records[0]["answer"] = "record hash를 갱신하지 않은 변조"
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[records],
            expected_error="answer_sha256 mismatch",
        )

    def test_rejects_error_bearing_and_empty_answers(self) -> None:
        cases = self.cases()
        error_records = self.answer_records(
            cases,
            selected_ids=["q-core-1"],
            condition_id="C0",
            error={"message": "generation failed"},
        )
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[error_records],
            expected_error="has error",
        )
        empty_records = self.answer_records(
            cases,
            selected_ids=["q-core-1"],
            condition_id="C0",
            empty_answer=True,
        )
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[empty_records],
            expected_error="is empty",
        )

    def test_rejects_duplicate_answers_across_artifacts(self) -> None:
        cases = self.cases()
        records = self.answer_records(
            cases, selected_ids=["q-core-1"], condition_id="C0"
        )
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[records, records],
            expected_error="duplicate answer_id across artifacts",
        )

    def test_rejects_mixed_experiments(self) -> None:
        cases = self.cases()
        first = self.answer_records(
            cases,
            selected_ids=["q-core-1"],
            condition_id="C0",
            experiment_id="experiment-a",
        )
        second = self.answer_records(
            cases,
            selected_ids=["q-core-1"],
            condition_id="C1",
            experiment_id="experiment-b",
        )
        self.assert_cli_rejects(
            cases=cases,
            records_by_file=[first, second],
            expected_error="mix experiment_id",
        )

    def test_rejects_output_that_aliases_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.cases()
            cases_path = self.write_cases(root, cases)
            answer_path = self.write_answers(
                root,
                "answers.jsonl",
                self.answer_records(
                    cases, selected_ids=["q-core-1"], condition_id="C0"
                ),
            )
            outputs = self.output_paths(root)
            outputs["packet"] = answer_path
            original = answer_path.read_bytes()

            result = self.run_cli(
                cases_path=cases_path,
                answer_paths=[answer_path],
                outputs=outputs,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("must not alias", result.stderr)
            self.assertEqual(answer_path.read_bytes(), original)
            self.assertFalse(outputs["mapping"].exists())

    def test_non_check_mode_never_replaces_existing_bundle_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.cases()
            cases_path = self.write_cases(root, cases)
            answer_path = self.write_answers(
                root,
                "answers.jsonl",
                self.answer_records(
                    cases, selected_ids=["q-core-1"], condition_id="C0"
                ),
            )
            outputs = self.output_paths(root)
            outputs["labels_b"].write_text("owner data\n", encoding="utf-8")

            result = self.run_cli(
                cases_path=cases_path,
                answer_paths=[answer_path],
                outputs=outputs,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(
                outputs["labels_b"].read_text(encoding="utf-8"), "owner data\n"
            )
            self.assertFalse(outputs["mapping"].exists())
            self.assertFalse(outputs["packet"].exists())

    def test_symlink_input_is_rejected_before_bundle_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.cases()
            cases_path = self.write_cases(root, cases)
            answer_path = self.write_answers(
                root,
                "answers.jsonl",
                self.answer_records(
                    cases, selected_ids=["q-core-1"], condition_id="C0"
                ),
            )
            answer_link = root / "answers-link.jsonl"
            answer_link.symlink_to(answer_path)
            outputs = self.output_paths(root)

            result = self.run_cli(
                cases_path=cases_path,
                answer_paths=[answer_link],
                outputs=outputs,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("must not be symlinks", result.stderr)
            self.assertFalse(any(path.exists() for path in outputs.values()))


if __name__ == "__main__":
    unittest.main()
