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
SPEC = importlib.util.spec_from_file_location(
    "build_evidence_manifest", ROOT / "scripts" / "build_evidence_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
manifest_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_module)
ARTIFACT_SPEC = importlib.util.spec_from_file_location(
    "service_eval_artifacts_for_evidence_manifest_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
assert ARTIFACT_SPEC is not None and ARTIFACT_SPEC.loader is not None
artifact_module = importlib.util.module_from_spec(ARTIFACT_SPEC)
ARTIFACT_SPEC.loader.exec_module(artifact_module)


class EvidenceManifestTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_jsonl_stats_accept_complete_unique_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_jsonl(
                path,
                [
                    {"id": "q1", "judge": {"score": 2}},
                    {"id": "q2", "judge": {"score": 1}},
                ],
            )

            stats = manifest_module.jsonl_stats(path, expected_count=2)

            self.assertTrue(stats["complete"])
            self.assertEqual(stats["records"], 2)
            self.assertEqual(stats["unique_ids"], 2)
            self.assertEqual(stats["judge_scores"], {"2": 1, "1": 1})

    def test_jsonl_stats_reject_count_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_jsonl(
                path,
                [
                    {"id": "q1", "judge": {"score": 2}},
                    {"id": "q1", "judge": {"score": 0}},
                ],
            )

            stats = manifest_module.jsonl_stats(path, expected_count=3)

            self.assertFalse(stats["complete"])
            self.assertEqual(stats["duplicate_ids"], ["q1"])
            self.assertIn("record count 2 != expected 3", stats["problems"])

    def test_jsonl_stats_can_require_complete_judge_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_jsonl(
                path,
                [
                    {"id": "q1", "judge": {"score": 2}},
                    {"id": "q2", "judge": {"score": None}},
                ],
            )

            stats = manifest_module.jsonl_stats(
                path, expected_count=2, require_judge_scores=True
            )

            self.assertFalse(stats["complete"])
            self.assertIn("1 record(s) have no judge score", stats["problems"])

    def test_build_manifest_hashes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            artifact = repo_root / "result.jsonl"
            self.write_jsonl(artifact, [{"id": "q1", "judge": {"score": 2}}])

            manifest, problems = manifest_module.build_manifest(
                repo_root=repo_root,
                experiment_id="unit-test",
                artifacts=[artifact],
                expectations={artifact.resolve(): 1},
                metadata={"model": "test-model"},
                require_clean=False,
            )

            self.assertEqual(problems, [])
            self.assertTrue(manifest["validation"]["ok"])
            self.assertEqual(manifest["metadata"], {"model": "test-model"})
            self.assertEqual(len(manifest["artifacts"][0]["sha256"]), 64)

    def test_jsonl_stats_uses_stable_answer_id_for_answer_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            self.write_jsonl(
                path,
                [
                    {
                        "record_type": "answer",
                        "answer_id": "answer_1",
                        "case_id": "q1",
                        "answer_sha256": "a" * 64,
                    }
                ],
            )

            stats = manifest_module.jsonl_stats(path, expected_count=1)

        self.assertTrue(stats["complete"])
        self.assertEqual(stats["id_field"], "answer_id")
        self.assertEqual(stats["unique_ids"], 1)

    def test_manifest_cross_checks_answer_judgment_hash_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            answers = repo_root / "answers.jsonl"
            judgments = repo_root / "judgments.jsonl"
            answer = {
                "record_type": "answer",
                "experiment_id": "exp",
                "condition_id": "c1",
                "generation_run_id": "g1",
                "case_id": "q1",
                "answer_id": "answer_1",
                "answer_sha256": "a" * 64,
            }
            judgment = {
                "record_type": "judgment",
                "experiment_id": "exp",
                "condition_id": "c1",
                "generation_run_id": "g1",
                "judge_run_id": "j1",
                "case_id": "q1",
                "judgment_id": "judgment_1",
                "answer_id": "answer_1",
                "answer_sha256": "b" * 64,
                "judge": {"score": 2},
            }
            self.write_jsonl(answers, [answer])
            self.write_jsonl(judgments, [judgment])

            manifest, problems = manifest_module.build_manifest(
                repo_root=repo_root,
                experiment_id="exp",
                artifacts=[answers, judgments],
                expectations={answers.resolve(): 1, judgments.resolve(): 1},
                metadata={},
                require_clean=False,
            )

        self.assertFalse(manifest["validation"]["ok"])
        self.assertTrue(
            any("answer_sha256 mismatch" in problem for problem in problems)
        )

    def test_human_label_jsonl_requires_unique_answer_and_blind_item_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviewer-a.jsonl"
            base = {
                "schema_version": "pnu.human-answer-label.v1",
                "label_kind": "independent",
                "reviewer_id": "human-a",
            }
            self.write_jsonl(
                path,
                [
                    {**base, "answer_id": "a1", "blind_item_id": "blind-1"},
                    {**base, "answer_id": "a1", "blind_item_id": "blind-2"},
                    {**base, "answer_id": "a3", "blind_item_id": "blind-2"},
                ],
            )

            stats = manifest_module.jsonl_stats(path, expected_count=3)

            self.assertFalse(stats["complete"])
            self.assertTrue(stats["human_label_identity_required"])
            self.assertEqual(stats["duplicate_ids"], ["a1"])
            self.assertEqual(stats["duplicate_blind_item_ids"], ["blind-2"])

    def test_mixed_or_malformed_human_label_rows_cannot_bypass_identity_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed-human-labels.jsonl"
            self.write_jsonl(
                path,
                [
                    {
                        "schema_version": "pnu.human-answer-label.v1",
                        "record_type": "human_label",
                        "answer_id": "answer-1",
                        "blind_item_id": "blind-1",
                        "label_kind": "independent",
                        "reviewer_id": "human-a",
                    },
                    {
                        "id": "disguised-generic-row",
                        "answer_id": "answer-1",
                    },
                ],
            )

            stats = manifest_module.jsonl_stats(path, expected_count=2)

            self.assertFalse(stats["complete"])
            self.assertTrue(stats["human_label_identity_required"])
            self.assertEqual(stats["duplicate_ids"], ["answer-1"])
            self.assertEqual(stats["invalid_human_label_identity_records"], 1)
            self.assertGreater(stats["missing_blind_item_ids"], 0)

    def test_final_partial_selection_requires_exactly_two_judgment_groups(self) -> None:
        answers = [
            {
                "record_type": "answer",
                "experiment_id": "exp",
                "condition_id": "c0",
                "generation_run_id": "run1",
                "case_id": case_id,
                "answer_id": answer_id,
                "answer_sha256": digest,
            }
            for case_id, answer_id, digest in (
                ("q1", "answer-1", "a" * 64),
                ("q2", "answer-2", "b" * 64),
            )
        ]
        binding = {"selection_id": "fixed-selection"}
        binding_json = json.dumps(
            binding,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        selection = {
            "selection_id": "fixed-selection",
            "answer_group": ["exp", "c0", "run1"],
            "selected_judge_eligible_answer_ids": ["answer-1"],
            "answers_artifact_sha256": "c" * 64,
            "binding_json": binding_json,
            "final_protocol_selection": True,
        }

        def judgment(judge_run_id: str) -> dict:
            return {
                "record_type": "judgment",
                "experiment_id": "exp",
                "condition_id": "c0",
                "generation_run_id": "run1",
                "judge_run_id": judge_run_id,
                "case_id": "q1",
                "judgment_id": f"judgment-{judge_run_id}",
                "answer_id": "answer-1",
                "answer_sha256": "a" * 64,
                "answers_artifact_sha256": "c" * 64,
                "judge_repeat_selection": binding,
            }

        _, incomplete = manifest_module.validate_eval_artifact_links(
            [*answers, judgment("judge-r2")],
            partial_selections=[selection],
        )
        self.assertTrue(
            any("expected exactly 2 additional Judge runs" in item for item in incomplete)
        )

        links, complete = manifest_module.validate_eval_artifact_links(
            [*answers, judgment("judge-r2"), judgment("judge-r3")],
            partial_selections=[selection],
        )
        self.assertEqual(complete, [])
        self.assertEqual(
            links["partial_selection_judgment_group_counts"],
            {"fixed-selection": 2},
        )

    def test_partial_judgment_group_needs_explicit_bound_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            answers = repo_root / "answers.jsonl"
            judgments = repo_root / "partial-judgments.jsonl"
            answer_rows = [
                {
                    "record_type": "answer",
                    "experiment_id": "exp",
                    "condition_id": "c0",
                    "generation_run_id": "run1",
                    "case_id": case_id,
                    "answer_id": f"answer-{case_id}",
                    "answer_sha256": character * 64,
                }
                for case_id, character in (("q1", "a"), ("q2", "b"))
            ]
            self.write_jsonl(answers, answer_rows)
            answer_artifact_sha = hashlib.sha256(answers.read_bytes()).hexdigest()
            judgment = {
                "record_type": "judgment",
                "experiment_id": "exp",
                "condition_id": "c0",
                "generation_run_id": "run1",
                "judge_run_id": "judge-r2",
                "case_id": "q1",
                "judgment_id": "judgment-q1-r2",
                "answer_id": "answer-q1",
                "answer_sha256": "a" * 64,
                "answers_artifact_sha256": answer_artifact_sha,
                "judge": {"score": 2},
            }
            self.write_jsonl(judgments, [judgment])

            _, unsafe_problems = manifest_module.build_manifest(
                repo_root=repo_root,
                experiment_id="exp",
                artifacts=[answers, judgments],
                expectations={},
                metadata={},
                require_clean=False,
            )
            self.assertTrue(
                any("no unique explicit selection binding" in item for item in unsafe_problems)
            )

            selected_ids = ["answer-q1"]
            identity = {
                "schema_version": "pnu.judge-repeat-selection.v1",
                "answers_artifact_sha256": answer_artifact_sha,
                "selected_answer_ids": selected_ids,
                "selected_answer_ids_sha256": hashlib.sha256(
                    json.dumps(
                        selected_ids,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
            selection_sha = hashlib.sha256(
                json.dumps(
                    identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            selection = repo_root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        **identity,
                        "selection_id": "judge_repeat_selection_" + selection_sha[:24],
                        "selection_sha256": selection_sha,
                    }
                ),
                encoding="utf-8",
            )
            selection_artifact_sha = hashlib.sha256(
                selection.read_bytes()
            ).hexdigest()
            judgment["judge_repeat_selection"] = {
                "schema_version": identity["schema_version"],
                "selection_id": "judge_repeat_selection_" + selection_sha[:24],
                "selection_sha256": selection_sha,
                "selection_artifact_sha256": selection_artifact_sha,
                "answers_artifact_sha256": answer_artifact_sha,
                "selected_answer_count": 1,
                "selected_answer_ids_sha256": identity[
                    "selected_answer_ids_sha256"
                ],
            }
            self.write_jsonl(judgments, [judgment])
            manifest, problems = manifest_module.build_manifest(
                repo_root=repo_root,
                experiment_id="exp",
                artifacts=[answers, judgments, selection],
                expectations={},
                metadata={},
                require_clean=False,
                selection_manifests=[selection],
            )

            self.assertEqual(problems, [])
            self.assertTrue(manifest["validation"]["ok"])
            self.assertEqual(
                manifest["eval_artifact_links"]["used_partial_selection_count"], 1
            )

    def test_terminal_answer_needs_no_judgment_and_cannot_be_judged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            answers = repo_root / "answers.jsonl"
            judgments = repo_root / "judgments.jsonl"
            shared = {
                "schema_version": artifact_module.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "exp",
                "condition_id": "c0",
                "generation_run_id": "run2",
            }
            success = {
                **shared,
                "case_id": "q1",
                "answer": "ok",
                "slot_outcome": "answer",
                "answer_eligible_for_judge": True,
                "collector_config": {"max_attempts": 3},
                "collector_config_sha256": artifact_module.sha256_json(
                    {"max_attempts": 3}
                ),
                "request_attempts": [
                    {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
                ],
                "collection_attempt_number": 1,
            }
            attempts = [
                {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
            ]
            terminal = {
                **shared,
                "case_id": "q2",
                "answer": "[SERVICE_ERROR]",
                "slot_outcome": "service_error",
                "answer_eligible_for_judge": False,
                "collector_config": {"max_attempts": 3},
                "collector_config_sha256": artifact_module.sha256_json(
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
            success = artifact_module.build_answer_identity(success)
            terminal = artifact_module.build_answer_identity(terminal)
            self.write_jsonl(answers, [success, terminal])
            answer_artifact_sha = hashlib.sha256(answers.read_bytes()).hexdigest()
            judge_config = {"provider": "fixture", "model": "judge-v1"}
            success_judgment = artifact_module.build_judgment_identity({
                "schema_version": artifact_module.JUDGMENT_SCHEMA_VERSION,
                "record_type": "judgment",
                "experiment_id": "exp",
                "condition_id": "c0",
                "generation_run_id": "run2",
                "judge_run_id": "judge-r1",
                "case_id": "q1",
                "answer_id": success["answer_id"],
                "answer_sha256": success["answer_sha256"],
                "answer_record_sha256": artifact_module.sha256_json(success),
                "answers_artifact_sha256": answer_artifact_sha,
                "judge_config": judge_config,
                "judge_config_sha256": artifact_module.sha256_json(judge_config),
                "judge": {"score": 2, "grounded_fully_correct": True},
                "error": None,
            })
            self.write_jsonl(judgments, [success_judgment])

            manifest, problems = manifest_module.build_manifest(
                repo_root=repo_root,
                experiment_id="exp",
                artifacts=[answers, judgments],
                expectations={},
                metadata={},
                require_clean=False,
            )
            self.assertEqual(problems, [])
            self.assertTrue(manifest["eval_artifact_links"]["complete"])

            self.write_jsonl(
                judgments,
                [
                    success_judgment,
                    artifact_module.build_judgment_identity({
                        **{
                            key: value
                            for key, value in success_judgment.items()
                            if key not in {"judgment_id", "record_sha256"}
                        },
                        "case_id": "q2",
                        "answer_id": terminal["answer_id"],
                        "answer_sha256": terminal["answer_sha256"],
                        "answer_record_sha256": artifact_module.sha256_json(
                            terminal
                        ),
                        "judge": {
                            "score": 0,
                            "grounded_fully_correct": False,
                        },
                    }),
                ],
            )
            _, invalid = manifest_module.build_manifest(
                repo_root=repo_root,
                experiment_id="exp",
                artifacts=[answers, judgments],
                expectations={},
                metadata={},
                require_clean=False,
            )
            self.assertTrue(
                any("references terminal service-error" in item for item in invalid)
            )

    def test_wholly_omitted_successful_answer_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            c0_answers = repo_root / "c0-answers.jsonl"
            c1_answers = repo_root / "c1-answers.jsonl"
            judgments = repo_root / "c0-judgments.jsonl"
            base = {
                "record_type": "answer",
                "experiment_id": "exp",
                "generation_run_id": "run1",
                "case_id": "q1",
            }
            c0 = {
                **base,
                "condition_id": "c0",
                "answer_id": "answer-c0-q1",
                "answer_sha256": "a" * 64,
            }
            c1 = {
                **base,
                "condition_id": "c1",
                "answer_id": "answer-c1-q1",
                "answer_sha256": "b" * 64,
            }
            self.write_jsonl(c0_answers, [c0])
            self.write_jsonl(c1_answers, [c1])
            c0_sha = hashlib.sha256(c0_answers.read_bytes()).hexdigest()
            self.write_jsonl(
                judgments,
                [
                    {
                        "record_type": "judgment",
                        "experiment_id": "exp",
                        "condition_id": "c0",
                        "generation_run_id": "run1",
                        "judge_run_id": "judge-r1",
                        "case_id": "q1",
                        "judgment_id": "judgment-c0-q1",
                        "answer_id": c0["answer_id"],
                        "answer_sha256": c0["answer_sha256"],
                        "answers_artifact_sha256": c0_sha,
                        "judge": {"score": 2},
                    }
                ],
            )

            _, problems = manifest_module.build_manifest(
                repo_root=repo_root,
                experiment_id="exp",
                artifacts=[c0_answers, c1_answers, judgments],
                expectations={},
                metadata={},
                require_clean=False,
            )

            self.assertTrue(
                any(
                    "Judge-eligible answer group has no judgment group" in item
                    and "c1" in item
                    for item in problems
                )
            )

    def test_cli_rejects_manifest_output_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.jsonl"
            self.write_jsonl(artifact, [{"id": "q1"}])
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts" / "build_evidence_manifest.py"),
                    "--experiment-id",
                    "alias-test",
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(artifact),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not overwrite", result.stderr)

    def test_cli_publishes_immutable_authoritative_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.jsonl"
            self.write_jsonl(artifact, [{"id": "q1"}])
            output = root / "manifest.json"
            command = [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "build_evidence_manifest.py"),
                "--experiment-id",
                "immutable-test",
                "--artifact",
                str(artifact),
                "--output",
                str(output),
            ]

            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["output_publication"][
                    "authoritative_completion_artifact"
                ],
                "json",
            )

            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), payload
            )

    def test_cli_rejects_symlink_input_and_broken_symlink_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.jsonl"
            self.write_jsonl(artifact, [{"id": "q1"}])
            artifact_link = root / "artifact-link.jsonl"
            artifact_link.symlink_to(artifact)
            output = root / "manifest.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts" / "build_evidence_manifest.py"),
                    "--experiment-id",
                    "symlink-test",
                    "--artifact",
                    str(artifact_link),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not be symlinks", result.stderr)
            self.assertFalse(output.exists())

            output.symlink_to(root / "missing-target.json")
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts" / "build_evidence_manifest.py"),
                    "--experiment-id",
                    "symlink-test",
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("already exists", result.stderr)
            self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
