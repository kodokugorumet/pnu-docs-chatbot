from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_final_generation_schedule as runner  # noqa: E402
from service_eval_artifacts import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    build_answer_identity,
    record_sha256,
    sha256_json,
)
from tests import test_holdout_gold as holdout_fixture  # noqa: E402

EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "scheduled_evaluate_service_answers",
    SCRIPTS / "evaluate_service_answers.py",
)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
evaluator = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluator)


class FinalGenerationScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cases_path = self.root / "holdout-v2.jsonl"
        self.cases = holdout_fixture.HoldoutGoldTests().valid_cases()
        self.cases_path.write_text(
            "".join(
                json.dumps(case, ensure_ascii=False) + "\n"
                for case in self.cases
            ),
            encoding="utf-8",
        )
        self.output_root = self.root / "run"
        self.schedule_path = self.root / "meta" / "schedule.json"
        self.dev_path = self.root / "dev.json"
        self.signoff_path = self.root / "signoff.json"
        self.review_packet_path = self.root / "review-packet.md"
        self.review_a_path = self.root / "review-a.json"
        self.review_b_path = self.root / "review-b.json"
        self.corpus_path = self.root / "corpus.sqlite"
        self.dev_path.write_text('{"source":"dev"}\n', encoding="utf-8")
        self.review_packet_path.write_text("# review packet\n", encoding="utf-8")
        self.review_a_path.write_text('{"reviewer_slot":"A"}\n', encoding="utf-8")
        self.review_b_path.write_text('{"reviewer_slot":"B"}\n', encoding="utf-8")
        packet_sha = hashlib.sha256(self.review_packet_path.read_bytes()).hexdigest()
        response_provenance = [
            {
                "reviewer_slot": slot,
                "reviewer_id": reviewer_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "path": path.name,
            }
            for slot, reviewer_id, path in (
                ("A", "reviewer-a", self.review_a_path),
                ("B", "reviewer-b", self.review_b_path),
            )
        ]
        self.signoff_path.write_text(
            json.dumps(
                {
                    "schema_version": runner.signoff_artifacts.SIGNOFF_SCHEMA_VERSION,
                    "cases_sha256": hashlib.sha256(
                        self.cases_path.read_bytes()
                    ).hexdigest(),
                    "review_packet_path": self.review_packet_path.name,
                    "review_packet_sha256": packet_sha,
                    "review_response_sha256s": response_provenance,
                    "case_signoffs": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.original_signoff_bytes = self.signoff_path.read_bytes()
        self.corpus_path.write_bytes(b"sqlite-fixture-snapshot")
        self.index_sha256 = hashlib.sha256(
            self.corpus_path.read_bytes()
        ).hexdigest()
        self.source_manifest_sha256 = "c" * 64
        self.schedule = runner.build_schedule(
            self.cases_path,
            self.cases,
            experiment_id="final-fixture",
            output_root=self.output_root,
            expected_corpus_revision="corpus-fixture",
            expected_git_commit="1" * 40,
            expected_index_sha256=self.index_sha256,
            expected_source_manifest_sha256=self.source_manifest_sha256,
        )
        self.gate_binding = runner.build_gate_input_binding(
            self.schedule,
            dev_paths=[self.dev_path],
            corpus_index=self.corpus_path,
            signoff_path=self.signoff_path,
            repo_root=self.root,
        )
        self.execute_gate_patcher = mock.patch.object(
            runner, "_revalidate_execute_gates",
            side_effect=lambda *_args: self.gate_summary(),
        )
        self.execute_gate_patcher.start()

    def tearDown(self) -> None:
        self.execute_gate_patcher.stop()
        self.temp.cleanup()

    def write_schedule(self) -> None:
        runner.write_new_schedule(self.schedule_path, self.schedule)

    def gate_summary(self) -> dict:
        return {
            "ok": True,
            "cases_sha256": self.schedule["cases_sha256"],
            "gates": {
                name: {"ok": True}
                for name in ("schema", "dev", "corpus", "signoff")
            },
            "errors": [],
            "gate_input_binding": self.gate_binding,
            "gate_input_binding_sha256": sha256_json(self.gate_binding),
        }

    def execute_gate_kwargs(self) -> dict:
        return {
            "dev_paths": [self.dev_path],
            "corpus_index": self.corpus_path,
            "signoff_path": self.signoff_path,
        }

    def frozen_git_identity(self, _repo_root: Path) -> dict:
        return {"commit": "1" * 40, "clean": True}

    def test_schedule_fixes_all_counts_order_and_ab_ba_balance(self) -> None:
        summary = self.schedule["summary"]
        self.assertEqual(summary["core_case_count"], 27)
        self.assertEqual(summary["challenge_case_count"], 9)
        self.assertEqual(summary["generation_run_count"], 3)
        self.assertEqual(summary["core_pair_count"], 81)
        self.assertEqual(summary["core_ab_pair_count"], 41)
        self.assertEqual(summary["core_ba_pair_count"], 40)
        self.assertEqual(summary["core_call_count"], 162)
        self.assertEqual(summary["challenge_c1_call_count"], 27)
        self.assertEqual(summary["total_call_count"], 189)
        self.assertEqual(len(self.schedule["artifacts"]), 9)
        assignment = self.schedule["order_assignment"]
        self.assertEqual(assignment["seed"], 20260914)
        self.assertEqual(
            assignment["algorithm"], "sha256-category-stratified-balanced-v1"
        )
        self.assertEqual(
            assignment["assignment_sha256"],
            "b40de1c50967a1f18a6fe96a7b340520384fa36a000196a6560d59a3cc44883a",
        )
        by_category: dict[str, list[str]] = {}
        case_categories = {str(case["id"]): str(case["category"]) for case in self.cases}
        for row in assignment["assignments"]:
            by_category.setdefault(case_categories[row["case_id"]], []).append(
                row["pair_label"]
            )
        self.assertTrue(
            all(sorted((values.count("AB"), values.count("BA"))) == [4, 5]
                for values in by_category.values())
        )

        calls = self.schedule["calls"]
        self.assertEqual(
            [row["call_order"] for row in calls], list(range(1, 190))
        )
        first_case = self.cases[0]["id"]
        self.assertEqual(
            [
                (row["case_id"], row["generation_run_id"], row["condition_id"])
                for row in calls[:6]
            ],
            [
                (first_case, run_id, condition)
                for run_id in runner.RUN_IDS
                for condition in (
                    (["c0", "c1"] if self.schedule["order_assignment"]["assignments"][(runner.RUN_IDS.index(run_id))]["pair_label"] == "AB" else ["c1", "c0"])
                )
            ],
        )
        self.assertTrue(
            all(
                row["phase"] == "challenge" and row["condition_id"] == "c1"
                for row in calls[162:]
            )
        )
        self.assertEqual(
            {row["generation_run_id"] for row in calls[162:]},
            {"run1", "run2", "run3"},
        )

    def test_written_schedule_rebuilds_exactly_and_cannot_be_overwritten(self) -> None:
        self.write_schedule()
        self.assertEqual(runner.validate_schedule(self.schedule_path), self.schedule)
        with self.assertRaisesRegex(runner.ScheduleError, "will not be overwritten"):
            runner.write_new_schedule(self.schedule_path, self.schedule)

    def test_schedule_tampering_is_rejected_even_if_digest_is_recomputed(self) -> None:
        tampered = copy.deepcopy(self.schedule)
        tampered["calls"][0]["condition_id"] = "c1"
        tampered["schedule_sha256"] = runner._schedule_digest(tampered)
        tampered["schedule_id"] = (
            "final_generation_" + tampered["schedule_sha256"][:24]
        )
        self.schedule_path.parent.mkdir(parents=True)
        self.schedule_path.write_text(
            json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(runner.ScheduleError, "not the canonical"):
            runner.validate_schedule(self.schedule_path)

    def test_cases_file_hash_change_is_rejected(self) -> None:
        self.write_schedule()
        self.cases_path.write_text(
            self.cases_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(runner.ScheduleError, "SHA-256"):
            runner.validate_schedule(self.schedule_path)

    def _collector_config(
        self,
        artifact: dict,
        *,
        gate_hash: str = "a" * 64,
    ) -> dict:
        shared = self.schedule["controls"]["shared"]
        condition = self.schedule["controls"]["conditions"][
            artifact["condition_id"]
        ]
        return {
            "api_base": condition["api_base"],
            "cases_path": self.schedule["cases_path"],
            "cases_sha256": self.schedule["cases_sha256"],
            "cases_canonical_sha256": self.schedule[
                "cases_canonical_sha256"
            ],
            "selected_case_ids_sha256": artifact[
                "expected_case_ids_sha256"
            ],
            "provider": shared["provider"],
            "model": shared["model"],
            "institution": None,
            "context_k": 8,
            "expected_generation_max_context_chars": 24000,
            "expected_generation_max_output_tokens": 900,
            "expected_generation_sampling_parameters": [],
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "expected_corpus_revision": "corpus-fixture",
            "expected_source_manifest_sha256": self.source_manifest_sha256,
            "expected_retrieval_tuning": condition[
                "expected_retrieval_tuning"
            ],
            "expected_context_chunks_per_document": 2,
            "eval_trace": True,
            "allow_unpinned": False,
            "max_attempts": 3,
            "retry_backoff_seconds": 2.0,
            "request_timeout_seconds": 180.0,
            "inter_call_sleep_seconds": 2.0,
        "errors_path": artifact["errors_path"],
        "attempt_journal_path": artifact["journal_path"],
            "server_config": {
                "parser_profile": "cascade",
                "retrieval_mode": "bm25",
                "corpus_revision": "corpus-fixture",
                "retrieval_tuning": condition["expected_retrieval_tuning"],
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": True,
                "generation": {
                    "provider": "frontier",
                    "configured": True,
                    "model": "gemini-3.5-flash-lite",
                    "max_context_chars": 24000,
                    "max_output_tokens": 900,
                    "sampling_parameters": [],
                },
                "freeze": {
                    "startup_git_commit": "1" * 40,
                    "startup_worktree_clean": True,
                    "startup_code_sha256": "7" * 64,
                    "process_started_at": "2026-09-02T00:00:00+00:00",
                },
                "profile_index": {
                    "sha256": self.index_sha256,
                    "size_bytes": self.corpus_path.stat().st_size,
                    "source_manifest_sha256": self.source_manifest_sha256,
                },
            },
            "final_authorization": runner._expected_authorization_metadata(
                self.schedule, artifact, gate_hash, self.gate_binding
            ),
        }

    def _answer_for_call(self, call_order: int) -> tuple[Path, dict]:
        entry = self.schedule["calls"][call_order - 1]
        artifacts = runner._artifact_lookup(self.schedule)
        artifact = artifacts[entry["artifact_id"]]
        config = self._collector_config(artifact)
        system_sha = hashlib.sha256(b"system").hexdigest()
        prompt_sha = hashlib.sha256(b"prompt").hexdigest()
        request_config = {
            "provider": "gemini",
            "model_requested": "gemini-3.5-flash-lite",
            "api_style": "generateContent",
            "generation_config": {"maxOutputTokens": 900},
            "prompt_used": True,
            "system_instruction_sha256": system_sha,
        }
        record = {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "record_type": "answer",
            "experiment_id": self.schedule["experiment_id"],
            "condition_id": entry["condition_id"],
            "generation_run_id": entry["generation_run_id"],
            "case_id": entry["case_id"],
            "id": entry["case_id"],
            "case_sha256": entry["case_sha256"],
            "collector_config": config,
            "collector_config_sha256": sha256_json(config),
            "call_order": call_order,
            "collection_schedule": runner._expected_record_schedule_metadata(
                self.schedule, entry
            ),
            "answer": "fixture answer",
            "slot_outcome": "answer",
            "answer_eligible_for_judge": True,
            "request_attempts": [
                {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
            ],
            "collection_attempt_number": 1,
            "generation": {
                "requested": "frontier",
                "used": "frontier",
                "model": "gemini-3.5-flash-lite",
                "prompt_sha256": prompt_sha,
                "system_instruction_sha256": system_sha,
                "request_config": request_config,
                "request_config_sha256": sha256_json(request_config),
            },
            "evaluation_trace": {
                "schema_version": 1,
                "raw_draft": "fixture answer",
                "sanitized_draft": "fixture answer",
                "generation_input": {
                    "user_prompt": "prompt",
                    "prompt_sha256": prompt_sha,
                    "system_instruction": "system",
                    "system_instruction_sha256": system_sha,
                },
                "retrieval_stages": {"final_contexts": []},
            },
        }
        return Path(artifact["answers_path"]), build_answer_identity(record)

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_slot_journal(self, record: dict) -> None:
        self._append_journal_event(record, "slot_started", None)

    def _append_journal_event(
        self, record: dict, event: str, outcome: str | None
    ) -> None:
        artifact = runner._artifact_lookup(self.schedule)[
            record["collection_schedule"]["artifact_id"]
        ]
        payload = {
            "schema_version": "pnu.final-generation-slot-journal.v1",
            "event": event,
            "schedule_id": self.schedule["schedule_id"],
            "schedule_sha256": self.schedule["schedule_sha256"],
            "artifact_id": artifact["artifact_id"],
            "call_order": record["call_order"],
            "case_id": record["case_id"],
            "case_sha256": record["case_sha256"],
            "max_chat_attempts": 3,
            "outcome": outcome,
        }
        payload["record_sha256"] = sha256_json(payload)
        payload["journal_event_id"] = "journal_" + sha256_json(payload)[:24]
        self._append(Path(artifact["journal_path"]), payload)

    def _terminal_for_call(self, call_order: int) -> tuple[Path, Path, dict, dict]:
        path, record = self._answer_for_call(call_order)
        attempts = [
            {"attempt_number": number, "status": "error", "elapsed_ms": 1.0,
             "error_type": "TimeoutError", "error": "timeout",
             "retryable": True, "http_status": None}
            for number in (1, 2, 3)
        ]
        error = evaluator.build_collection_error_record(
            record, attempt_number=1, stage="chat_transport",
            error=TimeoutError("timeout"), request_attempts=attempts,
        )
        for key in ("answer_id", "answer_sha256", "record_sha256"):
            record.pop(key, None)
        record.update({
            "answer": "[SERVICE_ERROR]", "slot_outcome": "service_error",
            "answer_eligible_for_judge": False,
            "collection_attempt_number": 1,
            "request_attempts": attempts,
            "service_error": {
                "stage": "chat_transport", "type": "TimeoutError",
                "message": "timeout", "retryable": True,
                "http_status": None, "request_attempts": attempts,
            },
        })
        record = build_answer_identity(record)
        artifact = runner._artifact_lookup(self.schedule)[
            record["collection_schedule"]["artifact_id"]
        ]
        return path, Path(artifact["errors_path"]), record, error

    def test_empty_and_valid_prefix_artifacts_are_audited(self) -> None:
        empty = runner.audit_schedule_artifacts(self.schedule)
        self.assertEqual(empty["completed_call_count"], 0)
        self.assertEqual(empty["next_call_order"], 1)

        path, record = self._answer_for_call(1)
        self._append(path, record)
        self._append_slot_journal(record)
        self._append_journal_event(record, "slot_completed", "answer")
        progress = runner.audit_schedule_artifacts(self.schedule)
        self.assertEqual(progress["completed_call_count"], 1)
        self.assertEqual(progress["next_call_order"], 2)
        self.assertEqual(progress["gate_summary_sha256"], "a" * 64)

    def test_out_of_order_missing_call_is_rejected(self) -> None:
        path, record = self._answer_for_call(2)
        self._append(path, record)
        self._append_slot_journal(record)
        self._append_journal_event(record, "slot_completed", "answer")
        with self.assertRaisesRegex(runner.ScheduleError, "global call_order prefix"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_completed_answer_without_completed_wal_is_rejected(self) -> None:
        path, record = self._answer_for_call(1)
        self._append(path, record)
        self._append_slot_journal(record)
        with self.assertRaisesRegex(
            runner.ScheduleError, "exact started/completed"
        ):
            runner.audit_schedule_artifacts(self.schedule)

    def test_duplicate_answer_is_rejected(self) -> None:
        path, record = self._answer_for_call(1)
        self._append(path, record)
        self._append_slot_journal(record)
        self._append(path, record)
        self._append_slot_journal(record)
        with self.assertRaisesRegex(ValueError, "duplicate answer_id"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_rehashed_control_mismatch_is_rejected(self) -> None:
        path, record = self._answer_for_call(1)
        record["collector_config"]["model"] = "other-model"
        record["collector_config_sha256"] = sha256_json(
            record["collector_config"]
        )
        record["record_sha256"] = record_sha256(record)
        self._append(path, record)
        with self.assertRaisesRegex(runner.ScheduleError, "model expected"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_unregistered_existing_output_is_rejected(self) -> None:
        unexpected = self.output_root / "generation" / "rogue.answers.jsonl"
        unexpected.parent.mkdir(parents=True)
        unexpected.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(runner.ScheduleError, "unexpected generation"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_draft_and_failed_signoff_stop_before_collection(self) -> None:
        draft_path = self.root / "holdout-v2.draft.jsonl"
        draft_path.write_bytes(self.cases_path.read_bytes())
        draft_schedule = runner.build_schedule(
            draft_path,
            self.cases,
            experiment_id="draft-fixture",
            output_root=self.root / "draft-run",
            expected_corpus_revision="corpus-fixture",
            expected_git_commit="1" * 40,
            expected_index_sha256=self.index_sha256,
            expected_source_manifest_sha256=self.source_manifest_sha256,
        )
        validator = mock.Mock()
        with self.assertRaisesRegex(runner.ScheduleError, "draft holdout"):
            runner.require_frozen_holdout(
                draft_schedule,
                dev_paths=[self.dev_path],
                corpus_index=self.corpus_path,
                signoff_path=self.signoff_path,
                repo_root=self.root,
                validator=validator,
                git_identity=self.frozen_git_identity,
            )
        validator.assert_not_called()

        validator.return_value = {
            "ok": False,
            "cases_sha256": self.schedule["cases_sha256"],
            "gates": {
                "schema": {"ok": True},
                "dev": {"ok": True},
                "corpus": {"ok": True},
                "signoff": {"ok": False},
            },
            "errors": ["signoff: two reviewers required"],
        }
        with self.assertRaisesRegex(runner.ScheduleError, "signoff"):
            runner.require_frozen_holdout(
                self.schedule,
                dev_paths=[self.dev_path],
                corpus_index=self.corpus_path,
                signoff_path=self.signoff_path,
                repo_root=self.root,
                validator=validator,
                git_identity=self.frozen_git_identity,
            )

    def test_passed_four_gate_summary_is_bound_to_cases_sha(self) -> None:
        gates = {
            name: {"ok": True}
            for name in ("schema", "dev", "corpus", "signoff")
        }
        summary = {
            "ok": True,
            "cases_sha256": self.schedule["cases_sha256"],
            "gates": gates,
            "errors": [],
        }
        result = runner.require_frozen_holdout(
            self.schedule,
            dev_paths=[self.dev_path],
            corpus_index=self.corpus_path,
            signoff_path=self.signoff_path,
            repo_root=self.root,
            validator=mock.Mock(return_value=summary),
            git_identity=self.frozen_git_identity,
        )
        self.assertEqual(result["cases_sha256"], summary["cases_sha256"])
        self.assertEqual(result["gate_input_binding"], self.gate_binding)

        wrong = {**summary, "cases_sha256": "0" * 64}
        with self.assertRaisesRegex(runner.ScheduleError, "cases_sha256"):
            runner.require_frozen_holdout(
                self.schedule,
                dev_paths=[self.dev_path],
                corpus_index=self.corpus_path,
                signoff_path=self.signoff_path,
                repo_root=self.root,
                validator=mock.Mock(return_value=wrong),
                git_identity=self.frozen_git_identity,
            )

    def test_local_dirty_or_wrong_git_state_blocks_generation_gate(self) -> None:
        validator = mock.Mock()
        for git in (
            {"commit": "1" * 40, "clean": False},
            {"commit": "2" * 40, "clean": True},
        ):
            with self.subTest(git=git), self.assertRaisesRegex(
                runner.ScheduleError, "clean frozen local Git commit"
            ):
                runner.require_frozen_holdout(
                    self.schedule,
                    dev_paths=[self.dev_path],
                    corpus_index=self.corpus_path,
                    signoff_path=self.signoff_path,
                    repo_root=self.root,
                    validator=validator,
                    git_identity=lambda _root, value=git: value,
                )
        validator.assert_not_called()

    def test_gate_input_mutation_during_validator_is_rejected(self) -> None:
        summary = {
            "ok": True,
            "cases_sha256": self.schedule["cases_sha256"],
            "gates": {
                name: {"ok": True}
                for name in ("schema", "dev", "corpus", "signoff")
            },
            "errors": [],
        }

        def mutating_validator(*_args: object, **_kwargs: object) -> dict:
            self.signoff_path.write_text('{"reviewers":["one"]}\n', encoding="utf-8")
            return summary

        with self.assertRaisesRegex(runner.ScheduleError, "changed while"):
            runner.require_frozen_holdout(
                self.schedule,
                dev_paths=[self.dev_path],
                corpus_index=self.corpus_path,
                signoff_path=self.signoff_path,
                repo_root=self.root,
                validator=mutating_validator,
                git_identity=self.frozen_git_identity,
            )

    def test_gate_binding_rehashes_packet_and_review_responses(self) -> None:
        self.assertEqual(
            self.gate_binding["review_packet"]["path"],
            str(self.review_packet_path.resolve()),
        )
        self.assertEqual(
            [row["reviewer_slot"] for row in self.gate_binding["review_responses"]],
            ["A", "B"],
        )
        for path in (
            self.review_packet_path,
            self.review_a_path,
            self.review_b_path,
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"tamper\n")
                with self.assertRaisesRegex(runner.ScheduleError, "byte SHA"):
                    runner.validate_gate_input_binding(
                        self.schedule, self.gate_binding
                    )
                path.write_bytes(original)
                runner.validate_gate_input_binding(self.schedule, self.gate_binding)

    def test_started_without_answer_is_uncertain_and_never_resumed(self) -> None:
        _path, record = self._answer_for_call(1)
        self._append_slot_journal(record)
        with self.assertRaisesRegex(runner.ScheduleError, "uncertain started"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_foreign_journal_identity_is_rejected(self) -> None:
        _path, record = self._answer_for_call(1)
        self._append_slot_journal(record)
        artifact = runner._artifact_lookup(self.schedule)[
            record["collection_schedule"]["artifact_id"]
        ]
        journal_path = Path(artifact["journal_path"])
        event = json.loads(journal_path.read_text(encoding="utf-8"))
        event["case_id"] = "foreign"
        payload = dict(event)
        payload.pop("record_sha256")
        payload.pop("journal_event_id")
        event["record_sha256"] = sha256_json(payload)
        with journal_path.open("w", encoding="utf-8") as sink:
            event_payload = dict(event)
            event_payload.pop("journal_event_id", None)
            event["journal_event_id"] = "journal_" + sha256_json(event_payload)[:24]
            sink.write(json.dumps(event) + "\n")
        with self.assertRaisesRegex(runner.ScheduleError, "identity/control"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_completed_and_poisoned_or_mismatched_outcome_is_rejected(self) -> None:
        path, record = self._answer_for_call(1)
        self._append(path, record)
        self._append_slot_journal(record)
        self._append_journal_event(record, "slot_completed", "service_error")
        with self.assertRaisesRegex(runner.ScheduleError, "outcome mismatch"):
            runner.audit_schedule_artifacts(self.schedule)

        artifact = runner._artifact_lookup(self.schedule)[
            record["collection_schedule"]["artifact_id"]
        ]
        Path(artifact["journal_path"]).unlink()
        self._append_slot_journal(record)
        self._append_journal_event(record, "slot_completed", "answer")
        self._append_journal_event(record, "slot_poisoned", "response_control")
        with self.assertRaisesRegex(runner.ScheduleError, "poisoned.*transition"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_terminal_requires_matching_sidecar_and_rejects_mismatch(self) -> None:
        path, errors_path, terminal, error = self._terminal_for_call(1)
        self._append(path, terminal)
        self._append_slot_journal(terminal)
        self._append_journal_event(terminal, "slot_completed", "service_error")
        with self.assertRaisesRegex(runner.ScheduleError, "exactly one"):
            runner.audit_schedule_artifacts(self.schedule)
        self._append(errors_path, error)
        error["request_attempts"][0]["elapsed_ms"] = 2.0
        error["record_sha256"] = record_sha256(error)
        errors_path.write_text(json.dumps(error) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(runner.ScheduleError, "sidecar mismatch"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_sidecar_only_and_success_with_error_are_rejected(self) -> None:
        path, errors_path, terminal, error = self._terminal_for_call(1)
        self._append(errors_path, error)
        with self.assertRaisesRegex(runner.ScheduleError, "sidecar-only"):
            runner.audit_schedule_artifacts(self.schedule)

        self._append(path, self._answer_for_call(1)[1])
        self._append_slot_journal(terminal)
        self._append_journal_event(terminal, "slot_completed", "answer")
        with self.assertRaisesRegex(runner.ScheduleError, "successful answer"):
            runner.audit_schedule_artifacts(self.schedule)

    def test_execution_passes_all_189_calls_to_collector_in_order(self) -> None:
        state = {"completed": 0}
        observed: list[tuple[int, str, str, str]] = []

        def fake_audit(_schedule: dict) -> dict:
            count = state["completed"]
            total = len(self.schedule["calls"])
            return {
                "complete": count == total,
                "completed_call_count": count,
                "next_call_order": None if count == total else count + 1,
            }

        def fake_collector(
            argv: list[str],
            *,
            final_authorization: object,
        ) -> None:
            entry = final_authorization.entry
            only_index = argv.index("--only") + 1
            self.assertEqual(argv[only_index], entry["case_id"])
            self.assertIn("--allow-partial", argv)
            self.assertEqual(entry["call_order"], state["completed"] + 1)
            observed.append(
                (
                    entry["call_order"],
                    entry["case_id"],
                    entry["generation_run_id"],
                    entry["condition_id"],
                )
            )
            state["completed"] += 1

        result = runner.execute_schedule(
            self.schedule,
            self.gate_summary(),
            dry_run=False,
            collector_main=fake_collector,
            audit_fn=fake_audit,
            **self.execute_gate_kwargs(),
        )
        self.assertTrue(result["complete"])
        self.assertEqual(len(observed), 189)
        self.assertEqual([row[0] for row in observed], list(range(1, 190)))

    def test_live_execution_rejects_existing_run_lock_before_collection(self) -> None:
        lock_path = self.output_root / ".final-generation-run.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text('{"pid":999999}\n', encoding="utf-8")
        collector = mock.Mock()

        with self.assertRaisesRegex(ValueError, "run lock already exists"):
            runner.execute_schedule(
                self.schedule,
                self.gate_summary(),
                dry_run=False,
                collector_main=collector,
                audit_fn=mock.Mock(),
                **self.execute_gate_kwargs(),
            )

        collector.assert_not_called()

    def test_programmatic_execute_rejects_empty_gates_and_mutated_signoff(self) -> None:
        invalid = {**self.gate_summary(), "gates": {}}
        with self.assertRaisesRegex(
            runner.ScheduleError, "validator-authentic|byte SHA"
        ):
            runner.execute_schedule(
                self.schedule, invalid, dry_run=True, **self.execute_gate_kwargs()
            )
        self.signoff_path.write_text('{"reviewers":["one"]}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            runner.ScheduleError, "validator-authentic|byte SHA"
        ):
            runner.execute_schedule(
                self.schedule, self.gate_summary(), dry_run=True,
                **self.execute_gate_kwargs(),
            )
        self.signoff_path.write_bytes(self.original_signoff_bytes)
        forged_binding = runner.build_gate_input_binding(
            self.schedule,
            dev_paths=[self.dev_path],
            corpus_index=self.corpus_path,
            signoff_path=self.signoff_path,
            repo_root=self.root,
        )
        forged = {
            **self.gate_summary(),
            "gate_input_binding": forged_binding,
            "gate_input_binding_sha256": sha256_json(forged_binding),
        }
        self.execute_gate_patcher.stop()
        with self.assertRaises(runner.ScheduleError):
            runner.execute_schedule(
                self.schedule, forged, dry_run=True,
                dev_paths=[self.dev_path], corpus_index=self.corpus_path,
                signoff_path=self.signoff_path,
            )
        self.execute_gate_patcher.start()

    def test_dry_run_never_invokes_collector(self) -> None:
        output = io.StringIO()
        collector = mock.Mock()
        with redirect_stdout(output):
            result = runner.execute_schedule(
                self.schedule,
                self.gate_summary(),
                dry_run=True,
                collector_main=collector,
                **self.execute_gate_kwargs(),
            )
        collector.assert_not_called()
        self.assertEqual(result["completed_call_count"], 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 189)
        first = json.loads(lines[0])
        self.assertEqual(first["call_order"], 1)
        self.assertFalse(first["standalone_executable"])
        self.assertTrue(first["internal_final_authorization_required"])
        self.assertIn("--allow-partial", first["collector_argv_preview"])

    def test_authorization_rejects_any_collector_control_mismatch(self) -> None:
        entry = self.schedule["calls"][0]
        artifact = runner._artifact_lookup(self.schedule)[entry["artifact_id"]]
        authorization = runner._issue_generation_authorization(
            schedule=self.schedule,
            entry=entry,
            artifact=artifact,
            gate_summary_sha256=sha256_json(self.gate_summary()),
            gate_input_binding=self.gate_binding,
        )
        shared = self.schedule["controls"]["shared"]
        condition = self.schedule["controls"]["conditions"]["c0"]
        controls = {
            "api_base": condition["api_base"],
            "provider": shared["provider"],
            "model": shared["model"],
            "institution": None,
            "context_k": 8,
            "expected_generation_max_context_chars": 24000,
            "expected_generation_max_output_tokens": 900,
            "expected_generation_sampling_parameters": [],
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "expected_corpus_revision": "corpus-fixture",
            "expected_retrieval_tuning": False,
            "expected_context_chunks_per_document": 2,
            "eval_trace": True,
            "allow_unpinned": False,
            "max_attempts": 3,
            "retry_backoff_seconds": 2.0,
            "request_timeout_seconds": 180.0,
            "inter_call_sleep_seconds": 2.0,
            "expected_git_commit": "1" * 40,
            "expected_index_sha256": self.index_sha256,
            "expected_source_manifest_sha256": self.source_manifest_sha256,
        }
        with mock.patch.object(
            runner, "require_frozen_holdout", return_value=self.gate_summary()
        ):
            accepted = runner.validate_generation_authorization(
                authorization.payload,
            cases_path=self.cases_path,
            output_path=Path(artifact["answers_path"]),
            errors_path=Path(artifact["errors_path"]),
            journal_path=Path(artifact["journal_path"]),
            experiment_id="final-fixture",
            condition_id="c0",
            generation_run_id="run1",
            selected_case_ids=[entry["case_id"]],
                collector_controls=controls,
            )
        self.assertEqual(
            accepted["record"]["call_order"], entry["call_order"]
        )
        mismatched = {**controls, "expected_retrieval_tuning": True}
        with mock.patch.object(
            runner, "require_frozen_holdout", return_value=self.gate_summary()
        ), self.assertRaisesRegex(runner.ScheduleError, "control"):
            runner.validate_generation_authorization(
                authorization.payload,
                cases_path=self.cases_path,
                output_path=Path(artifact["answers_path"]),
                errors_path=Path(artifact["errors_path"]),
                journal_path=Path(artifact["journal_path"]),
                experiment_id="final-fixture",
                condition_id="c0",
                generation_run_id="run1",
                selected_case_ids=[entry["case_id"]],
                collector_controls=mismatched,
            )

    def test_trusted_authorization_rejects_fabricated_gate_digest(self) -> None:
        entry = self.schedule["calls"][0]
        artifact = runner._artifact_lookup(self.schedule)[entry["artifact_id"]]
        authorization = runner._issue_generation_authorization(
            schedule=self.schedule,
            entry=entry,
            artifact=artifact,
            gate_summary_sha256="b" * 64,
            gate_input_binding=self.gate_binding,
        )
        authentic_gate = self.gate_summary()
        self.assertNotEqual("b" * 64, sha256_json(authentic_gate))
        with mock.patch.object(
            runner, "require_frozen_holdout", return_value=authentic_gate
        ), self.assertRaisesRegex(runner.ScheduleError, "validator-authentic"):
            runner.validate_generation_authorization(authorization.payload)

    def test_trusted_authorization_rejects_out_of_order_call(self) -> None:
        entry = self.schedule["calls"][1]
        artifact = runner._artifact_lookup(self.schedule)[entry["artifact_id"]]
        authentic_gate = self.gate_summary()
        authorization = runner._issue_generation_authorization(
            schedule=self.schedule,
            entry=entry,
            artifact=artifact,
            gate_summary_sha256=sha256_json(authentic_gate),
            gate_input_binding=self.gate_binding,
        )
        with mock.patch.object(
            runner, "require_frozen_holdout", return_value=authentic_gate
        ), self.assertRaisesRegex(runner.ScheduleError, "next scheduled call"):
            runner.validate_generation_authorization(authorization.payload)

    def test_public_evaluator_rejects_holdout_before_health_call(self) -> None:
        cases_path = self.root / "one-holdout.jsonl"
        cases_path.write_text(
            json.dumps(
                {
                    "id": "holdout-one",
                    "split": "holdout-core",
                    "query": "질문",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        health = mock.Mock()
        with mock.patch.object(evaluator, "call_health", health), self.assertRaises(
            SystemExit
        ) as caught:
            evaluator.main(
                [
                    "--cases",
                    str(cases_path),
                    "--out",
                    str(self.root / "answer.jsonl"),
                    "--experiment-id",
                    "unsafe",
                    "--condition-id",
                    "c1",
                    "--generation-run-id",
                    "run1",
                    "--provider",
                    "extractive",
                    "--parser-profile",
                    "cascade",
                    "--retrieval-mode",
                    "bm25",
                    "--only",
                    "holdout-one",
                    "--allow-partial",
                    "--allow-unpinned",
                ]
            )
        self.assertEqual(caught.exception.code, 2)
        health.assert_not_called()

    def test_marker_subclass_authorization_is_rejected_before_health(self) -> None:
        class ForgedSubclass:
            def authorize_evaluator_invocation(self, **_kwargs: object) -> dict:
                return {}

        ForgedSubclass.__name__ = "FinalCollectionAuthorization"
        ForgedSubclass.__module__ = "run_final_generation_schedule"
        forged = ForgedSubclass()

        with mock.patch.object(evaluator, "call_health") as health:
            with self.assertRaises(SystemExit):
                evaluator.main(
                    runner.build_collector_argv(
                        self.schedule, self.schedule["calls"][0]
                    ),
                    final_authorization=forged,
                )
        health.assert_not_called()

    def test_exact_authorization_with_tampered_payload_is_rejected(self) -> None:
        authorization = runner._issue_generation_authorization(
            schedule=self.schedule,
            entry={**self.schedule["calls"][0], "case_id": "tampered"},
            artifact=runner._artifact_lookup(self.schedule)[
                self.schedule["calls"][0]["artifact_id"]
            ],
            gate_summary_sha256=sha256_json(self.gate_summary()),
            gate_input_binding=self.gate_binding,
        )
        with mock.patch.object(
            runner, "require_frozen_holdout", return_value=self.gate_summary()
        ), self.assertRaises(runner.ScheduleError):
            runner.validate_generation_authorization(
                authorization.payload,
                cases_path=self.cases_path,
                output_path=Path(authorization.artifact["answers_path"]),
                errors_path=Path(authorization.artifact["errors_path"]),
                journal_path=Path(authorization.artifact["journal_path"]),
                experiment_id=self.schedule["experiment_id"],
                condition_id="c0", generation_run_id="run1",
                selected_case_ids=["tampered"], collector_controls={},
            )

    def test_concrete_authorization_replaces_stale_pre_auth_case_snapshot(self) -> None:
        entry = self.schedule["calls"][0]
        artifact = runner._artifact_lookup(self.schedule)[entry["artifact_id"]]
        authorization = runner._issue_generation_authorization(
            schedule=self.schedule,
            entry=entry,
            artifact=artifact,
            gate_summary_sha256=sha256_json(self.gate_summary()),
            gate_input_binding=self.gate_binding,
        )
        health = {
            "ready": True,
            "parser_profiles": [
                {
                    "id": "cascade",
                    "ready": True,
                    "corpus_revision": "corpus-fixture",
                    "index_sha256": self.index_sha256,
                    "source_manifest_sha256": self.source_manifest_sha256,
                    "index_size_bytes": self.corpus_path.stat().st_size,
                    "retrieval_modes": [{"id": "bm25", "ready": True}],
                }
            ],
            "service_config": {
                "retrieval_tuning": False,
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": True,
                "generation": {
                    "provider": "frontier",
                    "configured": True,
                    "allowed_models": ["gemini-3.5-flash-lite"],
                    "models": {
                        "gemini-3.5-flash-lite": {
                            "max_context_chars": 24000,
                            "max_output_tokens": 900,
                            "sampling_parameters": [],
                        }
                    },
                },
                "freeze": {
                    "startup_git_commit": "1" * 40,
                    "startup_worktree_clean": True,
                    "startup_code_sha256": "9" * 64,
                    "process_started_at": "fixture",
                },
            },
        }
        system_sha = hashlib.sha256(b"system").hexdigest()
        prompt_sha = hashlib.sha256(b"prompt").hexdigest()
        request_config = {
            "provider": "gemini",
            "model_requested": "gemini-3.5-flash-lite",
            "api_style": "generateContent",
            "generation_config": {"maxOutputTokens": 900},
            "prompt_used": True,
            "system_instruction_sha256": system_sha,
        }
        response = {
            "answer": "승인된 모의 답변",
            "cited_answer": "승인된 모의 답변",
            "institution": None,
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "generation": {
                "requested": "frontier",
                "used": "frontier",
                "model": "gemini-3.5-flash-lite",
                "prompt_sha256": prompt_sha,
                "system_instruction_sha256": system_sha,
                "request_config": request_config,
                "request_config_sha256": sha256_json(request_config),
            },
            "evaluation_trace": {
                "schema_version": 1,
                "raw_draft": "승인된 모의 답변",
                "sanitized_draft": "승인된 모의 답변",
                "generation_input": {
                    "user_prompt": "prompt",
                    "prompt_sha256": prompt_sha,
                    "system_instruction": "system",
                    "system_instruction_sha256": system_sha,
                },
                "retrieval_stages": {"final_contexts": []},
            },
            "results": [],
        }
        argv = runner.build_collector_argv(self.schedule, entry)
        stale_cases = copy.deepcopy(self.cases)
        stale_cases[0]["query"] = "변조된 stale 질문"
        observed_queries: list[str] = []

        def fake_chat(_base: str, case: dict, *_args: object, **_kwargs: object) -> dict:
            observed_queries.append(case["query"])
            return response

        with mock.patch.object(
            evaluator, "call_health", return_value=health
        ), mock.patch.object(
            evaluator, "load_cases", return_value=stale_cases
        ), mock.patch.object(
            evaluator, "call_chat", side_effect=fake_chat
        ), mock.patch.object(
            evaluator.time, "sleep", return_value=None
        ), mock.patch.object(
            runner, "require_frozen_holdout", return_value=self.gate_summary()
        ):
            evaluator.main(argv, final_authorization=authorization)

        self.assertEqual(observed_queries, [self.cases[0]["query"]])
        record = json.loads(Path(artifact["answers_path"]).read_text(encoding="utf-8"))
        self.assertEqual(record["call_order"], 1)
        self.assertEqual(record["query"], self.cases[0]["query"])
        self.assertEqual(
            record["collector_config"]["final_authorization"]["schedule_id"],
            self.schedule["schedule_id"],
        )
        self.assertEqual(
            record["collector_config"]["selected_case_ids_sha256"],
            artifact["expected_case_ids_sha256"],
        )

    def test_callable_fake_authorization_is_rejected_before_health(self) -> None:
        entry = self.schedule["calls"][0]

        class FakeAuthorization:
            def authorize_evaluator_invocation(self, **_kwargs: object) -> dict:
                return {}

        with mock.patch.object(evaluator, "call_health") as health:
            with self.assertRaises(SystemExit):
                evaluator.main(
                    runner.build_collector_argv(self.schedule, entry),
                    final_authorization=FakeAuthorization(),
                )
        health.assert_not_called()

    def test_retry_exhaustion_is_terminal_and_resume_never_replays_slot(self) -> None:
        entry = self.schedule["calls"][0]
        artifact = runner._artifact_lookup(self.schedule)[entry["artifact_id"]]
        authorization = runner._issue_generation_authorization(
            schedule=self.schedule, entry=entry, artifact=artifact,
            gate_summary_sha256=sha256_json(self.gate_summary()),
            gate_input_binding=self.gate_binding,
        )
        health = {
            "ready": True,
            "parser_profiles": [{
                "id": "cascade", "ready": True,
                "corpus_revision": "corpus-fixture",
                "index_sha256": self.index_sha256,
                "source_manifest_sha256": self.source_manifest_sha256,
                "index_size_bytes": self.corpus_path.stat().st_size,
                "retrieval_modes": [{"id": "bm25", "ready": True}],
            }],
            "service_config": {
                "retrieval_tuning": False,
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": True,
                "generation": {
                    "provider": "frontier", "configured": True,
                    "allowed_models": ["gemini-3.5-flash-lite"],
                    "models": {"gemini-3.5-flash-lite": {
                        "max_context_chars": 24000,
                        "max_output_tokens": 900,
                        "sampling_parameters": [],
                    }},
                },
                "freeze": {
                    "startup_git_commit": "1" * 40,
                    "startup_worktree_clean": True,
                    "startup_code_sha256": "9" * 64,
                    "process_started_at": "fixture",
                },
            },
        }
        calls = mock.Mock(
            side_effect=TimeoutError("synthetic retryable timeout")
        )
        with mock.patch.object(evaluator, "call_health", return_value=health), \
             mock.patch.object(evaluator, "call_chat", calls), \
             mock.patch.object(evaluator.time, "sleep", return_value=None), \
             mock.patch.object(runner, "require_frozen_holdout", return_value=self.gate_summary()):
            evaluator.main(
                runner.build_collector_argv(self.schedule, entry),
                final_authorization=authorization,
            )
        self.assertEqual(calls.call_count, 3)
        terminal = json.loads(Path(artifact["answers_path"]).read_text())
        self.assertEqual(terminal["slot_outcome"], "service_error")
        self.assertFalse(terminal["answer_eligible_for_judge"])
        self.assertEqual(len(terminal["service_error"]["request_attempts"]), 3)
        progress = runner.audit_schedule_artifacts(self.schedule)
        self.assertEqual(progress["next_call_order"], 2)
        # A resumed orchestration audits the completed prefix and selects slot
        # 2; it never invokes the collector for slot 1 again.
        self.assertEqual(calls.call_count, 3)

    def test_main_module_alias_preserves_concrete_capability_identity(self) -> None:
        script = ROOT / "scripts" / "run_final_generation_schedule.py"
        argv = sys.argv
        try:
            sys.argv = [str(script), "--help"]
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(caught.exception.code, 0)
            from service_eval_artifacts import ExactFinalCollectionAuthorization
            self.assertIs(
                type(
                runner._issue_generation_authorization(
                    schedule=self.schedule,
                    entry=self.schedule["calls"][0],
                    artifact=runner._artifact_lookup(self.schedule)[
                        self.schedule["calls"][0]["artifact_id"]
                    ],
                    gate_summary_sha256="b" * 64,
                    gate_input_binding=self.gate_binding,
                )),
                ExactFinalCollectionAuthorization,
            )
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
