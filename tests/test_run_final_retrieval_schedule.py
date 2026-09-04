from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_final_retrieval_schedule",
    ROOT / "scripts" / "run_final_retrieval_schedule.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


class FinalRetrievalScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_sha = "a" * 64
        self.cases = [
            {
                "id": f"core-{index:02d}",
                "split": "holdout-core",
                "family_id": f"family-{index:02d}",
                "query": f"question {index}",
                "required_claims": [],
            }
            for index in range(27)
        ]
        self.cases_path = self.root / "holdout-v2.jsonl"
        write_jsonl(self.cases_path, self.cases)
        indexes = {}
        for profile in ("baseline", "challenger", "cascade"):
            path = self.root / f"{profile}.sqlite"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE index_meta (key TEXT, value TEXT)")
            revision = f"revision-{profile}"
            connection.executemany(
                "INSERT INTO index_meta VALUES (?, ?)",
                [
                    ("profile", profile),
                    ("corpus_revision", revision),
                    ("source_manifest_sha256", self.source_sha),
                ],
            )
            connection.commit()
            connection.close()
            indexes[profile] = (path, revision)
        self.lane_inputs = {}
        for lane_id in runner.LANE_IDS:
            profile = runner.LANE_PROFILES[lane_id]
            path, revision = indexes[profile]
            self.lane_inputs[lane_id] = {
                "api_base": f"http://127.0.0.1/{lane_id}",
                "index_path": path,
                "corpus_revision": revision,
            }
        self.schedule = runner.build_schedule(
            self.cases_path,
            self.cases,
            experiment_id="final-retrieval-fixture",
            output_root=self.root / "out",
            expected_git_commit="b" * 40,
            expected_source_manifest_sha256=self.source_sha,
            lane_inputs=self.lane_inputs,
        )
        self.dev = self.root / "dev.json"
        self.signoff = self.root / "signoff.json"
        self.review_packet = self.root / "review-packet.md"
        self.review_a = self.root / "review-a.json"
        self.review_b = self.root / "review-b.json"
        self.dev.write_text("{}\n", encoding="utf-8")
        self.review_packet.write_text("# review packet\n", encoding="utf-8")
        self.review_a.write_text('{"reviewer_slot":"A"}\n', encoding="utf-8")
        self.review_b.write_text('{"reviewer_slot":"B"}\n', encoding="utf-8")
        response_provenance = [
            {
                "reviewer_slot": slot,
                "reviewer_id": reviewer_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "path": path.name,
            }
            for slot, reviewer_id, path in (
                ("A", "reviewer-a", self.review_a),
                ("B", "reviewer-b", self.review_b),
            )
        ]
        self.signoff.write_text(
            json.dumps(
                {
                    "schema_version": runner.signoff_artifacts.SIGNOFF_SCHEMA_VERSION,
                    "cases_sha256": hashlib.sha256(
                        self.cases_path.read_bytes()
                    ).hexdigest(),
                    "review_packet_path": self.review_packet.name,
                    "review_packet_sha256": hashlib.sha256(
                        self.review_packet.read_bytes()
                    ).hexdigest(),
                    "review_response_sha256s": response_provenance,
                    "case_signoffs": [],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.original_signoff_bytes = self.signoff.read_bytes()

    def gate(self) -> dict:
        def valid(*args, **kwargs):
            del args, kwargs
            return {
                "ok": True,
                "gates": {
                    key: {"ok": True}
                    for key in ("schema", "dev", "corpus", "signoff")
                },
            }

        return runner.require_final_gates(
            self.schedule,
            dev_paths=[self.dev],
            signoff_path=self.signoff,
            repo_root=self.root,
            validator=valid,
            git_identity=lambda _root: {"commit": "b" * 40, "clean": True},
        )

    def test_schedule_is_exactly_case_major_four_lane_108_calls(self) -> None:
        calls = self.schedule["calls"]
        self.assertEqual(len(calls), 108)
        self.assertEqual([row["call_order"] for row in calls], list(range(1, 109)))
        self.assertEqual(
            [row["condition_id"] for row in calls[:4]], list(runner.LANE_IDS)
        )
        self.assertEqual(len({row["case_id"] for row in calls}), 27)
        self.assertEqual(self.schedule["controls"]["provider"], "extractive")
        self.assertIsNone(self.schedule["controls"]["model"])
        self.assertTrue(all(lane["source_manifest_sha256"] == self.source_sha for lane in self.schedule["lanes"].values()))

    def test_authorization_pins_one_case_lane_index_and_offline_provider(self) -> None:
        gate = self.gate()
        entry = self.schedule["calls"][0]
        artifact = runner._artifact_map(self.schedule)[entry["artifact_id"]]
        capability = runner._issue_retrieval_authorization(
            self.schedule, entry, artifact, gate
        )
        argv = runner.build_collector_argv(self.schedule, entry)
        self.assertIn("extractive", argv)
        self.assertNotIn("--model", argv)
        controls = {
            "api_base": self.schedule["lanes"]["par-b"]["api_base"],
            "provider": "extractive",
            "model": None,
            "institution": None,
            "context_k": 8,
            "parser_profile": "baseline",
            "retrieval_mode": "bm25",
            "expected_corpus_revision": "revision-baseline",
            "expected_retrieval_tuning": True,
            "expected_context_chunks_per_document": 2,
            "expected_generation_max_context_chars": None,
            "expected_generation_max_output_tokens": None,
            "expected_generation_sampling_parameters": None,
            "expected_git_commit": "b" * 40,
            "expected_index_sha256": self.schedule["lanes"]["par-b"]["index"]["sha256"],
            "expected_source_manifest_sha256": self.source_sha,
            "eval_trace": True,
            "allow_unpinned": False,
            "max_attempts": 3,
            "retry_backoff_seconds": 2.0,
            "request_timeout_seconds": 180.0,
            "inter_call_sleep_seconds": 0.0,
        }
        with mock.patch.object(
            runner, "require_final_gates", return_value=gate
        ) as canonical_gate:
            authorized = runner.validate_retrieval_authorization(
                capability.payload,
                cases_path=self.cases_path,
                output_path=Path(artifact["answers_path"]),
                errors_path=Path(artifact["errors_path"]),
                journal_path=Path(artifact["journal_path"]),
                experiment_id=self.schedule["experiment_id"],
                condition_id="par-b",
                generation_run_id=runner.RUN_ID,
                selected_case_ids=[entry["case_id"]],
                collector_controls=controls,
            )
        canonical_gate.assert_called_once_with(
            self.schedule,
            dev_paths=[self.dev.resolve()],
            signoff_path=self.signoff.resolve(),
        )
        self.assertEqual(
            authorized["collector"]["collection_purpose"], "final_retrieval"
        )
        self.assertEqual(authorized["record"]["call_order"], 1)
        controls["provider"] = "frontier"
        with mock.patch.object(
            runner, "require_final_gates", return_value=gate
        ), self.assertRaisesRegex(runner.RetrievalScheduleError, "provider"):
            runner.validate_retrieval_authorization(
                capability.payload,
                cases_path=self.cases_path,
                output_path=Path(artifact["answers_path"]),
                errors_path=Path(artifact["errors_path"]),
                journal_path=Path(artifact["journal_path"]),
                experiment_id=self.schedule["experiment_id"],
                condition_id="par-b",
                generation_run_id=runner.RUN_ID,
                selected_case_ids=[entry["case_id"]],
                collector_controls=controls,
            )

    def test_trusted_authorization_rejects_fabricated_semantic_gate(self) -> None:
        authentic_gate = self.gate()
        fabricated_gate = {
            **authentic_gate,
            "profiles": {},
            "git": {"commit": "b" * 40, "clean": False},
            "gate_summary_sha256": "f" * 64,
        }
        entry = self.schedule["calls"][0]
        artifact = runner._artifact_map(self.schedule)[entry["artifact_id"]]
        authorization = runner._issue_retrieval_authorization(
            self.schedule, entry, artifact, fabricated_gate
        )
        with mock.patch.object(
            runner, "require_final_gates", return_value=authentic_gate
        ) as canonical_gate, self.assertRaisesRegex(
            runner.RetrievalScheduleError, "validator-authentic"
        ):
            runner.validate_retrieval_authorization(
                authorization.payload,
                cases_path=self.cases_path,
                output_path=Path(artifact["answers_path"]),
                errors_path=Path(artifact["errors_path"]),
                journal_path=Path(artifact["journal_path"]),
                experiment_id=self.schedule["experiment_id"],
                condition_id="par-b",
                generation_run_id=runner.RUN_ID,
                selected_case_ids=[entry["case_id"]],
                collector_controls={},
            )
        canonical_gate.assert_called_once()

    def test_gate_rejects_dirty_freeze_and_toctou(self) -> None:
        with self.assertRaisesRegex(runner.RetrievalScheduleError, "clean frozen"):
            runner.require_final_gates(
                self.schedule,
                dev_paths=[self.dev],
                signoff_path=self.signoff,
                repo_root=self.root,
                validator=lambda *args, **kwargs: {},
                git_identity=lambda _root: {"commit": "b" * 40, "clean": False},
            )

        calls = 0

        def mutating_validator(*args, **kwargs):
            nonlocal calls
            del args, kwargs
            calls += 1
            if calls == 1:
                self.signoff.write_text('{"changed":true}\n', encoding="utf-8")
            return {
                "ok": True,
                "gates": {key: {"ok": True} for key in ("schema", "dev", "corpus", "signoff")},
            }

        with self.assertRaisesRegex(runner.RetrievalScheduleError, "TOCTOU"):
            runner.require_final_gates(
                self.schedule,
                dev_paths=[self.dev],
                signoff_path=self.signoff,
                repo_root=self.root,
                validator=mutating_validator,
                git_identity=lambda _root: {"commit": "b" * 40, "clean": True},
            )

    def test_schedule_is_exclusive_and_journal_uncertainty_poisoned(self) -> None:
        path = self.root / "schedule.json"
        runner.write_new_schedule(path, self.schedule)
        with self.assertRaisesRegex(runner.RetrievalScheduleError, "overwrite"):
            runner.write_new_schedule(path, self.schedule)

        artifact = self.schedule["artifacts"][0]
        event = {
            "schema_version": "pnu.final-generation-slot-journal.v1",
            "event": "slot_started",
            "schedule_id": self.schedule["schedule_id"],
            "schedule_sha256": self.schedule["schedule_sha256"],
            "artifact_id": artifact["artifact_id"],
            "call_order": 1,
            "case_id": self.cases[0]["id"],
            "case_sha256": self.schedule["calls"][0]["case_sha256"],
            "max_chat_attempts": 3,
            "outcome": None,
        }
        event["record_sha256"] = runner.sha256_json(event)
        event["journal_event_id"] = "journal_" + runner.sha256_json(event)[:24]
        write_jsonl(Path(artifact["journal_path"]), [event])
        with self.assertRaisesRegex(runner.RetrievalScheduleError, "uncertain"):
            runner.audit_artifacts(self.schedule)

    def test_gate_binding_is_rehashed_between_logical_calls(self) -> None:
        gate = self.gate()
        runner.validate_gate_binding(self.schedule, gate["inputs"])
        self.signoff.write_text('{"changed":true}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            runner.RetrievalScheduleError, "changed after sign-off"
        ):
            runner.validate_gate_binding(self.schedule, gate["inputs"])

        self.signoff.write_bytes(self.original_signoff_bytes)
        gate = self.gate()
        index = Path(self.schedule["lanes"]["par-b"]["index"]["path"])
        with index.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(
            runner.RetrievalScheduleError, "changed after sign-off"
        ):
            runner.validate_gate_binding(self.schedule, gate["inputs"])

    def test_gate_binding_rehashes_packet_and_review_responses(self) -> None:
        gate = self.gate()
        self.assertEqual(
            gate["inputs"]["review_packet"]["path"],
            str(self.review_packet.resolve()),
        )
        self.assertEqual(
            [row["reviewer_slot"] for row in gate["inputs"]["review_responses"]],
            ["A", "B"],
        )
        for path in (self.review_packet, self.review_a, self.review_b):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"tamper\n")
                with self.assertRaisesRegex(
                    runner.RetrievalScheduleError, "changed after sign-off"
                ):
                    runner.validate_gate_binding(self.schedule, gate["inputs"])
                path.write_bytes(original)
                runner.validate_gate_binding(self.schedule, gate["inputs"])


if __name__ == "__main__":
    unittest.main()
