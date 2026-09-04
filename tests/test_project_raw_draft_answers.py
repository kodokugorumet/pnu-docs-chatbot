from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "project_raw_draft_answers.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


artifacts = load_module(
    "service_eval_artifacts_for_raw_draft_projection_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
projector = load_module(
    "project_raw_draft_answers_for_test",
    SCRIPT,
)


class ProjectRawDraftAnswersTests(unittest.TestCase):
    def source_answer(self, *, case_id: str = "q1") -> dict:
        collector_config = {
            "collector": "fixture",
            "max_attempts": 3,
            "provider": "frontier",
            "model": "fixture-model",
            "retrieval_mode": "bm25",
            "final_authorization": {"schedule_id": "fixture-final"},
        }
        context = {
            "rank": 1,
            "stage": "final_contexts",
            "source_number": 1,
            "chunk_id": "doc_fixture:cascade#0001",
            "document_id": "doc_fixture",
            "source_title": "등록금 납부 안내",
            "source_url": "https://example.test/registration",
            "text": "본등록 기간은 2026년 8월 24일부터 8월 27일까지입니다.",
        }
        record = {
            "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
            "record_type": "answer",
            "experiment_id": "source-experiment",
            "condition_id": "c1",
            "generation_run_id": "run1",
            "case_id": case_id,
            "id": case_id,
            "case_sha256": "a" * 64,
            "query": "본등록 기간은 언제인가요?",
            "answer": "- 저장 당시 최종 답변입니다.",
            "cited_answer": "- 저장 당시 최종 답변입니다. [1]",
            "claims": [{"text": "저장 당시 최종 답변입니다.", "supported": True}],
            "citations": [{"source_number": 1}],
            "postprocessing": {"input_claim_count": 1},
            "generator": "fixture-generator",
            "collector_config": collector_config,
            "collector_config_sha256": artifacts.sha256_json(collector_config),
            "retrieval": {"mode": "bm25"},
            "evaluation_trace": {
                "schema_version": 1,
                "raw_draft": (
                    "본등록 기간은 2026년 8월 24일부터 8월 27일까지입니다."
                ),
                "sanitized_draft": (
                    "본등록 기간은 2026년 8월 24일부터 8월 27일까지입니다."
                ),
                "retrieval_stages": {"final_contexts": [context]},
            },
            "slot_outcome": "answer",
            "answer_eligible_for_judge": True,
            "request_attempts": [
                {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
            ],
            "collection_attempt_number": 1,
        }
        return artifacts.build_answer_identity(record)

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            ),
            encoding="utf-8",
        )

    def run_cli(
        self,
        source: Path,
        output: Path,
        *,
        projection: str,
        experiment_id: str = "postprocessor-ab",
        condition_id: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT),
                "--answers",
                str(source),
                "--out",
                str(output),
                "--experiment-id",
                experiment_id,
                "--condition-id",
                condition_id or projection,
                "--projection",
                projection,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_raw_and_current_replay_share_bound_draft_and_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.answers.jsonl"
            source = self.source_answer()
            original = copy.deepcopy(source)
            self.write_jsonl(source_path, [source])
            source_artifact_sha = projector.sha256_file(source_path)

            raw = projector.project_answer(
                source,
                source_artifact_path=source_path,
                source_artifact_sha256=source_artifact_sha,
                experiment_id="postprocessor-ab",
                condition_id="raw-draft",
                projection="raw-draft",
            )
            current = projector.project_answer(
                source,
                source_artifact_path=source_path,
                source_artifact_sha256=source_artifact_sha,
                experiment_id="postprocessor-ab",
                condition_id="current-postprocessed",
                projection="current-postprocessed",
            )

            self.assertEqual(source, original)
            self.assertEqual(raw["answer"], source["evaluation_trace"]["sanitized_draft"])
            self.assertEqual(raw["claims"], [])
            self.assertEqual(raw["citations"], [])
            self.assertFalse(raw["postprocessing"]["applied"])
            self.assertEqual(
                current["answer"],
                "- 본등록 기간은 2026년 8월 24일부터 8월 27일까지입니다.",
            )
            self.assertTrue(current["claims"][0]["supported"])
            self.assertTrue(current["citations"])
            self.assertNotEqual(raw["answer_id"], current["answer_id"])

            raw_meta = raw["postprocessor_diagnostic"]
            current_meta = current["postprocessor_diagnostic"]
            self.assertEqual(raw_meta["primary_metric"], "llm_judge_score_0_1_2")
            self.assertFalse(raw_meta["grounded_fully_correct_eligible"])
            self.assertFalse(raw_meta["citation_performance_eligible"])
            self.assertFalse(raw_meta["service_performance_eligible"])
            self.assertEqual(
                raw_meta["source"]["answers_artifact_sha256"],
                source_artifact_sha,
            )
            self.assertEqual(raw_meta["source"]["answer_id"], source["answer_id"])
            self.assertEqual(
                raw_meta["source"]["answer_sha256"], source["answer_sha256"]
            )
            self.assertEqual(
                raw_meta["source"]["record_sha256"], source["record_sha256"]
            )
            self.assertEqual(
                raw_meta["source"]["sanitized_draft_sha256"],
                current_meta["source"]["sanitized_draft_sha256"],
            )
            self.assertEqual(
                raw_meta["source"]["final_contexts_sha256"],
                current_meta["source"]["final_contexts_sha256"],
            )
            self.assertEqual(
                raw_meta["projection"]["postprocessor_code_sha256"],
                current_meta["projection"]["postprocessor_code_sha256"],
            )
            self.assertNotIn("final_authorization", raw["collector_config"])
            self.assertNotIn(
                "final_authorization",
                json.dumps(raw_meta["source"], ensure_ascii=False),
            )
            self.assertEqual(
                raw_meta["source"]["collector_identity"],
                {
                    "provider": "frontier",
                    "model": "fixture-model",
                    "retrieval_mode": "bm25",
                },
            )
            self.assertNotIn("slot_outcome", raw)
            artifacts.validate_answer_record(raw)
            artifacts.validate_answer_record(current)

    def test_cli_publishes_revalidated_immutable_output_without_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.answers.jsonl"
            output_path = root / "raw.answers.jsonl"
            self.write_jsonl(source_path, [self.source_answer()])

            first = self.run_cli(source_path, output_path, projection="raw-draft")

            self.assertEqual(first.returncode, 0, first.stderr)
            published_bytes = output_path.read_bytes()
            record = json.loads(output_path.read_text(encoding="utf-8"))
            artifacts.validate_answer_record(record)
            self.assertEqual(record["condition_id"], "raw-draft")
            self.assertIn("diagnostic_only=true", first.stdout)

            second = self.run_cli(source_path, output_path, projection="raw-draft")
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("immutable output path already exists", second.stderr)
            self.assertEqual(output_path.read_bytes(), published_bytes)

            alias = self.run_cli(
                source_path,
                source_path,
                projection="current-postprocessed",
            )
            self.assertNotEqual(alias.returncode, 0)
            self.assertIn("must not overwrite or alias", alias.stderr)

    def test_corrupt_source_and_missing_trace_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt_path = root / "corrupt.answers.jsonl"
            corrupt = self.source_answer()
            corrupt["answer"] = "identity hash와 다른 답변"
            self.write_jsonl(corrupt_path, [corrupt])

            with self.assertRaisesRegex(ValueError, "answer_sha256 mismatch"):
                projector.project_artifact(
                    corrupt_path,
                    experiment_id="postprocessor-ab",
                    condition_id="raw-draft",
                    projection="raw-draft",
                )

            missing_path = root / "missing.answers.jsonl"
            missing = self.source_answer(case_id="q2")
            missing.pop("record_sha256")
            missing.pop("evaluation_trace")
            missing = artifacts.build_answer_identity(missing)
            self.write_jsonl(missing_path, [missing])
            result = self.run_cli(
                missing_path,
                root / "missing-output.jsonl",
                projection="raw-draft",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has no evaluation_trace", result.stderr)
            self.assertFalse((root / "missing-output.jsonl").exists())

    def test_target_identity_must_not_collide_with_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.answers.jsonl"
            self.write_jsonl(source_path, [self.source_answer()])

            result = self.run_cli(
                source_path,
                root / "projection.answers.jsonl",
                projection="raw-draft",
                experiment_id="source-experiment",
                condition_id="c1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must differ from the source identity", result.stderr)
            self.assertFalse((root / "projection.answers.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
