from __future__ import annotations

import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_postprocessor_judge_pairs.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


artifacts = load_module(
    "service_eval_artifacts_for_postprocessor_judge_pair_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
projector = load_module(
    "project_raw_draft_answers_for_judge_pair_test",
    ROOT / "scripts" / "project_raw_draft_answers.py",
)
analyzer = load_module(
    "analyze_postprocessor_judge_pairs_for_test",
    SCRIPT,
)


class AnalyzePostprocessorJudgePairsTests(unittest.TestCase):
    maxDiff = None

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )

    def source_answer(self, case_id: str) -> dict:
        text = f"{case_id} 신청은 학생지원시스템에서 할 수 있습니다."
        collector_config = {
            "collector": "fixture",
            "provider": "frontier",
            "model": "fixture-model",
            "retrieval_mode": "bm25",
        }
        context = {
            "rank": 1,
            "stage": "final_contexts",
            "source_number": 1,
            "chunk_id": f"doc:{case_id}#0001",
            "document_id": f"doc:{case_id}",
            "source_title": "부산대학교 안내",
            "text": text,
        }
        return artifacts.build_answer_identity(
            {
                "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "source-experiment",
                "condition_id": "c1",
                "generation_run_id": "run1",
                "case_id": case_id,
                "id": case_id,
                "case_sha256": artifacts.sha256_text(case_id),
                "query": f"{case_id} 신청 경로를 알려주세요.",
                "answer": text,
                "cited_answer": text,
                "claims": [],
                "citations": [],
                "postprocessing": {"input_claim_count": 1},
                "generator": "fixture-generator",
                "collector_config": collector_config,
                "collector_config_sha256": artifacts.sha256_json(collector_config),
                "retrieval": {"mode": "bm25"},
                "evaluation_trace": {
                    "schema_version": 1,
                    "raw_draft": text,
                    "sanitized_draft": text,
                    "retrieval_stages": {"final_contexts": [context]},
                },
            }
        )

    def projection_pair(self, root: Path) -> tuple[Path, Path, list[dict], list[dict]]:
        sources = [self.source_answer(case_id) for case_id in ("q1", "q2", "q3")]
        source_path = root / "source.answers.jsonl"
        self.write_jsonl(source_path, sources)
        source_sha = analyzer.sha256_file(source_path)
        raw = [
            projector.project_answer(
                source,
                source_artifact_path=source_path,
                source_artifact_sha256=source_sha,
                experiment_id="judge-pair-fixture",
                condition_id="raw-condition",
                projection="raw-draft",
            )
            for source in sources
        ]
        current = [
            projector.project_answer(
                source,
                source_artifact_path=source_path,
                source_artifact_sha256=source_sha,
                experiment_id="judge-pair-fixture",
                condition_id="current-condition",
                projection="current-postprocessed",
            )
            for source in sources
        ]
        raw_path = root / "raw.answers.jsonl"
        current_path = root / "current.answers.jsonl"
        self.write_jsonl(raw_path, raw)
        self.write_jsonl(current_path, current)
        return raw_path, current_path, raw, current

    def judgment_runs(
        self,
        root: Path,
        *,
        label: str,
        answer_path: Path,
        answers: list[dict],
        scores: dict[str, list[int]],
        invalid: tuple[str, int] | None = None,
        config_model: str = "fixture-judge",
    ) -> list[Path]:
        answer_artifact_sha = analyzer.sha256_file(answer_path)
        judge_config = {"provider": "fixture", "model": config_model}
        judge_config_sha = artifacts.sha256_json(judge_config)
        paths: list[Path] = []
        for repeat_index in range(3):
            run_id = f"{label}-r{repeat_index + 1}"
            records: list[dict] = []
            for answer in answers:
                score = scores[answer["case_id"]][repeat_index]
                is_invalid = invalid == (answer["case_id"], repeat_index)
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
                    "answers_artifact_sha256": answer_artifact_sha,
                    "judge_config": judge_config,
                    "judge_config_sha256": judge_config_sha,
                    "judge": (
                        {"score": None, "reason": "judge request failed"}
                        if is_invalid
                        else {
                            "score": score,
                            "grounded_fully_correct": score == 2,
                        }
                    ),
                    "error": "quote validation failed" if is_invalid else None,
                }
                if is_invalid:
                    record["raw_judge_response"] = json.dumps({"score": 2})
                records.append(artifacts.build_judgment_identity(record))
            path = root / f"{run_id}.jsonl"
            self.write_jsonl(path, records)
            paths.append(path)
        return paths

    def fixture(self, root: Path, *, invalid=True):
        raw_path, current_path, raw, current = self.projection_pair(root)
        raw_runs = self.judgment_runs(
            root,
            label="raw",
            answer_path=raw_path,
            answers=raw,
            scores={"q1": [1, 1, 1], "q2": [0, 0, 0], "q3": [2, 2, 2]},
            invalid=("q2", 1) if invalid else None,
        )
        current_runs = self.judgment_runs(
            root,
            label="current",
            answer_path=current_path,
            answers=current,
            scores={"q1": [2, 2, 2], "q2": [2, 2, 2], "q3": [1, 1, 1]},
        )
        return raw_path, current_path, raw_runs, current_runs

    def run_cli(
        self,
        raw_path: Path,
        current_path: Path,
        raw_runs: list[Path],
        current_runs: list[Path],
        json_out: Path,
        csv_out: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--raw-answers",
            str(raw_path),
            "--current-answers",
            str(current_path),
            "--raw-judgments",
            *(str(path) for path in raw_runs),
            "--current-judgments",
            *(str(path) for path in current_runs),
            "--json-out",
            str(json_out),
        ]
        if csv_out is not None:
            command.extend(["--csv-out", str(csv_out)])
        return subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )

    def test_invalid_repeat_is_reported_and_never_recovered_from_raw_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, current_path, raw_runs, current_runs = self.fixture(root)
            payload, _ = analyzer.build_analysis(
                raw_path, current_path, raw_runs, current_runs
            )

        self.assertEqual(payload["summary"]["question_count"], 3)
        self.assertEqual(payload["summary"]["effective_sample_n"], 2)
        self.assertEqual(payload["summary"]["judgment_call_count"], 18)
        self.assertEqual(payload["summary"]["not_comparable_case_ids"], ["q2"])
        self.assertEqual(payload["summary"]["current_win_count"], 1)
        self.assertEqual(payload["summary"]["tie_count"], 0)
        self.assertEqual(payload["summary"]["raw_win_count"], 1)
        self.assertEqual(payload["summary"]["raw_question_mean_score"], 1.5)
        self.assertEqual(payload["summary"]["current_question_mean_score"], 1.5)
        self.assertEqual(payload["summary"]["mean_delta_current_minus_raw"], 0)
        q2 = next(row for row in payload["cases"] if row["case_id"] == "q2")
        self.assertEqual(q2["raw"]["scores"], [0, None, 0])
        self.assertEqual(q2["raw"]["errors"], [None, "quote validation failed", None])
        self.assertIsNone(q2["raw"]["mean_score"])
        self.assertFalse(q2["fully_comparable"])
        self.assertEqual(q2["outcome"], "not_comparable")
        self.assertTrue(
            all(
                row["raw_judge_response_score_recovery_attempted"] is False
                for row in q2["raw"]["repeats"]
            )
        )
        self.assertFalse(payload["eligibility"]["final_service_performance_eligible"])
        self.assertFalse(payload["eligibility"]["grounded_fully_correct_eligible"])
        self.assertFalse(payload["eligibility"]["citation_performance_eligible"])
        self.assertFalse(
            payload["methodology"]["judge_repeats_count_as_additional_samples"]
        )

    def test_cli_publishes_immutable_json_and_optional_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, current_path, raw_runs, current_runs = self.fixture(root)
            json_out = root / "analysis.json"
            csv_out = root / "analysis.csv"
            result = self.run_cli(
                raw_path,
                current_path,
                raw_runs,
                current_runs,
                json_out,
                csv_out,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"],
                "pnu.postprocessor-judge-pair-analysis.v1",
            )
            self.assertIn("effective_sample_n=2", result.stdout)
            self.assertEqual(
                payload["output_publication"]["required_companion_artifacts"][0][
                    "sha256"
                ],
                analyzer.sha256_file(csv_out),
            )
            with csv_out.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(json.loads(rows[1]["raw_scores"]), [0, None, 0])

            original = json_out.read_bytes()
            second = self.run_cli(
                raw_path,
                current_path,
                raw_runs,
                current_runs,
                json_out,
                csv_out,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("immutable output path already exists", second.stderr)
            self.assertEqual(json_out.read_bytes(), original)

    def test_artifact_hash_binding_and_exact_case_set_fail_closed(self):
        for mutation, message in (("hash", "answers_artifact_sha256 mismatch"), ("case", "case count")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw_path, current_path, raw_runs, current_runs = self.fixture(root)
                records = [json.loads(line) for line in raw_runs[0].read_text().splitlines()]
                if mutation == "hash":
                    changed = copy.deepcopy(records[0])
                    changed["answers_artifact_sha256"] = "0" * 64
                    records[0] = artifacts.build_judgment_identity(changed)
                else:
                    records.pop()
                self.write_jsonl(raw_runs[0], records)

                with self.assertRaisesRegex(ValueError, message):
                    analyzer.build_analysis(
                        raw_path, current_path, raw_runs, current_runs
                    )

    def test_mismatched_judge_config_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, current_path, raw, current = self.projection_pair(root)
            raw_runs = self.judgment_runs(
                root,
                label="raw",
                answer_path=raw_path,
                answers=raw,
                scores={case: [2, 2, 2] for case in ("q1", "q2", "q3")},
            )
            current_runs = self.judgment_runs(
                root,
                label="current",
                answer_path=current_path,
                answers=current,
                scores={case: [2, 2, 2] for case in ("q1", "q2", "q3")},
                config_model="different-judge",
            )
            with self.assertRaisesRegex(ValueError, "one identical Judge config"):
                analyzer.build_analysis(raw_path, current_path, raw_runs, current_runs)


if __name__ == "__main__":
    unittest.main()
