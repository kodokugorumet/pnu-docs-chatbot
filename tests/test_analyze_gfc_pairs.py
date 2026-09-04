from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_gfc_pairs.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


artifacts = load_module(
    "service_eval_artifacts_for_gfc_pairs_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
repeat_summary = load_module(
    "summarize_judge_repeats_for_gfc_pairs_test",
    ROOT / "scripts" / "summarize_judge_repeats.py",
)
analyzer = load_module(
    "analyze_gfc_pairs_for_test",
    SCRIPT,
)


class AnalyzeGfcPairsTests(unittest.TestCase):
    maxDiff = None

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def build_summary(
        self,
        root: Path,
        *,
        condition: str,
        values_by_case: dict[str, list[bool]],
        experiment_id: str = "paired-gfc-fixture",
    ) -> Path:
        repeat_counts = {len(values) for values in values_by_case.values()}
        if len(repeat_counts) != 1:
            raise AssertionError("fixture repeat counts must be equal")
        repeat_count = next(iter(repeat_counts))

        answer_path = root / f"{condition}-answers.jsonl"
        answers: list[dict] = []
        for case_id in values_by_case:
            answers.append(
                artifacts.build_answer_identity(
                    {
                        "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                        "record_type": "answer",
                        "experiment_id": experiment_id,
                        "condition_id": condition,
                        "generation_run_id": "run1",
                        "case_id": case_id,
                        "id": case_id,
                        "answer": f"{condition} answer for {case_id}",
                        "error": None,
                    }
                )
            )
        self.write_jsonl(answer_path, answers)
        answer_artifact_sha = repeat_summary.sha256_file(answer_path)

        judgment_paths: list[Path] = []
        for repeat_index in range(repeat_count):
            judge_config = {"provider": "fixture", "model": "fixture-judge"}
            judge_config_sha = artifacts.sha256_json(judge_config)
            judgment_path = root / f"{condition}-judge-r{repeat_index + 1}.jsonl"
            judgments: list[dict] = []
            for answer in answers:
                gfc = values_by_case[answer["case_id"]][repeat_index]
                judgments.append(
                    artifacts.build_judgment_identity(
                        {
                            "schema_version": artifacts.JUDGMENT_SCHEMA_VERSION,
                            "record_type": "judgment",
                            "experiment_id": experiment_id,
                            "condition_id": condition,
                            "generation_run_id": "run1",
                            "judge_run_id": f"{condition}-judge-r{repeat_index + 1}",
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
                )
            self.write_jsonl(judgment_path, judgments)
            judgment_paths.append(judgment_path)

        payload, _ = repeat_summary.aggregate_repeats(
            answer_path=answer_path,
            judgment_paths=judgment_paths,
        )
        summary_path = root / f"{condition}-summary.json"
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary_path

    def run_cli(
        self,
        *,
        c0_summary: Path,
        c1_summary: Path,
        json_out: Path,
        csv_out: Path,
        condition_a: str | None = None,
        condition_b: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--c0-summary",
            str(c0_summary),
            "--c1-summary",
            str(c1_summary),
            "--json-out",
            str(json_out),
            "--csv-out",
            str(csv_out),
            "--bootstrap",
            "500",
            "--sign-flip-iterations",
            "500",
            "--seed",
            "17",
        ]
        if condition_a is not None:
            command.extend(["--condition-a", condition_a])
        if condition_b is not None:
            command.extend(["--condition-b", condition_b])
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def standard_pair(self, root: Path) -> tuple[Path, Path]:
        c0 = self.build_summary(
            root,
            condition="c0",
            values_by_case={
                "q1": [True, True, True],
                "q2": [False, False, False],
                "q3": [False, False, False],
            },
        )
        c1 = self.build_summary(
            root,
            condition="c1",
            values_by_case={
                "q1": [True, True, True],
                "q2": [False, False, False],
                "q3": [True, True, True],
            },
        )
        return c0, c1

    def test_cli_revalidates_artifacts_and_analyzes_dev3_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)
            json_out = root / "analysis.json"
            csv_out = root / "analysis.csv"

            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=json_out,
                csv_out=csv_out,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "pnu.gfc-paired-analysis.v1")
            self.assertTrue(
                payload["integrity_verification"]["summaries_rebuilt_from_source_artifacts"]
            )
            self.assertFalse(
                payload["integrity_verification"][
                    "cross_condition_answer_id_equality_required"
                ]
            )
            self.assertEqual(payload["primary"]["question_count"], 3)
            self.assertEqual(payload["primary"]["effective_sample_n"], 3)
            self.assertEqual(
                payload["methodology"]["judge_repeat_count_per_condition"], 3
            )
            self.assertFalse(
                payload["methodology"]["judge_repeats_count_as_additional_samples"]
            )
            self.assertAlmostEqual(payload["primary"]["c0_mean_gfc_rate"], 1 / 3)
            self.assertAlmostEqual(payload["primary"]["c1_mean_gfc_rate"], 2 / 3)
            self.assertAlmostEqual(
                payload["primary"]["mean_delta_c1_minus_c0"], 1 / 3
            )
            self.assertEqual(payload["primary"]["c1_higher_count"], 1)
            self.assertEqual(payload["primary"]["equal_count"], 2)
            self.assertEqual(payload["primary"]["c1_lower_count"], 0)
            self.assertEqual(
                payload["primary"]["paired_sign_flip_two_sided"]["method"],
                "exact paired sign-flip randomization test",
            )
            self.assertEqual(
                payload["sensitivity"]["paired_2x2"],
                {"both_gfc": 1, "c0_only": 0, "c1_only": 1, "neither_gfc": 1},
            )
            self.assertEqual(
                payload["sensitivity"]["exact_mcnemar_two_sided"]["p_value"], 1.0
            )
            self.assertNotEqual(
                payload["cases"][0]["c0_answer_id"],
                payload["cases"][0]["c1_answer_id"],
            )

            with csv_out.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[2]["case_id"], "q3")
            self.assertEqual(float(rows[2]["delta_c1_minus_c0"]), 1.0)
            self.assertEqual(
                json.loads(rows[2]["c1_grounded_fully_correct_values"]),
                [True, True, True],
            )

    def test_answer_artifact_sha_mismatch_fails_closed_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)
            payload = json.loads(c0.read_text(encoding="utf-8"))
            payload["inputs"]["answers"]["sha256"] = "0" * 64
            c0.write_text(json.dumps(payload), encoding="utf-8")
            json_out = root / "analysis.json"
            csv_out = root / "analysis.csv"

            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=json_out,
                csv_out=csv_out,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("artifact SHA-256 mismatch", result.stderr)
            self.assertFalse(json_out.exists())
            self.assertFalse(csv_out.exists())

    def test_cli_accepts_explicit_nondefault_condition_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition_a = "raw-draft"
            condition_b = "current-postprocessed"
            summary_a = self.build_summary(
                root,
                condition=condition_a,
                values_by_case={
                    "q1": [False, False, False],
                    "q2": [True, True, True],
                },
            )
            summary_b = self.build_summary(
                root,
                condition=condition_b,
                values_by_case={
                    "q1": [True, True, True],
                    "q2": [True, True, True],
                },
            )
            json_out = root / "analysis.json"
            csv_out = root / "analysis.csv"

            result = self.run_cli(
                c0_summary=summary_a,
                c1_summary=summary_b,
                json_out=json_out,
                csv_out=csv_out,
                condition_a=condition_a,
                condition_b=condition_b,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["condition_binding"],
                {
                    "a": condition_a,
                    "b": condition_b,
                    "legacy_v1_field_aliases": {"c0": "a", "c1": "b"},
                },
            )
            self.assertEqual(payload["inputs"]["c0"]["condition_id"], condition_a)
            self.assertEqual(payload["inputs"]["c1"]["condition_id"], condition_b)
            self.assertIn(f"{condition_a}=", result.stdout)
            self.assertIn(f"{condition_b}=", result.stdout)

    def test_explicit_condition_binding_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)

            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=root / "analysis.json",
                csv_out=root / "analysis.csv",
                condition_a="raw-draft",
                condition_b="current-postprocessed",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected condition_id 'raw-draft'", result.stderr)
            self.assertFalse((root / "analysis.json").exists())
            self.assertFalse((root / "analysis.csv").exists())

    def test_judgment_metadata_binding_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)
            payload = json.loads(c1.read_text(encoding="utf-8"))
            payload["inputs"]["judgments"][0]["judge_run_id"] = "tampered-run"
            c1.write_text(json.dumps(payload), encoding="utf-8")

            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=root / "analysis.json",
                csv_out=root / "analysis.csv",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("judge_run_id differs from source artifact", result.stderr)

    def test_summary_answer_id_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)
            payload = json.loads(c0.read_text(encoding="utf-8"))
            payload["cases"][0]["answer_id"] = "answer_tampered"
            c0.write_text(json.dumps(payload), encoding="utf-8")

            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=root / "analysis.json",
                csv_out=root / "analysis.csv",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("case rows differ from source artifacts", result.stderr)

    def test_duplicate_and_null_case_values_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)
            payload = json.loads(c0.read_text(encoding="utf-8"))
            payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
            c0.write_text(json.dumps(payload), encoding="utf-8")
            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=root / "duplicate.json",
                csv_out=root / "duplicate.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate case_id", result.stderr)

            # Rebuild clean files, then inject a null repeated value.
            c0, c1 = self.standard_pair(root)
            payload = json.loads(c0.read_text(encoding="utf-8"))
            payload["cases"][0]["grounded_fully_correct_values"][1] = None
            c0.write_text(json.dumps(payload), encoding="utf-8")
            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=root / "null.json",
                csv_out=root / "null.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("null is not allowed", result.stderr)

    def test_cross_condition_missing_case_and_repeat_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, _ = self.standard_pair(root)
            c1_missing = self.build_summary(
                root,
                condition="c1",
                values_by_case={
                    "q1": [True, True, True],
                    "q2": [False, False, False],
                },
            )
            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1_missing,
                json_out=root / "missing.json",
                csv_out=root / "missing.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("case_id set mismatch", result.stderr)

            c1_two_repeats = self.build_summary(
                root,
                condition="c1",
                values_by_case={
                    "q1": [True, True],
                    "q2": [False, False],
                    "q3": [True, True],
                },
            )
            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1_two_repeats,
                json_out=root / "repeat.json",
                csv_out=root / "repeat.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("repeat count mismatch", result.stderr)

    def test_wrong_condition_and_experiment_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)
            result = self.run_cli(
                c0_summary=c1,
                c1_summary=c0,
                json_out=root / "condition.json",
                csv_out=root / "condition.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected condition_id 'c0'", result.stderr)

            c1_other_experiment = self.build_summary(
                root,
                condition="c1",
                experiment_id="other-experiment",
                values_by_case={
                    "q1": [True, True, True],
                    "q2": [False, False, False],
                    "q3": [True, True, True],
                },
            )
            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1_other_experiment,
                json_out=root / "experiment.json",
                csv_out=root / "experiment.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("experiment_id mismatch", result.stderr)

    def test_even_repeat_tie_majority_is_rejected_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0 = self.build_summary(
                root,
                condition="c0",
                values_by_case={"q1": [True, False]},
            )
            c1 = self.build_summary(
                root,
                condition="c1",
                values_by_case={"q1": [True, True]},
            )
            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=root / "analysis.json",
                csv_out=root / "analysis.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("gfc_majority is null", result.stderr)

    def test_sign_flip_exact_and_deterministic_monte_carlo(self) -> None:
        exact = analyzer.paired_sign_flip_test(
            [0, 0, 3],
            repeat_count=3,
            monte_carlo_iterations=100,
            seed=7,
        )
        self.assertEqual(exact["method"], "exact paired sign-flip randomization test")
        self.assertEqual(exact["enumerated_assignments"], 2)
        self.assertEqual(exact["p_value"], 1.0)

        first = analyzer.paired_sign_flip_test(
            [1, 2, -1, 3],
            repeat_count=3,
            monte_carlo_iterations=1_000,
            seed=19,
            exact_max_nonzero=0,
        )
        second = analyzer.paired_sign_flip_test(
            [1, 2, -1, 3],
            repeat_count=3,
            monte_carlo_iterations=1_000,
            seed=19,
            exact_max_nonzero=0,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["method"],
            "deterministic Monte Carlo paired sign-flip randomization test",
        )
        self.assertEqual(first["seed"], 19)

    def test_exact_mcnemar_known_value_and_no_discordance(self) -> None:
        self.assertEqual(
            analyzer.exact_mcnemar(c0_only=0, c1_only=5)["p_value"],
            0.0625,
        )
        self.assertEqual(
            analyzer.exact_mcnemar(c0_only=0, c1_only=0)["p_value"],
            1.0,
        )

    def test_output_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c0, c1 = self.standard_pair(root)
            result = self.run_cli(
                c0_summary=c0,
                c1_summary=c1,
                json_out=c0,
                csv_out=root / "analysis.csv",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overwrite input summaries", result.stderr)


if __name__ == "__main__":
    unittest.main()
