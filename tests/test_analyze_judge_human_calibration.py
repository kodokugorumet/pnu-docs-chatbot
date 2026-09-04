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
SCRIPT = ROOT / "scripts" / "analyze_judge_human_calibration.py"
ARTIFACTS_SPEC = importlib.util.spec_from_file_location(
    "service_eval_artifacts_for_human_calibration_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
assert ARTIFACTS_SPEC is not None and ARTIFACTS_SPEC.loader is not None
artifacts = importlib.util.module_from_spec(ARTIFACTS_SPEC)
ARTIFACTS_SPEC.loader.exec_module(artifacts)


class AnalyzeJudgeHumanCalibrationTests(unittest.TestCase):
    maxDiff = None

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def answer_record(self, case_id: str, condition_id: str = "c1") -> dict:
        return artifacts.build_answer_identity(
            {
                "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "human-calibration-fixture",
                "condition_id": condition_id,
                "generation_run_id": "generation-1",
                "case_id": case_id,
                "id": case_id,
                "answer": f"answer for {case_id} under {condition_id}",
                "error": None,
            }
        )

    def judgment_record(
        self,
        answer: dict,
        *,
        score: int,
        gfc: bool,
        run_id: str = "judge-1",
        abstention: str = "not_applicable",
        injection_obedience: bool | None = None,
        error: str | None = None,
    ) -> dict:
        judge_config = {"provider": "fixture", "model": "judge-fixture"}
        judge = {
            "score": score,
            "grounded_fully_correct": gfc,
            "abstention": abstention,
        }
        if injection_obedience is not None:
            judge["injection_obedience"] = injection_obedience
        return artifacts.build_judgment_identity(
            {
                "schema_version": artifacts.JUDGMENT_SCHEMA_VERSION,
                "record_type": "judgment",
                "experiment_id": answer["experiment_id"],
                "condition_id": answer["condition_id"],
                "generation_run_id": answer["generation_run_id"],
                "judge_run_id": run_id,
                "case_id": answer["case_id"],
                "answer_id": answer["answer_id"],
                "answer_sha256": answer["answer_sha256"],
                "judge_config": judge_config,
                "judge_config_sha256": artifacts.sha256_json(judge_config),
                "judge": judge,
                "error": error,
            }
        )

    def human_labels(
        self,
        answer: dict,
        *,
        independent_a: tuple[int, bool],
        independent_b: tuple[int, bool],
        adjudicated: tuple[int, bool],
        split: str | None = None,
        challenge_type: str | None = None,
        correct_abstention: bool | None = None,
        injection_obedience: bool | None = None,
    ) -> list[dict]:
        labels: list[dict] = []
        for reviewer_id, label_kind, (score, gfc) in (
            ("human-a", "independent", independent_a),
            ("human-b", "independent", independent_b),
            ("consensus", "adjudicated", adjudicated),
        ):
            record = {
                "schema_version": "pnu.human-answer-label.v1",
                "record_type": "human_label",
                "blind_item_id": f"blind-{answer['answer_id']}",
                "answer_id": answer["answer_id"],
                "case_id": answer["case_id"],
                "reviewer_id": reviewer_id,
                "label_kind": label_kind,
                "score": score,
                "grounded_fully_correct": gfc,
                "uncertain": False,
                "notes": None,
            }
            if split is not None:
                record["split"] = split
            if challenge_type is not None:
                record["challenge_type"] = challenge_type
            # Nulls exercise the review-packet placeholder representation.
            record["correct_abstention"] = correct_abstention
            record["injection_obedience"] = injection_obedience
            labels.append(record)
        return labels

    def run_cli(
        self,
        *,
        judgments: list[Path],
        human_labels: Path | list[Path],
        json_out: Path,
        csv_out: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        human_label_paths = (
            [human_labels] if isinstance(human_labels, Path) else human_labels
        )
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--judgments",
            *(str(path) for path in judgments),
            "--human-labels",
            *(str(path) for path in human_label_paths),
            "--json-out",
            str(json_out),
        ]
        if csv_out is not None:
            command.extend(["--csv-out", str(csv_out)])
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def make_artifacts(
        self,
        root: Path,
        rows: list[
            tuple[
                str,
                tuple[int, bool],
                tuple[int, bool],
                tuple[int, bool],
                tuple[int, bool],
            ]
        ],
        *,
        split: str | None = None,
    ) -> tuple[Path, Path, list[dict], list[dict]]:
        judgments: list[dict] = []
        labels: list[dict] = []
        answers: list[dict] = []
        for case_id, judge_value, human_a, human_b, adjudicated in rows:
            answer = self.answer_record(case_id)
            answers.append(answer)
            judgments.append(
                self.judgment_record(
                    answer,
                    score=judge_value[0],
                    gfc=judge_value[1],
                )
            )
            labels.extend(
                self.human_labels(
                    answer,
                    independent_a=human_a,
                    independent_b=human_b,
                    adjudicated=adjudicated,
                    split=split,
                )
            )
        judgment_path = root / "judgments.jsonl"
        human_path = root / "human-labels.jsonl"
        self.write_jsonl(judgment_path, judgments)
        self.write_jsonl(human_path, labels)
        return judgment_path, human_path, judgments, labels

    def test_pass_fixture_computes_gate_agreement_challenge_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core_rows = [
                ("q1", (2, True), (2, True), (2, True), (2, True)),
                ("q2", (1, False), (1, False), (0, False), (1, False)),
                ("q3", (2, True), (2, True), (1, False), (2, True)),
                ("q4", (0, False), (0, False), (0, False), (0, False)),
            ]
            judgment_path, human_path, judgments, labels = self.make_artifacts(
                root, core_rows, split="holdout-core"
            )

            abstention_answer = self.answer_record("q5")
            judgments.append(
                self.judgment_record(
                    abstention_answer,
                    score=2,
                    gfc=True,
                    abstention="appropriate",
                )
            )
            labels.extend(
                self.human_labels(
                    abstention_answer,
                    independent_a=(2, True),
                    independent_b=(2, True),
                    adjudicated=(2, True),
                    split="holdout-challenge",
                    challenge_type="unanswerable",
                    correct_abstention=True,
                )
            )

            injection_answer = self.answer_record("q6")
            judgments.append(
                self.judgment_record(
                    injection_answer,
                    score=2,
                    gfc=True,
                    injection_obedience=False,
                )
            )
            labels.extend(
                self.human_labels(
                    injection_answer,
                    independent_a=(2, True),
                    independent_b=(2, True),
                    adjudicated=(2, True),
                    split="holdout-challenge",
                    challenge_type="prompt_injection",
                    injection_obedience=False,
                )
            )
            self.write_jsonl(judgment_path, judgments)
            self.write_jsonl(human_path, labels)

            json_out = root / "calibration.json"
            csv_out = root / "calibration.csv"
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=json_out,
                csv_out=csv_out,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"], "pnu.judge-human-calibration.v1"
            )
            summary = payload["summary"]
            self.assertEqual(summary["answer_count"], 6)
            self.assertEqual(summary["independent_human_rating_count"], 12)
            self.assertEqual(summary["adjudicated_human_rating_count"], 6)
            self.assertAlmostEqual(
                summary["human_human"]["gfc"]["raw_agreement"], 5 / 6
            )
            self.assertIsNotNone(
                summary["human_human"]["score_0_1_2"][
                    "quadratic_weighted_kappa"
                ]
            )
            self.assertEqual(
                summary["judge_vs_adjudicated_human_gfc"]["accuracy"], 1.0
            )
            self.assertEqual(
                summary["judge_vs_adjudicated_human_gfc"][
                    "balanced_accuracy"
                ],
                1.0,
            )
            gate = summary["protocol_gate"]
            self.assertEqual(gate["scope"], "core_split_only")
            self.assertEqual(gate["answer_count"], 4)
            self.assertTrue(gate["passed"])
            self.assertEqual(
                gate["checks"],
                {
                    "raw_agreement": True,
                    "balanced_accuracy": True,
                    "cohen_kappa": True,
                    "macro_f1": True,
                },
            )

            abstention = summary["challenge"]["correct_abstention"]
            self.assertEqual(abstention["answer_count"], 1)
            self.assertEqual(abstention["adjudicated_success_rate"], 1.0)
            self.assertEqual(abstention["judge_comparable_answer_count"], 1)
            injection = summary["challenge"]["injection_obedience"]
            self.assertEqual(injection["answer_count"], 1)
            self.assertEqual(
                injection["adjudicated_injection_resistance_rate"], 1.0
            )

            with csv_out.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 6)
            self.assertEqual(csv_rows[0]["reviewer_a_id"], "human-a")
            self.assertTrue(csv_rows[0]["blind_item_id"].startswith("blind-answer_"))

    def test_separate_packet_label_files_are_loaded_and_cross_file_duplicates_fail(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [("q1", (2, True), (2, True), (2, True), (2, True))]
            judgment_path, _, _, labels = self.make_artifacts(root, rows)
            # Match build_answer_review_packet.py's templates: record_type is
            # intentionally absent and only the editable nulls have been filled.
            for label in labels:
                label.pop("record_type")
            labels_a_path = root / "labels-a.jsonl"
            labels_b_path = root / "labels-b.jsonl"
            adjudication_path = root / "adjudication.jsonl"
            self.write_jsonl(labels_a_path, [labels[0]])
            self.write_jsonl(labels_b_path, [labels[1]])
            self.write_jsonl(adjudication_path, [labels[2]])
            output = root / "three-file-calibration.json"

            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=[
                    labels_a_path,
                    labels_b_path,
                    adjudication_path,
                ],
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [entry["path"] for entry in payload["inputs"]["human_labels"]],
                [
                    str(labels_a_path),
                    str(labels_b_path),
                    str(adjudication_path),
                ],
            )
            self.assertTrue(
                all(
                    len(entry["sha256"]) == 64
                    for entry in payload["inputs"]["human_labels"]
                )
            )

            duplicate_path = root / "duplicate-label.jsonl"
            self.write_jsonl(duplicate_path, [labels[0]])
            duplicate_output = root / "duplicate-across-files.json"
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=[
                    labels_a_path,
                    duplicate_path,
                    labels_b_path,
                    adjudication_path,
                ],
                json_out=duplicate_output,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate human label across artifacts", result.stderr)
            self.assertFalse(duplicate_output.exists())

    def test_fail_fixture_does_not_pass_protocol_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ("q1", (2, True), (0, False), (0, False), (0, False)),
                ("q2", (0, False), (2, True), (2, True), (2, True)),
                ("q3", (2, True), (0, False), (0, False), (0, False)),
                ("q4", (0, False), (2, True), (2, True), (2, True)),
            ]
            judgment_path, human_path, _, _ = self.make_artifacts(root, rows)
            output = root / "fail.json"
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            comparison = payload["summary"]["judge_vs_adjudicated_human_gfc"]
            self.assertEqual(comparison["accuracy"], 0.0)
            self.assertEqual(comparison["balanced_accuracy"], 0.0)
            self.assertEqual(comparison["cohen_kappa"], -1.0)
            self.assertEqual(comparison["macro_f1"], 0.0)
            self.assertEqual(comparison["false_pass_rate"], 1.0)
            self.assertFalse(payload["summary"]["protocol_gate"]["passed"])

    def test_gate_thresholds_are_inclusive_and_false_pass_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            human_gfc = [False] * 5 + [True] * 5
            judge_gfc = [False] * 4 + [True, False] + [True] * 4
            rows = [
                (
                    f"q{index}",
                    (2 if predicted else 0, predicted),
                    (2 if actual else 0, actual),
                    (2 if actual else 0, actual),
                    (2 if actual else 0, actual),
                )
                for index, (actual, predicted) in enumerate(
                    zip(human_gfc, judge_gfc), 1
                )
            ]
            judgment_path, human_path, _, _ = self.make_artifacts(root, rows)
            output = root / "threshold-boundary.json"

            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
            comparison = summary["judge_vs_adjudicated_human_gfc"]
            self.assertEqual(
                comparison["confusion_matrix"],
                {
                    "orientation": (
                        "rows=adjudicated_human_actual, columns=judge_predicted"
                    ),
                    "true_negative": 4,
                    "false_positive": 1,
                    "false_negative": 1,
                    "true_positive": 4,
                },
            )
            self.assertAlmostEqual(comparison["raw_agreement"], 0.80)
            self.assertAlmostEqual(comparison["balanced_accuracy"], 0.80)
            self.assertAlmostEqual(comparison["cohen_kappa"], 0.60)
            self.assertAlmostEqual(comparison["macro_f1"], 0.80)
            self.assertAlmostEqual(comparison["false_pass_rate"], 0.20)
            self.assertTrue(summary["protocol_gate"]["passed"])
            self.assertEqual(
                summary["protocol_gate"]["checks"],
                {
                    "raw_agreement": True,
                    "balanced_accuracy": True,
                    "cohen_kappa": True,
                    "macro_f1": True,
                },
            )

    def test_quadratic_weighted_kappa_uses_fixed_score_distance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviewer_a_scores = [0, 0, 1, 2, 2]
            reviewer_b_scores = [0, 1, 1, 1, 2]
            rows = [
                (
                    f"q{index}",
                    (0, False),
                    (score_a, False),
                    (score_b, False),
                    (0, False),
                )
                for index, (score_a, score_b) in enumerate(
                    zip(reviewer_a_scores, reviewer_b_scores), 1
                )
            ]
            judgment_path, human_path, _, _ = self.make_artifacts(root, rows)
            output = root / "qwk.json"

            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            metric = json.loads(output.read_text(encoding="utf-8"))["summary"][
                "human_human"
            ]["score_0_1_2"]
            self.assertTrue(metric["quadratic_weighted_kappa_defined"])
            self.assertAlmostEqual(metric["quadratic_weighted_kappa"], 2 / 3)

    def test_degenerate_kappa_is_null_and_cannot_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                (case_id, (0, False), (0, False), (0, False), (0, False))
                for case_id in ("q1", "q2", "q3")
            ]
            judgment_path, human_path, _, _ = self.make_artifacts(root, rows)
            output = root / "degenerate.json"
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
            self.assertIsNone(summary["human_human"]["gfc"]["cohen_kappa"])
            self.assertIsNone(
                summary["human_human"]["score_0_1_2"][
                    "quadratic_weighted_kappa"
                ]
            )
            comparison = summary["judge_vs_adjudicated_human_gfc"]
            self.assertEqual(comparison["accuracy"], 1.0)
            self.assertIsNone(comparison["balanced_accuracy"])
            self.assertFalse(comparison["balanced_accuracy_defined"])
            self.assertIsNone(comparison["cohen_kappa"])
            self.assertEqual(comparison["macro_f1"], 0.5)
            self.assertIsNone(comparison["false_pass_rate"])
            self.assertFalse(summary["protocol_gate"]["passed"])
            self.assertFalse(
                summary["protocol_gate"]["checks"]["balanced_accuracy"]
            )
            self.assertFalse(summary["protocol_gate"]["checks"]["cohen_kappa"])

    def test_missing_adjudication_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [("q1", (2, True), (2, True), (2, True), (2, True))]
            judgment_path, human_path, _, labels = self.make_artifacts(root, rows)
            self.write_jsonl(
                human_path,
                [record for record in labels if record["label_kind"] == "independent"],
            )
            output = root / "missing.json"
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly 1 adjudicated label", result.stderr)
            self.assertFalse(output.exists())

    def test_missing_all_labels_for_a_judgment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ("q1", (2, True), (2, True), (2, True), (2, True)),
                ("q2", (0, False), (0, False), (0, False), (0, False)),
            ]
            judgment_path, human_path, _, labels = self.make_artifacts(root, rows)
            q1_case_id = "q1"
            self.write_jsonl(
                human_path,
                [record for record in labels if record["case_id"] == q1_case_id],
            )
            output = root / "missing-all.json"
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing human labels for judged answer_id", result.stderr)
            self.assertFalse(output.exists())

    def test_duplicate_human_label_and_duplicate_judgment_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [("q1", (2, True), (2, True), (2, True), (2, True))]
            judgment_path, human_path, judgments, labels = self.make_artifacts(
                root, rows
            )

            self.write_jsonl(human_path, [*labels, labels[0]])
            duplicate_human_output = root / "duplicate-human.json"
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=duplicate_human_output,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate human label", result.stderr)
            self.assertFalse(duplicate_human_output.exists())

            self.write_jsonl(human_path, labels)
            second_judgment_path = root / "same-judgment.jsonl"
            self.write_jsonl(second_judgment_path, judgments)
            duplicate_judge_output = root / "duplicate-judge.json"
            result = self.run_cli(
                judgments=[judgment_path, second_judgment_path],
                human_labels=human_path,
                json_out=duplicate_judge_output,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate judgment_id across artifacts", result.stderr)
            self.assertFalse(duplicate_judge_output.exists())

    def test_blind_item_id_cannot_be_reused_across_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ("q1", (2, True), (2, True), (2, True), (2, True)),
                ("q2", (0, False), (0, False), (0, False), (0, False)),
            ]
            judgment_path, human_path, _, labels = self.make_artifacts(root, rows)
            q1_blind_id = next(
                record["blind_item_id"]
                for record in labels
                if record["case_id"] == "q1"
            )
            for record in labels:
                if record["case_id"] == "q2":
                    record["blind_item_id"] = q1_blind_id
            self.write_jsonl(human_path, labels)
            output = root / "duplicate-blind-id.json"

            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("blind_item_id", result.stderr)
            self.assertIn("is reused by answer_id", result.stderr)
            self.assertFalse(output.exists())

    def test_challenge_type_requires_its_human_boolean_on_all_labels(self) -> None:
        scenarios = (
            ("unanswerable", "correct_abstention"),
            ("prompt-injection", "injection_obedience"),
        )
        for challenge_type, required_field in scenarios:
            with self.subTest(challenge_type=challenge_type):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    answer = self.answer_record("q1")
                    judgment_path = root / "judgments.jsonl"
                    human_path = root / "human-labels.jsonl"
                    self.write_jsonl(
                        judgment_path,
                        [self.judgment_record(answer, score=2, gfc=True)],
                    )
                    labels = self.human_labels(
                        answer,
                        independent_a=(2, True),
                        independent_b=(2, True),
                        adjudicated=(2, True),
                        split="holdout-challenge",
                        challenge_type=challenge_type,
                    )
                    self.write_jsonl(human_path, labels)
                    output = root / "missing-challenge-label.json"

                    result = self.run_cli(
                        judgments=[judgment_path],
                        human_labels=human_path,
                        json_out=output,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        f"requires {required_field} on all three human labels",
                        result.stderr,
                    )
                    self.assertFalse(output.exists())

    def test_scope_ambiguity_allows_null_challenge_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer = self.answer_record("q1")
            judgment_path = root / "judgments.jsonl"
            human_path = root / "human-labels.jsonl"
            self.write_jsonl(
                judgment_path,
                [self.judgment_record(answer, score=1, gfc=False)],
            )
            labels = self.human_labels(
                answer,
                independent_a=(1, False),
                independent_b=(1, False),
                adjudicated=(1, False),
                split="holdout-challenge",
                challenge_type="scope-version-ambiguity",
            )
            self.write_jsonl(human_path, labels)
            output = root / "scope-ambiguity.json"

            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(output.read_text(encoding="utf-8"))["cases"][0]
            self.assertIsNone(row["challenge_labels"]["correct_abstention"])
            self.assertIsNone(row["challenge_labels"]["injection_obedience"])

    def test_unknown_missing_and_human_error_records_fail_closed(self) -> None:
        scenarios = (
            (
                "unknown-field",
                lambda record: record.update({"grounded_fully_correct_typo": True}),
                "unknown fields: grounded_fully_correct_typo",
            ),
            (
                "missing-field",
                lambda record: record.pop("uncertain"),
                "missing required fields: uncertain",
            ),
            (
                "human-error",
                lambda record: record.update({"error": "review export failed"}),
                "human label has error: review export failed",
            ),
        )
        for name, mutate, expected_error in scenarios:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    rows = [("q1", (2, True), (2, True), (2, True), (2, True))]
                    judgment_path, human_path, _, labels = self.make_artifacts(
                        root, rows
                    )
                    changed = [dict(record) for record in labels]
                    mutate(changed[0])
                    self.write_jsonl(human_path, changed)
                    output = root / f"{name}.json"

                    result = self.run_cli(
                        judgments=[judgment_path],
                        human_labels=human_path,
                        json_out=output,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected_error, result.stderr)
                    self.assertFalse(output.exists())

    def test_unfilled_review_packet_template_fails_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [("q1", (2, True), (2, True), (2, True), (2, True))]
            judgment_path, human_path, _, labels = self.make_artifacts(root, rows)
            unfilled = [dict(record) for record in labels]
            for record in unfilled:
                record.update(
                    {
                        "reviewer_id": None,
                        "score": None,
                        "grounded_fully_correct": None,
                        "uncertain": None,
                    }
                )
            self.write_jsonl(human_path, unfilled)
            output = root / "unfilled-packet.json"

            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=output,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reviewer_id must be a non-empty string", result.stderr)
            self.assertFalse(output.exists())

    def test_error_records_case_mismatch_and_unfilled_placeholders_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [("q1", (2, True), (2, True), (2, True), (2, True))]
            judgment_path, human_path, judgments, labels = self.make_artifacts(
                root, rows
            )

            answer = self.answer_record("q1")
            self.write_jsonl(
                judgment_path,
                [self.judgment_record(answer, score=2, gfc=True, error="timeout")],
            )
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=root / "judge-error.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has error: timeout", result.stderr)

            self.write_jsonl(judgment_path, judgments)
            mismatched = [dict(record) for record in labels]
            mismatched[0]["case_id"] = "wrong-case"
            self.write_jsonl(human_path, mismatched)
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=root / "case-mismatch.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("case_id mismatch", result.stderr)

            placeholder = [dict(record) for record in labels]
            placeholder[0]["score"] = None
            self.write_jsonl(human_path, placeholder)
            result = self.run_cli(
                judgments=[judgment_path],
                human_labels=human_path,
                json_out=root / "placeholder.json",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("score must be integer 0, 1, or 2", result.stderr)

    def test_help_documents_required_inputs_without_reading_real_v8(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--judgments", result.stdout)
        self.assertIn("--human-labels", result.stdout)
        self.assertIn(
            "pnu.human-answer-label.v1", "".join(result.stdout.split())
        )


if __name__ == "__main__":
    unittest.main()
