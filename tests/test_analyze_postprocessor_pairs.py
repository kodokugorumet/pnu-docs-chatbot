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
SCRIPT = ROOT / "scripts" / "analyze_postprocessor_pairs.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


artifacts = load_module(
    "service_eval_artifacts_for_postprocessor_pair_test",
    ROOT / "scripts" / "service_eval_artifacts.py",
)
projector = load_module(
    "project_raw_draft_answers_for_pair_test",
    ROOT / "scripts" / "project_raw_draft_answers.py",
)


class AnalyzePostprocessorPairsTests(unittest.TestCase):
    def source_answer(self, case_id: str) -> dict:
        if case_id == "q-retained-and-deleted":
            draft = (
                "본등록 기간은 2026년 8월 24일부터 8월 27일까지입니다.\n"
                "근거에 없는 신청 수수료는 10,000원입니다."
            )
            query = "본등록 기간과 신청 수수료를 알려주세요."
            context_text = (
                "본등록 기간은 2026년 8월 24일부터 8월 27일까지입니다."
            )
        elif case_id == "q-standard-refusal":
            draft = "제공된 검색 근거에서 질문에 답할 내용을 확인할 수 없습니다."
            query = "확인할 수 없는 내용을 알려주세요."
            context_text = "이 문서는 교내 시설 이용 안내입니다."
        elif case_id == "q-format-only":
            draft = "장학금 신청은 학생지원시스템에서 할 수 있습니다."
            query = "장학금 신청 경로를 알려주세요."
            context_text = "장학금 신청은 학생지원시스템에서 할 수 있습니다."
        else:
            raise AssertionError(f"unknown fixture case {case_id}")

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
            "text": context_text,
        }
        record = {
            "schema_version": artifacts.ANSWER_SCHEMA_VERSION,
            "record_type": "answer",
            "experiment_id": "source-experiment",
            "condition_id": "c1",
            "generation_run_id": "run1",
            "case_id": case_id,
            "id": case_id,
            "case_sha256": artifacts.sha256_text(case_id),
            "query": query,
            "answer": "- 저장 당시 서비스 답변입니다.",
            "cited_answer": "- 저장 당시 서비스 답변입니다. [1]",
            "claims": [],
            "citations": [],
            "postprocessing": {"input_claim_count": 1},
            "generator": "fixture-generator",
            "collector_config": collector_config,
            "collector_config_sha256": artifacts.sha256_json(collector_config),
            "retrieval": {"mode": "bm25"},
            "evaluation_trace": {
                "schema_version": 1,
                "raw_draft": draft,
                "sanitized_draft": draft,
                "retrieval_stages": {"final_contexts": [context]},
            },
        }
        return artifacts.build_answer_identity(record)

    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def projection_pair(self, root: Path) -> tuple[Path, Path]:
        source_path = root / "source.answers.jsonl"
        sources = [
            self.source_answer("q-retained-and-deleted"),
            self.source_answer("q-standard-refusal"),
            self.source_answer("q-format-only"),
        ]
        self.write_jsonl(source_path, sources)
        source_sha = projector.sha256_file(source_path)
        raw = [
            projector.project_answer(
                source,
                source_artifact_path=source_path,
                source_artifact_sha256=source_sha,
                experiment_id="postprocessor-pair",
                condition_id="raw-draft",
                projection="raw-draft",
            )
            for source in sources
        ]
        current = [
            projector.project_answer(
                source,
                source_artifact_path=source_path,
                source_artifact_sha256=source_sha,
                experiment_id="postprocessor-pair",
                condition_id="current-postprocessed",
                projection="current-postprocessed",
            )
            for source in sources
        ]
        raw_path = root / "raw.answers.jsonl"
        current_path = root / "current.answers.jsonl"
        self.write_jsonl(raw_path, raw)
        self.write_jsonl(current_path, current)
        return raw_path, current_path

    def resign(self, record: dict) -> dict:
        changed = copy.deepcopy(record)
        changed.pop("record_sha256", None)
        return artifacts.build_answer_identity(changed)

    def run_cli(
        self,
        raw_path: Path,
        current_path: Path,
        output_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPT),
                "--raw-answers",
                str(raw_path),
                "--current-answers",
                str(current_path),
                "--out",
                str(output_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_analysis_reports_deterministic_projection_effects(self) -> None:
        analyzer = load_module("analyze_postprocessor_pairs_for_test", SCRIPT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, current_path = self.projection_pair(root)

            first = analyzer.build_analysis(raw_path, current_path)
            second = analyzer.build_analysis(raw_path, current_path)

        self.assertEqual(
            analyzer.render_json(first), analyzer.render_json(second)
        )
        self.assertEqual(first["schema_version"], "pnu.postprocessor-pair-analysis.v2")
        self.assertEqual(first["summary"]["n"], 3)
        self.assertEqual(first["summary"]["answer_bytes_changed"], 2)
        self.assertEqual(first["summary"]["answer_bytes_tied"], 1)
        self.assertEqual(first["summary"]["content_units_changed"], 1)
        self.assertEqual(first["summary"]["content_units_tied"], 2)
        self.assertNotIn("answer_changed", first["summary"])
        self.assertEqual(first["summary"]["raw_claim_counts"]["total"], 4)
        self.assertEqual(first["summary"]["final_claim_counts"]["total"], 3)
        self.assertEqual(first["summary"]["sentence_units"]["kept"], 3)
        self.assertEqual(first["summary"]["sentence_units"]["deleted"], 1)
        self.assertEqual(first["summary"]["sentence_units"]["added"], 0)
        self.assertEqual(first["summary"]["final_standard_refusal_count"], 1)
        self.assertEqual(
            first["summary"]["postprocessing_rejection_reason_counts"],
            {"critical_value_mismatch": 1, "model_abstention": 1},
        )
        self.assertIn("does not measure factuality", first["warning"])
        self.assertFalse(first["integrity_verification"]["external_llm_called"])
        self.assertTrue(
            first["integrity_verification"]["source_provenance_exact"]
        )

        by_id = {row["case_id"]: row for row in first["per_case"]}
        changed = by_id["q-retained-and-deleted"]
        self.assertTrue(changed["answer_bytes_changed"])
        self.assertTrue(changed["content_units_changed"])
        self.assertEqual(changed["raw"]["claim_count"], 2)
        self.assertEqual(changed["final"]["claim_count"], 1)
        self.assertEqual(changed["sentence_projection"]["kept_count"], 1)
        self.assertEqual(changed["sentence_projection"]["deleted_count"], 1)
        self.assertEqual(
            changed["critical_values"]["lost"], ["amount_krw:10000"]
        )
        self.assertEqual(changed["critical_values"]["added"], [])
        self.assertFalse(changed["final"]["standard_refusal"])

        refusal = by_id["q-standard-refusal"]
        self.assertFalse(refusal["answer_bytes_changed"])
        self.assertFalse(refusal["content_units_changed"])
        self.assertTrue(refusal["final"]["standard_refusal"])
        self.assertEqual(refusal["final"]["standard_refusal_type"], "model_abstention")

        format_only = by_id["q-format-only"]
        self.assertTrue(format_only["answer_bytes_changed"])
        self.assertFalse(format_only["content_units_changed"])
        self.assertEqual(format_only["sentence_projection"]["deleted_count"], 0)
        self.assertEqual(format_only["sentence_projection"]["added_count"], 0)
        self.assertNotIn("answer_changed", format_only)

    def test_cli_publishes_immutable_json_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, current_path = self.projection_pair(root)
            output = root / "analysis.json"

            first = self.run_cli(raw_path, current_path, output)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            payload = json.loads(original)
            self.assertEqual(payload["summary"]["n"], 3)
            self.assertIn("content_units_changed=1", first.stdout)
            self.assertIn("external_llm_calls=false", first.stdout)

            second = self.run_cli(raw_path, current_path, output)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("immutable output path already exists", second.stderr)
            self.assertEqual(output.read_bytes(), original)

            alias = self.run_cli(raw_path, current_path, raw_path)
            self.assertNotEqual(alias.returncode, 0)
            self.assertIn("must not overwrite or alias", alias.stderr)

    def test_answer_integrity_and_case_set_mismatch_fail_closed(self) -> None:
        analyzer = load_module("analyze_postprocessor_pairs_integrity_test", SCRIPT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path, current_path = self.projection_pair(root)
            raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
            current = [
                json.loads(line) for line in current_path.read_text().splitlines()
            ]

            corrupt = copy.deepcopy(raw)
            corrupt[0]["answer"] = "tampered"
            self.write_jsonl(raw_path, corrupt)
            with self.assertRaisesRegex(ValueError, "answer_sha256 mismatch"):
                analyzer.build_analysis(raw_path, current_path)

            self.write_jsonl(raw_path, raw)
            self.write_jsonl(current_path, current[:1])
            with self.assertRaisesRegex(ValueError, "case set mismatch"):
                analyzer.build_analysis(raw_path, current_path)

    def test_source_hashes_must_match_exactly(self) -> None:
        analyzer = load_module("analyze_postprocessor_pairs_hash_test", SCRIPT)
        keys = (
            "answer_sha256",
            "sanitized_draft_sha256",
            "final_contexts_sha256",
        )
        for key in keys:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw_path, current_path = self.projection_pair(root)
                current = [
                    json.loads(line)
                    for line in current_path.read_text(encoding="utf-8").splitlines()
                ]
                current[0]["postprocessor_diagnostic"]["source"][key] = "0" * 64
                current[0] = self.resign(current[0])
                self.write_jsonl(current_path, current)

                with self.assertRaisesRegex(ValueError, "source provenance mismatch"):
                    analyzer.build_analysis(raw_path, current_path)

    def test_projection_and_score_only_gates_fail_closed(self) -> None:
        analyzer = load_module("analyze_postprocessor_pairs_gate_test", SCRIPT)
        mutations = (
            ("projection_mode", "raw-draft"),
            ("diagnostic_only", False),
            ("primary_metric", "grounded_fully_correct"),
            ("service_performance_eligible", True),
            ("grounded_fully_correct_eligible", True),
            ("citation_performance_eligible", True),
        )
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw_path, current_path = self.projection_pair(root)
                current = [
                    json.loads(line)
                    for line in current_path.read_text(encoding="utf-8").splitlines()
                ]
                current[0]["postprocessor_diagnostic"][key] = value
                current[0] = self.resign(current[0])
                self.write_jsonl(current_path, current)

                with self.assertRaisesRegex(ValueError, "diagnostic projection"):
                    analyzer.build_analysis(raw_path, current_path)


if __name__ == "__main__":
    unittest.main()
