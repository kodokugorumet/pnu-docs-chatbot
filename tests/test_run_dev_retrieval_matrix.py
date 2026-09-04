from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_dev_retrieval_matrix",
    ROOT / "scripts" / "run_dev_retrieval_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


class RunDevRetrievalMatrixTests(unittest.TestCase):
    def test_protocol_matrix_has_exact_eight_nonfactorial_conditions(self) -> None:
        self.assertEqual(
            [condition.id for condition in matrix.CONDITIONS],
            ["PAR-B", "PAR-CH", "C0", "C1", "D-K", "H-K", "D-S", "H-S"],
        )
        by_id = matrix.CONDITIONS_BY_ID
        self.assertTrue(by_id["PAR-B"].service_tuning)
        self.assertTrue(by_id["PAR-CH"].service_tuning)
        self.assertTrue(by_id["C1"].service_tuning)
        self.assertFalse(by_id["C0"].service_tuning)
        for condition_id in ("D-K", "H-K", "D-S", "H-S"):
            self.assertEqual(by_id[condition_id].parser_profile, "cascade")
            self.assertFalse(by_id[condition_id].service_tuning)

    def test_collector_command_is_external_llm_free_and_pins_top_eight(self) -> None:
        condition = matrix.CONDITIONS_BY_ID["H-K"]
        command = matrix.build_collector_command(
            condition,
            python_executable="python-test",
            cases_path=Path("cases.jsonl"),
            output_path=Path("h_k.answers.jsonl"),
            experiment_id="dev-matrix",
            api_base="http://127.0.0.1:8120",
            corpus_revision="cascade-revision",
        )

        self.assertIn("extractive", command)
        self.assertEqual(command[command.index("--context-k") + 1], "8")
        self.assertEqual(command[command.index("--retrieval-mode") + 1], "kure_hybrid")
        self.assertEqual(
            command[command.index("--expected-retrieval-tuning") + 1], "off"
        )
        self.assertIn("--eval-trace", command)
        self.assertNotIn("gemini", command)
        self.assertNotIn("frontier", command)

    def test_server_commands_separate_tuned_and_untuned_processes(self) -> None:
        indexes = {
            "baseline": Path("baseline.sqlite"),
            "challenger": Path("challenger.sqlite"),
            "cascade": Path("cascade.sqlite"),
        }
        tuned = matrix.build_server_command(
            python_executable="python-test",
            port=8120,
            indexes=indexes,
            learned_root=Path("dense"),
            service_tuning=True,
        )
        untuned = matrix.build_server_command(
            python_executable="python-test",
            port=8120,
            indexes=indexes,
            learned_root=Path("dense"),
            service_tuning=False,
        )

        self.assertNotIn("--no-retrieval-tuning", tuned)
        self.assertIn("--no-retrieval-tuning", untuned)
        self.assertIn("--context-chunks-per-document", tuned)
        self.assertIn("--learned-dense-root", untuned)

    def test_record_validation_rejects_external_generation(self) -> None:
        condition = matrix.CONDITIONS_BY_ID["C1"]
        record = self._record(condition)
        record["response_config"]["generation_used"] = "frontier"

        with self.assertRaisesRegex(matrix.MatrixError, "external/non-extractive"):
            matrix._validate_record_controls(record, condition)

    def test_compact_projection_keeps_hashes_but_drops_repeated_text(self) -> None:
        condition = matrix.CONDITIONS_BY_ID["C0"]
        record = self._record(condition)
        record.update(
            {
                "answer_id": "answer-1",
                "record_sha256": "record-sha",
                "case_id": "q1",
                "sources": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "source_title": "정답",
                        "preview": "very large preview",
                        "locations": [{"page": 1}],
                    }
                ],
            }
        )

        compact = matrix.compact_record(
            record,
            condition,
            source_artifact_sha256="source-sha",
        )

        rendered = json.dumps(compact, ensure_ascii=False)
        self.assertNotIn("very large preview", rendered)
        self.assertNotIn('"text":', rendered)
        self.assertIn("text-sha", rendered)
        self.assertEqual(compact["source_record_sha256"], "record-sha")
        self.assertEqual(len(compact["compact_record_sha256"]), 64)

    def test_summary_hides_cascade_chunk_gold_for_parser_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.jsonl"
            cases = [
                {
                    "id": "q1",
                    "query": "질문",
                    "expected": {"source_title_contains": "정답"},
                    "evidence": [{"chunk_id": "cascade:gold"}],
                }
            ]
            cases_path.write_text(
                "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in cases),
                encoding="utf-8",
            )
            for condition_id in ("PAR-B", "C1"):
                condition = matrix.CONDITIONS_BY_ID[condition_id]
                record = self._record(condition)
                record.update(
                    {
                        "case_id": "q1",
                        "id": "q1",
                        "latency_ms": 10.0,
                        "answer": "답",
                        "sources": [
                            {
                                "document_id": "doc-1",
                                "chunk_id": (
                                    "cascade:gold"
                                    if condition.parser_profile == "cascade"
                                    else "baseline:gold"
                                ),
                                "source_title": "정답 문서",
                                "source_host": "www.pusan.ac.kr",
                            }
                        ],
                    }
                )
                (root / condition.artifact_name).write_text(
                    json.dumps(record, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            summary, rows = matrix.summarize_outputs(
                cases_path=cases_path,
                output_dir=root,
            )
            reports = {row["id"]: row for row in summary["conditions"]}

            self.assertFalse(reports["PAR-B"]["gold_chunk"]["available"])
            self.assertEqual(reports["C1"]["gold_chunk"]["8"]["any_rate"], 1.0)
            csv_by_id = {row["condition"]: row for row in rows}
            self.assertIsNone(csv_by_id["PAR-B"]["any_gold_chunk_at_8"])
            self.assertEqual(csv_by_id["C1"]["any_gold_chunk_at_8"], 1.0)

    @staticmethod
    def _record(condition):
        final = [
            {
                "rank": 1,
                "stage": "final_contexts",
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "근거",
            }
        ]
        stages = {
            name: [dict(final[0], stage=name)]
            for name in (
                "raw_bm25",
                "post_retrieval_pool",
                "post_neighbor_expansion",
                "final_contexts",
            )
        }
        for entries in stages.values():
            entries[0]["text_sha256"] = "text-sha"
        return {
            "condition_id": condition.id,
            "collector_config": {
                "provider": "extractive",
                "context_k": 8,
                "parser_profile": condition.parser_profile,
                "retrieval_mode": condition.retrieval_mode,
                "expected_retrieval_tuning": condition.service_tuning,
                "expected_context_chunks_per_document": 2,
                "eval_trace": True,
            },
            "response_config": {
                "parser_profile": condition.parser_profile,
                "retrieval_mode": condition.retrieval_mode,
                "generation_used": "extractive",
            },
            "evaluation_trace": {
                "schema_version": 1,
                "retrieval_stages": stages,
            },
            "sources": [{}],
        }


if __name__ == "__main__":
    unittest.main()
