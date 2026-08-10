from __future__ import annotations

import csv
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bm25_search import ensure_schema, make_search_text
from evaluate_pnu_retrieval import (
    EvaluationError,
    SOURCE_MANIFEST_META_KEY,
    compare_indexes,
    evaluate_index,
    load_cases,
    main,
    result_matches_expected,
    validate_cases,
)

TEST_MANIFEST_SHA256 = "a" * 64


def unmatched_case_ids_for_manifest(
    cases: List[Dict[str, Any]],
    manifest_path: Path,
) -> List[str]:
    """Cross-check metadata gold against any supplied curated manifest.

    Parser output renames ``input_relative_path`` to ``relative_path``.  The
    conversion below mirrors that boundary so this helper checks the same
    fields that the evaluator will see without depending on a generated
    manifest in the test suite.
    """

    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    searchable_rows = [
        {
            "document_id": row.get("document_id"),
            "source_host": row.get("source_host"),
            "category": row.get("category"),
            "source_title": row.get("source_title"),
            "source_path": row.get("source_path"),
            "relative_path": row.get("input_relative_path"),
            "metadata": row,
        }
        for row in manifest_rows
    ]
    return [
        case["id"]
        for case in cases
        if not any(
            result_matches_expected(row, case["expected"])
            for row in searchable_rows
        )
    ]


class PnuRetrievalEvaluationTests(unittest.TestCase):
    def make_index(
        self,
        path: Path,
        *,
        source_manifest_sha256: Optional[str] = TEST_MANIFEST_SHA256,
    ) -> None:
        connection = sqlite3.connect(str(path))
        ensure_schema(connection)
        columns = [
            "chunk_id",
            "doc_id",
            "document_id",
            "chunk_index",
            "institution",
            "source_path",
            "relative_path",
            "file_name",
            "extension",
            "parser",
            "source_title",
            "source_url",
            "download_url",
            "source_host",
            "fetched_at",
            "published_at",
            "category",
            "include_reason",
            "crawl_storage_path",
            "source_aliases_json",
            "char_count",
            "page_start",
            "page_end",
            "section_path_json",
            "table_ids_json",
            "block_ids_json",
            "locations_json",
            "corpus_revision",
            "text",
        ]
        chunks: List[Dict[str, Any]] = [
            {
                "chunk_id": "academic-chunk",
                "doc_id": "academic-doc",
                "document_id": "academic-doc",
                "chunk_index": 0,
                "institution": "부산대학교",
                "source_path": "src/data/부산대학교/휴학복학.pdf",
                "relative_path": "부산대학교/휴학복학.pdf",
                "file_name": "휴학복학.pdf",
                "extension": ".pdf",
                "parser": "synthetic",
                "source_title": "부산대학교 휴학 및 복학 신청 안내",
                "source_url": "https://www.pusan.ac.kr/notice/academic",
                "download_url": "https://www.pusan.ac.kr/download/academic",
                "source_host": "www.pusan.ac.kr",
                "fetched_at": "2026-07-25T00:00:00+00:00",
                "published_at": None,
                "category": "academic",
                "include_reason": "test",
                "crawl_storage_path": "content/부산대학교/휴학복학.pdf",
                "source_aliases_json": "[]",
                "char_count": 24,
                "page_start": 1,
                "page_end": 1,
                "section_path_json": "null",
                "table_ids_json": "[]",
                "block_ids_json": "[]",
                "locations_json": "[]",
                "corpus_revision": "synthetic-v1",
                "text": "휴학 신청과 복학 신청 절차 및 제출 서류 안내",
            },
            {
                "chunk_id": "scholarship-chunk",
                "doc_id": "scholarship-doc",
                "document_id": "scholarship-doc",
                "chunk_index": 0,
                "institution": "부산대학교",
                "source_path": "src/data/부산대학교/장학금.pdf",
                "relative_path": "부산대학교/장학금.pdf",
                "file_name": "장학금.pdf",
                "extension": ".pdf",
                "parser": "synthetic",
                "source_title": "국가근로장학금 학생 신청 안내",
                "source_url": "https://www.pusan.ac.kr/notice/scholarship",
                "download_url": "https://www.pusan.ac.kr/download/scholarship",
                "source_host": "www.pusan.ac.kr",
                "fetched_at": "2026-07-25T00:00:00+00:00",
                "published_at": None,
                "category": "scholarship",
                "include_reason": "test",
                "crawl_storage_path": "content/부산대학교/장학금.pdf",
                "source_aliases_json": "[]",
                "char_count": 22,
                "page_start": 1,
                "page_end": 1,
                "section_path_json": "null",
                "table_ids_json": "[]",
                "block_ids_json": "[]",
                "locations_json": "[]",
                "corpus_revision": "synthetic-v1",
                "text": "국가근로장학금 신청 기간과 학생 신청 방법",
            },
        ]
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            "INSERT INTO chunks ({}) VALUES ({})".format(
                ", ".join(columns),
                placeholders,
            ),
            [[chunk[column] for column in columns] for chunk in chunks],
        )
        for chunk in chunks:
            search_chunk = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "institution": chunk["institution"],
                    "file_name": chunk["file_name"],
                    "relative_path": chunk["relative_path"],
                    "extension": chunk["extension"],
                    "source_title": chunk["source_title"],
                    "source_host": chunk["source_host"],
                    "category": chunk["category"],
                },
            }
            connection.execute(
                "INSERT INTO chunk_fts (chunk_id, search_text) VALUES (?, ?)",
                (chunk["chunk_id"], make_search_text(search_chunk)),
            )
        if source_manifest_sha256 is not None:
            connection.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?)",
                (SOURCE_MANIFEST_META_KEY, source_manifest_sha256),
            )
        connection.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            ("chunk_count", str(len(chunks))),
        )
        connection.commit()
        connection.close()

    def synthetic_cases(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "academic_hit",
                "query": "휴학 신청",
                "expected": {
                    "source_host": "www.pusan.ac.kr",
                    "category": "academic",
                    "source_title_contains": "휴학 및 복학",
                },
                "k": 2,
                "requires_validation": False,
            },
            {
                "id": "known_miss",
                "query": "국가근로장학금 신청",
                "expected": {"document_id": "not-the-scholarship-document"},
                "k": 2,
                "requires_validation": False,
            },
            {
                "id": "awaiting_gold_review",
                "query": "검증 대기 질의",
                "expected": {"source_host": "www.pusan.ac.kr"},
                "k": 2,
                "requires_validation": True,
            },
        ]

    def test_shipped_suite_is_24_grounded_schema_valid_cases(self) -> None:
        cases = load_cases(
            ROOT / "config" / "pnu-retrieval-eval.jsonl",
            expected_count=24,
        )

        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case["id"] for case in cases}), 24)
        self.assertTrue(all(case["note"] for case in cases))
        self.assertTrue(
            all(
                case["expected"].get("source_host", "").endswith(".pusan.ac.kr")
                for case in cases
            )
        )

    def test_manifest_gold_cross_check_helper_uses_parser_relative_path(self) -> None:
        cases = validate_cases(
            [
                {
                    "id": "title_gold",
                    "query": "등록금 납부",
                    "expected": {
                        "source_host": "www.pusan.ac.kr",
                        "category": "registration",
                        "source_title_contains": "등록금 납부 계획",
                    },
                    "k": 5,
                },
                {
                    "id": "path_gold",
                    "query": "대학원 외국인 전형",
                    "expected": {
                        "source_host": "go.pusan.ac.kr",
                        "relative_path": "부산대학교/첨부파일/go.pusan.ac.kr/guide.pdf",
                    },
                    "k": 5,
                },
            ]
        )
        manifest_rows = [
            {
                "input_relative_path": "부산대학교/첨부파일/www.pusan.ac.kr/tuition.pdf",
                "source_host": "www.pusan.ac.kr",
                "category": "registration",
                "source_title": "2026학년도 재학생 등록금 납부 계획.pdf",
            },
            {
                "input_relative_path": "부산대학교/첨부파일/go.pusan.ac.kr/guide.pdf",
                "source_host": "go.pusan.ac.kr",
                "category": "international",
                "source_title": "PDF",
            },
        ]

        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "curated-manifest.jsonl"
            manifest_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in manifest_rows
                ),
                encoding="utf-8",
            )
            missing = unmatched_case_ids_for_manifest(cases, manifest_path)

        self.assertEqual(missing, [])

    def test_schema_rejects_duplicate_ids_and_unknown_gold_fields(self) -> None:
        case = self.synthetic_cases()[0]
        with self.assertRaisesRegex(EvaluationError, "duplicate case id"):
            validate_cases([case, case])

        invalid = dict(case)
        invalid["id"] = "invalid_gold"
        invalid["expected"] = {"answer_text": "fabricated answer"}
        with self.assertRaisesRegex(EvaluationError, "unsupported field"):
            validate_cases([invalid])

    def test_metadata_match_uses_nested_metadata_and_all_criteria(self) -> None:
        row = {
            "document_id": "doc-1",
            "source_path": "src/data/doc.pdf",
            "metadata": {
                "source_host": "WWW.PUSAN.AC.KR",
                "category": "academic",
                "source_title": "  부산대학교   휴학 및 복학 안내  ",
            },
        }

        self.assertTrue(
            result_matches_expected(
                row,
                {
                    "document_id": "doc-1",
                    "source_host": "www.pusan.ac.kr",
                    "category": "ACADEMIC",
                    "source_title_contains": "휴학 및 복학",
                    "source_path": "src/data/doc.pdf",
                },
            )
        )
        self.assertFalse(
            result_matches_expected(
                row,
                {
                    "source_host": "www.pusan.ac.kr",
                    "category": "scholarship",
                },
            )
        )

    def test_hit_at_k_mrr_and_unevaluable_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "bm25.sqlite"
            self.make_index(index_path)

            result = evaluate_index(
                "baseline",
                index_path,
                self.synthetic_cases(),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metrics"]["total_cases"], 3)
        self.assertEqual(result["metrics"]["scored_cases"], 2)
        self.assertEqual(result["metrics"]["unevaluable_cases"], 1)
        self.assertEqual(result["metrics"]["hits"], 1)
        self.assertEqual(result["metrics"]["misses"], 1)
        self.assertEqual(result["metrics"]["hit_at_k"], 0.5)
        self.assertEqual(result["metrics"]["mrr"], 0.5)
        self.assertEqual(
            [case["status"] for case in result["cases"]],
            ["hit", "miss", "unevaluable"],
        )
        self.assertEqual(result["cases"][0]["rank"], 1)
        self.assertEqual(
            result["cases"][2]["reason"],
            "requires_validation",
        )
        self.assertGreaterEqual(
            result["cases"][0]["returned_chunk_count"],
            result["cases"][0]["returned_document_count"],
        )

    def test_document_ranking_survives_many_chunks_from_one_wrong_document(
        self,
    ) -> None:
        wrong_chunks = [
            {
                "chunk_id": "wrong-chunk-{}".format(index),
                "document_id": "wrong-document",
                "relative_path": "부산대학교/잘못된문서.pdf",
                "source_host": "www.pusan.ac.kr",
                "category": "academic",
                "source_title": "반복되는 잘못된 문서",
            }
            for index in range(300)
        ]
        expected_chunk = {
            "chunk_id": "expected-chunk",
            "document_id": "expected-document",
            "relative_path": "부산대학교/정답문서.pdf",
            "source_host": "www.pusan.ac.kr",
            "category": "academic",
            "source_title": "휴학 신청 정답 문서",
        }
        ranked_chunks = wrong_chunks + [expected_chunk]
        requested_limits: List[int] = []

        def crowded_search(
            index_path: Path,
            query: str,
            top_k: int,
            institution: str,
        ) -> List[Dict[str, Any]]:
            requested_limits.append(top_k)
            return ranked_chunks[:top_k]

        case = {
            "id": "document_level_rank",
            "query": "휴학 신청",
            "expected": {"document_id": "expected-document"},
            "k": 2,
            "requires_validation": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "placeholder.sqlite"
            index_path.touch()
            result = evaluate_index(
                "profile",
                index_path,
                [case],
                searcher=crowded_search,
            )

        self.assertEqual(requested_limits, [250, 500])
        self.assertEqual(result["metrics"]["hits"], 1)
        self.assertEqual(result["metrics"]["mrr"], 0.5)
        self.assertEqual(result["cases"][0]["status"], "hit")
        self.assertEqual(result["cases"][0]["rank"], 2)
        self.assertEqual(result["cases"][0]["returned_chunk_count"], 301)
        self.assertEqual(result["cases"][0]["returned_document_count"], 2)
        self.assertEqual(result["cases"][0]["returned_count"], 301)
        self.assertTrue(result["cases"][0]["candidate_pool"]["exhausted"])
        self.assertEqual(
            result["cases"][0]["candidate_pool"]["stop_reason"],
            "enough_unique_documents",
        )

    def test_unvalidated_case_does_not_call_searcher(self) -> None:
        case = {
            "id": "unvalidated_only",
            "query": "아직 검증하지 않은 질의",
            "expected": {"source_host": "www.pusan.ac.kr"},
            "k": 5,
            "requires_validation": True,
        }

        def unexpected_search(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            raise AssertionError("search must not run for unvalidated gold")

        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "placeholder.sqlite"
            index_path.touch()
            result = evaluate_index(
                "profile",
                index_path,
                [case],
                searcher=unexpected_search,
            )

        self.assertEqual(result["metrics"]["scored_cases"], 0)
        self.assertIsNone(result["metrics"]["hit_at_k"])
        self.assertIsNone(result["metrics"]["mrr"])
        self.assertIsNone(result["cases"][0]["returned_chunk_count"])
        self.assertIsNone(result["cases"][0]["returned_document_count"])

    def test_cli_writes_json_and_csv_for_multiple_named_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "bm25.sqlite"
            cases_path = root / "cases.jsonl"
            json_output = root / "comparison.json"
            csv_output = root / "comparison.csv"
            self.make_index(index_path)
            cases_path.write_text(
                "".join(
                    json.dumps(case, ensure_ascii=False) + "\n"
                    for case in self.synthetic_cases()
                ),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "evaluate",
                        "--cases",
                        str(cases_path),
                        "--index",
                        "baseline={}".format(index_path),
                        "--index",
                        "challenger={}".format(index_path),
                        "--json-output",
                        str(json_output),
                        "--csv-output",
                        str(csv_output),
                    ]
                )
            report = json.loads(json_output.read_text(encoding="utf-8"))
            with csv_output.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["ok"])
        self.assertTrue(report["compatibility"]["compatible"])
        self.assertFalse(report["compatibility"]["override_used"])
        self.assertEqual(
            report["compatibility"]["common_source_manifest_sha256"],
            TEST_MANIFEST_SHA256,
        )
        self.assertEqual(
            [item["name"] for item in report["indexes"]],
            ["baseline", "challenger"],
        )
        self.assertEqual(
            [row["index_name"] for row in csv_rows],
            ["baseline", "challenger"],
        )
        self.assertTrue(
            all(
                row["source_manifest_sha256"] == TEST_MANIFEST_SHA256
                for row in csv_rows
            )
        )
        self.assertIn('"ok": true', stdout.getvalue())

    def test_mismatched_source_manifests_block_comparison_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.sqlite"
            challenger = root / "challenger.sqlite"
            self.make_index(
                baseline,
                source_manifest_sha256="a" * 64,
            )
            self.make_index(
                challenger,
                source_manifest_sha256="b" * 64,
            )
            report = compare_indexes(
                [("baseline", baseline), ("challenger", challenger)],
                self.synthetic_cases(),
            )

        self.assertFalse(report["ok"])
        self.assertFalse(report["compatibility"]["compatible"])
        self.assertFalse(report["compatibility"]["comparison_allowed"])
        self.assertFalse(report["compatibility"]["override_used"])
        self.assertEqual(
            report["compatibility"]["reason"],
            "mismatched_source_manifest_sha256",
        )
        self.assertEqual(
            report["compatibility"]["distinct_source_manifest_sha256"],
            ["a" * 64, "b" * 64],
        )
        self.assertEqual(
            [item["status"] for item in report["indexes"]],
            ["not_evaluated", "not_evaluated"],
        )
        self.assertTrue(all(item["metrics"] is None for item in report["indexes"]))

    def test_cli_opt_out_allows_legacy_and_mixed_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            modern = root / "modern.sqlite"
            legacy = root / "legacy.sqlite"
            cases_path = root / "cases.jsonl"
            report_path = root / "comparison.json"
            self.make_index(modern)
            self.make_index(legacy, source_manifest_sha256=None)
            cases_path.write_text(
                "".join(
                    json.dumps(case, ensure_ascii=False) + "\n"
                    for case in self.synthetic_cases()
                ),
                encoding="utf-8",
            )

            blocked = compare_indexes(
                [("modern", modern), ("legacy", legacy)],
                self.synthetic_cases(),
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "evaluate",
                        "--cases",
                        str(cases_path),
                        "--index",
                        "modern={}".format(modern),
                        "--index",
                        "legacy={}".format(legacy),
                        "--allow-mixed-provenance",
                        "--json-output",
                        str(report_path),
                    ]
                )
            overridden = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertFalse(blocked["compatibility"]["comparison_allowed"])
        self.assertEqual(
            blocked["compatibility"]["reason"],
            "missing_source_manifest_sha256",
        )
        self.assertEqual(blocked["compatibility"]["legacy_indexes"], ["legacy"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(overridden["ok"])
        self.assertFalse(overridden["compatibility"]["compatible"])
        self.assertTrue(overridden["compatibility"]["comparison_allowed"])
        self.assertTrue(overridden["compatibility"]["override_used"])
        self.assertFalse(overridden["compatibility"]["enforced"])
        self.assertEqual(
            overridden["indexes"][1]["provenance"]["status"],
            "legacy",
        )
        self.assertIn('"override_used": true', stdout.getvalue())

    def test_compare_keeps_an_error_row_for_a_missing_profile_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.sqlite"
            placeholder = Path(temporary) / "placeholder.sqlite"
            placeholder.touch()
            report = compare_indexes(
                [("missing", missing), ("available", placeholder)],
                [
                    {
                        "id": "unvalidated_only",
                        "query": "대기",
                        "expected": {"source_host": "www.pusan.ac.kr"},
                        "k": 1,
                        "requires_validation": True,
                    }
                ],
                allow_mixed_provenance=True,
            )

        self.assertFalse(report["ok"])
        self.assertEqual(report["indexes"][0]["status"], "error")
        self.assertEqual(report["indexes"][1]["status"], "ok")
        self.assertIn("missing index DB", report["indexes"][0]["error"])


if __name__ == "__main__":
    unittest.main()
