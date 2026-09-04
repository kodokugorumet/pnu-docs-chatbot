from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_parser_audit import evaluate_profile
from scripts.parser_audit import (
    AuditValidationError,
    ProfileArtifact,
    find_anchor,
    normalize_anchor,
    sha256_file,
    validate_canonical_bindings,
    validate_dataset_schema,
)


SOURCE_MANIFEST_SHA = "1" * 64


class ParserAuditTests(unittest.TestCase):
    def build_fixture(
        self,
        root: Path,
        *,
        index_anchor_count: int = 3,
        profile: str = "cascade",
    ) -> dict:
        raw_path = root / "raw" / "sample.hwp"
        raw_path.parent.mkdir()
        raw_path.write_bytes(b"source-bound-parser-audit")
        source_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        document_id = "doc_" + "a" * 24
        anchors = [
            {
                "id": f"anchor-{offset}",
                "kind": kind,
                "text": text,
                "accepted_variants": [],
                "source_chunk_hint": f"{document_id}:cascade#000{offset - 1}",
            }
            for offset, (kind, text) in enumerate(
                [
                    ("date", "2026. 9. 1.(화)부터 9. 3.(목)까지"),
                    ("numeric", "지원금은 1인당 최대 50만원입니다"),
                    ("procedure", "신청서를 학과 사무실에 제출합니다"),
                ],
                start=1,
            )
        ]
        case = {
            "schema_version": 1,
            "id": "fixture-case",
            "group": "hwp_hwpx",
            "source": {
                "document_id": document_id,
                "extension": ".hwp",
                "source_path": "raw/sample.hwp",
                "relative_path": "fixture/sample.hwp",
                "source_sha256": source_sha,
                "source_title": "parser audit fixture.hwp",
            },
            "route": {
                "kind": "hwp_structure",
                "cascade_selected_parsers": ["cascade/hwplib"],
                "cascade_table_count": 1,
            },
            "anchors": anchors,
            "verification": {
                "status": "mechanically_verified",
                "canonical_profile": "cascade",
                "method": "raw-sha+run-chunk+index-chunk",
            },
        }

        run_dir = root / "run"
        run_dir.mkdir()
        document = {
            **case["source"],
            "selected_parsers": ["cascade/hwplib"],
            "table_count": 1,
        }
        (run_dir / "documents.jsonl").write_text(
            json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        chunk_records = [
            {
                "chunk_id": anchor["source_chunk_hint"].replace(
                    ":cascade#", f":{profile}#"
                ),
                "doc_id": document_id,
                "chunk_index": offset,
                "text": anchor["text"],
                "metadata": {
                    "source_path": case["source"]["source_path"],
                    "relative_path": case["source"]["relative_path"],
                    "extension": ".hwp",
                },
            }
            for offset, anchor in enumerate(anchors)
        ]
        (run_dir / "chunks.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in chunk_records
            ),
            encoding="utf-8",
        )
        manifest = {
            "profile": profile,
            "run_id": "fixture-run",
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "files": {
                name: sha256_file(run_dir / name)
                for name in ("documents.jsonl", "chunks.jsonl")
            },
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        index_path = root / "index.sqlite"
        with sqlite3.connect(index_path) as connection:
            connection.execute("CREATE TABLE index_meta (key TEXT, value TEXT)")
            connection.executemany(
                "INSERT INTO index_meta VALUES (?, ?)",
                [
                    ("profile", profile),
                    ("run_id", "fixture-run"),
                    ("source_manifest_sha256", SOURCE_MANIFEST_SHA),
                    ("chunk_count", str(index_anchor_count)),
                ],
            )
            connection.execute(
                """
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    chunk_index INTEGER,
                    source_path TEXT,
                    relative_path TEXT,
                    extension TEXT,
                    parser TEXT,
                    source_title TEXT,
                    text TEXT
                )
                """
            )
            for offset, anchor in enumerate(anchors[:index_anchor_count]):
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        anchor["source_chunk_hint"].replace(
                            ":cascade#", f":{profile}#"
                        ),
                        document_id,
                        offset,
                        case["source"]["source_path"],
                        case["source"]["relative_path"],
                        ".hwp",
                        f"{profile}/hwplib",
                        case["source"]["source_title"],
                        anchor["text"],
                    ),
                )
        return {
            "case": case,
            "artifact": ProfileArtifact(profile, run_dir, index_path),
            "raw_path": raw_path,
        }

    def test_normalized_match_tolerates_layout_punctuation_only(self) -> None:
        anchor = {
            "text": "수수료(60,000원) - 현금만 가능",
            "accepted_variants": [],
        }
        self.assertEqual(
            normalize_anchor("수수료 60,000원\n현금만 가능"),
            normalize_anchor(anchor["text"]),
        )
        self.assertIsNotNone(find_anchor("수수료 60,000원\n현금만 가능", anchor))
        self.assertIsNone(find_anchor("수수료 50,000원 현금만 가능", anchor))

    def test_schema_and_canonical_binding_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            records = [fixture["case"]]
            summary = validate_dataset_schema(
                records, Path(temporary), enforce_balance=False
            )
            binding = validate_canonical_bindings(records, fixture["artifact"])

        self.assertEqual(summary["documents"], 1)
        self.assertEqual(summary["anchors"], 3)
        self.assertEqual(binding, {"documents": 1, "anchors": 3})

    def test_raw_sha_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            case = copy.deepcopy(fixture["case"])
            case["source"]["source_sha256"] = "0" * 64
            with self.assertRaisesRegex(AuditValidationError, "SHA-256 mismatch"):
                validate_dataset_schema(
                    [case], Path(temporary), enforce_balance=False
                )

    def test_profile_evaluation_separates_run_and_index_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary), index_anchor_count=2)
            summary, rows = evaluate_profile(
                [fixture["case"]], fixture["artifact"]
            )

        self.assertEqual(summary["run_anchor_hits"], 3)
        self.assertEqual(summary["index_anchor_hits"], 2)
        self.assertEqual(summary["end_to_end_anchor_hits"], 2)
        self.assertEqual(summary["documents_all_anchors"], 0)
        self.assertFalse(rows[-1]["index_hit"])

    def test_profile_scoring_is_independent_of_cascade_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary), profile="baseline")
            summary, rows = evaluate_profile(
                [fixture["case"]], fixture["artifact"]
            )

        self.assertEqual(summary["end_to_end_anchor_hits"], 3)
        self.assertTrue(
            all(":baseline#" in str(row["index_chunk_id"]) for row in rows)
        )
        self.assertTrue(
            all(
                ":cascade#" in anchor["source_chunk_hint"]
                for anchor in fixture["case"]["anchors"]
            )
        )

    def test_index_source_identity_mismatch_blocks_end_to_end_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            with sqlite3.connect(fixture["artifact"].index_path) as connection:
                connection.execute(
                    "UPDATE chunks SET source_path = 'raw/wrong.hwp'"
                )
            summary, rows = evaluate_profile(
                [fixture["case"]], fixture["artifact"]
            )

        self.assertEqual(summary["run_anchor_hits"], 3)
        self.assertEqual(summary["index_anchor_hits"], 3)
        self.assertEqual(summary["end_to_end_anchor_hits"], 0)
        self.assertTrue(all(not row["index_source_identity_ok"] for row in rows))

    def test_canonical_binding_rejects_wrong_index_source_title(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            with sqlite3.connect(fixture["artifact"].index_path) as connection:
                connection.execute(
                    "UPDATE chunks SET source_title = 'wrong source.pdf'"
                )
            with self.assertRaisesRegex(
                AuditValidationError, "canonical index source_title mismatch"
            ):
                validate_canonical_bindings(
                    [fixture["case"]], fixture["artifact"]
                )


if __name__ == "__main__":
    unittest.main()
