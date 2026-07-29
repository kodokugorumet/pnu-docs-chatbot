from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bm25_search import build_dense_index, build_index, search_bm25_candidates
from rag.corpus import inspect_corpus
from rag.retrieval import DenseIndex


def jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CorpusGateTests(unittest.TestCase):
    def make_run(
        self,
        root: Path,
        *,
        decision: str = "pass",
        run_id: str = "run-1",
        institution: str = "부산대학교",
    ) -> Path:
        run_dir = root / run_id / "cascade"
        run_dir.mkdir(parents=True)
        document_id = "doc-1" if run_id == "run-1" else f"{run_id}-doc"
        chunk_id = "chunk-1" if run_id == "run-1" else f"{run_id}-chunk"
        documents = [
            {
                "document_id": document_id,
                "status": "parsed",
                "quality": {"decision": decision},
            }
        ]
        blocks = [
            {
                "document_id": document_id,
                "block_id": "table-parent",
                "block_type": "table",
                "text": "구분 등록금",
                "reading_order": 0,
                "parser": "test",
                "page": None,
                "section_path": ["제1장", "등록"],
                "table_id": "table-1",
                "row": None,
                "column": None,
            },
            {
                "document_id": document_id,
                "block_id": "cell-0-0",
                "block_type": "table_cell",
                "text": "구분",
                "reading_order": 1,
                "parser": "test",
                "page": None,
                "section_path": ["제1장", "등록"],
                "table_id": "table-1",
                "row": 0,
                "column": 0,
            },
            {
                "document_id": document_id,
                "block_id": "cell-0-1",
                "block_type": "table_cell",
                "text": "등록금",
                "reading_order": 2,
                "parser": "test",
                "page": None,
                "section_path": ["제1장", "등록"],
                "table_id": "table-1",
                "row": 0,
                "column": 1,
            },
        ]
        chunks = [
            {
                "chunk_id": chunk_id,
                "doc_id": document_id,
                "chunk_index": 0,
                "text": "구분 등록금 납부 안내",
                "char_count": 12,
                "metadata": {
                    "institution": institution,
                    "file_name": "등록금.hwp",
                    "relative_path": "부산대학교/등록금.hwp",
                    "source_path": "src/data/부산대학교/등록금.hwp",
                    "extension": ".hwp",
                    "parser": "test",
                    "source_title": "효원학사 등록금 납부 안내",
                    "source_url": "https://www.pusan.ac.kr/notice/1",
                    "download_url": "https://www.pusan.ac.kr/download/1",
                    "source_host": "www.pusan.ac.kr",
                    "fetched_at": "2026-07-25T01:00:00+00:00",
                    "published_at": None,
                    "category": "registration",
                    "include_reason": "core_host",
                    "crawl_storage_path": "content/부산대학교/등록금.hwp",
                    "source_aliases": [
                        "https://www.pusan.ac.kr/notice/1?alias=1"
                    ],
                    "page_start": None,
                    "page_end": None,
                    "section_path": ["제1장", "등록"],
                    "table_ids": ["table-1"],
                    "block_ids": ["table-parent"],
                },
            },
            {
                "chunk_id": "empty-chunk",
                "doc_id": document_id,
                "chunk_index": 1,
                "text": "   ",
                "char_count": 3,
                "metadata": {"block_ids": [], "table_ids": []},
            },
        ]

        primary = {
            "documents.jsonl": jsonl(documents),
            "blocks.jsonl": jsonl(blocks),
            "chunks.jsonl": jsonl(chunks),
            "attempts.jsonl": "",
            "parse_report.csv": "document_id,status\n",
            "parse_summary.json": "{}\n",
        }
        for name, contents in primary.items():
            (run_dir / name).write_text(contents, encoding="utf-8")

        manifest = {
            "run_id": run_id,
            "profile": "cascade",
            "block_schema_version": 1,
            "source_manifest_sha256": hashlib.sha256(
                f"source:{run_id}".encode("utf-8")
            ).hexdigest(),
            "selection_counts": {
                "manifest_entries": 1,
                "selected_files": 1,
            },
            "files": {name: sha256(run_dir / name) for name in primary},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return run_dir

    def test_verified_run_keeps_only_explicit_block_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run(Path(tmp))
            report = inspect_corpus(run_dir / "chunks.jsonl")
            chunk = next(
                json.loads(line)
                for line in (run_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                if "chunk-1" in line
            )

            locations = report.locations_for_chunk(chunk)
            explicit_cell = report.locations_for_chunk(
                {
                    "doc_id": "doc-1",
                    "metadata": {"block_ids": ["cell-0-1"]},
                }
            )

        self.assertTrue(report.is_verified_run)
        self.assertEqual(
            report.source_manifest_sha256,
            hashlib.sha256(b"source:run-1").hexdigest(),
        )
        self.assertEqual(
            report.selection_counts,
            {"manifest_entries": 1, "selected_files": 1},
        )
        self.assertIn("empty-chunk", report.excluded_chunk_ids)
        self.assertEqual(
            [item["block_id"] for item in locations],
            ["table-parent"],
        )
        self.assertEqual(locations[0]["table_id"], "table-1")
        self.assertEqual(
            (explicit_cell[0]["row"], explicit_cell[0]["column"]),
            (0, 1),
        )

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run(Path(tmp))
            with (run_dir / "chunks.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")

            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                inspect_corpus(run_dir / "chunks.jsonl")

    def test_suspect_documents_are_quarantined_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run(Path(tmp), decision="suspect")
            with self.assertRaisesRegex(RuntimeError, "rejected every document"):
                inspect_corpus(run_dir / "chunks.jsonl")

            report = inspect_corpus(run_dir / "chunks.jsonl", allow_suspect=True)

        self.assertEqual(report.allowed_document_ids, frozenset({"doc-1"}))

    def test_bm25_index_and_results_keep_corpus_revision_and_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run(Path(tmp))
            index_path = Path(tmp) / "bm25.sqlite"

            summary = build_index(
                run_dir / "chunks.jsonl",
                index_path,
                batch_size=10,
                require_manifest=True,
            )
            readonly_uri = "file:{}?mode=ro".format(index_path.resolve())
            readonly = sqlite3.connect(readonly_uri, uri=True)
            journal_mode = readonly.execute("PRAGMA journal_mode").fetchone()[0]
            readonly.close()
            results = search_bm25_candidates(
                index_path,
                "등록금",
                5,
                "부산대학교",
                include_text=True,
            )
            connection = sqlite3.connect(index_path)
            stored_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            metadata = dict(
                connection.execute("SELECT key, value FROM index_meta")
            )
            connection.close()

        self.assertTrue(summary["verified_run"])
        self.assertEqual(journal_mode, "delete")
        self.assertEqual(
            summary["source_manifest_sha256"],
            hashlib.sha256(b"source:run-1").hexdigest(),
        )
        self.assertEqual(
            metadata["source_manifest_sha256"],
            summary["source_manifest_sha256"],
        )
        self.assertEqual(
            json.loads(metadata["selection_counts"]),
            {"manifest_entries": 1, "selected_files": 1},
        )
        self.assertEqual(stored_count, 1)
        self.assertEqual(results[0]["document_id"], "doc-1")
        self.assertEqual(results[0]["corpus_revision"], summary["corpus_revision"])
        self.assertEqual(len(results[0]["locations"]), 1)
        self.assertEqual(results[0]["locations"][0]["table_id"], "table-1")
        self.assertEqual(results[0]["location"]["section_path"], ["제1장", "등록"])
        self.assertEqual(
            results[0]["source_url"],
            "https://www.pusan.ac.kr/notice/1",
        )
        self.assertEqual(
            results[0]["source_title"],
            "효원학사 등록금 납부 안내",
        )
        self.assertEqual(
            results[0]["source_aliases"],
            ["https://www.pusan.ac.kr/notice/1?alias=1"],
        )

    def test_bm25_indexes_source_title_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run(Path(tmp))
            index_path = Path(tmp) / "bm25.sqlite"
            build_index(
                run_dir / "chunks.jsonl",
                index_path,
                batch_size=10,
                require_manifest=True,
            )

            results = search_bm25_candidates(
                index_path,
                "효원학사",
                5,
                "부산대학교",
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_id"], "doc-1")

    def test_multi_source_index_is_verified_and_preserves_source_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pnu = self.make_run(
                root,
                run_id="run-pnu",
                institution="부산대학교",
            )
            kisa = self.make_run(
                root,
                run_id="run-kisa",
                institution="한국인터넷진흥원",
            )
            index_path = root / "bm25.sqlite"

            summary = build_index(
                [pnu / "chunks.jsonl", kisa / "chunks.jsonl"],
                index_path,
                batch_size=10,
                require_manifest=True,
            )
            connection = sqlite3.connect(index_path)
            revisions = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT corpus_revision FROM chunks"
                )
            }
            metadata = dict(
                connection.execute("SELECT key, value FROM index_meta")
            )
            connection.close()

        self.assertTrue(summary["verified_run"])
        self.assertEqual(summary["source_count"], 2)
        self.assertEqual(summary["chunk_count"], 2)
        self.assertEqual(set(summary["institutions"]), {
            "부산대학교",
            "한국인터넷진흥원",
        })
        self.assertEqual(len(revisions), 2)
        self.assertTrue(summary["corpus_revision"].startswith("collection:"))
        self.assertEqual(metadata["source_count"], "2")
        self.assertIsNone(summary["source_manifest_sha256"])
        self.assertEqual(
            len(json.loads(metadata["source_manifest_sha256s"])),
            2,
        )

    def test_invalid_curated_source_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self.make_run(Path(tmp))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_manifest_sha256"] = "not-a-digest"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "source_manifest_sha256",
            ):
                inspect_corpus(run_dir / "chunks.jsonl")

    def test_multi_source_index_rejects_duplicate_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self.make_run(root)

            with self.assertRaisesRegex(RuntimeError, "Duplicate chunk_id"):
                build_index(
                    [run_dir / "chunks.jsonl", run_dir / "chunks.jsonl"],
                    root / "bm25.sqlite",
                    batch_size=10,
                    require_manifest=True,
                )

    def test_dense_index_keeps_revision_and_exact_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = self.make_run(root)
            bm25_path = root / "bm25.sqlite"
            dense_path = root / "dense.sqlite"
            bm25 = build_index(
                run_dir / "chunks.jsonl",
                bm25_path,
                batch_size=10,
                require_manifest=True,
            )

            dense_summary = build_dense_index(
                bm25_path,
                dense_path,
                dimensions=32,
            )
            dense = DenseIndex.load(dense_path)
            hits = dense.search("등록금 납부", top_k=1)

        self.assertEqual(dense_summary["corpus_revision"], bm25["corpus_revision"])
        self.assertEqual(dense.corpus_revision, bm25["corpus_revision"])
        self.assertEqual(len(hits[0].locations), 1)
        self.assertEqual(hits[0].locations[0].table_id, "table-1")
        self.assertEqual(
            hits[0].source_url,
            "https://www.pusan.ac.kr/notice/1",
        )
        self.assertEqual(
            hits[0].source_title,
            "효원학사 등록금 납부 안내",
        )


if __name__ == "__main__":
    unittest.main()
