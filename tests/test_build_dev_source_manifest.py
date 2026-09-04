from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_dev_source_manifest",
    ROOT / "scripts" / "build_dev_source_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
manifest_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_module)


class BuildDevSourceManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        raw = root / "downloads" / "notice.pdf"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"frozen source bytes")
        index = root / "index.sqlite"
        connection = sqlite3.connect(index)
        connection.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                source_path TEXT,
                source_title TEXT,
                source_url TEXT,
                corpus_revision TEXT
            )
            """
        )
        for chunk_id in ("doc-a:cascade#0000", "doc-a:cascade#0001"):
            connection.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    "doc-a",
                    "downloads/notice.pdf",
                    "2026학년도 등록 안내.pdf",
                    "https://www.pusan.ac.kr/notice/1#top",
                    "revision-a",
                ),
            )
        connection.commit()
        connection.close()
        return raw, index

    def test_build_manifest_resolves_and_deduplicates_gold_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, index = self._fixture(root)
            cases = [
                {
                    "id": "dev-1",
                    "evidence": [{"chunk_id": "doc-a:cascade#0000"}],
                },
                {
                    "id": "dev-2",
                    "evidence": [{"chunk_id": "doc-a:cascade#0001"}],
                },
            ]

            result = manifest_module.build_manifest(
                cases,
                index_path=index,
                repo_root=root,
                cases_sha256="a" * 64,
            )

            self.assertEqual(result["document_count"], 1)
            self.assertEqual(result["corpus_revisions"], ["revision-a"])
            document = result["documents"][0]
            self.assertEqual(document["case_ids"], ["dev-1", "dev-2"])
            self.assertEqual(
                document["evidence_chunk_ids"],
                ["doc-a:cascade#0000", "doc-a:cascade#0001"],
            )
            self.assertEqual(
                document["source_sha256"], manifest_module.sha256_file(raw)
            )
            self.assertEqual(
                document["normalized_title_without_year"], "등록 안내"
            )
            self.assertEqual(
                document["source_url"], "https://www.pusan.ac.kr/notice/1"
            )

    def test_missing_gold_chunk_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, index = self._fixture(root)

            with self.assertRaisesRegex(ValueError, "resolved to 0 corpus rows"):
                manifest_module.build_manifest(
                    [
                        {
                            "id": "dev-1",
                            "evidence": [{"chunk_id": "missing#0000"}],
                        }
                    ],
                    index_path=index,
                    repo_root=root,
                )


if __name__ == "__main__":
    unittest.main()
