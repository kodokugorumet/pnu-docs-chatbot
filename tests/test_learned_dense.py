from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bm25_search import build_index, load_chunks_by_ids
from rag.learned_dense import LearnedDenseIndex


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional learned-dense dependency")
class LearnedDenseIndexTests(unittest.TestCase):
    def test_loads_aligned_artifact_and_returns_dense_scores(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_path = root / "chunks.jsonl"
            source_index = root / "bm25.sqlite"
            artifact = root / "artifact"
            artifact.mkdir()
            rows = [
                {
                    "chunk_id": "a#0000",
                    "doc_id": "a",
                    "chunk_index": 0,
                    "text": "휴학 신청 안내",
                    "metadata": {"institution": "부산대학교"},
                },
                {
                    "chunk_id": "b#0000",
                    "doc_id": "b",
                    "chunk_index": 0,
                    "text": "장학금 신청 안내",
                    "metadata": {"institution": "부산대학교"},
                },
            ]
            chunks_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
            build_index(chunks_path, source_index, batch_size=10)
            connection = sqlite3.connect(str(source_index))
            revision = dict(
                connection.execute("SELECT key, value FROM index_meta")
            )["corpus_revision"]
            connection.close()

            np.save(
                artifact / "vectors.npy",
                np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                allow_pickle=False,
            )
            (artifact / "chunk_ids.jsonl").write_text(
                '"a#0000"\n"b#0000"\n', encoding="utf-8"
            )
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": {
                            "id": "test/model",
                            "revision": "revision",
                            "max_sequence_length": 512,
                            "prompts": {"query": "query: "},
                        },
                        "source": {"corpus_revision": revision},
                        "vectors": {
                            "file": "vectors.npy",
                            "count": 2,
                            "dimensions": 2,
                            "dtype": "float32",
                            "normalized": True,
                        },
                        "chunk_ids": {
                            "file": "chunk_ids.jsonl",
                            "count": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            index = LearnedDenseIndex(
                artifact,
                source_index=source_index,
                row_loader=lambda ids: load_chunks_by_ids(source_index, ids),
            )

            class FakeEmbedder:
                loaded = True
                device = "cpu"

                def embed(self, texts: list[str]) -> list[list[float]]:
                    return [[0.0, 1.0] for _ in texts]

            index.embedder = FakeEmbedder()
            hits = index.search("장학", top_k=1, institution="부산대학교")

            self.assertEqual(index.query_prefix, "query: ")
            self.assertEqual(hits[0].chunk_id, "b#0000")
            self.assertEqual(hits[0].retrieval.dense.score, 1.0)
            self.assertEqual(hits[0].text, "장학금 신청 안내")

    def test_rejects_corpus_revision_mismatch(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_path = root / "chunks.jsonl"
            source_index = root / "bm25.sqlite"
            artifact = root / "artifact"
            artifact.mkdir()
            chunks_path.write_text(
                json.dumps(
                    {
                        "chunk_id": "a#0000",
                        "doc_id": "a",
                        "text": "본문",
                        "metadata": {"institution": "부산대학교"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            build_index(chunks_path, source_index, batch_size=10)
            np.save(
                artifact / "vectors.npy",
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                allow_pickle=False,
            )
            (artifact / "chunk_ids.jsonl").write_text(
                '"a#0000"\n', encoding="utf-8"
            )
            (artifact / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": {"id": "test/model", "prompts": {}},
                        "source": {"corpus_revision": "stale"},
                        "vectors": {
                            "file": "vectors.npy",
                            "count": 1,
                            "dimensions": 2,
                            "dtype": "float32",
                            "normalized": True,
                        },
                        "chunk_ids": {
                            "file": "chunk_ids.jsonl",
                            "count": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "corpus revision"):
                LearnedDenseIndex(
                    artifact,
                    source_index=source_index,
                    row_loader=lambda ids: [],
                )


if __name__ == "__main__":
    unittest.main()
