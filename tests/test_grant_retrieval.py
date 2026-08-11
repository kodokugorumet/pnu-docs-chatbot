from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bm25_search import build_index  # noqa: E402
from grant_retrieval import build_searcher  # noqa: E402
from rag import learned_dense  # noqa: E402


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

# 어휘로만 찾을 수 있는 문서(c)와 의미 벡터로만 찾을 수 있는 문서(b)를 함께 둔다.
CHUNKS = [
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
    {
        "chunk_id": "c#0000",
        "doc_id": "c",
        "chunk_index": 0,
        "text": "국내 여비 지급표 일비 이만원",
        "metadata": {"institution": "국가법령"},
    },
]

# 질의 벡터 [0, 1]과의 코사인 유사도: b=1.0 > c≈0.707 > a=0.0
VECTORS = [[1.0, 0.0], [0.0, 1.0], [0.7071068, 0.7071068]]


def _build_corpus(root: Path) -> tuple[Path, Path]:
    import numpy as np

    chunks_path = root / "chunks.jsonl"
    source_index = root / "bm25.sqlite"
    artifact = root / "artifact"
    artifact.mkdir()
    chunks_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in CHUNKS),
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
        np.asarray(VECTORS, dtype=np.float32),
        allow_pickle=False,
    )
    (artifact / "chunk_ids.jsonl").write_text(
        "".join(json.dumps(row["chunk_id"]) + "\n" for row in CHUNKS),
        encoding="utf-8",
    )
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {
                    "id": "test/model",
                    "revision": "revision",
                    "max_sequence_length": 512,
                    "prompts": {"query": ""},
                },
                "source": {"corpus_revision": revision},
                "vectors": {
                    "file": "vectors.npy",
                    "count": len(CHUNKS),
                    "dimensions": 2,
                    "dtype": "float32",
                    "normalized": True,
                },
                "chunk_ids": {
                    "file": "chunk_ids.jsonl",
                    "count": len(CHUNKS),
                },
            }
        ),
        encoding="utf-8",
    )
    return source_index, artifact


def _fake_embed(self, texts):  # noqa: ANN001
    return [[0.0, 1.0] for _ in texts]


class GrantRetrievalModeTests(unittest.TestCase):
    def test_bm25_mode_needs_no_dense_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks_path = root / "chunks.jsonl"
            source_index = root / "bm25.sqlite"
            chunks_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in CHUNKS
                ),
                encoding="utf-8",
            )
            build_index(chunks_path, source_index, batch_size=10)

            hits = build_searcher(source_index, "bm25")("일비", 5)

            self.assertTrue(hits)
            self.assertIsInstance(hits[0], dict)
            self.assertEqual(hits[0]["chunk_id"], "c#0000")
            # 평가·채점 코드가 읽는 키가 dict에 실려야 한다.
            for key in ("institution", "source_title", "file_name", "document_id"):
                self.assertIn(key, hits[0])

    def test_rejects_unknown_mode_and_missing_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown retrieval mode"):
            build_searcher(Path("unused.sqlite"), "rrf")
        with self.assertRaisesRegex(ValueError, "dense artifact"):
            build_searcher(Path("unused.sqlite"), "hybrid")


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional learned-dense dependency")
class GrantRetrievalLearnedModeTests(unittest.TestCase):
    def test_dense_mode_ranks_by_vector_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_index, artifact = _build_corpus(Path(temporary))
            with mock.patch.object(
                learned_dense.SentenceTransformerQueryEmbedder,
                "embed",
                _fake_embed,
            ):
                hits = build_searcher(source_index, "dense", artifact)("일비", 3)

            ids = [hit["chunk_id"] for hit in hits]
            # 어휘가 전혀 겹치지 않는 b가 벡터 유사도로 1위여야 한다.
            self.assertEqual(ids[0], "b#0000")
            self.assertIsInstance(hits[0], dict)
            self.assertEqual(hits[0]["text"], "장학금 신청 안내")
            self.assertTrue(hits[0]["preview"])

    def test_hybrid_mode_merges_both_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_index, artifact = _build_corpus(Path(temporary))
            with mock.patch.object(
                learned_dense.SentenceTransformerQueryEmbedder,
                "embed",
                _fake_embed,
            ):
                hits = build_searcher(source_index, "hybrid", artifact)("일비", 5)

            ids = {hit["chunk_id"] for hit in hits}
            # BM25만 찾는 c와 dense만 찾는 b가 둘 다 살아남아야 융합이 된 것이다.
            self.assertIn("c#0000", ids)
            self.assertIn("b#0000", ids)


class GrantRouterFallbackTests(unittest.TestCase):
    """폴백층(D50): 등록 문서는 1군이 top_k를 못 채울 때만 진입한다."""

    @staticmethod
    def _hit(chunk_id: str, file_name: str, institution: str = "한국연구재단") -> dict:
        return {"chunk_id": chunk_id, "file_name": file_name,
                "institution": institution}

    def test_fallback_docs_cannot_evict_primary_evidence(self) -> None:
        from rag.grant_router import filter_hits
        fb = frozenset({"종합매뉴얼.pdf"})
        # D49 재현: 폴백 문서가 상위권을 점유해도 1군 근거가 슬롯을 지킨다.
        hits = [
            self._hit("m#0", "종합매뉴얼.pdf"),
            self._hit("m#1", "종합매뉴얼.pdf"),
            self._hit("a#0", "정답근거.pdf"),
            self._hit("b#0", "보조근거.pdf"),
            self._hit("c#0", "제3근거.pdf"),
        ]
        out = filter_hits(hits, ["한국연구재단"], 3, fallback_files=fb)
        self.assertEqual([h["chunk_id"] for h in out], ["a#0", "b#0", "c#0"])

    def test_fallback_docs_fill_remaining_slots(self) -> None:
        from rag.grant_router import filter_hits
        fb = frozenset({"종합매뉴얼.pdf"})
        hits = [
            self._hit("m#0", "종합매뉴얼.pdf"),
            self._hit("a#0", "정답근거.pdf"),
        ]
        out = filter_hits(hits, ["한국연구재단"], 3, fallback_files=fb)
        # 1군이 부족하면 폴백이 잔여 슬롯에 진입한다 (순서: 1군 → 폴백).
        self.assertEqual([h["chunk_id"] for h in out], ["a#0", "m#0"])

    def test_fallback_applies_to_unscoped_queries_too(self) -> None:
        from rag.grant_router import filter_hits
        fb = frozenset({"종합매뉴얼.pdf"})
        hits = [
            self._hit("m#0", "종합매뉴얼.pdf"),
            self._hit("a#0", "정답근거.pdf"),
            self._hit("b#0", "보조근거.pdf"),
        ]
        out = filter_hits(hits, None, 2, fallback_files=fb)
        self.assertEqual([h["chunk_id"] for h in out], ["a#0", "b#0"])

    def test_no_fallback_set_preserves_existing_behavior(self) -> None:
        from rag.grant_router import filter_hits
        hits = [
            self._hit("m#0", "종합매뉴얼.pdf"),
            self._hit("a#0", "정답근거.pdf"),
        ]
        out = filter_hits(hits, ["한국연구재단"], 2)
        self.assertEqual([h["chunk_id"] for h in out], ["m#0", "a#0"])


if __name__ == "__main__":
    unittest.main()
