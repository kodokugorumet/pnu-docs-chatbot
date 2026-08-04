"""Runtime adapter for reproducible SentenceTransformer dense artifacts.

The vector matrix is memory-mapped, while the comparatively expensive query
model is loaded only when its retrieval mode is used for the first time.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

from .models import SearchHit, normalize_search_hit


RowLoader = Callable[
    [Sequence[str]], Sequence[Union[SearchHit, Mapping[str, Any]]]
]


def _read_chunk_ids(path: Path) -> tuple[str, ...]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid chunk ID JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"chunk ID at {path}:{line_number} must be non-empty"
                )
            values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("learned dense artifact contains duplicate chunk IDs")
    return tuple(values)


def _sqlite_corpus(index_path: Path) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    if not index_path.is_file():
        raise FileNotFoundError(f"missing BM25 index: {index_path}")
    connection = sqlite3.connect(str(index_path))
    try:
        rows = connection.execute(
            "SELECT chunk_id, COALESCE(institution, '') FROM chunks ORDER BY chunk_id"
        ).fetchall()
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM index_meta"
            )
        }
    finally:
        connection.close()
    return (
        tuple(str(row[0]) for row in rows),
        tuple(str(row[1]) for row in rows),
        metadata.get("corpus_revision"),
    )


class SentenceTransformerQueryEmbedder:
    """Lazy, serialized SentenceTransformer inference for one query model."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None,
        query_prefix: str = "",
        max_sequence_length: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.query_prefix = query_prefix
        self.max_sequence_length = max_sequence_length
        self.kind = f"sentence_transformer:{model_id}"
        self._model: Any = None
        self._device: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        requested = os.environ.get("RAG_EMBEDDING_DEVICE", "auto").strip().lower()
        if requested not in {"auto", "mps", "cpu"}:
            requested = "auto"
        device = (
            "mps"
            if requested == "auto" and torch.backends.mps.is_available()
            else "cpu" if requested == "auto" else requested
        )
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS embedding was requested but is unavailable")
        local_only = os.environ.get("RAG_EMBEDDING_LOCAL_ONLY", "1").strip().lower()
        model = SentenceTransformer(
            self.model_id,
            revision=self.revision,
            device=device,
            local_files_only=local_only not in {"0", "false", "no"},
        )
        if self.max_sequence_length:
            model.max_seq_length = min(
                int(model.max_seq_length), int(self.max_sequence_length)
            )
        if device == "mps":
            model.half()
        self._model = model
        self._device = device
        return model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("embed expects a sequence of strings")
        if not texts:
            return []
        with self._lock:
            import numpy as np
            import torch

            model = self._load()
            inputs = [self.query_prefix + str(value or "") for value in texts]
            with torch.inference_mode():
                vectors = model.encode(
                    inputs,
                    batch_size=max(1, len(inputs)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            if self._device == "mps":
                torch.mps.synchronize()
            array = np.asarray(vectors, dtype=np.float32)
            return array.tolist()


class LearnedDenseIndex:
    """Exact cosine search over a verified, normalized NumPy matrix."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        artifact_dir: Path,
        *,
        source_index: Path,
        row_loader: RowLoader,
    ) -> None:
        import numpy as np

        self.path = Path(artifact_dir)
        self.source_index = Path(source_index)
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported learned dense artifact schema version")

        model = manifest.get("model")
        source = manifest.get("source")
        vectors_meta = manifest.get("vectors")
        ids_meta = manifest.get("chunk_ids")
        if not all(
            isinstance(value, Mapping)
            for value in (model, source, vectors_meta, ids_meta)
        ):
            raise ValueError("learned dense manifest is incomplete")

        vector_name = str(vectors_meta.get("file") or "")
        ids_name = str(ids_meta.get("file") or "")
        if Path(vector_name).name != vector_name or Path(ids_name).name != ids_name:
            raise ValueError("artifact filenames must not escape their directory")
        vectors_path = self.path / vector_name
        ids_path = self.path / ids_name
        self._vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        self._chunk_ids = _read_chunk_ids(ids_path)

        count = int(vectors_meta.get("count", 0))
        dimensions = int(vectors_meta.get("dimensions", 0))
        if self._vectors.shape != (count, dimensions):
            raise ValueError("learned dense vector shape does not match manifest")
        if str(self._vectors.dtype) != str(vectors_meta.get("dtype")):
            raise ValueError("learned dense vector dtype does not match manifest")
        if len(self._chunk_ids) != count or int(ids_meta.get("count", 0)) != count:
            raise ValueError("learned dense chunk count does not match manifest")
        if vectors_meta.get("normalized") is not True:
            raise ValueError("learned dense vectors must be normalized")

        corpus_ids, institutions, corpus_revision = _sqlite_corpus(self.source_index)
        artifact_revision = str(source.get("corpus_revision") or "").strip() or None
        if not corpus_revision or artifact_revision != corpus_revision:
            raise ValueError("learned dense corpus revision does not match BM25")
        if corpus_ids != self._chunk_ids:
            raise ValueError("learned dense chunk IDs do not align with BM25")

        model_id = str(model.get("id") or "").strip()
        if not model_id:
            raise ValueError("learned dense manifest has no model ID")
        prompts = model.get("prompts")
        query_prefix = (
            str(prompts.get("query") or "")
            if isinstance(prompts, Mapping)
            else ""
        )
        self.embedder = SentenceTransformerQueryEmbedder(
            model_id,
            revision=str(model.get("revision") or "").strip() or None,
            query_prefix=query_prefix,
            max_sequence_length=(
                int(model["max_sequence_length"])
                if model.get("max_sequence_length")
                else None
            ),
        )
        self.embedding_kind = self.embedder.kind
        self.model_id = model_id
        self.model_revision = self.embedder.revision
        self.query_prefix = query_prefix
        self.corpus_revision = corpus_revision
        self.dimensions = dimensions
        self._row_loader = row_loader
        self._institution_positions: dict[str, Any] = {}
        for position, institution in enumerate(institutions):
            self._institution_positions.setdefault(institution, []).append(position)
        self._institution_positions = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in self._institution_positions.items()
        }

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def public_status(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "model_revision": self.model_revision,
            "embedding_kind": self.embedding_kind,
            "dimensions": self.dimensions,
            "chunk_count": len(self),
            "corpus_revision": self.corpus_revision,
            "query_prefix": self.query_prefix,
            "model_loaded": self.embedder.loaded,
            "device": self.embedder.device,
            "index": str(self.path),
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        institution: str | None = None,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        import numpy as np

        if top_k <= 0 or len(self) == 0:
            return []
        if institution is not None:
            positions = self._institution_positions.get(institution)
            if positions is None or positions.size == 0:
                return []
        else:
            positions = None

        vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise ValueError("query embedding dimension does not match index")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("query embedding must be finite and non-zero")
        if abs(norm - 1.0) > 1e-4:
            vector = vector / norm

        scores = np.clip(
            np.asarray(self._vectors @ vector, dtype=np.float32), -1.0, 1.0
        )
        candidate_scores = scores if positions is None else scores[positions]
        requested = min(len(candidate_scores), max(top_k, top_k * 8, 64))
        if requested == 0:
            return []
        if requested == len(candidate_scores):
            local_positions = np.arange(requested)
        else:
            local_positions = np.argpartition(
                -candidate_scores, requested - 1
            )[:requested]
        global_positions = (
            local_positions if positions is None else positions[local_positions]
        )
        ranked = sorted(
            (
                (float(scores[position]), self._chunk_ids[int(position)])
                for position in global_positions
            ),
            key=lambda value: (-value[0], value[1]),
        )
        rows = self._row_loader([chunk_id for _, chunk_id in ranked])
        hits_by_id = {
            hit.chunk_id: hit
            for hit in (
                value
                if isinstance(value, SearchHit)
                else normalize_search_hit(value, stage=None)
                for value in rows
            )
        }
        required = dict(filters or {})
        if institution:
            required["institution"] = institution
        result: list[SearchHit] = []
        for score, chunk_id in ranked:
            hit = hits_by_id.get(chunk_id)
            if hit is None:
                continue
            if any(
                (
                    getattr(hit, key, hit.metadata.get(key))
                    not in expected
                    if isinstance(expected, (set, frozenset, list, tuple))
                    else getattr(hit, key, hit.metadata.get(key)) != expected
                )
                for key, expected in required.items()
            ):
                continue
            rank = len(result) + 1
            result.append(
                hit.with_stage(
                    "dense",
                    rank=rank,
                    score=score,
                    kind=self.embedding_kind,
                ).with_final_rank(rank)
            )
            if len(result) >= top_k:
                break
        return result
