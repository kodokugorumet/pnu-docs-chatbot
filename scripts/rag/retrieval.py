"""Hybrid retrieval primitives with deterministic local fallbacks.

The implementation is intentionally dependency-free:

* a deterministic hashing embedder provides an offline dense baseline;
* an optional OpenAI-compatible ``/v1/embeddings`` client can replace it;
* ``DenseIndex`` supports in-memory, JSONL, and SQLite candidates;
* reciprocal-rank fusion records every stage explicitly; and
* ``HybridRetriever`` keeps serving one lane when the other lane fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from .models import (
    Location,
    RetrievalScores,
    SearchHit,
    StageScore,
    normalize_search_hit,
)


TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+|\d+")
ARTICLE_HEADING_RE = re.compile(
    r"제\s*\d+\s*조(?:의\s*\d+)?\s*[（(]([^)\n）]{1,80})[)）]"
)
HANGUL_RE = re.compile(r"^[가-힣]+$")
SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_RRF_K = 60


class EmbeddingProvider(Protocol):
    """Minimal interface shared by local and remote embedding providers."""

    kind: str

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...


def _embedding_terms(value: str) -> List[str]:
    terms: List[str] = []
    for match in TOKEN_RE.finditer(value.lower()):
        token = match.group(0)
        if len(token) <= 1 and not token.isdigit():
            continue
        terms.append("w:" + token)
        if HANGUL_RE.fullmatch(token) and len(token) >= 2:
            terms.extend(
                "b:" + token[index : index + 2]
                for index in range(len(token) - 1)
            )
    return terms


def _normalize_vector(vector: Sequence[float]) -> List[float]:
    normalized: List[float] = []
    for value in vector:
        if isinstance(value, bool):
            raise ValueError("embedding vectors must contain finite numbers")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "embedding vectors must contain finite numbers"
            ) from exc
        if not math.isfinite(parsed):
            raise ValueError("embedding vectors must contain finite numbers")
        normalized.append(parsed)
    if not normalized:
        raise ValueError("embedding vectors must not be empty")
    return normalized


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


class HashingEmbedder:
    """Deterministic signed feature hashing suitable for an offline baseline."""

    kind = "local_hashing_v1"

    def __init__(self, dimensions: int = 256) -> None:
        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions < 8
        ):
            raise ValueError("dimensions must be an integer of at least 8")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("embed expects a sequence of strings")
        result: List[List[float]] = []
        for value in texts:
            text = value if isinstance(value, str) else str(value or "")
            counts = Counter(_embedding_terms(text))
            vector = [0.0] * self.dimensions
            for term, count in counts.items():
                digest = hashlib.sha256(term.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dimensions
                sign = 1.0 if digest[8] & 1 else -1.0
                vector[bucket] += sign * (1.0 + math.log(float(count)))
            norm = _vector_norm(vector)
            if norm > 0:
                vector = [item / norm for item in vector]
            result.append(vector)
        return result


class OpenAICompatibleEmbeddingClient:
    """Small stdlib client for an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        dimensions: Optional[int] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        base = base_url.rstrip("/")
        if base.endswith("/v1/embeddings"):
            endpoint = base
        elif base.endswith("/v1"):
            endpoint = base + "/embeddings"
        else:
            endpoint = base + "/v1/embeddings"
        self.endpoint = endpoint
        self.model = model.strip()
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.dimensions = dimensions
        self.extra_headers = dict(extra_headers or {})
        self.kind = "openai_compatible:{}".format(self.model)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("embed expects a sequence of strings")
        inputs = [
            value if isinstance(value, str) else str(value or "")
            for value in texts
        ]
        if not inputs:
            return []
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": inputs,
        }
        if self.dimensions is not None:
            payload["dimensions"] = int(self.dimensions)
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "embedding endpoint returned HTTP {}: {}".format(
                    exc.code, detail[:300]
                )
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "embedding endpoint is unavailable: {}".format(exc.reason)
            ) from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "embedding endpoint returned invalid JSON"
            ) from exc
        data = decoded.get("data") if isinstance(decoded, Mapping) else None
        if not isinstance(data, list):
            raise RuntimeError("embedding response is missing data")
        ordered = sorted(
            enumerate(data),
            key=lambda pair: (
                pair[1].get("index", pair[0])
                if isinstance(pair[1], Mapping)
                else pair[0]
            ),
        )
        vectors = []
        for _, item in ordered:
            if not isinstance(item, Mapping):
                raise RuntimeError("embedding response item is invalid")
            vectors.append(_normalize_vector(item.get("embedding", [])))
        if len(vectors) != len(inputs):
            raise RuntimeError(
                "embedding response count does not match input count"
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise RuntimeError("embedding response dimensions are inconsistent")
        return vectors


def _embed_many(
    provider: EmbeddingProvider | Callable[[Sequence[str]], Sequence[Sequence[float]]],
    texts: Sequence[str],
) -> List[List[float]]:
    embed = getattr(provider, "embed", None)
    raw = embed(texts) if callable(embed) else provider(texts)
    vectors = [_normalize_vector(vector) for vector in raw]
    if len(vectors) != len(texts):
        raise ValueError("embedding count does not match text count")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError("embedding dimensions are inconsistent")
    return vectors


def _provider_kind(provider: Any) -> str:
    value = getattr(provider, "kind", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return type(provider).__name__


@dataclass(frozen=True)
class _DenseRecord:
    hit: SearchHit
    vector: Tuple[float, ...]
    norm: float


class DenseIndex:
    """A small exact-cosine index backed by memory and optional SQLite."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        records: Sequence[_DenseRecord],
        *,
        embedder: EmbeddingProvider
        | Callable[[Sequence[str]], Sequence[Sequence[float]]],
        path: Optional[Path] = None,
        embedding_kind: Optional[str] = None,
        corpus_revision: Optional[str] = None,
    ) -> None:
        self._records = tuple(records)
        self.embedder = embedder
        self.path = Path(path) if path is not None else None
        self.embedding_kind = embedding_kind or _provider_kind(embedder)
        self.corpus_revision = (
            str(corpus_revision).strip() if corpus_revision else None
        )
        dimensions = {len(record.vector) for record in self._records}
        if len(dimensions) > 1:
            raise ValueError("dense index contains mixed vector dimensions")
        self.dimensions = next(iter(dimensions), None)

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[SearchHit | Mapping[str, Any]],
        *,
        embedder: Optional[
            EmbeddingProvider
            | Callable[[Sequence[str]], Sequence[Sequence[float]]]
        ] = None,
        index_path: Optional[Path] = None,
        corpus_revision: Optional[str] = None,
    ) -> "DenseIndex":
        provider = embedder or HashingEmbedder()
        unique: Dict[str, SearchHit] = {}
        for value in rows:
            hit = (
                value
                if isinstance(value, SearchHit)
                else normalize_search_hit(value, stage=None)
            )
            if hit.chunk_id not in unique:
                unique[hit.chunk_id] = hit
        hits = [unique[key] for key in sorted(unique)]
        vectors = _embed_many(
            provider, [hit.text or hit.preview for hit in hits]
        )
        records = [
            _DenseRecord(
                hit=hit,
                vector=tuple(vector),
                norm=_vector_norm(vector),
            )
            for hit, vector in zip(hits, vectors)
        ]
        result = cls(
            records,
            embedder=provider,
            path=index_path,
            embedding_kind=_provider_kind(provider),
            corpus_revision=corpus_revision,
        )
        if index_path is not None:
            result.save_atomic(Path(index_path))
        return result

    @classmethod
    def from_jsonl(
        cls,
        chunks_path: Path,
        *,
        embedder: Optional[
            EmbeddingProvider
            | Callable[[Sequence[str]], Sequence[Sequence[float]]]
        ] = None,
        index_path: Optional[Path] = None,
        corpus_revision: Optional[str] = None,
    ) -> "DenseIndex":
        path = Path(chunks_path)

        def records() -> Iterable[Mapping[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "invalid JSON at {}:{}".format(
                                path, line_number
                            )
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise ValueError(
                            "JSONL row at {}:{} must be an object".format(
                                path, line_number
                            )
                        )
                    yield value

        return cls.from_rows(
            records(),
            embedder=embedder,
            index_path=index_path,
            corpus_revision=corpus_revision,
        )

    @classmethod
    def from_sqlite(
        cls,
        source_path: Path,
        *,
        table: str = "chunks",
        query: Optional[str] = None,
        embedder: Optional[
            EmbeddingProvider
            | Callable[[Sequence[str]], Sequence[Sequence[float]]]
        ] = None,
        index_path: Optional[Path] = None,
        corpus_revision: Optional[str] = None,
    ) -> "DenseIndex":
        path = Path(source_path)
        if (
            index_path is not None
            and Path(index_path).resolve() == path.resolve()
        ):
            raise ValueError("dense index path must differ from source SQLite")
        if query is None:
            if not SQL_IDENTIFIER_RE.fullmatch(table):
                raise ValueError("invalid SQLite table name")
            query = "SELECT * FROM {} ORDER BY chunk_id".format(table)
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        try:
            rows = [dict(row) for row in connection.execute(query)]
        finally:
            connection.close()
        return cls.from_rows(
            rows,
            embedder=embedder,
            index_path=index_path,
            corpus_revision=corpus_revision,
        )

    @classmethod
    def from_bm25_index(
        cls,
        source_path: Path,
        *,
        embedder: Optional[
            EmbeddingProvider
            | Callable[[Sequence[str]], Sequence[Sequence[float]]]
        ] = None,
        index_path: Optional[Path] = None,
    ) -> "DenseIndex":
        """Build from the canonical BM25 store without losing citation data."""

        path = Path(source_path)
        if (
            index_path is not None
            and Path(index_path).resolve() == path.resolve()
        ):
            raise ValueError("dense index path must differ from source SQLite")
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        try:
            meta = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM index_meta"
                )
            }
            raw_rows = connection.execute(
                "SELECT * FROM chunks ORDER BY chunk_id"
            ).fetchall()
        finally:
            connection.close()

        def decode(value: Any, default: Any) -> Any:
            if value in (None, ""):
                return default
            try:
                return json.loads(str(value))
            except (TypeError, json.JSONDecodeError):
                return default

        rows = []
        for raw_row in raw_rows:
            row = dict(raw_row)
            locations = decode(row.pop("locations_json", None), [])
            section_path = decode(row.pop("section_path_json", None), None)
            table_ids = decode(row.pop("table_ids_json", None), [])
            block_ids = decode(row.pop("block_ids_json", None), [])
            source_aliases = decode(
                row.pop("source_aliases_json", None), []
            )
            metadata = {
                "corpus_revision": row.get("corpus_revision"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "section_path": section_path,
                "table_ids": table_ids,
                "block_ids": block_ids,
                "source_title": row.get("source_title"),
                "source_url": row.get("source_url"),
                "download_url": row.get("download_url"),
                "source_host": row.get("source_host"),
                "fetched_at": row.get("fetched_at"),
                "published_at": row.get("published_at"),
                "category": row.get("category"),
                "include_reason": row.get("include_reason"),
                "crawl_storage_path": row.get("crawl_storage_path"),
                "source_aliases": source_aliases,
            }
            row.update(
                {
                    "locations": locations,
                    "source_aliases": source_aliases,
                    "metadata": metadata,
                    "preview": row.get("text", ""),
                }
            )
            rows.append(row)

        return cls.from_rows(
            rows,
            embedder=embedder,
            index_path=index_path,
            corpus_revision=meta.get("corpus_revision"),
        )

    @classmethod
    def from_source(
        cls,
        source_path: Path,
        **kwargs: Any,
    ) -> "DenseIndex":
        suffix = Path(source_path).suffix.lower()
        if suffix in {".sqlite", ".sqlite3", ".db"}:
            return cls.from_sqlite(source_path, **kwargs)
        return cls.from_jsonl(source_path, **kwargs)

    def save_atomic(self, index_path: Path) -> None:
        """Write a complete SQLite index and atomically replace the target."""

        target = Path(index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".{}.".format(target.name),
            suffix=".tmp",
            dir=str(target.parent),
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            connection = sqlite3.connect(str(temporary))
            try:
                connection.executescript(
                    """
                    PRAGMA journal_mode=DELETE;
                    PRAGMA synchronous=FULL;
                    CREATE TABLE dense_entries (
                      position INTEGER PRIMARY KEY,
                      chunk_id TEXT NOT NULL UNIQUE,
                      row_json TEXT NOT NULL,
                      vector_json TEXT NOT NULL,
                      vector_norm REAL NOT NULL
                    );
                    CREATE TABLE dense_meta (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL
                    );
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO dense_entries (
                      position, chunk_id, row_json, vector_json, vector_norm
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            index,
                            record.hit.chunk_id,
                            json.dumps(
                                record.hit.to_dict(),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                            json.dumps(
                                list(record.vector),
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                            record.norm,
                        )
                        for index, record in enumerate(self._records)
                    ],
                )
                meta = {
                    "schema_version": str(self.SCHEMA_VERSION),
                    "embedding_kind": self.embedding_kind,
                    "dimensions": str(self.dimensions or 0),
                    "count": str(len(self._records)),
                    "corpus_revision": self.corpus_revision or "",
                }
                connection.executemany(
                    "INSERT INTO dense_meta (key, value) VALUES (?, ?)",
                    sorted(meta.items()),
                )
                connection.commit()
            finally:
                connection.close()
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(target))
            self.path = target
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(
        cls,
        index_path: Path,
        *,
        embedder: Optional[
            EmbeddingProvider
            | Callable[[Sequence[str]], Sequence[Sequence[float]]]
        ] = None,
    ) -> "DenseIndex":
        path = Path(index_path)
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        try:
            meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key, value FROM dense_meta"
                )
            }
            rows = connection.execute(
                """
                SELECT row_json, vector_json, vector_norm
                FROM dense_entries
                ORDER BY position
                """
            ).fetchall()
        finally:
            connection.close()
        if meta.get("schema_version") != str(cls.SCHEMA_VERSION):
            raise ValueError("unsupported dense index schema version")
        dimensions = int(meta.get("dimensions", "0"))
        embedding_kind = meta.get("embedding_kind", "")
        provider = embedder
        if provider is None:
            if embedding_kind == HashingEmbedder.kind:
                provider = HashingEmbedder(dimensions=dimensions)
            else:
                raise ValueError(
                    "an embedding provider is required to load this index"
                )
        records = []
        for row in rows:
            value = json.loads(row["row_json"])
            vector = tuple(_normalize_vector(json.loads(row["vector_json"])))
            if dimensions and len(vector) != dimensions:
                raise ValueError("dense index vector dimension mismatch")
            records.append(
                _DenseRecord(
                    hit=normalize_search_hit(value, stage=None),
                    vector=vector,
                    norm=float(row["vector_norm"]),
                )
            )
        return cls(
            records,
            embedder=provider,
            path=path,
            embedding_kind=embedding_kind,
            corpus_revision=meta.get("corpus_revision") or None,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        institution: Optional[str] = None,
        *,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[SearchHit]:
        if top_k <= 0 or not self._records:
            return []
        query_vector = _embed_many(self.embedder, [query])[0]
        if self.dimensions is not None and len(query_vector) != self.dimensions:
            raise ValueError("query embedding dimension does not match index")
        query_norm = _vector_norm(query_vector)
        required = dict(filters or {})
        if institution:
            required["institution"] = institution
        scored = []
        for record in self._records:
            if not _matches_filters(record.hit, required):
                continue
            if query_norm == 0 or record.norm == 0:
                similarity = 0.0
            else:
                similarity = sum(
                    left * right
                    for left, right in zip(query_vector, record.vector)
                ) / (query_norm * record.norm)
            scored.append((similarity, record.hit.chunk_id, record.hit))
        scored.sort(key=lambda item: (-item[0], item[1]))
        result = []
        for rank, (score, _, hit) in enumerate(scored[:top_k], start=1):
            result.append(
                hit.with_stage(
                    "dense",
                    rank=rank,
                    score=score,
                    kind=self.embedding_kind,
                ).with_final_rank(rank)
            )
        return result


def _matches_filters(hit: SearchHit, filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = (
            getattr(hit, key)
            if hasattr(hit, key)
            else hit.metadata.get(key)
        )
        if isinstance(expected, (set, frozenset, list, tuple)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _hit_richness(hit: SearchHit) -> Tuple[int, int, int]:
    return (
        len(hit.text),
        len(hit.locations),
        len(hit.metadata),
    )


def _best_stage(
    values: Iterable[Optional[StageScore]],
) -> Optional[StageScore]:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda value: (
            value.rank if value.rank is not None else 10**12,
            -(value.score if value.score is not None else -math.inf),
            value.kind or "",
        ),
    )[0]


def _merge_occurrences(
    occurrences: Sequence[Tuple[str, int, SearchHit]],
) -> SearchHit:
    ordered = sorted(
        occurrences,
        key=lambda item: (
            tuple(-value for value in _hit_richness(item[2])),
            item[0],
            item[1],
        ),
    )
    base = ordered[0][2]
    retrieval = RetrievalScores(
        bm25=_best_stage(item[2].retrieval.bm25 for item in ordered),
        dense=_best_stage(item[2].retrieval.dense for item in ordered),
        rrf=_best_stage(item[2].retrieval.rrf for item in ordered),
        reranker=_best_stage(item[2].retrieval.reranker for item in ordered),
    )
    locations: List[Location] = []
    seen_locations = set()
    for _, _, hit in ordered:
        for location in hit.locations:
            key = (
                location.block_id,
                location.page,
                location.section_path,
                location.table_id,
                location.row,
                location.column,
            )
            if key not in seen_locations:
                seen_locations.add(key)
                locations.append(location)
    return replace(
        base,
        locations=tuple(locations),
        retrieval=retrieval,
        final_rank=None,
    )


def reciprocal_rank_fusion(
    lanes: Mapping[
        str, Sequence[SearchHit | Mapping[str, Any]]
    ],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[SearchHit]:
    """Fuse ranked lanes with ``sum(1 / (k + rank))``.

    Duplicate chunk IDs are counted once per lane.  Lanes and ties are sorted
    deterministically, so mapping insertion order cannot change the result.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("RRF k must be a non-negative integer")
    occurrences: Dict[str, List[Tuple[str, int, SearchHit]]] = {}
    scores: Dict[str, float] = {}
    best_ranks: Dict[str, int] = {}

    for lane_name in sorted(lanes):
        values = lanes[lane_name]
        seen = set()
        unique_rank = 0
        for value in values:
            chunk_id = (
                value.chunk_id
                if isinstance(value, SearchHit)
                else value.get("chunk_id")
            )
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique_rank += 1
            stage = lane_name if lane_name in {"bm25", "dense"} else None
            hit = (
                value
                if isinstance(value, SearchHit)
                else normalize_search_hit(value, stage=None)
            )
            if stage is not None:
                existing = getattr(hit.retrieval, stage)
                hit = hit.with_stage(
                    stage,
                    rank=unique_rank,
                    score=(
                        existing.score
                        if existing is not None
                        else (
                            value.get("score")
                            if isinstance(value, Mapping)
                            else None
                        )
                    ),
                    kind=(
                        existing.kind
                        if existing is not None and existing.kind
                        else stage
                    ),
                )
            occurrences.setdefault(chunk_id, []).append(
                (lane_name, unique_rank, hit)
            )
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / float(k + unique_rank)
            )
            best_ranks[chunk_id] = min(
                best_ranks.get(chunk_id, unique_rank),
                unique_rank,
            )

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_ranks[chunk_id],
            chunk_id,
        ),
    )
    if top_k is not None:
        ordered_ids = ordered_ids[: max(0, int(top_k))]
    result = []
    for rank, chunk_id in enumerate(ordered_ids, start=1):
        hit = _merge_occurrences(occurrences[chunk_id])
        result.append(
            hit.with_stage(
                "rrf",
                rank=rank,
                score=scores[chunk_id],
                kind="reciprocal_rank_fusion:k={}".format(k),
            ).with_final_rank(rank)
        )
    return result


def _lexical_tokens(value: str) -> List[str]:
    return [
        match.group(0).lower()
        for match in TOKEN_RE.finditer(value)
        if len(match.group(0)) > 1 or match.group(0).isdigit()
    ]


def _heading_match_coverage(query_terms: set[str], text: str) -> float:
    """Measure topic matches in legal-style article headings.

    A term in ``제64조(휴학)`` is substantially stronger evidence than the
    same term appearing incidentally in a long body paragraph.
    """

    headings = " ".join(ARTICLE_HEADING_RE.findall(text.lower()))
    if not headings:
        return 0.0
    matched = sum(1 for term in query_terms if term in headings)
    return matched / len(query_terms)


def _primary_policy_title_bonus(hit: SearchHit) -> float:
    """Prefer a selected institution's own top-level policy document.

    This stays conservative: the institution name must be immediately followed
    by a policy-document label after punctuation is removed.  Affiliated-unit
    documents such as ``부산대학교 사범대학부설고등학교 학칙`` therefore do
    not receive the boost unless the query terms independently support them.
    """

    if not hit.institution or not hit.file_name:
        return 0.0
    institution = re.sub(r"[^0-9A-Za-z가-힣]+", "", hit.institution.lower())
    file_name = re.sub(r"[^0-9A-Za-z가-힣]+", "", hit.file_name.lower())
    if not institution or institution not in file_name:
        return 0.0
    labels = ("학칙", "규정", "규칙", "지침", "요강")
    return 1.0 if any(
        institution + label in file_name for label in labels
    ) else 0.0


def lexical_fallback_score(
    query: str,
    hit: SearchHit | Mapping[str, Any],
    base_rank: int,
) -> float:
    """Current lexical reranking formula, separated from BM25 retrieval."""

    canonical = (
        hit
        if isinstance(hit, SearchHit)
        else normalize_search_hit(hit, stage=None)
    )
    query_terms = set(_lexical_tokens(query)[:24])
    if not query_terms:
        return 0.0
    body_terms = set(_lexical_tokens(canonical.text or canonical.preview))
    metadata_terms = set(
        _lexical_tokens(
            " ".join(
                value
                for value in (
                    canonical.institution,
                    canonical.file_name,
                    canonical.relative_path,
                    canonical.source_title,
                    canonical.category,
                )
                if value
            )
        )
    )
    body_coverage = len(query_terms & body_terms) / len(query_terms)
    metadata_coverage = len(query_terms & metadata_terms) / len(query_terms)
    normalized_query = re.sub(r"\s+", " ", query.lower()).strip()
    normalized_text = re.sub(
        r"\s+", " ", (canonical.text or canonical.preview).lower()
    ).strip()
    phrase_bonus = (
        1.0
        if normalized_query and normalized_query in normalized_text
        else 0.0
    )
    heading_coverage = _heading_match_coverage(query_terms, normalized_text)
    primary_policy_bonus = _primary_policy_title_bonus(canonical)
    rank_signal = 1.0 / max(1, int(base_rank))
    return (
        body_coverage * 0.42
        + metadata_coverage * 0.10
        + phrase_bonus * 0.12
        + heading_coverage * 0.36
        + primary_policy_bonus * 0.12
        + rank_signal * 0.10
    )


def lexical_fallback_rerank(
    query: str,
    hits: Sequence[SearchHit | Mapping[str, Any]],
    *,
    top_k: Optional[int] = None,
) -> List[SearchHit]:
    scored = []
    for position, value in enumerate(hits, start=1):
        hit = (
            value
            if isinstance(value, SearchHit)
            else normalize_search_hit(value, stage=None)
        )
        base_rank = hit.final_rank or position
        scored.append(
            (
                lexical_fallback_score(query, hit, base_rank),
                base_rank,
                hit.chunk_id,
                hit,
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    if top_k is not None:
        limit = max(0, int(top_k))
        selected = []
        deferred = []
        document_counts: Dict[str, int] = {}
        for item in scored:
            document_id = item[3].document_id
            if document_counts.get(document_id, 0) >= 2:
                deferred.append(item)
                continue
            selected.append(item)
            document_counts[document_id] = (
                document_counts.get(document_id, 0) + 1
            )
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            selected.extend(deferred[: limit - len(selected)])
        scored = selected
    result = []
    for rank, (score, _, _, hit) in enumerate(scored, start=1):
        result.append(
            hit.with_stage(
                "reranker",
                rank=rank,
                score=score,
                kind="lexical_heading_diverse",
            ).with_final_rank(rank)
        )
    return result


@dataclass(frozen=True)
class HybridSearchResult:
    hits: Tuple[SearchHit, ...]
    trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        object.__setattr__(self, "trace", dict(self.trace))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "trace": dict(self.trace),
        }


class HybridRetriever:
    """Coordinate injected BM25 and dense lanes with graceful fallback."""

    def __init__(
        self,
        *,
        bm25_search: Optional[Callable[..., Sequence[Any]]] = None,
        dense_index: Optional[DenseIndex] = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = 4,
        reranker: Optional[Callable[..., Sequence[SearchHit]]] = (
            lexical_fallback_rerank
        ),
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        self.bm25_search = bm25_search
        self.dense_index = dense_index
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 8,
        institution: Optional[str] = None,
        *,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> HybridSearchResult:
        if top_k <= 0:
            return HybridSearchResult(
                (),
                {
                    "lanes": {},
                    "fusion": {
                        "status": "skipped",
                        "kind": "rrf",
                        "k": self.rrf_k,
                    },
                    "reranker": {"status": "skipped"},
                },
            )
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        lane_hits: Dict[str, Sequence[SearchHit]] = {}
        lane_trace: Dict[str, Dict[str, Any]] = {}

        if self.bm25_search is None:
            lane_trace["bm25"] = {"status": "disabled", "count": 0}
        else:
            try:
                raw = self.bm25_search(
                    query=query,
                    top_k=candidate_k,
                    institution=institution,
                )
                if isinstance(raw, Mapping) and isinstance(
                    raw.get("results"), Sequence
                ):
                    raw = raw["results"]
                normalized = []
                seen = set()
                for value in raw or []:
                    hit = (
                        value
                        if isinstance(value, SearchHit)
                        else normalize_search_hit(value, stage=None)
                    )
                    if hit.chunk_id in seen:
                        continue
                    if institution and hit.institution != institution:
                        continue
                    if filters and not _matches_filters(hit, filters):
                        continue
                    seen.add(hit.chunk_id)
                    normalized.append(
                        hit.with_stage(
                            "bm25",
                            rank=len(normalized) + 1,
                            score=(
                                value.get("score")
                                if isinstance(value, Mapping)
                                else (
                                    hit.retrieval.bm25.score
                                    if hit.retrieval.bm25 is not None
                                    else None
                                )
                            ),
                            kind=(
                                hit.retrieval.bm25.kind
                                if hit.retrieval.bm25 is not None
                                and hit.retrieval.bm25.kind
                                else "sqlite_fts5_bm25"
                            ),
                        )
                    )
                lane_hits["bm25"] = normalized
                lane_trace["bm25"] = {
                    "status": "ok",
                    "count": len(normalized),
                }
            except Exception as exc:
                lane_trace["bm25"] = _failure_trace(exc)

        if self.dense_index is None:
            lane_trace["dense"] = {"status": "disabled", "count": 0}
        else:
            try:
                dense = self.dense_index.search(
                    query,
                    candidate_k,
                    institution,
                    filters=filters,
                )
                lane_hits["dense"] = dense
                lane_trace["dense"] = {
                    "status": "ok",
                    "count": len(dense),
                    "kind": self.dense_index.embedding_kind,
                }
            except Exception as exc:
                lane_trace["dense"] = _failure_trace(exc)

        active = {
            name: values
            for name, values in lane_hits.items()
            if values
        }
        fused = reciprocal_rank_fusion(
            active,
            k=self.rrf_k,
            top_k=candidate_k,
        )
        fusion_trace = {
            "status": "ok" if active else "empty",
            "kind": "rrf",
            "k": self.rrf_k,
            "input_counts": {
                name: len(values) for name, values in sorted(active.items())
            },
            "count": len(fused),
        }

        if self.reranker is None:
            final = [
                hit.with_final_rank(rank)
                for rank, hit in enumerate(fused[:top_k], start=1)
            ]
            reranker_trace = {
                "status": "disabled",
                "count": len(final),
            }
        else:
            try:
                final = list(
                    self.reranker(query, fused, top_k=top_k)
                )
                reranker_trace = {
                    "status": "ok",
                    "kind": (
                        final[0].retrieval.reranker.kind
                        if final
                        and final[0].retrieval.reranker is not None
                        else "custom"
                    ),
                    "count": len(final),
                }
            except Exception as exc:
                final = [
                    hit.with_final_rank(rank)
                    for rank, hit in enumerate(fused[:top_k], start=1)
                ]
                reranker_trace = _failure_trace(exc)
                reranker_trace["count"] = len(final)

        return HybridSearchResult(
            tuple(final),
            {
                "lanes": lane_trace,
                "fusion": fusion_trace,
                "reranker": reranker_trace,
            },
        )


def _failure_trace(exc: BaseException) -> Dict[str, Any]:
    return {
        "status": "error",
        "count": 0,
        "error_type": type(exc).__name__,
        "error": str(exc)[:300],
    }
