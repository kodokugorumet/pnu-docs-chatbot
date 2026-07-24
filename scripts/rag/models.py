"""Stable, JSON-friendly contracts for retrieval and citation results.

The parser's canonical ``Block`` remains the source of truth for document
locations.  This module deliberately does not infer a page, table cell, row, or
column from chunk ordering or text.  If a legacy chunk only carries
``metadata.block_ids``, the resulting locations contain those block IDs and
leave all unavailable coordinates as ``None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


RETRIEVAL_STAGES = ("bm25", "dense", "rrf", "reranker")


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(field_name))
    return value


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _optional_int(value: Any, *, minimum: int = 0) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= minimum else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= minimum else None
    return None


def _optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _section_path(value: Any) -> Optional[Tuple[str, ...]]:
    if value is None or isinstance(value, (str, bytes)):
        return None
    if not isinstance(value, Sequence):
        return None
    normalized = tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )
    return normalized or None


@dataclass(frozen=True)
class Location:
    """One exact-or-partial source block location.

    Page numbers are one-based.  Table rows and columns are zero-based.  Every
    value may be unknown, except that callers normally provide ``block_id``.
    """

    block_id: Optional[str] = None
    page: Optional[int] = None
    section_path: Optional[Tuple[str, ...]] = None
    table_id: Optional[str] = None
    row: Optional[int] = None
    column: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _optional_string(self.block_id))
        object.__setattr__(self, "page", _optional_int(self.page, minimum=1))
        object.__setattr__(
            self, "section_path", _section_path(self.section_path)
        )
        object.__setattr__(self, "table_id", _optional_string(self.table_id))
        object.__setattr__(self, "row", _optional_int(self.row, minimum=0))
        object.__setattr__(
            self, "column", _optional_int(self.column, minimum=0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "page": self.page,
            "section_path": (
                list(self.section_path)
                if self.section_path is not None
                else None
            ),
            "table_id": self.table_id,
            "row": self.row,
            "column": self.column,
        }


@dataclass(frozen=True)
class StageScore:
    """Rank and score emitted by one retrieval stage."""

    rank: Optional[int] = None
    score: Optional[float] = None
    kind: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", _optional_int(self.rank, minimum=1))
        object.__setattr__(self, "score", _optional_float(self.score))
        object.__setattr__(self, "kind", _optional_string(self.kind))

    @classmethod
    def from_value(cls, value: Any) -> Optional["StageScore"]:
        if value is None:
            return None
        if isinstance(value, StageScore):
            return value
        if isinstance(value, Mapping):
            return cls(
                rank=value.get("rank"),
                score=value.get("score"),
                kind=value.get("kind"),
            )
        score = _optional_float(value)
        return cls(score=score) if score is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class RetrievalScores:
    """Explicit stage scores; absent lanes serialize as ``null``."""

    bm25: Optional[StageScore] = None
    dense: Optional[StageScore] = None
    rrf: Optional[StageScore] = None
    reranker: Optional[StageScore] = None

    @classmethod
    def from_mapping(cls, value: Any) -> "RetrievalScores":
        if isinstance(value, RetrievalScores):
            return value
        source = value if isinstance(value, Mapping) else {}
        return cls(
            **{
                stage: StageScore.from_value(source.get(stage))
                for stage in RETRIEVAL_STAGES
            }
        )
    def with_stage(
        self,
        stage: str,
        *,
        rank: Optional[int],
        score: Optional[float],
        kind: Optional[str],
    ) -> "RetrievalScores":
        if stage not in RETRIEVAL_STAGES:
            raise ValueError("unknown retrieval stage: {}".format(stage))
        return replace(
            self,
            **{
                stage: StageScore(
                    rank=rank,
                    score=score,
                    kind=kind,
                )
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            stage: (
                getattr(self, stage).to_dict()
                if getattr(self, stage) is not None
                else None
            )
            for stage in RETRIEVAL_STAGES
        }


@dataclass(frozen=True)
class SearchHit:
    """Canonical backend-facing retrieval result."""

    chunk_id: str
    document_id: str
    chunk_index: Optional[int] = None
    text: str = ""
    preview: str = ""
    char_count: int = 0
    institution: Optional[str] = None
    file_name: Optional[str] = None
    source_path: Optional[str] = None
    relative_path: Optional[str] = None
    extension: Optional[str] = None
    parser: Optional[str] = None
    locations: Tuple[Location, ...] = ()
    retrieval: RetrievalScores = field(default_factory=RetrievalScores)
    final_rank: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "chunk_id", _non_empty_string(self.chunk_id, "chunk_id")
        )
        object.__setattr__(
            self,
            "document_id",
            _non_empty_string(self.document_id, "document_id"),
        )
        object.__setattr__(
            self, "chunk_index", _optional_int(self.chunk_index, minimum=0)
        )
        if not isinstance(self.text, str):
            object.__setattr__(self, "text", str(self.text or ""))
        if not isinstance(self.preview, str):
            object.__setattr__(self, "preview", str(self.preview or ""))
        if not self.preview and self.text:
            object.__setattr__(self, "preview", self.text)
        resolved_count = _optional_int(self.char_count, minimum=0)
        if resolved_count is None or (
            resolved_count == 0 and len(self.text) > 0
        ):
            resolved_count = len(self.text)
        object.__setattr__(self, "char_count", resolved_count)
        for name in (
            "institution",
            "file_name",
            "source_path",
            "relative_path",
            "extension",
            "parser",
        ):
            object.__setattr__(
                self, name, _optional_string(getattr(self, name))
            )
        normalized_locations = tuple(
            item
            if isinstance(item, Location)
            else _location_from_mapping(item)
            for item in self.locations
            if isinstance(item, (Location, Mapping))
        )
        object.__setattr__(
            self,
            "locations",
            tuple(item for item in normalized_locations if item is not None),
        )
        object.__setattr__(
            self,
            "retrieval",
            RetrievalScores.from_mapping(self.retrieval),
        )
        object.__setattr__(
            self, "final_rank", _optional_int(self.final_rank, minimum=1)
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def with_stage(
        self,
        stage: str,
        *,
        rank: Optional[int],
        score: Optional[float],
        kind: Optional[str],
    ) -> "SearchHit":
        return replace(
            self,
            retrieval=self.retrieval.with_stage(
                stage,
                rank=rank,
                score=score,
                kind=kind,
            ),
        )

    def with_final_rank(self, rank: Optional[int]) -> "SearchHit":
        return replace(self, final_rank=rank)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize using ``document_id`` only (never the legacy ``doc_id``)."""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "preview": self.preview,
            "char_count": self.char_count,
            "institution": self.institution,
            "file_name": self.file_name,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "extension": self.extension,
            "parser": self.parser,
            "locations": [item.to_dict() for item in self.locations],
            "retrieval": self.retrieval.to_dict(),
            "final_rank": self.final_rank,
            "metadata": dict(self.metadata),
        }


def _location_from_mapping(value: Any) -> Optional[Location]:
    if not isinstance(value, Mapping):
        return None
    location = Location(
        block_id=value.get("block_id"),
        page=(
            value.get("page")
            if value.get("page") is not None
            else value.get("page_number")
        ),
        section_path=value.get("section_path"),
        table_id=value.get("table_id"),
        row=(
            value.get("row")
            if value.get("row") is not None
            else value.get("row_index")
        ),
        column=(
            value.get("column")
            if value.get("column") is not None
            else value.get("column_index")
        ),
    )
    if all(item is None for item in location.to_dict().values()):
        return None
    return location


def _iter_explicit_location_values(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Iterable[Any]:
    for source in (row, metadata):
        for key in ("locations", "block_locations"):
            values = source.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes)
            ):
                yield from values
        for key in ("blocks", "source_blocks"):
            values = source.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes)
            ):
                yield from values
        for key in ("location", "block"):
            value = source.get(key)
            if isinstance(value, Mapping):
                yield value

    # A row may itself be a canonical block or an explicitly joined block.
    if "block_id" in row and any(
        key in row
        for key in (
            "page",
            "page_number",
            "section_path",
            "table_id",
            "row",
            "row_index",
            "column",
            "column_index",
        )
    ):
        yield row


def locations_from_row(row: Mapping[str, Any]) -> Tuple[Location, ...]:
    """Return only locations supported by explicit block/location metadata."""

    metadata_value = row.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    result = []
    seen = set()

    for value in _iter_explicit_location_values(row, metadata):
        location = _location_from_mapping(value)
        if location is None:
            continue
        key = (
            location.block_id,
            location.page,
            location.section_path,
            location.table_id,
            location.row,
            location.column,
        )
        if key not in seen:
            seen.add(key)
            result.append(location)

    # Legacy structured chunks only carry block IDs.  Retain that provenance,
    # but do not copy a page range or table ID onto every block: those values
    # are not exact block locations and would fabricate precision.
    block_ids = metadata.get("block_ids", row.get("block_ids"))
    if isinstance(block_ids, Sequence) and not isinstance(
        block_ids, (str, bytes)
    ):
        known_ids = {item.block_id for item in result}
        for value in block_ids:
            block_id = _optional_string(value)
            if block_id is None or block_id in known_ids:
                continue
            result.append(Location(block_id=block_id))
            known_ids.add(block_id)

    return tuple(result)


def _value_from_row_or_metadata(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    key: str,
) -> Any:
    value = row.get(key)
    return value if value is not None else metadata.get(key)


def normalize_search_hit(
    row: Mapping[str, Any],
    *,
    stage: Optional[str] = "bm25",
    rank: Optional[int] = None,
    stage_kind: Optional[str] = None,
) -> SearchHit:
    """Normalize legacy or new chunk/search rows into :class:`SearchHit`.

    Legacy ``doc_id`` is accepted on input.  Serialization always emits
    ``document_id``.
    """

    if isinstance(row, SearchHit):
        hit = row
    elif not isinstance(row, Mapping):
        raise TypeError("search hit row must be a mapping")
    else:
        metadata_value = row.get("metadata")
        metadata = (
            dict(metadata_value)
            if isinstance(metadata_value, Mapping)
            else {}
        )
        document_id = row.get("document_id") or row.get("doc_id")
        text_value = row.get("text")
        preview_value = row.get("preview")
        text = (
            text_value
            if isinstance(text_value, str)
            else (
                preview_value if isinstance(preview_value, str) else ""
            )
        )
        preview = (
            preview_value
            if isinstance(preview_value, str)
            else text
        )
        hit = SearchHit(
            chunk_id=row.get("chunk_id"),
            document_id=document_id,
            chunk_index=row.get("chunk_index"),
            text=text,
            preview=preview,
            char_count=(
                row.get("char_count")
                if row.get("char_count") is not None
                else len(text)
            ),
            institution=_value_from_row_or_metadata(
                row, metadata, "institution"
            ),
            file_name=_value_from_row_or_metadata(
                row, metadata, "file_name"
            ),
            source_path=_value_from_row_or_metadata(
                row, metadata, "source_path"
            ),
            relative_path=_value_from_row_or_metadata(
                row, metadata, "relative_path"
            ),
            extension=_value_from_row_or_metadata(
                row, metadata, "extension"
            ),
            parser=_value_from_row_or_metadata(row, metadata, "parser"),
            locations=locations_from_row(row),
            retrieval=RetrievalScores.from_mapping(row.get("retrieval")),
            final_rank=row.get("final_rank"),
            metadata=metadata,
        )

    if stage is not None:
        if stage not in RETRIEVAL_STAGES:
            raise ValueError("unknown retrieval stage: {}".format(stage))
        existing = getattr(hit.retrieval, stage)
        score = (
            row.get("score")
            if isinstance(row, Mapping) and row.get("score") is not None
            else (existing.score if existing is not None else None)
        )
        resolved_rank = (
            rank
            if rank is not None
            else (
                row.get("rank")
                if isinstance(row, Mapping)
                else None
            )
        )
        if resolved_rank is None and existing is not None:
            resolved_rank = existing.rank
        kind = (
            stage_kind
            or (existing.kind if existing is not None else None)
            or stage
        )
        if score is not None or resolved_rank is not None or existing is not None:
            hit = hit.with_stage(
                stage,
                rank=resolved_rank,
                score=score,
                kind=kind,
            )
    return hit


@dataclass(frozen=True)
class CitationV1:
    """Citation payload tied to a retrieved chunk and its exact locations."""

    chunk_id: str
    document_id: str
    excerpt: str
    locations: Tuple[Location, ...] = ()
    source_number: Optional[int] = None
    citation_id: Optional[str] = None
    institution: Optional[str] = None
    file_name: Optional[str] = None
    source_path: Optional[str] = None
    relative_path: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "chunk_id", _non_empty_string(self.chunk_id, "chunk_id")
        )
        object.__setattr__(
            self,
            "document_id",
            _non_empty_string(self.document_id, "document_id"),
        )
        if not isinstance(self.excerpt, str):
            object.__setattr__(self, "excerpt", str(self.excerpt or ""))
        object.__setattr__(self, "locations", tuple(self.locations))
        object.__setattr__(
            self,
            "source_number",
            _optional_int(self.source_number, minimum=1),
        )
        for name in (
            "citation_id",
            "institution",
            "file_name",
            "source_path",
            "relative_path",
        ):
            object.__setattr__(
                self, name, _optional_string(getattr(self, name))
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "source_number": self.source_number,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "excerpt": self.excerpt,
            "locations": [item.to_dict() for item in self.locations],
            "institution": self.institution,
            "file_name": self.file_name,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
        }


def citation_from_hit(
    hit: SearchHit | Mapping[str, Any],
    retrieved_chunk_ids: Iterable[str | SearchHit | Mapping[str, Any]],
    *,
    excerpt: Optional[str] = None,
    source_number: Optional[int] = None,
    citation_id: Optional[str] = None,
) -> CitationV1:
    """Build a citation only when ``hit`` belongs to the retrieved result set."""

    canonical = (
        hit
        if isinstance(hit, SearchHit)
        else normalize_search_hit(hit, stage=None)
    )
    allowed = set()
    for value in retrieved_chunk_ids:
        if isinstance(value, SearchHit):
            allowed.add(value.chunk_id)
        elif isinstance(value, Mapping):
            chunk_id = _optional_string(value.get("chunk_id"))
            if chunk_id is not None:
                allowed.add(chunk_id)
        else:
            chunk_id = _optional_string(value)
            if chunk_id is not None:
                allowed.add(chunk_id)
    if canonical.chunk_id not in allowed:
        raise ValueError(
            "citation source was not present in retrieved chunks: {}".format(
                canonical.chunk_id
            )
        )
    return CitationV1(
        citation_id=citation_id,
        source_number=source_number,
        chunk_id=canonical.chunk_id,
        document_id=canonical.document_id,
        excerpt=(
            excerpt
            if excerpt is not None
            else (canonical.preview or canonical.text)
        ),
        # Reuse the canonical tuple verbatim; do not enrich or infer locations.
        locations=canonical.locations,
        institution=canonical.institution,
        file_name=canonical.file_name,
        source_path=canonical.source_path,
        relative_path=canonical.relative_path,
    )
