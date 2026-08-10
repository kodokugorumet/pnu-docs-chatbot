"""Canonical parser data models.

The canonical block is deliberately small. Parser-specific geometry, OCR
confidence, merge spans, and other rich output belong in ``raw_artifacts`` or
attempt metadata so every adapter can emit the same stable eleven fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


BLOCK_FIELDS = (
    "document_id",
    "block_id",
    "block_type",
    "text",
    "reading_order",
    "parser",
    "page",
    "section_path",
    "table_id",
    "row",
    "column",
)

BLOCK_TYPES = frozenset(
    {
        "heading",
        "paragraph",
        "list_item",
        "table",
        "table_cell",
        "caption",
    }
)

ATTEMPT_STATUSES = frozenset(
    {
        "success",
        "suspect",
        "empty",
        "error",
        "timeout",
        "unavailable",
        "unsupported",
        "skipped",
    }
)


@dataclass(frozen=True)
class Block:
    """A canonical document block with exactly the v1 eleven-field schema."""

    document_id: str
    block_id: str
    block_type: str
    text: str
    reading_order: int
    parser: str
    page: Optional[int] = None
    section_path: Optional[Sequence[str]] = None
    table_id: Optional[str] = None
    row: Optional[int] = None
    column: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("block_id must be a non-empty string")
        if self.block_type not in BLOCK_TYPES:
            raise ValueError(
                "block_type must be one of {}".format(", ".join(sorted(BLOCK_TYPES)))
            )
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if (
            isinstance(self.reading_order, bool)
            or not isinstance(self.reading_order, int)
            or self.reading_order < 0
        ):
            raise ValueError("reading_order must be a zero-based non-negative integer")
        if not isinstance(self.parser, str) or not self.parser:
            raise ValueError("parser must be a non-empty string")
        if self.page is not None and (
            isinstance(self.page, bool)
            or not isinstance(self.page, int)
            or self.page < 1
        ):
            raise ValueError("page must be null or a one-based positive integer")
        if self.section_path is not None:
            if isinstance(self.section_path, (str, bytes)):
                raise TypeError("section_path must be null or a sequence of strings")
            normalized_path = tuple(self.section_path)
            if any(not isinstance(item, str) or not item for item in normalized_path):
                raise ValueError("section_path entries must be non-empty strings")
            object.__setattr__(self, "section_path", normalized_path)
        if self.row is not None and (
            isinstance(self.row, bool) or not isinstance(self.row, int) or self.row < 0
        ):
            raise ValueError("row must be null or a zero-based non-negative integer")
        if self.column is not None and (
            isinstance(self.column, bool)
            or not isinstance(self.column, int)
            or self.column < 0
        ):
            raise ValueError("column must be null or a zero-based non-negative integer")
        if self.table_id is not None and (
            not isinstance(self.table_id, str) or not self.table_id
        ):
            raise ValueError("table_id must be null or a non-empty string")

        if self.block_type in {"table", "table_cell"} and not self.table_id:
            raise ValueError("{} blocks require table_id".format(self.block_type))
        if self.block_type == "table_cell":
            if self.row is None or self.column is None:
                raise ValueError("table_cell blocks require row and column")
        elif self.row is not None or self.column is not None:
            raise ValueError("row and column are only valid for table_cell blocks")

    def to_dict(self) -> Dict[str, Any]:
        """Return all eleven fields in their canonical order, including nulls."""

        return {
            "document_id": self.document_id,
            "block_id": self.block_id,
            "block_type": self.block_type,
            "text": self.text,
            "reading_order": self.reading_order,
            "parser": self.parser,
            "page": self.page,
            "section_path": (
                list(self.section_path) if self.section_path is not None else None
            ),
            "table_id": self.table_id,
            "row": self.row,
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Block":
        missing = [name for name in BLOCK_FIELDS if name not in value]
        extra = [name for name in value if name not in BLOCK_FIELDS]
        if missing or extra:
            details = []
            if missing:
                details.append("missing={}".format(",".join(missing)))
            if extra:
                details.append("extra={}".format(",".join(sorted(extra))))
            raise ValueError("invalid canonical block fields: " + " ".join(details))
        return cls(**{name: value[name] for name in BLOCK_FIELDS})


@dataclass(frozen=True)
class SourceDocument:
    """Input identity and source metadata shared by every parser adapter."""

    path: Path
    document_id: str
    relative_path: str
    source_sha256: str = ""
    source_path: str = ""
    file_name: str = ""
    extension: str = ""
    size_bytes: int = 0
    institution: str = ""
    mime_type: Optional[str] = None
    profile: str = ""
    selected_parser: Optional[str] = None
    quality: Optional[Mapping[str, Any]] = None
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    download_url: Optional[str] = None
    source_host: Optional[str] = None
    fetched_at: Optional[str] = None
    published_at: Optional[str] = None
    category: Optional[str] = None
    include_reason: Optional[str] = None
    source_aliases: Sequence[str] = field(default_factory=tuple)
    crawl_storage_path: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.document_id:
            raise ValueError("document_id must be non-empty")
        if not self.relative_path:
            raise ValueError("relative_path must be non-empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not self.source_path:
            object.__setattr__(self, "source_path", self.relative_path)
        if not self.file_name:
            object.__setattr__(self, "file_name", self.path.name)
        if not self.extension:
            object.__setattr__(self, "extension", self.path.suffix.lower())
        if self.quality is not None:
            object.__setattr__(self, "quality", dict(self.quality))
        optional_text_fields = (
            "source_title",
            "source_url",
            "download_url",
            "source_host",
            "fetched_at",
            "published_at",
            "category",
            "include_reason",
            "crawl_storage_path",
        )
        for name in optional_text_fields:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError("{} must be null or a string".format(name))
        if isinstance(self.source_aliases, (str, bytes)):
            raise TypeError("source_aliases must be a sequence of strings")
        aliases = tuple(self.source_aliases)
        if any(not isinstance(value, str) for value in aliases):
            raise TypeError("source_aliases must contain only strings")
        object.__setattr__(self, "source_aliases", aliases)

    @property
    def doc_id(self) -> str:
        """Compatibility alias used by the current BM25 records."""

        return self.document_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "source_sha256": self.source_sha256,
            "institution": self.institution,
            "mime_type": self.mime_type,
            "profile": self.profile,
            "selected_parser": self.selected_parser,
            "quality": dict(self.quality) if self.quality is not None else None,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "download_url": self.download_url,
            "source_host": self.source_host,
            "fetched_at": self.fetched_at,
            "published_at": self.published_at,
            "category": self.category,
            "include_reason": self.include_reason,
            "source_aliases": list(self.source_aliases),
            "crawl_storage_path": self.crawl_storage_path,
        }


@dataclass(frozen=True)
class Attempt:
    """One adapter execution, including failures that caused a fallback."""

    parser: str
    status: str
    reason: str = ""
    elapsed_ms: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    exit_code: Optional[int] = None
    stderr: Optional[str] = None
    timeout: bool = False
    parser_version: Optional[str] = None
    model: Optional[str] = None
    device: Optional[str] = None
    fallback_reason: Optional[str] = None
    peak_memory_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.parser:
            raise ValueError("attempt parser must be non-empty")
        if self.status not in ATTEMPT_STATUSES:
            raise ValueError(
                "attempt status must be one of {}".format(
                    ", ".join(sorted(ATTEMPT_STATUSES))
                )
            )
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parser": self.parser,
            "status": self.status,
            "reason": self.reason,
            "elapsed_ms": self.elapsed_ms,
            "metadata": dict(self.metadata),
            "error_type": self.error_type,
            "exit_code": self.exit_code,
            "stderr": self.stderr,
            "timeout": self.timeout,
            "parser_version": self.parser_version,
            "model": self.model,
            "device": self.device,
            "fallback_reason": self.fallback_reason,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass
class ParseResult:
    """Normalized result returned by all adapters and profile pipelines."""

    document: SourceDocument
    blocks: List[Block] = field(default_factory=list)
    attempts: List[Attempt] = field(default_factory=list)
    raw_artifacts: List[Any] = field(default_factory=list)

    @property
    def source(self) -> SourceDocument:
        """Compatibility alias for adapters that call the input ``source``."""

        return self.document

    def validate(self) -> None:
        seen_ids = set()
        orders = []
        for block in self.blocks:
            if block.document_id != self.document.document_id:
                raise ValueError(
                    "block {} belongs to {}, expected {}".format(
                        block.block_id,
                        block.document_id,
                        self.document.document_id,
                    )
                )
            if block.block_id in seen_ids:
                raise ValueError("duplicate block_id: {}".format(block.block_id))
            seen_ids.add(block.block_id)
            orders.append(block.reading_order)
        if orders and sorted(orders) != list(range(len(orders))):
            raise ValueError("reading_order must be contiguous from zero")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "raw_artifacts": list(self.raw_artifacts),
        }
