"""Stable core contracts for the document parsing pipelines."""

from .chunking import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    blocks_to_legacy_chunks,
)
from .ids import (
    id_component,
    make_block_id,
    make_document_id,
    make_table_id,
    normalize_relative_path,
    source_sha256,
)
from .jsonl import canonical_json, iter_jsonl, read_jsonl, write_jsonl
from .models import (
    ATTEMPT_STATUSES,
    BLOCK_FIELDS,
    BLOCK_TYPES,
    Attempt,
    Block,
    ParseResult,
    SourceDocument,
)
from .quality import (
    QualityAssessment,
    QualityMetrics,
    assess_quality,
    calculate_quality_metrics,
)
from .text import clean_text

__all__ = [
    "ATTEMPT_STATUSES",
    "BLOCK_FIELDS",
    "BLOCK_TYPES",
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_CHUNK_OVERLAP",
    "Attempt",
    "Block",
    "ParseResult",
    "QualityAssessment",
    "QualityMetrics",
    "SourceDocument",
    "assess_quality",
    "blocks_to_legacy_chunks",
    "calculate_quality_metrics",
    "canonical_json",
    "clean_text",
    "id_component",
    "iter_jsonl",
    "make_block_id",
    "make_document_id",
    "make_table_id",
    "normalize_relative_path",
    "read_jsonl",
    "source_sha256",
    "write_jsonl",
]
