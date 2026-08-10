"""Deterministic identifiers used by canonical parser outputs."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import PurePosixPath
from typing import Optional


SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_relative_path(value: str) -> str:
    """Normalize separators and Unicode without resolving against the host OS."""

    normalized = unicodedata.normalize("NFC", str(value)).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = str(PurePosixPath(normalized))
    if normalized in {"", "."}:
        raise ValueError("relative_path must be non-empty")
    if normalized.startswith("/"):
        raise ValueError("relative_path must not be absolute")
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("relative_path must not escape its data root")
    return normalized


def source_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_document_id(relative_path: str, digest: str) -> str:
    """Build ``doc_<24 hex>`` from path identity and the full source digest."""

    if not SHA256_RE.fullmatch(digest):
        raise ValueError("source digest must be a 64-character SHA-256 hex string")
    identity = "{}\0{}".format(
        normalize_relative_path(relative_path), digest.lower()
    ).encode("utf-8")
    return "doc_" + hashlib.sha256(identity).hexdigest()[:24]


def id_component(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).strip()
    component = re.sub(r"[^0-9A-Za-z._-]+", "-", normalized).strip("-")
    if not component:
        raise ValueError("ID component must contain at least one safe character")
    return component


def make_block_id(
    document_id: str,
    reading_order: int,
    profile: Optional[str] = None,
) -> str:
    if not document_id:
        raise ValueError("document_id must be non-empty")
    if (
        isinstance(reading_order, bool)
        or not isinstance(reading_order, int)
        or reading_order < 0
    ):
        raise ValueError("reading_order must be non-negative")
    prefix = document_id
    if profile:
        prefix += ":" + id_component(profile)
    return "{}:b{:06d}".format(prefix, reading_order)


def make_table_id(
    document_id: str,
    table_index: int,
    profile: Optional[str] = None,
) -> str:
    if not document_id:
        raise ValueError("document_id must be non-empty")
    if (
        isinstance(table_index, bool)
        or not isinstance(table_index, int)
        or table_index < 0
    ):
        raise ValueError("table_index must be non-negative")
    prefix = document_id
    if profile:
        prefix += ":" + id_component(profile)
    return "{}:t{:04d}".format(prefix, table_index)
