"""Validated corpus ingestion for the retrieval indexes.

Parser runs are immutable artifacts.  This module is the boundary between
those artifacts and the mutable retrieval indexes: it verifies the primary
manifest checksums, applies the quality gate, validates every canonical Block,
and exposes citation locations without inventing missing coordinates.

Legacy ``chunks.jsonl`` files without a run manifest remain supported so the
existing MVP index can still be rebuilt.  They receive a content-hash revision
and no block-level locations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PRIMARY_RUN_FILES = (
    "documents.jsonl",
    "blocks.jsonl",
    "chunks.jsonl",
    "attempts.jsonl",
    "parse_report.csv",
    "parse_summary.json",
)


@dataclass(frozen=True)
class CorpusGateReport:
    """The immutable facts retrieval builders may use from a parser run."""

    corpus_revision: str
    run_id: str | None
    profile: str | None
    manifest_sha256: str
    allowed_document_ids: frozenset[str] | None
    excluded_document_ids: frozenset[str] = frozenset()
    excluded_chunk_ids: frozenset[str] = frozenset()
    block_locations: dict[str, dict[str, Any]] = field(default_factory=dict)
    table_locations: dict[tuple[str, str], tuple[dict[str, Any], ...]] = field(
        default_factory=dict
    )

    @property
    def is_verified_run(self) -> bool:
        return self.run_id is not None

    def allows_chunk(self, chunk: dict[str, Any]) -> bool:
        chunk_id = str(chunk.get("chunk_id") or "")
        document_id = str(chunk.get("document_id") or chunk.get("doc_id") or "")
        if not chunk_id or chunk_id in self.excluded_chunk_ids:
            return False
        if self.allowed_document_ids is not None and document_id not in self.allowed_document_ids:
            return False
        return bool(str(chunk.get("text") or "").strip())

    def locations_for_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Return canonical block locations for a chunk.

        A table chunk currently references its parent table block.  We expand
        that parent to its canonical cells here so row/column citations are
        retained while table-cell text remains excluded from retrieval.
        """

        metadata = chunk.get("metadata") or {}
        document_id = str(chunk.get("document_id") or chunk.get("doc_id") or "")
        locations: list[dict[str, Any]] = []
        seen: set[str] = set()

        for block_id in _string_list(metadata.get("block_ids")):
            location = self.block_locations.get(block_id)
            if location and block_id not in seen:
                locations.append(dict(location))
                seen.add(block_id)

        for table_id in _string_list(metadata.get("table_ids")):
            for location in self.table_locations.get((document_id, table_id), ()):
                block_id = str(location.get("block_id") or "")
                if block_id and block_id not in seen:
                    locations.append(dict(location))
                    seen.add(block_id)

        return locations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _canonical_location(block: Any) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "page": block.page,
        "section_path": list(block.section_path) if block.section_path is not None else None,
        "table_id": block.table_id,
        "row": block.row,
        "column": block.column,
    }


def _load_manifest(run_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid run manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Run manifest must be an object: {manifest_path}")
    if manifest.get("block_schema_version") != 1:
        raise RuntimeError(
            f"Unsupported block_schema_version={manifest.get('block_schema_version')!r}"
        )
    checksums = manifest.get("files")
    if not isinstance(checksums, dict):
        raise RuntimeError("Run manifest is missing primary file checksums")
    for file_name in PRIMARY_RUN_FILES:
        expected = checksums.get(file_name)
        path = run_dir / file_name
        if not isinstance(expected, str) or not path.is_file():
            raise RuntimeError(f"Run manifest is missing {file_name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Corpus checksum mismatch for {file_name}: expected {expected}, got {actual}"
            )
    return manifest, sha256_file(manifest_path)


def inspect_corpus(
    chunks_path: Path,
    *,
    allow_suspect: bool = False,
    require_manifest: bool = False,
) -> CorpusGateReport:
    """Inspect and gate a chunks source before either index is built."""

    chunks_path = chunks_path.resolve()
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    run_dir = chunks_path.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        if require_manifest:
            raise RuntimeError(f"A verified parser run is required: {manifest_path}")
        chunks_hash = sha256_file(chunks_path)
        excluded = frozenset(
            str(chunk.get("chunk_id") or "")
            for chunk in iter_jsonl(chunks_path)
            if not str(chunk.get("text") or "").strip()
        )
        return CorpusGateReport(
            corpus_revision=f"legacy:{chunks_hash[:24]}",
            run_id=None,
            profile=None,
            manifest_sha256=chunks_hash,
            allowed_document_ids=None,
            excluded_chunk_ids=excluded,
        )

    manifest, manifest_hash = _load_manifest(run_dir)
    allowed_documents: set[str] = set()
    excluded_documents: set[str] = set()
    for document in iter_jsonl(run_dir / "documents.jsonl"):
        document_id = str(document.get("document_id") or "")
        quality = document.get("quality") or {}
        decision = str(quality.get("decision") or "")
        allowed = (
            bool(document_id)
            and document.get("status") == "parsed"
            and (decision == "pass" or (allow_suspect and decision == "suspect"))
        )
        (allowed_documents if allowed else excluded_documents).add(document_id)
    if not allowed_documents:
        raise RuntimeError("Corpus gate rejected every document")

    # Importing the parser model keeps this boundary pinned to the exact v1
    # schema, including rejection of both missing and extra keys.
    try:
        from ..document_parsing.core.models import Block
    except ImportError:  # Direct CLI imports with scripts/ on sys.path.
        from document_parsing.core.models import Block

    block_locations: dict[str, dict[str, Any]] = {}
    table_locations_mutable: dict[tuple[str, str], list[dict[str, Any]]] = {}
    block_documents: dict[str, str] = {}
    for payload in iter_jsonl(run_dir / "blocks.jsonl"):
        block = Block.from_dict(payload)
        if block.document_id not in allowed_documents:
            continue
        location = _canonical_location(block)
        block_locations[block.block_id] = location
        block_documents[block.block_id] = block.document_id
        if block.table_id:
            table_locations_mutable.setdefault(
                (block.document_id, block.table_id), []
            ).append(location)

    excluded_chunks: set[str] = set()
    for chunk in iter_jsonl(chunks_path):
        chunk_id = str(chunk.get("chunk_id") or "")
        document_id = str(chunk.get("document_id") or chunk.get("doc_id") or "")
        text = str(chunk.get("text") or "")
        if document_id not in allowed_documents or not text.strip():
            excluded_chunks.add(chunk_id)
            continue
        metadata = chunk.get("metadata") or {}
        for block_id in _string_list(metadata.get("block_ids")):
            if block_id not in block_locations:
                raise RuntimeError(
                    f"Chunk {chunk_id} references a missing or quarantined block {block_id}"
                )
            if block_documents[block_id] != document_id:
                raise RuntimeError(
                    f"Chunk {chunk_id} references block {block_id} from another document"
                )

    run_id = str(manifest.get("run_id") or run_dir.parent.name)
    profile = str(manifest.get("profile") or run_dir.name)
    table_locations = {
        key: tuple(sorted(values, key=lambda item: str(item["block_id"])))
        for key, values in table_locations_mutable.items()
    }
    return CorpusGateReport(
        corpus_revision=f"{run_id}:{profile}:{manifest_hash[:24]}",
        run_id=run_id,
        profile=profile,
        manifest_sha256=manifest_hash,
        allowed_document_ids=frozenset(allowed_documents),
        excluded_document_ids=frozenset(excluded_documents),
        excluded_chunk_ids=frozenset(excluded_chunks),
        block_locations=block_locations,
        table_locations=table_locations,
    )
