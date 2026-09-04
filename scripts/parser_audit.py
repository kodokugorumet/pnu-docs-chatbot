#!/usr/bin/env python3
"""Shared schema and artifact helpers for the parser-profile audit.

The parser audit is intentionally separate from retrieval and answer-generation
evaluation.  It asks a narrower question: does each parser profile preserve a
small, source-bound set of atomic facts in its run output and search index?
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PROFILES = ("baseline", "challenger", "cascade")
GROUPS = ("hwp_hwpx", "digital_pdf", "ocr_table_pdf")
GROUP_TARGETS = {group: 6 for group in GROUPS}
ANCHORS_PER_DOCUMENT = 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOCUMENT_ID_RE = re.compile(r"^doc_[0-9a-f]{24}$")
ANCHOR_KINDS = frozenset(
    {"text", "numeric", "date", "procedure", "table_relation"}
)


DEFAULT_ARTIFACTS = {
    "baseline": (
        "processed/runs/20260725-pnu-curated-baseline-v5/baseline",
        "processed/index/pnu-20260725-curated-baseline-v5-allow-suspect.sqlite",
    ),
    "challenger": (
        "processed/runs/20260725-pnu-curated-challenger-v5/challenger",
        "processed/index/pnu-20260725-curated-challenger-v5-allow-suspect.sqlite",
    ),
    "cascade": (
        "processed/runs/20260725-pnu-curated-cascade-v5/cascade",
        "processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite",
    ),
}


class AuditValidationError(ValueError):
    """Raised when the audit set or one of its bound artifacts is invalid."""


@dataclass(frozen=True)
class ProfileArtifact:
    profile: str
    run_dir: Path
    index_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_anchor(value: str) -> str:
    """Return a conservative format-tolerant key for exact anchor matching.

    NFKC handles full-width forms and compatibility characters.  Removing only
    non-alphanumeric characters tolerates parser-specific whitespace and table
    separators without introducing synonym or fuzzy semantic matching.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def anchor_variants(anchor: Mapping[str, Any]) -> tuple[str, ...]:
    values = [anchor.get("text")]
    values.extend(anchor.get("accepted_variants") or [])
    return tuple(str(value) for value in values if isinstance(value, str) and value)


def find_anchor(text: str, anchor: Mapping[str, Any]) -> str | None:
    haystack = normalize_anchor(text)
    for variant in anchor_variants(anchor):
        needle = normalize_anchor(variant)
        if needle and needle in haystack:
            return variant
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditValidationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise AuditValidationError(
                    f"{path}:{line_number}: each JSONL row must be an object"
                )
            record["_line_number"] = line_number
            records.append(record)
    return records


def _require_string(
    value: Any,
    field: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AuditValidationError(f"{field} must be a non-empty trimmed string")
    if pattern is not None and not pattern.fullmatch(value):
        raise AuditValidationError(f"{field} has an invalid format: {value!r}")
    return value


def _safe_repo_path(repo_root: Path, raw_path: str, field: str) -> Path:
    value = Path(raw_path)
    if value.is_absolute() or ".." in value.parts:
        raise AuditValidationError(f"{field} must be a safe repository-relative path")
    resolved_root = repo_root.resolve()
    resolved = (repo_root / value).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AuditValidationError(f"{field} escapes the repository") from exc
    return resolved


def validate_case_schema(record: Mapping[str, Any], repo_root: Path) -> None:
    line = record.get("_line_number", "?")
    prefix = f"line {line}"
    if record.get("schema_version") != SCHEMA_VERSION:
        raise AuditValidationError(
            f"{prefix}: schema_version must equal {SCHEMA_VERSION}"
        )
    case_id = _require_string(record.get("id"), f"{prefix}.id")
    group = _require_string(record.get("group"), f"{case_id}.group")
    if group not in GROUPS:
        raise AuditValidationError(f"{case_id}.group must be one of {GROUPS}")

    source = record.get("source")
    if not isinstance(source, dict):
        raise AuditValidationError(f"{case_id}.source must be an object")
    document_id = _require_string(
        source.get("document_id"),
        f"{case_id}.source.document_id",
        pattern=DOCUMENT_ID_RE,
    )
    source_path = _require_string(
        source.get("source_path"), f"{case_id}.source.source_path"
    )
    relative_path = _require_string(
        source.get("relative_path"), f"{case_id}.source.relative_path"
    )
    extension = _require_string(
        source.get("extension"), f"{case_id}.source.extension"
    ).lower()
    source_sha256 = _require_string(
        source.get("source_sha256"),
        f"{case_id}.source.source_sha256",
        pattern=SHA256_RE,
    )
    _require_string(source.get("source_title"), f"{case_id}.source.source_title")
    if group == "hwp_hwpx" and extension not in {".hwp", ".hwpx"}:
        raise AuditValidationError(f"{case_id}: hwp_hwpx requires .hwp or .hwpx")
    if group in {"digital_pdf", "ocr_table_pdf"} and extension != ".pdf":
        raise AuditValidationError(f"{case_id}: {group} requires .pdf")
    if not relative_path.endswith(extension):
        raise AuditValidationError(
            f"{case_id}: relative_path does not end with {extension}"
        )

    resolved_source = _safe_repo_path(
        repo_root, source_path, f"{case_id}.source.source_path"
    )
    if not resolved_source.is_file():
        raise AuditValidationError(f"{case_id}: raw source is missing: {source_path}")
    actual_sha256 = sha256_file(resolved_source)
    if actual_sha256 != source_sha256:
        raise AuditValidationError(
            f"{case_id}: raw source SHA-256 mismatch: "
            f"expected {source_sha256}, got {actual_sha256}"
        )

    route = record.get("route")
    if not isinstance(route, dict):
        raise AuditValidationError(f"{case_id}.route must be an object")
    route_kind = _require_string(route.get("kind"), f"{case_id}.route.kind")
    selected_parsers = route.get("cascade_selected_parsers")
    if not isinstance(selected_parsers, list) or not selected_parsers or any(
        not isinstance(value, str) or not value for value in selected_parsers
    ):
        raise AuditValidationError(
            f"{case_id}.route.cascade_selected_parsers must be a non-empty string list"
        )
    table_count = route.get("cascade_table_count")
    if not isinstance(table_count, int) or table_count < 0:
        raise AuditValidationError(
            f"{case_id}.route.cascade_table_count must be a non-negative integer"
        )
    parser_key = " ".join(selected_parsers).casefold()
    if group == "hwp_hwpx" and route_kind != "hwp_structure":
        raise AuditValidationError(f"{case_id}: hwp_hwpx requires hwp_structure")
    if group == "digital_pdf":
        if route_kind != "digital_text":
            raise AuditValidationError(f"{case_id}: digital_pdf requires digital_text")
        if "pp-structure" in parser_key or "tesseract" in parser_key:
            raise AuditValidationError(
                f"{case_id}: digital_pdf may not declare an OCR parser"
            )
    if group == "ocr_table_pdf":
        if route_kind not in {"ocr", "table"}:
            raise AuditValidationError(
                f"{case_id}: ocr_table_pdf requires route kind ocr or table"
            )
        if route_kind == "ocr" and not (
            "pp-structure" in parser_key or "tesseract" in parser_key
        ):
            raise AuditValidationError(
                f"{case_id}: OCR route lacks OCR parser evidence"
            )
        if route_kind == "table" and table_count <= 0:
            raise AuditValidationError(
                f"{case_id}: table route requires a positive table count"
            )

    anchors = record.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != ANCHORS_PER_DOCUMENT:
        raise AuditValidationError(
            f"{case_id}.anchors must contain exactly {ANCHORS_PER_DOCUMENT} items"
        )
    local_ids: set[str] = set()
    local_texts: set[str] = set()
    for offset, anchor in enumerate(anchors, start=1):
        if not isinstance(anchor, dict):
            raise AuditValidationError(f"{case_id}.anchors[{offset}] must be an object")
        anchor_id = _require_string(
            anchor.get("id"), f"{case_id}.anchors[{offset}].id"
        )
        if anchor_id in local_ids:
            raise AuditValidationError(f"{case_id}: duplicate anchor id {anchor_id}")
        local_ids.add(anchor_id)
        kind = _require_string(
            anchor.get("kind"), f"{case_id}.{anchor_id}.kind"
        )
        if kind not in ANCHOR_KINDS:
            raise AuditValidationError(
                f"{case_id}.{anchor_id}.kind must be one of {sorted(ANCHOR_KINDS)}"
            )
        text = _require_string(
            anchor.get("text"), f"{case_id}.{anchor_id}.text"
        )
        normalized = normalize_anchor(text)
        if len(normalized) < 8:
            raise AuditValidationError(
                f"{case_id}.{anchor_id}.text is too short to be distinctive"
            )
        if normalized in local_texts:
            raise AuditValidationError(f"{case_id}: duplicate normalized anchor text")
        local_texts.add(normalized)
        variants = anchor.get("accepted_variants", [])
        if not isinstance(variants, list) or any(
            not isinstance(value, str) or not value.strip() for value in variants
        ):
            raise AuditValidationError(
                f"{case_id}.{anchor_id}.accepted_variants must be a string list"
            )
        hint = _require_string(
            anchor.get("source_chunk_hint"),
            f"{case_id}.{anchor_id}.source_chunk_hint",
        )
        if not hint.startswith(f"{document_id}:cascade#"):
            raise AuditValidationError(
                f"{case_id}.{anchor_id}: source_chunk_hint must name the same "
                "document in the cascade profile"
            )

    verification = record.get("verification")
    if not isinstance(verification, dict):
        raise AuditValidationError(f"{case_id}.verification must be an object")
    if verification.get("canonical_profile") != "cascade":
        raise AuditValidationError(
            f"{case_id}.verification.canonical_profile must be cascade"
        )
    if verification.get("method") != "raw-sha+run-chunk+index-chunk":
        raise AuditValidationError(
            f"{case_id}.verification.method must bind raw, run, and index"
        )


def validate_dataset_schema(
    records: Sequence[Mapping[str, Any]],
    repo_root: Path,
    *,
    enforce_balance: bool = True,
) -> dict[str, int]:
    if not records:
        raise AuditValidationError("audit dataset is empty")
    case_ids: set[str] = set()
    document_ids: set[str] = set()
    anchor_ids: set[str] = set()
    groups: Counter[str] = Counter()
    for record in records:
        validate_case_schema(record, repo_root)
        case_id = str(record["id"])
        document_id = str(record["source"]["document_id"])
        if case_id in case_ids:
            raise AuditValidationError(f"duplicate case id: {case_id}")
        if document_id in document_ids:
            raise AuditValidationError(f"duplicate document id: {document_id}")
        case_ids.add(case_id)
        document_ids.add(document_id)
        groups[str(record["group"])] += 1
        for anchor in record["anchors"]:
            anchor_id = str(anchor["id"])
            if anchor_id in anchor_ids:
                raise AuditValidationError(f"duplicate global anchor id: {anchor_id}")
            anchor_ids.add(anchor_id)
    if enforce_balance and dict(groups) != GROUP_TARGETS:
        raise AuditValidationError(
            f"group balance must equal {GROUP_TARGETS}, got {dict(groups)}"
        )
    expected_anchors = len(records) * ANCHORS_PER_DOCUMENT
    if len(anchor_ids) != expected_anchors:
        raise AuditValidationError(
            f"expected {expected_anchors} unique anchors, got {len(anchor_ids)}"
        )
    return {
        "documents": len(records),
        "anchors": len(anchor_ids),
        **{f"group_{group}": groups[group] for group in GROUPS},
    }


def load_index_meta(index_path: Path) -> dict[str, str]:
    with sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True) as connection:
        try:
            rows = connection.execute("SELECT key, value FROM index_meta").fetchall()
        except sqlite3.Error as exc:
            raise AuditValidationError(
                f"cannot read index_meta from {index_path}: {exc}"
            ) from exc
    return {str(key): str(value) for key, value in rows}


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditValidationError(f"{path} must contain a JSON object")
    return value


def validate_profile_artifact(artifact: ProfileArtifact) -> dict[str, Any]:
    if artifact.profile not in PROFILES:
        raise AuditValidationError(f"unknown profile: {artifact.profile}")
    if not artifact.run_dir.is_dir():
        raise AuditValidationError(f"missing run directory: {artifact.run_dir}")
    if not artifact.index_path.is_file():
        raise AuditValidationError(f"missing index: {artifact.index_path}")
    manifest = load_run_manifest(artifact.run_dir)
    if manifest.get("profile") != artifact.profile:
        raise AuditValidationError(
            f"{artifact.run_dir}: expected profile {artifact.profile}, "
            f"got {manifest.get('profile')!r}"
        )
    for relative in ("documents.jsonl", "chunks.jsonl"):
        path = artifact.run_dir / relative
        expected = (manifest.get("files") or {}).get(relative)
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise AuditValidationError(
                f"{artifact.run_dir}: manifest lacks a valid hash for {relative}"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise AuditValidationError(
                f"{path}: hash mismatch: expected {expected}, got {actual}"
            )
    meta = load_index_meta(artifact.index_path)
    if meta.get("profile") != artifact.profile:
        raise AuditValidationError(
            f"{artifact.index_path}: expected profile {artifact.profile}, "
            f"got {meta.get('profile')!r}"
        )
    if str(manifest.get("run_id")) != meta.get("run_id"):
        raise AuditValidationError(
            f"{artifact.profile}: run/index run_id mismatch"
        )
    run_source_sha = manifest.get("source_manifest_sha256") or (
        manifest.get("derived_from") or {}
    ).get("curated_manifest_sha256")
    index_source_sha = meta.get("source_manifest_sha256")
    if not run_source_sha or run_source_sha != index_source_sha:
        raise AuditValidationError(
            f"{artifact.profile}: run/index source manifest mismatch"
        )
    return {
        "profile": artifact.profile,
        "run_id": manifest.get("run_id"),
        "source_manifest_sha256": index_source_sha,
        "run_manifest_sha256": sha256_file(artifact.run_dir / "run_manifest.json"),
        "index_sha256": sha256_file(artifact.index_path),
        "index_chunk_count": int(meta.get("chunk_count", "0")),
    }


def selected_documents(
    documents_path: Path, document_ids: set[str]
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with documents_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            document_id = str(record.get("document_id") or "")
            if document_id in document_ids:
                found[document_id] = record
    return found


def selected_run_chunks(
    chunks_path: Path, document_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    found = {document_id: [] for document_id in document_ids}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            record = json.loads(raw_line)
            document_id = str(record.get("doc_id") or record.get("document_id") or "")
            if document_id in found:
                found[document_id].append(record)
    for records in found.values():
        records.sort(key=lambda value: int(value.get("chunk_index", 0)))
    return found


def selected_index_chunks(
    index_path: Path, document_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    found = {document_id: [] for document_id in document_ids}
    if not document_ids:
        return found
    placeholders = ",".join("?" for _ in document_ids)
    sql = (
        "SELECT chunk_id, doc_id, chunk_index, source_path, relative_path, "
        "extension, parser, source_title, text FROM chunks WHERE doc_id IN ("
        + placeholders
        + ") ORDER BY doc_id, chunk_index"
    )
    with sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(sql, tuple(document_ids)).fetchall()
    for row in rows:
        record = dict(row)
        found[str(record["doc_id"])].append(record)
    return found


def match_anchor_in_chunks(
    chunks: Iterable[Mapping[str, Any]], anchor: Mapping[str, Any]
) -> dict[str, Any] | None:
    for chunk in chunks:
        matched_variant = find_anchor(str(chunk.get("text") or ""), anchor)
        if matched_variant is not None:
            return {
                "chunk_id": chunk.get("chunk_id"),
                "matched_variant": matched_variant,
            }
    return None


def validate_canonical_bindings(
    records: Sequence[Mapping[str, Any]],
    artifact: ProfileArtifact,
) -> dict[str, int]:
    if artifact.profile != "cascade":
        raise AuditValidationError("canonical binding artifact must be cascade")
    validate_profile_artifact(artifact)
    document_ids = {str(record["source"]["document_id"]) for record in records}
    documents = selected_documents(artifact.run_dir / "documents.jsonl", document_ids)
    run_chunks = selected_run_chunks(artifact.run_dir / "chunks.jsonl", document_ids)
    index_chunks = selected_index_chunks(artifact.index_path, sorted(document_ids))
    verified = 0
    for record in records:
        case_id = str(record["id"])
        source = record["source"]
        document_id = str(source["document_id"])
        document = documents.get(document_id)
        if document is None:
            raise AuditValidationError(
                f"{case_id}: canonical run is missing document {document_id}"
            )
        for field in (
            "source_path",
            "relative_path",
            "source_sha256",
            "extension",
            "source_title",
        ):
            if document.get(field) != source.get(field):
                raise AuditValidationError(
                    f"{case_id}: canonical document {field} mismatch"
                )
        selected_parsers = document.get("selected_parsers") or []
        if selected_parsers != record["route"]["cascade_selected_parsers"]:
            raise AuditValidationError(
                f"{case_id}: cascade_selected_parsers does not match canonical run"
            )
        if document.get("table_count") != record["route"]["cascade_table_count"]:
            raise AuditValidationError(
                f"{case_id}: cascade_table_count does not match canonical run"
            )
        run_by_id = {
            str(chunk.get("chunk_id")): chunk for chunk in run_chunks.get(document_id, [])
        }
        index_by_id = {
            str(chunk.get("chunk_id")): chunk
            for chunk in index_chunks.get(document_id, [])
        }
        if not index_by_id:
            raise AuditValidationError(
                f"{case_id}: canonical index is missing document {document_id}"
            )
        for chunk in index_by_id.values():
            for field in (
                "source_path",
                "relative_path",
                "extension",
                "source_title",
            ):
                if chunk.get(field) != source.get(field):
                    raise AuditValidationError(
                        f"{case_id}: canonical index {field} mismatch"
                    )
        for anchor in record["anchors"]:
            hint = str(anchor["source_chunk_hint"])
            run_chunk = run_by_id.get(hint)
            if run_chunk is None:
                raise AuditValidationError(
                    f"{case_id}.{anchor['id']}: canonical run lacks hinted chunk {hint}"
                )
            if find_anchor(str(run_chunk.get("text") or ""), anchor) is None:
                raise AuditValidationError(
                    f"{case_id}.{anchor['id']}: anchor is absent from canonical run hint"
                )
            index_chunk = index_by_id.get(hint)
            if index_chunk is None:
                raise AuditValidationError(
                    f"{case_id}.{anchor['id']}: canonical index lacks hinted chunk {hint}"
                )
            if find_anchor(str(index_chunk.get("text") or ""), anchor) is None:
                raise AuditValidationError(
                    f"{case_id}.{anchor['id']}: anchor is absent from canonical index hint"
                )
            verified += 1
    return {"documents": len(records), "anchors": verified}
