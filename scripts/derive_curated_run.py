#!/usr/bin/env python3
"""Derive a verified immutable parser run from a curated corpus manifest.

The source parser run is verified before any selection occurs.  Selected
records are copied into a temporary run, referenced raw parser artifacts are
hard-linked when possible (copied otherwise), and the complete derived run is
verified before it is atomically published.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

# Direct CLI execution puts scripts/, not the repository root, on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.document_parsing.output import (
    DATA_FILES,
    REPORT_FIELDS,
    sha256_path,
    verify_profile_run,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESERVED_OUTPUT_FILES = frozenset(DATA_FILES) | frozenset({"run_manifest.json"})
REQUIRED_MANIFEST_FIELDS = frozenset(
    {"input_relative_path", "sha256", "size_bytes"}
)
OPTIONAL_MANIFEST_FIELDS = frozenset(
    {
        "source_title",
        "source_url",
        "download_url",
        "source_host",
        "fetched_at",
        "published_at",
        "crawl_storage_path",
        "source_aliases",
        "category",
        "include_reason",
    }
)
ALLOWED_MANIFEST_FIELDS = REQUIRED_MANIFEST_FIELDS | OPTIONAL_MANIFEST_FIELDS


class DerivationError(RuntimeError):
    """Raised when a derived immutable run cannot be created safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DerivationError("{} must be a non-empty string".format(label))
    rendered = value.replace("\\", "/")
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise DerivationError("{} is unsafe: {!r}".format(label, value))
    result = pure.as_posix()
    if not result:
        raise DerivationError("{} must be non-empty".format(label))
    return result


def _validate_http_url(value: str, field: str, line_number: int) -> str:
    if (
        not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise DerivationError(
            "curated manifest line {} {} must be a non-empty HTTP(S) URL".format(
                line_number,
                field,
            )
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise DerivationError(
            "curated manifest line {} {} has an invalid URL: {}".format(
                line_number,
                field,
                exc,
            )
        )
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or (port is not None and not (1 <= port <= 65535))
    ):
        raise DerivationError(
            "curated manifest line {} {} must be an HTTP(S) URL".format(
                line_number,
                field,
            )
        )
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _iter_jsonl(path: Path, label: str) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DerivationError(
                        "{}:{} is invalid JSON: {}".format(
                            label,
                            line_number,
                            exc,
                        )
                    )
                if not isinstance(value, dict):
                    raise DerivationError(
                        "{}:{} must be an object".format(label, line_number)
                    )
                yield value
    except (OSError, UnicodeError) as exc:
        raise DerivationError(
            "cannot read {}: {}: {}".format(label, type(exc).__name__, exc)
        )


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DerivationError(
            "{} is invalid: {}: {}".format(label, type(exc).__name__, exc)
        )
    if not isinstance(value, dict):
        raise DerivationError("{} must contain an object".format(label))
    return value


def _read_curated_manifest(
    path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], str, int]:
    if path.is_symlink():
        raise DerivationError("curated manifest must not be a symlink")
    if not path.is_file():
        raise DerivationError("curated manifest is not a file: {}".format(path))
    records: Dict[str, Dict[str, Any]] = {}
    total_bytes = 0
    for line_number, record in enumerate(
        _iter_jsonl(path, "curated manifest"),
        start=1,
    ):
        missing = sorted(REQUIRED_MANIFEST_FIELDS - set(record))
        if missing:
            raise DerivationError(
                "curated manifest line {} is missing fields: {}".format(
                    line_number,
                    ", ".join(missing),
                )
            )
        unexpected = sorted(set(record) - ALLOWED_MANIFEST_FIELDS)
        if unexpected:
            raise DerivationError(
                "curated manifest line {} has unsupported fields: {}".format(
                    line_number,
                    ", ".join(unexpected),
                )
            )
        relative_path = _safe_relative_path(
            record.get("input_relative_path"),
            "input_relative_path",
        )
        digest = str(record.get("sha256") or "").lower()
        if not SHA256_RE.fullmatch(digest):
            raise DerivationError(
                "curated manifest line {} has invalid sha256".format(line_number)
            )
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DerivationError(
                "curated manifest line {} has invalid size_bytes".format(
                    line_number
                )
            )
        if relative_path in records:
            raise DerivationError(
                "curated manifest contains duplicate input_relative_path: {}".format(
                    relative_path
                )
            )
        normalized = dict(record)
        normalized["input_relative_path"] = relative_path
        normalized["sha256"] = digest
        for field in OPTIONAL_MANIFEST_FIELDS:
            if field == "source_aliases":
                value = normalized.get(field, [])
                if not isinstance(value, list) or any(
                    not isinstance(item, str) for item in value
                ):
                    raise DerivationError(
                        "curated manifest line {} source_aliases must be "
                        "a string list".format(line_number)
                    )
                normalized[field] = [
                    _validate_http_url(item, field, line_number)
                    for item in value
                ]
                continue
            value = normalized.get(field)
            if value is not None and not isinstance(value, str):
                raise DerivationError(
                    "curated manifest line {} {} must be null or a string".format(
                        line_number,
                        field,
                    )
                )
            if field in {"source_url", "download_url"} and value is not None:
                value = _validate_http_url(value, field, line_number)
            normalized[field] = value
        records[relative_path] = normalized
        total_bytes += size
    if not records:
        raise DerivationError("curated manifest is empty")
    return records, sha256_path(path), total_bytes


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")


def _filter_jsonl(
    source_path: Path,
    destination_path: Path,
    id_field: str,
    selected_document_ids: Set[str],
    source_metadata_by_document_id: Optional[
        Mapping[str, Mapping[str, Any]]
    ] = None,
) -> Tuple[int, int]:
    count = 0
    table_count = 0

    def selected() -> Iterable[Dict[str, Any]]:
        nonlocal count, table_count
        for record in _iter_jsonl(source_path, source_path.name):
            document_id = record.get(id_field)
            if document_id not in selected_document_ids:
                continue
            if source_metadata_by_document_id is not None:
                source_metadata = source_metadata_by_document_id.get(
                    str(document_id)
                )
                if source_metadata is None:
                    raise DerivationError(
                        "{} references selected document without manifest "
                        "metadata: {}".format(source_path.name, document_id)
                    )
                existing_metadata = record.get("metadata")
                if not isinstance(existing_metadata, dict):
                    raise DerivationError(
                        "{} selected record has invalid metadata: {}".format(
                            source_path.name,
                            document_id,
                        )
                    )
                record = dict(record)
                merged_metadata = dict(existing_metadata)
                merged_metadata.update(source_metadata)
                record["metadata"] = merged_metadata
            count += 1
            if source_path.name == "blocks.jsonl" and record.get("block_type") == "table":
                table_count += 1
            yield record

    _write_jsonl(destination_path, selected())
    return count, table_count


def _safe_artifact_source(root: Path, value: Any) -> Tuple[str, Path]:
    relative = _safe_relative_path(value, "raw artifact path")
    top_level = PurePosixPath(relative).parts[0]
    if top_level in RESERVED_OUTPUT_FILES:
        raise DerivationError(
            "raw artifact collides with a run data file: {}".format(relative)
        )
    path = root.joinpath(*PurePosixPath(relative).parts)
    if path.is_symlink():
        raise DerivationError("raw artifact must not be a symlink: {}".format(relative))
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DerivationError(
            "raw artifact is unavailable {}: {}: {}".format(
                relative,
                type(exc).__name__,
                exc,
            )
        )
    if not _is_relative_to(resolved, root) or not resolved.is_file():
        raise DerivationError(
            "raw artifact escapes the source run: {}".format(relative)
        )
    return relative, resolved


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
        return "hardlink"
    except OSError:
        try:
            shutil.copy2(str(source), str(destination))
            return "copy"
        except OSError as exc:
            raise DerivationError(
                "cannot materialize raw artifact {}: {}: {}".format(
                    source,
                    type(exc).__name__,
                    exc,
                )
            )


def _write_report(
    source: Path,
    destination: Path,
    document_paths: Sequence[str],
) -> None:
    rows: Dict[str, Dict[str, str]] = {}
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REPORT_FIELDS:
                raise DerivationError("source parse_report.csv has invalid columns")
            for row in reader:
                relative_path = row.get("relative_path")
                if not relative_path:
                    raise DerivationError(
                        "source parse_report.csv contains a row without relative_path"
                    )
                if relative_path in rows:
                    raise DerivationError(
                        "source parse_report.csv contains duplicate relative_path: "
                        + relative_path
                    )
                rows[relative_path] = row
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DerivationError(
            "cannot read source parse_report.csv: {}: {}".format(
                type(exc).__name__,
                exc,
            )
        )
    missing = [path for path in document_paths if path not in rows]
    if missing:
        raise DerivationError(
            "source parse_report.csv is missing selected documents: {}".format(
                ", ".join(missing[:10])
            )
        )
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for relative_path in document_paths:
            writer.writerow(rows[relative_path])


def derive_curated_run(
    source_run: Path,
    curated_manifest: Path,
    output_dir: Path,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and verify a self-contained immutable subset of a parser run."""

    requested_source = Path(source_run)
    if requested_source.is_symlink():
        raise DerivationError("source run must not be a symlink")
    source_root = requested_source.resolve()
    requested_curated_manifest = Path(curated_manifest)
    if requested_curated_manifest.is_symlink():
        raise DerivationError("curated manifest must not be a symlink")
    curated_manifest = requested_curated_manifest.resolve()
    output_dir = Path(output_dir)
    output_root = output_dir.resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise DerivationError("output directory already exists: {}".format(output_dir))
    if not source_root.is_dir():
        raise DerivationError("source run is not a directory: {}".format(source_root))
    if _is_relative_to(output_root, source_root) or _is_relative_to(
        source_root, output_root
    ):
        raise DerivationError("source and output run directories must not overlap")

    selected_manifest, manifest_digest, selected_source_bytes = (
        _read_curated_manifest(curated_manifest)
    )
    source_verification = verify_profile_run(source_root)
    if not source_verification.get("valid"):
        raise DerivationError(
            "source run failed verification: {}".format(
                "; ".join(source_verification.get("errors") or [])
            )
        )
    source_summary = _read_json_object(
        source_root / "parse_summary.json",
        "source parse_summary.json",
    )
    source_manifest = _read_json_object(
        source_root / "run_manifest.json",
        "source run_manifest.json",
    )
    profile = source_manifest.get("profile")
    if not isinstance(profile, str) or not profile:
        raise DerivationError("source run manifest has no valid profile")
    source_run_id = source_manifest.get("run_id")
    if not isinstance(source_run_id, str) or not source_run_id:
        raise DerivationError("source run manifest has no valid run_id")
    derived_run_id = run_id or "{}-curated-{}".format(
        source_run_id,
        manifest_digest[:12],
    )
    if not isinstance(derived_run_id, str) or not derived_run_id.strip():
        raise DerivationError("run_id must be a non-empty string")
    derived_run_id = derived_run_id.strip()

    selected_documents: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()
    found_paths: Set[str] = set()
    raw_artifact_values: Set[str] = set()
    source_metadata_by_document_id: Dict[str, Dict[str, Any]] = {}
    status_counts: Counter = Counter()
    for document in _iter_jsonl(
        source_root / "documents.jsonl",
        "source documents.jsonl",
    ):
        relative_path = document.get("relative_path")
        if relative_path not in selected_manifest:
            continue
        if relative_path in found_paths:
            raise DerivationError(
                "source run contains duplicate selected relative_path: {}".format(
                    relative_path
                )
            )
        expected = selected_manifest[str(relative_path)]
        document_sha = str(document.get("source_sha256") or "").lower()
        if document_sha != expected["sha256"]:
            raise DerivationError(
                "selected source sha256 mismatch for {}: manifest={} run={}".format(
                    relative_path,
                    expected["sha256"],
                    document_sha or "<missing>",
                )
            )
        document_size = document.get("size_bytes")
        if document_size != expected["size_bytes"]:
            raise DerivationError(
                "selected source size mismatch for {}: manifest={} run={}".format(
                    relative_path,
                    expected["size_bytes"],
                    document_size,
                )
            )
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise DerivationError(
                "selected source document has no document_id: {}".format(relative_path)
            )
        if document_id in selected_ids:
            raise DerivationError(
                "source run contains duplicate selected document_id: {}".format(
                    document_id
                )
            )
        source_metadata = {
            field: (
                list(expected[field])
                if field == "source_aliases"
                else expected[field]
            )
            for field in OPTIONAL_MANIFEST_FIELDS
        }
        document = dict(document)
        document.update(source_metadata)
        raw_values = document.get("raw_artifacts") or []
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, list):
            raise DerivationError(
                "selected source document has invalid raw_artifacts: {}".format(
                    relative_path
                )
            )
        for value in raw_values:
            raw_artifact_values.add(
                _safe_relative_path(value, "raw artifact path")
            )
        status = document.get("status")
        if not isinstance(status, str) or not status:
            raise DerivationError(
                "selected source document has invalid status: {}".format(relative_path)
            )
        status_counts[status] += 1
        selected_documents.append(document)
        selected_ids.add(document_id)
        source_metadata_by_document_id[document_id] = source_metadata
        found_paths.add(str(relative_path))

    missing_paths = sorted(set(selected_manifest) - found_paths)
    if missing_paths:
        raise DerivationError(
            "source run is missing curated documents: {}".format(
                ", ".join(missing_paths[:20])
            )
        )
    if not selected_documents:
        raise DerivationError("no source documents matched the curated manifest")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".{}.".format(output_root.name),
            dir=str(output_root.parent),
        )
    )
    published = False
    try:
        _write_jsonl(temporary / "documents.jsonl", selected_documents)
        block_count, table_count = _filter_jsonl(
            source_root / "blocks.jsonl",
            temporary / "blocks.jsonl",
            "document_id",
            selected_ids,
        )
        chunk_count, _ = _filter_jsonl(
            source_root / "chunks.jsonl",
            temporary / "chunks.jsonl",
            "doc_id",
            selected_ids,
            source_metadata_by_document_id,
        )
        _filter_jsonl(
            source_root / "attempts.jsonl",
            temporary / "attempts.jsonl",
            "document_id",
            selected_ids,
        )
        document_paths = [
            str(document["relative_path"]) for document in selected_documents
        ]
        _write_report(
            source_root / "parse_report.csv",
            temporary / "parse_report.csv",
            document_paths,
        )

        source_raw_hashes = source_manifest.get("raw_artifacts")
        if not isinstance(source_raw_hashes, dict):
            raise DerivationError(
                "source run manifest raw_artifacts must be an object"
            )
        derived_raw_hashes: Dict[str, str] = {}
        raw_materialization_counts: Counter = Counter()
        for value in sorted(raw_artifact_values):
            relative, source_path = _safe_artifact_source(source_root, value)
            destination = temporary.joinpath(*PurePosixPath(relative).parts)
            method = _link_or_copy(source_path, destination)
            raw_materialization_counts[method] += 1
            actual_digest = sha256_path(destination)
            if actual_digest != source_raw_hashes.get(relative):
                raise DerivationError(
                    "raw artifact checksum changed during materialization: {}".format(
                        relative
                    )
                )
            derived_raw_hashes[relative] = actual_digest

        created_at = _utc_now()
        counts = Counter(status_counts)
        counts["blocks"] = block_count
        counts["tables"] = table_count
        counts["chunks"] = chunk_count
        summary: Dict[str, Any] = {
            "schema_version": source_summary.get("schema_version", 1),
            "run_id": derived_run_id,
            "profile": profile,
            "input": source_summary.get(
                "input",
                source_manifest.get("input_root", ""),
            ),
            "output": str(output_root),
            "file_count": len(selected_documents),
            "counts": dict(sorted(counts.items())),
            "chunk_chars": source_summary.get("chunk_chars"),
            "chunk_overlap": source_summary.get("chunk_overlap"),
            "created_at": created_at,
        }
        (temporary / "parse_summary.json").write_text(
            _canonical_json(summary) + "\n",
            encoding="utf-8",
        )

        file_hashes = {
            name: sha256_path(temporary / name)
            for name in DATA_FILES
        }
        derived_manifest: Dict[str, Any] = {
            "schema_version": source_manifest.get("schema_version", 1),
            "block_schema_version": source_manifest.get(
                "block_schema_version",
                1,
            ),
            "run_id": derived_run_id,
            "profile": profile,
            "created_at": created_at,
            "input_root": source_manifest.get(
                "input_root",
                source_summary.get("input", ""),
            ),
            "output_dir": str(output_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "config": dict(source_manifest.get("config") or {}),
            "runtime": dict(source_manifest.get("runtime") or {}),
            "source_manifest_sha256": manifest_digest,
            "selection_counts": {
                "manifest_entries": len(selected_manifest),
                "selected_files": len(selected_documents),
            },
            "files": file_hashes,
            "raw_artifacts": derived_raw_hashes,
            "derived_from": {
                "source_run": str(source_root),
                "source_run_id": source_run_id,
                "source_run_manifest_sha256": sha256_path(
                    source_root / "run_manifest.json"
                ),
                "curated_manifest": str(curated_manifest),
                "curated_manifest_sha256": manifest_digest,
                "selected_source_bytes": selected_source_bytes,
                "raw_artifact_materialization": dict(
                    sorted(raw_materialization_counts.items())
                ),
            },
        }
        (temporary / "run_manifest.json").write_text(
            _canonical_json(derived_manifest) + "\n",
            encoding="utf-8",
        )

        temporary_verification = verify_profile_run(temporary)
        if not temporary_verification.get("valid"):
            raise DerivationError(
                "derived run failed verification before publication: {}".format(
                    "; ".join(temporary_verification.get("errors") or [])
                )
            )
        if output_root.exists() or output_root.is_symlink():
            raise DerivationError(
                "output directory appeared during publication: {}".format(
                    output_root
                )
            )
        os.rename(str(temporary), str(output_root))
        published = True
        final_verification = verify_profile_run(output_root)
        if not final_verification.get("valid"):
            raise DerivationError(
                "published derived run failed verification: {}".format(
                    "; ".join(final_verification.get("errors") or [])
                )
            )
        return {
            "output": str(output_root),
            "summary": summary,
            "manifest": derived_manifest,
            "verification": final_verification,
        }
    except BaseException:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        if published and output_root.exists():
            shutil.rmtree(str(output_root))
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--curated-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = derive_curated_run(
            source_run=args.source_run,
            curated_manifest=args.curated_manifest,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except (DerivationError, ValueError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    sys.stdout.write(
        _canonical_json(
            {
                "output": result["output"],
                "summary": result["summary"],
                "verification": result["verification"],
            }
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
