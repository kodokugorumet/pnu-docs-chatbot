"""Deterministic run artifacts and validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from scripts.document_parsing.core import (
    ATTEMPT_STATUSES,
    Block,
    blocks_to_legacy_chunks,
    canonical_json,
    iter_jsonl,
)
from scripts.document_parsing.pipeline import PipelineConfig, PipelineOutcome


DATA_FILES = (
    "documents.jsonl",
    "blocks.jsonl",
    "chunks.jsonl",
    "attempts.jsonl",
    "parse_report.csv",
    "parse_summary.json",
)
JSONL_FILES = (
    "documents.jsonl",
    "blocks.jsonl",
    "chunks.jsonl",
    "attempts.jsonl",
)
REPORT_FIELDS = (
    "status",
    "reason",
    "institution",
    "relative_path",
    "extension",
    "detected_format",
    "selected_parsers",
    "char_count",
    "block_count",
    "table_count",
    "chunk_count",
    "attempt_count",
    "size_bytes",
)
PARSER_SENTINEL = re.compile(
    "\ue000(?:TABLE_START|TABLE_END|ROW|CELL|PARA)\ue001"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_profile_run(
    output_dir: Path,
    outcomes: Iterable[PipelineOutcome],
    config: PipelineConfig,
    input_root: Path,
    run_id: str,
    runtime_report: Optional[Mapping[str, Any]] = None,
    chunk_chars: int = 1800,
    chunk_overlap: int = 250,
    published_output_dir: Optional[Path] = None,
    source_manifest_sha256: Optional[str] = None,
    selection_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ordered: Iterable[PipelineOutcome]
    if isinstance(outcomes, Sequence):
        ordered = sorted(outcomes, key=lambda item: item.source.relative_path)
    else:
        ordered = outcomes

    counts: Counter = Counter()
    raw_artifact_paths = set()
    file_count = 0
    previous_relative_path: Optional[str] = None
    temporary_paths: Dict[str, str] = {}
    jsonl_handles: Dict[str, Any] = {}
    report_handle: Any = None
    try:
        for name in JSONL_FILES:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".{}.".format(name),
                suffix=".tmp",
                dir=str(root),
            )
            temporary_paths[name] = temporary_name
            jsonl_handles[name] = os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".parse_report.csv.",
            suffix=".tmp",
            dir=str(root),
        )
        temporary_paths["parse_report.csv"] = temporary_name
        report_handle = os.fdopen(
            descriptor, "w", encoding="utf-8-sig", newline=""
        )
        report_writer = csv.DictWriter(
            report_handle, fieldnames=REPORT_FIELDS
        )
        report_writer.writeheader()

        for outcome in ordered:
            source = outcome.source
            if (
                previous_relative_path is not None
                and source.relative_path < previous_relative_path
            ):
                raise ValueError(
                    "streamed outcomes must be ordered by relative_path"
                )
            previous_relative_path = source.relative_path
            file_count += 1
            result = outcome.result
            selected_parsers = list(outcome.selected_parsers)
            parser_value = "+".join(selected_parsers)
            document_blocks = sorted(
                result.blocks,
                key=lambda block: (block.reading_order, block.block_id),
            )
            document_text = "\n\n".join(
                block.text
                for block in document_blocks
                if block.block_type != "table_cell" and block.text
            )
            raw_artifacts = [
                _relative_artifact(item, root)
                for item in result.raw_artifacts
            ]
            raw_artifacts = [
                item for item in raw_artifacts if item is not None
            ]
            raw_artifact_paths.update(raw_artifacts)

            document = source.to_dict()
            document.update(
                {
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "detected_format": outcome.sniffed_format,
                    "mime_type": outcome.mime_type,
                    "selected_parser": (
                        selected_parsers[0]
                        if len(selected_parsers) == 1
                        else None
                    ),
                    "selected_parsers": selected_parsers,
                    "parser": parser_value,
                    "quality": outcome.quality.to_dict(),
                    "block_count": len(document_blocks),
                    "table_count": sum(
                        block.block_type == "table"
                        for block in document_blocks
                    ),
                    "char_count": len(document_text),
                    "text": document_text,
                    "raw_artifacts": raw_artifacts,
                }
            )
            _write_jsonl_record(
                jsonl_handles["documents.jsonl"], document
            )
            for block in document_blocks:
                _write_jsonl_record(
                    jsonl_handles["blocks.jsonl"], block
                )

            for attempt_index, attempt in enumerate(result.attempts):
                attempt_record = {
                    "document_id": source.document_id,
                    "relative_path": source.relative_path,
                    "attempt_index": attempt_index,
                }
                attempt_record.update(attempt.to_dict())
                _write_jsonl_record(
                    jsonl_handles["attempts.jsonl"], attempt_record
                )

            document_chunks: List[Dict[str, Any]] = []
            if outcome.status == "parsed" and document_blocks:
                metadata = {
                    "institution": source.institution,
                    "source_path": source.source_path,
                    "relative_path": source.relative_path,
                    "file_name": source.file_name,
                    "extension": source.extension,
                    "parser": parser_value,
                    "source_title": source.source_title,
                    "source_url": source.source_url,
                    "download_url": source.download_url,
                    "source_host": source.source_host,
                    "fetched_at": source.fetched_at,
                    "published_at": source.published_at,
                    "category": source.category,
                    "include_reason": source.include_reason,
                    "source_aliases": list(source.source_aliases),
                    "crawl_storage_path": source.crawl_storage_path,
                }
                document_chunks = blocks_to_legacy_chunks(
                    document_blocks,
                    metadata=metadata,
                    doc_id=source.document_id,
                    max_chars=chunk_chars,
                    overlap=chunk_overlap,
                )
                for chunk in document_chunks:
                    chunk["chunk_id"] = "{}:{}#{:04d}".format(
                        source.document_id,
                        config.profile,
                        int(chunk["chunk_index"]),
                    )
                    _write_jsonl_record(
                        jsonl_handles["chunks.jsonl"], chunk
                    )

            table_count = sum(
                block.block_type == "table" for block in document_blocks
            )
            counts[outcome.status] += 1
            counts["blocks"] += len(document_blocks)
            counts["tables"] += table_count
            counts["chunks"] += len(document_chunks)
            report_writer.writerow(
                _csv_safe_row(
                    {
                    "status": outcome.status,
                    "reason": outcome.reason,
                    "institution": source.institution,
                    "relative_path": source.relative_path,
                    "extension": source.extension,
                    "detected_format": outcome.sniffed_format,
                    "selected_parsers": parser_value,
                    "char_count": len(document_text),
                    "block_count": len(document_blocks),
                    "table_count": table_count,
                    "chunk_count": len(document_chunks),
                    "attempt_count": len(result.attempts),
                    "size_bytes": source.size_bytes,
                    }
                )
            )

        for handle in list(jsonl_handles.values()) + [report_handle]:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for name in JSONL_FILES + ("parse_report.csv",):
            os.replace(temporary_paths[name], str(root / name))
        temporary_paths.clear()
    except BaseException:
        for handle in list(jsonl_handles.values()) + [report_handle]:
            if handle is not None and not handle.closed:
                handle.close()
        for temporary_name in temporary_paths.values():
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise

    created_at = datetime.now(timezone.utc).isoformat()
    published_root = Path(published_output_dir or root).resolve()
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "profile": config.profile,
        "input": str(Path(input_root).resolve()),
        "output": str(published_root),
        "file_count": file_count,
        "counts": dict(sorted(counts.items())),
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "created_at": created_at,
    }
    _atomic_write_json(root / "parse_summary.json", summary)

    hashes = {
        name: sha256_path(root / name)
        for name in DATA_FILES
        if (root / name).is_file()
    }
    raw_hashes = {
        relative: sha256_path(root / relative)
        for relative in sorted(raw_artifact_paths)
    }
    if source_manifest_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", source_manifest_sha256
    ):
        raise ValueError(
            "source_manifest_sha256 must be a lowercase SHA-256 digest"
        )
    recorded_selection_counts = dict(selection_counts or {})
    if not recorded_selection_counts:
        recorded_selection_counts["selected_files"] = file_count
    for name, value in recorded_selection_counts.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                "selection_counts must map non-empty strings to "
                "non-negative integers"
            )
    if (
        "selected_files" in recorded_selection_counts
        and recorded_selection_counts["selected_files"] != file_count
    ):
        raise ValueError(
            "selection_counts selected_files does not match parsed file_count"
        )
    manifest = {
        "schema_version": 1,
        "block_schema_version": 1,
        "run_id": run_id,
        "profile": config.profile,
        "created_at": created_at,
        "input_root": str(Path(input_root).resolve()),
        "output_dir": str(published_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": {
            "expect_korean": config.expect_korean,
            "min_chars": config.min_chars,
            "enable_vl_review": config.enable_vl_review,
            "fast_timeout_seconds": config.fast_timeout_seconds,
            "tesseract_timeout_seconds": config.tesseract_timeout_seconds,
            "heavy_timeout_seconds": config.heavy_timeout_seconds,
            "max_blocks_per_document": config.max_blocks_per_document,
            "chunk_chars": chunk_chars,
            "chunk_overlap": chunk_overlap,
        },
        "runtime": dict(runtime_report or {}),
        "source_manifest_sha256": source_manifest_sha256,
        "selection_counts": recorded_selection_counts,
        "files": hashes,
        "raw_artifacts": raw_hashes,
    }
    _atomic_write_json(root / "run_manifest.json", manifest)
    return {
        "output": str(root),
        "summary": summary,
        "manifest": manifest,
    }


def verify_profile_run(output_dir: Path) -> Dict[str, Any]:
    """Stream-verify an immutable profile run without loading the corpus."""

    requested_root = Path(output_dir)
    if requested_root.is_symlink():
        raise ValueError("verification target must not be a symlink")
    root = requested_root.resolve()
    if root.name == "current" and root.parent.name == "processed":
        raise ValueError(
            "verification target must be an immutable run, not processed/current"
        )
    required = set(DATA_FILES) | {"run_manifest.json"}
    missing = sorted(
        name for name in required if not (root / name).is_file()
    )
    if missing:
        return {
            "valid": False,
            "output": str(root),
            "errors": [
                "missing files: {}".format(", ".join(missing))
            ],
        }

    errors: List[str] = []
    for name in required:
        if (root / name).is_symlink():
            errors.append("{} must not be a symlink".format(name))

    manifest = _read_json_object_for_verification(
        root / "run_manifest.json",
        "run_manifest.json",
        errors,
    )
    summary = _read_json_object_for_verification(
        root / "parse_summary.json",
        "parse_summary.json",
        errors,
    )
    profile = manifest.get("profile")
    if not isinstance(profile, str) or not profile:
        errors.append("manifest profile must be a non-empty string")
        profile = ""
    if summary.get("profile") != profile:
        errors.append("summary and manifest profile do not match")
    if summary.get("run_id") != manifest.get("run_id"):
        errors.append("summary and manifest run_id do not match")

    documents: Dict[str, Dict[str, Any]] = {}
    document_order: List[str] = []
    statuses: Counter = Counter()
    referenced_raw_artifacts = set()
    document_count = 0
    previous_relative_path: Optional[str] = None
    for index, item in enumerate(
        _iter_jsonl_for_verification(
            root / "documents.jsonl",
            "documents.jsonl",
            errors,
        ),
        start=1,
    ):
        document_count += 1
        if not isinstance(item, Mapping):
            errors.append(
                "documents.jsonl:{} must be an object".format(index)
            )
            continue
        document_id = item.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            errors.append(
                "documents.jsonl:{} has invalid document_id".format(
                    index
                )
            )
            continue
        if document_id in documents:
            errors.append(
                "documents.jsonl contains duplicate document_id values"
            )
            continue
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(
                "documents.jsonl:{} has invalid relative_path".format(
                    index
                )
            )
            relative_path = ""
        if (
            previous_relative_path is not None
            and relative_path < previous_relative_path
        ):
            errors.append(
                "documents.jsonl is not ordered by relative_path"
            )
        previous_relative_path = relative_path
        raw_values = item.get("raw_artifacts", [])
        if not isinstance(raw_values, list) or any(
            not isinstance(value, str) or not value
            for value in raw_values
        ):
            errors.append(
                "documents.jsonl:{} has invalid raw_artifacts".format(
                    index
                )
            )
            raw_values = []
        referenced_raw_artifacts.update(raw_values)
        text = item.get("text")
        if not isinstance(text, str):
            errors.append(
                "documents.jsonl:{} text must be a string".format(index)
            )
            text = ""
        status = item.get("status")
        if not isinstance(status, str) or not status:
            errors.append(
                "documents.jsonl:{} has invalid status".format(index)
            )
            status = ""
        else:
            statuses[status] += 1
        documents[document_id] = {
            "relative_path": relative_path,
            "institution": item.get("institution", ""),
            "status": status,
            "block_count": item.get("block_count"),
            "table_count": item.get("table_count"),
            "char_count": item.get("char_count"),
            "text_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
        }
        document_order.append(document_id)

    known_documents = set(documents)
    block_count = 0
    block_counts: Counter = Counter()
    table_counts: Counter = Counter()
    text_lengths: Counter = Counter()
    text_digests: Dict[str, Any] = {}
    text_started = set()
    current_document: Optional[str] = None
    seen_block_documents = set()
    current_tables: Dict[str, Dict[str, Any]] = {}

    def finish_tables() -> None:
        for table_id, state in current_tables.items():
            if state["parents"] != 1:
                errors.append(
                    "{} must have exactly one table parent".format(
                        table_id
                    )
                )
            elif state["bad_cell_order"]:
                errors.append(
                    "{} cells must follow the table parent".format(
                        table_id
                    )
                )

    for index, value in enumerate(
        _iter_jsonl_for_verification(
            root / "blocks.jsonl",
            "blocks.jsonl",
            errors,
        ),
        start=1,
    ):
        block_count += 1
        try:
            if not isinstance(value, Mapping):
                raise TypeError("canonical block must be an object")
            block = Block.from_dict(value)
        except (TypeError, ValueError) as exc:
            errors.append("blocks.jsonl:{}: {}".format(index, exc))
            continue
        if block.document_id not in known_documents:
            errors.append(
                "block {} references unknown document {}".format(
                    block.block_id,
                    block.document_id,
                )
            )
            continue
        if block.document_id != current_document:
            finish_tables()
            current_tables = {}
            if block.document_id in seen_block_documents:
                errors.append(
                    "blocks.jsonl documents must be contiguous"
                )
            seen_block_documents.add(block.document_id)
            current_document = block.document_id
        expected_order = block_counts[block.document_id]
        if block.reading_order != expected_order:
            errors.append(
                "{} has non-contiguous reading_order".format(
                    block.document_id
                )
            )
        if profile:
            expected_id = "{}:{}:b{:06d}".format(
                block.document_id,
                profile,
                expected_order,
            )
            if block.block_id != expected_id:
                if block.reading_order < expected_order:
                    errors.append(
                        "duplicate block_id: {}".format(block.block_id)
                    )
                else:
                    errors.append(
                        "unexpected block_id: {}".format(block.block_id)
                    )
        block_counts[block.document_id] += 1
        if block.block_type == "table":
            table_counts[block.document_id] += 1
        if block.block_type != "table_cell" and block.text:
            digest = text_digests.setdefault(
                block.document_id,
                hashlib.sha256(),
            )
            if block.document_id in text_started:
                digest.update(b"\n\n")
                text_lengths[block.document_id] += 2
            digest.update(block.text.encode("utf-8"))
            text_lengths[block.document_id] += len(block.text)
            text_started.add(block.document_id)
        if block.table_id:
            state = current_tables.setdefault(
                block.table_id,
                {
                    "parents": 0,
                    "parent_order": None,
                    "bad_cell_order": False,
                },
            )
            if block.block_type == "table":
                state["parents"] += 1
                state["parent_order"] = block.reading_order
            elif block.block_type == "table_cell":
                parent_order = state["parent_order"]
                if (
                    parent_order is None
                    or block.reading_order <= parent_order
                ):
                    state["bad_cell_order"] = True
        if PARSER_SENTINEL.search(block.text):
            errors.append(
                "block {} contains an internal parser sentinel".format(
                    block.block_id
                )
            )
    finish_tables()

    chunk_count = 0
    chunk_counts: Counter = Counter()
    current_chunk_document: Optional[str] = None
    seen_chunk_documents = set()
    for index, chunk in enumerate(
        _iter_jsonl_for_verification(
            root / "chunks.jsonl",
            "chunks.jsonl",
            errors,
        ),
        start=1,
    ):
        chunk_count += 1
        required_chunk_fields = {
            "chunk_id",
            "doc_id",
            "chunk_index",
            "text",
            "char_count",
            "metadata",
        }
        if not isinstance(chunk, Mapping):
            errors.append(
                "chunks.jsonl:{} must be an object".format(index)
            )
            continue
        missing_fields = required_chunk_fields - set(chunk)
        extra_fields = set(chunk) - required_chunk_fields
        if missing_fields or extra_fields:
            details = []
            if missing_fields:
                details.append(
                    "missing {}".format(
                        ",".join(sorted(missing_fields))
                    )
                )
            if extra_fields:
                details.append(
                    "extra {}".format(
                        ",".join(sorted(extra_fields))
                    )
                )
            errors.append(
                "chunks.jsonl:{} {}".format(
                    index,
                    "; ".join(details),
                )
            )
            continue
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        chunk_index = chunk.get("chunk_index")
        text = chunk.get("text")
        char_count = chunk.get("char_count")
        metadata = chunk.get("metadata")
        if not isinstance(chunk_id, str) or not chunk_id:
            errors.append(
                "chunks.jsonl:{} has invalid chunk_id".format(index)
            )
            continue
        if not isinstance(doc_id, str) or not doc_id:
            errors.append(
                "chunk {} has invalid doc_id".format(chunk_id)
            )
            doc_id = ""
        elif doc_id not in known_documents:
            errors.append(
                "chunk {} references unknown document".format(chunk_id)
            )
        if doc_id and doc_id != current_chunk_document:
            if doc_id in seen_chunk_documents:
                errors.append(
                    "chunks.jsonl documents must be contiguous"
                )
            seen_chunk_documents.add(doc_id)
            current_chunk_document = doc_id
        expected_index = chunk_counts[doc_id]
        valid_index = (
            not isinstance(chunk_index, bool)
            and isinstance(chunk_index, int)
            and chunk_index >= 0
        )
        if not valid_index:
            errors.append(
                "chunk {} has invalid chunk_index".format(chunk_id)
            )
        elif chunk_index != expected_index:
            errors.append(
                "{} has non-contiguous or duplicate chunk_index".format(
                    doc_id
                )
            )
        if doc_id and profile:
            expected_id = "{}:{}#{:04d}".format(
                doc_id,
                profile,
                expected_index,
            )
            if chunk_id != expected_id:
                if (
                    valid_index
                    and chunk_index < expected_index
                    and chunk_id
                    == "{}:{}#{:04d}".format(
                        doc_id,
                        profile,
                        chunk_index,
                    )
                ):
                    errors.append(
                        "duplicate chunk_id: {}".format(chunk_id)
                    )
                else:
                    errors.append(
                        "unexpected chunk_id: {}".format(chunk_id)
                    )
        if doc_id:
            chunk_counts[doc_id] += 1
        if not isinstance(text, str):
            errors.append(
                "chunk {} text must be a string".format(chunk_id)
            )
        elif PARSER_SENTINEL.search(text):
            errors.append(
                "chunk {} contains an internal parser sentinel".format(
                    chunk_id
                )
            )
        if (
            isinstance(char_count, bool)
            or not isinstance(char_count, int)
            or not isinstance(text, str)
            or char_count != len(text)
        ):
            errors.append(
                "chunk {} char_count mismatch".format(chunk_id)
            )
        if not isinstance(metadata, Mapping):
            errors.append(
                "chunk {} metadata must be an object".format(chunk_id)
            )
            continue
        metadata_block_ids = metadata.get("block_ids") or []
        if isinstance(
            metadata_block_ids,
            (str, bytes),
        ) or not isinstance(metadata_block_ids, Sequence):
            errors.append(
                "chunk {} has invalid block_ids".format(chunk_id)
            )
            continue
        if len(set(metadata_block_ids)) != len(metadata_block_ids):
            errors.append(
                "chunk {} has duplicate block_ids".format(chunk_id)
            )
        if any(
            not _valid_block_reference(
                block_id,
                doc_id,
                profile,
                block_counts.get(doc_id, 0),
            )
            for block_id in metadata_block_ids
        ):
            errors.append(
                "chunk {} references unknown blocks".format(chunk_id)
            )

    attempt_count = 0
    attempt_counts: Counter = Counter()
    for index, attempt in enumerate(
        _iter_jsonl_for_verification(
            root / "attempts.jsonl",
            "attempts.jsonl",
            errors,
        ),
        start=1,
    ):
        attempt_count += 1
        if not isinstance(attempt, Mapping):
            errors.append(
                "attempts.jsonl:{} must be an object".format(index)
            )
            continue
        document_id = attempt.get("document_id")
        if document_id not in known_documents:
            errors.append(
                "attempts.jsonl:{} references unknown document".format(
                    index
                )
            )
        else:
            expected_index = attempt_counts[document_id]
            if attempt.get("attempt_index") != expected_index:
                errors.append(
                    "{} has non-contiguous attempt_index".format(
                        document_id
                    )
                )
            attempt_counts[document_id] += 1
        if attempt.get("status") not in ATTEMPT_STATUSES:
            errors.append(
                "attempts.jsonl:{} has invalid status".format(index)
            )

    for document_id, expected in documents.items():
        digest = text_digests.get(document_id)
        text_sha256 = (
            digest.hexdigest()
            if digest is not None
            else hashlib.sha256(b"").hexdigest()
        )
        comparisons = {
            "block_count": block_counts[document_id],
            "table_count": table_counts[document_id],
            "char_count": text_lengths[document_id],
        }
        for field, actual in comparisons.items():
            if expected.get(field) != actual:
                errors.append(
                    "{} {} mismatch".format(document_id, field)
                )
        if expected["text_sha256"] != text_sha256:
            errors.append("{} text mismatch".format(document_id))

    actual_counts = Counter(statuses)
    actual_counts["blocks"] = block_count
    actual_counts["tables"] = sum(table_counts.values())
    actual_counts["chunks"] = chunk_count
    if summary.get("file_count") != document_count:
        errors.append("parse_summary file_count mismatch")
    if summary.get("counts") != dict(sorted(actual_counts.items())):
        errors.append("parse_summary counts mismatch")

    _verify_report_for_run(
        root / "parse_report.csv",
        document_order,
        documents,
        block_counts,
        table_counts,
        chunk_counts,
        attempt_counts,
        errors,
    )

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        errors.append("manifest files must be an object")
        manifest_files = {}
    missing_hashes = sorted(set(DATA_FILES) - set(manifest_files))
    extra_hashes = sorted(set(manifest_files) - set(DATA_FILES))
    if missing_hashes:
        errors.append(
            "manifest missing checksums: {}".format(
                ", ".join(missing_hashes)
            )
        )
    if extra_hashes:
        errors.append(
            "manifest has unexpected checksums: {}".format(
                ", ".join(extra_hashes)
            )
        )
    hash_mismatches = []
    for name in DATA_FILES:
        expected = manifest_files.get(name)
        path = root / name
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or path.is_symlink()
            or sha256_path(path) != expected
        ):
            hash_mismatches.append(name)
    if hash_mismatches:
        errors.append(
            "manifest checksum mismatch: {}".format(
                ", ".join(sorted(hash_mismatches))
            )
        )

    raw_hashes = manifest.get("raw_artifacts")
    if not isinstance(raw_hashes, dict):
        errors.append("manifest raw_artifacts must be an object")
        raw_hashes = {}
    declared_raw_artifacts = set(raw_hashes)
    missing_raw_hashes = sorted(
        referenced_raw_artifacts - declared_raw_artifacts
    )
    extra_raw_hashes = sorted(
        declared_raw_artifacts - referenced_raw_artifacts
    )
    if missing_raw_hashes:
        errors.append(
            "manifest missing raw artifact checksums: {}".format(
                ", ".join(missing_raw_hashes)
            )
        )
    if extra_raw_hashes:
        errors.append(
            "manifest has unreferenced raw artifact checksums: {}".format(
                ", ".join(extra_raw_hashes)
            )
        )
    invalid_raw = []
    for relative in sorted(referenced_raw_artifacts):
        path = _safe_run_relative(root, relative)
        expected = raw_hashes.get(relative)
        if (
            path is None
            or not path.is_file()
            or path.is_symlink()
            or not isinstance(expected, str)
            or len(expected) != 64
            or sha256_path(path) != expected
        ):
            invalid_raw.append(relative)
    if invalid_raw:
        errors.append(
            "raw artifact checksum mismatch: {}".format(
                ", ".join(invalid_raw)
            )
        )

    result_hashes: Dict[str, str] = {}
    for name in sorted(required):
        path = root / name
        try:
            if path.is_file() and not path.is_symlink():
                result_hashes[name] = sha256_path(path)
        except OSError as exc:
            errors.append(
                "{} cannot be hashed: {}: {}".format(
                    name,
                    type(exc).__name__,
                    exc,
                )
            )
    return {
        "valid": not errors,
        "output": str(root),
        "documents": document_count,
        "blocks": block_count,
        "chunks": chunk_count,
        "attempts": attempt_count,
        "errors": errors,
        "hashes": result_hashes,
    }


def _read_json_object_for_verification(
    path: Path,
    label: str,
    errors: List[str],
) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(
            "{} is invalid: {}: {}".format(
                label,
                type(exc).__name__,
                exc,
            )
        )
        return {}
    if not isinstance(value, dict):
        errors.append("{} must contain an object".format(label))
        return {}
    return value


def _iter_jsonl_for_verification(
    path: Path,
    label: str,
    errors: List[str],
) -> Iterable[Any]:
    try:
        yield from iter_jsonl(path)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        errors.append(
            "{} is invalid: {}: {}".format(
                label,
                type(exc).__name__,
                exc,
            )
        )


def _valid_block_reference(
    block_id: Any,
    document_id: str,
    profile: str,
    block_count: int,
) -> bool:
    if not isinstance(block_id, str) or not document_id or not profile:
        return False
    prefix = "{}:{}:b".format(document_id, profile)
    if not block_id.startswith(prefix):
        return False
    suffix = block_id[len(prefix):]
    return (
        len(suffix) == 6
        and suffix.isdigit()
        and int(suffix) < block_count
    )


def _verify_report_for_run(
    path: Path,
    document_order: Sequence[str],
    documents: Mapping[str, Mapping[str, Any]],
    block_counts: Mapping[str, int],
    table_counts: Mapping[str, int],
    chunk_counts: Mapping[str, int],
    attempt_counts: Mapping[str, int],
    errors: List[str],
) -> None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REPORT_FIELDS:
                errors.append("parse_report.csv has invalid columns")
                return
            row_count = 0
            for row_count, row in enumerate(reader, start=1):
                if row_count > len(document_order):
                    errors.append("parse_report.csv has extra rows")
                    continue
                document_id = document_order[row_count - 1]
                document = documents[document_id]
                expected = {
                    "status": _csv_safe_value(document["status"]),
                    "institution": _csv_safe_value(
                        document["institution"]
                    ),
                    "relative_path": _csv_safe_value(
                        document["relative_path"]
                    ),
                    "block_count": str(block_counts[document_id]),
                    "table_count": str(table_counts[document_id]),
                    "chunk_count": str(chunk_counts[document_id]),
                    "attempt_count": str(attempt_counts[document_id]),
                }
                for field, value in expected.items():
                    if row.get(field) != value:
                        errors.append(
                            "parse_report.csv:{} {} mismatch".format(
                                row_count,
                                field,
                            )
                        )
            if row_count != len(document_order):
                errors.append("parse_report.csv row count mismatch")
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(
            "parse_report.csv is invalid: {}: {}".format(
                type(exc).__name__,
                exc,
            )
        )


def _verify_profile_run_legacy(output_dir: Path) -> Dict[str, Any]:
    root = Path(output_dir).resolve()
    if root.name == "current" and root.parent.name == "processed":
        raise ValueError("verification target must be an immutable run, not processed/current")
    required = set(DATA_FILES) | {"run_manifest.json"}
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        return {
            "valid": False,
            "output": str(root),
            "errors": ["missing files: {}".format(", ".join(missing))],
        }

    errors: List[str] = []
    documents = _read_jsonl_for_verification(
        root / "documents.jsonl", "documents.jsonl", errors
    )
    block_values = _read_jsonl_for_verification(
        root / "blocks.jsonl", "blocks.jsonl", errors
    )
    chunks = _read_jsonl_for_verification(
        root / "chunks.jsonl", "chunks.jsonl", errors
    )
    attempts = _read_jsonl_for_verification(
        root / "attempts.jsonl", "attempts.jsonl", errors
    )
    document_ids: List[str] = []
    referenced_raw_artifacts = set()
    for index, item in enumerate(documents, start=1):
        if not isinstance(item, Mapping):
            errors.append("documents.jsonl:{} must be an object".format(index))
            continue
        document_id = item.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            errors.append(
                "documents.jsonl:{} has invalid document_id".format(index)
            )
            continue
        document_ids.append(document_id)
        raw_values = item.get("raw_artifacts", [])
        if not isinstance(raw_values, list) or any(
            not isinstance(value, str) or not value for value in raw_values
        ):
            errors.append(
                "documents.jsonl:{} has invalid raw_artifacts".format(index)
            )
            continue
        referenced_raw_artifacts.update(raw_values)
    if len(set(document_ids)) != len(document_ids):
        errors.append("documents.jsonl contains duplicate document_id values")
    known_documents = set(document_ids)

    blocks: List[Block] = []
    for index, value in enumerate(block_values, start=1):
        try:
            if not isinstance(value, Mapping):
                raise TypeError("canonical block must be an object")
            block = Block.from_dict(value)
            blocks.append(block)
        except (TypeError, ValueError) as exc:
            errors.append("blocks.jsonl:{}: {}".format(index, exc))
            continue
        if block.document_id not in known_documents:
            errors.append(
                "block {} references unknown document {}".format(
                    block.block_id, block.document_id
                )
            )

    blocks_by_document: Dict[str, List[Block]] = defaultdict(list)
    block_ids = set()
    tables: Dict[str, List[Block]] = defaultdict(list)
    for block in blocks:
        if block.block_id in block_ids:
            errors.append("duplicate block_id: {}".format(block.block_id))
        block_ids.add(block.block_id)
        blocks_by_document[block.document_id].append(block)
        if block.table_id:
            tables[block.table_id].append(block)
        if PARSER_SENTINEL.search(block.text):
            errors.append(
                "block {} contains an internal parser sentinel".format(
                    block.block_id
                )
            )
    for document_id, values in blocks_by_document.items():
        orders = sorted(block.reading_order for block in values)
        if orders != list(range(len(values))):
            errors.append(
                "{} has non-contiguous reading_order".format(document_id)
            )
    for table_id, values in tables.items():
        parents = [block for block in values if block.block_type == "table"]
        if len(parents) != 1:
            errors.append(
                "{} must have exactly one table parent".format(table_id)
            )
            continue
        parent_order = parents[0].reading_order
        if any(
            block.block_type == "table_cell"
            and block.reading_order <= parent_order
            for block in values
        ):
            errors.append("{} cells must follow the table parent".format(table_id))

    chunk_ids = set()
    chunk_indices_by_document: Dict[str, List[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks, start=1):
        required_chunk_fields = {
            "chunk_id",
            "doc_id",
            "chunk_index",
            "text",
            "char_count",
            "metadata",
        }
        if not isinstance(chunk, Mapping):
            errors.append("chunks.jsonl:{} must be an object".format(index))
            continue
        actual_chunk_fields = set(chunk)
        missing_fields = required_chunk_fields - actual_chunk_fields
        extra_fields = actual_chunk_fields - required_chunk_fields
        if missing_fields or extra_fields:
            details = []
            if missing_fields:
                details.append("missing {}".format(",".join(sorted(missing_fields))))
            if extra_fields:
                details.append("extra {}".format(",".join(sorted(extra_fields))))
            errors.append(
                "chunks.jsonl:{} {}".format(
                    index, "; ".join(details)
                )
            )
            continue
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        text = chunk.get("text")
        chunk_index = chunk.get("chunk_index")
        char_count = chunk.get("char_count")
        metadata = chunk.get("metadata")
        if not isinstance(chunk_id, str) or not chunk_id:
            errors.append("chunks.jsonl:{} has invalid chunk_id".format(index))
            continue
        if chunk_id in chunk_ids:
            errors.append("duplicate chunk_id: {}".format(chunk_id))
        chunk_ids.add(chunk_id)
        if not isinstance(doc_id, str) or not doc_id:
            errors.append("chunk {} has invalid doc_id".format(chunk_id))
        elif doc_id not in known_documents:
            errors.append(
                "chunk {} references unknown document".format(chunk_id)
            )
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            errors.append("chunk {} has invalid chunk_index".format(chunk_id))
        elif isinstance(doc_id, str):
            chunk_indices_by_document[doc_id].append(chunk_index)
        if not isinstance(text, str):
            errors.append("chunk {} text must be a string".format(chunk_id))
        elif PARSER_SENTINEL.search(text):
            errors.append(
                "chunk {} contains an internal parser sentinel".format(chunk_id)
            )
        if (
            isinstance(char_count, bool)
            or not isinstance(char_count, int)
            or not isinstance(text, str)
            or char_count != len(text)
        ):
            errors.append(
                "chunk {} char_count mismatch".format(chunk_id)
            )
        if not isinstance(metadata, Mapping):
            errors.append("chunk {} metadata must be an object".format(chunk_id))
            continue
        metadata_block_ids = metadata.get("block_ids") or []
        if isinstance(metadata_block_ids, (str, bytes)) or not isinstance(
            metadata_block_ids, Sequence
        ):
            errors.append("chunk {} has invalid block_ids".format(chunk_id))
            continue
        unknown = set(metadata_block_ids) - block_ids
        if unknown:
            errors.append(
                "chunk {} references unknown blocks".format(chunk_id)
            )
    for document_id, indices in chunk_indices_by_document.items():
        if sorted(indices) != list(range(len(indices))):
            errors.append(
                "{} has non-contiguous or duplicate chunk_index".format(
                    document_id
                )
            )

    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            errors.append("attempts.jsonl:{} must be an object".format(index))
            continue
        if attempt.get("document_id") not in known_documents:
            errors.append(
                "attempts.jsonl:{} references unknown document".format(index)
            )
        if attempt.get("status") not in ATTEMPT_STATUSES:
            errors.append(
                "attempts.jsonl:{} has invalid status".format(index)
            )

    try:
        manifest = json.loads(
            (root / "run_manifest.json").read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            errors.append("run_manifest.json must contain an object")
            manifest = {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(
            "run_manifest.json is invalid: {}: {}".format(type(exc).__name__, exc)
        )
        manifest = {}
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        errors.append("manifest files must be an object")
        manifest_files = {}
    missing_hashes = sorted(set(DATA_FILES) - set(manifest_files))
    extra_hashes = sorted(set(manifest_files) - set(DATA_FILES))
    if missing_hashes:
        errors.append(
            "manifest missing checksums: {}".format(", ".join(missing_hashes))
        )
    if extra_hashes:
        errors.append(
            "manifest has unexpected checksums: {}".format(
                ", ".join(extra_hashes)
            )
        )
    hash_mismatches = []
    for name in DATA_FILES:
        expected = manifest_files.get(name)
        path = root / name
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or sha256_path(path) != expected
        ):
            hash_mismatches.append(name)
    if hash_mismatches:
        errors.append(
            "manifest checksum mismatch: {}".format(
                ", ".join(sorted(hash_mismatches))
            )
        )

    raw_hashes = manifest.get("raw_artifacts")
    if not isinstance(raw_hashes, dict):
        errors.append("manifest raw_artifacts must be an object")
        raw_hashes = {}
    declared_raw_artifacts = set(raw_hashes)
    missing_raw_hashes = sorted(
        referenced_raw_artifacts - declared_raw_artifacts
    )
    extra_raw_hashes = sorted(
        declared_raw_artifacts - referenced_raw_artifacts
    )
    if missing_raw_hashes:
        errors.append(
            "manifest missing raw artifact checksums: {}".format(
                ", ".join(missing_raw_hashes)
            )
        )
    if extra_raw_hashes:
        errors.append(
            "manifest has unreferenced raw artifact checksums: {}".format(
                ", ".join(extra_raw_hashes)
            )
        )
    invalid_raw = []
    for relative in sorted(referenced_raw_artifacts):
        path = _safe_run_relative(root, relative)
        expected = raw_hashes.get(relative)
        if (
            path is None
            or not path.is_file()
            or not isinstance(expected, str)
            or len(expected) != 64
            or sha256_path(path) != expected
        ):
            invalid_raw.append(relative)
    if invalid_raw:
        errors.append(
            "raw artifact checksum mismatch: {}".format(
                ", ".join(invalid_raw)
            )
        )

    result_hashes: Dict[str, str] = {}
    for name in sorted(required):
        path = root / name
        try:
            if path.is_file():
                result_hashes[name] = sha256_path(path)
        except OSError as exc:
            errors.append(
                "{} cannot be hashed: {}: {}".format(
                    name, type(exc).__name__, exc
                )
            )
    return {
        "valid": not errors,
        "output": str(root),
        "documents": len(documents),
        "blocks": len(blocks),
        "chunks": len(chunks),
        "attempts": len(attempts),
        "errors": errors,
        "hashes": result_hashes,
    }


def _relative_artifact(value: Any, output_dir: Path) -> Optional[str]:
    try:
        path = Path(str(value)).resolve()
    except (OSError, ValueError):
        return None
    if not path.is_file():
        return None
    try:
        return path.relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        # Never leak or persist arbitrary absolute paths in portable run data.
        return None


def _safe_run_relative(root: Path, relative: str) -> Optional[Path]:
    if not relative or Path(relative).is_absolute():
        return None
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
    except (OSError, ValueError):
        return None
    return candidate


def _read_jsonl_for_verification(
    path: Path,
    label: str,
    errors: List[str],
) -> List[Any]:
    try:
        return list(iter_jsonl(path))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        errors.append(
            "{} is invalid: {}: {}".format(label, type(exc).__name__, exc)
        )
        return []


def _write_jsonl_record(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value))
    handle.write("\n")


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(_csv_safe_row(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _csv_safe_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        field: _csv_safe_value(row.get(field, ""))
        for field in REPORT_FIELDS
    }


def _csv_safe_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
