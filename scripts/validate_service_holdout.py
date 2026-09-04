#!/usr/bin/env python3
"""Fail-closed preflight for the frozen 36-case PNU service holdout v2.

The command passes only when four independent gates pass: holdout schema and
composition, DEV source-identity separation, evidence provenance against the
frozen corpus, and two-person sign-off for every case.

Before distributing a review packet, ``--pre-review`` runs the first three
machine gates without accepting or fabricating a human sign-off.

Example:
  python3 scripts/validate_service_holdout.py \
      --cases config/pnu-service-answer-holdout-v2.jsonl \
      --dev-manifest evidence/dev-source-manifest.json \
      --signoff evidence/holdout-v2-signoff.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:
    from .holdout_gold import (
        CHALLENGE_SPLIT,
        CORE_SPLIT,
        canonical_source_url,
        collect_holdout_validation_errors,
        collect_source_identities,
        load_json_or_jsonl,
        match_evidence,
        sha256_file,
    )
except ImportError:  # Direct execution: python scripts/validate_service_holdout.py
    script_directory = str(Path(__file__).resolve().parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    from holdout_gold import (  # type: ignore
        CHALLENGE_SPLIT,
        CORE_SPLIT,
        canonical_source_url,
        collect_holdout_validation_errors,
        collect_source_identities,
        load_json_or_jsonl,
        match_evidence,
        sha256_file,
    )

try:
    from . import build_holdout_signoff as signoff_artifacts
except ImportError:  # Direct execution and importlib-based tests.
    script_directory = str(Path(__file__).resolve().parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    import build_holdout_signoff as signoff_artifacts  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-holdout-v2.jsonl"
DEFAULT_CORPUS_INDEX = (
    REPO_ROOT
    / "processed"
    / "index"
    / "pnu-20260725-curated-cascade-v5-allow-suspect.sqlite"
)
REQUIRED_DEV_IDENTITIES = {
    "family_ids": "source document family IDs",
    "source_sha256s": "source SHA-256 values",
    "normalized_titles": "normalized source titles",
    "source_urls": "source URLs",
}
REQUIRED_REVIEWER_CHECKS = (
    "source_verified",
    "gold_verified",
    "answerability_verified",
    "label_verified",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "required JSON/JSONL DEV source manifest; repeat to check multiple "
            "manifests"
        ),
    )
    parser.add_argument(
        "--dev-input",
        type=Path,
        action="append",
        default=[],
        help="optional additional DEV JSON/JSONL input",
    )
    parser.add_argument(
        "--dev-cases",
        type=Path,
        action="append",
        default=[],
        help="optional additional DEV JSONL cases input",
    )
    parser.add_argument(
        "--corpus-index",
        type=Path,
        default=DEFAULT_CORPUS_INDEX,
        help="frozen SQLite corpus index used to verify every evidence option",
    )
    review_mode = parser.add_mutually_exclusive_group(required=True)
    review_mode.add_argument(
        "--signoff",
        type=Path,
        help="JSON object containing the completed two-person sign-off manifest",
    )
    review_mode.add_argument(
        "--pre-review",
        action="store_true",
        help=(
            "validate schema, DEV separation, and corpus evidence before human "
            "review; omit the sign-off gate"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one machine-readable summary object",
    )
    return parser


def _gate(errors: Sequence[str], **details: Any) -> dict[str, Any]:
    return {"ok": not errors, "errors": list(errors), **details}


def _load_cases_snapshot(path: Path) -> tuple[list[dict[str, Any]], bytes, str]:
    """Read, strictly parse, and hash one immutable cases-byte snapshot."""

    payload = signoff_artifacts._read_regular_bytes(path, label="cases")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: must be UTF-8: {exc}") from exc

    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = signoff_artifacts._load_json_bytes(
            line.encode("utf-8"), source=f"{path}:{line_number}"
        )
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        cases.append(value)
    return cases, payload, signoff_artifacts.sha256_bytes(payload)


def _new_errors(
    baseline: Sequence[str], combined: Sequence[str]
) -> list[str]:
    """Return multiset entries added to combined beyond baseline."""

    remaining = Counter(baseline)
    added: list[str] = []
    for error in combined:
        if remaining[error]:
            remaining[error] -= 1
        else:
            added.append(error)
    return added


def _validate_dev_gate(
    cases: Sequence[Mapping[str, Any]],
    schema_errors: Sequence[str],
    dev_paths: Sequence[Path],
) -> dict[str, Any]:
    errors: list[str] = []
    merged = {key: set() for key in REQUIRED_DEV_IDENTITIES}
    if not dev_paths:
        errors.append("at least one --dev-manifest is required")

    for path in dev_paths:
        try:
            identities = collect_source_identities(load_json_or_jsonl(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"DEV manifest {path}: {exc}")
            continue
        for key, label in REQUIRED_DEV_IDENTITIES.items():
            values = identities[key]
            if not values:
                errors.append(f"DEV manifest {path} contains no {label}")
            merged[key].update(values)

    combined = collect_holdout_validation_errors(
        cases,
        dev_family_ids=merged["family_ids"],
        dev_source_sha256s=merged["source_sha256s"],
        dev_normalized_titles=merged["normalized_titles"],
        dev_source_urls=merged["source_urls"],
    )
    leak_errors = _new_errors(schema_errors, combined)
    # DEV identities case-fold hashes. Supplement the shared validator's leak
    # report for an authored uppercase holdout SHA so casing cannot bypass the
    # separation gate.
    reported_sha_leaks = {
        error
        for error in leak_errors
        if "DEV source SHA-256 leak" in error
    }
    for case in cases:
        if case.get("split") != CORE_SPLIT:
            continue
        case_id = str(case.get("id") or "").strip()
        for claim in case.get("required_claims", []):
            if not isinstance(claim, Mapping):
                continue
            for option in claim.get("evidence_options", []):
                if not isinstance(option, Mapping):
                    continue
                source_sha = str(option.get("source_sha256") or "").strip().casefold()
                if source_sha not in merged["source_sha256s"]:
                    continue
                marker = f"DEV source SHA-256 leak {source_sha!r}"
                if any(marker in error for error in reported_sha_leaks):
                    continue
                error = f"dataset: {marker} in {case_id}"
                leak_errors.append(error)
                reported_sha_leaks.add(error)
    errors.extend(leak_errors)
    return _gate(
        errors,
        inputs=[str(path) for path in dev_paths],
        family_count=len(merged["family_ids"]),
        source_sha256_count=len(merged["source_sha256s"]),
        normalized_title_count=len(merged["normalized_titles"]),
        source_url_count=len(merged["source_urls"]),
    )


def _iter_evidence_options(
    cases: Sequence[Mapping[str, Any]],
):
    for case_index, case in enumerate(cases):
        case_id = str(case.get("id") or f"cases[{case_index}]").strip()
        claims = case.get("required_claims")
        if not isinstance(claims, list):
            continue
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                continue
            claim_id = str(claim.get("claim_id") or claim_index).strip()
            options = claim.get("evidence_options")
            if not isinstance(options, list):
                continue
            for option_index, option in enumerate(options):
                if isinstance(option, Mapping):
                    label = (
                        f"{case_id}.required_claims[{claim_id!r}]."
                        f"evidence_options[{option_index}]"
                    )
                    yield label, claim, option


def _normalized_metadata(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _resolve_source_path(repo_root: Path, source_path: str) -> tuple[Optional[Path], str]:
    raw = Path(source_path)
    if raw.is_absolute():
        return None, "source_path must be relative to the repository root"
    root = repo_root.resolve()
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None, "source_path escapes the repository root"
    return resolved, ""


def _validate_corpus_gate(
    cases: Sequence[Mapping[str, Any]],
    corpus_index: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    option_count = 0
    verified_option_count = 0
    source_hash_cache: dict[Path, str] = {}
    connection: Optional[sqlite3.Connection] = None

    try:
        uri = f"file:{corpus_index.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        required_columns = {
            "chunk_id",
            "chunk_index",
            "document_id",
            "source_path",
            "source_title",
            "source_url",
            "text",
        }
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            errors.append(
                "corpus index chunks table is missing columns: "
                + ", ".join(missing_columns)
            )
            return _gate(
                errors,
                index_path=str(corpus_index),
                evidence_option_count=0,
                verified_evidence_option_count=0,
            )

        for label, claim, option in _iter_evidence_options(cases):
            option_count += 1
            before = len(errors)
            document_id = str(option.get("document_id") or "").strip()
            rows = (
                connection.execute(
                    """
                    SELECT chunk_id, chunk_index, document_id, source_path,
                           source_title, source_url, text
                    FROM chunks
                    WHERE document_id = ?
                    ORDER BY chunk_index, chunk_id
                    """,
                    (document_id,),
                ).fetchall()
                if document_id
                else []
            )
            if not rows:
                errors.append(
                    f"{label}: document_id {document_id!r} is absent from corpus index"
                )
                continue

            expected_path = str(option.get("source_path") or "").strip()
            actual_paths = {
                str(row["source_path"] or "").strip() for row in rows
            }
            if actual_paths != {expected_path}:
                errors.append(
                    f"{label}: source_path mismatch; expected {expected_path!r}, "
                    f"index has {sorted(actual_paths)!r}"
                )

            expected_title = _normalized_metadata(option.get("source_title"))
            actual_titles = {
                _normalized_metadata(row["source_title"]) for row in rows
            }
            if actual_titles != {expected_title}:
                errors.append(
                    f"{label}: source_title mismatch; expected {expected_title!r}, "
                    f"index has {sorted(actual_titles)!r}"
                )

            expected_url = canonical_source_url(option.get("source_url"))
            actual_urls = {canonical_source_url(row["source_url"]) for row in rows}
            if actual_urls != {expected_url}:
                errors.append(
                    f"{label}: source_url mismatch; expected {expected_url!r}, "
                    f"index has {sorted(actual_urls)!r}"
                )

            resolved_source, path_error = _resolve_source_path(
                repo_root, expected_path
            )
            if path_error:
                errors.append(f"{label}: {path_error}")
            elif resolved_source is None or not resolved_source.is_file():
                errors.append(
                    f"{label}: raw source file does not exist: {expected_path!r}"
                )
            else:
                actual_sha = source_hash_cache.get(resolved_source)
                if actual_sha is None:
                    actual_sha = sha256_file(resolved_source)
                    source_hash_cache[resolved_source] = actual_sha
                expected_sha = str(option.get("source_sha256") or "").casefold()
                if actual_sha != expected_sha:
                    errors.append(
                        f"{label}: source_sha256 mismatch; expected "
                        f"{expected_sha!r}, raw file has {actual_sha!r}"
                    )

            aggregate_text = "\n".join(str(row["text"] or "") for row in rows)
            match = match_evidence(
                {"document_id": document_id, "text": aggregate_text},
                option,
                claim=claim,
            )
            if not match.get("matched"):
                errors.append(
                    f"{label}: quote/claim does not match indexed document "
                    f"({match.get('reason', 'unknown')})"
                )

            if len(errors) == before:
                verified_option_count += 1
    except (OSError, sqlite3.Error) as exc:
        errors.append(f"cannot verify corpus index {corpus_index}: {exc}")
    finally:
        if connection is not None:
            connection.close()

    return _gate(
        errors,
        index_path=str(corpus_index),
        evidence_option_count=option_count,
        verified_evidence_option_count=verified_option_count,
    )


def _signoff_records(payload: Any) -> Optional[list[Any]]:
    if isinstance(payload, Mapping):
        records = payload.get("case_signoffs")
        if isinstance(records, list):
            return records
    return None


def _resolve_review_artifact_path(
    repo_root: Path,
    raw_path: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise ValueError(f"{label}: path must be a non-empty, trimmed string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label}: path must be repository-relative")

    root = repo_root.resolve()
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label}: symlink path components are not allowed")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes the repository root") from exc
    return resolved


def _read_repo_review_artifact(
    repo_root: Path,
    raw_path: Any,
    *,
    label: str,
) -> tuple[Path, bytes]:
    """Read a repo-relative file without following any symlink component."""

    resolved = _resolve_review_artifact_path(repo_root, raw_path, label=label)
    relative = Path(raw_path)
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptors: list[int] = []
    file_descriptor: Optional[int] = None
    try:
        directory_descriptors.append(os.open(repo_root.resolve(), directory_flags))
        for component in relative.parts[:-1]:
            descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptors[-1],
            )
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError(
                    f"{label}: non-directory path component is not allowed"
                )
            directory_descriptors.append(descriptor)

        file_descriptor = os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=directory_descriptors[-1],
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}: must be a regular file: {resolved}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        if (
            (before.st_dev, before.st_ino)
            != (after.st_dev, after.st_ino)
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise ValueError(f"{label}: changed while being read: {resolved}")

        # Re-resolve after the descriptor read so a swapped visible parent is
        # also rejected, while every actual open above remained beneath root.
        current_resolved = _resolve_review_artifact_path(
            repo_root, raw_path, label=label
        )
        current = os.lstat(current_resolved)
        if (
            stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise ValueError(f"{label}: path identity changed while being read")
        return current_resolved, b"".join(chunks)
    except (OSError, IndexError) as exc:
        raise ValueError(f"{label}: cannot securely open {resolved}: {exc}") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _validate_signoff_provenance(
    payload: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_cases_sha256: str,
    cases_path: Path,
    signoff_path: Path,
    repo_root: Path,
) -> tuple[list[str], dict[str, dict[str, Any]], Optional[str], Optional[str]]:
    errors: list[str] = []
    bound_reviews: dict[str, dict[str, Any]] = {}
    packet_sha256: Optional[str] = None
    packet_display_path: Optional[str] = None

    if payload.get("schema_version") != signoff_artifacts.SIGNOFF_SCHEMA_VERSION:
        errors.append(
            "sign-off manifest schema_version must be "
            f"{signoff_artifacts.SIGNOFF_SCHEMA_VERSION!r}"
        )

    artifact_paths: list[tuple[str, Path]] = [
        ("cases", cases_path),
        ("sign-off", signoff_path),
    ]
    if signoff_artifacts.paths_alias(cases_path, signoff_path):
        errors.append(
            f"sign-off manifest aliases cases: {signoff_path}"
        )

    packet_path: Optional[Path] = None
    packet_payload: Optional[bytes] = None
    try:
        packet_display_path = str(payload.get("review_packet_path") or "") or None
        packet_path, packet_payload = _read_repo_review_artifact(
            repo_root,
            payload.get("review_packet_path"),
            label="sign-off review_packet_path",
        )
        for other_label, other_path in artifact_paths:
            if signoff_artifacts.paths_alias(packet_path, other_path):
                raise ValueError(
                    "sign-off review_packet_path aliases "
                    f"{other_label}: {packet_path}"
                )
        if not packet_payload:
            raise ValueError("review packet must not be empty")
        packet_sha256 = signoff_artifacts.sha256_bytes(packet_payload)
        artifact_paths.append(("review packet", packet_path))
    except (OSError, ValueError) as exc:
        errors.append(str(exc))

    signed_packet_sha = payload.get("review_packet_sha256")
    if (
        not isinstance(signed_packet_sha, str)
        or not signoff_artifacts.SHA256_PATTERN.fullmatch(signed_packet_sha)
    ):
        errors.append("sign-off review_packet_sha256 must be a lowercase SHA-256")
    elif packet_sha256 is not None and signed_packet_sha != packet_sha256:
        errors.append(
            "sign-off review_packet_sha256 does not match the current packet bytes"
        )

    raw_provenance = payload.get("review_response_sha256s")
    if not isinstance(raw_provenance, list) or len(raw_provenance) != 2:
        errors.append(
            "sign-off review_response_sha256s must contain exactly 2 records"
        )
        provenance_records = raw_provenance if isinstance(raw_provenance, list) else []
    else:
        provenance_records = raw_provenance

    case_ids = [str(case.get("id") or "").strip() for case in cases]
    for index, record in enumerate(provenance_records):
        source = f"review_response_sha256s[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{source}: must be an object")
            continue
        expected_keys = {"reviewer_slot", "reviewer_id", "sha256", "path"}
        if set(record) != expected_keys:
            errors.append(f"{source}: keys must be exactly {sorted(expected_keys)}")

        slot = record.get("reviewer_slot")
        if slot not in {"A", "B"}:
            errors.append(f"{source}.reviewer_slot: must be 'A' or 'B'")
            continue
        if slot in bound_reviews:
            errors.append(f"{source}.reviewer_slot: duplicate slot {slot!r}")
            continue

        response_path: Optional[Path] = None
        response_payload: Optional[bytes] = None
        try:
            response_path, response_payload = _read_repo_review_artifact(
                repo_root,
                record.get("path"),
                label=f"{source}.path",
            )
            for other_label, other_path in artifact_paths:
                if signoff_artifacts.paths_alias(response_path, other_path):
                    raise ValueError(
                        f"{source}.path aliases {other_label}: {response_path}"
                    )
            artifact_paths.append((f"Reviewer {slot} response", response_path))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

        signed_response_sha = record.get("sha256")
        if (
            not isinstance(signed_response_sha, str)
            or not signoff_artifacts.SHA256_PATTERN.fullmatch(signed_response_sha)
        ):
            errors.append(f"{source}.sha256: must be a lowercase SHA-256")
        elif (
            response_payload is not None
            and signed_response_sha
            != signoff_artifacts.sha256_bytes(response_payload)
        ):
            errors.append(f"{source}.sha256: does not match response bytes")

        if response_payload is None or packet_payload is None:
            continue
        try:
            reviewer_id, normalized_id, reviews = (
                signoff_artifacts.validate_completed_response(
                    response_payload,
                    source=str(response_path),
                    expected_slot=slot,
                    expected_case_ids=case_ids,
                    expected_cases_sha256=expected_cases_sha256,
                    expected_packet_sha256=signoff_artifacts.sha256_bytes(
                        packet_payload
                    ),
                )
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if record.get("reviewer_id") != reviewer_id:
            errors.append(
                f"{source}.reviewer_id: does not match the bound response"
            )
            continue
        bound_reviews[slot] = {
            "reviewer_id": reviewer_id,
            "normalized_id": normalized_id,
            "records": reviews,
        }

    if set(bound_reviews) != {"A", "B"}:
        errors.append("sign-off provenance must validate Reviewer A and B responses")
    elif (
        bound_reviews["A"]["normalized_id"]
        == bound_reviews["B"]["normalized_id"]
    ):
        errors.append("sign-off provenance requires two distinct reviewer IDs")

    return errors, bound_reviews, packet_sha256, packet_display_path


def _validate_signoff_gate(
    cases: Sequence[Mapping[str, Any]],
    signoff_path: Optional[Path],
    *,
    expected_cases_sha256: str,
    cases_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    errors: list[str] = []
    approved_case_count = 0
    reviewer_roster: Optional[frozenset[str]] = None
    if signoff_path is None:
        return _gate(
            ["--signoff is required"],
            path=None,
            record_count=0,
            approved_case_count=0,
        )

    try:
        signoff_payload = signoff_artifacts._read_regular_bytes(
            signoff_path,
            label="sign-off manifest",
        )
        payload = signoff_artifacts._load_json_bytes(
            signoff_payload,
            source=str(signoff_path),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _gate(
            [f"sign-off manifest {signoff_path}: {exc}"],
            path=str(signoff_path),
            record_count=0,
            approved_case_count=0,
        )
    if not isinstance(payload, Mapping):
        return _gate(
            ["sign-off manifest must be a JSON object"],
            path=str(signoff_path),
            record_count=0,
            approved_case_count=0,
        )
    expected_manifest_keys = {
        "schema_version",
        "cases_sha256",
        "review_packet_path",
        "review_packet_sha256",
        "review_response_sha256s",
        "case_signoffs",
    }
    if set(payload) != expected_manifest_keys:
        errors.append(
            "sign-off manifest keys must match the provenance-bound schema exactly"
        )

    raw_cases_sha256 = payload.get("cases_sha256")
    signed_cases_sha256 = (
        raw_cases_sha256 if isinstance(raw_cases_sha256, str) else ""
    )
    if not signoff_artifacts.SHA256_PATTERN.fullmatch(signed_cases_sha256):
        errors.append(
            "sign-off manifest cases_sha256: must be a lowercase SHA-256 and "
            "must bind approvals to the exact holdout file"
        )
    elif signed_cases_sha256 != expected_cases_sha256:
        errors.append(
            "sign-off manifest cases_sha256 mismatch; expected "
            f"{expected_cases_sha256!r}, found {signed_cases_sha256!r}"
        )

    provenance_errors, bound_reviews, packet_sha256, packet_path = (
        _validate_signoff_provenance(
            payload,
            cases,
            expected_cases_sha256=expected_cases_sha256,
            cases_path=cases_path,
            signoff_path=signoff_path,
            repo_root=repo_root,
        )
    )
    errors.extend(provenance_errors)
    records = _signoff_records(payload)
    if records is None:
        return _gate(
            errors
            + [
                "sign-off manifest must be a JSON object containing a list "
                "under 'case_signoffs'"
            ],
            path=str(signoff_path),
            record_count=0,
            approved_case_count=0,
        )

    expected_case_ids = {
        str(case.get("id") or "").strip()
        for case in cases
        if str(case.get("id") or "").strip()
    }
    expected_case_order = [
        str(case.get("id") or "").strip()
        for case in cases
        if str(case.get("id") or "").strip()
    ]
    seen_case_ids: Counter[str] = Counter()
    for index, record in enumerate(records):
        record_path = f"signoffs[{index}]"
        before = len(errors)
        if not isinstance(record, Mapping):
            errors.append(f"{record_path}: must be an object")
            continue
        if set(record) != {"case_id", "reviewers"}:
            errors.append(
                f"{record_path}: keys must be exactly ['case_id', 'reviewers']"
            )
        raw_case_id = record.get("case_id")
        case_id = raw_case_id if isinstance(raw_case_id, str) else ""
        if not case_id or case_id != case_id.strip():
            errors.append(
                f"{record_path}.case_id: must be a non-empty, trimmed string"
            )
        else:
            seen_case_ids[case_id] += 1
            if case_id not in expected_case_ids:
                errors.append(f"{record_path}.case_id: unknown case {case_id!r}")
            if index >= len(expected_case_order) or case_id != expected_case_order[index]:
                errors.append(
                    f"{record_path}.case_id: must preserve cases-file order"
                )

        reviewers = record.get("reviewers")
        if not isinstance(reviewers, list):
            errors.append(f"{record_path}.reviewers: must be a list")
            continue
        if len(reviewers) != 2:
            errors.append(
                f"{record_path}.reviewers: requires exactly 2 reviewer records"
            )
        reviewer_ids: list[str] = []
        reviewer_slots: list[str] = []
        for reviewer_index, reviewer in enumerate(reviewers):
            reviewer_path = f"{record_path}.reviewers[{reviewer_index}]"
            if not isinstance(reviewer, Mapping):
                errors.append(f"{reviewer_path}: must be an object")
                continue
            expected_reviewer_keys = {
                "reviewer_slot",
                "reviewer_id",
                *REQUIRED_REVIEWER_CHECKS,
                "decision",
                "notes",
            }
            if set(reviewer) != expected_reviewer_keys:
                errors.append(
                    f"{reviewer_path}: keys must match the merged response schema"
                )
            reviewer_slot = reviewer.get("reviewer_slot")
            if reviewer_slot not in {"A", "B"}:
                errors.append(f"{reviewer_path}.reviewer_slot: must be 'A' or 'B'")
            else:
                reviewer_slots.append(reviewer_slot)
            reviewer_id = str(reviewer.get("reviewer_id") or "").strip()
            if not reviewer_id:
                errors.append(
                    f"{reviewer_path}.reviewer_id: must be a non-empty string"
                )
            else:
                reviewer_ids.append(
                    unicodedata.normalize("NFKC", reviewer_id).casefold()
                )
            for field in REQUIRED_REVIEWER_CHECKS:
                if reviewer.get(field) is not True:
                    errors.append(f"{reviewer_path}.{field}: must be true")
            if reviewer.get("decision") != "PASS":
                errors.append(f"{reviewer_path}.decision: must be exactly 'PASS'")
            if not isinstance(reviewer.get("notes"), str):
                errors.append(f"{reviewer_path}.notes: must be a string")

            if (
                reviewer_slot in bound_reviews
                and index < len(bound_reviews[reviewer_slot]["records"])
            ):
                bound = bound_reviews[reviewer_slot]
                source_review = bound["records"][index]
                expected_values = {
                    "reviewer_slot": reviewer_slot,
                    "reviewer_id": bound["reviewer_id"],
                    **{
                        field: source_review[field]
                        for field in REQUIRED_REVIEWER_CHECKS
                    },
                    "decision": source_review["decision"],
                    "notes": source_review["notes"],
                }
                if any(
                    reviewer.get(field) != value
                    for field, value in expected_values.items()
                ):
                    errors.append(
                        f"{reviewer_path}: does not exactly match the bound "
                        f"Reviewer {reviewer_slot} response"
                    )
        distinct_reviewers = set(reviewer_ids)
        if len(distinct_reviewers) != 2:
            errors.append(
                f"{record_path}.reviewers: requires exactly 2 distinct reviewers"
            )
        if len(reviewer_ids) != len(distinct_reviewers):
            errors.append(f"{record_path}.reviewers: duplicate reviewer_id")
        if set(reviewer_slots) != {"A", "B"} or len(reviewer_slots) != 2:
            errors.append(
                f"{record_path}.reviewers: requires exactly one Reviewer A and B"
            )

        if len(reviewers) == 2 and len(distinct_reviewers) == 2:
            current_roster = frozenset(distinct_reviewers)
            if reviewer_roster is None:
                reviewer_roster = current_roster
            elif current_roster != reviewer_roster:
                errors.append(
                    f"{record_path}.reviewers: must use the same 2-person "
                    "reviewer roster for every case"
                )

        if len(errors) == before:
            approved_case_count += 1

    for case_id, count in sorted(seen_case_ids.items()):
        if count > 1:
            errors.append(
                f"sign-off case {case_id!r} must occur exactly once; found {count}"
            )
    missing = sorted(expected_case_ids - set(seen_case_ids))
    for case_id in missing:
        errors.append(f"sign-off missing case {case_id!r}")

    return _gate(
        errors,
        path=str(signoff_path),
        cases_sha256=signed_cases_sha256 or None,
        record_count=len(records),
        approved_case_count=approved_case_count,
        reviewer_ids=sorted(reviewer_roster or ()),
        review_packet_path=packet_path,
        review_packet_sha256=packet_sha256,
        validated_review_response_count=len(bound_reviews),
    )


def _summary_from_gates(
    cases_path: Path,
    cases_sha256: str,
    cases: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    split_counts = Counter(str(case.get("split") or "") for case in cases)
    role_cases = sum(
        1
        for case in cases
        if case.get("split") == CORE_SPLIT and case.get("role") is not None
    )
    all_errors = [
        f"{gate_name}: {error}"
        for gate_name, gate in gates.items()
        for error in gate.get("errors", [])
    ]
    return {
        "ok": all(bool(gate.get("ok")) for gate in gates.values()),
        "cases_path": str(cases_path),
        "cases_sha256": cases_sha256,
        "case_count": len(cases),
        "core_count": split_counts[CORE_SPLIT],
        "challenge_count": split_counts[CHALLENGE_SPLIT],
        "role_case_count": role_cases,
        "dev_family_count": gates["dev"].get("family_count", 0),
        "dev_inputs": gates["dev"].get("inputs", []),
        "gates": dict(gates),
        "errors": all_errors,
    }


def validate_paths(
    cases_path: Path,
    dev_paths: Sequence[Path],
    *,
    corpus_index: Path = DEFAULT_CORPUS_INDEX,
    signoff_path: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
    pre_review: bool = False,
) -> dict[str, Any]:
    cases, cases_payload, cases_sha256 = _load_cases_snapshot(cases_path)
    schema_errors = collect_holdout_validation_errors(cases)
    gates: dict[str, dict[str, Any]] = {
        "schema": _gate(schema_errors),
        "dev": _validate_dev_gate(cases, schema_errors, dev_paths),
        "corpus": _validate_corpus_gate(
            cases, corpus_index, repo_root=repo_root
        ),
    }
    if not pre_review:
        gates["signoff"] = _validate_signoff_gate(
            cases,
            signoff_path,
            expected_cases_sha256=cases_sha256,
            cases_path=cases_path,
            repo_root=repo_root,
        )
    try:
        current_cases_payload = signoff_artifacts._read_regular_bytes(
            cases_path, label="cases"
        )
        if current_cases_payload != cases_payload:
            gates["schema"]["errors"].append(
                "cases bytes changed during validation"
            )
            gates["schema"]["ok"] = False
    except (OSError, ValueError) as exc:
        gates["schema"]["errors"].append(
            f"cannot confirm cases snapshot after validation: {exc}"
        )
        gates["schema"]["ok"] = False
    return _summary_from_gates(cases_path, cases_sha256, cases, gates)


def _load_failure_summary(
    cases_path: Path,
    error: Exception,
    *,
    pre_review: bool = False,
) -> dict[str, Any]:
    message = str(error)
    gates = {
        "schema": _gate([message]),
        "dev": _gate(["not evaluated because cases could not be loaded"]),
        "corpus": _gate(["not evaluated because cases could not be loaded"]),
    }
    if not pre_review:
        gates["signoff"] = _gate(
            ["not evaluated because cases could not be loaded"]
        )
    return {
        "ok": False,
        "cases_path": str(cases_path),
        "gates": gates,
        "errors": [
            f"{gate_name}: {gate_error}"
            for gate_name, gate in gates.items()
            for gate_error in gate["errors"]
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dev_paths = list(args.dev_manifest) + list(args.dev_input) + list(args.dev_cases)
    try:
        summary = validate_paths(
            args.cases,
            dev_paths,
            corpus_index=args.corpus_index,
            signoff_path=args.signoff,
            pre_review=args.pre_review,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        summary = _load_failure_summary(
            args.cases,
            exc,
            pre_review=args.pre_review,
        )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        for error in summary.get("errors", []):
            print(f"ERROR {error}")
        gates = summary.get("gates", {})
        gate_status = " ".join(
            f"{name}={'pass' if gate.get('ok') else 'fail'}"
            for name, gate in gates.items()
        )
        print(
            "cases={case_count} core={core_count} challenge={challenge_count} "
            "role_cases={role_case_count} {gate_status} errors={error_count}".format(
                case_count=summary.get("case_count", 0),
                core_count=summary.get("core_count", 0),
                challenge_count=summary.get("challenge_count", 0),
                role_case_count=summary.get("role_case_count", 0),
                gate_status=gate_status,
                error_count=len(summary.get("errors", [])),
            )
        )
        if summary.get("cases_sha256"):
            print(f"cases_sha256={summary['cases_sha256']}")
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
