#!/usr/bin/env python3
"""Create and merge SHA-bound, two-person holdout review responses.

The create mode publishes separate Reviewer A/B templates without clobbering
existing files.  The merge mode accepts only two complete, independent PASS
responses for the exact cases and review-packet bytes, then emits the sign-off
shape consumed by ``validate_service_holdout.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .immutable_outputs import (
        paths_alias,
        publish_immutable_texts,
        reject_symlink_inputs,
        require_new_outputs,
    )
except ImportError:  # Direct execution and importlib-based tests.
    script_directory = str(Path(__file__).resolve().parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    from immutable_outputs import (  # type: ignore
        paths_alias,
        publish_immutable_texts,
        reject_symlink_inputs,
        require_new_outputs,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-holdout-v2.draft.jsonl"
DEFAULT_REVIEW_PACKET = REPO_ROOT / "docs" / "holdout-v2-human-review.md"

RESPONSE_SCHEMA_VERSION = "pnu.holdout-independent-review.v1"
SIGNOFF_SCHEMA_VERSION = "pnu.holdout-signoff.v1"
REVIEW_CHECKS = (
    "source_verified",
    "gold_verified",
    "answerability_verified",
    "label_verified",
)
RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "reviewer_slot",
        "cases_sha256",
        "review_packet_sha256",
        "reviewer_id",
        "independent_review_confirmed",
        "case_reviews",
    }
)
CASE_REVIEW_KEYS = frozenset(
    {"case_id", *REVIEW_CHECKS, "decision", "notes"}
)
REVIEWER_PLACEHOLDERS = {
    "__reviewer_a_id__",
    "__reviewer_b_id__",
    "reviewer_id",
    "reviewer id",
    "placeholder",
    "replace_me",
    "replace me",
    "todo",
    "tbd",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON value {value!r} is not allowed")


def _load_json_bytes(payload: bytes, *, source: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: must be UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid strict JSON: {exc}") from exc


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    """Read one exact regular-file snapshot through inode-bound directories."""

    reject_symlink_inputs([path])
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label}: cannot resolve {path}: {exc}") from exc

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptors: list[int] = []
    descriptor: int | None = None
    try:
        directory_descriptors.append(os.open(resolved.anchor, directory_flags))
        components = resolved.parts[1:]
        for component in components[:-1]:
            child = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptors[-1],
            )
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise ValueError(
                    f"{label}: non-directory path component is not allowed"
                )
            directory_descriptors.append(child)
        descriptor = os.open(
            components[-1],
            file_flags,
            dir_fd=directory_descriptors[-1],
        )
    except (OSError, IndexError, ValueError) as exc:
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)
        raise ValueError(f"{label}: cannot securely open {path}: {exc}") from exc

    try:
        assert descriptor is not None
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label}: must be a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)

    identity_before = (before.st_dev, before.st_ino)
    identity_after = (after.st_dev, after.st_ino)
    metadata_before = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    metadata_after = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after or metadata_before != metadata_after:
        raise ValueError(f"{label}: changed while being read: {path}")
    try:
        current_resolved = path.resolve(strict=True)
        current = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label}: disappeared after being read: {path}") from exc
    if (
        current_resolved != resolved
        or stat.S_ISLNK(current.st_mode)
        or (current.st_dev, current.st_ino) != identity_after
    ):
        raise ValueError(f"{label}: path identity changed while being read: {path}")
    return b"".join(chunks)


def _reject_aliases(named_paths: Sequence[tuple[str, Path]]) -> None:
    for index, (left_name, left_path) in enumerate(named_paths):
        for right_name, right_path in named_paths[index + 1 :]:
            if paths_alias(left_path, right_path):
                raise ValueError(
                    f"{left_name} and {right_name} must be distinct, non-alias paths: "
                    f"{left_path} / {right_path}"
                )


def _assert_unchanged(snapshots: Sequence[tuple[str, Path, bytes]]) -> None:
    for label, path, expected in snapshots:
        current = _read_regular_bytes(path, label=label)
        if current != expected:
            raise ValueError(
                f"{label}: bytes changed before immutable publication: {path}"
            )


def load_case_ids(payload: bytes, *, source: str) -> list[str]:
    """Load exact JSONL case IDs and reject ambiguous records or duplicates."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: must be UTF-8: {exc}") from exc

    case_ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = _load_json_bytes(
            line.encode("utf-8"), source=f"{source}:{line_number}"
        )
        if not isinstance(value, Mapping):
            raise ValueError(f"{source}:{line_number}: case must be a JSON object")
        case_id = value.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id != case_id.strip()
        ):
            raise ValueError(
                f"{source}:{line_number}: id must be a non-empty, trimmed string"
            )
        if case_id in seen:
            raise ValueError(f"{source}:{line_number}: duplicate id {case_id!r}")
        seen.add(case_id)
        case_ids.append(case_id)
    if not case_ids:
        raise ValueError(f"{source}: contains no cases")
    return case_ids


def _render_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def response_template(
    case_ids: Sequence[str],
    *,
    reviewer_slot: str,
    cases_sha256: str,
    review_packet_sha256: str,
) -> dict[str, Any]:
    if reviewer_slot not in {"A", "B"}:
        raise ValueError("reviewer_slot must be A or B")
    if not SHA256_PATTERN.fullmatch(cases_sha256):
        raise ValueError("cases_sha256 must be a lowercase SHA-256")
    if not SHA256_PATTERN.fullmatch(review_packet_sha256):
        raise ValueError("review_packet_sha256 must be a lowercase SHA-256")
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("case_ids must be non-empty and unique")

    reviews: list[dict[str, Any]] = []
    for case_id in case_ids:
        review: dict[str, Any] = {"case_id": case_id}
        review.update({check: None for check in REVIEW_CHECKS})
        review.update({"decision": "PENDING", "notes": ""})
        reviews.append(review)
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "reviewer_slot": reviewer_slot,
        "cases_sha256": cases_sha256,
        "review_packet_sha256": review_packet_sha256,
        "reviewer_id": f"__REVIEWER_{reviewer_slot}_ID__",
        "independent_review_confirmed": False,
        "case_reviews": reviews,
    }


def _template_texts(
    cases_payload: bytes,
    packet_payload: bytes,
    *,
    cases_source: str,
) -> tuple[str, str, list[str]]:
    if not packet_payload:
        raise ValueError("review packet must not be empty")
    case_ids = load_case_ids(cases_payload, source=cases_source)
    cases_sha256 = sha256_bytes(cases_payload)
    packet_sha256 = sha256_bytes(packet_payload)
    reviewer_a = response_template(
        case_ids,
        reviewer_slot="A",
        cases_sha256=cases_sha256,
        review_packet_sha256=packet_sha256,
    )
    reviewer_b = response_template(
        case_ids,
        reviewer_slot="B",
        cases_sha256=cases_sha256,
        review_packet_sha256=packet_sha256,
    )
    return _render_json(reviewer_a), _render_json(reviewer_b), case_ids


def create_templates(
    *,
    cases_path: Path,
    review_packet_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
) -> int:
    inputs = [("cases", cases_path), ("review packet", review_packet_path)]
    outputs = [
        ("Reviewer A template", reviewer_a_path),
        ("Reviewer B template", reviewer_b_path),
    ]
    reject_symlink_inputs([path for _, path in inputs])
    _reject_aliases([*inputs, *outputs])
    require_new_outputs([path for _, path in outputs])

    cases_payload = _read_regular_bytes(cases_path, label="cases")
    packet_payload = _read_regular_bytes(review_packet_path, label="review packet")
    reviewer_a_text, reviewer_b_text, case_ids = _template_texts(
        cases_payload,
        packet_payload,
        cases_source=str(cases_path),
    )
    _assert_unchanged(
        [
            ("cases", cases_path, cases_payload),
            ("review packet", review_packet_path, packet_payload),
        ]
    )
    _reject_aliases([*inputs, *outputs])
    # A is authoritative and linked last: its existence certifies B was also
    # published and directory-fsynced by publish_immutable_texts.
    publish_immutable_texts(
        {
            reviewer_a_path: reviewer_a_text,
            reviewer_b_path: reviewer_b_text,
        },
        authoritative_path=reviewer_a_path,
    )
    return len(case_ids)


def check_templates(
    *,
    cases_path: Path,
    review_packet_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
) -> int:
    named_paths = [
        ("cases", cases_path),
        ("review packet", review_packet_path),
        ("Reviewer A template", reviewer_a_path),
        ("Reviewer B template", reviewer_b_path),
    ]
    reject_symlink_inputs([path for _, path in named_paths])
    _reject_aliases(named_paths)
    cases_payload = _read_regular_bytes(cases_path, label="cases")
    packet_payload = _read_regular_bytes(review_packet_path, label="review packet")
    expected_a, expected_b, case_ids = _template_texts(
        cases_payload,
        packet_payload,
        cases_source=str(cases_path),
    )
    actual_a = _read_regular_bytes(reviewer_a_path, label="Reviewer A template")
    actual_b = _read_regular_bytes(reviewer_b_path, label="Reviewer B template")
    if actual_a != expected_a.encode("utf-8"):
        raise ValueError("Reviewer A template is missing or stale")
    if actual_b != expected_b.encode("utf-8"):
        raise ValueError("Reviewer B template is missing or stale")
    _assert_unchanged(
        [
            ("cases", cases_path, cases_payload),
            ("review packet", review_packet_path, packet_payload),
            ("Reviewer A template", reviewer_a_path, actual_a),
            ("Reviewer B template", reviewer_b_path, actual_b),
        ]
    )
    _reject_aliases(named_paths)
    return len(case_ids)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    source: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    raise ValueError(
        f"{source}: keys must match the template exactly ({', '.join(details)})"
    )


def _normalized_reviewer_id(value: Any, *, source: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{source}.reviewer_id: must be a non-empty, trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{source}.reviewer_id: control characters are not allowed")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if normalized in REVIEWER_PLACEHOLDERS:
        raise ValueError(f"{source}.reviewer_id: template placeholder is not allowed")
    return value, normalized


def _validate_case_sequence(
    records: Sequence[Mapping[str, Any]],
    expected_case_ids: Sequence[str],
    *,
    source: str,
) -> None:
    actual_case_ids: list[str] = []
    for index, record in enumerate(records):
        case_id = record.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError(
                f"{source}.case_reviews[{index}].case_id: must be a string"
            )
        actual_case_ids.append(case_id)
    if actual_case_ids == list(expected_case_ids):
        return

    actual_counts = Counter(actual_case_ids)
    duplicates = sorted(
        case_id for case_id, count in actual_counts.items() if count > 1
    )
    expected = set(expected_case_ids)
    actual = set(actual_case_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if duplicates:
        details.append(f"duplicate={duplicates}")
    if missing:
        details.append(f"missing={missing}")
    if extra:
        details.append(f"extra={extra}")
    if not details:
        details.append("case order differs from the cases file")
    raise ValueError(
        f"{source}.case_reviews must contain every current case exactly once "
        f"and in cases-file order ({'; '.join(details)})"
    )


def validate_completed_response(
    payload: bytes,
    *,
    source: str,
    expected_slot: str,
    expected_case_ids: Sequence[str],
    expected_cases_sha256: str,
    expected_packet_sha256: str,
) -> tuple[str, str, list[Mapping[str, Any]]]:
    value = _load_json_bytes(payload, source=source)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: response must be a JSON object")
    _require_exact_keys(value, RESPONSE_KEYS, source=source)
    if value.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError(f"{source}.schema_version: unsupported or stale template")
    if value.get("reviewer_slot") != expected_slot:
        raise ValueError(f"{source}.reviewer_slot: must be {expected_slot!r}")
    if value.get("cases_sha256") != expected_cases_sha256:
        raise ValueError(f"{source}.cases_sha256: stale or mismatched cases bytes")
    if value.get("review_packet_sha256") != expected_packet_sha256:
        raise ValueError(
            f"{source}.review_packet_sha256: stale or mismatched review packet bytes"
        )
    reviewer_id, normalized_id = _normalized_reviewer_id(
        value.get("reviewer_id"), source=source
    )
    if value.get("independent_review_confirmed") is not True:
        raise ValueError(
            f"{source}.independent_review_confirmed: must be the JSON boolean true"
        )

    raw_records = value.get("case_reviews")
    if not isinstance(raw_records, list):
        raise ValueError(f"{source}.case_reviews: must be a list")
    records: list[Mapping[str, Any]] = []
    for index, record in enumerate(raw_records):
        record_source = f"{source}.case_reviews[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{record_source}: must be an object")
        _require_exact_keys(record, CASE_REVIEW_KEYS, source=record_source)
        records.append(record)
    _validate_case_sequence(records, expected_case_ids, source=source)

    for index, record in enumerate(records):
        record_source = f"{source}.case_reviews[{index}]"
        for check in REVIEW_CHECKS:
            if record.get(check) is not True:
                raise ValueError(
                    f"{record_source}.{check}: must be the JSON boolean true"
                )
        if record.get("decision") != "PASS":
            raise ValueError(f"{record_source}.decision: must be exactly 'PASS'")
        if not isinstance(record.get("notes"), str):
            raise ValueError(f"{record_source}.notes: must be a string")
    return reviewer_id, normalized_id, records


def _provenance_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "review provenance artifacts must be inside the repository: "
            f"{resolved}"
        ) from exc


def merged_signoff(
    *,
    case_ids: Sequence[str],
    cases_sha256: str,
    review_packet_sha256: str,
    review_packet_path: Path,
    reviewer_a_id: str,
    reviewer_a_records: Sequence[Mapping[str, Any]],
    reviewer_a_sha256: str,
    reviewer_a_path: Path,
    reviewer_b_id: str,
    reviewer_b_records: Sequence[Mapping[str, Any]],
    reviewer_b_sha256: str,
    reviewer_b_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    case_signoffs: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids):
        reviewers: list[dict[str, Any]] = []
        for slot, reviewer_id, records in (
            ("A", reviewer_a_id, reviewer_a_records),
            ("B", reviewer_b_id, reviewer_b_records),
        ):
            source_record = records[index]
            reviewer = {
                "reviewer_slot": slot,
                "reviewer_id": reviewer_id,
                **{check: source_record[check] for check in REVIEW_CHECKS},
                "decision": source_record["decision"],
                "notes": source_record["notes"],
            }
            reviewers.append(reviewer)
        case_signoffs.append({"case_id": case_id, "reviewers": reviewers})

    return {
        "schema_version": SIGNOFF_SCHEMA_VERSION,
        "cases_sha256": cases_sha256,
        "review_packet_sha256": review_packet_sha256,
        "review_packet_path": _provenance_path(
            review_packet_path, repo_root=repo_root
        ),
        "review_response_sha256s": [
            {
                "reviewer_slot": "A",
                "reviewer_id": reviewer_a_id,
                "sha256": reviewer_a_sha256,
                "path": _provenance_path(reviewer_a_path, repo_root=repo_root),
            },
            {
                "reviewer_slot": "B",
                "reviewer_id": reviewer_b_id,
                "sha256": reviewer_b_sha256,
                "path": _provenance_path(reviewer_b_path, repo_root=repo_root),
            },
        ],
        "case_signoffs": case_signoffs,
    }


def merge_responses(
    *,
    cases_path: Path,
    review_packet_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    output_path: Path,
    repo_root: Path = REPO_ROOT,
) -> int:
    inputs = [
        ("cases", cases_path),
        ("review packet", review_packet_path),
        ("Reviewer A response", reviewer_a_path),
        ("Reviewer B response", reviewer_b_path),
    ]
    named_paths = [*inputs, ("merged sign-off output", output_path)]
    reject_symlink_inputs([path for _, path in inputs])
    _reject_aliases(named_paths)
    require_new_outputs([output_path])

    cases_payload = _read_regular_bytes(cases_path, label="cases")
    packet_payload = _read_regular_bytes(review_packet_path, label="review packet")
    reviewer_a_payload = _read_regular_bytes(
        reviewer_a_path, label="Reviewer A response"
    )
    reviewer_b_payload = _read_regular_bytes(
        reviewer_b_path, label="Reviewer B response"
    )
    if not packet_payload:
        raise ValueError("review packet must not be empty")
    case_ids = load_case_ids(cases_payload, source=str(cases_path))
    cases_sha256 = sha256_bytes(cases_payload)
    packet_sha256 = sha256_bytes(packet_payload)

    reviewer_a_id, normalized_a, reviewer_a_records = validate_completed_response(
        reviewer_a_payload,
        source=str(reviewer_a_path),
        expected_slot="A",
        expected_case_ids=case_ids,
        expected_cases_sha256=cases_sha256,
        expected_packet_sha256=packet_sha256,
    )
    reviewer_b_id, normalized_b, reviewer_b_records = validate_completed_response(
        reviewer_b_payload,
        source=str(reviewer_b_path),
        expected_slot="B",
        expected_case_ids=case_ids,
        expected_cases_sha256=cases_sha256,
        expected_packet_sha256=packet_sha256,
    )
    if normalized_a == normalized_b:
        raise ValueError(
            "Reviewer A and Reviewer B must have distinct reviewer IDs after "
            "Unicode normalization"
        )

    result = merged_signoff(
        case_ids=case_ids,
        cases_sha256=cases_sha256,
        review_packet_sha256=packet_sha256,
        review_packet_path=review_packet_path,
        reviewer_a_id=reviewer_a_id,
        reviewer_a_records=reviewer_a_records,
        reviewer_a_sha256=sha256_bytes(reviewer_a_payload),
        reviewer_a_path=reviewer_a_path,
        reviewer_b_id=reviewer_b_id,
        reviewer_b_records=reviewer_b_records,
        reviewer_b_sha256=sha256_bytes(reviewer_b_payload),
        reviewer_b_path=reviewer_b_path,
        repo_root=repo_root,
    )
    rendered = _render_json(result)
    _assert_unchanged(
        [
            ("cases", cases_path, cases_payload),
            ("review packet", review_packet_path, packet_payload),
            ("Reviewer A response", reviewer_a_path, reviewer_a_payload),
            ("Reviewer B response", reviewer_b_path, reviewer_b_payload),
        ]
    )
    _reject_aliases(named_paths)
    publish_immutable_texts({output_path: rendered}, authoritative_path=output_path)
    return len(case_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--create-templates",
        dest="mode",
        action="store_const",
        const="create",
        help="immutably create separate Reviewer A and B response templates",
    )
    modes.add_argument(
        "--merge",
        dest="mode",
        action="store_const",
        const="merge",
        help="validate two completed responses and immutably merge a sign-off",
    )
    modes.add_argument(
        "--check-templates",
        dest="mode",
        action="store_const",
        const="check",
        help="read only; fail if either response template is missing or stale",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--review-packet", type=Path, default=DEFAULT_REVIEW_PACKET
    )
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="merged sign-off JSON; required only with --merge",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "merge" and args.output is None:
        parser.error("--merge requires --output")
    if args.mode != "merge" and args.output is not None:
        parser.error("--output is valid only with --merge")

    try:
        if args.mode == "create":
            count = create_templates(
                cases_path=args.cases,
                review_packet_path=args.review_packet,
                reviewer_a_path=args.review_a,
                reviewer_b_path=args.review_b,
            )
            print(
                f"wrote Reviewer A/B templates for {count} cases: "
                f"{args.review_a}, {args.review_b}"
            )
            return 0
        if args.mode == "check":
            try:
                count = check_templates(
                    cases_path=args.cases,
                    review_packet_path=args.review_packet,
                    reviewer_a_path=args.review_a,
                    reviewer_b_path=args.review_b,
                )
            except (OSError, ValueError) as exc:
                print(f"stale: {exc}", file=sys.stderr)
                return 1
            print(f"ok: both review templates are current for {count} cases")
            return 0

        assert args.output is not None
        count = merge_responses(
            cases_path=args.cases,
            review_packet_path=args.review_packet,
            reviewer_a_path=args.review_a,
            reviewer_b_path=args.review_b,
            output_path=args.output,
            repo_root=REPO_ROOT,
        )
        print(f"wrote {args.output} ({count} cases, 2 independent reviewers)")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
