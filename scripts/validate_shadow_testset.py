#!/usr/bin/env python3
"""Validate Shadow60 composition, evidence integrity, and DEV source isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from holdout_gold import (  # noqa: E402
    canonical_source_url,
    normalize_evidence_text,
    normalize_title_without_year,
)


DEFAULT_CASES = ROOT / "config" / "pnu-service-shadow60-v1.jsonl"
DEFAULT_INDEX = (
    ROOT
    / "processed"
    / "index"
    / "pnu-20260725-curated-cascade-v5-allow-suspect.sqlite"
)
DEFAULT_MANIFEST = (
    ROOT
    / "processed"
    / "curation"
    / "20260725-pnu-curated-v5"
    / "curated-manifest.jsonl"
)
DEFAULT_DEV_MANIFEST = ROOT / "config" / "pnu-service-dev-source-manifest.json"
EXPECTED_BUCKETS = {
    "simple": 24,
    "multi": 18,
    "role_variant": 8,
    "challenge": 10,
}
EXPECTED_CHALLENGES = {
    "unanswerable": 4,
    "scope_version_ambiguity": 3,
    "prompt_injection": 3,
}
ALLOWED_ROLES = {"pnu-student", "pnu-staff", "pnu-researcher"}
CRAWL_PREFIX = "downloads/pnu-web-crawl/"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _load_documents(index_path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT document_id, source_path, source_title, source_url,
                   category, group_concat(text, char(10)) AS text
            FROM chunks
            GROUP BY document_id, source_path, source_title, source_url, category
            """
        ).fetchall()
    finally:
        connection.close()
    return {str(row["document_id"]): dict(row) for row in rows}


def _load_source_shas(manifest_path: Path) -> dict[str, str]:
    return {
        str(row["crawl_storage_path"]): str(row["sha256"])
        for row in _read_jsonl(manifest_path)
        if row.get("crawl_storage_path") and row.get("sha256")
    }


def _dev_fingerprints(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    return {
        "document_id": {str(row.get("document_id") or "") for row in documents},
        "source_sha256": {str(row.get("source_sha256") or "") for row in documents},
        "normalized_title_without_year": {
            str(row.get("normalized_title_without_year") or "") for row in documents
        },
        "source_url": {
            canonical_source_url(row.get("source_url")) for row in documents
        },
    }


def _check_equal_list(
    errors: list[str], case_id: str, field: str, actual: list[str], expected: list[str]
) -> None:
    if actual != expected:
        errors.append(f"{case_id}.{field}: expected {expected!r}, got {actual!r}")


def collect_errors(
    cases: list[dict[str, Any]],
    *,
    documents: dict[str, dict[str, Any]],
    source_shas: dict[str, str],
    dev: dict[str, set[str]],
) -> list[str]:
    errors: list[str] = []
    ids = [str(case.get("id") or "") for case in cases]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if len(cases) != 60:
        errors.append(f"expected 60 cases, got {len(cases)}")
    if duplicates:
        errors.append(f"duplicate case ids: {duplicates}")

    bucket_counts = Counter(str(case.get("shadow_bucket") or "") for case in cases)
    if dict(bucket_counts) != EXPECTED_BUCKETS:
        errors.append(f"bucket counts: expected {EXPECTED_BUCKETS}, got {dict(bucket_counts)}")
    challenge_counts = Counter(
        str(case.get("challenge_type") or "")
        for case in cases
        if case.get("shadow_bucket") == "challenge"
    )
    if dict(challenge_counts) != EXPECTED_CHALLENGES:
        errors.append(
            f"challenge counts: expected {EXPECTED_CHALLENGES}, got {dict(challenge_counts)}"
        )

    by_id = {str(case.get("id") or ""): case for case in cases}
    core_sources: list[str] = []
    for case in cases:
        case_id = str(case.get("id") or "<missing-id>")
        bucket = str(case.get("shadow_bucket") or "")
        query = str(case.get("query") or "").strip()
        if not query:
            errors.append(f"{case_id}.query: missing")
        if bucket in {"simple", "multi"}:
            doc_ids = {
                str(option.get("document_id") or "")
                for claim in case.get("required_claims", [])
                for option in claim.get("evidence_options", [])
            }
            if len(doc_ids) != 1:
                errors.append(f"{case_id}: core case must use exactly one source document")
            core_sources.extend(doc_ids)
            expected_claims = 1 if bucket == "simple" else 2
            if len(case.get("required_claims", [])) != expected_claims:
                errors.append(
                    f"{case_id}: {bucket} case must contain {expected_claims} required claims"
                )
        if bucket == "role_variant":
            base = by_id.get(str(case.get("variant_of") or ""))
            if base is None or base.get("shadow_bucket") not in {"simple", "multi"}:
                errors.append(f"{case_id}: invalid role variant parent")
            elif case.get("family_id") != base.get("family_id"):
                errors.append(f"{case_id}: role variant must share parent family_id")
            if case.get("role") not in ALLOWED_ROLES:
                errors.append(f"{case_id}: unsupported role {case.get('role')!r}")
        if bucket == "challenge":
            challenge_type = case.get("challenge_type")
            if challenge_type == "prompt_injection":
                if case.get("answerable") is not True or not case.get("required_claims"):
                    errors.append(f"{case_id}: prompt injection must retain answerable gold")
            elif case.get("answerable") is not False or case.get("required_claims"):
                errors.append(f"{case_id}: non-injection challenge must be unanswerable")

        option_sources: list[dict[str, str]] = []
        for claim_index, claim in enumerate(case.get("required_claims", []), start=1):
            description = str(claim.get("description") or "").strip()
            critical_values = [str(value) for value in claim.get("critical_values", [])]
            options = claim.get("evidence_options", [])
            if not description or not critical_values or not options:
                errors.append(
                    f"{case_id}.required_claims[{claim_index}]: incomplete atomic claim"
                )
                continue
            for option_index, option in enumerate(options, start=1):
                document_id = str(option.get("document_id") or "")
                document = documents.get(document_id)
                label = f"{case_id}.claim[{claim_index}].option[{option_index}]"
                if document is None:
                    errors.append(f"{label}: unknown document_id {document_id!r}")
                    continue
                source_path = str(document["source_path"])
                manifest_key = (
                    source_path[len(CRAWL_PREFIX) :]
                    if source_path.startswith(CRAWL_PREFIX)
                    else ""
                )
                expected_sha = source_shas.get(manifest_key, "")
                expected = {
                    "document_id": document_id,
                    "source_sha256": expected_sha,
                    "source_path": source_path,
                    "source_url": canonical_source_url(document.get("source_url")),
                    "source_title": str(document["source_title"]),
                    "normalized_title_without_year": normalize_title_without_year(
                        document["source_title"]
                    ),
                }
                for field, expected_value in expected.items():
                    if str(option.get(field) or "") != expected_value:
                        errors.append(
                            f"{label}.{field}: expected {expected_value!r}, "
                            f"got {option.get(field)!r}"
                        )
                quote = str(option.get("quote") or "")
                normalized_quote = normalize_evidence_text(quote)
                normalized_document = normalize_evidence_text(document["text"])
                if not normalized_quote or normalized_quote not in normalized_document:
                    errors.append(f"{label}.quote: not found in indexed document")
                for value in critical_values:
                    if normalize_evidence_text(value) not in normalized_quote:
                        errors.append(
                            f"{label}: critical value {value!r} absent from evidence quote"
                        )
                option_sources.append(expected)

                for fingerprint, option_value in (
                    ("document_id", document_id),
                    ("source_sha256", expected_sha),
                    (
                        "normalized_title_without_year",
                        expected["normalized_title_without_year"],
                    ),
                    ("source_url", expected["source_url"]),
                ):
                    if option_value and option_value in dev[fingerprint]:
                        errors.append(
                            f"{label}: overlaps DEV by {fingerprint}={option_value!r}"
                        )

        unique_sources: dict[str, dict[str, str]] = {}
        for source in option_sources:
            unique_sources[source["document_id"]] = source
        ordered_sources = list(unique_sources.values())
        _check_equal_list(
            errors,
            case_id,
            "source_sha256s",
            list(case.get("source_sha256s", [])),
            [source["source_sha256"] for source in ordered_sources],
        )
        _check_equal_list(
            errors,
            case_id,
            "source_paths",
            list(case.get("source_paths", [])),
            [source["source_path"] for source in ordered_sources],
        )
        _check_equal_list(
            errors,
            case_id,
            "normalized_source_titles",
            list(case.get("normalized_source_titles", [])),
            [source["normalized_title_without_year"] for source in ordered_sources],
        )

    duplicate_core_sources = sorted(
        source for source, count in Counter(core_sources).items() if source and count > 1
    )
    if len(set(core_sources)) != 42:
        errors.append(
            f"expected 42 unique core source documents, got {len(set(core_sources))}; "
            f"duplicates={duplicate_core_sources}"
        )
    return errors


def validate_paths(
    cases_path: Path,
    *,
    index_path: Path,
    source_manifest_path: Path,
    dev_manifest_path: Path,
) -> dict[str, Any]:
    cases = _read_jsonl(cases_path)
    errors = collect_errors(
        cases,
        documents=_load_documents(index_path),
        source_shas=_load_source_shas(source_manifest_path),
        dev=_dev_fingerprints(dev_manifest_path),
    )
    if errors:
        raise ValueError("shadow testset validation failed:\n- " + "\n- ".join(errors))
    return {
        "cases_path": str(cases_path),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "bucket_counts": dict(
            Counter(str(case["shadow_bucket"]) for case in cases)
        ),
        "challenge_counts": dict(
            Counter(
                str(case["challenge_type"])
                for case in cases
                if case["shadow_bucket"] == "challenge"
            )
        ),
        "unique_core_source_documents": len(
            {
                option["document_id"]
                for case in cases
                if case["shadow_bucket"] in {"simple", "multi"}
                for claim in case["required_claims"]
                for option in claim["evidence_options"]
            }
        ),
        "dev_source_overlap_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dev-source-manifest", type=Path, default=DEFAULT_DEV_MANIFEST)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_paths(
        args.cases,
        index_path=args.index,
        source_manifest_path=args.source_manifest,
        dev_manifest_path=args.dev_source_manifest,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
        report["report_path"] = str(args.report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
