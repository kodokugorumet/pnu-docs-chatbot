#!/usr/bin/env python3
"""Evaluate atomic-anchor preservation for the three parser profiles."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from scripts.parser_audit import (
        DEFAULT_ARTIFACTS,
        PROFILES,
        AuditValidationError,
        ProfileArtifact,
        match_anchor_in_chunks,
        read_jsonl,
        selected_documents,
        selected_index_chunks,
        selected_run_chunks,
        sha256_file,
        validate_canonical_bindings,
        validate_dataset_schema,
        validate_profile_artifact,
    )
else:
    from parser_audit import (
        DEFAULT_ARTIFACTS,
        PROFILES,
        AuditValidationError,
        ProfileArtifact,
        match_anchor_in_chunks,
        read_jsonl,
        selected_documents,
        selected_index_chunks,
        selected_run_chunks,
        sha256_file,
        validate_canonical_bindings,
        validate_dataset_schema,
        validate_profile_artifact,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_artifacts() -> list[ProfileArtifact]:
    return [
        ProfileArtifact(
            profile,
            REPO_ROOT / DEFAULT_ARTIFACTS[profile][0],
            REPO_ROOT / DEFAULT_ARTIFACTS[profile][1],
        )
        for profile in PROFILES
    ]


def parse_artifact(value: str) -> ProfileArtifact:
    profile, separator, remainder = value.partition("=")
    run, inner_separator, index = remainder.partition("::")
    if (
        not separator
        or not inner_separator
        or profile not in PROFILES
        or not run
        or not index
    ):
        raise argparse.ArgumentTypeError(
            "artifact must be PROFILE=RUN_DIR::INDEX.sqlite"
        )
    return ProfileArtifact(profile, Path(run), Path(index))


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def evaluate_profile(
    records: Sequence[Mapping[str, Any]], artifact: ProfileArtifact
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = validate_profile_artifact(artifact)
    document_ids = [str(record["source"]["document_id"]) for record in records]
    selected = set(document_ids)
    documents = selected_documents(artifact.run_dir / "documents.jsonl", selected)
    run_chunks = selected_run_chunks(artifact.run_dir / "chunks.jsonl", selected)
    index_chunks = selected_index_chunks(artifact.index_path, document_ids)

    rows: list[dict[str, Any]] = []
    group_totals: Counter[str] = Counter()
    group_hits: Counter[str] = Counter()
    document_all: dict[str, bool] = {}
    documents_available = 0
    source_identity_ok = 0

    for record in records:
        case_id = str(record["id"])
        group = str(record["group"])
        source = record["source"]
        document_id = str(source["document_id"])
        document = documents.get(document_id)
        document_available = document is not None and bool(run_chunks.get(document_id))
        if document_available:
            documents_available += 1
        run_identity_ok = bool(document) and all(
            document.get(field) == source.get(field)
            for field in (
                "source_path",
                "relative_path",
                "source_sha256",
                "extension",
            )
        )
        indexed = index_chunks.get(document_id, [])
        index_identity_ok = bool(indexed) and all(
            chunk.get(field) == source.get(field)
            for chunk in indexed
            for field in (
                "source_path",
                "relative_path",
                "extension",
                "source_title",
            )
        )
        identity_ok = bool(run_identity_ok and index_identity_ok)
        if identity_ok:
            source_identity_ok += 1
        all_hit = True
        for anchor in record["anchors"]:
            run_match = match_anchor_in_chunks(run_chunks.get(document_id, []), anchor)
            index_match = match_anchor_in_chunks(index_chunks.get(document_id, []), anchor)
            end_to_end = bool(identity_ok and run_match and index_match)
            all_hit = all_hit and end_to_end
            group_totals[group] += 1
            if end_to_end:
                group_hits[group] += 1
            rows.append(
                {
                    "profile": artifact.profile,
                    "case_id": case_id,
                    "group": group,
                    "document_id": document_id,
                    "anchor_id": anchor["id"],
                    "anchor_kind": anchor["kind"],
                    "document_available": document_available,
                    "run_source_identity_ok": run_identity_ok,
                    "index_source_identity_ok": index_identity_ok,
                    "source_identity_ok": identity_ok,
                    "run_hit": bool(run_match),
                    "run_chunk_id": run_match.get("chunk_id") if run_match else None,
                    "index_hit": bool(index_match),
                    "index_chunk_id": index_match.get("chunk_id") if index_match else None,
                    "end_to_end_hit": end_to_end,
                }
            )
        document_all[case_id] = all_hit

    anchor_count = len(rows)
    run_hits = sum(1 for row in rows if row["run_hit"])
    index_hits = sum(1 for row in rows if row["index_hit"])
    end_to_end_hits = sum(1 for row in rows if row["end_to_end_hit"])
    all_document_hits = sum(document_all.values())
    group_summary = {
        group: {
            "anchors": group_totals[group],
            "hits": group_hits[group],
            "anchor_recall": _ratio(group_hits[group], group_totals[group]),
            "documents_all_anchors": sum(
                1
                for record in records
                if record["group"] == group and document_all[str(record["id"])]
            ),
        }
        for group in sorted(group_totals)
    }
    summary = {
        **identity,
        "documents": len(records),
        "documents_available": documents_available,
        "source_identity_documents": source_identity_ok,
        "anchors": anchor_count,
        "run_anchor_hits": run_hits,
        "run_anchor_recall": _ratio(run_hits, anchor_count),
        "index_anchor_hits": index_hits,
        "index_anchor_recall": _ratio(index_hits, anchor_count),
        "end_to_end_anchor_hits": end_to_end_hits,
        "end_to_end_anchor_recall": _ratio(end_to_end_hits, anchor_count),
        "documents_all_anchors": all_document_hits,
        "all_anchor_document_rate": _ratio(all_document_hits, len(records)),
        "groups": group_summary,
    }
    return summary, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "config/pnu-parser-audit-v1.jsonl",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        type=parse_artifact,
        help="PROFILE=RUN_DIR::INDEX.sqlite; repeat once per profile",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = read_jsonl(args.cases)
        schema = validate_dataset_schema(records, REPO_ROOT, enforce_balance=True)
        artifacts = args.artifact or default_artifacts()
        by_profile = {artifact.profile: artifact for artifact in artifacts}
        if set(by_profile) != set(PROFILES) or len(artifacts) != len(PROFILES):
            raise AuditValidationError(
                f"exactly one artifact is required for each profile: {PROFILES}"
            )
        validate_canonical_bindings(records, by_profile["cascade"])

        summaries: list[dict[str, Any]] = []
        detail_rows: list[dict[str, Any]] = []
        source_manifests: set[str] = set()
        for profile in PROFILES:
            summary, rows = evaluate_profile(records, by_profile[profile])
            summaries.append(summary)
            detail_rows.extend(rows)
            source_manifests.add(str(summary["source_manifest_sha256"]))
        if len(source_manifests) != 1:
            raise AuditValidationError(
                "profile artifacts do not share one source manifest SHA-256"
            )
        result = {
            "ok": True,
            "schema_version": 1,
            "metric_version": "pnu.parser-atomic-anchor-v1",
            "cases_path": str(args.cases),
            "cases_sha256": sha256_file(args.cases),
            "dataset": schema,
            "common_source_manifest_sha256": next(iter(source_manifests)),
            "profiles": summaries,
            "details": detail_rows,
            "interpretation": {
                "headline_eligible": False,
                "scope": "parser-profile preservation audit",
                "excludes": [
                    "retrieval ranking quality",
                    "answer generation quality",
                    "LLM judge scores",
                ],
                "selection_note": (
                    "The audit is source-bound and format-balanced, but its 18 "
                    "documents are a targeted diagnostic sample rather than a "
                    "random estimate of the full corpus."
                ),
            },
        }
    except (AuditValidationError, OSError, sqlite3.Error) as exc:
        result = {"ok": False, "error": str(exc)}
        detail_rows = []

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    if args.csv_out and result["ok"]:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(detail_rows[0]) if detail_rows else []
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)
    print(rendered, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
