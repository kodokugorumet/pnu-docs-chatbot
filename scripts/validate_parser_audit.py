#!/usr/bin/env python3
"""Validate the source-bound 18-document parser audit set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

if __package__:
    from scripts.parser_audit import (
        DEFAULT_ARTIFACTS,
        AuditValidationError,
        ProfileArtifact,
        read_jsonl,
        sha256_file,
        validate_canonical_bindings,
        validate_dataset_schema,
    )
else:
    from parser_audit import (
        DEFAULT_ARTIFACTS,
        AuditValidationError,
        ProfileArtifact,
        read_jsonl,
        sha256_file,
        validate_canonical_bindings,
        validate_dataset_schema,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "config/pnu-parser-audit-v1.jsonl",
    )
    parser.add_argument(
        "--canonical-run",
        type=Path,
        default=REPO_ROOT / DEFAULT_ARTIFACTS["cascade"][0],
    )
    parser.add_argument(
        "--canonical-index",
        type=Path,
        default=REPO_ROOT / DEFAULT_ARTIFACTS["cascade"][1],
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = read_jsonl(args.cases)
        schema = validate_dataset_schema(records, REPO_ROOT, enforce_balance=True)
        bindings = validate_canonical_bindings(
            records,
            ProfileArtifact("cascade", args.canonical_run, args.canonical_index),
        )
        result = {
            "ok": True,
            "schema_version": 1,
            "cases_path": str(args.cases),
            "cases_sha256": sha256_file(args.cases),
            "schema_gate": {"ok": True, **schema},
            "raw_source_gate": {
                "ok": True,
                "documents_sha256_verified": schema["documents"],
            },
            "canonical_binding_gate": {"ok": True, **bindings},
            "scope": "parser-profile audit only; not a RAG retrieval/generation metric",
        }
    except (AuditValidationError, OSError) as exc:
        result = {"ok": False, "error": str(exc)}

    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
