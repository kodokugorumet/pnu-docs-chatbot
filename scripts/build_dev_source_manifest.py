#!/usr/bin/env python3
"""Resolve DEV gold chunks to immutable source identities.

The resulting manifest is an input to ``validate_service_holdout.py``.  It is
derived from the frozen DEV cases, the canonical SQLite corpus, and the raw
download bytes; no retrieval output or model answer is consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from holdout_gold import (  # noqa: E402
    canonical_source_url,
    load_jsonl,
    normalize_title_without_year,
    sha256_file,
)


DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-eval.jsonl"
DEFAULT_INDEX = (
    REPO_ROOT
    / "processed"
    / "index"
    / "pnu-20260725-curated-cascade-v5-allow-suspect.sqlite"
)
DEFAULT_OUT = REPO_ROOT / "config" / "pnu-service-dev-source-manifest.json"
SCHEMA_VERSION = "pnu.service-dev-source-manifest.v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_path(value: Any, *, repo_root: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("corpus row is missing source_path")
    path = Path(raw)
    resolved = (path if path.is_absolute() else repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"source_path escapes repository: {raw!r}") from exc
    if not resolved.is_file():
        raise ValueError(f"raw source file does not exist: {raw!r}")
    return resolved


def _family_id(normalized_title: str) -> str:
    fingerprint = sha256_bytes(normalized_title.encode("utf-8"))[:20]
    return f"dev-title-family-{fingerprint}"


def build_manifest(
    cases: Sequence[Mapping[str, Any]],
    *,
    index_path: Path,
    repo_root: Path = REPO_ROOT,
    cases_sha256: str | None = None,
) -> dict[str, Any]:
    evidence_owners: dict[str, set[str]] = defaultdict(set)
    for case_index, case in enumerate(cases):
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"cases[{case_index}] is missing id")
        evidence = case.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"case {case_id!r} has no evidence")
        for evidence_index, item in enumerate(evidence):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"case {case_id!r} evidence[{evidence_index}] is not an object"
                )
            chunk_id = str(item.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(
                    f"case {case_id!r} evidence[{evidence_index}] is missing chunk_id"
                )
            evidence_owners[chunk_id].add(case_id)

    documents: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    try:
        corpus_revisions: set[str] = set()
        for chunk_id in sorted(evidence_owners):
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, source_path, source_title,
                       source_url, corpus_revision
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(
                    f"gold chunk {chunk_id!r} resolved to {len(rows)} corpus rows"
                )
            row = rows[0]
            document_id = str(row["document_id"] or "").strip()
            title = str(row["source_title"] or "").strip()
            normalized_title = normalize_title_without_year(title)
            source_url = canonical_source_url(row["source_url"])
            if not document_id or not title or not normalized_title or not source_url:
                raise ValueError(
                    f"gold chunk {chunk_id!r} has incomplete source identity"
                )
            raw_path = _source_path(row["source_path"], repo_root=repo_root)
            relative_path = raw_path.relative_to(repo_root.resolve()).as_posix()
            corpus_revision = str(row["corpus_revision"] or "").strip()
            if corpus_revision:
                corpus_revisions.add(corpus_revision)
            identity = {
                "document_id": document_id,
                "source_document_family_id": _family_id(normalized_title),
                "source_sha256": sha256_file(raw_path),
                "source_path": relative_path,
                "source_title": title,
                "normalized_title_without_year": normalized_title,
                "source_url": source_url,
            }
            existing = documents.get(document_id)
            if existing is None:
                documents[document_id] = {
                    **identity,
                    "case_ids": sorted(evidence_owners[chunk_id]),
                    "evidence_chunk_ids": [chunk_id],
                }
            else:
                for key, value in identity.items():
                    if existing.get(key) != value:
                        raise ValueError(
                            f"document {document_id!r} has inconsistent {key}"
                        )
                existing["case_ids"] = sorted(
                    set(existing["case_ids"]) | evidence_owners[chunk_id]
                )
                existing["evidence_chunk_ids"].append(chunk_id)
    finally:
        connection.close()

    return {
        "schema_version": SCHEMA_VERSION,
        "cases_sha256": cases_sha256,
        "corpus_index_path": index_path.resolve().relative_to(
            repo_root.resolve()
        ).as_posix(),
        "corpus_index_sha256": sha256_file(index_path),
        "corpus_revisions": sorted(corpus_revisions),
        "document_count": len(documents),
        "documents": [documents[key] for key in sorted(documents)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cases_bytes = args.cases.read_bytes()
    manifest = build_manifest(
        load_jsonl(args.cases),
        index_path=args.index,
        cases_sha256=sha256_bytes(cases_bytes),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "documents": manifest["document_count"],
                "cases_sha256": manifest["cases_sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
