#!/usr/bin/env python3
"""Evaluate one or more PNU BM25 indexes with auditable metadata gold.

The benchmark deliberately scores document retrieval, not generated answers.
Every scored case identifies an expected source using metadata that is carried
from the crawl manifest into the BM25 index.  Cases whose gold has not yet been
checked can remain in the suite with ``requires_validation: true``; they are
reported explicitly and excluded from Hit@k and MRR.

Examples:
  python scripts/evaluate_pnu_retrieval.py validate
  python scripts/evaluate_pnu_retrieval.py evaluate \
    --index baseline=processed/index/pnu-baseline.sqlite \
    --index challenger=processed/index/pnu-challenger.sqlite \
    --json-output processed/eval/pnu-comparison.json \
    --csv-output processed/eval/pnu-comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import unicodedata
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .bm25_search import search_index
except ImportError:  # Direct CLI execution.
    from bm25_search import search_index


DEFAULT_CASES = Path("config/pnu-retrieval-eval.jsonl")
DEFAULT_INSTITUTION = "부산대학교"
SCHEMA_VERSION = 1
# Search is chunk-ranked, while this benchmark is document-ranked.  Fetch a
# deliberately broad chunk pool so one long document cannot occupy every slot
# before document-level deduplication.
DOCUMENT_CHUNK_POOL_MULTIPLIER = 50
MIN_DOCUMENT_CHUNK_POOL = 250
FALLBACK_DOCUMENT_CHUNK_CAP = 10_000
SOURCE_MANIFEST_META_KEY = "source_manifest_sha256"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
CASE_FIELDS = frozenset(
    {
        "id",
        "query",
        "expected",
        "k",
        "requires_validation",
        "institution",
        "note",
    }
)
EXPECTED_FIELDS = frozenset(
    {
        "source_host",
        "category",
        "source_title_contains",
        "source_path",
        "relative_path",
        "document_id",
    }
)
EXACT_TEXT_FIELDS = frozenset(
    {
        "source_host",
        "category",
        "source_path",
        "relative_path",
        "document_id",
    }
)
CSV_FIELDS = (
    "index_name",
    "index_path",
    "status",
    "provenance_status",
    "source_manifest_sha256",
    "provenance_compatible",
    "provenance_override_used",
    "total_cases",
    "scored_cases",
    "unevaluable_cases",
    "hits",
    "misses",
    "hit_at_k",
    "mrr",
    "error",
)

SearchFunction = Callable[..., List[Dict[str, Any]]]


class EvaluationError(ValueError):
    """Raised for an invalid case file or evaluation request."""


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError("{} must be a non-empty string".format(label))
    return value.strip()


def _normalized_text(value: Any) -> str:
    rendered = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(rendered.casefold().split())


def _normalize_expected(value: Any, label: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise EvaluationError("{} must be an object".format(label))
    unknown = sorted(set(value) - EXPECTED_FIELDS)
    if unknown:
        raise EvaluationError(
            "{} has unsupported field(s): {}".format(label, ", ".join(unknown))
        )
    if not value:
        raise EvaluationError("{} must contain at least one criterion".format(label))
    return {
        field: _nonempty_string(raw_value, "{}.{}".format(label, field))
        for field, raw_value in value.items()
    }


def validate_cases(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Validate and normalize benchmark rows while preserving their order."""

    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for row_number, raw_row in enumerate(rows, start=1):
        label = "case {}".format(row_number)
        if not isinstance(raw_row, dict):
            raise EvaluationError("{} must be a JSON object".format(label))
        unknown = sorted(set(raw_row) - CASE_FIELDS)
        if unknown:
            raise EvaluationError(
                "{} has unsupported field(s): {}".format(label, ", ".join(unknown))
            )
        missing = sorted({"id", "query", "expected", "k"} - set(raw_row))
        if missing:
            raise EvaluationError(
                "{} is missing field(s): {}".format(label, ", ".join(missing))
            )

        case_id = _nonempty_string(raw_row.get("id"), "{}.id".format(label))
        if not CASE_ID_RE.fullmatch(case_id):
            raise EvaluationError(
                "{}.id must match {}".format(label, CASE_ID_RE.pattern)
            )
        if case_id in seen_ids:
            raise EvaluationError("duplicate case id: {}".format(case_id))
        seen_ids.add(case_id)

        query = _nonempty_string(raw_row.get("query"), "{}.query".format(label))
        k = raw_row.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
            raise EvaluationError("{}.k must be an integer from 1 to 100".format(label))
        requires_validation = raw_row.get("requires_validation", False)
        if not isinstance(requires_validation, bool):
            raise EvaluationError(
                "{}.requires_validation must be a boolean".format(label)
            )

        case: Dict[str, Any] = {
            "id": case_id,
            "query": query,
            "expected": _normalize_expected(
                raw_row.get("expected"),
                "{}.expected".format(label),
            ),
            "k": k,
            "requires_validation": requires_validation,
        }
        if "institution" in raw_row:
            case["institution"] = _nonempty_string(
                raw_row.get("institution"),
                "{}.institution".format(label),
            )
        if "note" in raw_row:
            case["note"] = _nonempty_string(
                raw_row.get("note"),
                "{}.note".format(label),
            )
        normalized.append(case)

    if not normalized:
        raise EvaluationError("case file is empty")
    if expected_count is not None:
        if expected_count < 1:
            raise EvaluationError("expected_count must be positive")
        if len(normalized) != expected_count:
            raise EvaluationError(
                "expected {} cases, found {}".format(expected_count, len(normalized))
            )
    return normalized


def load_cases(
    path: Path,
    *,
    expected_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load UTF-8 JSONL cases and return their validated representation."""

    path = Path(path)
    if not path.is_file():
        raise EvaluationError("missing case file: {}".format(path))
    rows: List[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationError(
                        "invalid JSON at {}:{}: {}".format(path, line_number, exc)
                    ) from exc
                if not isinstance(value, dict):
                    raise EvaluationError(
                        "JSONL row at {}:{} must be an object".format(
                            path,
                            line_number,
                        )
                    )
                rows.append(value)
    except UnicodeError as exc:
        raise EvaluationError("{} must be UTF-8 JSONL".format(path)) from exc
    except OSError as exc:
        raise EvaluationError("cannot read {}: {}".format(path, exc)) from exc
    return validate_cases(rows, expected_count=expected_count)


def _result_value(result: Mapping[str, Any], field: str) -> Any:
    if field == "document_id":
        return result.get("document_id") or result.get("doc_id")
    value = result.get(field)
    if value not in (None, ""):
        return value
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(field)
    return None


def result_matches_expected(
    result: Mapping[str, Any],
    expected: Mapping[str, str],
) -> bool:
    """Return true only when every declared metadata criterion matches."""

    for field, expected_value in expected.items():
        actual = _result_value(result, field)
        if field == "source_title_contains":
            title = _result_value(result, "source_title")
            if _normalized_text(expected_value) not in _normalized_text(title):
                return False
        elif field in {"source_host", "category"}:
            if _normalized_text(actual) != _normalized_text(expected_value):
                return False
        elif field in EXACT_TEXT_FIELDS:
            if str(actual or "") != expected_value:
                return False
        else:  # Defensive guard for callers that bypass schema validation.
            return False
    return True


def _matched_result_summary(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": _result_value(row, "document_id"),
        "chunk_id": row.get("chunk_id"),
        "source_title": _result_value(row, "source_title"),
        "source_host": _result_value(row, "source_host"),
        "category": _result_value(row, "category"),
        "source_path": _result_value(row, "source_path"),
        "relative_path": _result_value(row, "relative_path"),
    }


def _document_identity(row: Mapping[str, Any], ordinal: int) -> str:
    """Return the most stable available identity for document deduplication."""

    document_id = _result_value(row, "document_id")
    if document_id not in (None, ""):
        return "document_id:{}".format(document_id)
    for field in ("relative_path", "source_path"):
        value = _result_value(row, field)
        if value not in (None, ""):
            return "{}:{}".format(field, value)
    chunk_id = row.get("chunk_id")
    if chunk_id not in (None, ""):
        return "chunk_id:{}".format(chunk_id)
    # Rows without any stable source identity must not collapse together.
    return "anonymous_result:{}".format(ordinal)


def _rank_unique_documents(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> List[Mapping[str, Any]]:
    """Preserve chunk rank, retain the first chunk per document, then truncate."""

    seen = set()
    documents: List[Mapping[str, Any]] = []
    for ordinal, row in enumerate(rows):
        identity = _document_identity(row, ordinal)
        if identity in seen:
            continue
        seen.add(identity)
        documents.append(row)
    return documents[:limit]


def _adaptive_document_candidates(
    index_path: Path,
    query: str,
    k: int,
    institution: Optional[str],
    *,
    searcher: SearchFunction,
    index_chunk_count: Optional[int],
    chunk_count_source: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Mapping[str, Any]], Dict[str, Any]]:
    """Expand the chunk pool until k documents or all known chunks are seen."""

    known_chunk_count = (
        index_chunk_count
        if isinstance(index_chunk_count, int) and index_chunk_count >= 0
        else None
    )
    candidate_cap = (
        max(1, known_chunk_count)
        if known_chunk_count is not None
        else FALLBACK_DOCUMENT_CHUNK_CAP
    )
    requested = min(
        max(k * DOCUMENT_CHUNK_POOL_MULTIPLIER, MIN_DOCUMENT_CHUNK_POOL),
        candidate_cap,
    )
    requested_limits: List[int] = []

    while True:
        requested_limits.append(requested)
        rows = searcher(index_path, query, requested, institution)
        documents = _rank_unique_documents(rows, k)
        short_page = len(rows) < requested
        known_index_exhausted = (
            known_chunk_count is not None and requested >= candidate_cap
        )
        exhausted = short_page or known_index_exhausted
        cap_reached = requested >= candidate_cap

        if len(documents) >= k:
            stop_reason = "enough_unique_documents"
            break
        if exhausted:
            stop_reason = (
                "index_chunk_count_exhausted"
                if known_index_exhausted
                else "search_results_exhausted"
            )
            break
        if cap_reached:
            stop_reason = "fallback_chunk_cap_reached"
            break
        requested = min(requested * 2, candidate_cap)

    audit = {
        "requested_limits": requested_limits,
        "attempts": len(requested_limits),
        "final_requested_limit": requested_limits[-1],
        "index_chunk_count": known_chunk_count,
        "chunk_count_source": chunk_count_source,
        "candidate_cap": candidate_cap,
        "exhausted": exhausted,
        "cap_reached": cap_reached,
        "stop_reason": stop_reason,
    }
    return rows, documents, audit


def evaluate_index(
    name: str,
    index_path: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    default_institution: Optional[str] = DEFAULT_INSTITUTION,
    searcher: SearchFunction = search_index,
    index_chunk_count: Optional[int] = None,
    chunk_count_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single index and return case detail plus aggregate metrics."""

    index_name = _nonempty_string(name, "index name")
    index_path = Path(index_path)
    if not index_path.is_file():
        raise EvaluationError("missing index DB: {}".format(index_path))
    validated = validate_cases(cases)
    if index_chunk_count is None:
        inspected = inspect_index_provenance(index_name, index_path)
        index_chunk_count = inspected.get("chunk_count")
        chunk_count_source = inspected.get("chunk_count_source")

    details: List[Dict[str, Any]] = []
    hits = 0
    reciprocal_rank_sum = 0.0
    scored = 0
    k_groups: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "hits": 0}
    )

    for case in validated:
        detail: Dict[str, Any] = {
            "id": case["id"],
            "query": case["query"],
            "k": case["k"],
            "expected": dict(case["expected"]),
        }
        if case["requires_validation"]:
            detail.update(
                {
                    "status": "unevaluable",
                    "reason": "requires_validation",
                    "rank": None,
                    "reciprocal_rank": None,
                    "returned_count": None,
                    "returned_chunk_count": None,
                    "returned_document_count": None,
                    "candidate_pool": None,
                }
            )
            details.append(detail)
            continue

        institution = case.get("institution", default_institution)
        rows, documents, candidate_pool = _adaptive_document_candidates(
            index_path,
            case["query"],
            case["k"],
            institution,
            searcher=searcher,
            index_chunk_count=index_chunk_count,
            chunk_count_source=chunk_count_source,
        )
        matching_rank: Optional[int] = None
        matching_row: Optional[Mapping[str, Any]] = None
        for rank, row in enumerate(documents, start=1):
            if result_matches_expected(row, case["expected"]):
                matching_rank = rank
                matching_row = row
                break

        incomplete_candidates = (
            matching_rank is None
            and len(documents) < case["k"]
            and not candidate_pool["exhausted"]
        )
        if incomplete_candidates:
            detail.update(
                {
                    "status": "unevaluable",
                    "reason": "document_candidate_pool_not_exhausted",
                    "rank": None,
                    "reciprocal_rank": None,
                    "returned_count": len(rows),
                    "returned_chunk_count": len(rows),
                    "returned_document_count": len(documents),
                    "candidate_pool": candidate_pool,
                }
            )
            details.append(detail)
            continue

        scored += 1
        k_groups[case["k"]]["cases"] += 1
        if matching_rank is None:
            detail.update(
                {
                    "status": "miss",
                    "rank": None,
                    "reciprocal_rank": 0.0,
                    # Backward-compatible alias for the chunk result count.
                    "returned_count": len(rows),
                    "returned_chunk_count": len(rows),
                    "returned_document_count": len(documents),
                    "candidate_pool": candidate_pool,
                }
            )
        else:
            reciprocal_rank = 1.0 / matching_rank
            hits += 1
            reciprocal_rank_sum += reciprocal_rank
            k_groups[case["k"]]["hits"] += 1
            detail.update(
                {
                    "status": "hit",
                    "rank": matching_rank,
                    "reciprocal_rank": round(reciprocal_rank, 8),
                    # Backward-compatible alias for the chunk result count.
                    "returned_count": len(rows),
                    "returned_chunk_count": len(rows),
                    "returned_document_count": len(documents),
                    "candidate_pool": candidate_pool,
                    "matched": _matched_result_summary(matching_row or {}),
                }
            )
        details.append(detail)

    unevaluable = len(validated) - scored
    hit_rate = (hits / scored) if scored else None
    mrr = (reciprocal_rank_sum / scored) if scored else None
    breakdown = {
        str(k): {
            "cases": values["cases"],
            "hits": values["hits"],
            "hit_at_k": (
                round(values["hits"] / values["cases"], 8)
                if values["cases"]
                else None
            ),
        }
        for k, values in sorted(k_groups.items())
    }
    return {
        "name": index_name,
        "path": str(index_path),
        "status": "ok",
        "metrics": {
            "total_cases": len(validated),
            "scored_cases": scored,
            "unevaluable_cases": unevaluable,
            "hits": hits,
            "misses": scored - hits,
            "hit_at_k": round(hit_rate, 8) if hit_rate is not None else None,
            "mrr": round(mrr, 8) if mrr is not None else None,
            "by_k": breakdown,
        },
        "cases": details,
    }


def parse_named_indexes(values: Sequence[str]) -> List[Tuple[str, Path]]:
    """Parse repeated ``NAME=PATH`` arguments and reject ambiguous names."""

    if not values:
        raise EvaluationError("at least one --index NAME=PATH is required")
    parsed: List[Tuple[str, Path]] = []
    names = set()
    for value in values:
        if "=" not in value:
            raise EvaluationError(
                "index must use NAME=PATH syntax: {}".format(value)
            )
        name, rendered_path = value.split("=", 1)
        name = _nonempty_string(name, "index name")
        rendered_path = _nonempty_string(rendered_path, "index path")
        if name in names:
            raise EvaluationError("duplicate index name: {}".format(name))
        names.add(name)
        parsed.append((name, Path(rendered_path)))
    return parsed


def inspect_index_provenance(name: str, index_path: Path) -> Dict[str, Any]:
    """Read the source-manifest identity from an index without mutating it."""

    path = Path(index_path)
    provenance: Dict[str, Any] = {
        "name": name,
        "path": str(path),
        "index_meta_key": SOURCE_MANIFEST_META_KEY,
        "source_manifest_sha256": None,
        "chunk_count": None,
        "chunk_count_source": None,
        "status": "error",
        "error": None,
    }
    if not path.is_file():
        provenance["error"] = "missing index DB: {}".format(path)
        return provenance

    uri = "file:{}?mode=ro".format(
        urllib.parse.quote(str(path.resolve()), safe="/:")
    )
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        meta_rows = dict(
            connection.execute(
                "SELECT key, value FROM index_meta WHERE key IN (?, ?)",
                (SOURCE_MANIFEST_META_KEY, "chunk_count"),
            ).fetchall()
        )
    except sqlite3.Error as exc:
        provenance["error"] = "{}: {}".format(type(exc).__name__, exc)
        return provenance
    finally:
        if connection is not None:
            connection.close()

    raw_chunk_count = meta_rows.get("chunk_count")
    try:
        chunk_count = int(str(raw_chunk_count).strip())
    except (TypeError, ValueError):
        chunk_count = -1
    if chunk_count >= 0:
        provenance["chunk_count"] = chunk_count
        provenance["chunk_count_source"] = "index_meta.chunk_count"
    else:
        fallback_connection: Optional[sqlite3.Connection] = None
        try:
            fallback_connection = sqlite3.connect(uri, uri=True)
            fallback_connection.execute("PRAGMA query_only = ON")
            counted = fallback_connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()
            if counted is not None and int(counted[0]) >= 0:
                provenance["chunk_count"] = int(counted[0])
                provenance["chunk_count_source"] = "chunks_table_count"
        except (sqlite3.Error, TypeError, ValueError):
            pass
        finally:
            if fallback_connection is not None:
                fallback_connection.close()

    value = str(meta_rows.get(SOURCE_MANIFEST_META_KEY) or "").strip()
    if not value:
        provenance["status"] = "legacy"
        provenance["error"] = "missing non-empty index_meta.{}".format(
            SOURCE_MANIFEST_META_KEY
        )
        return provenance
    provenance["status"] = "ok"
    provenance["source_manifest_sha256"] = value
    return provenance


def _assess_provenance_compatibility(
    indexes: Sequence[Mapping[str, Any]],
    *,
    allow_mixed_provenance: bool,
) -> Dict[str, Any]:
    error_indexes = [
        str(item["name"]) for item in indexes if item.get("status") == "error"
    ]
    legacy_indexes = [
        str(item["name"]) for item in indexes if item.get("status") == "legacy"
    ]
    hashes = sorted(
        {
            str(item["source_manifest_sha256"])
            for item in indexes
            if item.get("source_manifest_sha256")
        }
    )

    if error_indexes:
        compatible = False
        reason = "provenance_read_error"
    elif legacy_indexes:
        compatible = False
        reason = "missing_source_manifest_sha256"
    elif len(hashes) != 1:
        compatible = False
        reason = "mismatched_source_manifest_sha256"
    else:
        compatible = True
        reason = "matching_source_manifest_sha256"

    override_used = bool(allow_mixed_provenance and not compatible)
    return {
        "required_index_meta_key": SOURCE_MANIFEST_META_KEY,
        "compatible": compatible,
        "comparison_allowed": compatible or allow_mixed_provenance,
        "enforced": not allow_mixed_provenance,
        "override_used": override_used,
        "reason": reason,
        "common_source_manifest_sha256": hashes[0] if compatible else None,
        "distinct_source_manifest_sha256": hashes,
        "legacy_indexes": legacy_indexes,
        "error_indexes": error_indexes,
    }


def compare_indexes(
    named_indexes: Sequence[Tuple[str, Path]],
    cases: Sequence[Mapping[str, Any]],
    *,
    case_file: Optional[Path] = None,
    default_institution: Optional[str] = DEFAULT_INSTITUTION,
    searcher: SearchFunction = search_index,
    allow_mixed_provenance: bool = False,
) -> Dict[str, Any]:
    """Evaluate only provenance-compatible indexes unless explicitly overridden."""

    validated = validate_cases(cases)
    if not named_indexes:
        raise EvaluationError("at least one named index is required")
    provenance_indexes = [
        inspect_index_provenance(name, path) for name, path in named_indexes
    ]
    compatibility = _assess_provenance_compatibility(
        provenance_indexes,
        allow_mixed_provenance=allow_mixed_provenance,
    )
    provenance_by_name = {
        str(item["name"]): item for item in provenance_indexes
    }
    results: List[Dict[str, Any]] = []
    if not compatibility["comparison_allowed"]:
        for name, path in named_indexes:
            result = {
                "name": name,
                "path": str(path),
                "status": "not_evaluated",
                "error": "provenance compatibility check failed: {}".format(
                    compatibility["reason"]
                ),
                "metrics": None,
                "cases": [],
                "provenance": provenance_by_name[str(name)],
            }
            results.append(result)
    else:
        for name, path in named_indexes:
            try:
                result = evaluate_index(
                    name,
                    path,
                    validated,
                    default_institution=default_institution,
                    searcher=searcher,
                    index_chunk_count=provenance_by_name[str(name)].get(
                        "chunk_count"
                    ),
                    chunk_count_source=provenance_by_name[str(name)].get(
                        "chunk_count_source"
                    ),
                )
            except Exception as exc:
                result = {
                    "name": name,
                    "path": str(path),
                    "status": "error",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "metrics": None,
                    "cases": [],
                }
            result["provenance"] = provenance_by_name[str(name)]
            results.append(result)

    case_path = Path(case_file) if case_file is not None else None
    case_digest = (
        hashlib.sha256(case_path.read_bytes()).hexdigest()
        if case_path is not None and case_path.is_file()
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_file": str(case_path) if case_path is not None else None,
        "case_file_sha256": case_digest,
        "case_count": len(validated),
        "scored_case_count": sum(
            not case["requires_validation"] for case in validated
        ),
        "unevaluable_case_count": sum(
            case["requires_validation"] for case in validated
        ),
        "provenance": {
            "required_index_meta_key": SOURCE_MANIFEST_META_KEY,
            "indexes": provenance_indexes,
        },
        "compatibility": compatibility,
        "ok": all(result["status"] == "ok" for result in results),
        "indexes": results,
    }


def comparison_csv(report: Mapping[str, Any]) -> str:
    """Render one aggregate row per index for spreadsheet comparison."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    compatibility = report.get("compatibility") or {}
    for index in report.get("indexes", []):
        metrics = index.get("metrics") or {}
        provenance = index.get("provenance") or {}
        writer.writerow(
            {
                "index_name": index.get("name"),
                "index_path": index.get("path"),
                "status": index.get("status"),
                "provenance_status": provenance.get("status"),
                "source_manifest_sha256": provenance.get(
                    "source_manifest_sha256"
                ),
                "provenance_compatible": compatibility.get("compatible"),
                "provenance_override_used": compatibility.get("override_used"),
                "total_cases": metrics.get("total_cases"),
                "scored_cases": metrics.get("scored_cases"),
                "unevaluable_cases": metrics.get("unevaluable_cases"),
                "hits": metrics.get("hits"),
                "misses": metrics.get("misses"),
                "hit_at_k": metrics.get("hit_at_k"),
                "mrr": metrics.get("mrr"),
                "error": index.get("error", ""),
            }
        )
    return buffer.getvalue()


def _atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def write_comparison(
    report: Mapping[str, Any],
    *,
    json_output: Optional[Path] = None,
    csv_output: Optional[Path] = None,
) -> None:
    """Write requested comparison artifacts atomically."""

    if (
        json_output is not None
        and csv_output is not None
        and Path(json_output).resolve() == Path(csv_output).resolve()
    ):
        raise EvaluationError("JSON and CSV outputs must be different paths")
    if json_output is not None:
        _atomic_write_text(
            Path(json_output),
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    if csv_output is not None:
        _atomic_write_text(Path(csv_output), comparison_csv(report))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate the JSONL case schema without querying an index.",
    )
    validate.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    validate.add_argument(
        "--expected-count",
        type=int,
        help="Optionally require an exact number of cases.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Compare one or more named BM25 indexes.",
    )
    evaluate.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    evaluate.add_argument(
        "--index",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named BM25 SQLite index. Repeat to compare profiles.",
    )
    evaluate.add_argument(
        "--institution",
        default=DEFAULT_INSTITUTION,
        help=(
            "Default institution filter; use an empty string to search all "
            "institutions (default: 부산대학교)."
        ),
    )
    evaluate.add_argument(
        "--allow-mixed-provenance",
        "--allow-legacy-or-mixed-indexes",
        action="store_true",
        help=(
            "Explicitly bypass the source_manifest_sha256 compatibility gate "
            "for legacy or mixed-corpus indexes. The report still records the "
            "incompatibility and override."
        ),
    )
    evaluate.add_argument("--json-output", type=Path)
    evaluate.add_argument("--csv-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            cases = load_cases(
                arguments.cases,
                expected_count=arguments.expected_count,
            )
            summary = {
                "schema_version": SCHEMA_VERSION,
                "case_file": str(arguments.cases),
                "case_count": len(cases),
                "scored_case_count": sum(
                    not case["requires_validation"] for case in cases
                ),
                "unevaluable_case_count": sum(
                    case["requires_validation"] for case in cases
                ),
                "valid": True,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        cases = load_cases(arguments.cases)
        named_indexes = parse_named_indexes(arguments.index)
        default_institution = arguments.institution.strip() or None
        report = compare_indexes(
            named_indexes,
            cases,
            case_file=arguments.cases,
            default_institution=default_institution,
            allow_mixed_provenance=arguments.allow_mixed_provenance,
        )
        write_comparison(
            report,
            json_output=arguments.json_output,
            csv_output=arguments.csv_output,
        )
        if arguments.json_output is None and arguments.csv_output is None:
            print(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            summary = {
                "ok": report["ok"],
                "compatibility": report["compatibility"],
                "provenance": report["provenance"],
                "json_output": (
                    str(arguments.json_output)
                    if arguments.json_output is not None
                    else None
                ),
                "csv_output": (
                    str(arguments.csv_output)
                    if arguments.csv_output is not None
                    else None
                ),
                "indexes": [
                    {
                        "name": item["name"],
                        "status": item["status"],
                        "metrics": item["metrics"],
                        "error": item.get("error"),
                    }
                    for item in report["indexes"]
                ],
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    except EvaluationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
