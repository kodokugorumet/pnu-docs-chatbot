#!/usr/bin/env python3
"""Compare two complete service retrieval runs without an LLM judge.

The command joins each service JSONL to the DEV case file, recomputes document
and gold-chunk metrics from the saved ``sources`` array, and emits one JSON
summary plus one case-level CSV.  It intentionally ignores judge fields: this
artifact answers whether retrieval/context selection changed, not whether the
generated answer is correct.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_module
import json
import math
import re
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-eval.jsonl"
DOCUMENT_CUTOFFS = (1, 3, 5)
EVIDENCE_CUTOFFS = (5, 8)
SCHEMA_VERSION = "service-retrieval-ab/v4"
MIN_NORMALIZED_EVIDENCE_QUOTE_CHARS = 20
MIN_NORMALIZED_EVIDENCE_FRAGMENT_CHARS = 10


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    return records


def load_cases(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        case_id = str(record.get("id") or "")
        if not case_id:
            raise ValueError(f"{path}: case without id")
        if case_id in cases:
            raise ValueError(f"{path}: duplicate case id {case_id}")
        order.append(case_id)
        cases[case_id] = record
    if not order:
        raise ValueError(f"{path}: no cases")
    return order, cases


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_complete_run(
    path: Path,
    expected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Load one judge-free run and require exactly the DEV case ID set."""

    by_id: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        case_id = str(record.get("case_id") or record.get("id") or "")
        if not case_id:
            raise ValueError(f"{path}: record without id")
        if case_id in by_id:
            raise ValueError(f"{path}: duplicate id {case_id}")
        if record.get("error"):
            raise ValueError(f"{path}: {case_id} has error: {record['error']}")
        if not isinstance(record.get("sources"), list):
            raise ValueError(f"{path}: {case_id} has no sources list")
        latency = record.get("latency_ms")
        if not _finite_number(latency) or float(latency) < 0:
            raise ValueError(f"{path}: {case_id} has invalid latency_ms {latency!r}")
        by_id[case_id] = record

    actual_ids = set(by_id)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            f"{path}: incomplete ID set; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    return by_id


def normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def document_key(source: dict[str, Any], position: int = 0) -> str:
    identifier = str(
        source.get("document_id")
        or source.get("doc_id")
        or source.get("chunk_id")
        or ""
    )
    if identifier:
        return identifier
    source_url = normalized(source.get("source_url"))
    if source_url:
        return f"url:{source_url}"
    title_host = (
        normalized(source.get("source_title")),
        normalized(source.get("source_host")),
    )
    if any(title_host):
        return "title-host:" + "|".join(title_host)
    return f"anonymous-position:{position}"


def document_values(
    record: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    """Return title-based document rank after folding repeated chunks."""

    expected = case.get("expected") or {}
    wanted_title = normalized(expected.get("source_title_contains"))
    wanted_host = normalized(expected.get("source_host"))
    seen_documents: set[str] = set()
    rank: int | None = None
    host_match: bool | None = None

    for position, source in enumerate(record.get("sources") or [], 1):
        key = document_key(source, position)
        if key in seen_documents:
            continue
        seen_documents.add(key)
        document_rank = len(seen_documents)
        if wanted_title and wanted_title in normalized(source.get("source_title")):
            rank = document_rank
            host_match = (
                not wanted_host
                or wanted_host == normalized(source.get("source_host"))
            )
            break

    hit_at_k = {
        str(k): rank is not None and rank <= k
        for k in DOCUMENT_CUTOFFS
    }
    return {
        "rank": rank,
        "hit_at_k": hit_at_k,
        "reciprocal_rank_at_5": (
            1.0 / rank if rank is not None and rank <= 5 else 0.0
        ),
        # A mirror host remains a title hit, matching the service evaluator.
        "host_match": host_match,
    }


def evidence_values(
    record: dict[str, Any],
    case: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    """Compute exact required-gold-chunk coverage at one context cutoff."""

    cutoff = max(0, int(k))
    wanted: set[str] = set()
    for _, item in _required_evidence_items(case):
        if item.get("chunk_id"):
            wanted.add(str(item["chunk_id"]))
    returned = {
        str(source.get("chunk_id"))
        for source in (record.get("sources") or [])[:cutoff]
        if source.get("chunk_id")
    }
    found = sorted(wanted & returned)
    missing = sorted(wanted - returned)
    return {
        "k": cutoff,
        "any_matched": bool(found),
        "all_matched": bool(wanted) and not missing,
        "recall": len(found) / len(wanted) if wanted else 0.0,
        "found_chunk_ids": found,
        "missing_chunk_ids": missing,
        "wanted": len(wanted),
    }


def _required_evidence_items(
    case: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    required_items: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(case.get("evidence") or []):
        if not isinstance(item, dict):
            raise ValueError(f"evidence[{index}] must be an object")
        required = item.get("required_for_answer", True)
        if not isinstance(required, bool):
            raise ValueError(
                f"evidence[{index}].required_for_answer must be boolean"
            )
        if required:
            required_items.append((index, item))
    return required_items


def _normalized_evidence_text(value: Any) -> str:
    normalized_value = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", normalized_value).casefold()


def _source_evidence_text(source: dict[str, Any]) -> str:
    return "\n".join(
        str(source.get(field) or "")
        for field in ("preview", "text", "excerpt")
    )


def evidence_content_values(
    record: dict[str, Any],
    case: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    """Measure required evidence items by exact chunk or verbatim quote.

    Exact chunk IDs remain the primary deterministic retrieval diagnostic.
    This companion metric conservatively recovers duplicate-document or
    re-chunked copies only when the complete normalized gold quote is present
    in a returned source. It does not use fuzzy or semantic matching.
    """

    cutoff = max(0, int(k))
    sources = (record.get("sources") or [])[:cutoff]
    normalized_sources = [
        _normalized_evidence_text(_source_evidence_text(source))
        for source in sources
    ]
    matches: list[dict[str, Any]] = []
    missing_evidence_indices: list[int] = []

    required_items = _required_evidence_items(case)
    for evidence_index, item in required_items:
        expected_chunk_id = str(item.get("chunk_id") or "")
        matched_source_index: int | None = None
        method: str | None = None

        if expected_chunk_id:
            matched_source_index = next(
                (
                    source_index
                    for source_index, source in enumerate(sources)
                    if str(source.get("chunk_id") or "")
                    == expected_chunk_id
                ),
                None,
            )
            if matched_source_index is not None:
                method = "exact_chunk_id"

        normalized_quote = _normalized_evidence_text(item.get("quote"))
        if (
            matched_source_index is None
            and len(normalized_quote)
            >= MIN_NORMALIZED_EVIDENCE_QUOTE_CHARS
        ):
            matched_source_index = next(
                (
                    source_index
                    for source_index, source_text in enumerate(
                        normalized_sources
                    )
                    if normalized_quote in source_text
                ),
                None,
            )
            if matched_source_index is not None:
                method = "normalized_quote"

        normalized_quote_lines = [
            _normalized_evidence_text(line)
            for line in str(item.get("quote") or "").splitlines()
            if len(_normalized_evidence_text(line))
            >= MIN_NORMALIZED_EVIDENCE_FRAGMENT_CHARS
        ]
        if (
            matched_source_index is None
            and len(normalized_quote_lines) >= 2
        ):
            matched_source_index = next(
                (
                    source_index
                    for source_index, source_text in enumerate(
                        normalized_sources
                    )
                    if all(
                        fragment in source_text
                        for fragment in normalized_quote_lines
                    )
                ),
                None,
            )
            if matched_source_index is not None:
                method = "normalized_quote_lines"

        if matched_source_index is None:
            missing_evidence_indices.append(evidence_index)
            continue

        matched_source = sources[matched_source_index]
        matches.append(
            {
                "evidence_index": evidence_index,
                "expected_chunk_id": expected_chunk_id or None,
                "source_rank": matched_source_index + 1,
                "source_chunk_id": matched_source.get("chunk_id"),
                "method": method,
            }
        )

    wanted = len(required_items)
    return {
        "k": cutoff,
        "any_matched": bool(matches),
        "all_matched": bool(wanted) and not missing_evidence_indices,
        "recall": len(matches) / wanted if wanted else 0.0,
        "matches": matches,
        "missing_evidence_indices": missing_evidence_indices,
        "wanted": wanted,
    }


def linear_quantile(values: Iterable[float], probability: float) -> float:
    """Return an R-7/NumPy-style linearly interpolated quantile."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _source_signature(record: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(source.get("chunk_id") or ""),
            str(source.get("source_title") or ""),
            str(source.get("source_host") or ""),
        )
        for source in record.get("sources") or []
    )


def _chunk_ids(record: dict[str, Any]) -> list[str]:
    return [
        str(source.get("chunk_id") or "")
        for source in record.get("sources") or []
    ]


def _source_titles(record: dict[str, Any]) -> list[str]:
    return [
        str(source.get("source_title") or "")
        for source in record.get("sources") or []
    ]


def _unique_document_count(record: dict[str, Any]) -> int:
    return len(
        {
            document_key(source, position)
            for position, source in enumerate(record.get("sources") or [], 1)
        }
    )


def _answer_hash(record: dict[str, Any]) -> str:
    answer = str(record.get("answer") or "")
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def record_values(
    record: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    return {
        "document": document_values(record, case),
        "evidence_at_k": {
            str(k): evidence_values(record, case, k)
            for k in EVIDENCE_CUTOFFS
        },
        "evidence_content_at_k": {
            str(k): evidence_content_values(record, case, k)
            for k in EVIDENCE_CUTOFFS
        },
        "latency_ms": float(record["latency_ms"]),
        "unique_documents": _unique_document_count(record),
        "source_signature": _source_signature(record),
        "source_chunk_ids": _chunk_ids(record),
        "source_titles": _source_titles(record),
        "answer": str(record.get("answer") or ""),
        "answer_sha256": _answer_hash(record),
    }


def summarize_arm(
    order: list[str],
    per_case: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    question_count = len(order)
    document: dict[str, Any] = {}
    for k in DOCUMENT_CUTOFFS:
        count = sum(
            bool(per_case[case_id]["document"]["hit_at_k"][str(k)])
            for case_id in order
        )
        document[f"hit_at_{k}"] = {
            "count": count,
            "rate": count / question_count,
        }
    document["mrr_at_5"] = mean(
        float(per_case[case_id]["document"]["reciprocal_rank_at_5"])
        for case_id in order
    )

    gold_chunk: dict[str, Any] = {}
    for k in EVIDENCE_CUTOFFS:
        values = [
            per_case[case_id]["evidence_at_k"][str(k)]
            for case_id in order
        ]
        any_count = sum(bool(value["any_matched"]) for value in values)
        all_count = sum(bool(value["all_matched"]) for value in values)
        gold_chunk[str(k)] = {
            "metric_family": "exact_required_gold_chunk_coverage",
            "any_count": any_count,
            "any_rate": any_count / question_count,
            "all_count": all_count,
            "all_rate": all_count / question_count,
            "mean_gold_chunk_recall": mean(
                float(value["recall"]) for value in values
            ),
        }

    required_evidence_item: dict[str, Any] = {}
    for k in EVIDENCE_CUTOFFS:
        values = [
            per_case[case_id]["evidence_content_at_k"][str(k)]
            for case_id in order
        ]
        any_count = sum(bool(value["any_matched"]) for value in values)
        all_count = sum(bool(value["all_matched"]) for value in values)
        required_evidence_item[str(k)] = {
            "metric_family": (
                "required_evidence_item_exact_chunk_or_normalized_quote_v2"
            ),
            "any_count": any_count,
            "any_rate": any_count / question_count,
            "all_count": all_count,
            "all_rate": all_count / question_count,
            "mean_recall": mean(float(value["recall"]) for value in values),
        }

    latencies = [per_case[case_id]["latency_ms"] for case_id in order]
    unique_documents = [
        per_case[case_id]["unique_documents"] for case_id in order
    ]
    return {
        "questions": question_count,
        "document": document,
        "gold_chunk": gold_chunk,
        "required_evidence_item": required_evidence_item,
        "latency_ms": {
            "mean": mean(latencies),
            "p50": linear_quantile(latencies, 0.50),
            "p95": linear_quantile(latencies, 0.95),
            "quantile_method": "linear interpolation at (n - 1) * p",
        },
        "unique_documents": {
            "mean": mean(unique_documents),
            "minimum": min(unique_documents),
            "maximum": max(unique_documents),
        },
    }


def _transition(before: bool, after: bool) -> str:
    if after and not before:
        return "gained"
    if before and not after:
        return "lost"
    if before:
        return "retained"
    return "absent"


def _transition_lists(
    order: list[str],
    per_a: dict[str, dict[str, Any]],
    per_b: dict[str, dict[str, Any]],
    getter: Callable[[dict[str, Any]], bool],
) -> dict[str, list[str]]:
    gained = [
        case_id
        for case_id in order
        if getter(per_b[case_id]) and not getter(per_a[case_id])
    ]
    lost = [
        case_id
        for case_id in order
        if getter(per_a[case_id]) and not getter(per_b[case_id])
    ]
    return {"gained": gained, "lost": lost}


def _ordered_difference(left: list[str], right: list[str]) -> list[str]:
    right_set = set(right)
    return [value for value in left if value not in right_set]


def _chunk_rank_changes(
    before: list[str],
    after: list[str],
) -> list[dict[str, Any]]:
    before_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(before, 1)
        if chunk_id
    }
    after_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(after, 1)
        if chunk_id
    }
    return [
        {
            "chunk_id": chunk_id,
            "rank_a": before_ranks[chunk_id],
            "rank_b": after_ranks[chunk_id],
        }
        for chunk_id in before
        if (
            chunk_id
            and chunk_id in after_ranks
            and before_ranks[chunk_id] != after_ranks[chunk_id]
        )
    ]


def _numeric_deltas(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for k in DOCUMENT_CUTOFFS:
        key = f"document_hit_at_{k}_rate"
        deltas[key] = (
            summary_b["document"][f"hit_at_{k}"]["rate"]
            - summary_a["document"][f"hit_at_{k}"]["rate"]
        )
    deltas["document_mrr_at_5"] = (
        summary_b["document"]["mrr_at_5"]
        - summary_a["document"]["mrr_at_5"]
    )
    for k in EVIDENCE_CUTOFFS:
        for metric in ("any_rate", "all_rate", "mean_gold_chunk_recall"):
            deltas[f"gold_chunk_{metric}_at_{k}"] = (
                summary_b["gold_chunk"][str(k)][metric]
                - summary_a["gold_chunk"][str(k)][metric]
            )
        for metric in ("any_rate", "all_rate", "mean_recall"):
            deltas[f"required_evidence_item_{metric}_at_{k}"] = (
                summary_b["required_evidence_item"][str(k)][metric]
                - summary_a["required_evidence_item"][str(k)][metric]
            )
    for metric in ("mean", "p50", "p95"):
        deltas[f"latency_ms_{metric}"] = (
            summary_b["latency_ms"][metric]
            - summary_a["latency_ms"][metric]
        )
    deltas["unique_documents_mean"] = (
        summary_b["unique_documents"]["mean"]
        - summary_a["unique_documents"]["mean"]
    )
    return deltas


def analyze_runs(
    order: list[str],
    cases: dict[str, dict[str, Any]],
    run_a: dict[str, dict[str, Any]],
    run_b: dict[str, dict[str, Any]],
    *,
    label_a: str,
    label_b: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not order:
        raise ValueError("at least one case is required")
    if not label_a or not label_b or label_a == label_b:
        raise ValueError("A/B labels must be non-empty and distinct")

    per_a = {
        case_id: record_values(run_a[case_id], cases[case_id])
        for case_id in order
    }
    per_b = {
        case_id: record_values(run_b[case_id], cases[case_id])
        for case_id in order
    }
    summary_a = summarize_arm(order, per_a)
    summary_b = summarize_arm(order, per_b)

    case_changes: dict[str, dict[str, list[str]]] = {}
    for k in DOCUMENT_CUTOFFS:
        case_changes[f"document_hit_at_{k}"] = _transition_lists(
            order,
            per_a,
            per_b,
            lambda value, cutoff=str(k): bool(
                value["document"]["hit_at_k"][cutoff]
            ),
        )
    for k in EVIDENCE_CUTOFFS:
        case_changes[f"any_gold_chunk_at_{k}"] = _transition_lists(
            order,
            per_a,
            per_b,
            lambda value, cutoff=str(k): bool(
                value["evidence_at_k"][cutoff]["any_matched"]
            ),
        )
        case_changes[f"all_gold_chunk_at_{k}"] = _transition_lists(
            order,
            per_a,
            per_b,
            lambda value, cutoff=str(k): bool(
                value["evidence_at_k"][cutoff]["all_matched"]
            ),
        )
        case_changes[f"any_required_evidence_item_at_{k}"] = (
            _transition_lists(
                order,
                per_a,
                per_b,
                lambda value, cutoff=str(k): bool(
                    value["evidence_content_at_k"][cutoff]["any_matched"]
                ),
            )
        )
        case_changes[f"all_required_evidence_item_at_{k}"] = (
            _transition_lists(
                order,
                per_a,
                per_b,
                lambda value, cutoff=str(k): bool(
                    value["evidence_content_at_k"][cutoff]["all_matched"]
                ),
            )
        )

    source_changed = [
        case_id
        for case_id in order
        if per_a[case_id]["source_signature"]
        != per_b[case_id]["source_signature"]
    ]
    source_rank_changed = [
        case_id
        for case_id in order
        if per_a[case_id]["document"]["rank"]
        != per_b[case_id]["document"]["rank"]
    ]
    answer_changed = [
        case_id
        for case_id in order
        if per_a[case_id]["answer"] != per_b[case_id]["answer"]
    ]
    chunk_rank_changed = [
        case_id
        for case_id in order
        if _chunk_rank_changes(
            per_a[case_id]["source_chunk_ids"],
            per_b[case_id]["source_chunk_ids"],
        )
    ]

    case_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for case_id in order:
        before = per_a[case_id]
        after = per_b[case_id]
        before_chunks = before["source_chunk_ids"]
        after_chunks = after["source_chunk_ids"]
        chunk_rank_changes = _chunk_rank_changes(before_chunks, after_chunks)
        metric_changes: dict[str, str] = {}
        for k in DOCUMENT_CUTOFFS:
            metric_changes[f"document_hit_at_{k}"] = _transition(
                bool(before["document"]["hit_at_k"][str(k)]),
                bool(after["document"]["hit_at_k"][str(k)]),
            )
        for k in EVIDENCE_CUTOFFS:
            metric_changes[f"any_gold_chunk_at_{k}"] = _transition(
                bool(before["evidence_at_k"][str(k)]["any_matched"]),
                bool(after["evidence_at_k"][str(k)]["any_matched"]),
            )
            metric_changes[f"all_gold_chunk_at_{k}"] = _transition(
                bool(before["evidence_at_k"][str(k)]["all_matched"]),
                bool(after["evidence_at_k"][str(k)]["all_matched"]),
            )
            metric_changes[f"any_required_evidence_item_at_{k}"] = (
                _transition(
                    bool(
                        before["evidence_content_at_k"][str(k)][
                            "any_matched"
                        ]
                    ),
                    bool(
                        after["evidence_content_at_k"][str(k)][
                            "any_matched"
                        ]
                    ),
                )
            )
            metric_changes[f"all_required_evidence_item_at_{k}"] = (
                _transition(
                    bool(
                        before["evidence_content_at_k"][str(k)][
                            "all_matched"
                        ]
                    ),
                    bool(
                        after["evidence_content_at_k"][str(k)][
                            "all_matched"
                        ]
                    ),
                )
            )

        changes = {
            "metrics": metric_changes,
            "source_changed": (
                before["source_signature"] != after["source_signature"]
            ),
            "source_rank_changed": (
                before["document"]["rank"] != after["document"]["rank"]
            ),
            "chunk_rank_changed": bool(chunk_rank_changes),
            "chunk_rank_changes": chunk_rank_changes,
            "answer_changed": before["answer"] != after["answer"],
            "added_chunk_ids": _ordered_difference(
                after_chunks,
                before_chunks,
            ),
            "removed_chunk_ids": _ordered_difference(
                before_chunks,
                after_chunks,
            ),
            "source_order_changed": before_chunks != after_chunks,
        }
        case_row = {
            "id": case_id,
            "query": str(cases[case_id].get("query") or ""),
            "category": cases[case_id].get("category"),
            "role": cases[case_id].get("role"),
            label_a: {
                "document": before["document"],
                "evidence_at_k": before["evidence_at_k"],
                "evidence_content_at_k": before[
                    "evidence_content_at_k"
                ],
                "latency_ms": before["latency_ms"],
                "unique_documents": before["unique_documents"],
                "source_chunk_ids": before_chunks,
                "source_titles": before["source_titles"],
                "answer_sha256": before["answer_sha256"],
            },
            label_b: {
                "document": after["document"],
                "evidence_at_k": after["evidence_at_k"],
                "evidence_content_at_k": after[
                    "evidence_content_at_k"
                ],
                "latency_ms": after["latency_ms"],
                "unique_documents": after["unique_documents"],
                "source_chunk_ids": after_chunks,
                "source_titles": after["source_titles"],
                "answer_sha256": after["answer_sha256"],
            },
            "changes": changes,
        }
        case_rows.append(case_row)

        csv_row: dict[str, Any] = {
            "id": case_id,
            "query": str(cases[case_id].get("query") or ""),
            "category": cases[case_id].get("category"),
            "role": cases[case_id].get("role"),
            "label_a": label_a,
            "label_b": label_b,
            "document_rank_a": before["document"]["rank"],
            "document_rank_b": after["document"]["rank"],
            "document_mrr_at_5_a": before["document"]["reciprocal_rank_at_5"],
            "document_mrr_at_5_b": after["document"]["reciprocal_rank_at_5"],
            "latency_ms_a": before["latency_ms"],
            "latency_ms_b": after["latency_ms"],
            "latency_ms_delta_b_minus_a": (
                after["latency_ms"] - before["latency_ms"]
            ),
            "unique_documents_a": before["unique_documents"],
            "unique_documents_b": after["unique_documents"],
            "source_changed": changes["source_changed"],
            "source_rank_changed": changes["source_rank_changed"],
            "chunk_rank_changed": changes["chunk_rank_changed"],
            "chunk_rank_changes": changes["chunk_rank_changes"],
            "answer_changed": changes["answer_changed"],
            "added_chunk_ids": changes["added_chunk_ids"],
            "removed_chunk_ids": changes["removed_chunk_ids"],
            "source_chunk_ids_a": before_chunks,
            "source_chunk_ids_b": after_chunks,
            "source_titles_a": before["source_titles"],
            "source_titles_b": after["source_titles"],
        }
        for k in DOCUMENT_CUTOFFS:
            csv_row[f"document_hit_at_{k}_a"] = before["document"][
                "hit_at_k"
            ][str(k)]
            csv_row[f"document_hit_at_{k}_b"] = after["document"][
                "hit_at_k"
            ][str(k)]
            csv_row[f"document_hit_at_{k}_change"] = metric_changes[
                f"document_hit_at_{k}"
            ]
        for k in EVIDENCE_CUTOFFS:
            before_evidence = before["evidence_at_k"][str(k)]
            after_evidence = after["evidence_at_k"][str(k)]
            csv_row[f"any_gold_chunk_at_{k}_a"] = before_evidence[
                "any_matched"
            ]
            csv_row[f"any_gold_chunk_at_{k}_b"] = after_evidence[
                "any_matched"
            ]
            csv_row[f"all_gold_chunk_at_{k}_a"] = before_evidence[
                "all_matched"
            ]
            csv_row[f"all_gold_chunk_at_{k}_b"] = after_evidence[
                "all_matched"
            ]
            csv_row[f"gold_chunk_recall_at_{k}_a"] = before_evidence[
                "recall"
            ]
            csv_row[f"gold_chunk_recall_at_{k}_b"] = after_evidence[
                "recall"
            ]
            csv_row[f"any_gold_chunk_at_{k}_change"] = metric_changes[
                f"any_gold_chunk_at_{k}"
            ]
            csv_row[f"all_gold_chunk_at_{k}_change"] = metric_changes[
                f"all_gold_chunk_at_{k}"
            ]
        csv_rows.append(csv_row)

    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "judge-free paired service retrieval comparison",
        "metric_notes": {
            "document": (
                "Document rank folds repeated chunks by document_id and uses "
                "normalized source-title substring matching. Host mismatch is "
                "diagnostic only."
            ),
            "gold_chunk": (
                "Required DEV evidence is a set of exact Cascade chunk IDs; "
                "items marked required_for_answer=false are excluded. Any/All "
                "and mean recall are required-chunk coverage diagnostics, not "
                "atomic claim Evidence Recall."
            ),
            "required_evidence_item": (
                "Each required DEV evidence item matches by exact chunk ID or "
                "only when its complete NFKC/whitespace-normalized quote occurs "
                "in a returned source. For multi-line quotes, every normalized "
                "line may occur in the same source in a different order. This "
                "conservative companion metric separates duplicate-document/"
                "re-chunking misses from content misses; it uses no fuzzy or "
                "semantic matching."
            ),
            "latency_ms": (
                "Client-observed latency from one complete run; p50/p95 use "
                "linear interpolation."
            ),
            "answer_changed": (
                "Exact answer-string change only; no LLM judge is invoked."
            ),
            "rank_changes": (
                "source_rank_changed means the expected document rank changed; "
                "chunk_rank_changes lists shared chunks whose final context rank changed."
            ),
        },
        "labels": {"a": label_a, "b": label_b},
        "arms": {label_a: summary_a, label_b: summary_b},
        "comparison": {
            "deltas_b_minus_a": _numeric_deltas(summary_a, summary_b),
            "case_changes": case_changes,
            "source_changed": source_changed,
            "source_rank_changed": source_rank_changed,
            "expected_document_rank_changed": source_rank_changed,
            "chunk_rank_changed": chunk_rank_changed,
            "answer_changed": answer_changed,
            "source_changed_count": len(source_changed),
            "source_rank_changed_count": len(source_rank_changed),
            "expected_document_rank_changed_count": len(source_rank_changed),
            "chunk_rank_changed_count": len(chunk_rank_changed),
            "answer_changed_count": len(answer_changed),
        },
        "cases": case_rows,
    }
    return report, csv_rows


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write CSV without case rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {key: _serialize_csv_value(value) for key, value in row.items()}
            )


def _html(value: Any) -> str:
    return html_module.escape(str(value or ""), quote=True)


def _html_bool(value: bool) -> str:
    return "✓" if value else "✕"


def _html_sources(record: dict[str, Any]) -> str:
    sources = record.get("sources") or []
    if not sources:
        return '<p class="empty">출처 없음</p>'
    items: list[str] = []
    for rank, source in enumerate(sources, 1):
        title = _html(source.get("source_title") or "제목 없음")
        host = _html(source.get("source_host") or "")
        chunk_id = _html(source.get("chunk_id") or "chunk ID 없음")
        document_id = _html(
            source.get("document_id") or source.get("doc_id") or ""
        )
        items.append(
            '<li><span class="source-rank">'
            f"{rank}</span><div><strong>{title}</strong>"
            f'<small>{host}</small><code>{chunk_id}</code>'
            f'<small class="doc-id">{document_id}</small></div></li>'
        )
    return '<ol class="sources">' + "".join(items) + "</ol>"


def _html_metric_rows(
    case_row: dict[str, Any],
    label_a: str,
    label_b: str,
) -> str:
    before = case_row[label_a]
    after = case_row[label_b]
    changes = case_row["changes"]["metrics"]
    rows: list[str] = []
    for k in EVIDENCE_CUTOFFS:
        before_evidence = before["evidence_at_k"][str(k)]
        after_evidence = after["evidence_at_k"][str(k)]
        for key, label in (
            ("any_matched", f"Any-Gold-Chunk@{k}"),
            ("all_matched", f"All-Gold-Chunks@{k}"),
        ):
            change_key = (
                f"any_gold_chunk_at_{k}"
                if key == "any_matched"
                else f"all_gold_chunk_at_{k}"
            )
            status = changes[change_key]
            rows.append(
                f'<tr class="metric-{_html(status)}"><th>{label}</th>'
                f"<td>{_html_bool(bool(before_evidence[key]))}</td>"
                f"<td>{_html_bool(bool(after_evidence[key]))}</td>"
                f'<td><span class="status {status}">{_html(status)}</span></td></tr>'
            )
        recall_a = float(before_evidence["recall"])
        recall_b = float(after_evidence["recall"])
        delta = recall_b - recall_a
        recall_status = "gained" if delta > 0 else "lost" if delta < 0 else "retained"
        rows.append(
            f'<tr class="metric-{recall_status}"><th>GoldChunkRecall@{k}</th>'
            f"<td>{recall_a:.3f}</td><td>{recall_b:.3f}</td>"
            f'<td><span class="status {recall_status}">{delta:+.3f}</span></td></tr>'
        )
    return "".join(rows)


def render_html(
    report: dict[str, Any],
    order: list[str],
    cases: dict[str, dict[str, Any]],
    run_a: dict[str, dict[str, Any]],
    run_b: dict[str, dict[str, Any]],
    *,
    label_a: str,
    label_b: str,
) -> str:
    """Render a self-contained, filterable side-by-side comparison."""

    by_id = {row["id"]: row for row in report["cases"]}
    arm_a = report["arms"][label_a]
    arm_b = report["arms"][label_b]
    title = f"{label_a} vs {label_b} · Retrieval DEV 비교"
    parts = [
        "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_html(title)}</title>",
        """<style>
        :root{color-scheme:light;--ink:#172033;--muted:#667085;--line:#dfe3ea;
        --panel:#fff;--bg:#f4f6f9;--good:#087443;--good-bg:#eaf8f0;
        --bad:#b42318;--bad-bg:#fff0ee;--accent:#1e5eff}
        *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5}
        header{padding:32px max(20px,calc((100vw - 1440px)/2));background:#132238;color:white}
        header h1{margin:0 0 8px;font-size:26px}header p{margin:0;color:#cbd5e1}
        main{max-width:1440px;margin:auto;padding:24px 20px 64px}.summary{display:grid;
        grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:18px}
        .metric-card,.case{background:var(--panel);border:1px solid var(--line);border-radius:14px}
        .metric-card{padding:16px}.metric-card small{display:block;color:var(--muted)}
        .metric-card strong{font-size:22px}.toolbar{position:sticky;top:0;z-index:2;
        display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px;margin-bottom:16px;
        background:rgba(244,246,249,.94);backdrop-filter:blur(8px);border:1px solid var(--line);
        border-radius:12px}.toolbar label{font-weight:700}.toolbar select,.toolbar input{
        border:1px solid #b9c1ce;border-radius:8px;padding:9px 11px;background:white}
        .toolbar input{min-width:260px;flex:1}.case{margin-bottom:16px;overflow:hidden}
        .case-head{padding:16px 18px;border-bottom:1px solid var(--line)}.case-head h2{
        margin:0 0 4px;font-size:18px}.query{font-size:16px}.badges{display:flex;gap:6px;
        flex-wrap:wrap;margin-top:8px}.badge,.status{display:inline-block;border-radius:999px;
        padding:2px 8px;font-size:12px;background:#eef1f5;color:#475467}.badge.gained,
        .status.gained{color:var(--good);background:var(--good-bg)}.badge.lost,
        .status.lost{color:var(--bad);background:var(--bad-bg)}.compare{display:grid;
        grid-template-columns:1fr 1fr}.arm{min-width:0;padding:18px}.arm+ .arm{
        border-left:1px solid var(--line)}.arm h3{margin:0 0 12px}.latency{color:var(--muted)}
        pre{white-space:pre-wrap;word-break:break-word;background:#f8fafc;border:1px solid var(--line);
        border-radius:9px;padding:12px;min-height:110px;max-height:320px;overflow:auto;font:13px/1.55 ui-monospace,SFMono-Regular,monospace}
        .sources{list-style:none;margin:0;padding:0}.sources li{display:flex;gap:9px;padding:9px 0;
        border-top:1px solid #edf0f4}.source-rank{display:grid;place-items:center;flex:0 0 24px;
        height:24px;border-radius:50%;background:#e9efff;color:#1746b5;font-weight:700;font-size:12px}
        .sources strong,.sources small,.sources code{display:block;word-break:break-word}
        .sources small{color:var(--muted)}.sources code{font-size:11px;color:#344054}.doc-id{font-size:10px}
        .evidence{padding:0 18px 18px}.evidence table{width:100%;border-collapse:collapse}
        th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}.empty{color:var(--muted)}
        [hidden]{display:none!important}@media(max-width:820px){.compare{grid-template-columns:1fr}
        .arm+ .arm{border-left:0;border-top:1px solid var(--line)}}
        </style></head><body>""",
        f"<header><h1>{_html(title)}</h1><p>judge 없이 저장된 sources와 DEV gold chunk를 직접 비교합니다. 답변 열은 extractive 출력이며 LLM 생성 품질 평가가 아닙니다.</p></header>",
        "<main><section class=\"summary\">",
    ]
    for metric_label, value_a, value_b in (
        (
            "Document Hit@5",
            arm_a["document"]["hit_at_5"]["count"],
            arm_b["document"]["hit_at_5"]["count"],
        ),
        (
            "Any-Gold-Chunk@8",
            arm_a["gold_chunk"]["8"]["any_count"],
            arm_b["gold_chunk"]["8"]["any_count"],
        ),
        (
            "All-Gold-Chunks@8",
            arm_a["gold_chunk"]["8"]["all_count"],
            arm_b["gold_chunk"]["8"]["all_count"],
        ),
        (
            "Latency p50",
            f"{arm_a['latency_ms']['p50']:.1f}ms",
            f"{arm_b['latency_ms']['p50']:.1f}ms",
        ),
        (
            "평균 unique docs",
            f"{arm_a['unique_documents']['mean']:.2f}",
            f"{arm_b['unique_documents']['mean']:.2f}",
        ),
    ):
        parts.append(
            '<article class="metric-card">'
            f"<small>{_html(metric_label)}</small><strong>{_html(value_a)} → {_html(value_b)}</strong>"
            f"<small>{_html(label_a)} → {_html(label_b)}</small></article>"
        )
    parts.extend(
        [
            "</section>",
            '<section class="toolbar" aria-label="신규/상실/변경 필터">',
            '<label for="case-filter">신규/상실/변경 필터</label>',
            '<select id="case-filter"><option value="all">전체</option>',
            '<option value="gained">근거 신규</option><option value="lost">근거 상실</option>',
            '<option value="changed">모든 변경</option><option value="source">출처 변경</option>',
            '<option value="answer">답변 변경</option></select>',
            '<input id="case-search" type="search" placeholder="case ID 또는 질문 검색">',
            '<span id="visible-count"></span></section><section id="cases">',
        ]
    )

    for case_id in order:
        case = cases[case_id]
        row = by_id[case_id]
        changes = row["changes"]
        statuses = set(changes["metrics"].values())
        gained = "gained" in statuses
        lost = "lost" in statuses
        changed = (
            gained
            or lost
            or changes["source_changed"]
            or changes["answer_changed"]
        )
        badges: list[str] = []
        if gained:
            badges.append('<span class="badge gained">근거 신규</span>')
        if lost:
            badges.append('<span class="badge lost">근거 상실</span>')
        if changes["source_changed"]:
            badges.append('<span class="badge">출처 변경</span>')
        if changes["answer_changed"]:
            badges.append('<span class="badge">답변 변경</span>')
        if not badges:
            badges.append('<span class="badge">변경 없음</span>')
        query = _html(case.get("query") or "")
        category = _html(case.get("category") or "")
        role = _html(case.get("role") or "")
        parts.append(
            '<article class="case" '
            f'data-gained="{int(gained)}" data-lost="{int(lost)}" '
            f'data-changed="{int(changed)}" '
            f'data-source="{int(bool(changes["source_changed"]))}" '
            f'data-answer="{int(bool(changes["answer_changed"]))}">'
            '<div class="case-head">'
            f"<h2>{_html(case_id)} · {category}{' · ' + role if role else ''}</h2>"
            f'<div class="query">{query}</div><div class="badges">{"".join(badges)}</div></div>'
            '<div class="compare">'
        )
        for label, record, values in (
            (label_a, run_a[case_id], row[label_a]),
            (label_b, run_b[case_id], row[label_b]),
        ):
            parts.append(
                '<section class="arm">'
                f"<h3>{_html(label)}</h3>"
                f'<div class="latency">latency {float(values["latency_ms"]):.3f}ms · '
                f'unique docs {int(values["unique_documents"])}</div>'
                f"<h4>답변</h4><pre>{_html(record.get('answer') or '')}</pre>"
                f"<h4>출처·청크 순서</h4>{_html_sources(record)}</section>"
            )
        parts.append(
            '</div><div class="evidence"><h3>Gold chunk 변화</h3><table>'
            f"<thead><tr><th>지표</th><th>{_html(label_a)}</th>"
            f"<th>{_html(label_b)}</th><th>변화</th></tr></thead><tbody>"
            f"{_html_metric_rows(row, label_a, label_b)}</tbody></table></div></article>"
        )

    parts.append(
        """</section></main><script>
        (()=>{const filter=document.querySelector('#case-filter');
        const search=document.querySelector('#case-search');const count=document.querySelector('#visible-count');
        const cards=[...document.querySelectorAll('.case')];function apply(){const mode=filter.value;
        const needle=search.value.trim().toLocaleLowerCase('ko');let visible=0;for(const card of cards){
        const flag=mode==='all'||(mode==='gained'&&card.dataset.gained==='1')||
        (mode==='lost'&&card.dataset.lost==='1')||(mode==='changed'&&card.dataset.changed==='1')||
        (mode==='source'&&card.dataset.source==='1')||(mode==='answer'&&card.dataset.answer==='1');
        const text=!needle||card.textContent.toLocaleLowerCase('ko').includes(needle);
        card.hidden=!(flag&&text);if(!card.hidden)visible++;}count.textContent=`${visible} / ${cards.length}`;}
        filter.addEventListener('change',apply);search.addEventListener('input',apply);apply();})();
        </script></body></html>"""
    )
    return "".join(parts)


def write_html(
    path: Path,
    report: dict[str, Any],
    order: list[str],
    cases: dict[str, dict[str, Any]],
    run_a: dict[str, dict[str, Any]],
    run_b: dict[str, dict[str, Any]],
    *,
    label_a: str,
    label_b: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_html(
            report,
            order,
            cases,
            run_a,
            run_b,
            label_a=label_a,
            label_b=label_b,
        ),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--label-a", default="baseline")
    parser.add_argument("--label-b", default="candidate")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--html-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        order, cases = load_cases(args.cases)
        expected_ids = set(order)
        run_a = load_complete_run(args.run_a, expected_ids)
        run_b = load_complete_run(args.run_b, expected_ids)
        report, csv_rows = analyze_runs(
            order,
            cases,
            run_a,
            run_b,
            label_a=args.label_a,
            label_b=args.label_b,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    report["inputs"] = {
        "cases": {
            "path": str(args.cases),
            "sha256": file_sha256(args.cases),
        },
        args.label_a: {
            "path": str(args.run_a),
            "sha256": file_sha256(args.run_a),
        },
        args.label_b: {
            "path": str(args.run_b),
            "sha256": file_sha256(args.run_b),
        },
    }
    write_json(args.json_out, report)
    write_csv(args.csv_out, csv_rows)
    if args.html_out:
        write_html(
            args.html_out,
            report,
            order,
            cases,
            run_a,
            run_b,
            label_a=args.label_a,
            label_b=args.label_b,
        )

    summary_a = report["arms"][args.label_a]
    summary_b = report["arms"][args.label_b]
    print(
        f"{args.label_a}: Hit@5="
        f"{summary_a['document']['hit_at_5']['count']}/{len(order)}, "
        f"Any-Required-Gold-Chunk@8="
        f"{summary_a['gold_chunk']['8']['any_count']}/{len(order)}, "
        f"All-Required-Evidence@8="
        f"{summary_a['required_evidence_item']['8']['all_count']}/{len(order)}, "
        f"latency p50={summary_a['latency_ms']['p50']:.3f}ms"
    )
    print(
        f"{args.label_b}: Hit@5="
        f"{summary_b['document']['hit_at_5']['count']}/{len(order)}, "
        f"Any-Required-Gold-Chunk@8="
        f"{summary_b['gold_chunk']['8']['any_count']}/{len(order)}, "
        f"All-Required-Evidence@8="
        f"{summary_b['required_evidence_item']['8']['all_count']}/{len(order)}, "
        f"latency p50={summary_b['latency_ms']['p50']:.3f}ms"
    )
    print(f"JSON: {args.json_out}")
    print(f"CSV: {args.csv_out}")
    if args.html_out:
        print(f"HTML: {args.html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
