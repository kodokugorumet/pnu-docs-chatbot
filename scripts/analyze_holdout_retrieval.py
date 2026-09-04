#!/usr/bin/env python3
"""Fail-closed paired retrieval analysis for the frozen holdout-v2 Core.

This command is deliberately offline: final mode reads the four complete PAR-B,
PAR-CH, C0, and C1 service artifacts and their embedded evaluation traces.  It
never calls the service or an LLM.  The legacy ``retrieval_hit`` field is ignored
because holdout-v2 gold is profile-independent and lives in
``required_claims[].evidence_options``.

Example::

    python3 scripts/analyze_holdout_retrieval.py \
      --cases config/pnu-service-answer-holdout-v2.jsonl \
      --c0 processed/eval/final/c0-core-run1.answers.jsonl \
      --c1 processed/eval/final/c1-core-run1.answers.jsonl \
      --cases-sha256 "$CASES_FILE_SHA256" \
      --c0-sha256 "$C0_ARTIFACT_SHA256" \
      --c1-sha256 "$C1_ARTIFACT_SHA256" \
      --expected-experiment-id final-20260909 \
      --expected-generation-run-id run1 \
      --expected-corpus-revision "$CORPUS_REVISION" \
      --json-out processed/eval/final/retrieval-c0-c1.json \
      --csv-out processed/eval/final/retrieval-c0-c1.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_service_answers import (  # noqa: E402
    ATOMIC_EVIDENCE_MATCHER_VERSION,
    atomic_evidence_hit,
    final_evaluation_contexts,
    validate_atomic_evidence_schema,
)
from holdout_gold import (  # noqa: E402
    CORE_SPLIT,
    load_jsonl,
    sha256_file,
    validate_holdout_records,
)
from immutable_outputs import publish_immutable_texts  # noqa: E402
from service_eval_artifacts import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    build_answer_identity,
    sha256_json,
    validate_answer_record,
)


SCHEMA_VERSION = "pnu.holdout-retrieval-paired-analysis.v1"
FINAL_SCHEMA_VERSION = "pnu.final-holdout-retrieval-analysis.v2"
SOURCE_CUTOFFS = (1, 3, 5)
EVIDENCE_CUTOFFS = (5, 8)
PRESELECTION_CUTOFF = 50
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_SEED = 20_260_914
EXPECTED_CONTEXT_K = 8
EXPECTED_PARSER_PROFILE = "cascade"
EXPECTED_RETRIEVAL_MODE = "bm25"
EXPECTED_CONTEXT_CHUNKS_PER_DOCUMENT = 2
TRACE_STAGES = (
    "raw_bm25",
    "post_retrieval_pool",
    "post_neighbor_expansion",
    "final_contexts",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

METRIC_KINDS = {
    **{f"source_hit_at_{cutoff}": "binary" for cutoff in SOURCE_CUTOFFS},
    **{
        f"evidence_recall_at_{cutoff}": "continuous"
        for cutoff in EVIDENCE_CUTOFFS
    },
    **{
        f"all_evidence_at_{cutoff}": "binary"
        for cutoff in EVIDENCE_CUTOFFS
    },
    "preselection_mrr_at_50": "continuous",
    "candidate_recall_at_50": "continuous",
    "retrieval_timing_ms": "continuous",
}


class RetrievalAnalysisError(ValueError):
    """An integrity, completeness, or experiment-control failure."""


def _required_text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise RetrievalAnalysisError(f"{label} must be a non-empty string")
    return result


def _expected_sha256(value: str, label: str) -> str:
    normalized = str(value or "").strip().casefold()
    if not SHA256_RE.fullmatch(normalized):
        raise RetrievalAnalysisError(
            f"{label} must be exactly 64 lowercase hexadecimal characters"
        )
    return normalized


def _verify_file_sha256(path: Path, expected: str, label: str) -> str:
    wanted = _expected_sha256(expected, label)
    actual = sha256_file(path)
    if actual != wanted:
        raise RetrievalAnalysisError(
            f"{label} mismatch for {path}: expected {wanted}, got {actual}"
        )
    return actual


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_path_separation(
    *, inputs: Sequence[Path], outputs: Sequence[Path]
) -> None:
    """Reject aliases before any output is opened or parent directory is made."""

    input_paths = [_resolved(path) for path in inputs]
    output_paths = [_resolved(path) for path in outputs]
    if len(input_paths) != len(set(input_paths)):
        raise RetrievalAnalysisError("input paths must be distinct")
    if len(output_paths) != len(set(output_paths)):
        raise RetrievalAnalysisError("output paths must be distinct")
    collisions = sorted(set(input_paths).intersection(output_paths))
    if collisions:
        raise RetrievalAnalysisError(
            "input/output path alias is forbidden: "
            + ", ".join(str(path) for path in collisions)
        )
    all_paths = [*inputs, *outputs]
    for left_index, left in enumerate(all_paths):
        if not left.exists():
            continue
        for right in all_paths[left_index + 1 :]:
            if not right.exists():
                continue
            try:
                aliases = left.samefile(right)
            except OSError:
                aliases = False
            if aliases:
                raise RetrievalAnalysisError(
                    f"filesystem alias is forbidden: {left} and {right}"
                )


def load_frozen_core_cases(
    path: Path, *, expected_file_sha256: str
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, str]]:
    file_sha = _verify_file_sha256(path, expected_file_sha256, "cases SHA-256")
    records = load_jsonl(path)
    try:
        validate_holdout_records(records)
    except ValueError as exc:
        raise RetrievalAnalysisError(f"invalid holdout-v2 cases: {exc}") from exc

    canonical_sha = sha256_json(records)
    selected = [record for record in records if record.get("split") == CORE_SPLIT]
    if not selected:
        raise RetrievalAnalysisError("holdout-v2 contains no Core cases")
    order: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    for record in selected:
        case_id = _required_text(record.get("id"), "case id")
        if case_id in cases:
            raise RetrievalAnalysisError(f"duplicate Core case id {case_id!r}")
        try:
            validate_atomic_evidence_schema(record)
        except ValueError as exc:
            raise RetrievalAnalysisError(f"{case_id}: {exc}") from exc
        if not record.get("required_claims"):
            raise RetrievalAnalysisError(
                f"{case_id}: Core retrieval analysis requires required_claims"
            )
        order.append(case_id)
        cases[case_id] = record
    return order, cases, {
        "file_sha256": file_sha,
        "canonical_cases_sha256": canonical_sha,
        "selected_case_ids_sha256": sha256_json(order),
    }


def _fold_identifier(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold()


def gold_document_ids(case: Mapping[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    for claim in case.get("required_claims") or []:
        if not isinstance(claim, Mapping):
            continue
        for option in claim.get("evidence_options") or []:
            if not isinstance(option, Mapping):
                continue
            document_id = _fold_identifier(option.get("document_id"))
            if document_id:
                identifiers.add(document_id)
    if not identifiers:
        raise RetrievalAnalysisError(
            f"{case.get('id')}: no profile-independent gold document IDs"
        )
    return identifiers


def _validate_trace_rows(
    rows: Any,
    *,
    case_id: str,
    stage: str,
    corpus_revision: str,
    exact_length: int | None = None,
    minimum_length: int | None = None,
    maximum_length: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise RetrievalAnalysisError(
            f"{case_id}: evaluation_trace.{stage} must be an array"
        )
    if exact_length is not None and len(rows) != exact_length:
        raise RetrievalAnalysisError(
            f"{case_id}: evaluation_trace.{stage} must contain exactly "
            f"{exact_length} rows, found {len(rows)}"
        )
    if minimum_length is not None and len(rows) < minimum_length:
        raise RetrievalAnalysisError(
            f"{case_id}: evaluation_trace.{stage} must contain at least "
            f"{minimum_length} rows, found {len(rows)}"
        )
    if maximum_length is not None and len(rows) > maximum_length:
        raise RetrievalAnalysisError(
            f"{case_id}: evaluation_trace.{stage} must contain at most "
            f"{maximum_length} rows, found {len(rows)}"
        )

    seen_chunk_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise RetrievalAnalysisError(
                f"{case_id}: evaluation_trace.{stage}[{rank - 1}] must be an object"
            )
        if row.get("rank") != rank:
            raise RetrievalAnalysisError(
                f"{case_id}: evaluation_trace.{stage} rank sequence is invalid "
                f"at position {rank}"
            )
        if row.get("stage") != stage:
            raise RetrievalAnalysisError(
                f"{case_id}: evaluation_trace.{stage}[{rank - 1}] has stage "
                f"{row.get('stage')!r}"
            )
        chunk_id = _required_text(
            row.get("chunk_id"),
            f"{case_id}: evaluation_trace.{stage}[{rank - 1}].chunk_id",
        )
        _required_text(
            row.get("document_id") or row.get("doc_id"),
            f"{case_id}: evaluation_trace.{stage}[{rank - 1}].document_id",
        )
        if chunk_id in seen_chunk_ids:
            raise RetrievalAnalysisError(
                f"{case_id}: evaluation_trace.{stage} repeats chunk_id {chunk_id!r}"
            )
        seen_chunk_ids.add(chunk_id)
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RetrievalAnalysisError(
                f"{case_id}: evaluation_trace.{stage}[{rank - 1}].text "
                "must be non-empty"
            )
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if row.get("text_sha256") != text_sha:
            raise RetrievalAnalysisError(
                f"{case_id}: evaluation_trace.{stage}[{rank - 1}] "
                "text_sha256 mismatch"
            )
        if row.get("corpus_revision") != corpus_revision:
            raise RetrievalAnalysisError(
                f"{case_id}: evaluation_trace.{stage}[{rank - 1}] corpus "
                "revision mismatch"
            )
        if stage == "final_contexts" and row.get("source_number") != rank:
            raise RetrievalAnalysisError(
                f"{case_id}: final_contexts source_number must equal rank {rank}"
            )
        if stage == "raw_bm25":
            retrieval = row.get("retrieval")
            bm25 = retrieval.get("bm25") if isinstance(retrieval, dict) else None
            if not isinstance(bm25, dict):
                raise RetrievalAnalysisError(
                    f"{case_id}: raw_bm25[{rank - 1}] lacks BM25 provenance"
                )
            bm25_rank = bm25.get("rank")
            bm25_score = bm25.get("score")
            if (
                not isinstance(bm25_rank, int)
                or isinstance(bm25_rank, bool)
                or bm25_rank <= 0
                or not isinstance(bm25_score, (int, float))
                or isinstance(bm25_score, bool)
                or not math.isfinite(float(bm25_score))
            ):
                raise RetrievalAnalysisError(
                    f"{case_id}: raw_bm25[{rank - 1}] has invalid BM25 "
                    "rank/score provenance"
                )
        validated.append(row)
    return validated


def _public_provider(provider: str) -> str:
    return "frontier" if provider == "gemini" else provider


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RetrievalAnalysisError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_record_controls(
    record: dict[str, Any],
    case: dict[str, Any],
    *,
    expected_condition: str,
    cases_file_sha256: str,
    canonical_cases_sha256: str,
    selected_case_ids_sha256: str,
    expected_profile: str = EXPECTED_PARSER_PROFILE,
    expected_tuning: bool | None = None,
    expected_revision: str | None = None,
    expected_index_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case_id = str(case["id"])
    if record.get("error") not in (None, False, ""):
        raise RetrievalAnalysisError(f"{case_id}: answer record contains an error")
    try:
        validate_answer_record(record)
    except ValueError as exc:
        raise RetrievalAnalysisError(f"{case_id}: invalid answer record: {exc}") from exc
    if not str(record.get("answer") or "").strip():
        raise RetrievalAnalysisError(f"{case_id}: answer record is incomplete/empty")

    _assert_equal(record.get("case_id"), case_id, f"{case_id}: case_id")
    _assert_equal(
        record.get("case_sha256"), sha256_json(case), f"{case_id}: case_sha256"
    )
    _assert_equal(
        record.get("condition_id"),
        expected_condition,
        f"{case_id}: condition_id",
    )

    collector = record.get("collector_config")
    if not isinstance(collector, dict):
        raise RetrievalAnalysisError(f"{case_id}: collector_config must be an object")
    _assert_equal(
        record.get("collector_config_sha256"),
        sha256_json(collector),
        f"{case_id}: collector_config_sha256",
    )
    if expected_tuning is None:
        expected_tuning = expected_condition == "c1"
    fixed_collector_controls = {
        "cases_sha256": cases_file_sha256,
        "cases_canonical_sha256": canonical_cases_sha256,
        "selected_case_ids_sha256": selected_case_ids_sha256,
        "context_k": EXPECTED_CONTEXT_K,
        "parser_profile": expected_profile,
        "retrieval_mode": EXPECTED_RETRIEVAL_MODE,
        "expected_retrieval_tuning": expected_tuning,
        "expected_context_chunks_per_document": (
            EXPECTED_CONTEXT_CHUNKS_PER_DOCUMENT
        ),
        "eval_trace": True,
        "allow_unpinned": False,
    }
    for key, expected in fixed_collector_controls.items():
        _assert_equal(
            collector.get(key), expected, f"{case_id}: collector_config.{key}"
        )
    corpus_revision = _required_text(
        collector.get("expected_corpus_revision"),
        f"{case_id}: collector_config.expected_corpus_revision",
    )
    if expected_revision is not None:
        _assert_equal(
            corpus_revision, expected_revision,
            f"{case_id}: collector_config.expected_corpus_revision",
        )
    if expected_index_sha256 is not None:
        _assert_equal(
            collector.get("expected_index_sha256"), expected_index_sha256,
            f"{case_id}: collector_config.expected_index_sha256",
        )
    if expected_source_manifest_sha256 is not None:
        _assert_equal(
            collector.get("expected_source_manifest_sha256"),
            expected_source_manifest_sha256,
            f"{case_id}: collector_config.expected_source_manifest_sha256",
        )
    provider = _required_text(
        collector.get("provider"), f"{case_id}: collector_config.provider"
    )
    model = collector.get("model")
    if model is not None and not isinstance(model, str):
        raise RetrievalAnalysisError(
            f"{case_id}: collector_config.model must be a string or null"
        )

    server = collector.get("server_config")
    if not isinstance(server, dict):
        raise RetrievalAnalysisError(
            f"{case_id}: collector_config.server_config must be an object"
        )
    server_controls = {
        "parser_profile": expected_profile,
        "retrieval_mode": EXPECTED_RETRIEVAL_MODE,
        "corpus_revision": corpus_revision,
        "retrieval_tuning": expected_tuning,
        "context_chunks_per_document": EXPECTED_CONTEXT_CHUNKS_PER_DOCUMENT,
        "evaluation_trace_enabled": True,
    }
    for key, expected in server_controls.items():
        _assert_equal(
            server.get(key), expected, f"{case_id}: server_config.{key}"
        )

    request = record.get("request")
    if not isinstance(request, dict):
        raise RetrievalAnalysisError(f"{case_id}: request must be an object")
    request_controls = {
        "question": case.get("query"),
        "top_k": EXPECTED_CONTEXT_K,
        "institution": collector.get("institution"),
        "provider": provider,
        "parser_profile": expected_profile,
        "retrieval_mode": EXPECTED_RETRIEVAL_MODE,
        "eval_trace": True,
    }
    if model is not None:
        request_controls["model"] = model
    if case.get("role") is not None:
        request_controls["role"] = case.get("role")
    for key, expected in request_controls.items():
        _assert_equal(request.get(key), expected, f"{case_id}: request.{key}")

    response_config = record.get("response_config")
    if not isinstance(response_config, dict):
        raise RetrievalAnalysisError(
            f"{case_id}: response_config must be an object"
        )
    response_controls = {
        "institution": collector.get("institution"),
        "parser_profile": expected_profile,
        "retrieval_mode": EXPECTED_RETRIEVAL_MODE,
        "generation_requested": _public_provider(provider),
        "generation_used": _public_provider(provider),
        "generation_model": model,
    }
    for key, expected in response_controls.items():
        _assert_equal(
            response_config.get(key), expected, f"{case_id}: response_config.{key}"
        )

    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise RetrievalAnalysisError(f"{case_id}: generation must be an object")
    generation_controls = {
        "requested": _public_provider(provider),
        "used": _public_provider(provider),
        "model": model,
    }
    for key, expected in generation_controls.items():
        _assert_equal(
            generation.get(key), expected, f"{case_id}: generation.{key}"
        )

    retrieval = record.get("retrieval")
    if not isinstance(retrieval, dict):
        raise RetrievalAnalysisError(f"{case_id}: retrieval must be an object")
    retrieval_controls = {
        "mode": EXPECTED_RETRIEVAL_MODE,
        "service_tuning": expected_tuning,
        "max_chunks_per_document": EXPECTED_CONTEXT_CHUNKS_PER_DOCUMENT,
    }
    for key, expected in retrieval_controls.items():
        _assert_equal(retrieval.get(key), expected, f"{case_id}: retrieval.{key}")

    if expected_index_sha256 is not None or expected_source_manifest_sha256 is not None:
        profile_index = server.get("profile_index")
        if not isinstance(profile_index, dict):
            raise RetrievalAnalysisError(f"{case_id}: server profile_index missing")
        if expected_index_sha256 is not None:
            _assert_equal(
                profile_index.get("sha256"), expected_index_sha256,
                f"{case_id}: server profile index SHA-256",
            )
        if expected_source_manifest_sha256 is not None:
            _assert_equal(
                profile_index.get("source_manifest_sha256"),
                expected_source_manifest_sha256,
                f"{case_id}: server source manifest SHA-256",
            )

    if schedule is not None:
        calls = {
            (entry["condition_id"], entry["case_id"]): entry
            for entry in schedule.get("calls", [])
        }
        entry = calls.get((expected_condition, case_id))
        if entry is None:
            raise RetrievalAnalysisError(f"{case_id}: missing scheduled lane call")
        collection = record.get("collection_schedule")
        if not isinstance(collection, dict):
            raise RetrievalAnalysisError(f"{case_id}: collection_schedule missing")
        for key, expected in {
            "collection_purpose": "final_retrieval",
            "schedule_id": schedule.get("schedule_id"),
            "schedule_sha256": schedule.get("schedule_sha256"),
            "call_order": entry.get("call_order"),
            "artifact_id": entry.get("artifact_id"),
            "family_id": entry.get("family_id"),
            "condition_position": entry.get("condition_position"),
        }.items():
            _assert_equal(collection.get(key), expected, f"{case_id}: schedule.{key}")
        authorization = collector.get("final_authorization")
        if not isinstance(authorization, dict):
            raise RetrievalAnalysisError(f"{case_id}: final_authorization missing")
        for key, expected in {
            "collection_purpose": "final_retrieval",
            "schedule_id": schedule.get("schedule_id"),
            "schedule_sha256": schedule.get("schedule_sha256"),
            "cases_sha256": schedule.get("cases_sha256"),
            "cases_canonical_sha256": schedule.get("cases_canonical_sha256"),
            "shared_source_manifest_sha256": schedule.get(
                "shared_source_manifest_sha256"
            ),
            "artifact_id": entry.get("artifact_id"),
            "expected_case_count": len(schedule.get("calls", [])) // 4,
            "expected_case_ids_sha256": schedule.get(
                "selected_case_ids_sha256"
            ),
        }.items():
            _assert_equal(
                authorization.get(key), expected,
                f"{case_id}: final_authorization.{key}",
            )
        binding = authorization.get("gate_input_binding")
        if not isinstance(binding, dict) or authorization.get(
            "gate_input_binding_sha256"
        ) != sha256_json(binding):
            raise RetrievalAnalysisError(f"{case_id}: gate binding invalid")
        _assert_equal(
            entry.get("case_sha256"), sha256_json(case),
            f"{case_id}: scheduled case SHA-256",
        )
        gate_hash = authorization.get("gate_summary_sha256")
        if not isinstance(gate_hash, str) or not SHA256_RE.fullmatch(gate_hash):
            raise RetrievalAnalysisError(f"{case_id}: gate summary SHA invalid")

    trace = record.get("evaluation_trace")
    if not isinstance(trace, dict) or trace.get("schema_version") != 1:
        raise RetrievalAnalysisError(
            f"{case_id}: evaluation_trace schema_version must equal 1"
        )
    stages = trace.get("retrieval_stages")
    if not isinstance(stages, dict):
        raise RetrievalAnalysisError(
            f"{case_id}: evaluation_trace.retrieval_stages must be an object"
        )
    validated_stages: dict[str, list[dict[str, Any]]] = {}
    for stage in TRACE_STAGES:
        validated_stages[stage] = _validate_trace_rows(
            stages.get(stage),
            case_id=case_id,
            stage=stage,
            corpus_revision=corpus_revision,
            maximum_length=EXPECTED_CONTEXT_K if stage == "final_contexts" else None,
        )

    final_rows = validated_stages["final_contexts"]
    _assert_equal(
        retrieval.get("result_count"),
        len(final_rows),
        f"{case_id}: retrieval.result_count",
    )
    expanded_rows = validated_stages["post_neighbor_expansion"]
    cursor = 0
    for final_rank, final_row in enumerate(final_rows, start=1):
        matched_index = next(
            (
                index
                for index in range(cursor, len(expanded_rows))
                if expanded_rows[index].get("chunk_id") == final_row.get("chunk_id")
            ),
            None,
        )
        if matched_index is None:
            raise RetrievalAnalysisError(
                f"{case_id}: final_contexts is not an ordered subsequence of "
                "post_neighbor_expansion"
            )
        expanded = expanded_rows[matched_index]
        for key in ("document_id", "text_sha256", "corpus_revision"):
            _assert_equal(
                final_row.get(key),
                expanded.get(key),
                f"{case_id}: final_contexts[{final_rank - 1}].{key} provenance",
            )
        cursor = matched_index + 1

    timing = trace.get("timing_ms")
    if not isinstance(timing, dict):
        raise RetrievalAnalysisError(f"{case_id}: evaluation_trace.timing_ms missing")
    retrieval_timing = timing.get("retrieval")
    if (
        not isinstance(retrieval_timing, (int, float))
        or isinstance(retrieval_timing, bool)
        or not math.isfinite(float(retrieval_timing))
        or retrieval_timing < 0
    ):
        raise RetrievalAnalysisError(f"{case_id}: invalid retrieval timing")

    sources = record.get("sources")
    if not isinstance(sources, list):
        raise RetrievalAnalysisError(f"{case_id}: sources must be an array")
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise RetrievalAnalysisError(
                f"{case_id}: sources[{source_index}] must be an object"
            )
        _assert_equal(
            source.get("corpus_revision"),
            corpus_revision,
            f"{case_id}: sources[{source_index}].corpus_revision",
        )
    final_contexts = final_evaluation_contexts(
        {"results": sources, "evaluation_trace": trace}
    )
    if final_contexts is None:
        raise RetrievalAnalysisError(
            f"{case_id}: sources and final_contexts failed integrity binding"
        )

    recomputed_at_k = {
        str(cutoff): atomic_evidence_hit(final_contexts, case, k=cutoff)
        for cutoff in EVIDENCE_CUTOFFS
    }
    stored_at_k = record.get("atomic_evidence_at_k")
    if not isinstance(stored_at_k, dict):
        raise RetrievalAnalysisError(
            f"{case_id}: atomic_evidence_at_k must be an object"
        )
    for cutoff in EVIDENCE_CUTOFFS:
        key = str(cutoff)
        _assert_equal(
            stored_at_k.get(key),
            recomputed_at_k[key],
            f"{case_id}: atomic_evidence_at_k[{key!r}]",
        )

    control_signature = {
        "experiment_id": _required_text(
            record.get("experiment_id"), f"{case_id}: experiment_id"
        ),
        "generation_run_id": _required_text(
            record.get("generation_run_id"), f"{case_id}: generation_run_id"
        ),
        "corpus_revision": corpus_revision,
        "provider": provider,
        "model": model,
        "institution": collector.get("institution"),
        "parser_profile": expected_profile,
        "retrieval_mode": EXPECTED_RETRIEVAL_MODE,
        "context_k": EXPECTED_CONTEXT_K,
        "context_chunks_per_document": EXPECTED_CONTEXT_CHUNKS_PER_DOCUMENT,
        "cases_sha256": collector.get("cases_sha256"),
        "cases_canonical_sha256": collector.get("cases_canonical_sha256"),
        "selected_case_ids_sha256": collector.get(
            "selected_case_ids_sha256"
        ),
        "index_sha256": collector.get("expected_index_sha256"),
        "collector_config_sha256": record.get("collector_config_sha256"),
        "gate_summary_sha256": (
            (collector.get("final_authorization") or {}).get("gate_summary_sha256")
            if schedule is not None else None
        ),
        "schedule_id": schedule.get("schedule_id") if schedule is not None else None,
        "schedule_sha256": schedule.get("schedule_sha256") if schedule is not None else None,
        "source_manifest_sha256": expected_source_manifest_sha256,
    }
    return (
        validated_stages["final_contexts"],
        validated_stages["raw_bm25"][:PRESELECTION_CUTOFF],
        {**control_signature, "retrieval_timing_ms": float(retrieval_timing)},
    )


def _source_hit(
    contexts: Sequence[Mapping[str, Any]], gold_ids: set[str], cutoff: int
) -> bool:
    return any(
        _fold_identifier(row.get("document_id") or row.get("doc_id")) in gold_ids
        for row in contexts[:cutoff]
    )


def compute_case_metrics(
    *,
    case: dict[str, Any],
    final_contexts: list[dict[str, Any]],
    raw_bm25_at_50: list[dict[str, Any]],
    retrieval_timing_ms: float,
) -> dict[str, float | bool]:
    gold_ids = gold_document_ids(case)
    values: dict[str, float | bool] = {
        f"source_hit_at_{cutoff}": _source_hit(
            final_contexts, gold_ids, cutoff
        )
        for cutoff in SOURCE_CUTOFFS
    }
    for cutoff in EVIDENCE_CUTOFFS:
        atomic = atomic_evidence_hit(final_contexts, case, k=cutoff)
        values[f"evidence_recall_at_{cutoff}"] = float(atomic["recall"])
        values[f"all_evidence_at_{cutoff}"] = bool(atomic["all_matched"])

    first_gold_rank = next(
        (
            rank
            for rank, row in enumerate(raw_bm25_at_50, start=1)
            if _fold_identifier(row.get("document_id") or row.get("doc_id"))
            in gold_ids
        ),
        None,
    )
    values["preselection_mrr_at_50"] = (
        1.0 / first_gold_rank if first_gold_rank is not None else 0.0
    )
    candidate_atomic = atomic_evidence_hit(
        raw_bm25_at_50, case, k=PRESELECTION_CUTOFF
    )
    values["candidate_recall_at_50"] = float(candidate_atomic["recall"])
    values["retrieval_timing_ms"] = retrieval_timing_ms
    return values


def load_complete_condition(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_condition: str,
    order: Sequence[str],
    cases: Mapping[str, dict[str, Any]],
    case_hashes: Mapping[str, str],
    expected_profile: str = EXPECTED_PARSER_PROFILE,
    expected_tuning: bool | None = None,
    expected_revision: str | None = None,
    expected_index_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, dict[str, float | bool]],
    dict[str, Any],
    str,
]:
    artifact_sha = _verify_file_sha256(
        path, expected_file_sha256, f"{expected_condition} artifact SHA-256"
    )
    records = load_jsonl(path)
    if not records:
        raise RetrievalAnalysisError(f"{path}: answer artifact is empty")

    by_case: dict[str, dict[str, Any]] = {}
    seen_answer_ids: set[str] = set()
    for index, record in enumerate(records):
        case_id = _required_text(
            record.get("case_id"), f"{path}:{index + 1}: case_id"
        )
        if case_id in by_case:
            raise RetrievalAnalysisError(f"{path}: duplicate case_id {case_id!r}")
        answer_id = _required_text(
            record.get("answer_id"), f"{path}:{index + 1}: answer_id"
        )
        if answer_id in seen_answer_ids:
            raise RetrievalAnalysisError(f"{path}: duplicate answer_id {answer_id!r}")
        seen_answer_ids.add(answer_id)
        by_case[case_id] = record

    expected_ids = set(order)
    actual_ids = set(by_case)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise RetrievalAnalysisError(
            f"{path}: incomplete Core case set; missing={missing or 'none'}, "
            f"extra={extra or 'none'}"
        )

    metrics: dict[str, dict[str, float | bool]] = {}
    signatures: list[dict[str, Any]] = []
    for case_id in order:
        final_contexts, raw_bm25_at_50, signature = _validate_record_controls(
            by_case[case_id],
            cases[case_id],
            expected_condition=expected_condition,
            cases_file_sha256=case_hashes["file_sha256"],
            canonical_cases_sha256=case_hashes["canonical_cases_sha256"],
            selected_case_ids_sha256=case_hashes["selected_case_ids_sha256"],
            expected_profile=expected_profile,
            expected_tuning=expected_tuning,
            expected_revision=expected_revision,
            expected_index_sha256=expected_index_sha256,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            schedule=schedule,
        )
        signatures.append(signature)
        metrics[case_id] = compute_case_metrics(
            case=cases[case_id],
            final_contexts=final_contexts,
            raw_bm25_at_50=raw_bm25_at_50,
            retrieval_timing_ms=float(signature["retrieval_timing_ms"]),
        )

    stable_signatures = [
        {key: value for key, value in signature.items() if key != "retrieval_timing_ms"}
        for signature in signatures
    ]
    first_signature = stable_signatures[0]
    for case_id, signature in zip(order[1:], stable_signatures[1:]):
        if signature != first_signature:
            differing = sorted(
                key
                for key in set(first_signature).union(signature)
                if first_signature.get(key) != signature.get(key)
            )
            raise RetrievalAnalysisError(
                f"{path}: mixed run/control values at {case_id}: {differing}"
            )
    return metrics, first_signature, artifact_sha


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise RetrievalAnalysisError("cannot compute a quantile of no values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_family_bootstrap(
    deltas: Mapping[str, float],
    families: Mapping[str, str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        raise RetrievalAnalysisError("bootstrap iterations must be positive")
    if not deltas:
        raise RetrievalAnalysisError("paired bootstrap requires at least one case")
    clusters: dict[str, list[str]] = defaultdict(list)
    for case_id in deltas:
        family_id = _required_text(families.get(case_id), f"{case_id}: family_id")
        clusters[family_id].append(case_id)
    family_ids = sorted(clusters)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_cases: list[str] = []
        for _family_slot in family_ids:
            sampled_cases.extend(clusters[rng.choice(family_ids)])
        samples.append(mean(deltas[case_id] for case_id in sampled_cases))
    return {
        "method": "paired cluster bootstrap by frozen family_id",
        "unit": "question family",
        "clusters": len(family_ids),
        "questions": len(deltas),
        "iterations": iterations,
        "seed": seed,
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
    }


def exact_mcnemar(c0: Sequence[bool], c1: Sequence[bool]) -> dict[str, Any]:
    if len(c0) != len(c1) or not c0:
        raise RetrievalAnalysisError("McNemar requires non-empty paired values")
    both_pass = sum(a and b for a, b in zip(c0, c1))
    c0_only = sum(a and not b for a, b in zip(c0, c1))
    c1_only = sum((not a) and b for a, b in zip(c0, c1))
    neither = len(c0) - both_pass - c0_only - c1_only
    discordant = c0_only + c1_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(min(c0_only, c1_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "method": "exact two-sided McNemar (binomial discordant-pair test)",
        "both_pass": both_pass,
        "c0_only": c0_only,
        "c1_only": c1_only,
        "neither": neither,
        "discordant": discordant,
        "p_value": p_value,
    }


def summarize_condition(
    order: Sequence[str], metrics: Mapping[str, Mapping[str, float | bool]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, kind in METRIC_KINDS.items():
        values = [float(metrics[case_id][metric]) for case_id in order]
        summary: dict[str, Any] = {
            "kind": kind,
            "questions": len(values),
            "mean": mean(values),
        }
        if kind == "binary":
            summary["hits"] = int(sum(values))
        if metric == "retrieval_timing_ms":
            summary["p50"] = _quantile(values, 0.50)
            summary["p95"] = _quantile(values, 0.95)
        result[metric] = summary
    return result


def compare_metrics(
    order: Sequence[str],
    metrics_c0: Mapping[str, Mapping[str, float | bool]],
    metrics_c1: Mapping[str, Mapping[str, float | bool]],
    families: Mapping[str, str],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric, kind in METRIC_KINDS.items():
        values_c0 = [float(metrics_c0[case_id][metric]) for case_id in order]
        values_c1 = [float(metrics_c1[case_id][metric]) for case_id in order]
        deltas = {
            case_id: float(metrics_c1[case_id][metric])
            - float(metrics_c0[case_id][metric])
            for case_id in order
        }
        result: dict[str, Any] = {
            "kind": kind,
            "questions": len(order),
            "c0_mean": mean(values_c0),
            "c1_mean": mean(values_c1),
            "mean_delta_c1_minus_c0": mean(deltas.values()),
            "gained_case_ids": [
                case_id for case_id in order if deltas[case_id] > 0
            ],
            "lost_case_ids": [
                case_id for case_id in order if deltas[case_id] < 0
            ],
            "tied_case_ids": [
                case_id for case_id in order if deltas[case_id] == 0
            ],
            "paired_family_bootstrap_95ci": paired_family_bootstrap(
                deltas,
                families,
                iterations=bootstrap_iterations,
                seed=seed,
            ),
        }
        if kind == "binary":
            result["exact_mcnemar_two_sided"] = exact_mcnemar(
                [bool(metrics_c0[case_id][metric]) for case_id in order],
                [bool(metrics_c1[case_id][metric]) for case_id in order],
            )
        comparison[metric] = result
    return comparison


def build_strata(
    order: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    metrics_c0: Mapping[str, Mapping[str, float | bool]],
    metrics_c1: Mapping[str, Mapping[str, float | bool]],
    *,
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for case_id in order:
        value = _required_text(cases[case_id].get(field), f"{case_id}: {field}")
        groups[value].append(case_id)
    result: dict[str, Any] = {}
    for value in sorted(groups):
        case_ids = groups[value]
        metrics: dict[str, Any] = {}
        for metric, kind in METRIC_KINDS.items():
            c0_values = [float(metrics_c0[case_id][metric]) for case_id in case_ids]
            c1_values = [float(metrics_c1[case_id][metric]) for case_id in case_ids]
            metrics[metric] = {
                "kind": kind,
                "c0_mean": mean(c0_values),
                "c1_mean": mean(c1_values),
                "mean_delta_c1_minus_c0": mean(
                    right - left for left, right in zip(c0_values, c1_values)
                ),
            }
        result[value] = {
            "questions": len(case_ids),
            "case_ids": case_ids,
            "metrics": metrics,
        }
    return result


def analyze_pair(
    *,
    order: Sequence[str],
    cases: Mapping[str, dict[str, Any]],
    metrics_c0: Mapping[str, Mapping[str, float | bool]],
    metrics_c1: Mapping[str, Mapping[str, float | bool]],
    controls: Mapping[str, Any],
    input_metadata: Mapping[str, Any],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    families = {
        case_id: _required_text(cases[case_id].get("family_id"), "family_id")
        for case_id in order
    }
    case_rows: list[dict[str, Any]] = []
    for case_id in order:
        c0_values = dict(metrics_c0[case_id])
        c1_values = dict(metrics_c1[case_id])
        case_rows.append(
            {
                "case_id": case_id,
                "family_id": families[case_id],
                "category": cases[case_id].get("category"),
                "difficulty_type": cases[case_id].get("difficulty_type"),
                "c0": c0_values,
                "c1": c1_values,
                "delta_c1_minus_c0": {
                    metric: float(c1_values[metric]) - float(c0_values[metric])
                    for metric in METRIC_KINDS
                },
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_unit": "Core question family; one deterministic retrieval run",
        "metric_definitions": {
            "source_hit_at_k": (
                "any profile-independent gold document_id in the first k "
                "actual final-context positions"
            ),
            "evidence_recall_at_k": (
                "fraction of required atomic claims matched by document_id plus "
                "exact/fuzzy quote or table relation in the first k final contexts"
            ),
            "all_evidence_at_k": (
                "all required atomic claims matched in the first k final contexts"
            ),
            "preselection_mrr_at_50": (
                "reciprocal rank of the first profile-independent gold document_id "
                "in evaluation_trace.raw_bm25[:50], else zero"
            ),
            "candidate_recall_at_50": (
                "fraction of required atomic claims evidence-matched in "
                "evaluation_trace.raw_bm25[:50]"
            ),
            "atomic_evidence_matcher_version": ATOMIC_EVIDENCE_MATCHER_VERSION,
        },
        "analysis_config": {
            "split": CORE_SPLIT,
            "source_cutoffs": list(SOURCE_CUTOFFS),
            "evidence_cutoffs": list(EVIDENCE_CUTOFFS),
            "preselection_cutoff": PRESELECTION_CUTOFF,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": seed,
        },
        "inputs": dict(input_metadata),
        "controls": dict(controls),
        "questions": len(order),
        "condition_summaries": {
            "c0": summarize_condition(order, metrics_c0),
            "c1": summarize_condition(order, metrics_c1),
        },
        "paired_comparison": compare_metrics(
            order,
            metrics_c0,
            metrics_c1,
            families,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        ),
        "strata": {
            "category": build_strata(
                order,
                cases,
                metrics_c0,
                metrics_c1,
                field="category",
            ),
            "difficulty_type": build_strata(
                order,
                cases,
                metrics_c0,
                metrics_c1,
                field="difficulty_type",
            ),
        },
        "cases": case_rows,
    }
    result["analysis_sha256"] = sha256_json(result)
    return result


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm family-wise adjusted p-values with monotonic step-down values."""

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(raw)))
        adjusted[name] = running
    return adjusted


def analyze_four_lanes(
    *,
    order: Sequence[str],
    cases: Mapping[str, dict[str, Any]],
    lane_metrics: Mapping[str, Mapping[str, Mapping[str, float | bool]]],
    lane_controls: Mapping[str, Mapping[str, Any]],
    input_metadata: Mapping[str, Any],
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    families = {case_id: str(cases[case_id]["family_id"]) for case_id in order}
    primary = compare_metrics(
        order, lane_metrics["c0"], lane_metrics["c1"], families,
        bootstrap_iterations=bootstrap_iterations, seed=seed,
    )
    parser_pairs = {
        "par-b_vs_par-ch": ("par-b", "par-ch"),
        "par-b_vs_c1": ("par-b", "c1"),
        "par-ch_vs_c1": ("par-ch", "c1"),
    }
    exploratory = {
        name: compare_metrics(
            order, lane_metrics[left], lane_metrics[right], families,
            bootstrap_iterations=bootstrap_iterations, seed=seed,
        )
        for name, (left, right) in parser_pairs.items()
    }
    for metric, kind in METRIC_KINDS.items():
        if kind != "binary":
            continue
        adjusted = holm_adjust(
            {
                name: comparison[metric]["exact_mcnemar_two_sided"]["p_value"]
                for name, comparison in exploratory.items()
            }
        )
        for name, value in adjusted.items():
            exploratory[name][metric]["holm_adjusted_p_value"] = value

    gate_hashes = {controls.get("gate_summary_sha256") for controls in lane_controls.values()}
    if len(gate_hashes) != 1 or None in gate_hashes:
        raise RetrievalAnalysisError("four lanes do not share one final gate summary")
    case_rows = []
    for case_id in order:
        case_rows.append(
            {
                "case_id": case_id,
                "family_id": families[case_id],
                "category": cases[case_id].get("category"),
                "difficulty_type": cases[case_id].get("difficulty_type"),
                "lanes": {lane: dict(lane_metrics[lane][case_id]) for lane in lane_metrics},
                "primary_delta_c1_minus_c0": {
                    metric: float(lane_metrics["c1"][case_id][metric])
                    - float(lane_metrics["c0"][case_id][metric])
                    for metric in METRIC_KINDS
                },
            }
        )
    result = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "analysis_unit": "27 paired Core question families",
        "metric_definitions": {
            "source_identity": "profile-independent evidence_options.document_id; never chunk_id",
            "atomic_matcher_version": ATOMIC_EVIDENCE_MATCHER_VERSION,
            "raw_candidate_scope": "all raw_bm25 lengths accepted; metrics use only [:50]",
            "final_context_scope": "0..8 contexts; ordered subsequence of post_neighbor_expansion",
        },
        "analysis_config": {
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": seed,
            "primary_contrast": "c0_vs_c1",
            "parser_contrasts": "exploratory_with_Holm_for_binary_McNemar",
        },
        "inputs": dict(input_metadata),
        "controls": {lane: dict(value) for lane, value in lane_controls.items()},
        "questions": len(order),
        "condition_summaries": {
            lane: summarize_condition(order, metrics) for lane, metrics in lane_metrics.items()
        },
        "primary_c0_vs_c1": primary,
        "exploratory_parser_contrasts": exploratory,
        "strata": {
            "category": build_strata(order, cases, lane_metrics["c0"], lane_metrics["c1"], field="category"),
            "difficulty_type": build_strata(order, cases, lane_metrics["c0"], lane_metrics["c1"], field="difficulty_type"),
        },
        "cases": case_rows,
    }
    result["analysis_sha256"] = sha256_json(result)
    return result


def _cross_condition_controls(
    c0: Mapping[str, Any], c1: Mapping[str, Any]
) -> dict[str, Any]:
    comparable_keys = (
        "experiment_id",
        "generation_run_id",
        "cases_sha256",
        "cases_canonical_sha256",
        "selected_case_ids_sha256",
        "corpus_revision",
        "index_sha256",
        "provider",
        "model",
        "institution",
        "parser_profile",
        "retrieval_mode",
        "context_k",
        "context_chunks_per_document",
        "gate_summary_sha256",
    )
    mismatches = [key for key in comparable_keys if c0.get(key) != c1.get(key)]
    if mismatches:
        raise RetrievalAnalysisError(
            "C0/C1 controls are not paired: " + ", ".join(mismatches)
        )
    return {
        key: c0.get(key)
        for key in comparable_keys
    } | {
        "c0_retrieval_tuning": False,
        "c1_retrieval_tuning": True,
        "c0_collector_config_sha256": c0.get("collector_config_sha256"),
        "c1_collector_config_sha256": c1.get("collector_config_sha256"),
    }


def write_case_csv(path: Path, cases: Iterable[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "case_id": case["case_id"],
            "family_id": case["family_id"],
            "category": case.get("category"),
            "difficulty_type": case.get("difficulty_type"),
        }
        for metric in METRIC_KINDS:
            row[f"c0_{metric}"] = case["c0"][metric]
            row[f"c1_{metric}"] = case["c1"][metric]
            row[f"delta_{metric}"] = case["delta_c1_minus_c0"][metric]
        rows.append(row)
    if not rows:
        raise RetrievalAnalysisError("cannot write an empty case CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _four_lane_csv(cases: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for case in cases:
        row = {
            "case_id": case["case_id"], "family_id": case["family_id"],
            "category": case.get("category"),
            "difficulty_type": case.get("difficulty_type"),
        }
        for lane, metrics in case["lanes"].items():
            for metric, value in metrics.items():
                row[f"{lane}_{metric}"] = value
        for metric, value in case["primary_delta_c1_minus_c0"].items():
            row[f"c1_minus_c0_{metric}"] = value
        rows.append(row)
    if not rows:
        raise RetrievalAnalysisError("cannot serialize empty final retrieval CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_final_bundle(
    *, json_path: Path, csv_path: Path, manifest_path: Path,
    report: Mapping[str, Any], schedule: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = (json_path, csv_path, manifest_path)
    validate_path_separation(inputs=(), outputs=outputs)
    json_text = json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    csv_text = _four_lane_csv(report["cases"])
    json_bytes = json_text.encode("utf-8")
    csv_bytes = csv_text.encode("utf-8")
    manifest = {
        "schema_version": "pnu.final-retrieval-analysis-completion.v1",
        "complete": True,
        "schedule_id": schedule["schedule_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "analysis_sha256": report["analysis_sha256"],
        "analyzer_sha256": sha256_file(Path(__file__)),
        "matcher_version": ATOMIC_EVIDENCE_MATCHER_VERSION,
        "matcher_sha256": sha256_file(
            REPO_ROOT / "scripts" / "evaluate_service_answers.py"
        ),
        "outputs": {
            "json": {"path": str(json_path), "sha256": _sha256_bytes(json_bytes)},
            "csv": {"path": str(csv_path), "sha256": _sha256_bytes(csv_bytes)},
        },
        "output_publication": {
            "authoritative_completion_artifact": "manifest",
            "required_companion_artifacts": ["json", "csv"],
            "contract": (
                "The immutable completion manifest is published only after "
                "the JSON and CSV companions are durably published."
            ),
        },
    }
    manifest_text = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    try:
        publish_immutable_texts(
            {
                json_path: json_text,
                csv_path: csv_text,
                manifest_path: manifest_text,
            },
            authoritative_path=manifest_path,
        )
    except (OSError, ValueError) as exc:
        raise RetrievalAnalysisError(str(exc)) from exc
    return manifest


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--c0", type=Path, required=True)
    parser.add_argument("--c1", type=Path, required=True)
    parser.add_argument("--par-b", type=Path)
    parser.add_argument("--par-ch", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--completion-manifest", type=Path)
    parser.add_argument(
        "--cases-sha256",
        required=True,
        help="expected byte-level SHA-256 of --cases",
    )
    parser.add_argument(
        "--c0-sha256",
        required=True,
        help="expected byte-level SHA-256 of --c0",
    )
    parser.add_argument(
        "--c1-sha256",
        required=True,
        help="expected byte-level SHA-256 of --c1",
    )
    parser.add_argument("--par-b-sha256")
    parser.add_argument("--par-ch-sha256")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument(
        "--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-experiment-id", required=True)
    parser.add_argument("--expected-generation-run-id", required=True)
    parser.add_argument("--expected-corpus-revision")
    return parser


def _validate_final_schedule(
    schedule_path: Path,
    *,
    cases_path: Path,
    case_hashes: Mapping[str, str],
    order: Sequence[str],
    lane_paths: Mapping[str, Path],
) -> dict[str, Any]:
    from run_final_retrieval_schedule import (  # noqa: PLC0415
        EXPECTED_CALL_COUNT,
        LANE_IDS,
        audit_artifacts,
        load_schedule,
    )

    schedule = load_schedule(schedule_path)
    _assert_equal(
        Path(schedule.get("cases_path", "")).resolve(), cases_path.resolve(),
        "schedule cases_path",
    )
    _assert_equal(schedule.get("cases_sha256"), case_hashes["file_sha256"], "schedule cases SHA")
    _assert_equal(
        schedule.get("cases_canonical_sha256"),
        case_hashes["canonical_cases_sha256"], "schedule canonical cases SHA",
    )
    _assert_equal(
        schedule.get("selected_case_ids_sha256"),
        case_hashes["selected_case_ids_sha256"], "schedule selected IDs SHA",
    )
    artifacts = {item.get("condition_id"): item for item in schedule.get("artifacts", [])}
    if set(artifacts) != set(LANE_IDS):
        raise RetrievalAnalysisError("schedule must contain exactly four lane artifacts")
    for lane in LANE_IDS:
        _assert_equal(
            Path(artifacts[lane].get("answers_path", "")).resolve(),
            lane_paths[lane].resolve(), f"schedule {lane} answers path",
        )
    _assert_equal(
        schedule["lanes"]["c0"]["index"]["sha256"],
        schedule["lanes"]["c1"]["index"]["sha256"],
        "schedule C0/C1 cascade index SHA-256",
    )
    calls = schedule.get("calls")
    expected_pairs = [(case_id, lane) for case_id in order for lane in LANE_IDS]
    actual_pairs = []
    for expected_order, entry in enumerate(calls or [], 1):
        _assert_equal(entry.get("call_order"), expected_order, "schedule call_order")
        actual_pairs.append((entry.get("case_id"), entry.get("condition_id")))
    if len(actual_pairs) != EXPECTED_CALL_COUNT or actual_pairs != expected_pairs:
        raise RetrievalAnalysisError("schedule call order is not frozen case-major 27x4")
    try:
        audit = audit_artifacts(schedule)
    except (OSError, ValueError) as exc:
        raise RetrievalAnalysisError(
            f"final retrieval schedule artifact audit failed: {exc}"
        ) from exc
    if (
        audit.get("complete") is not True
        or audit.get("completed_call_count") != EXPECTED_CALL_COUNT
        or audit.get("total_call_count") != EXPECTED_CALL_COUNT
    ):
        raise RetrievalAnalysisError(
            "final retrieval schedule is incomplete or contains uncertain slots"
        )
    schedule = dict(schedule)
    schedule["_verified_artifact_audit"] = audit
    return schedule


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.bootstrap <= 0:
            raise RetrievalAnalysisError("--bootstrap must be positive")
        validate_path_separation(
            inputs=(args.cases, args.c0, args.c1),
            outputs=(args.json_out, args.csv_out),
        )
        order, cases, case_hashes = load_frozen_core_cases(
            args.cases, expected_file_sha256=args.cases_sha256
        )
        final_values = (
            args.schedule, args.par_b, args.par_ch, args.par_b_sha256,
            args.par_ch_sha256, args.completion_manifest,
        )
        if any(value is not None for value in final_values):
            if not all(value is not None for value in final_values):
                raise RetrievalAnalysisError(
                    "final four-lane mode requires --schedule, --par-b/--par-ch, "
                    "both byte SHA values, and --completion-manifest"
                )
            lane_paths = {
                "par-b": args.par_b, "par-ch": args.par_ch,
                "c0": args.c0, "c1": args.c1,
            }
            validate_path_separation(
                inputs=(args.cases, args.schedule, *lane_paths.values()),
                outputs=(args.json_out, args.csv_out, args.completion_manifest),
            )
            schedule = _validate_final_schedule(
                args.schedule, cases_path=args.cases, case_hashes=case_hashes,
                order=order, lane_paths=lane_paths,
            )
            _assert_equal(schedule.get("experiment_id"), args.expected_experiment_id, "schedule experiment")
            run_ids = {entry.get("generation_run_id") for entry in schedule["calls"]}
            _assert_equal(run_ids, {args.expected_generation_run_id}, "schedule run")
            lane_shas = {
                "par-b": args.par_b_sha256, "par-ch": args.par_ch_sha256,
                "c0": args.c0_sha256, "c1": args.c1_sha256,
            }
            lane_metrics: dict[str, Any] = {}
            lane_controls: dict[str, Any] = {}
            inputs: dict[str, Any] = {
                "cases": {"path": str(args.cases), **case_hashes},
                "schedule": {
                    "path": str(args.schedule),
                    "file_sha256": sha256_file(args.schedule),
                    "schedule_id": schedule["schedule_id"],
                    "schedule_sha256": schedule["schedule_sha256"],
                    "artifact_audit": schedule["_verified_artifact_audit"],
                },
                "analyzer_sha256": sha256_file(Path(__file__)),
            }
            for lane in ("par-b", "par-ch", "c0", "c1"):
                lane_pin = schedule["lanes"][lane]
                metrics, controls, artifact_sha = load_complete_condition(
                    lane_paths[lane], expected_file_sha256=lane_shas[lane],
                    expected_condition=lane, order=order, cases=cases,
                    case_hashes=case_hashes,
                    expected_profile=lane_pin["parser_profile"],
                    expected_tuning=lane_pin["expected_retrieval_tuning"],
                    expected_revision=lane_pin["corpus_revision"],
                    expected_index_sha256=lane_pin["index"]["sha256"],
                    expected_source_manifest_sha256=schedule[
                        "shared_source_manifest_sha256"
                    ], schedule=schedule,
                )
                lane_metrics[lane] = metrics
                lane_controls[lane] = controls
                inputs[lane] = {"path": str(lane_paths[lane]), "file_sha256": artifact_sha}
            _cross_condition_controls(lane_controls["c0"], lane_controls["c1"])
            result = analyze_four_lanes(
                order=order, cases=cases, lane_metrics=lane_metrics,
                lane_controls=lane_controls, input_metadata=inputs,
                bootstrap_iterations=args.bootstrap, seed=args.seed,
            )
            write_final_bundle(
                json_path=args.json_out, csv_path=args.csv_out,
                manifest_path=args.completion_manifest, report=result,
                schedule=schedule,
            )
            print(f"four-lane final retrieval analysis complete: n={len(order)}")
            return 0
        if not args.expected_corpus_revision:
            raise RetrievalAnalysisError(
                "legacy two-lane mode requires --expected-corpus-revision"
            )
        metrics_c0, controls_c0, c0_sha = load_complete_condition(
            args.c0,
            expected_file_sha256=args.c0_sha256,
            expected_condition="c0",
            order=order,
            cases=cases,
            case_hashes=case_hashes,
        )
        metrics_c1, controls_c1, c1_sha = load_complete_condition(
            args.c1,
            expected_file_sha256=args.c1_sha256,
            expected_condition="c1",
            order=order,
            cases=cases,
            case_hashes=case_hashes,
        )
        controls = _cross_condition_controls(controls_c0, controls_c1)
        expected_controls = {
            "experiment_id": args.expected_experiment_id,
            "generation_run_id": args.expected_generation_run_id,
            "corpus_revision": args.expected_corpus_revision,
        }
        for key, expected in expected_controls.items():
            _assert_equal(controls.get(key), expected, f"paired controls.{key}")
        result = analyze_pair(
            order=order,
            cases=cases,
            metrics_c0=metrics_c0,
            metrics_c1=metrics_c1,
            controls=controls,
            input_metadata={
                "cases": {
                    "path": str(args.cases),
                    **case_hashes,
                },
                "c0": {"path": str(args.c0), "file_sha256": c0_sha},
                "c1": {"path": str(args.c1), "file_sha256": c1_sha},
                "analyzer_sha256": sha256_file(Path(__file__)),
            },
            bootstrap_iterations=args.bootstrap,
            seed=args.seed,
        )
    except (OSError, json.JSONDecodeError, RetrievalAnalysisError, ValueError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_case_csv(args.csv_out, result["cases"])

    c0_summary = result["condition_summaries"]["c0"]
    c1_summary = result["condition_summaries"]["c1"]
    print(
        "C0/C1 holdout retrieval: "
        f"n={result['questions']} "
        f"SourceHit@5={c0_summary['source_hit_at_5']['mean']:.3f}/"
        f"{c1_summary['source_hit_at_5']['mean']:.3f} "
        f"EvidenceRecall@8={c0_summary['evidence_recall_at_8']['mean']:.3f}/"
        f"{c1_summary['evidence_recall_at_8']['mean']:.3f} "
        f"AllEvidence@8={c0_summary['all_evidence_at_8']['mean']:.3f}/"
        f"{c1_summary['all_evidence_at_8']['mean']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
