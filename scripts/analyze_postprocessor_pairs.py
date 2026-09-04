#!/usr/bin/env python3
"""Compare paired raw-draft and current-postprocessed diagnostic projections.

This analyzer is deliberately deterministic and calls no model.  It describes
text removed or retained by the current postprocessor for projections derived
from the exact same saved answer, sanitized draft, and final contexts.  The
result is diagnostic attribution only; it is not a factuality, groundedness,
citation-quality, retrieval-quality, or end-to-end service metric.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from holdout_gold import sha256_file  # noqa: E402
from immutable_outputs import (  # noqa: E402
    paths_alias,
    publish_immutable_texts,
    reject_symlink_inputs,
    require_new_outputs,
)
from search_api import (  # noqa: E402
    extract_critical_values,
    normalize_text,
    split_draft_claims,
)
from service_eval_artifacts import (  # noqa: E402
    load_unique_jsonl,
    sha256_json,
    sha256_text,
    validate_answer_record,
)


OUTPUT_SCHEMA_VERSION = "pnu.postprocessor-pair-analysis.v2"
PROJECTION_SCHEMA_VERSION = "pnu.postprocessor-diagnostic-projection.v1"
RAW_MODE = "raw-draft"
CURRENT_MODE = "current-postprocessed"
PRIMARY_METRIC = "llm_judge_score_0_1_2"
COMPARISON_TARGET = "same_draft_postprocessor_effect"
WARNING = (
    "This deterministic postprocessor projection analysis does not measure "
    "factuality, groundedness, citation quality, retrieval quality, or "
    "end-to-end service performance. It makes no external LLM calls and must "
    "be interpreted only as a same-draft postprocessor effect diagnostic."
)

STANDARD_REFUSALS = {
    "no_results": (
        "검색된 근거 문서가 없습니다. 기관 범위나 질문 표현을 바꿔 "
        "다시 검색해 주세요."
    ),
    "model_abstention": (
        "제공된 검색 근거에서 질문에 답할 내용을 확인할 수 없습니다."
    ),
    "no_supported_claims": (
        "검색 결과는 있으나 답변 문장을 지지하는 근거를 충분히 "
        "확인하지 못했습니다."
    ),
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")

SOURCE_HASH_KEYS = (
    "answers_artifact_sha256",
    "answer_sha256",
    "record_sha256",
    "record_content_sha256",
    "collector_config_sha256",
    "sanitized_draft_sha256",
    "final_contexts_sha256",
)
SOURCE_TEXT_KEYS = (
    "answers_artifact_path",
    "answer_id",
    "experiment_id",
    "condition_id",
    "generation_run_id",
)


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _required_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _diagnostic_error(case_id: str, detail: str) -> ValueError:
    return ValueError(f"invalid diagnostic projection for {case_id}: {detail}")


def _normalize_unit(value: str) -> str:
    """Normalize one answer unit for conservative exact comparison."""

    without_marker = LIST_MARKER_RE.sub("", str(value or ""), count=1)
    return normalize_text(without_marker).casefold()


def _answer_units(answer: str) -> list[str]:
    return [unit for unit in split_draft_claims(answer) if _normalize_unit(unit)]


def _standard_refusal_type(answer: str) -> str | None:
    normalized = _normalize_unit(answer)
    for refusal_type, message in STANDARD_REFUSALS.items():
        if normalized == _normalize_unit(message):
            return refusal_type
    return None


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"total": 0, "mean": 0.0, "min": 0, "max": 0}
    total = sum(values)
    return {
        "total": total,
        "mean": round(total / len(values), 3),
        "min": min(values),
        "max": max(values),
    }


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _validate_diagnostic_record(
    record: dict[str, Any],
    *,
    expected_mode: str,
) -> Mapping[str, Any]:
    """Validate one signed projection and its embedded diagnostic contract."""

    validate_answer_record(record)
    case_id = _required_text(record.get("case_id"), label="case_id")
    if "judge" in record or "judgment" in record:
        raise _diagnostic_error(case_id, "inline judge output is forbidden")
    if record.get("slot_outcome") == "service_error" or "service_error" in record:
        raise _diagnostic_error(case_id, "service-error records are forbidden")

    diagnostic = _required_mapping(
        record.get("postprocessor_diagnostic"),
        label=f"postprocessor_diagnostic for {case_id}",
    )
    gates = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "projection_mode": expected_mode,
        "comparison_target": COMPARISON_TARGET,
        "primary_metric": PRIMARY_METRIC,
        "service_performance_eligible": False,
        "grounded_fully_correct_eligible": False,
        "citation_performance_eligible": False,
    }
    for key, expected in gates.items():
        if diagnostic.get(key) != expected:
            raise _diagnostic_error(
                case_id,
                f"{key} must be {expected!r}, got {diagnostic.get(key)!r}",
            )
    _required_text(
        diagnostic.get("interpretation"),
        label=f"diagnostic interpretation for {case_id}",
    )

    source = _required_mapping(
        diagnostic.get("source"), label=f"diagnostic source for {case_id}"
    )
    for key in SOURCE_TEXT_KEYS:
        _required_text(source.get(key), label=f"source {key} for {case_id}")
    for key in SOURCE_HASH_KEYS:
        _require_sha256(source.get(key), label=f"source {key} for {case_id}")
    collector_identity = source.get("collector_identity")
    if not isinstance(collector_identity, Mapping):
        raise _diagnostic_error(case_id, "source collector_identity must be an object")
    context_count = source.get("final_context_count")
    if isinstance(context_count, bool) or not isinstance(context_count, int):
        raise _diagnostic_error(case_id, "source final_context_count must be an integer")
    if context_count <= 0:
        raise _diagnostic_error(case_id, "source final_context_count must be positive")

    projection = _required_mapping(
        diagnostic.get("projection"),
        label=f"diagnostic projection detail for {case_id}",
    )
    expected_projection = {
        RAW_MODE: {
            "mode": RAW_MODE,
            "answer_field": "evaluation_trace.sanitized_draft",
            "claims": "emptied",
            "citations": "emptied",
            "postprocessor": "bypassed",
        },
        CURRENT_MODE: {
            "mode": CURRENT_MODE,
            "answer_field": "build_rag_response.answer",
            "claims": "current_replay",
            "citations": "current_replay",
            "postprocessor": "current_replay",
        },
    }[expected_mode]
    for key, expected in expected_projection.items():
        if projection.get(key) != expected:
            raise _diagnostic_error(
                case_id,
                f"projection.{key} must be {expected!r}",
            )
    _required_text(
        projection.get("postprocessor_code_path"),
        label=f"projection postprocessor_code_path for {case_id}",
    )
    postprocessor_code_sha = _require_sha256(
        projection.get("postprocessor_code_sha256"),
        label=f"projection postprocessor_code_sha256 for {case_id}",
    )

    collector = _required_mapping(
        record.get("collector_config"), label=f"collector_config for {case_id}"
    )
    if record.get("collector_config_sha256") != sha256_json(collector):
        raise _diagnostic_error(case_id, "collector_config_sha256 mismatch")
    expected_collector = {
        "collector": "project_raw_draft_answers.py",
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "source_answers_artifact_path": source["answers_artifact_path"],
        "source_answers_artifact_sha256": source["answers_artifact_sha256"],
        "source_collector_config_sha256": source["collector_config_sha256"],
        "projection_mode": expected_mode,
        "answer_projection": (
            "evaluation_trace.sanitized_draft"
            if expected_mode == RAW_MODE
            else "current build_rag_response(sanitized_draft, final_contexts)"
        ),
        "postprocessor_applied": expected_mode == CURRENT_MODE,
        "sanitized_draft_sha256": source["sanitized_draft_sha256"],
        "final_contexts_sha256": source["final_contexts_sha256"],
        "postprocessor_code_sha256": postprocessor_code_sha,
        "service_performance_eligible": False,
    }
    for key, expected in expected_collector.items():
        if collector.get(key) != expected:
            raise ValueError(
                f"source provenance mismatch for {case_id}: "
                f"collector_config.{key} does not match the diagnostic source"
            )

    retrieval = _required_mapping(
        record.get("retrieval"), label=f"retrieval for {case_id}"
    )
    if retrieval.get("service_performance_eligible") is not False:
        raise _diagnostic_error(
            case_id, "retrieval.service_performance_eligible must be false"
        )

    trace = _required_mapping(
        record.get("evaluation_trace"), label=f"evaluation_trace for {case_id}"
    )
    sanitized_draft = _required_text(
        trace.get("sanitized_draft"),
        label=f"evaluation_trace.sanitized_draft for {case_id}",
    )
    if sha256_text(sanitized_draft) != source["sanitized_draft_sha256"]:
        raise _diagnostic_error(case_id, "sanitized draft content/hash mismatch")
    retrieval_stages = _required_mapping(
        trace.get("retrieval_stages"),
        label=f"evaluation_trace.retrieval_stages for {case_id}",
    )
    final_contexts = _required_list(
        retrieval_stages.get("final_contexts"),
        label=f"evaluation_trace final_contexts for {case_id}",
    )
    if not final_contexts or any(not isinstance(item, Mapping) for item in final_contexts):
        raise _diagnostic_error(case_id, "final_contexts must contain objects")
    if len(final_contexts) != context_count:
        raise _diagnostic_error(case_id, "final_context_count mismatch")
    if sha256_json(final_contexts) != source["final_contexts_sha256"]:
        raise _diagnostic_error(case_id, "final contexts content/hash mismatch")

    claims = _required_list(record.get("claims"), label=f"claims for {case_id}")
    _required_list(record.get("citations"), label=f"citations for {case_id}")
    postprocessing = _required_mapping(
        record.get("postprocessing"), label=f"postprocessing for {case_id}"
    )
    answer = _required_text(record.get("answer"), label=f"answer for {case_id}")
    cited_answer = _required_text(
        record.get("cited_answer"), label=f"cited_answer for {case_id}"
    )

    if expected_mode == RAW_MODE:
        if claims or record.get("citations"):
            raise _diagnostic_error(case_id, "raw projection claims/citations must be empty")
        if postprocessing != {
            "mode": "bypassed_for_raw_draft_diagnostic",
            "applied": False,
        }:
            raise _diagnostic_error(case_id, "raw postprocessing bypass marker mismatch")
        if answer != sanitized_draft or cited_answer != answer:
            raise _diagnostic_error(case_id, "raw answer must equal the sanitized draft")
    else:
        draft_answer = _required_text(
            record.get("draft_answer"), label=f"draft_answer for {case_id}"
        )
        if draft_answer != sanitized_draft:
            raise _diagnostic_error(case_id, "current draft_answer/sanitized draft mismatch")
        _validate_current_postprocessing(
            case_id=case_id,
            draft_answer=draft_answer,
            final_answer=answer,
            claims=claims,
            postprocessing=postprocessing,
        )

    return diagnostic


def _nonnegative_int(value: Any, *, case_id: str, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _diagnostic_error(case_id, f"{label} must be a non-negative integer")
    return value


def _validate_current_postprocessing(
    *,
    case_id: str,
    draft_answer: str,
    final_answer: str,
    claims: Sequence[Any],
    postprocessing: Mapping[str, Any],
) -> None:
    input_count = _nonnegative_int(
        postprocessing.get("input_claim_count"),
        case_id=case_id,
        label="postprocessing.input_claim_count",
    )
    processed_count = _nonnegative_int(
        postprocessing.get("processed_claim_count"),
        case_id=case_id,
        label="postprocessing.processed_claim_count",
    )
    truncated_count = _nonnegative_int(
        postprocessing.get("truncated_claim_count"),
        case_id=case_id,
        label="postprocessing.truncated_claim_count",
    )
    if not isinstance(postprocessing.get("claims_truncated"), bool):
        raise _diagnostic_error(
            case_id, "postprocessing.claims_truncated must be boolean"
        )
    raw_units = _answer_units(draft_answer)
    if input_count != len(raw_units):
        raise _diagnostic_error(case_id, "input claim count/draft mismatch")
    if processed_count != len(claims):
        raise _diagnostic_error(case_id, "processed claim count/claims mismatch")
    if input_count - processed_count != truncated_count:
        raise _diagnostic_error(case_id, "truncated claim count mismatch")
    if postprocessing["claims_truncated"] != (truncated_count > 0):
        raise _diagnostic_error(case_id, "claims_truncated flag mismatch")

    claim_texts: list[str] = []
    supported_texts: list[str] = []
    reasons: list[str] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            raise _diagnostic_error(case_id, f"claim {index} must be an object")
        text = _required_text(
            claim.get("text"), label=f"claim {index} text for {case_id}"
        )
        supported = claim.get("supported")
        if not isinstance(supported, bool):
            raise _diagnostic_error(case_id, f"claim {index} supported must be boolean")
        reason = _required_text(
            claim.get("validation_reason"),
            label=f"claim {index} validation_reason for {case_id}",
        )
        claim_texts.append(text)
        reasons.append(reason)
        if supported:
            supported_texts.append(text)

    expected_inputs = raw_units[:processed_count]
    if [_normalize_unit(item) for item in claim_texts] != [
        _normalize_unit(item) for item in expected_inputs
    ]:
        raise _diagnostic_error(case_id, "attributed claim sequence/draft mismatch")

    if supported_texts:
        if [_normalize_unit(item) for item in _answer_units(final_answer)] != [
            _normalize_unit(item) for item in supported_texts
        ]:
            raise _diagnostic_error(case_id, "final answer/supported claim mismatch")
    else:
        refusal_type = _standard_refusal_type(final_answer)
        if refusal_type is None:
            raise _diagnostic_error(case_id, "unsupported-only output is not a standard refusal")
        expected_refusal = (
            "model_abstention"
            if claims and all(reason == "model_abstention" for reason in reasons)
            else "no_supported_claims"
        )
        if refusal_type != expected_refusal:
            raise _diagnostic_error(case_id, "standard refusal/rejection reason mismatch")


def _load_projection_artifact(
    path: Path, *, expected_mode: str
) -> tuple[str, list[dict[str, Any]]]:
    artifact_sha = sha256_file(path)
    records = load_unique_jsonl(path, key="answer_id")
    if not records:
        raise ValueError(f"{expected_mode} answer artifact is empty")
    seen_cases: set[str] = set()
    for record in records:
        _validate_diagnostic_record(record, expected_mode=expected_mode)
        case_id = record["case_id"]
        if case_id in seen_cases:
            raise ValueError(f"duplicate case_id {case_id} in {expected_mode} artifact")
        seen_cases.add(case_id)
    if sha256_file(path) != artifact_sha:
        raise ValueError(f"{expected_mode} answer artifact changed during validation")
    return artifact_sha, records


def _sentence_projection(raw_units: Sequence[str], final_units: Sequence[str]) -> dict[str, Any]:
    final_by_normalized: dict[str, deque[int]] = defaultdict(deque)
    for index, unit in enumerate(final_units):
        final_by_normalized[_normalize_unit(unit)].append(index)

    kept: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []
    matched_final: set[int] = set()
    for raw_index, raw_unit in enumerate(raw_units):
        normalized = _normalize_unit(raw_unit)
        candidates = final_by_normalized.get(normalized)
        if candidates:
            final_index = candidates.popleft()
            matched_final.add(final_index)
            kept.append(
                {
                    "raw_index": raw_index,
                    "final_index": final_index,
                    "raw_text": raw_unit,
                    "final_text": final_units[final_index],
                    "normalized_text": normalized,
                }
            )
        else:
            deleted.append(
                {
                    "raw_index": raw_index,
                    "text": raw_unit,
                    "normalized_text": normalized,
                }
            )

    added = [
        {
            "final_index": index,
            "text": unit,
            "normalized_text": _normalize_unit(unit),
        }
        for index, unit in enumerate(final_units)
        if index not in matched_final
    ]
    return {
        "matching": "normalized_exact_multiset",
        "kept_count": len(kept),
        "deleted_count": len(deleted),
        "added_count": len(added),
        "kept": kept,
        "deleted": deleted,
        "added": added,
    }


def _case_analysis(
    raw: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    case_id = raw["case_id"]
    raw_diagnostic = raw["postprocessor_diagnostic"]
    current_diagnostic = current["postprocessor_diagnostic"]
    raw_source = raw_diagnostic["source"]
    current_source = current_diagnostic["source"]
    if raw_source != current_source:
        raise ValueError(f"source provenance mismatch for {case_id}")
    if (
        raw_diagnostic["projection"]["postprocessor_code_sha256"]
        != current_diagnostic["projection"]["postprocessor_code_sha256"]
    ):
        raise ValueError(f"source provenance mismatch for {case_id}: postprocessor code")
    if raw.get("experiment_id") != current.get("experiment_id"):
        raise ValueError(f"projection experiment mismatch for {case_id}")
    if raw.get("generation_run_id") != current.get("generation_run_id"):
        raise ValueError(f"projection generation run mismatch for {case_id}")
    if raw.get("condition_id") == current.get("condition_id"):
        raise ValueError(f"projection conditions must differ for {case_id}")
    for key in ("case_sha256", "query"):
        if raw.get(key) != current.get(key):
            raise ValueError(f"paired {key} mismatch for {case_id}")
    if raw["answer"] != current.get("draft_answer"):
        raise ValueError(f"paired sanitized draft mismatch for {case_id}")

    raw_answer = raw["answer"]
    final_answer = current["answer"]
    raw_units = _answer_units(raw_answer)
    final_units = _answer_units(final_answer)
    sentence_projection = _sentence_projection(raw_units, final_units)
    raw_values = set(extract_critical_values(raw_answer))
    final_values = set(extract_critical_values(final_answer))
    rejection_reasons = Counter(
        str(claim["validation_reason"])
        for claim in current["claims"]
        if claim["supported"] is False
    )
    supported_claim_count = sum(
        1 for claim in current["claims"] if claim["supported"] is True
    )
    refusal_type = _standard_refusal_type(final_answer)

    return {
        "case_id": case_id,
        "query": raw["query"],
        "source_provenance": {
            "answer_id": raw_source["answer_id"],
            "answer_sha256": raw_source["answer_sha256"],
            "record_sha256": raw_source["record_sha256"],
            "sanitized_draft_sha256": raw_source["sanitized_draft_sha256"],
            "final_contexts_sha256": raw_source["final_contexts_sha256"],
            "final_context_count": raw_source["final_context_count"],
        },
        "answer_bytes_changed": (
            raw_answer.encode("utf-8") != final_answer.encode("utf-8")
        ),
        "content_units_changed": (
            sentence_projection["deleted_count"] > 0
            or sentence_projection["added_count"] > 0
        ),
        "raw": {
            "answer": raw_answer,
            "answer_sha256": raw["answer_sha256"],
            "chars": len(raw_answer),
            "claim_count": len(raw_units),
            "claims": raw_units,
        },
        "final": {
            "answer": final_answer,
            "answer_sha256": current["answer_sha256"],
            "chars": len(final_answer),
            "claim_count": len(final_units),
            "claims": final_units,
            "attributed_claim_count": len(current["claims"]),
            "supported_claim_count": supported_claim_count,
            "rejected_claim_count": len(current["claims"]) - supported_claim_count,
            "standard_refusal": refusal_type is not None,
            "standard_refusal_type": refusal_type,
        },
        "sentence_projection": sentence_projection,
        "critical_values": {
            "raw": sorted(raw_values),
            "final": sorted(final_values),
            "retained": sorted(raw_values & final_values),
            "lost": sorted(raw_values - final_values),
            "added": sorted(final_values - raw_values),
        },
        "postprocessing_rejection_reason_counts": _sorted_counts(rejection_reasons),
    }


def build_analysis(raw_path: Path, current_path: Path) -> dict[str, Any]:
    """Build a deterministic, integrity-checked paired diagnostic report."""

    raw_path = Path(raw_path)
    current_path = Path(current_path)
    reject_symlink_inputs([raw_path, current_path])
    if paths_alias(raw_path, current_path):
        raise ValueError("raw and current input artifacts must not alias")

    raw_sha, raw_records = _load_projection_artifact(
        raw_path, expected_mode=RAW_MODE
    )
    current_sha, current_records = _load_projection_artifact(
        current_path, expected_mode=CURRENT_MODE
    )
    raw_by_case = {record["case_id"]: record for record in raw_records}
    current_by_case = {record["case_id"]: record for record in current_records}
    raw_cases = set(raw_by_case)
    current_cases = set(current_by_case)
    if raw_cases != current_cases:
        missing_current = sorted(raw_cases - current_cases)
        missing_raw = sorted(current_cases - raw_cases)
        raise ValueError(
            "case set mismatch: "
            f"missing_current={missing_current}, missing_raw={missing_raw}"
        )

    per_case = [
        _case_analysis(raw_by_case[case_id], current_by_case[case_id])
        for case_id in sorted(raw_cases)
    ]
    answer_bytes_changed = sum(row["answer_bytes_changed"] for row in per_case)
    content_units_changed = sum(row["content_units_changed"] for row in per_case)
    raw_chars = [row["raw"]["chars"] for row in per_case]
    final_chars = [row["final"]["chars"] for row in per_case]
    raw_claims = [row["raw"]["claim_count"] for row in per_case]
    final_claims = [row["final"]["claim_count"] for row in per_case]
    rejection_reasons: Counter[str] = Counter()
    lost_values: Counter[str] = Counter()
    added_values: Counter[str] = Counter()
    retained_values: Counter[str] = Counter()
    for row in per_case:
        rejection_reasons.update(row["postprocessing_rejection_reason_counts"])
        lost_values.update(row["critical_values"]["lost"])
        added_values.update(row["critical_values"]["added"])
        retained_values.update(row["critical_values"]["retained"])

    summary = {
        "n": len(per_case),
        "answer_bytes_changed": answer_bytes_changed,
        "answer_bytes_tied": len(per_case) - answer_bytes_changed,
        "answer_bytes_changed_case_ids": [
            row["case_id"] for row in per_case if row["answer_bytes_changed"]
        ],
        "content_units_changed": content_units_changed,
        "content_units_tied": len(per_case) - content_units_changed,
        "content_units_changed_case_ids": [
            row["case_id"] for row in per_case if row["content_units_changed"]
        ],
        "raw_char_counts": _distribution(raw_chars),
        "final_char_counts": _distribution(final_chars),
        "raw_claim_counts": _distribution(raw_claims),
        "final_claim_counts": _distribution(final_claims),
        "sentence_units": {
            "raw": sum(raw_claims),
            "final": sum(final_claims),
            "kept": sum(row["sentence_projection"]["kept_count"] for row in per_case),
            "deleted": sum(
                row["sentence_projection"]["deleted_count"] for row in per_case
            ),
            "added": sum(row["sentence_projection"]["added_count"] for row in per_case),
        },
        "critical_values": {
            "retained": sum(retained_values.values()),
            "lost": sum(lost_values.values()),
            "added": sum(added_values.values()),
            "retained_by_value": _sorted_counts(retained_values),
            "lost_by_value": _sorted_counts(lost_values),
            "added_by_value": _sorted_counts(added_values),
        },
        "final_standard_refusal_count": sum(
            row["final"]["standard_refusal"] for row in per_case
        ),
        "final_standard_refusal_case_ids": [
            row["case_id"] for row in per_case if row["final"]["standard_refusal"]
        ],
        "postprocessing_rejection_reason_counts": _sorted_counts(rejection_reasons),
    }
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_type": "same-source-postprocessor-pair-diagnostic",
        "warning": WARNING,
        "methodology": {
            "paired_unit": "case_id",
            "answer_bytes_changed": "exact UTF-8 byte inequality",
            "sentence_unit_extractor": "search_api.split_draft_claims",
            "sentence_matching": "normalized exact multiset match",
            "content_units_changed": (
                "one or more normalized sentence units deleted or added; "
                "format-only byte changes are tied"
            ),
            "critical_value_extractor": "search_api.extract_critical_values",
            "external_llm_calls": False,
        },
        "inputs": {
            "raw": {
                "path": str(raw_path.resolve()),
                "artifact_sha256": raw_sha,
                "projection_mode": RAW_MODE,
            },
            "current": {
                "path": str(current_path.resolve()),
                "artifact_sha256": current_sha,
                "projection_mode": CURRENT_MODE,
            },
        },
        "integrity_verification": {
            "answer_records_validated": True,
            "case_sets_exact": True,
            "source_provenance_exact": True,
            "source_answer_hash_exact": True,
            "sanitized_draft_hash_exact": True,
            "final_contexts_hash_exact": True,
            "diagnostic_projection_gates_validated": True,
            "score_only_eligibility_validated": True,
            "input_artifacts_stable_during_validation": True,
            "external_llm_called": False,
        },
        "summary": summary,
        "per_case": per_case,
    }


def render_json(report: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-provenance raw-draft/current-postprocessed JSONL "
            "diagnostic projections without an external LLM call."
        )
    )
    parser.add_argument("--raw-answers", type=Path, required=True)
    parser.add_argument("--current-answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_path = args.raw_answers
    current_path = args.current_answers
    output_path = args.out
    try:
        if paths_alias(output_path, raw_path) or paths_alias(output_path, current_path):
            raise ValueError("--out must not overwrite or alias either input artifact")
        require_new_outputs([output_path])
        report = build_analysis(raw_path, current_path)
        if sha256_file(raw_path) != report["inputs"]["raw"]["artifact_sha256"]:
            raise ValueError("raw answer artifact changed before publication")
        if sha256_file(current_path) != report["inputs"]["current"]["artifact_sha256"]:
            raise ValueError("current answer artifact changed before publication")
        publish_immutable_texts(
            {output_path: render_json(report)}, authoritative_path=output_path
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    summary = report["summary"]
    print(
        f"n={summary['n']} "
        f"answer_bytes_changed={summary['answer_bytes_changed']} "
        f"answer_bytes_tied={summary['answer_bytes_tied']} "
        f"content_units_changed={summary['content_units_changed']} "
        f"content_units_tied={summary['content_units_tied']} "
        "external_llm_calls=false "
        f"out={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
