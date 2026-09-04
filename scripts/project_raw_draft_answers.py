#!/usr/bin/env python3
"""Project saved drafts into immutable postprocessor diagnostic artifacts.

The source generation artifact remains untouched.  ``raw-draft`` uses the saved
``evaluation_trace.sanitized_draft`` as ``answer`` and deliberately drops
postprocessor-produced claims and citations.  ``current-postprocessed`` replays
that same saved draft over the same saved final contexts through the current
``build_rag_response`` implementation.  No model is called in either mode.

This is a postprocessor diagnostic, not a service-performance lane.  The raw
lane intentionally bypasses attribution/citations; the replay lane executes
them only to reconstruct the current postprocessed answer.  Neither lane may
be used to report grounded-fully-correct or citation performance.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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
from service_eval_artifacts import (  # noqa: E402
    build_answer_identity,
    load_unique_jsonl,
    sha256_json,
    sha256_text,
    validate_answer_record,
)
from search_api import build_rag_response, number_sources  # noqa: E402


PROJECTION_SCHEMA_VERSION = "pnu.postprocessor-diagnostic-projection.v1"
PROJECTION_CHOICES = ("raw-draft", "current-postprocessed")


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _uniform_identity(records: Sequence[Mapping[str, Any]], key: str) -> str:
    values = {
        _required_text(record.get(key), label=f"source {key}")
        for record in records
    }
    if len(values) != 1:
        raise ValueError(f"source answer artifact mixes {key} values: {sorted(values)}")
    return next(iter(values))


def _collector_identity(config: Any) -> dict[str, Any]:
    """Return non-capability source identity fields safe to copy into provenance."""

    if not isinstance(config, Mapping):
        return {}
    allowed = (
        "provider",
        "model",
        "parser_profile",
        "retrieval_mode",
        "expected_corpus_revision",
        "expected_retrieval_tuning",
        "expected_context_chunks_per_document",
        "context_k",
    )
    return {key: copy.deepcopy(config[key]) for key in allowed if key in config}


def load_source_answers(path: Path) -> tuple[str, list[dict[str, Any]]]:
    artifact_sha_before = sha256_file(path)
    records = load_unique_jsonl(path, key="answer_id")
    if not records:
        raise ValueError("source answer artifact is empty")
    for record in records:
        validate_answer_record(record)
        if "judge" in record:
            raise ValueError(
                f"source answer {record['answer_id']} contains forbidden inline judge output"
            )
    if sha256_file(path) != artifact_sha_before:
        raise ValueError("source answer artifact changed during validation")
    return artifact_sha_before, records


def project_answer(
    source: Mapping[str, Any],
    *,
    source_artifact_path: Path,
    source_artifact_sha256: str,
    experiment_id: str,
    condition_id: str,
    projection: str,
) -> dict[str, Any]:
    """Return one integrity-bound, score-only postprocessor projection."""

    source_record = copy.deepcopy(dict(source))
    validate_answer_record(source_record)
    trace = source_record.get("evaluation_trace")
    if not isinstance(trace, Mapping):
        raise ValueError(
            f"source answer {source_record['answer_id']} has no evaluation_trace"
        )
    sanitized_draft = trace.get("sanitized_draft")
    if not isinstance(sanitized_draft, str) or not sanitized_draft.strip():
        raise ValueError(
            f"source answer {source_record['answer_id']} has no non-empty "
            "evaluation_trace.sanitized_draft"
        )
    retrieval_stages = trace.get("retrieval_stages")
    if not isinstance(retrieval_stages, Mapping):
        raise ValueError(
            f"source answer {source_record['answer_id']} has no retrieval stages"
        )
    final_contexts = retrieval_stages.get("final_contexts")
    if (
        not isinstance(final_contexts, list)
        or not final_contexts
        or any(not isinstance(context, Mapping) for context in final_contexts)
    ):
        raise ValueError(
            f"source answer {source_record['answer_id']} has no non-empty "
            "evaluation_trace.retrieval_stages.final_contexts"
        )
    if projection not in PROJECTION_CHOICES:
        raise ValueError(
            f"projection must be one of {', '.join(PROJECTION_CHOICES)}"
        )

    target_experiment = _required_text(experiment_id, label="experiment_id")
    target_condition = _required_text(condition_id, label="condition_id")
    if (
        target_experiment == source_record.get("experiment_id")
        and target_condition == source_record.get("condition_id")
    ):
        raise ValueError(
            "target experiment_id/condition_id must differ from the source "
            "identity so projected answer IDs cannot collide"
        )

    source_collector_config = source_record.get("collector_config")
    source_collector_config_sha256 = source_record.get("collector_config_sha256")
    draft_sha256 = sha256_text(sanitized_draft)
    contexts_sha256 = sha256_json(final_contexts)
    postprocessor_code_path = REPO_ROOT / "scripts" / "search_api.py"
    postprocessor_code_sha256 = sha256_file(postprocessor_code_path)
    projection_config = {
        "collector": "project_raw_draft_answers.py",
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "source_answers_artifact_path": str(source_artifact_path.resolve()),
        "source_answers_artifact_sha256": source_artifact_sha256,
        "source_collector_config_sha256": source_collector_config_sha256,
        "projection_mode": projection,
        "answer_projection": (
            "evaluation_trace.sanitized_draft"
            if projection == "raw-draft"
            else "current build_rag_response(sanitized_draft, final_contexts)"
        ),
        "postprocessor_applied": projection == "current-postprocessed",
        "sanitized_draft_sha256": draft_sha256,
        "final_contexts_sha256": contexts_sha256,
        "postprocessor_code_sha256": postprocessor_code_sha256,
        "service_performance_eligible": False,
    }

    projected = copy.deepcopy(source_record)
    for key in (
        "answer_id",
        "answer_sha256",
        "record_sha256",
        "slot_outcome",
        "answer_eligible_for_judge",
        "service_error",
        "request_attempts",
        "collection_attempt_number",
    ):
        projected.pop(key, None)
    projected["experiment_id"] = target_experiment
    projected["condition_id"] = target_condition
    if projection == "raw-draft":
        projected["answer"] = sanitized_draft
        projected["cited_answer"] = sanitized_draft
        projected["claims"] = []
        projected["citations"] = []
        projected["postprocessing"] = {
            "mode": "bypassed_for_raw_draft_diagnostic",
            "applied": False,
        }
    else:
        replay_contexts = number_sources(
            [copy.deepcopy(dict(context)) for context in final_contexts]
        )
        query = _required_text(source_record.get("query"), label="source query")
        generator = _required_text(
            source_record.get("generator") or "unknown",
            label="source generator",
        )
        replayed = build_rag_response(
            query,
            replay_contexts,
            sanitized_draft,
            generator,
        )
        projected["answer"] = replayed["answer"]
        projected["cited_answer"] = replayed["cited_answer"]
        projected["claims"] = replayed["claims"]
        projected["citations"] = replayed.get("citations") or []
        projected["postprocessing"] = replayed["postprocessing"]
        projected["draft_answer"] = sanitized_draft
    projected["collector_config"] = projection_config
    projected["collector_config_sha256"] = sha256_json(projection_config)

    retrieval = projected.get("retrieval")
    if isinstance(retrieval, Mapping):
        projected["retrieval"] = dict(retrieval)
        projected["retrieval"]["service_performance_eligible"] = False

    projected["postprocessor_diagnostic"] = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "diagnostic_only": True,
        "projection_mode": projection,
        "comparison_target": "same_draft_postprocessor_effect",
        "primary_metric": "llm_judge_score_0_1_2",
        "service_performance_eligible": False,
        "grounded_fully_correct_eligible": False,
        "citation_performance_eligible": False,
        "interpretation": (
            "Score-based postprocessor diagnostic only. Do not interpret this "
            "projection as grounded-fully-correct, citation, or end-to-end "
            "service performance."
        ),
        "source": {
            "answers_artifact_path": str(source_artifact_path.resolve()),
            "answers_artifact_sha256": source_artifact_sha256,
            "answer_id": source_record["answer_id"],
            "answer_sha256": source_record["answer_sha256"],
            "record_sha256": source_record["record_sha256"],
            "record_content_sha256": sha256_json(source_record),
            "experiment_id": source_record["experiment_id"],
            "condition_id": source_record["condition_id"],
            "generation_run_id": source_record["generation_run_id"],
            "collector_identity": _collector_identity(source_collector_config),
            "collector_config_sha256": source_collector_config_sha256,
            "sanitized_draft_sha256": draft_sha256,
            "final_contexts_sha256": contexts_sha256,
            "final_context_count": len(final_contexts),
        },
        "projection": {
            "mode": projection,
            "answer_field": (
                "evaluation_trace.sanitized_draft"
                if projection == "raw-draft"
                else "build_rag_response.answer"
            ),
            "claims": (
                "emptied" if projection == "raw-draft" else "current_replay"
            ),
            "citations": (
                "emptied" if projection == "raw-draft" else "current_replay"
            ),
            "postprocessor": (
                "bypassed" if projection == "raw-draft" else "current_replay"
            ),
            "postprocessor_code_path": str(postprocessor_code_path.resolve()),
            "postprocessor_code_sha256": postprocessor_code_sha256,
        },
    }

    projected = build_answer_identity(projected)
    validate_answer_record(projected)
    return projected


def project_artifact(
    source_path: Path,
    *,
    experiment_id: str,
    condition_id: str,
    projection: str,
) -> tuple[str, list[dict[str, Any]]]:
    source_artifact_sha256, records = load_source_answers(source_path)
    source_experiment = _uniform_identity(records, "experiment_id")
    source_condition = _uniform_identity(records, "condition_id")
    _uniform_identity(records, "generation_run_id")
    if experiment_id == source_experiment and condition_id == source_condition:
        raise ValueError(
            "target experiment_id/condition_id must differ from the source identity"
        )
    projected = [
        project_answer(
            record,
            source_artifact_path=source_path,
            source_artifact_sha256=source_artifact_sha256,
            experiment_id=experiment_id,
            condition_id=condition_id,
            projection=projection,
        )
        for record in records
    ]
    if len({record["answer_id"] for record in projected}) != len(projected):
        raise ValueError("projected answer IDs are not unique")
    if sha256_file(source_path) != source_artifact_sha256:
        raise ValueError("source answer artifact changed during projection")
    return source_artifact_sha256, projected


def render_jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in records
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--projection", choices=PROJECTION_CHOICES, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        reject_symlink_inputs([args.answers])
        if paths_alias(args.answers, args.out):
            raise ValueError("--out must not overwrite or alias --answers")
        require_new_outputs([args.out])
        source_sha, records = project_artifact(
            args.answers,
            experiment_id=args.experiment_id,
            condition_id=args.condition_id,
            projection=args.projection,
        )
        publish_immutable_texts(
            {args.out: render_jsonl(records)},
            authoritative_path=args.out,
        )

        output_sha_before = sha256_file(args.out)
        published = load_unique_jsonl(args.out, key="answer_id")
        if len(published) != len(records):
            raise ValueError("published projection record count mismatch")
        for record in published:
            validate_answer_record(record)
        if sha256_file(args.out) != output_sha_before:
            raise ValueError("published projection changed during revalidation")
        if sha256_file(args.answers) != source_sha:
            raise ValueError("source answer artifact changed during publication")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    print(
        f"projected_answers={len(records)} projection={args.projection} "
        "diagnostic_only=true "
        f"primary_metric=llm_judge_score_0_1_2 out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
