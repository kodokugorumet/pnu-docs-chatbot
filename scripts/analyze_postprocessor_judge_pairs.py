#!/usr/bin/env python3
"""Compare repeated Judge scores for same-draft postprocessor projections.

This is a diagnostic-only score analysis.  A question is comparable only when
all three raw judgments and all three current judgments are valid.  A judgment
with a non-null ``error`` or a missing/invalid ``judge.score`` is invalid; this
tool deliberately never parses ``raw_judge_response`` to recover a score.

The projections are explicitly ineligible for final service, grounded-fully-
correct, and citation-performance claims.  Repeated Judge calls measure Judge
stability and do not increase the effective sample size beyond the number of
fully comparable questions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_postprocessor_pairs import (  # noqa: E402
    CURRENT_MODE,
    RAW_MODE,
    build_analysis as validate_projection_pair,
)
from immutable_outputs import (  # noqa: E402
    paths_alias,
    publish_immutable_texts,
    reject_symlink_inputs,
    require_new_outputs,
)
from service_eval_artifacts import (  # noqa: E402
    load_unique_jsonl,
    sha256_json,
    validate_answer_record,
    validate_judgment_record,
)


OUTPUT_SCHEMA_VERSION = "pnu.postprocessor-judge-pair-analysis.v1"
REQUIRED_REPEAT_COUNT = 3
WARNING = (
    "Diagnostic same-draft postprocessor comparison only. Results are not "
    "eligible as final service, grounded-fully-correct, citation, retrieval, "
    "or end-to-end performance evidence. Questions with any invalid repeat "
    "are excluded from paired means and W/T/L; Judge repeats are not "
    "independent samples."
)


@dataclass(frozen=True)
class ScoreSide:
    answer_path: Path
    answer_sha256: str
    order: list[str]
    answers_by_case: dict[str, dict[str, Any]]
    judgment_artifacts: list[dict[str, Any]]
    repeats_by_case: dict[str, list[dict[str, Any]]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_answers(path: Path) -> tuple[str, list[str], dict[str, dict[str, Any]]]:
    artifact_sha = sha256_file(path)
    records = load_unique_jsonl(path, key="answer_id")
    if not records:
        raise ValueError(f"{path}: answer artifact is empty")
    order: list[str] = []
    by_case: dict[str, dict[str, Any]] = {}
    for record in records:
        validate_answer_record(record)
        case_id = str(record["case_id"])
        if case_id in by_case:
            raise ValueError(f"{path}: duplicate case_id {case_id}")
        if record.get("error") is not None:
            raise ValueError(f"{path}: answer {case_id} has error")
        if isinstance(record.get("collector_config"), Mapping) and isinstance(
            record["collector_config"].get("final_authorization"), Mapping
        ):
            raise ValueError(
                f"{path}: final-authorized answer {case_id} is forbidden in "
                "this diagnostic analyzer"
            )
        order.append(case_id)
        by_case[case_id] = record
    for field in ("experiment_id", "condition_id", "generation_run_id"):
        values = {str(record.get(field) or "") for record in records}
        if len(values) != 1 or "" in values:
            raise ValueError(f"{path}: answer artifact mixes or omits {field}")
    if sha256_file(path) != artifact_sha:
        raise ValueError(f"{path}: answer artifact changed during validation")
    return artifact_sha, order, by_case


def _error_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _validate_judgment_binding(
    record: dict[str, Any],
    *,
    answer: dict[str, Any],
    answer_artifact_sha256: str,
    path: Path,
) -> None:
    case_id = str(answer["case_id"])
    validate_judgment_record(record)
    expected = {
        "case_id": answer["case_id"],
        "answer_id": answer["answer_id"],
        "answer_sha256": answer["answer_sha256"],
        "experiment_id": answer["experiment_id"],
        "condition_id": answer["condition_id"],
        "generation_run_id": answer["generation_run_id"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{path}: {field} mismatch for case {case_id}")
    if record.get("answer_record_sha256") != sha256_json(answer):
        raise ValueError(f"{path}: answer_record_sha256 mismatch for case {case_id}")
    if record.get("answers_artifact_sha256") != answer_artifact_sha256:
        raise ValueError(f"{path}: answers_artifact_sha256 mismatch for case {case_id}")
    if record.get("judge_repeat_selection") is not None:
        raise ValueError(
            f"{path}: selection-bound judgments are not accepted by this "
            "complete diagnostic case-set analyzer"
        )


def _load_score_side(
    answer_path: Path,
    judgment_paths: Sequence[Path],
) -> ScoreSide:
    if len(judgment_paths) != REQUIRED_REPEAT_COUNT:
        raise ValueError(
            f"exactly {REQUIRED_REPEAT_COUNT} judgment artifacts are required "
            "per condition"
        )
    all_paths = [answer_path, *judgment_paths]
    reject_symlink_inputs(all_paths)
    for index, left in enumerate(all_paths):
        for right in all_paths[index + 1 :]:
            if paths_alias(left, right):
                raise ValueError("answer and judgment input artifacts must be distinct")

    answer_sha, order, answers_by_case = _load_answers(answer_path)
    answer_by_id = {
        str(answer["answer_id"]): answer for answer in answers_by_case.values()
    }
    repeats_by_case = {case_id: [] for case_id in order}
    artifact_metadata: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seen_artifact_shas: set[str] = set()

    for repeat_index, path in enumerate(judgment_paths, 1):
        artifact_sha = sha256_file(path)
        if artifact_sha in seen_artifact_shas:
            raise ValueError(f"duplicate judgment artifact SHA-256: {artifact_sha}")
        seen_artifact_shas.add(artifact_sha)
        records = load_unique_jsonl(path, key="judgment_id")
        if len(records) != len(order):
            raise ValueError(
                f"{path}: judgment case count {len(records)} does not match "
                f"answer case count {len(order)}"
            )
        records_by_case: dict[str, dict[str, Any]] = {}
        for record in records:
            validate_judgment_record(record)
            answer_id = str(record.get("answer_id") or "")
            answer = answer_by_id.get(answer_id)
            if answer is None:
                raise ValueError(f"{path}: judgment references unknown answer_id {answer_id}")
            case_id = str(record.get("case_id") or "")
            if case_id in records_by_case:
                raise ValueError(f"{path}: duplicate case_id {case_id}")
            _validate_judgment_binding(
                record,
                answer=answer,
                answer_artifact_sha256=answer_sha,
                path=path,
            )
            records_by_case[case_id] = record
        if set(records_by_case) != set(order):
            raise ValueError(f"{path}: judgment case set does not match answer artifact")

        run_ids = {str(record.get("judge_run_id") or "") for record in records}
        if len(run_ids) != 1 or "" in run_ids:
            raise ValueError(f"{path}: judgment artifact mixes or omits judge_run_id")
        run_id = next(iter(run_ids))
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate judge_run_id across artifacts: {run_id}")
        seen_run_ids.add(run_id)
        config_shas = {
            str(record.get("judge_config_sha256") or "") for record in records
        }
        if len(config_shas) != 1 or "" in config_shas:
            raise ValueError(f"{path}: judgment artifact mixes or omits judge_config_sha256")
        config_sha = next(iter(config_shas))

        for case_id in order:
            record = records_by_case[case_id]
            error = _error_text(record.get("error"))
            judge = record.get("judge")
            score = judge.get("score") if isinstance(judge, Mapping) else None
            valid = error is None and type(score) is int and score in (0, 1, 2)
            if not valid:
                score = None
                invalid_reason = (
                    "judgment_error" if error is not None else "missing_or_invalid_score"
                )
            else:
                invalid_reason = None
            repeats_by_case[case_id].append(
                {
                    "repeat": repeat_index,
                    "judge_run_id": run_id,
                    "status": "valid" if valid else "invalid",
                    "score": score,
                    "error": error,
                    "invalid_reason": invalid_reason,
                    "raw_judge_response_score_recovery_attempted": False,
                }
            )
        if sha256_file(path) != artifact_sha:
            raise ValueError(f"{path}: judgment artifact changed during validation")
        artifact_metadata.append(
            {
                "path": str(path.resolve()),
                "sha256": artifact_sha,
                "judge_run_id": run_id,
                "judge_config_sha256": config_sha,
            }
        )

    return ScoreSide(
        answer_path=answer_path.resolve(),
        answer_sha256=answer_sha,
        order=order,
        answers_by_case=answers_by_case,
        judgment_artifacts=artifact_metadata,
        repeats_by_case=repeats_by_case,
    )


def _side_case(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["score"] for row in repeats]
    errors = [row["error"] for row in repeats]
    fully_valid = all(row["status"] == "valid" for row in repeats)
    valid_scores = [int(score) for score in scores if type(score) is int]
    return {
        "repeats": repeats,
        "scores": scores,
        "errors": errors,
        "valid_repeat_count": len(valid_scores),
        "invalid_repeat_count": len(repeats) - len(valid_scores),
        "fully_valid": fully_valid,
        "mean_score": mean(valid_scores) if fully_valid else None,
        "all_valid_repeats_agree": len(set(valid_scores)) == 1 if fully_valid else None,
    }


def _score_distribution(values: Iterable[int]) -> dict[str, int]:
    materialized = list(values)
    return {str(score): materialized.count(score) for score in (0, 1, 2)}


def build_analysis(
    raw_answer_path: Path,
    current_answer_path: Path,
    raw_judgment_paths: Sequence[Path],
    current_judgment_paths: Sequence[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one fail-closed, question-level score-only comparison."""

    raw_answer_path = Path(raw_answer_path)
    current_answer_path = Path(current_answer_path)
    raw_judgment_paths = [Path(path) for path in raw_judgment_paths]
    current_judgment_paths = [Path(path) for path in current_judgment_paths]
    all_inputs = [
        raw_answer_path,
        current_answer_path,
        *raw_judgment_paths,
        *current_judgment_paths,
    ]
    reject_symlink_inputs(all_inputs)
    for index, left in enumerate(all_inputs):
        for right in all_inputs[index + 1 :]:
            if paths_alias(left, right):
                raise ValueError("all answer and judgment inputs must be distinct")

    projection_validation = validate_projection_pair(
        raw_answer_path, current_answer_path
    )
    raw = _load_score_side(raw_answer_path, raw_judgment_paths)
    current = _load_score_side(current_answer_path, current_judgment_paths)
    if raw.order != current.order:
        raise ValueError("raw/current answer case order mismatch")

    raw_first = raw.answers_by_case[raw.order[0]]
    current_first = current.answers_by_case[current.order[0]]
    if raw_first["experiment_id"] != current_first["experiment_id"]:
        raise ValueError("raw/current experiment_id mismatch")
    if raw_first["generation_run_id"] != current_first["generation_run_id"]:
        raise ValueError("raw/current generation_run_id mismatch")
    if raw_first["condition_id"] == current_first["condition_id"]:
        raise ValueError("raw/current condition_id must differ")

    config_shas = {
        item["judge_config_sha256"]
        for item in [*raw.judgment_artifacts, *current.judgment_artifacts]
    }
    if len(config_shas) != 1:
        raise ValueError("raw/current judgments do not use one identical Judge config")
    run_ids = [
        item["judge_run_id"]
        for item in [*raw.judgment_artifacts, *current.judgment_artifacts]
    ]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("judge_run_id values must be unique across both conditions")

    cases: list[dict[str, Any]] = []
    comparable_deltas: list[float] = []
    raw_call_scores: list[int] = []
    current_call_scores: list[int] = []
    for case_id in raw.order:
        raw_case = _side_case(raw.repeats_by_case[case_id])
        current_case = _side_case(current.repeats_by_case[case_id])
        raw_call_scores.extend(
            score for score in raw_case["scores"] if type(score) is int
        )
        current_call_scores.extend(
            score for score in current_case["scores"] if type(score) is int
        )
        comparable = raw_case["fully_valid"] and current_case["fully_valid"]
        if comparable:
            delta = current_case["mean_score"] - raw_case["mean_score"]
            comparable_deltas.append(delta)
            outcome = "current_win" if delta > 0 else "raw_win" if delta < 0 else "tie"
        else:
            delta = None
            outcome = "not_comparable"
        cases.append(
            {
                "case_id": case_id,
                "query": raw.answers_by_case[case_id].get("query"),
                "raw_answer_id": raw.answers_by_case[case_id]["answer_id"],
                "current_answer_id": current.answers_by_case[case_id]["answer_id"],
                "raw": raw_case,
                "current": current_case,
                "fully_comparable": comparable,
                "delta_current_minus_raw": delta,
                "outcome": outcome,
            }
        )

    comparable = [row for row in cases if row["fully_comparable"]]
    raw_means = [float(row["raw"]["mean_score"]) for row in comparable]
    current_means = [float(row["current"]["mean_score"]) for row in comparable]
    invalid_case_ids = [row["case_id"] for row in cases if not row["fully_comparable"]]
    summary = {
        "question_count": len(cases),
        "effective_sample_n": len(comparable),
        "judge_repeat_count_per_condition": REQUIRED_REPEAT_COUNT,
        "judgment_call_count": len(cases) * REQUIRED_REPEAT_COUNT * 2,
        "fully_comparable_case_ids": [row["case_id"] for row in comparable],
        "not_comparable_case_ids": invalid_case_ids,
        "raw_question_mean_score": mean(raw_means) if raw_means else None,
        "current_question_mean_score": mean(current_means) if current_means else None,
        "mean_delta_current_minus_raw": (
            mean(comparable_deltas) if comparable_deltas else None
        ),
        "current_win_count": sum(delta > 0 for delta in comparable_deltas),
        "tie_count": sum(delta == 0 for delta in comparable_deltas),
        "raw_win_count": sum(delta < 0 for delta in comparable_deltas),
        "raw_invalid_judgment_count": len(cases) * REQUIRED_REPEAT_COUNT
        - len(raw_call_scores),
        "current_invalid_judgment_count": len(cases) * REQUIRED_REPEAT_COUNT
        - len(current_call_scores),
        "raw_valid_call_score_distribution_descriptive_only": _score_distribution(
            raw_call_scores
        ),
        "current_valid_call_score_distribution_descriptive_only": _score_distribution(
            current_call_scores
        ),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_type": "same-source-postprocessor-judge-score-diagnostic",
        "warning": WARNING,
        "eligibility": {
            "diagnostic_only": True,
            "primary_metric": "llm_judge_score_0_1_2",
            "final_service_performance_eligible": False,
            "grounded_fully_correct_eligible": False,
            "citation_performance_eligible": False,
            "retrieval_performance_eligible": False,
        },
        "methodology": {
            "sampling_unit": "question/case_id",
            "judge_repeats_count_as_additional_samples": False,
            "required_repeats_per_condition": REQUIRED_REPEAT_COUNT,
            "fully_comparable_definition": (
                "all three raw and all three current judgments have error=null "
                "and integer judge.score in {0,1,2}"
            ),
            "invalid_judgment_policy": (
                "exclude its entire question from paired means and W/T/L; do "
                "not recover a score from raw_judge_response"
            ),
            "aggregation": (
                "mean repeated score within each fully comparable question, "
                "then arithmetic mean across fully comparable questions"
            ),
            "external_llm_calls_by_analyzer": False,
        },
        "inputs": {
            "raw": {
                "answers": {
                    "path": str(raw.answer_path),
                    "sha256": raw.answer_sha256,
                    "condition_id": raw_first["condition_id"],
                    "projection_mode": RAW_MODE,
                },
                "judgments": raw.judgment_artifacts,
            },
            "current": {
                "answers": {
                    "path": str(current.answer_path),
                    "sha256": current.answer_sha256,
                    "condition_id": current_first["condition_id"],
                    "projection_mode": CURRENT_MODE,
                },
                "judgments": current.judgment_artifacts,
            },
            "experiment_id": raw_first["experiment_id"],
            "generation_run_id": raw_first["generation_run_id"],
            "judge_config_sha256": next(iter(config_shas)),
        },
        "integrity_verification": {
            "answer_record_hashes_validated": True,
            "judgment_record_hashes_validated": True,
            "answer_artifact_hash_bindings_validated": True,
            "answer_record_hash_bindings_validated": True,
            "exact_case_sets_per_judgment_artifact": True,
            "raw_current_case_order_exact": True,
            "same_source_projection_provenance_validated": True,
            "diagnostic_eligibility_gates_validated": True,
            "judge_config_identical_across_conditions": True,
            "raw_judge_response_parsed_for_score": False,
            "external_llm_called_by_analyzer": False,
        },
        "projection_validation_artifact_hashes": {
            "raw": projection_validation["inputs"]["raw"]["artifact_sha256"],
            "current": projection_validation["inputs"]["current"]["artifact_sha256"],
        },
        "summary": summary,
        "cases": cases,
    }
    return payload, cases


def render_csv(rows: Iterable[dict[str, Any]]) -> str:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot render an empty case CSV")
    fieldnames = [
        "case_id",
        "fully_comparable",
        "outcome",
        "raw_scores",
        "current_scores",
        "raw_errors",
        "current_errors",
        "raw_mean_score",
        "current_mean_score",
        "delta_current_minus_raw",
    ]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in materialized:
        writer.writerow(
            {
                "case_id": row["case_id"],
                "fully_comparable": row["fully_comparable"],
                "outcome": row["outcome"],
                "raw_scores": json.dumps(row["raw"]["scores"], separators=(",", ":")),
                "current_scores": json.dumps(
                    row["current"]["scores"], separators=(",", ":")
                ),
                "raw_errors": json.dumps(
                    row["raw"]["errors"], ensure_ascii=False, separators=(",", ":")
                ),
                "current_errors": json.dumps(
                    row["current"]["errors"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "raw_mean_score": row["raw"]["mean_score"],
                "current_mean_score": row["current"]["mean_score"],
                "delta_current_minus_raw": row["delta_current_minus_raw"],
            }
        )
    return handle.getvalue()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-answers", type=Path, required=True)
    parser.add_argument("--current-answers", type=Path, required=True)
    parser.add_argument(
        "--raw-judgments", type=Path, nargs=REQUIRED_REPEAT_COUNT, required=True
    )
    parser.add_argument(
        "--current-judgments", type=Path, nargs=REQUIRED_REPEAT_COUNT, required=True
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [
        args.raw_answers,
        args.current_answers,
        *args.raw_judgments,
        *args.current_judgments,
    ]
    outputs = [args.json_out, *([args.csv_out] if args.csv_out else [])]
    try:
        reject_symlink_inputs(inputs)
        for source in inputs:
            for output in outputs:
                if paths_alias(source, output):
                    raise ValueError("output paths must not overwrite or alias inputs")
        require_new_outputs(outputs)
        payload, rows = build_analysis(
            args.raw_answers,
            args.current_answers,
            args.raw_judgments,
            args.current_judgments,
        )
        csv_text = render_csv(rows) if args.csv_out else None
        payload["output_publication"] = {
            "authoritative_completion_artifact": "json",
            "authoritative_path": str(args.json_out.resolve()),
            "required_companion_artifacts": (
                []
                if args.csv_out is None
                else [
                    {
                        "kind": "case_csv",
                        "path": str(args.csv_out.resolve()),
                        "sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
                        "bytes": len(csv_text.encode("utf-8")),
                    }
                ]
            ),
            "contract": (
                "The authoritative JSON is published only after every required "
                "companion is durably published; all output paths are immutable."
            ),
        }
        json_text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ) + "\n"
        output_texts = {args.json_out: json_text}
        if args.csv_out is not None:
            output_texts[args.csv_out] = csv_text
        for group in (payload["inputs"]["raw"], payload["inputs"]["current"]):
            if sha256_file(Path(group["answers"]["path"])) != group["answers"]["sha256"]:
                raise ValueError("answer artifact changed before publication")
            for artifact in group["judgments"]:
                if sha256_file(Path(artifact["path"])) != artifact["sha256"]:
                    raise ValueError("judgment artifact changed before publication")
        publish_immutable_texts(output_texts, authoritative_path=args.json_out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    summary = payload["summary"]
    print(
        f"questions={summary['question_count']} "
        f"effective_sample_n={summary['effective_sample_n']} "
        f"repeats={summary['judge_repeat_count_per_condition']} "
        f"raw_mean={summary['raw_question_mean_score']} "
        f"current_mean={summary['current_question_mean_score']} "
        f"delta={summary['mean_delta_current_minus_raw']} "
        f"W/T/L={summary['current_win_count']}/{summary['tie_count']}/"
        f"{summary['raw_win_count']} external_llm_calls=false"
    )
    print(f"wrote {args.json_out}")
    if args.csv_out is not None:
        print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
