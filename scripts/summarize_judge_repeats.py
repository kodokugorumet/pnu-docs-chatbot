#!/usr/bin/env python3
"""Validate and aggregate repeated LLM judgments for one answer artifact.

The immutable answer JSONL is the sampling unit. In the default full mode,
every judgment JSONL must contain exactly one valid judgment for every
Judge-eligible answer; terminal service-error slots are never judged and are
reported separately. Partial stability subsets require a selection manifest made
by ``build_judge_repeat_selection.py``; ad-hoc partial files are rejected.
Repeated judgments measure judge stability and never increase the
question-level sample size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from service_eval_artifacts import (  # noqa: E402
    join_answers_and_judgments,
    load_unique_jsonl,
    sha256_json,
    validate_answer_record,
)
from final_generation_slots import (  # noqa: E402
    answer_is_judge_eligible as validate_slot_disposition,
)
from immutable_outputs import (  # noqa: E402
    publish_immutable_texts,
    reject_symlink_inputs,
    require_new_outputs,
)


SUMMARY_SCHEMA_VERSION = "pnu.judge-repeat-summary.v1"
SELECTION_SCHEMA_VERSION = "pnu.judge-repeat-selection.v1"
FINAL_SELECTION_COUNT_PER_CONDITION = 9
FINAL_REPEAT_JUDGE_RUN_COUNT = 2
FINAL_CORE_ANSWER_COUNT = 27
FINAL_AUTHORIZATION_SCHEMA_VERSION = "pnu.final-generation-authorization.v1"
FINAL_CORE_SPLIT = "holdout-core"


def _selection_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "answers_artifact_sha256": payload["answers_artifact_sha256"],
        "selected_answer_ids": payload["selected_answer_ids"],
        "selected_answer_ids_sha256": payload["selected_answer_ids_sha256"],
    }


def load_selection_manifest(
    path: Path,
    *,
    answers_artifact_sha256: str,
    answer_order: list[str],
    answers_by_case: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    """Validate one immutable, answer-artifact-bound partial-repeat selection."""

    artifact_sha_before = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: selection manifest must be a JSON object")
    required = {
        "schema_version",
        "selection_id",
        "selection_sha256",
        "answers_artifact_sha256",
        "selected_answer_ids",
        "selected_answer_ids_sha256",
    }
    missing = sorted(required - payload.keys())
    unknown = sorted(payload.keys() - required)
    if missing:
        raise ValueError(
            f"{path}: selection manifest missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ValueError(
            f"{path}: selection manifest has unknown fields: {', '.join(unknown)}"
        )
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {SELECTION_SCHEMA_VERSION!r}"
        )
    if payload.get("answers_artifact_sha256") != answers_artifact_sha256:
        raise ValueError(f"{path}: answers_artifact_sha256 mismatch")

    selected_ids = payload.get("selected_answer_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError(f"{path}: selected_answer_ids must be a non-empty list")
    if any(not isinstance(value, str) or not value for value in selected_ids):
        raise ValueError(
            f"{path}: selected_answer_ids must contain non-empty strings"
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"{path}: duplicate selected_answer_ids")
    expected_ids_hash = sha256_json(selected_ids)
    if payload.get("selected_answer_ids_sha256") != expected_ids_hash:
        raise ValueError(f"{path}: selected_answer_ids_sha256 mismatch")

    answer_by_id = {
        str(answer["answer_id"]): answer for answer in answers_by_case.values()
    }
    unknown_ids = sorted(set(selected_ids) - set(answer_by_id))
    if unknown_ids:
        raise ValueError(
            f"{path}: selected_answer_ids contain unknown ids: {unknown_ids}"
        )
    full_answer_order = [
        str(answers_by_case[case_id]["answer_id"]) for case_id in answer_order
    ]
    canonical_selected_ids = [
        answer_id for answer_id in full_answer_order if answer_id in set(selected_ids)
    ]
    if selected_ids != canonical_selected_ids:
        raise ValueError(
            f"{path}: selected_answer_ids must preserve answer artifact order"
        )

    expected_selection_sha = sha256_json(_selection_identity(payload))
    if payload.get("selection_sha256") != expected_selection_sha:
        raise ValueError(f"{path}: selection_sha256 mismatch")
    expected_selection_id = f"judge_repeat_selection_{expected_selection_sha[:24]}"
    if payload.get("selection_id") != expected_selection_id:
        raise ValueError(
            f"{path}: selection_id mismatch; expected {expected_selection_id}"
        )
    if sha256_file(path) != artifact_sha_before:
        raise ValueError(f"{path}: selection manifest changed during validation")

    selected_by_case = {
        str(answer_by_id[answer_id]["case_id"]): answer_by_id[answer_id]
        for answer_id in selected_ids
    }
    selected_order = [str(answer_by_id[value]["case_id"]) for value in selected_ids]
    final_authorization_flags = {
        isinstance(answer.get("collector_config"), dict)
        and isinstance(
            answer["collector_config"].get("final_authorization"), dict
        )
        for answer in answers_by_case.values()
    }
    if len(final_authorization_flags) != 1:
        raise ValueError(f"{path}: answer artifact mixes final authorization state")
    final_authorized = next(iter(final_authorization_flags))
    if final_authorized:
        if len(selected_ids) != FINAL_SELECTION_COUNT_PER_CONDITION:
            raise ValueError(
                f"{path}: final selection has {len(selected_ids)} answers; expected "
                f"fixed {FINAL_SELECTION_COUNT_PER_CONDITION} per condition"
            )
        final_authorizations = [
            answer["collector_config"]["final_authorization"]
            for answer in answers_by_case.values()
        ]
        authorization = final_authorizations[0]
        if any(value != authorization for value in final_authorizations[1:]):
            raise ValueError(
                f"{path}: final answer artifact mixes authorization metadata"
            )
        condition_ids = {
            str(answer.get("condition_id") or "")
            for answer in answers_by_case.values()
        }
        generation_run_ids = {
            str(answer.get("generation_run_id") or "")
            for answer in answers_by_case.values()
        }
        if condition_ids not in ({"c0"}, {"c1"}):
            raise ValueError(
                f"{path}: final stability selection must be C0 or C1 Core"
            )
        condition_id = next(iter(condition_ids))
        if generation_run_ids != {"run1"}:
            raise ValueError(
                f"{path}: final stability selection must use generation run1"
            )
        if (
            authorization.get("schema_version")
            != FINAL_AUTHORIZATION_SCHEMA_VERSION
            or authorization.get("collection_purpose") != "final_generation"
            or authorization.get("split") != FINAL_CORE_SPLIT
            or authorization.get("artifact_id") != f"{condition_id}-core-run1"
            or authorization.get("expected_case_count") != FINAL_CORE_ANSWER_COUNT
            or len(full_answer_order) != FINAL_CORE_ANSWER_COUNT
            or authorization.get("expected_case_ids_sha256")
            != sha256_json(answer_order)
        ):
            raise ValueError(
                f"{path}: final stability selection is not bound to the complete "
                "27-answer run1 Core artifact"
            )
    terminal_selected = [
        case_id
        for case_id in selected_order
        if not answer_is_judge_eligible(selected_by_case[case_id], path=path)
    ]
    if terminal_selected:
        raise ValueError(
            f"{path}: fixed-sample Judge stability diagnosis is incomplete; "
            "selection contains terminal service-error slots and replacement "
            f"is forbidden: {terminal_selected}"
        )
    metadata = {
        "path": str(path),
        "sha256": artifact_sha_before,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_id": expected_selection_id,
        "selection_sha256": expected_selection_sha,
        "answers_artifact_sha256": answers_artifact_sha256,
        "selected_answer_count": len(selected_ids),
        "selected_answer_ids_sha256": expected_ids_hash,
        "final_protocol_selection": final_authorized,
    }
    return metadata, selected_order, selected_by_case


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answer_is_judge_eligible(answer: dict[str, Any], *, path: Path) -> bool:
    return validate_slot_disposition(
        answer,
        source=path,
        allow_legacy=True,
    )


def load_answers(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    records = load_unique_jsonl(path, key="answer_id")
    if not records:
        raise ValueError(f"{path}: answer artifact is empty")

    order: list[str] = []
    answers_by_case: dict[str, dict[str, Any]] = {}
    for answer in records:
        validate_answer_record(answer)
        case_id = str(answer["case_id"])
        if answer.get("error") is not None:
            raise ValueError(f"{path}: answer {case_id} has error: {answer['error']}")
        if "judge" in answer:
            raise ValueError(
                f"{path}: answer {case_id} contains inline judge output"
            )
        if case_id in answers_by_case:
            raise ValueError(f"{path}: duplicate case_id {case_id}")
        order.append(case_id)
        answers_by_case[case_id] = answer

    for field in ("experiment_id", "condition_id", "generation_run_id"):
        values = {str(answer[field]) for answer in records}
        if len(values) != 1:
            raise ValueError(f"{path}: answer artifact mixes {field}")
    return order, answers_by_case


def _validate_judgment_bindings(
    *,
    path: Path,
    records: list[dict[str, Any]],
    answers_by_case: dict[str, dict[str, Any]],
    answers_artifact_sha256: str,
    selection_metadata: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    if not records:
        raise ValueError(f"{path}: judgment artifact is empty")

    run_ids = {str(record.get("judge_run_id") or "") for record in records}
    if len(run_ids) != 1 or "" in run_ids:
        raise ValueError(f"{path}: judgment artifact mixes judge_run_id")
    judge_run_id = next(iter(run_ids))

    config_hashes = {
        str(record.get("judge_config_sha256") or "") for record in records
    }
    if len(config_hashes) > 1:
        raise ValueError(f"{path}: judgment artifact mixes judge_config_sha256")
    judge_config_sha256 = next(iter(config_hashes)) or None
    final_authorized = bool(answers_by_case) and all(
        isinstance(answer.get("collector_config"), dict)
        and isinstance(
            answer["collector_config"].get("final_authorization"), dict
        )
        for answer in answers_by_case.values()
    )
    if final_authorized and (
        judge_config_sha256 is None
        or any(
            not isinstance(record.get("judge_config"), dict)
            or not record["judge_config"]
            for record in records
        )
    ):
        raise ValueError(f"{path}: final judgments omit pinned judge_config")

    expected_selection_binding: dict[str, Any] | None = None
    if selection_metadata is not None:
        expected_selection_binding = {
            "schema_version": selection_metadata["schema_version"],
            "selection_id": selection_metadata["selection_id"],
            "selection_sha256": selection_metadata["selection_sha256"],
            "selection_artifact_sha256": selection_metadata["sha256"],
            "answers_artifact_sha256": answers_artifact_sha256,
            "selected_answer_count": selection_metadata["selected_answer_count"],
            "selected_answer_ids_sha256": selection_metadata[
                "selected_answer_ids_sha256"
            ],
        }

    for record in records:
        case_id = str(record.get("case_id") or "<missing>")
        if record.get("error") is not None:
            raise ValueError(
                f"{path}: judgment {case_id} has error: {record['error']}"
            )
        observed_selection = record.get("judge_repeat_selection")
        if observed_selection != expected_selection_binding:
            if expected_selection_binding is None:
                raise ValueError(
                    f"{path}: selection-bound judgment requires "
                    "--selection-manifest"
                )
            raise ValueError(
                f"{path}: judge_repeat_selection mismatch for {case_id}"
            )

    # This existing helper validates the judgment schema and record hash, then
    # requires an exact one-to-one answer_id/case_id/experiment/answer_sha256
    # join.  It rejects missing, extra, and duplicate judgments.
    joined = join_answers_and_judgments(answers_by_case, records)

    records_by_answer = {str(record["answer_id"]): record for record in records}
    for case_id, merged in joined.items():
        answer = answers_by_case[case_id]
        judgment = records_by_answer[str(answer["answer_id"])]

        for field in ("condition_id", "generation_run_id"):
            if field in judgment and judgment.get(field) != answer.get(field):
                raise ValueError(
                    f"{path}: {field} mismatch for answer_id {answer['answer_id']}"
                )

        bound_record_sha = judgment.get("answer_record_sha256")
        if bound_record_sha is not None and bound_record_sha != sha256_json(answer):
            raise ValueError(
                f"{path}: answer_record_sha256 mismatch for answer_id "
                f"{answer['answer_id']}"
            )
        bound_artifact_sha = judgment.get("answers_artifact_sha256")
        if (
            bound_artifact_sha is not None
            and bound_artifact_sha != answers_artifact_sha256
        ):
            raise ValueError(
                f"{path}: answers_artifact_sha256 mismatch for answer_id "
                f"{answer['answer_id']}"
            )

        judge = merged["judge"]
        score = judge.get("score")
        if type(score) is not int or score not in (0, 1, 2):
            raise ValueError(
                f"{path}: invalid judge score for answer_id {answer['answer_id']}"
            )
        gfc = judge.get("grounded_fully_correct")
        if type(gfc) is not bool:
            raise ValueError(
                f"{path}: invalid or missing grounded_fully_correct for "
                f"answer_id {answer['answer_id']}"
            )
        if gfc and score != 2:
            raise ValueError(
                f"{path}: grounded_fully_correct=true requires score=2 for "
                f"answer_id {answer['answer_id']}"
            )
    return judge_run_id, judge_config_sha256


def load_judgment_run(
    path: Path,
    *,
    answers_by_case: dict[str, dict[str, Any]],
    answers_artifact_sha256: str,
    selection_metadata: dict[str, Any] | None = None,
) -> tuple[str, str | None, str, dict[str, dict[str, Any]]]:
    artifact_sha_before = sha256_file(path)
    records = load_unique_jsonl(path, key="judgment_id")
    judge_run_id, judge_config_sha256 = _validate_judgment_bindings(
        path=path,
        records=records,
        answers_by_case=answers_by_case,
        answers_artifact_sha256=answers_artifact_sha256,
        selection_metadata=selection_metadata,
    )
    artifact_sha_after = sha256_file(path)
    if artifact_sha_after != artifact_sha_before:
        raise ValueError(f"{path}: judgment artifact changed during validation")
    return (
        judge_run_id,
        judge_config_sha256,
        artifact_sha_before,
        join_answers_and_judgments(answers_by_case, records),
    )


def unique_mode(values: list[int]) -> int | None:
    counts = Counter(values)
    highest = max(counts.values())
    winners = [value for value, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else None


def strict_boolean_majority(true_count: int, total: int) -> bool | None:
    if true_count * 2 > total:
        return True
    if (total - true_count) * 2 > total:
        return False
    return None


def aggregate_repeats(
    *,
    answer_path: Path,
    judgment_paths: list[Path],
    selection_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not judgment_paths:
        raise ValueError("at least one judgment artifact is required")

    answer_artifact_sha = sha256_file(answer_path)
    full_order, full_answers_by_case = load_answers(answer_path)
    if sha256_file(answer_path) != answer_artifact_sha:
        raise ValueError(f"{answer_path}: answer artifact changed during validation")

    selection_metadata: dict[str, Any] | None = None
    if selection_manifest_path is None:
        logical_order = full_order
        logical_answers_by_case = full_answers_by_case
    else:
        (
            selection_metadata,
            logical_order,
            logical_answers_by_case,
        ) = load_selection_manifest(
            selection_manifest_path,
            answers_artifact_sha256=answer_artifact_sha,
            answer_order=full_order,
            answers_by_case=full_answers_by_case,
        )
        if (
            selection_metadata["final_protocol_selection"]
            and len(judgment_paths) != FINAL_REPEAT_JUDGE_RUN_COUNT
        ):
            raise ValueError(
                "final partial Judge stability summary requires exactly "
                f"{FINAL_REPEAT_JUDGE_RUN_COUNT} additional judgment artifacts"
            )
    terminal_case_ids = [
        case_id
        for case_id in logical_order
        if not answer_is_judge_eligible(
            logical_answers_by_case[case_id], path=answer_path
        )
    ]
    terminal_set = set(terminal_case_ids)
    order = [case_id for case_id in logical_order if case_id not in terminal_set]
    answers_by_case = {
        case_id: logical_answers_by_case[case_id] for case_id in order
    }
    if not order:
        raise ValueError("selection contains no Judge-eligible answers")

    runs: list[dict[str, dict[str, Any]]] = []
    run_metadata: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for path in judgment_paths:
        judge_run_id, config_sha, judgment_artifact_sha, joined = load_judgment_run(
            path,
            answers_by_case=answers_by_case,
            answers_artifact_sha256=answer_artifact_sha,
            selection_metadata=selection_metadata,
        )
        if judge_run_id in seen_run_ids:
            raise ValueError(f"duplicate judge_run_id across artifacts: {judge_run_id}")
        seen_run_ids.add(judge_run_id)
        runs.append(joined)
        run_metadata.append(
            {
                "path": str(path),
                "sha256": judgment_artifact_sha,
                "judge_run_id": judge_run_id,
                "judge_config_sha256": config_sha,
            }
        )
    if sha256_file(answer_path) != answer_artifact_sha:
        raise ValueError(f"{answer_path}: answer artifact changed during aggregation")

    rows: list[dict[str, Any]] = []
    agreement_counts: Counter[str] = Counter()
    for case_id in order:
        answer = answers_by_case[case_id]
        scores = [int(run[case_id]["judge"]["score"]) for run in runs]
        gfc_values = [
            bool(run[case_id]["judge"]["grounded_fully_correct"])
            for run in runs
        ]
        gfc_true_count = sum(gfc_values)
        agreement_count = max(gfc_true_count, len(gfc_values) - gfc_true_count)
        agreement_label = f"{agreement_count}/{len(gfc_values)}"
        agreement_counts[agreement_label] += 1
        rows.append(
            {
                "case_id": case_id,
                "answer_id": answer["answer_id"],
                "scores": scores,
                "score_mode": unique_mode(scores),
                "score_mean": mean(scores),
                "grounded_fully_correct_values": gfc_values,
                "gfc_true_count": gfc_true_count,
                "gfc_majority": strict_boolean_majority(
                    gfc_true_count, len(gfc_values)
                ),
                "gfc_agreement_count": agreement_count,
                "gfc_agreement_label": agreement_label,
            }
        )

    majority_true = sum(row["gfc_majority"] is True for row in rows)
    majority_false = sum(row["gfc_majority"] is False for row in rows)
    no_majority = len(rows) - majority_true - majority_false
    agreement_counts_sorted = {
        label: agreement_counts[label]
        for label in sorted(
            agreement_counts,
            key=lambda value: tuple(int(part) for part in value.split("/")),
            reverse=True,
        )
    }
    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "inputs": {
            "answers": {
                "path": str(answer_path),
                "sha256": answer_artifact_sha,
            },
            "judgments": run_metadata,
        },
        "methodology": {
            "sample_unit": "question",
            "judge_repeats_count_as_additional_samples": False,
            "effective_sample_n": len(rows),
            "note": (
                "Judge repeats estimate judgment stability and do not increase "
                "the question-level statistical sample size."
            ),
            "score_aggregation": (
                "Per-question arithmetic mean, then arithmetic mean across questions"
            ),
            "score_mode_tie": None,
            "gfc_majority": "strict majority; an even split is null",
            "gfc_agreement": "larger binary vote count / judge repeat count",
        },
        "summary": {
            "question_count": len(rows),
            "effective_sample_n": len(rows),
            "judge_repeat_count": len(runs),
            "judgment_record_count": len(rows) * len(runs),
            "mean_score": mean(float(row["score_mean"]) for row in rows),
            "score_mode_tie_count": sum(row["score_mode"] is None for row in rows),
            "question_level_majority_gfc_true_count": majority_true,
            "question_level_majority_gfc_false_count": majority_false,
            "question_level_gfc_no_majority_count": no_majority,
            "question_level_majority_gfc_rate": majority_true / len(rows),
            "gfc_agreement_counts": agreement_counts_sorted,
            "gfc_agreement_rates": {
                label: count / len(rows)
                for label, count in agreement_counts_sorted.items()
            },
        },
        "cases": rows,
    }
    if selection_metadata is not None:
        payload["inputs"]["selection"] = selection_metadata
        payload["methodology"]["selection_mode"] = (
            "explicit immutable answer-id selection"
        )
        payload["methodology"]["full_answer_artifact_question_count"] = len(
            full_order
        )
        if selection_metadata["final_protocol_selection"]:
            payload["methodology"]["final_protocol_repeat_judge_run_count"] = (
                FINAL_REPEAT_JUDGE_RUN_COUNT
            )
    if terminal_case_ids:
        payload["methodology"]["selected_logical_slot_count"] = len(
            logical_order
        )
        payload["methodology"]["terminal_service_errors_are_judged"] = False
        payload["summary"]["terminal_service_error_count"] = len(
            terminal_case_ids
        )
        payload["summary"]["terminal_service_error_case_ids"] = (
            terminal_case_ids
        )
    return payload, rows


def render_csv(rows: Iterable[dict[str, Any]]) -> str:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty case CSV")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
    writer.writeheader()
    for row in materialized:
        serialized = dict(row)
        for field in ("scores", "grounded_fully_correct_values"):
            serialized[field] = json.dumps(serialized[field], separators=(",", ":"))
        writer.writerow(serialized)
    return handle.getvalue()


def _reject_input_output_aliases(
    *,
    answer_path: Path,
    judgment_paths: list[Path],
    json_out: Path,
    csv_out: Path | None,
    selection_manifest_path: Path | None = None,
) -> None:
    def aliases(left: Path, right: Path) -> bool:
        if left.resolve() == right.resolve():
            return True
        try:
            return left.exists() and right.exists() and left.samefile(right)
        except OSError:
            return False

    input_paths = [answer_path, *judgment_paths]
    if selection_manifest_path is not None:
        input_paths.append(selection_manifest_path)
    output_paths = [json_out]
    if csv_out is not None:
        output_paths.append(csv_out)
    if any(
        aliases(left, right)
        for index, left in enumerate(output_paths)
        for right in output_paths[index + 1 :]
    ):
        raise ValueError("--json-out and --csv-out must be different files")
    if any(aliases(source, target) for source in input_paths for target in output_paths):
        raise ValueError("output paths must not overwrite answer/judgment inputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, nargs="+", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help=(
            "enable partial-repeat mode using a pnu.judge-repeat-selection.v1 "
            "manifest bound to the complete --answers artifact"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inputs = [args.answers, *args.judgments]
    if args.selection_manifest is not None:
        inputs.append(args.selection_manifest)
    outputs = [args.json_out]
    if args.csv_out is not None:
        outputs.append(args.csv_out)
    try:
        reject_symlink_inputs(inputs)
        _reject_input_output_aliases(
            answer_path=args.answers,
            judgment_paths=args.judgments,
            json_out=args.json_out,
            csv_out=args.csv_out,
            selection_manifest_path=args.selection_manifest,
        )
        require_new_outputs(outputs)
        payload, rows = aggregate_repeats(
            answer_path=args.answers,
            judgment_paths=args.judgments,
            selection_manifest_path=args.selection_manifest,
        )
        csv_text = render_csv(rows) if args.csv_out is not None else None
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
                        "sha256": hashlib.sha256(
                            csv_text.encode("utf-8")
                        ).hexdigest(),
                        "bytes": len(csv_text.encode("utf-8")),
                    }
                ]
            ),
            "contract": (
                "The authoritative JSON is published only after every required "
                "companion is durably published; all output paths are immutable."
            ),
        }
        json_text = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        output_texts = {args.json_out: json_text}
        if args.csv_out is not None:
            output_texts[args.csv_out] = csv_text
        publish_immutable_texts(
            output_texts,
            authoritative_path=args.json_out,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    summary = payload["summary"]
    print(
        f"questions={summary['question_count']} "
        f"judge_repeats={summary['judge_repeat_count']} "
        f"effective_sample_n={summary['effective_sample_n']} "
        f"mean_score={summary['mean_score']:.4f} "
        f"majority_gfc_rate={summary['question_level_majority_gfc_rate']:.4f}"
    )
    print(f"wrote {args.json_out}")
    if args.csv_out is not None:
        print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
