#!/usr/bin/env python3
"""Create one immutable selection manifest for partial Judge-repeat analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from service_eval_artifacts import sha256_json  # noqa: E402
from immutable_outputs import (  # noqa: E402
    paths_alias,
    publish_immutable_texts,
    reject_symlink_inputs,
    require_new_outputs,
)
from summarize_judge_repeats import (  # noqa: E402
    FINAL_SELECTION_COUNT_PER_CONDITION,
    SELECTION_SCHEMA_VERSION,
    _selection_identity,
    answer_is_judge_eligible,
    load_answers,
    load_selection_manifest,
    sha256_file,
)


def build_selection(answer_path: Path, selected_case_ids: list[str]) -> dict:
    if not selected_case_ids:
        raise ValueError("at least one selected case id is required")
    if any(not case_id for case_id in selected_case_ids):
        raise ValueError("selected case ids must be non-empty")
    if len(set(selected_case_ids)) != len(selected_case_ids):
        raise ValueError("duplicate selected case ids")
    if len(selected_case_ids) != FINAL_SELECTION_COUNT_PER_CONDITION:
        raise ValueError(
            f"selected case count {len(selected_case_ids)} != fixed expected "
            f"{FINAL_SELECTION_COUNT_PER_CONDITION} (C0+C1 total 18)"
        )
    artifact_sha_before = sha256_file(answer_path)
    order, answers_by_case = load_answers(answer_path)
    unknown = sorted(set(selected_case_ids) - set(answers_by_case))
    if unknown:
        raise ValueError(f"unknown selected case ids: {unknown}")
    terminal_case_ids = [
        case_id
        for case_id in selected_case_ids
        if not answer_is_judge_eligible(
            answers_by_case[case_id], path=answer_path
        )
    ]
    if terminal_case_ids:
        raise ValueError(
            "fixed-sample Judge stability diagnosis is incomplete: selected "
            "case IDs contain terminal service-error slots and no replacement "
            f"is allowed: {terminal_case_ids}"
        )
    selected_set = set(selected_case_ids)
    canonical_case_ids = [case_id for case_id in order if case_id in selected_set]
    selected_answer_ids = [
        str(answers_by_case[case_id]["answer_id"]) for case_id in canonical_case_ids
    ]
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "answers_artifact_sha256": artifact_sha_before,
        "selected_answer_ids": selected_answer_ids,
        "selected_answer_ids_sha256": sha256_json(selected_answer_ids),
    }
    payload["selection_sha256"] = sha256_json(_selection_identity(payload))
    payload["selection_id"] = (
        "judge_repeat_selection_" + payload["selection_sha256"][:24]
    )
    if sha256_file(answer_path) != artifact_sha_before:
        raise ValueError("answer artifact changed during selection")
    return payload


def write_new(path: Path, payload: dict) -> None:
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    publish_immutable_texts({path: text}, authoritative_path=path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument(
        "--case-ids",
        required=True,
        help="comma-separated case IDs; output is canonicalized to answer order",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if paths_alias(args.answers, args.out):
        raise SystemExit("validation error: output must not overwrite --answers")
    selected_case_ids = [value.strip() for value in args.case_ids.split(",")]
    try:
        reject_symlink_inputs([args.answers])
        require_new_outputs([args.out])
        payload = build_selection(args.answers, selected_case_ids)
        write_new(args.out, payload)
        # Re-open through the consumer's validator before reporting success.
        order, answers_by_case = load_answers(args.answers)
        load_selection_manifest(
            args.out,
            answers_artifact_sha256=sha256_file(args.answers),
            answer_order=order,
            answers_by_case=answers_by_case,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc
    print(
        f"selection_id={payload['selection_id']} "
        f"selected_answers={len(payload['selected_answer_ids'])}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
