#!/usr/bin/env python3
"""Build a reproducibility manifest for final-report evidence artifacts.

The large ``processed/`` and ``downloads/`` trees are ignored by Git. This command
records the small set of files supporting a reported result, validates JSONL
completeness, and writes hashes plus execution metadata to a tracked location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from service_eval_artifacts import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    validate_answer_record,
    validate_judgment_record,
)
from final_generation_slots import (  # noqa: E402
    answer_is_judge_eligible as validate_slot_disposition,
)
from immutable_outputs import (  # noqa: E402
    publish_immutable_texts,
    reject_symlink_inputs,
    require_new_outputs,
)


SELECTION_SCHEMA_VERSION = "pnu.judge-repeat-selection.v1"
FINAL_SELECTION_COUNT_PER_CONDITION = 9
FINAL_REPEAT_JUDGE_RUN_COUNT = 2
FINAL_CORE_ANSWER_COUNT = 27
FINAL_AUTHORIZATION_SCHEMA_VERSION = "pnu.final-generation-authorization.v1"
FINAL_CORE_SPLIT = "holdout-core"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def jsonl_stats(
    path: Path,
    expected_count: int | None = None,
    require_judge_scores: bool = False,
) -> dict[str, Any]:
    records = load_jsonl(path)
    record_types = {
        str(record.get("record_type") or "")
        for record in records
        if record.get("record_type")
    }
    human_label_markers = [
        record.get("schema_version") == "pnu.human-answer-label.v1"
        or record.get("record_type") == "human_label"
        or "blind_item_id" in record
        or (
            "label_kind" in record
            and "reviewer_id" in record
            and "answer_id" in record
        )
        for record in records
    ]
    is_human_labels = bool(records) and any(human_label_markers)
    if record_types == {"answer"}:
        id_field = "answer_id"
    elif record_types == {"judgment"}:
        id_field = "judgment_id"
    elif is_human_labels:
        id_field = "answer_id"
    else:
        id_field = "id"
    ids = [record.get(id_field) for record in records]
    missing_ids = sum(value is None or value == "" for value in ids)
    normalized_ids = [str(value) for value in ids if value is not None and value != ""]
    id_counts: dict[str, int] = {}
    for value in normalized_ids:
        id_counts[value] = id_counts.get(value, 0) + 1
    duplicate_ids = sorted(value for value, count in id_counts.items() if count > 1)
    explicit_errors = sum(bool(record.get("error")) for record in records)

    judge_scores: dict[str, int] = {}
    missing_judge_scores = 0
    invalid_judge_scores = 0
    for record in records:
        judge = record.get("judge")
        score = judge.get("score") if isinstance(judge, dict) else None
        if score is None:
            missing_judge_scores += 1
        elif score not in (0, 1, 2):
            invalid_judge_scores += 1
        else:
            key = str(score)
            judge_scores[key] = judge_scores.get(key, 0) + 1

    problems: list[str] = []
    if expected_count is not None and len(records) != expected_count:
        problems.append(f"record count {len(records)} != expected {expected_count}")
    if missing_ids:
        problems.append(f"{missing_ids} record(s) have no {id_field}")
    if duplicate_ids:
        problems.append(f"duplicate ids: {', '.join(duplicate_ids)}")
    missing_blind_item_ids = 0
    duplicate_blind_item_ids: list[str] = []
    invalid_human_label_identity_records = 0
    if is_human_labels:
        invalid_human_label_identity_records = sum(
            record.get("schema_version") != "pnu.human-answer-label.v1"
            or not isinstance(record.get("answer_id"), str)
            or not str(record.get("answer_id") or "").strip()
            or not isinstance(record.get("blind_item_id"), str)
            or not str(record.get("blind_item_id") or "").strip()
            for record in records
        )
        blind_ids = [record.get("blind_item_id") for record in records]
        missing_blind_item_ids = sum(
            not isinstance(value, str) or not value.strip() for value in blind_ids
        )
        normalized_blind_ids = [
            str(value).strip()
            for value in blind_ids
            if isinstance(value, str) and value.strip()
        ]
        blind_counts: dict[str, int] = {}
        for value in normalized_blind_ids:
            blind_counts[value] = blind_counts.get(value, 0) + 1
        duplicate_blind_item_ids = sorted(
            value for value, count in blind_counts.items() if count > 1
        )
        if missing_blind_item_ids:
            problems.append(
                f"{missing_blind_item_ids} human label record(s) have no blind_item_id"
            )
        if duplicate_blind_item_ids:
            problems.append(
                "duplicate blind_item_id values: "
                + ", ".join(duplicate_blind_item_ids)
            )
        if invalid_human_label_identity_records:
            problems.append(
                f"{invalid_human_label_identity_records} record(s) in a human-label "
                "JSONL lack the required schema/answer_id/blind_item_id identity"
            )
    if require_judge_scores and explicit_errors:
        problems.append(f"{explicit_errors} record(s) have an explicit error")
    if require_judge_scores and missing_judge_scores:
        problems.append(f"{missing_judge_scores} record(s) have no judge score")
    if require_judge_scores and invalid_judge_scores:
        problems.append(f"{invalid_judge_scores} record(s) have an invalid judge score")

    return {
        "records": len(records),
        "id_field": id_field,
        "unique_ids": len(set(normalized_ids)),
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "human_label_identity_required": is_human_labels,
        "missing_blind_item_ids": missing_blind_item_ids,
        "duplicate_blind_item_ids": duplicate_blind_item_ids,
        "invalid_human_label_identity_records": (
            invalid_human_label_identity_records
        ),
        "explicit_errors": explicit_errors,
        "judge_scores": judge_scores,
        "missing_judge_scores": missing_judge_scores,
        "invalid_judge_scores": invalid_judge_scores,
        "judge_scores_required": require_judge_scores,
        "expected_records": expected_count,
        "complete": not problems,
        "problems": problems,
    }


def answer_is_judge_eligible(record: dict[str, Any]) -> bool:
    """Validate answer/service-error disposition, preserving legacy DEV rows."""
    return validate_slot_disposition(
        record,
        source="evidence manifest answer",
        allow_legacy=True,
    )


def validate_eval_artifact_links(
    records: Iterable[dict[str, Any]],
    *,
    partial_selections: Sequence[dict[str, Any]] = (),
    answer_artifact_shas: dict[str, set[str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Cross-check immutable answer and judgment records in one manifest."""

    answers = [record for record in records if record.get("record_type") == "answer"]
    judgments = [
        record for record in records if record.get("record_type") == "judgment"
    ]
    problems: list[str] = []
    answer_by_id: dict[str, dict[str, Any]] = {}
    judge_eligible_by_answer_id: dict[str, bool] = {}
    final_slot_by_answer_id: dict[str, bool] = {}
    duplicate_answer_ids: set[str] = set()
    for answer in answers:
        answer_id = str(answer.get("answer_id") or "")
        if not answer_id:
            problems.append("answer record is missing answer_id")
            continue
        if answer_id in answer_by_id:
            duplicate_answer_ids.add(answer_id)
        answer_by_id[answer_id] = answer
        final_slot_by_answer_id[answer_id] = (
            answer.get("slot_outcome") is not None
            or answer.get("answer_eligible_for_judge") is not None
            or answer.get("service_error") is not None
        )
        if answer.get("error") is not None:
            problems.append(f"answer {answer_id} has an explicit error")
        if (
            answer.get("schema_version") == ANSWER_SCHEMA_VERSION
            or answer.get("slot_outcome") is not None
            or answer.get("answer_eligible_for_judge") is not None
        ):
            try:
                validate_answer_record(answer)
            except ValueError as exc:
                problems.append(f"answer {answer_id} identity/hash invalid: {exc}")
        try:
            judge_eligible_by_answer_id[answer_id] = answer_is_judge_eligible(answer)
        except ValueError as exc:
            judge_eligible_by_answer_id[answer_id] = False
            problems.append(f"answer {answer_id} has invalid slot disposition: {exc}")
    if duplicate_answer_ids:
        problems.append(
            "duplicate answer_id across artifacts: "
            + ", ".join(sorted(duplicate_answer_ids))
        )

    seen_judgments: set[str] = set()
    duplicate_judgment_ids: set[str] = set()
    judgment_groups: dict[
        tuple[str, str, str, str], dict[str, Any]
    ] = {}
    for judgment in judgments:
        judgment_id = str(judgment.get("judgment_id") or "")
        if not judgment_id:
            problems.append("judgment record is missing judgment_id")
        elif judgment_id in seen_judgments:
            duplicate_judgment_ids.add(judgment_id)
        seen_judgments.add(judgment_id)
        if judgment.get("schema_version") == JUDGMENT_SCHEMA_VERSION:
            try:
                validate_judgment_record(judgment)
            except ValueError as exc:
                problems.append(
                    f"judgment {judgment_id or '<missing>'} identity/hash invalid: {exc}"
                )

        answer_id = str(judgment.get("answer_id") or "")
        answer = answer_by_id.get(answer_id)
        if answer is None:
            problems.append(
                f"judgment {judgment_id or '<missing>'} references unknown "
                f"answer_id {answer_id or '<missing>'}"
            )
            continue
        is_final_slot = final_slot_by_answer_id.get(answer_id, False)
        if judgment.get("error") is not None:
            problems.append(
                f"judgment {judgment_id or '<missing>'} has an explicit error"
            )
        if is_final_slot and judgment.get("schema_version") != JUDGMENT_SCHEMA_VERSION:
            problems.append(
                f"final judgment {judgment_id or '<missing>'} has invalid schema_version"
            )
        if not judge_eligible_by_answer_id.get(answer_id, False):
            problems.append(
                f"judgment {judgment_id or '<missing>'} references terminal "
                f"service-error answer_id {answer_id}"
            )
        expected_artifact_shas = (
            answer_artifact_shas.get(answer_id, set())
            if answer_artifact_shas is not None
            else set()
        )
        if answer_artifact_shas is not None and (
            len(expected_artifact_shas) != 1
            or judgment.get("answers_artifact_sha256")
            not in expected_artifact_shas
        ):
            problems.append(
                f"answers_artifact_sha256 mismatch for judgment "
                f"{judgment_id or '<missing>'} and answer_id {answer_id}"
            )
        for field in (
            "experiment_id",
            "condition_id",
            "generation_run_id",
            "case_id",
            "answer_sha256",
        ):
            if judgment.get(field) != answer.get(field):
                problems.append(
                    f"{field} mismatch for judgment {judgment_id or '<missing>'} "
                    f"and answer_id {answer_id}"
                )
        expected_answer_record_sha = hashlib.sha256(
            json.dumps(
                answer,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if (
            is_final_slot
            and judgment.get("answer_record_sha256")
            != expected_answer_record_sha
        ) or (
            not is_final_slot
            and judgment.get("answer_record_sha256") is not None
            and judgment.get("answer_record_sha256")
            != expected_answer_record_sha
        ):
            problems.append(
                f"answer_record_sha256 mismatch for judgment "
                f"{judgment_id or '<missing>'} and answer_id {answer_id}"
            )
        if is_final_slot:
            judge_config = judgment.get("judge_config")
            if not isinstance(judge_config, dict) or not judge_config:
                problems.append(
                    f"final judgment {judgment_id or '<missing>'} omits judge_config"
                )
            judge = judgment.get("judge")
            score = judge.get("score") if isinstance(judge, dict) else None
            gfc = (
                judge.get("grounded_fully_correct")
                if isinstance(judge, dict)
                else None
            )
            if type(score) is not int or score not in (0, 1, 2):
                problems.append(
                    f"final judgment {judgment_id or '<missing>'} has invalid score"
                )
            if type(gfc) is not bool or (gfc and score != 2):
                problems.append(
                    f"final judgment {judgment_id or '<missing>'} has invalid GFC"
                )
        group = (
            str(judgment.get("experiment_id") or ""),
            str(judgment.get("condition_id") or ""),
            str(judgment.get("generation_run_id") or ""),
            str(judgment.get("judge_run_id") or ""),
        )
        group_data = judgment_groups.setdefault(
            group,
            {
                "answer_ids": set(),
                "answer_id_counts": {},
                "answers_artifact_sha256s": set(),
                "missing_answers_artifact_sha256": 0,
                "judge_repeat_selection_values": set(),
                "missing_judge_repeat_selection": 0,
            },
        )
        group_data["answer_ids"].add(answer_id)
        group_data["answer_id_counts"][answer_id] = (
            group_data["answer_id_counts"].get(answer_id, 0) + 1
        )
        if judgment.get("answers_artifact_sha256") is not None:
            group_data["answers_artifact_sha256s"].add(
                str(judgment.get("answers_artifact_sha256"))
            )
        else:
            group_data["missing_answers_artifact_sha256"] += 1
        if judgment.get("judge_repeat_selection") is None:
            group_data["missing_judge_repeat_selection"] += 1
        else:
            group_data["judge_repeat_selection_values"].add(
                json.dumps(
                    judgment.get("judge_repeat_selection"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )

    if duplicate_judgment_ids:
        problems.append(
            "duplicate judgment_id across artifacts: "
            + ", ".join(sorted(duplicate_judgment_ids))
        )

    answer_groups: dict[tuple[str, str, str], set[str]] = {}
    for answer_id, answer in answer_by_id.items():
        group = (
            str(answer.get("experiment_id") or ""),
            str(answer.get("condition_id") or ""),
            str(answer.get("generation_run_id") or ""),
        )
        if judge_eligible_by_answer_id.get(answer_id, False):
            answer_groups.setdefault(group, set()).add(answer_id)
    selection_use_counts = {
        str(selection["selection_id"]): 0 for selection in partial_selections
    }
    for judgment_group, group_data in judgment_groups.items():
        judged_ids = group_data["answer_ids"]
        duplicate_answer_references = sorted(
            answer_id
            for answer_id, count in group_data["answer_id_counts"].items()
            if count > 1
        )
        if duplicate_answer_references:
            problems.append(
                f"judgment group {judgment_group} contains multiple judgments "
                "for answer_id(s): " + ", ".join(duplicate_answer_references)
            )
        answer_group = judgment_group[:3]
        expected_ids = answer_groups.get(answer_group, set())
        missing = sorted(expected_ids - judged_ids)
        extra = sorted(judged_ids - expected_ids)
        selection_fields_observed = bool(
            group_data["judge_repeat_selection_values"]
        )
        if missing or extra or selection_fields_observed:
            matching = [
                selection
                for selection in partial_selections
                if tuple(selection["answer_group"]) == answer_group
                and set(selection["selected_judge_eligible_answer_ids"])
                == judged_ids
                and group_data["missing_answers_artifact_sha256"] == 0
                and group_data["answers_artifact_sha256s"]
                == {selection["answers_artifact_sha256"]}
                and group_data["missing_judge_repeat_selection"] == 0
                and group_data["judge_repeat_selection_values"]
                == {selection["binding_json"]}
            ]
            if len(matching) == 1 and not extra:
                selection_id = str(matching[0]["selection_id"])
                selection_use_counts[selection_id] += 1
            else:
                problems.append(
                    "judgment group does not exactly cover its answer run and "
                    "has no unique explicit selection binding "
                    f"{judgment_group}: missing={missing or 'none'}, "
                    f"extra={extra or 'none'}"
                )

    if judgments:
        judged_answer_groups = {
            judgment_group[:3] for judgment_group in judgment_groups
        }
        for answer_group, expected_ids in answer_groups.items():
            if expected_ids and answer_group not in judged_answer_groups:
                problems.append(
                    "Judge-eligible answer group has no judgment group: "
                    f"{answer_group}; missing={sorted(expected_ids)}"
                )

    unused_selection_ids = sorted(
        str(selection["selection_id"])
        for selection in partial_selections
        if selection_use_counts[str(selection["selection_id"])] == 0
    )
    if unused_selection_ids:
        problems.append(
            "partial selection manifest(s) did not bind an incomplete judgment "
            "group: " + ", ".join(unused_selection_ids)
        )
    for selection in partial_selections:
        selection_id = str(selection["selection_id"])
        if (
            selection.get("final_protocol_selection") is True
            and selection_use_counts[selection_id]
            != FINAL_REPEAT_JUDGE_RUN_COUNT
        ):
            problems.append(
                f"final partial selection {selection_id} binds "
                f"{selection_use_counts[selection_id]} judgment group(s); expected "
                f"exactly {FINAL_REPEAT_JUDGE_RUN_COUNT} additional Judge runs"
            )

    return {
        "answer_records": len(answers),
        "unique_answer_ids": len(answer_by_id),
        "judgment_records": len(judgments),
        "unique_judgment_ids": len(seen_judgments),
        "judgment_groups": len(judgment_groups),
        "partial_selection_count": len(partial_selections),
        "used_partial_selection_count": sum(
            count > 0 for count in selection_use_counts.values()
        ),
        "partial_selection_judgment_group_counts": selection_use_counts,
        "complete": not problems,
        "problems": problems,
    }, problems


def load_partial_selection_manifest(path: Path) -> dict[str, Any]:
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
    if set(payload) != required:
        missing = sorted(required - payload.keys())
        unknown = sorted(payload.keys() - required)
        raise ValueError(
            f"{path}: invalid selection fields; missing={missing or 'none'}, "
            f"unknown={unknown or 'none'}"
        )
    if payload["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"{path}: invalid selection schema_version")
    selected_ids = payload["selected_answer_ids"]
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or any(not isinstance(value, str) or not value for value in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise ValueError(f"{path}: invalid selected_answer_ids")
    selected_hash = hashlib.sha256(
        json.dumps(
            selected_ids,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if payload["selected_answer_ids_sha256"] != selected_hash:
        raise ValueError(f"{path}: selected_answer_ids_sha256 mismatch")
    identity = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "answers_artifact_sha256": payload["answers_artifact_sha256"],
        "selected_answer_ids": selected_ids,
        "selected_answer_ids_sha256": selected_hash,
    }
    selection_hash = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if payload["selection_sha256"] != selection_hash:
        raise ValueError(f"{path}: selection_sha256 mismatch")
    expected_id = f"judge_repeat_selection_{selection_hash[:24]}"
    if payload["selection_id"] != expected_id:
        raise ValueError(f"{path}: selection_id mismatch")
    answer_sha = payload["answers_artifact_sha256"]
    if (
        not isinstance(answer_sha, str)
        or len(answer_sha) != 64
        or any(character not in "0123456789abcdef" for character in answer_sha)
    ):
        raise ValueError(f"{path}: invalid answers_artifact_sha256")
    return payload


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def parse_key_values(values: Iterable[str], flag: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} requires KEY=VALUE, got: {value}")
        key, item = value.split("=", 1)
        if not key:
            raise ValueError(f"{flag} key must not be empty: {value}")
        parsed[key] = item
    return parsed


def parse_expectations(values: Iterable[str], repo_root: Path) -> dict[Path, int]:
    parsed: dict[Path, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--expect-jsonl requires PATH=COUNT, got: {value}")
        raw_path, raw_count = value.rsplit("=", 1)
        count = int(raw_count)
        if count < 0:
            raise ValueError("expected JSONL count must be non-negative")
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        parsed[path.resolve()] = count
    return parsed


def build_manifest(
    *,
    repo_root: Path,
    experiment_id: str,
    artifacts: Iterable[Path],
    expectations: dict[Path, int],
    metadata: dict[str, str],
    require_clean: bool,
    require_judge_scores: set[Path] | None = None,
    selection_manifests: Sequence[Path] = (),
) -> tuple[dict[str, Any], list[str]]:
    head = git_value(repo_root, "rev-parse", "HEAD")
    status = git_value(repo_root, "status", "--porcelain")
    dirty = bool(status) if status is not None else None
    problems: list[str] = []
    require_judge_scores = require_judge_scores or set()
    if require_clean and dirty is not False:
        problems.append("Git worktree is not clean")

    artifact_entries: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    jsonl_records_by_artifact_sha: dict[str, list[dict[str, Any]]] = {}
    answer_artifact_shas: dict[str, set[str]] = {}
    resolved_artifacts: set[Path] = set()
    artifact_sha_by_path: dict[Path, str] = {}
    for raw_path in artifacts:
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        path = path.resolve()
        resolved_artifacts.add(path)
        if not path.is_file():
            problems.append(f"missing artifact: {path}")
            continue

        try:
            display_path = str(path.relative_to(repo_root))
        except ValueError:
            display_path = str(path)
        artifact_sha_before = sha256_file(path)
        artifact_bytes_before = path.stat().st_size
        entry: dict[str, Any] = {
            "path": display_path,
            "bytes": artifact_bytes_before,
            "sha256": artifact_sha_before,
        }
        artifact_sha_by_path[path] = str(entry["sha256"])
        expected_count = expectations.get(path)
        if path.suffix.lower() == ".jsonl":
            records = load_jsonl(path)
            eval_records.extend(records)
            jsonl_records_by_artifact_sha[str(entry["sha256"])] = records
            for record in records:
                if record.get("record_type") == "answer":
                    answer_id = str(record.get("answer_id") or "")
                    if answer_id:
                        answer_artifact_shas.setdefault(answer_id, set()).add(
                            str(entry["sha256"])
                        )
            stats = jsonl_stats(
                path,
                expected_count,
                require_judge_scores=path in require_judge_scores,
            )
            entry["jsonl"] = stats
            problems.extend(f"{display_path}: {item}" for item in stats["problems"])
        elif expected_count is not None:
            problems.append(f"expectation given for non-JSONL artifact: {display_path}")
        if (
            path.stat().st_size != artifact_bytes_before
            or sha256_file(path) != artifact_sha_before
        ):
            raise ValueError(f"artifact changed while building manifest: {path}")
        artifact_entries.append(entry)

    for expected_path in expectations:
        if expected_path not in resolved_artifacts:
            problems.append(f"expected JSONL is not listed as an artifact: {expected_path}")
    for scored_path in require_judge_scores:
        if scored_path not in resolved_artifacts:
            problems.append(f"judge-scored JSONL is not listed as an artifact: {scored_path}")

    normalized_selections: list[dict[str, Any]] = []
    seen_selection_ids: set[str] = set()
    seen_selection_paths: set[Path] = set()
    for raw_selection_path in selection_manifests:
        selection_path = (
            raw_selection_path
            if raw_selection_path.is_absolute()
            else repo_root / raw_selection_path
        ).resolve()
        if selection_path in seen_selection_paths:
            problems.append(f"duplicate partial selection path: {selection_path}")
            continue
        seen_selection_paths.add(selection_path)
        if selection_path not in resolved_artifacts:
            problems.append(
                "partial selection manifest is not listed as an artifact: "
                f"{selection_path}"
            )
            continue
        try:
            selection = load_partial_selection_manifest(selection_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"invalid partial selection {selection_path}: {exc}")
            continue
        selection_artifact_sha = sha256_file(selection_path)
        if selection_artifact_sha != artifact_sha_by_path.get(selection_path):
            problems.append(
                f"partial selection {selection_path} changed while building manifest"
            )
            continue
        selection_id = str(selection["selection_id"])
        if selection_id in seen_selection_ids:
            problems.append(f"duplicate partial selection_id: {selection_id}")
            continue
        seen_selection_ids.add(selection_id)
        answer_sha = str(selection["answers_artifact_sha256"])
        answer_records = jsonl_records_by_artifact_sha.get(answer_sha)
        if answer_records is None:
            problems.append(
                f"partial selection {selection_id} references an answer artifact "
                "SHA-256 that is not listed"
            )
            continue
        if not answer_records or any(
            record.get("record_type") != "answer" for record in answer_records
        ):
            problems.append(
                f"partial selection {selection_id} does not reference an answer JSONL"
            )
            continue
        answer_ids = [str(record.get("answer_id") or "") for record in answer_records]
        selected_ids = list(selection["selected_answer_ids"])
        unknown = sorted(set(selected_ids) - set(answer_ids))
        canonical_selected = [
            answer_id for answer_id in answer_ids if answer_id in set(selected_ids)
        ]
        if unknown:
            problems.append(
                f"partial selection {selection_id} contains unknown answer ids: {unknown}"
            )
            continue
        if selected_ids != canonical_selected:
            problems.append(
                f"partial selection {selection_id} does not preserve answer order"
            )
            continue
        selected_records = [
            record for record in answer_records if record.get("answer_id") in set(selected_ids)
        ]
        final_authorization_flags = {
            isinstance(record.get("collector_config"), dict)
            and isinstance(
                record["collector_config"].get("final_authorization"), dict
            )
            for record in answer_records
        }
        if len(final_authorization_flags) != 1:
            problems.append(
                f"partial selection {selection_id} references an answer artifact "
                "that mixes final authorization state"
            )
            continue
        final_protocol_selection = next(iter(final_authorization_flags))
        if final_protocol_selection:
            if len(selected_ids) != FINAL_SELECTION_COUNT_PER_CONDITION:
                problems.append(
                    f"partial selection {selection_id} has {len(selected_ids)} final "
                    f"answers; expected fixed {FINAL_SELECTION_COUNT_PER_CONDITION} "
                    "per condition"
                )
                continue
            final_authorizations = [
                record["collector_config"]["final_authorization"]
                for record in answer_records
            ]
            authorization = final_authorizations[0]
            condition_ids = {
                str(record.get("condition_id") or "") for record in answer_records
            }
            generation_run_ids = {
                str(record.get("generation_run_id") or "")
                for record in answer_records
            }
            case_ids = [str(record.get("case_id") or "") for record in answer_records]
            condition_id = (
                next(iter(condition_ids)) if len(condition_ids) == 1 else ""
            )
            if (
                any(value != authorization for value in final_authorizations[1:])
                or condition_ids not in ({"c0"}, {"c1"})
                or generation_run_ids != {"run1"}
                or authorization.get("schema_version")
                != FINAL_AUTHORIZATION_SCHEMA_VERSION
                or authorization.get("collection_purpose") != "final_generation"
                or authorization.get("split") != FINAL_CORE_SPLIT
                or authorization.get("artifact_id") != f"{condition_id}-core-run1"
                or authorization.get("expected_case_count")
                != FINAL_CORE_ANSWER_COUNT
                or len(answer_records) != FINAL_CORE_ANSWER_COUNT
                or authorization.get("expected_case_ids_sha256")
                != hashlib.sha256(
                    json.dumps(
                        case_ids,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            ):
                problems.append(
                    f"partial selection {selection_id} is not bound to a complete "
                    "C0/C1 generation-run1 Core artifact"
                )
                continue
        selected_judge_eligible_ids: list[str] = []
        selected_terminal_ids: list[str] = []
        invalid_selected_disposition = False
        for record in selected_records:
            try:
                if answer_is_judge_eligible(record):
                    selected_judge_eligible_ids.append(str(record["answer_id"]))
                else:
                    selected_terminal_ids.append(str(record["answer_id"]))
            except ValueError as exc:
                problems.append(
                    f"partial selection {selection_id} contains an answer with "
                    f"invalid slot disposition: {exc}"
                )
                invalid_selected_disposition = True
        if invalid_selected_disposition:
            continue
        if selected_terminal_ids:
            problems.append(
                f"partial selection {selection_id} makes the fixed-sample Judge "
                "stability diagnosis incomplete; terminal slots cannot be "
                f"replaced or selected: {selected_terminal_ids}"
            )
            continue
        groups = {
            (
                str(record.get("experiment_id") or ""),
                str(record.get("condition_id") or ""),
                str(record.get("generation_run_id") or ""),
            )
            for record in selected_records
        }
        if len(groups) != 1 or "" in next(iter(groups)):
            problems.append(
                f"partial selection {selection_id} mixes or omits answer group identity"
            )
            continue
        binding = {
            "schema_version": selection["schema_version"],
            "selection_id": selection_id,
            "selection_sha256": selection["selection_sha256"],
            "selection_artifact_sha256": selection_artifact_sha,
            "answers_artifact_sha256": answer_sha,
            "selected_answer_count": len(selected_ids),
            "selected_answer_ids_sha256": selection[
                "selected_answer_ids_sha256"
            ],
        }
        normalized_selections.append(
            {
                "selection_id": selection_id,
                "selection_sha256": selection["selection_sha256"],
                "selection_artifact_path": str(selection_path),
                "selection_artifact_sha256": selection_artifact_sha,
                "answers_artifact_sha256": answer_sha,
                "selected_answer_ids": selected_ids,
                "selected_judge_eligible_answer_ids": (
                    selected_judge_eligible_ids
                ),
                "selected_answer_ids_sha256": selection[
                    "selected_answer_ids_sha256"
                ],
                "answer_group": list(next(iter(groups))),
                "final_protocol_selection": final_protocol_selection,
                "binding": binding,
                "binding_json": json.dumps(
                    binding,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )

    eval_links, link_problems = validate_eval_artifact_links(
        eval_records,
        partial_selections=normalized_selections,
        answer_artifact_shas=answer_artifact_shas,
    )
    problems.extend(f"eval artifact links: {item}" for item in link_problems)

    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"root": str(repo_root), "commit": head, "dirty": dirty},
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "metadata": metadata,
        "artifacts": artifact_entries,
        "eval_artifact_links": eval_links,
        "partial_judgment_selections": [
            {
                key: value
                for key, value in selection.items()
                if key != "binding_json"
            }
            for selection in normalized_selections
        ],
        "validation": {"ok": not problems, "problems": problems},
    }
    return manifest, problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument(
        "--expect-jsonl",
        action="append",
        default=[],
        metavar="PATH=COUNT",
        help="validate record and unique-id completeness for one JSONL artifact",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="record an execution setting such as model, prompt, or index revision",
    )
    parser.add_argument(
        "--require-judge-scores",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="require every record in a JSONL artifact to have score 0, 1, or 2",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "explicitly authorize one partial judgment group with a "
            "pnu.judge-repeat-selection.v1 artifact (also list it with --artifact)"
        ),
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail validation unless the Git worktree is clean",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else repo_root / args.output
    artifact_inputs = [
        path if path.is_absolute() else repo_root / path for path in args.artifact
    ]
    selection_inputs = [
        path if path.is_absolute() else repo_root / path
        for path in args.selection_manifest
    ]
    try:
        reject_symlink_inputs([*artifact_inputs, *selection_inputs])
        if any(
            paths_alias(output, input_path)
            for input_path in [*artifact_inputs, *selection_inputs]
        ):
            raise ValueError("output path must not overwrite an input artifact")
        require_new_outputs([output])
        metadata = parse_key_values(args.metadata, "--metadata")
        expectations = parse_expectations(args.expect_jsonl, repo_root)
        scored_paths = {
            (path if path.is_absolute() else repo_root / path).resolve()
            for path in args.require_judge_scores
        }
        manifest, problems = build_manifest(
            repo_root=repo_root,
            experiment_id=args.experiment_id,
            artifacts=args.artifact,
            expectations=expectations,
            metadata=metadata,
            require_clean=args.require_clean,
            require_judge_scores=scored_paths,
            selection_manifests=args.selection_manifest,
        )
        manifest["output_publication"] = {
            "authoritative_completion_artifact": "json",
            "authoritative_path": str(output.resolve()),
            "required_companion_artifacts": [],
            "contract": (
                "This immutable JSON is the sole authoritative completion "
                "artifact and is atomically published after durable staging."
            ),
        }
        manifest_text = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        publish_immutable_texts(
            {output: manifest_text},
            authoritative_path=output,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {output}")
    if problems:
        for problem in problems:
            print(f"validation error: {problem}", file=sys.stderr)
        return 1
    print(f"validated {len(manifest['artifacts'])} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
