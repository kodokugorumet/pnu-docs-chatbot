#!/usr/bin/env python3
"""Validate and analyze LLM-Judge calibration against human labels.

Human-label JSONL schema (one JSON object per rating)
======================================================

One or more artifacts may be supplied.  This directly accepts the separate
reviewer-A, reviewer-B, and adjudication JSONL templates produced by
``build_answer_review_packet.py`` after every required placeholder is filled.

Required fields:

``answer_id``
    Non-empty string copied exactly from the judgment artifact.
``case_id``
    Non-empty string copied exactly from the judgment artifact.
``reviewer_id``
    Non-empty, stable reviewer identifier.  The same two independent reviewer
    IDs must cover every answer in one analysis.
``label_kind``
    Either ``"independent"`` or ``"adjudicated"``.
``score``
    Integer 0, 1, or 2.  Boolean values are not integers for this schema.
``grounded_fully_correct``
    JSON boolean.  ``true`` requires ``score == 2``.
``uncertain``
    JSON boolean preserving whether the human reviewer marked the decision as
    uncertain.

Optional fields:

``split``
    Non-empty string such as ``"holdout-core"`` or
    ``"holdout-challenge"``.  If used, it must be present for every answer.
``challenge_type``
    Non-empty string, consistent across the three labels for an answer.
``correct_abstention``
    Boolean challenge outcome or null.  When used for an answer, both
    independent labels and the adjudicated label must provide a boolean.
    It is required when ``challenge_type`` is ``"unanswerable"``.
``injection_obedience``
    Boolean challenge outcome or null (``true`` means the answer obeyed the
    attack).  When used for an answer, all three labels must provide a boolean.
    It is required when ``challenge_type`` is ``"prompt_injection"`` or
    ``"prompt-injection"``.
``notes``
    String (may be empty) or null.
``blind_item_id``
    Non-empty string identifying the condition-blinded review item.  When
    present it must agree across the three labels for an answer.
``schema_version``
    If present, must equal ``"pnu.human-answer-label.v1"``.
``record_type``
    If present, must equal ``"human_label"``.
``error``
    Must be null when present.  Error-bearing labels are rejected.

Unknown fields are rejected.  Every judged answer must have exactly two
``independent`` labels from two distinct reviewers and exactly one
``adjudicated`` label.  The judgment and human-label answer sets must match
exactly; duplicate, missing, extra, malformed, or error-bearing records abort
the analysis before output is written.

The protocol gate is computed on ``core``/``holdout-core`` records when split
metadata is supplied, otherwise on the complete input.  Its raw agreement is
the Judge-versus-adjudicated-human GFC accuracy, matching the final evaluation
protocol.  Balanced accuracy protects the gate from a deceptively high score
when human GFC labels are class-imbalanced.  Undefined balanced accuracy or
degenerate kappa never passes the gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from service_eval_artifacts import (  # noqa: E402
    expected_answer_id,
    load_unique_jsonl,
    validate_judgment_record,
)


OUTPUT_SCHEMA_VERSION = "pnu.judge-human-calibration.v1"
HUMAN_LABEL_SCHEMA_VERSION = "pnu.human-answer-label.v1"
LABEL_KINDS = frozenset({"independent", "adjudicated"})
CHALLENGE_BOOLEAN_FIELDS = (
    "correct_abstention",
    "injection_obedience",
)
HUMAN_REQUIRED_FIELDS = frozenset(
    {
        "answer_id",
        "case_id",
        "reviewer_id",
        "label_kind",
        "score",
        "grounded_fully_correct",
        "uncertain",
    }
)
HUMAN_OPTIONAL_FIELDS = frozenset(
    {
        "split",
        "challenge_type",
        *CHALLENGE_BOOLEAN_FIELDS,
        "notes",
        "blind_item_id",
        "schema_version",
        "record_type",
        "error",
    }
)
HUMAN_ALLOWED_FIELDS = HUMAN_REQUIRED_FIELDS | HUMAN_OPTIONAL_FIELDS
GATE_THRESHOLDS = {
    "raw_agreement": 0.80,
    "balanced_accuracy": 0.80,
    "cohen_kappa": 0.60,
    "macro_f1": 0.75,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(record: dict[str, Any], field: str, *, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: {field} must be a non-empty string")
    return value.strip()


def _validate_human_record(
    record: dict[str, Any], *, path: Path, line_number: int
) -> dict[str, Any]:
    label = f"{path}:{line_number}"
    missing = sorted(HUMAN_REQUIRED_FIELDS - record.keys())
    if missing:
        raise ValueError(f"{label}: missing required fields: {', '.join(missing)}")
    unknown = sorted(record.keys() - HUMAN_ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"{label}: unknown fields: {', '.join(unknown)}")

    value = dict(record)
    value["answer_id"] = _required_text(value, "answer_id", label=label)
    value["case_id"] = _required_text(value, "case_id", label=label)
    value["reviewer_id"] = _required_text(value, "reviewer_id", label=label)
    label_kind = _required_text(value, "label_kind", label=label)
    if label_kind not in LABEL_KINDS:
        raise ValueError(
            f"{label}: label_kind must be 'independent' or 'adjudicated'"
        )
    value["label_kind"] = label_kind

    score = value.get("score")
    if type(score) is not int or score not in (0, 1, 2):
        raise ValueError(f"{label}: score must be integer 0, 1, or 2")
    gfc = value.get("grounded_fully_correct")
    if type(gfc) is not bool:
        raise ValueError(f"{label}: grounded_fully_correct must be a boolean")
    if gfc and score != 2:
        raise ValueError(
            f"{label}: grounded_fully_correct=true requires score=2"
        )
    if type(value.get("uncertain")) is not bool:
        raise ValueError(f"{label}: uncertain must be a boolean")

    if "schema_version" in value:
        if value["schema_version"] != HUMAN_LABEL_SCHEMA_VERSION:
            raise ValueError(
                f"{label}: schema_version must be {HUMAN_LABEL_SCHEMA_VERSION!r}"
            )
    if "record_type" in value and value["record_type"] != "human_label":
        raise ValueError(f"{label}: record_type must be 'human_label'")
    if value.get("error") is not None:
        raise ValueError(f"{label}: human label has error: {value['error']}")

    for field in ("split", "challenge_type", "blind_item_id"):
        if field in value:
            value[field] = _required_text(value, field, label=label)
    if "notes" in value and value["notes"] is not None and not isinstance(
        value["notes"], str
    ):
        raise ValueError(f"{label}: notes must be a string or null")
    for field in CHALLENGE_BOOLEAN_FIELDS:
        if (
            field in value
            and value[field] is not None
            and type(value[field]) is not bool
        ):
            raise ValueError(f"{label}: {field} must be a boolean or null")
    return value


def load_human_labels(
    paths: Sequence[Path],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one human-label artifact is required")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("duplicate --human-labels path")

    metadata: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        artifact_sha_before = sha256_file(path)
        artifact_record_count = 0
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            record = _validate_human_record(
                raw, path=path, line_number=line_number
            )
            identity = (
                str(record["answer_id"]),
                str(record["label_kind"]),
                str(record["reviewer_id"]),
            )
            if identity in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate human label across "
                    f"artifacts for answer_id={identity[0]} "
                    f"label_kind={identity[1]} reviewer_id={identity[2]}"
                )
            seen.add(identity)
            records.append(record)
            artifact_record_count += 1
        if artifact_record_count == 0:
            raise ValueError(f"{path}: human-label artifact is empty")
        artifact_sha_after = sha256_file(path)
        if artifact_sha_after != artifact_sha_before:
            raise ValueError(
                f"{path}: human-label artifact changed during validation"
            )
        metadata.append({"path": str(path), "sha256": artifact_sha_before})
    return metadata, records


def _validate_judge_values(record: dict[str, Any], *, path: Path) -> None:
    case_id = str(record.get("case_id") or "<missing>")
    if record.get("error") is not None:
        raise ValueError(f"{path}: judgment {case_id} has error: {record['error']}")
    expected_id = expected_answer_id(record)
    if record.get("answer_id") != expected_id:
        raise ValueError(
            f"{path}: answer_id mismatch for case_id {case_id}: "
            f"expected {expected_id}, got {record.get('answer_id')}"
        )
    judge = record.get("judge")
    if not isinstance(judge, dict):
        raise ValueError(f"{path}: missing judge object for case_id {case_id}")
    score = judge.get("score")
    if type(score) is not int or score not in (0, 1, 2):
        raise ValueError(f"{path}: invalid judge score for case_id {case_id}")
    gfc = judge.get("grounded_fully_correct")
    if type(gfc) is not bool:
        raise ValueError(
            f"{path}: invalid or missing grounded_fully_correct for case_id {case_id}"
        )
    if gfc and score != 2:
        raise ValueError(
            f"{path}: grounded_fully_correct=true requires score=2 for "
            f"case_id {case_id}"
        )
    for field in CHALLENGE_BOOLEAN_FIELDS:
        if field in judge and type(judge[field]) is not bool:
            raise ValueError(f"{path}: judge.{field} must be a boolean for {case_id}")
    if "abstention" in judge and judge["abstention"] not in {
        "appropriate",
        "inappropriate",
        "not_applicable",
    }:
        raise ValueError(f"{path}: invalid judge.abstention for case_id {case_id}")


def load_judgments(
    paths: Sequence[Path],
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    if not paths:
        raise ValueError("at least one judgment artifact is required")
    metadata: list[dict[str, str]] = []
    by_answer: dict[str, dict[str, Any]] = {}
    seen_judgment_ids: set[str] = set()
    for path in paths:
        artifact_sha_before = sha256_file(path)
        records = load_unique_jsonl(path, key="judgment_id")
        if not records:
            raise ValueError(f"{path}: judgment artifact is empty")
        for record in records:
            validate_judgment_record(record)
            _validate_judge_values(record, path=path)
            judgment_id = str(record["judgment_id"])
            if judgment_id in seen_judgment_ids:
                raise ValueError(
                    f"duplicate judgment_id across artifacts: {judgment_id}"
                )
            seen_judgment_ids.add(judgment_id)
            answer_id = str(record["answer_id"])
            if answer_id in by_answer:
                raise ValueError(
                    f"multiple judgments for answer_id across artifacts: {answer_id}"
                )
            by_answer[answer_id] = record
        artifact_sha_after = sha256_file(path)
        if artifact_sha_after != artifact_sha_before:
            raise ValueError(f"{path}: judgment artifact changed during validation")
        metadata.append({"path": str(path), "sha256": artifact_sha_before})
    return metadata, by_answer


def _cohen_kappa(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must be non-empty and have equal length")
    size = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / size
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / size) * (right_counts[label] / size)
        for label in labels
    )
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    value = (observed - expected) / denominator
    return 0.0 if math.isclose(value, 0.0, abs_tol=1e-15) else value


def _quadratic_weighted_kappa(
    left: Sequence[int], right: Sequence[int]
) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("weighted-kappa inputs must be non-empty and equal length")
    categories = (0, 1, 2)
    size = len(left)
    observed_counts = Counter(zip(left, right))
    left_counts = Counter(left)
    right_counts = Counter(right)
    scale = float((len(categories) - 1) ** 2)
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for row in categories:
        for column in categories:
            weight = ((row - column) ** 2) / scale
            observed_disagreement += weight * observed_counts[(row, column)]
            expected_disagreement += (
                weight * left_counts[row] * right_counts[column] / size
            )
    if math.isclose(expected_disagreement, 0.0, abs_tol=1e-15):
        return None
    value = 1.0 - observed_disagreement / expected_disagreement
    return 0.0 if math.isclose(value, 0.0, abs_tol=1e-15) else value


def _binary_agreement(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("binary-agreement inputs must be non-empty and equal length")
    raw = sum(a == b for a, b in zip(left, right)) / len(left)
    kappa = _cohen_kappa(left, right)
    return {
        "n": len(left),
        "raw_agreement": raw,
        "cohen_kappa": kappa,
        "cohen_kappa_defined": kappa is not None,
    }


def _binary_comparison(
    *, actual: Sequence[bool], predicted: Sequence[bool]
) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("binary-comparison inputs must be non-empty and equal length")
    true_positive = sum(a and p for a, p in zip(actual, predicted))
    true_negative = sum((not a) and (not p) for a, p in zip(actual, predicted))
    false_positive = sum((not a) and p for a, p in zip(actual, predicted))
    false_negative = sum(a and (not p) for a, p in zip(actual, predicted))
    size = len(actual)
    accuracy = (true_positive + true_negative) / size
    actual_positive = true_positive + false_negative
    actual_negative = true_negative + false_positive
    true_positive_rate = (
        true_positive / actual_positive if actual_positive else None
    )
    true_negative_rate = (
        true_negative / actual_negative if actual_negative else None
    )
    balanced_accuracy = (
        (true_positive_rate + true_negative_rate) / 2.0
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    positive_f1_denominator = 2 * true_positive + false_positive + false_negative
    negative_f1_denominator = 2 * true_negative + false_positive + false_negative
    positive_f1 = (
        2 * true_positive / positive_f1_denominator
        if positive_f1_denominator
        else 0.0
    )
    negative_f1 = (
        2 * true_negative / negative_f1_denominator
        if negative_f1_denominator
        else 0.0
    )
    predicted_positive = true_positive + false_positive
    false_pass_rate = (
        false_positive / predicted_positive if predicted_positive else None
    )
    kappa = _cohen_kappa(actual, predicted)
    return {
        "n": size,
        "confusion_matrix": {
            "orientation": "rows=adjudicated_human_actual, columns=judge_predicted",
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "raw_agreement": accuracy,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_defined": balanced_accuracy is not None,
        "class_recall": {
            "false_specificity": true_negative_rate,
            "true_sensitivity": true_positive_rate,
        },
        "cohen_kappa": kappa,
        "cohen_kappa_defined": kappa is not None,
        "class_f1": {"false": negative_f1, "true": positive_f1},
        "macro_f1": (negative_f1 + positive_f1) / 2.0,
        "false_pass_rate": false_pass_rate,
        "false_pass_rate_definition": (
            "false_positive / all Judge GFC passes; null when Judge has no passes"
        ),
    }


def _same_optional_metadata(
    labels: Sequence[dict[str, Any]], *, answer_id: str, field: str
) -> Any:
    values = {label.get(field) for label in labels}
    if len(values) != 1:
        raise ValueError(
            f"human labels disagree on {field} for answer_id {answer_id}"
        )
    return next(iter(values))


def _is_core_split(split: str | None) -> bool:
    if split is None:
        return False
    normalized = split.strip().lower().replace("_", "-")
    return normalized == "core" or normalized.endswith("-core")


def _normalize_challenge_type(challenge_type: str | None) -> str | None:
    if challenge_type is None:
        return None
    return challenge_type.strip().lower().replace("-", "_")


def _derive_judge_challenge_value(
    judgment: dict[str, Any], field: str
) -> bool | None:
    judge = judgment["judge"]
    explicit = judge.get(field)
    if type(explicit) is bool:
        return explicit
    if field == "correct_abstention":
        abstention = judge.get("abstention")
        if abstention == "appropriate":
            return True
        if abstention == "inappropriate":
            return False
    return None


def _build_case_rows(
    *,
    judgments_by_answer: dict[str, dict[str, Any]],
    human_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in human_records:
        grouped[str(record["answer_id"])].append(record)

    judged_ids = set(judgments_by_answer)
    human_ids = set(grouped)
    missing = sorted(judged_ids - human_ids)
    extra = sorted(human_ids - judged_ids)
    if missing:
        raise ValueError(
            "missing human labels for judged answer_id(s): " + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "human labels reference unknown answer_id(s): " + ", ".join(extra)
        )

    independent_reviewer_ids = {
        str(record["reviewer_id"])
        for record in human_records
        if record["label_kind"] == "independent"
    }
    if len(independent_reviewer_ids) != 2:
        raise ValueError(
            "exactly two independent reviewer_ids must cover the analysis; "
            f"found {sorted(independent_reviewer_ids)}"
        )
    reviewer_order = tuple(sorted(independent_reviewer_ids))

    rows: list[dict[str, Any]] = []
    any_split = any(record.get("split") is not None for record in human_records)
    blind_items_to_answers: dict[str, str] = {}
    for answer_id in sorted(judgments_by_answer):
        judgment = judgments_by_answer[answer_id]
        labels = grouped[answer_id]
        independent = [
            record for record in labels if record["label_kind"] == "independent"
        ]
        adjudicated = [
            record for record in labels if record["label_kind"] == "adjudicated"
        ]
        if len(independent) != 2:
            raise ValueError(
                f"answer_id {answer_id}: expected exactly 2 independent labels; "
                f"found {len(independent)}"
            )
        if len(adjudicated) != 1:
            raise ValueError(
                f"answer_id {answer_id}: expected exactly 1 adjudicated label; "
                f"found {len(adjudicated)}"
            )
        reviewers = {str(record["reviewer_id"]) for record in independent}
        if reviewers != set(reviewer_order):
            raise ValueError(
                f"answer_id {answer_id}: independent labels must come from the "
                f"same two reviewers {list(reviewer_order)}; found {sorted(reviewers)}"
            )
        human_case_ids = {str(record["case_id"]) for record in labels}
        judge_case_id = str(judgment["case_id"])
        if human_case_ids != {judge_case_id}:
            raise ValueError(
                f"case_id mismatch for answer_id {answer_id}: "
                f"judge={judge_case_id}, human={sorted(human_case_ids)}"
            )

        split = _same_optional_metadata(labels, answer_id=answer_id, field="split")
        challenge_type = _same_optional_metadata(
            labels, answer_id=answer_id, field="challenge_type"
        )
        blind_item_id = _same_optional_metadata(
            labels, answer_id=answer_id, field="blind_item_id"
        )
        if blind_item_id is not None:
            prior_answer_id = blind_items_to_answers.get(blind_item_id)
            if prior_answer_id is not None and prior_answer_id != answer_id:
                raise ValueError(
                    f"blind_item_id {blind_item_id!r} is reused by answer_id "
                    f"{prior_answer_id} and {answer_id}"
                )
            blind_items_to_answers[blind_item_id] = answer_id
        if any_split and split is None:
            raise ValueError(
                "split is present in the dataset but missing for answer_id "
                f"{answer_id}"
            )
        challenge_values: dict[str, dict[str, bool] | None] = {}
        for field in CHALLENGE_BOOLEAN_FIELDS:
            concrete_count = sum(type(record.get(field)) is bool for record in labels)
            if concrete_count not in (0, 3):
                raise ValueError(
                    f"answer_id {answer_id}: {field} must be present on all three "
                    "human labels as booleans or be null/absent on all three"
                )
            if concrete_count == 0:
                challenge_values[field] = None
            else:
                independent_by_reviewer = {
                    str(record["reviewer_id"]): record for record in independent
                }
                challenge_values[field] = {
                    "reviewer_a": bool(
                        independent_by_reviewer[reviewer_order[0]][field]
                    ),
                    "reviewer_b": bool(
                        independent_by_reviewer[reviewer_order[1]][field]
                    ),
                    "adjudicated": bool(adjudicated[0][field]),
                }

        required_challenge_field = {
            "unanswerable": "correct_abstention",
            "prompt_injection": "injection_obedience",
        }.get(_normalize_challenge_type(challenge_type))
        if (
            required_challenge_field is not None
            and challenge_values[required_challenge_field] is None
        ):
            raise ValueError(
                f"answer_id {answer_id}: challenge_type {challenge_type!r} "
                f"requires {required_challenge_field} on all three human "
                "labels as booleans"
            )

        independent_by_reviewer = {
            str(record["reviewer_id"]): record for record in independent
        }
        reviewer_a = independent_by_reviewer[reviewer_order[0]]
        reviewer_b = independent_by_reviewer[reviewer_order[1]]
        adjudicated_label = adjudicated[0]
        rows.append(
            {
                "answer_id": answer_id,
                "case_id": judge_case_id,
                "blind_item_id": blind_item_id,
                "split": split,
                "challenge_type": challenge_type,
                "judge": {
                    "judgment_id": judgment["judgment_id"],
                    "score": int(judgment["judge"]["score"]),
                    "grounded_fully_correct": bool(
                        judgment["judge"]["grounded_fully_correct"]
                    ),
                },
                "independent": [
                    {
                        "reviewer_id": reviewer_a["reviewer_id"],
                        "score": reviewer_a["score"],
                        "grounded_fully_correct": reviewer_a[
                            "grounded_fully_correct"
                        ],
                        "uncertain": reviewer_a["uncertain"],
                        "notes": reviewer_a.get("notes"),
                    },
                    {
                        "reviewer_id": reviewer_b["reviewer_id"],
                        "score": reviewer_b["score"],
                        "grounded_fully_correct": reviewer_b[
                            "grounded_fully_correct"
                        ],
                        "uncertain": reviewer_b["uncertain"],
                        "notes": reviewer_b.get("notes"),
                    },
                ],
                "adjudicated": {
                    "reviewer_id": adjudicated_label["reviewer_id"],
                    "score": adjudicated_label["score"],
                    "grounded_fully_correct": adjudicated_label[
                        "grounded_fully_correct"
                    ],
                    "uncertain": adjudicated_label["uncertain"],
                    "notes": adjudicated_label.get("notes"),
                },
                "independent_gfc_agreement": reviewer_a[
                    "grounded_fully_correct"
                ]
                == reviewer_b["grounded_fully_correct"],
                "independent_score_agreement": reviewer_a["score"]
                == reviewer_b["score"],
                "challenge_labels": challenge_values,
            }
        )
    return rows, reviewer_order


def _human_human_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    left_gfc = [bool(row["independent"][0]["grounded_fully_correct"]) for row in rows]
    right_gfc = [bool(row["independent"][1]["grounded_fully_correct"]) for row in rows]
    left_scores = [int(row["independent"][0]["score"]) for row in rows]
    right_scores = [int(row["independent"][1]["score"]) for row in rows]
    qwk = _quadratic_weighted_kappa(left_scores, right_scores)
    return {
        "gfc": _binary_agreement(left_gfc, right_gfc),
        "score_0_1_2": {
            "n": len(rows),
            "quadratic_weighted_kappa": qwk,
            "quadratic_weighted_kappa_defined": qwk is not None,
        },
    }


def _judge_human_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _binary_comparison(
        actual=[bool(row["adjudicated"]["grounded_fully_correct"]) for row in rows],
        predicted=[bool(row["judge"]["grounded_fully_correct"]) for row in rows],
    )


def _challenge_metrics(
    rows: Sequence[dict[str, Any]],
    judgments_by_answer: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in CHALLENGE_BOOLEAN_FIELDS:
        applicable = [
            row for row in rows if row["challenge_labels"].get(field) is not None
        ]
        if not applicable:
            continue
        left = [
            bool(row["challenge_labels"][field]["reviewer_a"])
            for row in applicable
        ]
        right = [
            bool(row["challenge_labels"][field]["reviewer_b"])
            for row in applicable
        ]
        adjudicated = [
            bool(row["challenge_labels"][field]["adjudicated"])
            for row in applicable
        ]
        metric: dict[str, Any] = {
            "answer_count": len(applicable),
            "human_human": _binary_agreement(left, right),
            "adjudicated_true_count": sum(adjudicated),
            "adjudicated_true_rate": sum(adjudicated) / len(adjudicated),
        }
        if field == "injection_obedience":
            metric["desired_value"] = False
            metric["adjudicated_injection_resistance_rate"] = (
                sum(not value for value in adjudicated) / len(adjudicated)
            )
        else:
            metric["desired_value"] = True
            metric["adjudicated_success_rate"] = (
                sum(adjudicated) / len(adjudicated)
            )

        judge_pairs = [
            (
                adjudicated[index],
                _derive_judge_challenge_value(
                    judgments_by_answer[row["answer_id"]], field
                ),
            )
            for index, row in enumerate(applicable)
        ]
        comparable = [
            (actual, predicted)
            for actual, predicted in judge_pairs
            if predicted is not None
        ]
        metric["judge_comparable_answer_count"] = len(comparable)
        metric["judge_vs_adjudicated"] = (
            _binary_comparison(
                actual=[actual for actual, _ in comparable],
                predicted=[bool(predicted) for _, predicted in comparable],
            )
            if comparable
            else None
        )
        result[field] = metric
    return result


def analyze_calibration(
    *, judgment_paths: Sequence[Path], human_label_paths: Sequence[Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    judgment_metadata, judgments_by_answer = load_judgments(judgment_paths)
    human_metadata, human_records = load_human_labels(human_label_paths)
    rows, reviewer_ids = _build_case_rows(
        judgments_by_answer=judgments_by_answer,
        human_records=human_records,
    )

    if any(row["split"] is not None for row in rows):
        gate_rows = [row for row in rows if _is_core_split(row["split"])]
        gate_scope = "core_split_only"
    else:
        gate_rows = list(rows)
        gate_scope = "all_answers_no_split_metadata"

    gate_observed = _judge_human_metrics(gate_rows) if gate_rows else None
    gate_checks = {
        key: bool(
            gate_observed is not None
            and gate_observed[
                "raw_agreement" if key == "raw_agreement" else key
            ]
            is not None
            and gate_observed[
                "raw_agreement" if key == "raw_agreement" else key
            ]
            >= threshold
        )
        for key, threshold in GATE_THRESHOLDS.items()
    }
    gate = {
        "scope": gate_scope,
        "answer_count": len(gate_rows),
        "eligible": bool(gate_rows),
        "thresholds": dict(GATE_THRESHOLDS),
        "observed": (
            {
                "raw_agreement": gate_observed["raw_agreement"],
                "balanced_accuracy": gate_observed["balanced_accuracy"],
                "cohen_kappa": gate_observed["cohen_kappa"],
                "macro_f1": gate_observed["macro_f1"],
            }
            if gate_observed is not None
            else {
                "raw_agreement": None,
                "balanced_accuracy": None,
                "cohen_kappa": None,
                "macro_f1": None,
            }
        ),
        "checks": gate_checks,
        "passed": bool(gate_rows) and all(gate_checks.values()),
        "undefined_kappa_passes": False,
    }

    payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "inputs": {
            "judgments": judgment_metadata,
            "human_labels": human_metadata,
        },
        "human_label_schema": {
            "schema_version": HUMAN_LABEL_SCHEMA_VERSION,
            "required_fields": sorted(HUMAN_REQUIRED_FIELDS),
            "optional_fields": sorted(HUMAN_OPTIONAL_FIELDS),
            "label_kinds": sorted(LABEL_KINDS),
            "labels_per_answer": {
                "independent": 2,
                "adjudicated": 1,
            },
        },
        "methodology": {
            "sample_unit": "answer",
            "independent_reviewer_ids": list(reviewer_ids),
            "human_gfc_agreement": "raw agreement and unweighted Cohen's kappa",
            "human_score_agreement": (
                "Cohen's quadratic-weighted kappa over fixed categories 0,1,2"
            ),
            "macro_f1": (
                "unweighted mean of false-class and true-class F1; an absent "
                "class has F1=0"
            ),
            "balanced_accuracy": (
                "unweighted mean of GFC sensitivity and specificity; undefined "
                "unless both adjudicated-human classes occur"
            ),
            "gate": (
                "Judge versus adjudicated human GFC on core split, or all "
                "answers only when split metadata is entirely absent"
            ),
        },
        "summary": {
            "answer_count": len(rows),
            "judgment_record_count": len(judgments_by_answer),
            "independent_human_rating_count": len(rows) * 2,
            "adjudicated_human_rating_count": len(rows),
            "human_human": _human_human_metrics(rows),
            "judge_vs_adjudicated_human_gfc": _judge_human_metrics(rows),
            "protocol_gate": gate,
            "challenge": _challenge_metrics(rows, judgments_by_answer),
        },
        "cases": rows,
    }
    return payload, rows


def _csv_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        independent_a, independent_b = row["independent"]
        challenge = row["challenge_labels"]
        yield {
            "answer_id": row["answer_id"],
            "case_id": row["case_id"],
            "blind_item_id": row["blind_item_id"] or "",
            "split": row["split"] or "",
            "challenge_type": row["challenge_type"] or "",
            "judge_score": row["judge"]["score"],
            "judge_grounded_fully_correct": row["judge"][
                "grounded_fully_correct"
            ],
            "reviewer_a_id": independent_a["reviewer_id"],
            "reviewer_a_score": independent_a["score"],
            "reviewer_a_grounded_fully_correct": independent_a[
                "grounded_fully_correct"
            ],
            "reviewer_b_id": independent_b["reviewer_id"],
            "reviewer_b_score": independent_b["score"],
            "reviewer_b_grounded_fully_correct": independent_b[
                "grounded_fully_correct"
            ],
            "adjudicator_id": row["adjudicated"]["reviewer_id"],
            "adjudicated_score": row["adjudicated"]["score"],
            "adjudicated_grounded_fully_correct": row["adjudicated"][
                "grounded_fully_correct"
            ],
            "independent_gfc_agreement": row["independent_gfc_agreement"],
            "independent_score_agreement": row["independent_score_agreement"],
            "correct_abstention": (
                json.dumps(challenge["correct_abstention"], separators=(",", ":"))
                if challenge["correct_abstention"] is not None
                else ""
            ),
            "injection_obedience": (
                json.dumps(challenge["injection_obedience"], separators=(",", ":"))
                if challenge["injection_obedience"] is not None
                else ""
            ),
        }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    materialized = list(_csv_rows(rows))
    if not materialized:
        raise ValueError("cannot write an empty case CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _reject_input_output_aliases(
    *,
    judgment_paths: Sequence[Path],
    human_label_paths: Sequence[Path],
    json_out: Path,
    csv_out: Path | None,
) -> None:
    input_paths = {
        *(path.resolve() for path in human_label_paths),
        *(path.resolve() for path in judgment_paths),
    }
    output_paths = [json_out.resolve()]
    if csv_out is not None:
        output_paths.append(csv_out.resolve())
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("--json-out and --csv-out must be different files")
    if any(path in input_paths for path in output_paths):
        raise ValueError("output paths must not overwrite judgment/human inputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judgments",
        type=Path,
        nargs="+",
        required=True,
        help="one or more non-overlapping judge_service_answers.py JSONL files",
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "one combined file or separate reviewer-A, reviewer-B, and "
            "adjudication JSONL files"
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        _reject_input_output_aliases(
            judgment_paths=args.judgments,
            human_label_paths=args.human_labels,
            json_out=args.json_out,
            csv_out=args.csv_out,
        )
        payload, rows = analyze_calibration(
            judgment_paths=args.judgments,
            human_label_paths=args.human_labels,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.csv_out is not None:
        write_csv(args.csv_out, rows)

    summary = payload["summary"]
    judge_human = summary["judge_vs_adjudicated_human_gfc"]
    gate = summary["protocol_gate"]
    print(
        f"answers={summary['answer_count']} "
        f"judge_human_accuracy={judge_human['accuracy']:.4f} "
        f"balanced_accuracy={judge_human['balanced_accuracy']} "
        f"judge_human_kappa={judge_human['cohen_kappa']} "
        f"macro_f1={judge_human['macro_f1']:.4f} "
        f"gate_passed={str(gate['passed']).lower()}"
    )
    print(f"wrote {args.json_out}")
    if args.csv_out is not None:
        print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
