#!/usr/bin/env python3
"""Run a deterministic synthetic A/B evaluation of the claim semantic guard.

This microbenchmark isolates ``scripts.search_api.attribute_claim``.  It does
not estimate retrieval quality, generation quality, or end-to-end service
quality; those require separate held-out evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_api import attribute_claim  # noqa: E402


SCHEMA_VERSION = "pnu.grounding-guard-ablation.v1"
DEFAULT_CASES = REPO_ROOT / "config" / "pnu-grounding-adversarial-eval.jsonl"
DEFAULT_JSON = REPO_ROOT / "evidence" / "20260914" / "grounding-guard-ablation.json"
DEFAULT_CSV = REPO_ROOT / "evidence" / "20260914" / "grounding-guard-ablation.csv"
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
CONDITIONS = (
    ("legacy", False),
    ("tuned", True),
)


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")

        case_id = record.get("id")
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise ValueError(f"{path}:{line_number}: invalid case id {case_id!r}")
        if case_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate case id {case_id}")
        seen.add(case_id)

        claim = record.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"{path}:{line_number}: {case_id}: non-empty claim required")
        expected = record.get("expected_supported")
        if not isinstance(expected, bool):
            raise ValueError(
                f"{path}:{line_number}: {case_id}: expected_supported must be boolean"
            )
        expected_reason = record.get("expected_reason")
        if expected_reason is not None and (
            not isinstance(expected_reason, str) or not expected_reason
        ):
            raise ValueError(
                f"{path}:{line_number}: {case_id}: expected_reason must be a string"
            )
        sources = record.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{path}:{line_number}: {case_id}: sources required")
        source_ids: set[str] = set()
        for source_index, source in enumerate(sources, 1):
            if not isinstance(source, dict):
                raise ValueError(
                    f"{path}:{line_number}: {case_id}: source {source_index} "
                    "must be an object"
                )
            chunk_id = source.get("chunk_id")
            text = source.get("text")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"{path}:{line_number}: {case_id}: source {source_index} "
                    "requires chunk_id"
                )
            if chunk_id in source_ids:
                raise ValueError(
                    f"{path}:{line_number}: {case_id}: duplicate source chunk_id "
                    f"{chunk_id}"
                )
            source_ids.add(chunk_id)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{path}:{line_number}: {case_id}: source {source_index} "
                    "requires non-empty text"
                )
        expected_source_ids = record.get("expected_source_ids")
        if expected_source_ids is not None:
            if (
                not isinstance(expected_source_ids, list)
                or not all(isinstance(item, str) and item for item in expected_source_ids)
                or len(set(expected_source_ids)) != len(expected_source_ids)
            ):
                raise ValueError(
                    f"{path}:{line_number}: {case_id}: expected_source_ids must "
                    "contain unique strings"
                )
            unknown = set(expected_source_ids) - source_ids
            if unknown:
                raise ValueError(
                    f"{path}:{line_number}: {case_id}: unknown expected source ids "
                    f"{sorted(unknown)}"
                )
        cases.append(record)
    if not cases:
        raise ValueError(f"{path}: no cases")
    return cases


def _prediction(case: dict[str, Any], *, semantic_guard: bool) -> dict[str, Any]:
    result = attribute_claim(
        str(case["claim"]),
        list(case["sources"]),
        semantic_guard=semantic_guard,
    )
    supported = bool(result.get("supported"))
    reason = str(result.get("validation_reason") or "")
    source_ids = [str(value) for value in result.get("source_ids") or []]
    expected_supported = bool(case["expected_supported"])
    expected_reason = case.get("expected_reason")
    expected_source_ids = case.get("expected_source_ids")
    label_correct = supported == expected_supported
    reason_correct = expected_reason is None or reason == expected_reason
    source_ids_correct = (
        expected_source_ids is None or source_ids == expected_source_ids
    )
    return {
        "semantic_guard": semantic_guard,
        "supported": supported,
        "reason": reason,
        "source_ids": source_ids,
        "label_correct": label_correct,
        "reason_evaluated": expected_reason is not None,
        "reason_correct": reason_correct,
        "source_ids_evaluated": expected_source_ids is not None,
        "source_ids_correct": source_ids_correct,
        # The primary joint metric follows the benchmark target: supported label
        # plus validation reason.  Source selection is reported independently so
        # the one multi-source regression cannot inflate or depress that score.
        "joint_correct": label_correct and reason_correct,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _condition_metrics(
    rows: Sequence[dict[str, Any]], condition: str
) -> dict[str, Any]:
    total = len(rows)
    predictions = [row[condition] for row in rows]
    label_correct = sum(bool(value["label_correct"]) for value in predictions)
    reason_predictions = [
        value for value in predictions if value["reason_evaluated"]
    ]
    reason_correct = sum(bool(value["reason_correct"]) for value in reason_predictions)
    source_predictions = [
        value for value in predictions if value["source_ids_evaluated"]
    ]
    source_correct = sum(
        bool(value["source_ids_correct"]) for value in source_predictions
    )
    joint_correct = sum(bool(value["joint_correct"]) for value in predictions)

    true_positive = true_negative = false_positive = false_negative = 0
    for row in rows:
        expected = bool(row["expected_supported"])
        predicted = bool(row[condition]["supported"])
        if expected and predicted:
            true_positive += 1
        elif not expected and not predicted:
            true_negative += 1
        elif not expected and predicted:
            false_positive += 1
        else:
            false_negative += 1

    return {
        "case_count": total,
        "label_correct_count": label_correct,
        "label_accuracy": _rate(label_correct, total),
        "reason_evaluated_count": len(reason_predictions),
        "reason_correct_count": reason_correct,
        "reason_accuracy": _rate(reason_correct, len(reason_predictions)),
        "source_ids_evaluated_count": len(source_predictions),
        "source_ids_correct_count": source_correct,
        "source_ids_accuracy": _rate(source_correct, len(source_predictions)),
        "joint_correct_count": joint_correct,
        "joint_accuracy": _rate(joint_correct, total),
        "confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "label_error_case_ids": [
            row["case_id"] for row in rows if not row[condition]["label_correct"]
        ],
        "reason_error_case_ids": [
            row["case_id"]
            for row in rows
            if row[condition]["reason_evaluated"]
            and not row[condition]["reason_correct"]
        ],
        "source_ids_error_case_ids": [
            row["case_id"]
            for row in rows
            if row[condition]["source_ids_evaluated"]
            and not row[condition]["source_ids_correct"]
        ],
        "error_case_ids": [
            row["case_id"] for row in rows if not row[condition]["joint_correct"]
        ],
    }


def evaluate_cases(cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "case_id": str(case["id"]),
            "category": str(case.get("category") or "unspecified"),
            "claim": str(case["claim"]),
            "expected_supported": bool(case["expected_supported"]),
            "expected_reason": case.get("expected_reason"),
            "expected_source_ids": case.get("expected_source_ids"),
        }
        for condition, semantic_guard in CONDITIONS:
            row[condition] = _prediction(case, semantic_guard=semantic_guard)
        rows.append(row)

    expected_ids = [str(case["id"]) for case in cases]
    actual_ids = [str(row["case_id"]) for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ValueError(
            "incomplete evaluation: result IDs do not exactly match config IDs"
        )
    return rows


def build_report(cases_path: Path) -> dict[str, Any]:
    config_bytes = cases_path.read_bytes()
    cases = load_cases(cases_path)
    rows = evaluate_cases(cases)
    conditions = {
        condition: {
            "semantic_guard": semantic_guard,
            "metrics": _condition_metrics(rows, condition),
        }
        for condition, semantic_guard in CONDITIONS
    }
    legacy_metrics = conditions["legacy"]["metrics"]
    tuned_metrics = conditions["tuned"]["metrics"]
    fixed = [
        row["case_id"]
        for row in rows
        if not row["legacy"]["joint_correct"] and row["tuned"]["joint_correct"]
    ]
    regressions = [
        row["case_id"]
        for row in rows
        if row["legacy"]["joint_correct"] and not row["tuned"]["joint_correct"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "kind": "synthetic_adversarial_microbenchmark",
            "synthetic": True,
            "component_under_test": "scripts.search_api.attribute_claim",
            "estimand": (
                "supported-label, validation-reason, and selected-source "
                "behavior on curated relation-reversal cases"
            ),
            "limitations": [
                "This benchmark is curated from synthetic and notice-shaped examples.",
                "It does not estimate retrieval, LLM generation, or end-to-end service quality.",
                "Use a separate held-out service evaluation before claiming production gains.",
            ],
        },
        "config": {
            "path": _display_path(cases_path),
            "sha256": sha256_bytes(config_bytes),
            "case_count": len(cases),
            "case_ids": [str(case["id"]) for case in cases],
        },
        "conditions": conditions,
        "comparison": {
            "label_accuracy_delta": round(
                tuned_metrics["label_accuracy"] - legacy_metrics["label_accuracy"],
                6,
            ),
            "reason_accuracy_delta": round(
                tuned_metrics["reason_accuracy"] - legacy_metrics["reason_accuracy"],
                6,
            ),
            "joint_accuracy_delta": round(
                tuned_metrics["joint_accuracy"] - legacy_metrics["joint_accuracy"],
                6,
            ),
            "fixed_case_ids": fixed,
            "regression_case_ids": regressions,
        },
        "cases": rows,
    }


CSV_FIELDS = (
    "case_id",
    "category",
    "expected_supported",
    "expected_reason",
    "expected_source_ids",
    "legacy_supported",
    "legacy_reason",
    "legacy_source_ids",
    "legacy_label_correct",
    "legacy_reason_correct",
    "legacy_source_ids_correct",
    "legacy_joint_correct",
    "tuned_supported",
    "tuned_reason",
    "tuned_source_ids",
    "tuned_label_correct",
    "tuned_reason_correct",
    "tuned_source_ids_correct",
    "tuned_joint_correct",
)


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {
                "case_id": row["case_id"],
                "category": row["category"],
                "expected_supported": _csv_bool(row["expected_supported"]),
                "expected_reason": row["expected_reason"] or "",
                "expected_source_ids": canonical_json(row["expected_source_ids"] or []),
            }
            for condition, _semantic_guard in CONDITIONS:
                prediction = row[condition]
                output.update(
                    {
                        f"{condition}_supported": _csv_bool(prediction["supported"]),
                        f"{condition}_reason": prediction["reason"],
                        f"{condition}_source_ids": canonical_json(
                            prediction["source_ids"]
                        ),
                        f"{condition}_label_correct": _csv_bool(
                            prediction["label_correct"]
                        ),
                        f"{condition}_reason_correct": _csv_bool(
                            prediction["reason_correct"]
                        ),
                        f"{condition}_source_ids_correct": _csv_bool(
                            prediction["source_ids_correct"]
                        ),
                        f"{condition}_joint_correct": _csv_bool(
                            prediction["joint_correct"]
                        ),
                    }
                )
            writer.writerow(output)


def write_report(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(canonical_json(report, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, report["cases"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.cases)
    write_report(report, args.output_json, args.output_csv)
    legacy = report["conditions"]["legacy"]["metrics"]
    tuned = report["conditions"]["tuned"]["metrics"]
    print(
        "grounding guard ablation: "
        f"cases={report['config']['case_count']} "
        f"legacy_joint={legacy['joint_accuracy']:.3f} "
        f"tuned_joint={tuned['joint_accuracy']:.3f} "
        f"delta={report['comparison']['joint_accuracy_delta']:+.3f}"
    )
    print(f"json={args.output_json}")
    print(f"csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
