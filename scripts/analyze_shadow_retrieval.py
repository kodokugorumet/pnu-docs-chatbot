#!/usr/bin/env python3
"""Compare C0/C1 retrieval on the post-freeze Shadow60 diagnostic set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "config" / "pnu-service-shadow60-v1.jsonl"
DEFAULT_C0 = (
    ROOT
    / "processed"
    / "eval"
    / "preflight-20260904"
    / "shadow60-retrieval-v1"
    / "c0-extractive.answers.jsonl"
)
DEFAULT_C1 = (
    ROOT
    / "processed"
    / "eval"
    / "preflight-20260904"
    / "shadow60-retrieval-v1"
    / "c1-extractive.answers.jsonl"
)
DEFAULT_JSON_OUT = DEFAULT_C0.parent / "c0-vs-c1-retrieval.json"
DEFAULT_CSV_OUT = DEFAULT_C0.parent / "c0-vs-c1-retrieval.csv"
ANSWERABLE_BUCKETS = {"simple", "multi", "role_variant"}
K_VALUES = (5, 8)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.mean(items) if items else 0.0


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("McNemar inputs must be paired and non-empty")
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum((not a) and b for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "c0_only": left_only,
        "c1_only": right_only,
        "discordant": discordant,
        "p_value_two_sided_exact": p_value,
    }


def _cluster_bootstrap(
    deltas: dict[str, float],
    family_by_case: dict[str, str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for case_id in deltas:
        clusters[family_by_case[case_id]].append(case_id)
    families = sorted(clusters)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled: list[str] = []
        for _slot in families:
            sampled.extend(clusters[rng.choice(families)])
        samples.append(_mean(deltas[case_id] for case_id in sampled))
    return {
        "method": "paired question-family cluster bootstrap",
        "clusters": len(families),
        "iterations": iterations,
        "seed": seed,
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
    }


def _gold_document_ids(case: dict[str, Any]) -> set[str]:
    return {
        str(option["document_id"])
        for claim in case.get("required_claims", [])
        for option in claim.get("evidence_options", [])
        if option.get("document_id")
    }


def _metrics(record: dict[str, Any], case: dict[str, Any]) -> dict[str, float | bool]:
    contexts = record.get("evaluation_trace", {}).get("retrieval_stages", {}).get(
        "final_contexts"
    )
    if not isinstance(contexts, list):
        raise ValueError(f"{case['id']}: missing final_contexts trace")
    gold_documents = _gold_document_ids(case)
    metrics: dict[str, float | bool] = {}
    for k in K_VALUES:
        document_ids = {
            str(row.get("document_id") or row.get("doc_id") or "")
            for row in contexts[:k]
            if isinstance(row, dict)
        }
        metrics[f"source_hit_at_{k}"] = bool(gold_documents & document_ids)
        atomic = record.get("atomic_evidence_at_k", {}).get(str(k))
        if not isinstance(atomic, dict) or atomic.get("available") is not True:
            raise ValueError(f"{case['id']}: missing atomic evidence@{k}")
        metrics[f"all_evidence_at_{k}"] = bool(atomic.get("all_matched"))
        metrics[f"evidence_recall_at_{k}"] = float(atomic.get("recall", 0.0))
    metrics["latency_ms"] = float(record.get("latency_ms", 0.0))
    return metrics


def _load_condition(
    path: Path,
    *,
    expected_condition: str,
    cases_sha256: str,
    expected_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("id") or "")
        if not case_id or case_id in by_id:
            raise ValueError(f"{path}: missing or duplicate case id {case_id!r}")
        if row.get("condition_id") != expected_condition:
            raise ValueError(f"{case_id}: condition mismatch")
        if row.get("experiment_id") != "shadow60-retrieval-v1":
            raise ValueError(f"{case_id}: experiment mismatch")
        config = row.get("collector_config")
        if not isinstance(config, dict) or config.get("cases_sha256") != cases_sha256:
            raise ValueError(f"{case_id}: cases SHA mismatch")
        if row.get("generator") != "extractive":
            raise ValueError(f"{case_id}: retrieval run must use extractive provider")
        by_id[case_id] = row
    if set(by_id) != expected_ids:
        raise ValueError(
            f"{path}: case ids mismatch; missing={sorted(expected_ids - set(by_id))}, "
            f"extra={sorted(set(by_id) - expected_ids)}"
        )
    first_config = rows[0]["collector_config"]
    expected_tuning = expected_condition == "c1"
    if first_config.get("expected_retrieval_tuning") is not expected_tuning:
        raise ValueError(f"{path}: retrieval tuning control mismatch")
    return by_id, {
        "path": str(path),
        "sha256": _sha256(path),
        "records": len(rows),
        "collector_config_sha256": rows[0].get("collector_config_sha256"),
    }


def _condition_summary(
    case_ids: list[str], metrics: dict[str, dict[str, float | bool]]
) -> dict[str, Any]:
    result: dict[str, Any] = {"questions": len(case_ids)}
    for metric in (
        "source_hit_at_5",
        "source_hit_at_8",
        "all_evidence_at_5",
        "all_evidence_at_8",
    ):
        values = [bool(metrics[case_id][metric]) for case_id in case_ids]
        result[metric] = {"hits": sum(values), "rate": _mean(float(v) for v in values)}
    for metric in ("evidence_recall_at_5", "evidence_recall_at_8", "latency_ms"):
        result[metric] = _mean(float(metrics[case_id][metric]) for case_id in case_ids)
    return result


def analyze(
    cases: list[dict[str, Any]],
    c0_records: dict[str, dict[str, Any]],
    c1_records: dict[str, dict[str, Any]],
    *,
    inputs: dict[str, Any],
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases_by_id = {str(case["id"]): case for case in cases}
    selected = [
        str(case["id"])
        for case in cases
        if case.get("answerable") is True and case.get("required_claims")
    ]
    groups = {
        "core": [
            case_id
            for case_id in selected
            if cases_by_id[case_id].get("shadow_bucket") in {"simple", "multi"}
        ],
        "role_variant": [
            case_id
            for case_id in selected
            if cases_by_id[case_id].get("shadow_bucket") == "role_variant"
        ],
        "prompt_injection": [
            case_id
            for case_id in selected
            if cases_by_id[case_id].get("challenge_type") == "prompt_injection"
        ],
        "all_answerable": selected,
    }
    metrics_c0 = {
        case_id: _metrics(c0_records[case_id], cases_by_id[case_id])
        for case_id in selected
    }
    metrics_c1 = {
        case_id: _metrics(c1_records[case_id], cases_by_id[case_id])
        for case_id in selected
    }
    family_by_case = {
        case_id: str(cases_by_id[case_id]["family_id"]) for case_id in selected
    }
    comparisons: dict[str, Any] = {}
    for group_name, case_ids in groups.items():
        group_comparison: dict[str, Any] = {}
        for metric in (
            "source_hit_at_5",
            "source_hit_at_8",
            "all_evidence_at_5",
            "all_evidence_at_8",
            "evidence_recall_at_5",
            "evidence_recall_at_8",
        ):
            c0_values = [float(metrics_c0[case_id][metric]) for case_id in case_ids]
            c1_values = [float(metrics_c1[case_id][metric]) for case_id in case_ids]
            deltas = {
                case_id: float(metrics_c1[case_id][metric])
                - float(metrics_c0[case_id][metric])
                for case_id in case_ids
            }
            item: dict[str, Any] = {
                "c0_mean": _mean(c0_values),
                "c1_mean": _mean(c1_values),
                "delta_c1_minus_c0": _mean(deltas.values()),
                "gained_case_ids": [case_id for case_id in case_ids if deltas[case_id] > 0],
                "lost_case_ids": [case_id for case_id in case_ids if deltas[case_id] < 0],
                "paired_family_bootstrap_95ci": _cluster_bootstrap(
                    deltas,
                    family_by_case,
                    iterations=iterations,
                    seed=seed,
                ),
            }
            if metric.startswith(("source_hit", "all_evidence")):
                item["exact_mcnemar"] = _exact_mcnemar(
                    [bool(value) for value in c0_values],
                    [bool(value) for value in c1_values],
                )
            group_comparison[metric] = item
        comparisons[group_name] = {
            "questions": len(case_ids),
            "c0": _condition_summary(case_ids, metrics_c0),
            "c1": _condition_summary(case_ids, metrics_c1),
            "paired": group_comparison,
        }

    csv_rows: list[dict[str, Any]] = []
    for case_id in selected:
        case = cases_by_id[case_id]
        row: dict[str, Any] = {
            "case_id": case_id,
            "family_id": case["family_id"],
            "bucket": case["shadow_bucket"],
            "category": case["category"],
            "challenge_type": case.get("challenge_type", ""),
        }
        for condition, metrics in (("c0", metrics_c0), ("c1", metrics_c1)):
            for metric, value in metrics[case_id].items():
                row[f"{condition}_{metric}"] = value
        row["delta_all_evidence_at_8"] = (
            int(bool(metrics_c1[case_id]["all_evidence_at_8"]))
            - int(bool(metrics_c0[case_id]["all_evidence_at_8"]))
        )
        csv_rows.append(row)

    report = {
        "schema_version": "pnu.shadow-retrieval-analysis.v1",
        "interpretation": (
            "Post-freeze synthetic diagnostic only; it is not the human-reviewed "
            "final holdout and must not be used for further production tuning."
        ),
        "inputs": inputs,
        "selected_answerable_cases": len(selected),
        "excluded_unanswerable_or_ambiguity_cases": len(cases) - len(selected),
        "groups": comparisons,
        "case_change_counts_at_8": dict(
            Counter(
                "gained"
                if row["delta_all_evidence_at_8"] > 0
                else "lost"
                if row["delta_all_evidence_at_8"] < 0
                else "tied"
                for row in csv_rows
            )
        ),
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["analysis_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report, csv_rows


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--c0", type=Path, default=DEFAULT_C0)
    parser.add_argument("--c1", type=Path, default=DEFAULT_C1)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    if args.json_out.resolve() == args.csv_out.resolve():
        parser.error("output paths must differ")

    cases = _read_jsonl(args.cases)
    case_ids = {str(case.get("id") or "") for case in cases}
    cases_sha = _sha256(args.cases)
    c0, c0_input = _load_condition(
        args.c0,
        expected_condition="c0",
        cases_sha256=cases_sha,
        expected_ids=case_ids,
    )
    c1, c1_input = _load_condition(
        args.c1,
        expected_condition="c1",
        cases_sha256=cases_sha,
        expected_ids=case_ids,
    )
    report, rows = analyze(
        cases,
        c0,
        c1,
        inputs={
            "cases": {"path": str(args.cases), "sha256": cases_sha},
            "c0": c0_input,
            "c1": c1_input,
        },
        iterations=args.bootstrap,
        seed=args.seed,
    )
    _write_new(
        args.json_out,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "csv_out": str(args.csv_out),
                "analysis_sha256": report["analysis_sha256"],
                "core_all_evidence_at_8": {
                    "c0": report["groups"]["core"]["c0"]["all_evidence_at_8"],
                    "c1": report["groups"]["core"]["c1"]["all_evidence_at_8"],
                    "delta": report["groups"]["core"]["paired"]
                    ["all_evidence_at_8"]["delta_c1_minus_c0"],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
