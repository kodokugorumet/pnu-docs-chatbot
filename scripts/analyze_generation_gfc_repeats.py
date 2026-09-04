#!/usr/bin/env python3
"""Compare odd-numbered independent generation runs using majority-vote GFC.

One question is one statistical sample.  Generation repeats are reduced to a
strict within-question majority before the paired comparison.  Answer and
judgment artifacts are joined with the repository's hash-validated loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_service_ab import (  # noqa: E402
    DEFAULT_CASES,
    cluster_key,
    load_cases,
    load_complete_run,
    quantile,
)


SCHEMA_VERSION = "pnu.generation-gfc-paired-analysis.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_majority(values: list[bool]) -> bool:
    if not values or len(values) % 2 == 0:
        raise ValueError("GFC majority requires a non-empty odd number of runs")
    if any(type(value) is not bool for value in values):
        raise ValueError("GFC votes must be booleans")
    return sum(values) > len(values) // 2


def validate_run_bindings(
    runs: list[dict[str, dict[str, Any]]], order: list[str]
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    experiment_ids: set[str] = set()
    condition_ids: set[str] = set()
    generation_run_ids: list[str] = []
    for index, run in enumerate(runs):
        triples = {
            (
                str(run[case_id].get("experiment_id") or ""),
                str(run[case_id].get("condition_id") or ""),
                str(run[case_id].get("generation_run_id") or ""),
            )
            for case_id in order
        }
        if len(triples) != 1:
            raise ValueError(f"run {index + 1} has inconsistent artifact bindings")
        experiment_id, condition_id, generation_run_id = next(iter(triples))
        if not all((experiment_id, condition_id, generation_run_id)):
            raise ValueError(f"run {index + 1} has an empty artifact binding")
        experiment_ids.add(experiment_id)
        condition_ids.add(condition_id)
        generation_run_ids.append(generation_run_id)
    if len(experiment_ids) != 1:
        raise ValueError("experiment_id changed across generation repeats")
    if len(condition_ids) != 1:
        raise ValueError("condition_id changed across generation repeats")
    if len(set(generation_run_ids)) != len(generation_run_ids):
        raise ValueError("generation_run_id must be unique across repeats")
    return {
        "experiment_id": next(iter(experiment_ids)),
        "condition_id": next(iter(condition_ids)),
        "generation_run_ids": generation_run_ids,
    }


def summarize_condition(
    order: list[str],
    cases: dict[str, dict[str, Any]],
    runs: list[dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if len(runs) % 2 == 0:
        raise ValueError("GFC majority requires an odd number of generation runs")
    binding = validate_run_bindings(runs, order)
    per_question: dict[str, dict[str, Any]] = {}
    run_gfc_counts = [0 for _ in runs]
    category_values: dict[str, list[bool]] = {}
    for case_id in order:
        votes: list[bool] = []
        for index, run in enumerate(runs):
            value = run[case_id].get("judge", {}).get("grounded_fully_correct")
            if type(value) is not bool:
                raise ValueError(
                    f"{case_id}: run {index + 1} grounded_fully_correct must be boolean"
                )
            votes.append(value)
            run_gfc_counts[index] += int(value)
        majority = strict_majority(votes)
        category = str(cases[case_id].get("category") or "unknown")
        category_values.setdefault(category, []).append(majority)
        per_question[case_id] = {
            "votes": votes,
            "true_count": sum(votes),
            "majority": majority,
        }
    majority_count = sum(item["majority"] for item in per_question.values())
    summary = {
        **binding,
        "generation_repeat_count": len(runs),
        "question_count": len(order),
        "effective_sample_n": len(order),
        "run_gfc_counts": run_gfc_counts,
        "run_gfc_rates": [count / len(order) for count in run_gfc_counts],
        "majority_gfc_count": majority_count,
        "majority_gfc_rate": majority_count / len(order),
        "categories": {
            category: {
                "questions": len(values),
                "majority_gfc_count": sum(values),
                "majority_gfc_rate": sum(values) / len(values),
            }
            for category, values in sorted(category_values.items())
        },
    }
    return summary, per_question


def paired_cluster_bootstrap(
    deltas: dict[str, int], iterations: int, seed: int
) -> dict[str, Any]:
    clusters: dict[str, list[str]] = {}
    for case_id in deltas:
        clusters.setdefault(cluster_key(case_id), []).append(case_id)
    cluster_names = sorted(clusters)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_ids: list[str] = []
        for _ in cluster_names:
            sampled_ids.extend(clusters[rng.choice(cluster_names)])
        samples.append(mean(deltas[case_id] for case_id in sampled_ids))
    return {
        "method": "paired cluster bootstrap by base/role question family",
        "iterations": iterations,
        "seed": seed,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def exact_mcnemar_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare_conditions(
    order: list[str],
    cases: dict[str, dict[str, Any]],
    per_a: dict[str, dict[str, Any]],
    per_b: dict[str, dict[str, Any]],
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gains = losses = ties_true = ties_false = 0
    deltas: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for case_id in order:
        majority_a = bool(per_a[case_id]["majority"])
        majority_b = bool(per_b[case_id]["majority"])
        delta = int(majority_b) - int(majority_a)
        deltas[case_id] = delta
        if delta > 0:
            gains += 1
        elif delta < 0:
            losses += 1
        elif majority_a:
            ties_true += 1
        else:
            ties_false += 1
        rows.append(
            {
                "id": case_id,
                "category": cases[case_id].get("category"),
                "role": cases[case_id].get("role"),
                "gfc_votes_a": per_a[case_id]["votes"],
                "gfc_votes_b": per_b[case_id]["votes"],
                "gfc_true_count_a": per_a[case_id]["true_count"],
                "gfc_true_count_b": per_b[case_id]["true_count"],
                "majority_gfc_a": majority_a,
                "majority_gfc_b": majority_b,
                "delta_b_minus_a": delta,
            }
        )
    comparison = {
        "majority_gfc_rate_delta_b_minus_a": mean(deltas.values()),
        "gains_b": gains,
        "losses_b": losses,
        "ties_true": ties_true,
        "ties_false": ties_false,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(gains, losses),
        "bootstrap": paired_cluster_bootstrap(deltas, iterations, seed),
    }
    return comparison, rows


def input_metadata(
    answers: list[Path], judgments: list[Path]
) -> list[dict[str, Any]]:
    return [
        {
            "answers": {"path": str(answer), "sha256": sha256_file(answer)},
            "judgments": {"path": str(judgment), "sha256": sha256_file(judgment)},
        }
        for answer, judgment in zip(answers, judgments)
    ]


def write_new_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ValueError(f"immutable output path already exists: {path}") from exc


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
            writer.writeheader()
            for row in materialized:
                serialized = dict(row)
                serialized["gfc_votes_a"] = json.dumps(row["gfc_votes_a"])
                serialized["gfc_votes_b"] = json.dumps(row["gfc_votes_b"])
                writer.writerow(serialized)
    except FileExistsError as exc:
        raise ValueError(f"immutable output path already exists: {path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runs-a", type=Path, nargs="+", required=True)
    parser.add_argument("--judgments-a", type=Path, nargs="+", required=True)
    parser.add_argument("--runs-b", type=Path, nargs="+", required=True)
    parser.add_argument("--judgments-b", type=Path, nargs="+", required=True)
    parser.add_argument("--label-a", default="baseline")
    parser.add_argument("--label-b", default="candidate")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counts = {
        len(args.runs_a),
        len(args.judgments_a),
        len(args.runs_b),
        len(args.judgments_b),
    }
    if len(counts) != 1:
        raise SystemExit("answer and judgment run counts must match for A and B")
    repeat_count = len(args.runs_a)
    if repeat_count == 0 or repeat_count % 2 == 0:
        raise SystemExit("GFC majority requires a non-empty odd number of runs")
    if args.bootstrap <= 0:
        raise SystemExit("--bootstrap must be positive")
    if args.json_out.resolve() == args.csv_out.resolve():
        raise SystemExit("--json-out and --csv-out must differ")
    existing_outputs = [
        str(path) for path in (args.json_out, args.csv_out) if path.exists()
    ]
    if existing_outputs:
        raise SystemExit(
            "immutable output path already exists: " + ", ".join(existing_outputs)
        )

    order, cases = load_cases(args.cases)
    expected_ids = set(order)
    try:
        runs_a = [
            load_complete_run(
                answer,
                expected_ids,
                judgment_path=judgment,
                allow_legacy_inline=False,
            )
            for answer, judgment in zip(args.runs_a, args.judgments_a)
        ]
        runs_b = [
            load_complete_run(
                answer,
                expected_ids,
                judgment_path=judgment,
                allow_legacy_inline=False,
            )
            for answer, judgment in zip(args.runs_b, args.judgments_b)
        ]
        summary_a, per_a = summarize_condition(order, cases, runs_a)
        summary_b, per_b = summarize_condition(order, cases, runs_b)
        comparison, rows = compare_conditions(
            order, cases, per_a, per_b, args.bootstrap, args.seed
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "methodology": {
                "sample_unit": "question",
                "generation_repeats_count_as_additional_samples": False,
                "within_question_reducer": "strict boolean majority",
                "effective_sample_n": len(order),
            },
            "labels": {"a": args.label_a, "b": args.label_b},
            "cases": {"path": str(args.cases), "sha256": sha256_file(args.cases)},
            "inputs": {
                "a": input_metadata(args.runs_a, args.judgments_a),
                "b": input_metadata(args.runs_b, args.judgments_b),
            },
            "a": summary_a,
            "b": summary_b,
            "comparison": comparison,
            "questions": rows,
        }
        write_new_text(
            args.json_out,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        write_csv(args.csv_out, rows)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    low, high = comparison["bootstrap"]["ci95"]
    print(
        f"{args.label_a}: majority GFC {summary_a['majority_gfc_count']}/"
        f"{summary_a['question_count']} ({summary_a['majority_gfc_rate']:.4f})"
    )
    print(
        f"{args.label_b}: majority GFC {summary_b['majority_gfc_count']}/"
        f"{summary_b['question_count']} ({summary_b['majority_gfc_rate']:.4f})"
    )
    print(
        f"B-A={comparison['majority_gfc_rate_delta_b_minus_a']:+.4f}, "
        f"gains/losses={comparison['gains_b']}/{comparison['losses_b']}, "
        f"cluster-bootstrap 95% CI=[{low:+.4f}, {high:+.4f}], "
        f"exact McNemar p={comparison['exact_mcnemar_two_sided_p']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
