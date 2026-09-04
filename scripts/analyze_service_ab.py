#!/usr/bin/env python3
"""Strict paired A/B analysis for the 45-question service evaluation.

Unlike the lightweight progress summarizer, this command rejects partial or duplicate
runs, backfills evidence hits from saved source chunk IDs, verifies retrieval stability
across repeats, and produces question-level paired results plus a cluster bootstrap CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from service_eval_artifacts import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    build_answer_identity,
    build_judgment_identity,
    join_answers_and_judgments,
    load_unique_jsonl,
    validate_answer_record,
)

DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-eval.jsonl"
EVIDENCE_CUTOFFS = (5, 8)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    return records


def load_cases(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    records = load_jsonl(path)
    order: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("id") or "")
        if not case_id:
            raise ValueError(f"{path}: case without id")
        if case_id in cases:
            raise ValueError(f"{path}: duplicate case id {case_id}")
        order.append(case_id)
        cases[case_id] = record
    return order, cases


def load_complete_run(
    path: Path,
    expected_ids: set[str],
    *,
    judgment_path: Path | None = None,
    allow_legacy_inline: bool = True,
) -> dict[str, dict[str, Any]]:
    records = load_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id") or record.get("id") or "")
        if not case_id:
            raise ValueError(f"{path}: record without id")
        if case_id in by_id:
            raise ValueError(f"{path}: duplicate id {case_id}")
        if record.get("error"):
            raise ValueError(f"{path}: {case_id} has error: {record['error']}")
        if judgment_path is not None:
            if "judge" in record:
                raise ValueError(
                    f"{path}: {case_id} mixes inline judge with separate judgment"
                )
            try:
                validate_answer_record(record)
            except ValueError as exc:
                raise ValueError(f"{path}: {case_id}: {exc}") from exc
        elif allow_legacy_inline:
            judge = record.get("judge")
            score = judge.get("score") if isinstance(judge, dict) else None
            if score not in (0, 1, 2):
                raise ValueError(
                    f"{path}: {case_id} has invalid judge score {score!r}"
                )
        else:
            raise ValueError(
                f"{path}: separate judgment artifact is required; "
                "use --legacy-inline only for historical runs"
            )
        by_id[case_id] = record

    actual_ids = set(by_id)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(
            f"{path}: incomplete ID set; missing={missing or 'none'}, extra={extra or 'none'}"
        )
    if judgment_path is None:
        return by_id

    try:
        judgments = load_unique_jsonl(judgment_path, key="judgment_id")
        return join_answers_and_judgments(by_id, judgments)
    except ValueError as exc:
        raise ValueError(f"{judgment_path}: {exc}") from exc


def normalized(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def evidence_values_at_k(
    record: dict[str, Any], case: dict[str, Any], k: int
) -> dict[str, Any]:
    wanted_chunks: set[str] = set()
    for index, item in enumerate(case.get("evidence") or []):
        if not isinstance(item, dict):
            raise ValueError(f"evidence[{index}] must be an object")
        required = item.get("required_for_answer", True)
        if not isinstance(required, bool):
            raise ValueError(
                f"evidence[{index}].required_for_answer must be boolean"
            )
        if required and item.get("chunk_id"):
            wanted_chunks.add(str(item["chunk_id"]))
    returned_chunks = {
        str(source.get("chunk_id"))
        for source in (record.get("sources") or [])[: max(0, int(k))]
        if source.get("chunk_id")
    }
    evidence_chunks = sorted(wanted_chunks & returned_chunks)
    missing_chunks = sorted(wanted_chunks - returned_chunks)
    return {
        "k": int(k),
        "matched": bool(evidence_chunks),
        "all_matched": bool(wanted_chunks) and not missing_chunks,
        "recall": (
            len(evidence_chunks) / len(wanted_chunks)
            if wanted_chunks
            else 0.0
        ),
        "evidence_chunks": evidence_chunks,
        "missing_chunks": missing_chunks,
        "wanted": len(wanted_chunks),
    }


def retrieval_values(record: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    stored = record.get("retrieval_hit")
    if isinstance(stored, dict) and "matched" in stored:
        matched = bool(stored.get("matched"))
        rank = stored.get("rank") if matched else None
    else:
        want_title = normalized((case.get("expected") or {}).get("source_title_contains"))
        matched = False
        rank = None
        seen_titles: set[str] = set()
        for source in record.get("sources") or []:
            title = normalized(source.get("source_title"))
            if title in seen_titles:
                continue
            seen_titles.add(title)
            if want_title and want_title in title:
                matched = True
                rank = len(seen_titles)
                break

    evidence_at_k = {
        str(k): evidence_values_at_k(record, case, k)
        for k in EVIDENCE_CUTOFFS
    }
    context_evidence = evidence_at_k[str(max(EVIDENCE_CUTOFFS))]
    return {
        "hit": matched,
        "rank": int(rank) if rank is not None else None,
        # Backward-compatible aliases refer to final context@8, not @5.
        "evidence_hit": context_evidence["matched"],
        "evidence_chunks": context_evidence["evidence_chunks"],
        "evidence_at_k": evidence_at_k,
    }


def source_signature(record: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(source.get("source_title") or ""),
            str(source.get("source_host") or ""),
            str(source.get("chunk_id") or ""),
        )
        for source in record.get("sources") or []
    )


def condition_summary(
    order: list[str],
    cases: dict[str, dict[str, Any]],
    runs: list[dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not runs:
        raise ValueError("at least one run is required")

    per_question: dict[str, dict[str, Any]] = {}
    category_scores: dict[str, list[float]] = {}
    for case_id in order:
        scores = [int(run[case_id]["judge"]["score"]) for run in runs]
        retrievals = [retrieval_values(run[case_id], cases[case_id]) for run in runs]
        first_retrieval = retrievals[0]
        if any(item != first_retrieval for item in retrievals[1:]):
            raise ValueError(f"retrieval changed across repeated runs for {case_id}")
        signatures = [source_signature(run[case_id]) for run in runs]
        if any(item != signatures[0] for item in signatures[1:]):
            raise ValueError(f"source order changed across repeated runs for {case_id}")

        case_mean = mean(scores)
        category = str(cases[case_id].get("category") or "unknown")
        category_scores.setdefault(category, []).append(case_mean)
        per_question[case_id] = {
            "scores": scores,
            "mean": case_mean,
            "retrieval": first_retrieval,
            "source_signature": signatures[0],
        }

    run_means = [mean(int(run[case_id]["judge"]["score"]) for case_id in order) for run in runs]
    hit_count = sum(item["retrieval"]["hit"] for item in per_question.values())
    evidence_summary: dict[str, dict[str, Any]] = {}
    for k in EVIDENCE_CUTOFFS:
        values = [
            item["retrieval"]["evidence_at_k"][str(k)]
            for item in per_question.values()
        ]
        any_hits = sum(bool(value["matched"]) for value in values)
        all_hits = sum(bool(value["all_matched"]) for value in values)
        evidence_summary[str(k)] = {
            "metric_family": "exact_required_gold_chunk_coverage",
            "any_hits": any_hits,
            "any_hit_rate": any_hits / len(values),
            "all_hits": all_hits,
            "all_hit_rate": all_hits / len(values),
            "mean_gold_chunk_recall": mean(
                float(value["recall"]) for value in values
            ),
        }
    context_summary = evidence_summary[str(max(EVIDENCE_CUTOFFS))]
    reciprocal_ranks = [
        1.0 / item["retrieval"]["rank"] if item["retrieval"]["rank"] else 0.0
        for item in per_question.values()
    ]
    score_counts = {str(score): 0 for score in (0, 1, 2)}
    for run in runs:
        for case_id in order:
            key = str(run[case_id]["judge"]["score"])
            score_counts[key] += 1

    summary = {
        "runs": len(runs),
        "questions": len(order),
        "run_means": run_means,
        "mean": mean(item["mean"] for item in per_question.values()),
        "run_mean_sd": pstdev(run_means) if len(run_means) > 1 else 0.0,
        "score_counts": score_counts,
        "retrieval": {
            "hits": hit_count,
            "hit_rate": hit_count / len(order),
            "mrr": mean(reciprocal_ranks),
            # Legacy fields are retained for the existing visual review and
            # explicitly map to Any-Evidence@8.
            "evidence_hits": context_summary["any_hits"],
            "evidence_hit_rate": context_summary["any_hit_rate"],
            "evidence_metric_alias": "Any-Gold-Chunk@8",
            "evidence_at_k": evidence_summary,
        },
        "categories": {
            category: {"questions": len(values), "mean": mean(values)}
            for category, values in sorted(category_scores.items())
        },
    }
    return summary, per_question


def cluster_key(case_id: str) -> str:
    return case_id.split("_role_", 1)[0]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty list")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_cluster_bootstrap(
    deltas: dict[str, float], iterations: int, seed: int
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


def compare_conditions(
    order: list[str],
    cases: dict[str, dict[str, Any]],
    per_a: dict[str, dict[str, Any]],
    per_b: dict[str, dict[str, Any]],
    iterations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    wins = ties = losses = 0
    deltas: dict[str, float] = {}
    for case_id in order:
        delta = per_b[case_id]["mean"] - per_a[case_id]["mean"]
        deltas[case_id] = delta
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
        rows.append(
            {
                "id": case_id,
                "category": cases[case_id].get("category"),
                "role": cases[case_id].get("role"),
                "scores_a": per_a[case_id]["scores"],
                "scores_b": per_b[case_id]["scores"],
                "mean_a": per_a[case_id]["mean"],
                "mean_b": per_b[case_id]["mean"],
                "delta_b_minus_a": delta,
                "retrieval_hit_a": per_a[case_id]["retrieval"]["hit"],
                "retrieval_hit_b": per_b[case_id]["retrieval"]["hit"],
                "rank_a": per_a[case_id]["retrieval"]["rank"],
                "rank_b": per_b[case_id]["retrieval"]["rank"],
                "evidence_hit_a": per_a[case_id]["retrieval"]["evidence_hit"],
                "evidence_hit_b": per_b[case_id]["retrieval"]["evidence_hit"],
                "gold_chunk_recall_at_5_a": per_a[case_id]["retrieval"]["evidence_at_k"]["5"]["recall"],
                "gold_chunk_recall_at_5_b": per_b[case_id]["retrieval"]["evidence_at_k"]["5"]["recall"],
                "all_gold_chunks_at_5_a": per_a[case_id]["retrieval"]["evidence_at_k"]["5"]["all_matched"],
                "all_gold_chunks_at_5_b": per_b[case_id]["retrieval"]["evidence_at_k"]["5"]["all_matched"],
                "gold_chunk_recall_at_8_a": per_a[case_id]["retrieval"]["evidence_at_k"]["8"]["recall"],
                "gold_chunk_recall_at_8_b": per_b[case_id]["retrieval"]["evidence_at_k"]["8"]["recall"],
                "all_gold_chunks_at_8_a": per_a[case_id]["retrieval"]["evidence_at_k"]["8"]["all_matched"],
                "all_gold_chunks_at_8_b": per_b[case_id]["retrieval"]["evidence_at_k"]["8"]["all_matched"],
                "sources_changed": (
                    per_a[case_id]["source_signature"]
                    != per_b[case_id]["source_signature"]
                ),
            }
        )
    comparison = {
        "mean_delta_b_minus_a": mean(deltas.values()),
        "wins_b": wins,
        "ties": ties,
        "losses_b": losses,
        "bootstrap": paired_cluster_bootstrap(deltas, iterations, seed),
    }
    return comparison, rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["scores_a"] = json.dumps(row["scores_a"])
            serialized["scores_b"] = json.dumps(row["scores_b"])
            writer.writerow(serialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runs-a", type=Path, nargs="+", required=True)
    parser.add_argument("--runs-b", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--judgments-a",
        type=Path,
        nargs="+",
        help="--runs-a와 같은 순서의 별도 judgment JSONL",
    )
    parser.add_argument(
        "--judgments-b",
        type=Path,
        nargs="+",
        help="--runs-b와 같은 순서의 별도 judgment JSONL",
    )
    parser.add_argument(
        "--legacy-inline",
        action="store_true",
        help="과거 answer+judge 혼합 JSONL을 명시적으로 허용",
    )
    parser.add_argument("--label-a", default="baseline")
    parser.add_argument("--label-b", default="candidate")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.runs_a) != len(args.runs_b):
        raise SystemExit("A and B must have the same number of repeated runs")
    if args.legacy_inline:
        if args.judgments_a or args.judgments_b:
            raise SystemExit(
                "--legacy-inline cannot be combined with --judgments-a/b"
            )
    else:
        if not args.judgments_a or not args.judgments_b:
            raise SystemExit(
                "separate --judgments-a and --judgments-b are required; "
                "use --legacy-inline only for historical artifacts"
            )
        if len(args.judgments_a) != len(args.runs_a):
            raise SystemExit("--judgments-a count must match --runs-a")
        if len(args.judgments_b) != len(args.runs_b):
            raise SystemExit("--judgments-b count must match --runs-b")
    if args.bootstrap <= 0:
        raise SystemExit("--bootstrap must be positive")

    order, cases = load_cases(args.cases)
    expected_ids = set(order)
    try:
        runs_a = [
            load_complete_run(
                path,
                expected_ids,
                judgment_path=(
                    args.judgments_a[index]
                    if args.judgments_a
                    else None
                ),
                allow_legacy_inline=args.legacy_inline,
            )
            for index, path in enumerate(args.runs_a)
        ]
        runs_b = [
            load_complete_run(
                path,
                expected_ids,
                judgment_path=(
                    args.judgments_b[index]
                    if args.judgments_b
                    else None
                ),
                allow_legacy_inline=args.legacy_inline,
            )
            for index, path in enumerate(args.runs_b)
        ]
        summary_a, per_a = condition_summary(order, cases, runs_a)
        summary_b, per_b = condition_summary(order, cases, runs_b)
        comparison, rows = compare_conditions(
            order, cases, per_a, per_b, args.bootstrap, args.seed
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    result = {
        "schema_version": 1,
        "cases": str(args.cases),
        "labels": {"a": args.label_a, "b": args.label_b},
        "runs": {
            "a": [str(path) for path in args.runs_a],
            "b": [str(path) for path in args.runs_b],
            "judgments_a": [str(path) for path in (args.judgments_a or [])],
            "judgments_b": [str(path) for path in (args.judgments_b or [])],
            "legacy_inline": args.legacy_inline,
        },
        "a": summary_a,
        "b": summary_b,
        "comparison": comparison,
        "questions": rows,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.csv_out:
        write_csv(args.csv_out, rows)

    print(
        f"A {args.label_a}: mean={summary_a['mean']:.4f}, "
        f"Hit@5={summary_a['retrieval']['hit_rate']:.3f}, "
        f"MRR={summary_a['retrieval']['mrr']:.3f}, "
        f"RequiredGoldChunkRecall@5={summary_a['retrieval']['evidence_at_k']['5']['mean_gold_chunk_recall']:.3f}, "
        f"RequiredGoldChunkRecall@8={summary_a['retrieval']['evidence_at_k']['8']['mean_gold_chunk_recall']:.3f}"
    )
    print(
        f"B {args.label_b}: mean={summary_b['mean']:.4f}, "
        f"Hit@5={summary_b['retrieval']['hit_rate']:.3f}, "
        f"MRR={summary_b['retrieval']['mrr']:.3f}, "
        f"RequiredGoldChunkRecall@5={summary_b['retrieval']['evidence_at_k']['5']['mean_gold_chunk_recall']:.3f}, "
        f"RequiredGoldChunkRecall@8={summary_b['retrieval']['evidence_at_k']['8']['mean_gold_chunk_recall']:.3f}"
    )
    low, high = comparison["bootstrap"]["ci95"]
    print(
        f"B-A={comparison['mean_delta_b_minus_a']:+.4f}, "
        f"wins/ties/losses={comparison['wins_b']}/{comparison['ties']}/"
        f"{comparison['losses_b']}, cluster-bootstrap 95% CI=[{low:+.4f}, {high:+.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
