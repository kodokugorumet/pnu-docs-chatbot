#!/usr/bin/env python3
"""Decompose retrieval, evidence, Judge, and deterministic-guard failures.

Inputs are immutable answer JSONL and Judge v11 JSONL artifacts.  Causes are
reported twice: a multi-label set that preserves overlap and one deterministic
primary cause for a partition whose total equals the non-GFC count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


class FailureAnalysisError(ValueError):
    """Raised when input artifacts cannot be joined or validated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(paths: Sequence[Path], *, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise FailureAnalysisError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise FailureAnalysisError(f"{path}:{line_number}: expected object")
                row = dict(row)
                row["_input_path"] = str(path)
                rows.append(row)
    if not rows:
        raise FailureAnalysisError(f"{kind} inputs contain no records")
    return rows


def _answer_key(row: dict[str, Any]) -> str:
    value = str(row.get("answer_id") or "")
    if not value:
        raise FailureAnalysisError("record is missing answer_id")
    return value


def join_artifacts(
    answers: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    answer_by_id: dict[str, dict[str, Any]] = {}
    for answer in answers:
        answer_id = _answer_key(answer)
        if answer_id in answer_by_id:
            raise FailureAnalysisError(f"duplicate answer_id in answers: {answer_id}")
        answer_by_id[answer_id] = answer

    judgment_by_id: dict[str, dict[str, Any]] = {}
    for judgment in judgments:
        answer_id = _answer_key(judgment)
        if answer_id in judgment_by_id:
            raise FailureAnalysisError(f"duplicate answer_id in judgments: {answer_id}")
        if judgment.get("error") not in (None, ""):
            raise FailureAnalysisError(
                f"judgment has terminal error for {answer_id}: {judgment.get('error')}"
            )
        judgment_by_id[answer_id] = judgment

    missing_judgments = sorted(set(answer_by_id) - set(judgment_by_id))
    extra_judgments = sorted(set(judgment_by_id) - set(answer_by_id))
    if missing_judgments or extra_judgments:
        raise FailureAnalysisError(
            f"answer/judgment mismatch: missing={missing_judgments[:5]} "
            f"extra={extra_judgments[:5]}"
        )

    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for answer in answers:
        judgment = judgment_by_id[_answer_key(answer)]
        for field in ("case_id", "condition_id", "generation_run_id"):
            if str(answer.get(field) or "") != str(judgment.get(field) or ""):
                raise FailureAnalysisError(
                    f"{answer['answer_id']}: {field} differs between artifacts"
                )
        expected_answer_sha = judgment.get("answer_sha256")
        if expected_answer_sha and expected_answer_sha != answer.get("answer_sha256"):
            raise FailureAnalysisError(
                f"{answer['answer_id']}: answer_sha256 differs between artifacts"
            )
        joined.append((answer, judgment))
    return joined


def _bool_field(value: Any, field: str) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and isinstance(value.get(field), bool):
        return value[field]
    return None


def _retrieval_hit(answer: dict[str, Any]) -> bool | None:
    return _bool_field(answer.get("retrieval_hit"), "matched")


def _required_evidence_all_at_8(answer: dict[str, Any]) -> bool | None:
    evidence_at_k = answer.get("evidence_at_k")
    if isinstance(evidence_at_k, dict):
        value = evidence_at_k.get("8") or evidence_at_k.get(8)
        result = _bool_field(value, "all_matched")
        if result is not None:
            return result
    return _bool_field(answer.get("evidence_hit"), "all_matched")


def _judge_payload(judgment: dict[str, Any]) -> dict[str, Any]:
    value = judgment.get("judge")
    if not isinstance(value, dict):
        raise FailureAnalysisError(
            f"{judgment.get('answer_id')}: judgment has no judge object"
        )
    score = value.get("score")
    gfc = value.get("grounded_fully_correct")
    if score not in (0, 1, 2) or not isinstance(gfc, bool):
        raise FailureAnalysisError(
            f"{judgment.get('answer_id')}: invalid score/GFC: {score}/{gfc}"
        )
    return value


def _guard_rules(judgment: dict[str, Any]) -> list[str]:
    guard = judgment.get("deterministic_guard")
    guard = guard if isinstance(guard, dict) else {}
    rules = guard.get("rules")
    if not isinstance(rules, list):
        return []
    return [str(value) for value in rules if str(value)]


def failure_causes(judgment: dict[str, Any]) -> list[str]:
    judge = _judge_payload(judgment)
    if judge["grounded_fully_correct"]:
        return []
    claim_checks = judge.get("claim_checks")
    checks = claim_checks if isinstance(claim_checks, list) else []
    statuses = {
        str(item.get("status") or "")
        for item in checks
        if isinstance(item, dict)
    }
    causes: list[str] = []
    guard_rules = _guard_rules(judgment)
    if "answerable_clear_refusal_forces_score_zero" in guard_rules:
        causes.append("explicit_refusal_guard")
    if judge.get("abstention") == "inappropriate":
        causes.append("inappropriate_abstention")
    if "missing" in statuses:
        causes.append("required_claim_missing")
    unsupported_facts = judge.get("unsupported_facts")
    if (
        "unsupported" in statuses
        or isinstance(unsupported_facts, list)
        and bool(unsupported_facts)
    ):
        causes.append("unsupported_fact")
    contradictions = judge.get("contradictions")
    if (
        "contradicted" in statuses
        or isinstance(contradictions, list)
        and bool(contradictions)
    ):
        causes.append("contradiction")
    if judge.get("citation_support") == "partial":
        causes.append("partial_citation")
    return causes


PRIMARY_PRECEDENCE = (
    "contradiction",
    "unsupported_fact",
    "explicit_refusal_guard",
    "required_claim_missing",
    "partial_citation",
    "inappropriate_abstention",
)


def primary_cause(causes: Sequence[str]) -> str:
    for candidate in PRIMARY_PRECEDENCE:
        if candidate in causes:
            return candidate
    return "other_non_gfc"


def _truth_label(value: bool | None) -> str:
    return "true" if value is True else "false" if value is False else "unknown"


def _summarize_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    score_counts = Counter(str(item["judge_score"]) for item in items)
    gfc_count = sum(item["grounded_fully_correct"] for item in items)
    non_gfc = [item for item in items if not item["grounded_fully_correct"]]
    multilabel = Counter(
        cause for item in non_gfc for cause in item["failure_causes"]
    )
    primary = Counter(item["primary_failure_cause"] for item in non_gfc)
    guards = Counter(rule for item in items for rule in item["guard_rules"])
    cross = Counter(
        (
            _truth_label(item["retrieval_hit"]),
            _truth_label(item["required_evidence_all_at_8"]),
            str(item["judge_score"]),
            _truth_label(item["grounded_fully_correct"]),
        )
        for item in items
    )
    if sum(primary.values()) != len(non_gfc):
        raise FailureAnalysisError("primary cause partition does not equal non-GFC")
    return {
        "n": len(items),
        "mean_score": sum(item["judge_score"] for item in items) / len(items),
        "score_counts": dict(sorted(score_counts.items())),
        "gfc_count": gfc_count,
        "gfc_rate": gfc_count / len(items),
        "non_gfc_count": len(non_gfc),
        "failure_causes_multilabel": dict(sorted(multilabel.items())),
        "failure_primary_partition": dict(sorted(primary.items())),
        "guard_rule_counts": dict(sorted(guards.items())),
        "cross_table": [
            {
                "retrieval_hit": retrieval,
                "required_evidence_all_at_8": evidence,
                "judge_score": int(score),
                "grounded_fully_correct": gfc == "true",
                "count": count,
            }
            for (retrieval, evidence, score, gfc), count in sorted(cross.items())
        ],
    }


def analyze(
    joined: Sequence[tuple[dict[str, Any], dict[str, Any]]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for answer, judgment in joined:
        judge = _judge_payload(judgment)
        causes = failure_causes(judgment)
        row = {
            "case_id": str(answer.get("case_id") or ""),
            "answer_id": str(answer.get("answer_id") or ""),
            "condition_id": str(answer.get("condition_id") or "unknown"),
            "generation_run_id": str(
                answer.get("generation_run_id") or "unknown"
            ),
            "retrieval_hit": _retrieval_hit(answer),
            "required_evidence_all_at_8": _required_evidence_all_at_8(answer),
            "judge_score": int(judge["score"]),
            "grounded_fully_correct": bool(judge["grounded_fully_correct"]),
            "failure_causes": causes,
            "primary_failure_cause": (
                None if judge["grounded_fully_correct"] else primary_cause(causes)
            ),
            "guard_rules": _guard_rules(judgment),
            "citation_support": judge.get("citation_support"),
            "abstention": judge.get("abstention"),
            "judge_reason": judge.get("reason"),
            "answer_input_path": answer.get("_input_path"),
            "judgment_input_path": judgment.get("_input_path"),
        }
        rows.append(row)
        condition_rows[row["condition_id"]].append(row)

    summaries: dict[str, Any] = {}
    for condition_id, items in sorted(condition_rows.items()):
        summaries[condition_id] = _summarize_items(items)
        run_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            run_groups[item["generation_run_id"]].append(item)
        summaries[condition_id]["runs"] = {
            run_id: _summarize_items(run_items)
            for run_id, run_items in sorted(run_groups.items())
        }
    return {
        "schema_version": "pnu.generation-failure-analysis.v1",
        "methodology": {
            "gfc": "judge.grounded_fully_correct",
            "required_evidence": "answer.evidence_at_k['8'].all_matched",
            "multilabel_note": (
                "Cause counts overlap and can exceed non_gfc_count. "
                "failure_primary_partition is the exclusive partition."
            ),
            "required_claim_missing": "at least one claim_check status is missing",
            "partial_citation": "judge.citation_support is exactly partial",
        },
        "conditions": summaries,
        "records": rows,
    }


def validate_expected_summary(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    expected = json.loads(path.read_text(encoding="utf-8"))
    labels = expected.get("labels") if isinstance(expected, dict) else None
    labels = labels if isinstance(labels, dict) else {}
    checks: list[dict[str, Any]] = []
    for side in ("a", "b"):
        expected_condition = expected.get(side)
        if not isinstance(expected_condition, dict):
            continue
        condition_id = str(labels.get(side) or side).lower()
        observed = payload["conditions"].get(condition_id)
        if observed is None:
            raise FailureAnalysisError(
                f"expected summary condition {condition_id} is missing"
            )
        expected_mean = float(expected_condition.get("mean"))
        expected_counts = {
            str(key): int(value)
            for key, value in (expected_condition.get("score_counts") or {}).items()
        }
        mean_matches = abs(observed["mean_score"] - expected_mean) <= 1e-12
        counts_match = observed["score_counts"] == expected_counts
        gfc_matches_score_two = observed["gfc_count"] == expected_counts.get("2", 0)
        checks.append(
            {
                "condition_id": condition_id,
                "mean_matches": mean_matches,
                "score_counts_match": counts_match,
                "gfc_matches_score_2_count": gfc_matches_score_two,
            }
        )
        if not all((mean_matches, counts_match, gfc_matches_score_two)):
            raise FailureAnalysisError(
                f"{condition_id}: result disagrees with expected summary"
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "checks": checks,
        "all_match": bool(checks) and all(
            all(value for key, value in item.items() if key != "condition_id")
            for item in checks
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FailureAnalysisError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FailureAnalysisError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case_id",
        "answer_id",
        "condition_id",
        "generation_run_id",
        "retrieval_hit",
        "required_evidence_all_at_8",
        "judge_score",
        "grounded_fully_correct",
        "primary_failure_cause",
        "failure_causes",
        "guard_rules",
        "citation_support",
        "abstention",
    )
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for item in payload["records"]:
            writer.writerow(
                {
                    **{field: item.get(field) for field in fields},
                    "failure_causes": ";".join(item["failure_causes"]),
                    "guard_rules": ";".join(item["guard_rules"]),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", action="append", required=True, type=Path)
    parser.add_argument("--judgments", action="append", required=True, type=Path)
    parser.add_argument("--expected-summary", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = [*args.answers, *args.judgments]
    if args.expected_summary:
        input_paths.append(args.expected_summary)
    for path in input_paths:
        if not path.is_file() or path.is_symlink():
            raise FailureAnalysisError(
                f"input must be a regular non-symlink file: {path}"
            )
    if args.out_json.resolve() == args.out_csv.resolve():
        raise FailureAnalysisError("JSON and CSV outputs must differ")
    answers = load_jsonl(args.answers, kind="answer")
    judgments = load_jsonl(args.judgments, kind="judgment")
    payload = analyze(join_artifacts(answers, judgments))
    payload["inputs"] = {
        "answers": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.answers
        ],
        "judgments": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.judgments
        ],
    }
    if args.expected_summary:
        payload["expected_summary_validation"] = validate_expected_summary(
            payload, args.expected_summary
        )
    _write_json(args.out_json, payload)
    _write_csv(args.out_csv, payload)
    print(f"records={len(payload['records'])} conditions={len(payload['conditions'])}")
    for condition_id, summary in payload["conditions"].items():
        print(
            f"{condition_id}: mean={summary['mean_score']:.6f} "
            f"gfc={summary['gfc_count']}/{summary['n']} "
            f"non_gfc={summary['non_gfc_count']}"
        )
    print(f"json={args.out_json}")
    print(f"csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
