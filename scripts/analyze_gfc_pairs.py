#!/usr/bin/env python3
"""Fail-closed paired GFC analysis for two judge-repeat summaries.

The input summaries must use ``pnu.judge-repeat-summary.v1``.  This command
does not treat repeated judge calls as independent observations: one paired
question is one statistical sample.  Before analysis it re-opens every answer
and judgment artifact referenced by each summary, verifies its SHA-256, and
rebuilds the repeat summary to check answer/case/condition bindings.

The default condition binding remains C0 versus C1.  ``--condition-a`` and
``--condition-b`` allow other explicitly named conditions while legacy v1
output field names containing ``c0``/``c1`` remain aliases for A/B.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_judge_repeats import (  # noqa: E402
    SUMMARY_SCHEMA_VERSION,
    aggregate_repeats,
    load_answers,
    sha256_file,
    strict_boolean_majority,
)


OUTPUT_SCHEMA_VERSION = "pnu.gfc-paired-analysis.v1"
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_SIGN_FLIP_ITERATIONS = 100_000
DEFAULT_SEED = 20260914
EXACT_SIGN_FLIP_MAX_NONZERO = 20


@dataclass(frozen=True)
class ValidatedSummary:
    path: Path
    artifact_sha256: str
    condition_id: str
    experiment_id: str
    generation_run_id: str
    repeat_count: int
    order: list[str]
    cases_by_id: dict[str, dict[str, Any]]
    answer_artifact: dict[str, Any]
    judgment_artifacts: list[dict[str, Any]]


def _required_nonempty_text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: expected non-empty string")
    return value


def _required_int(value: Any, *, location: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{location}: expected integer >= {minimum}")
    return value


def _validate_sha256(value: Any, *, location: str) -> str:
    digest = _required_nonempty_text(value, location=location)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{location}: expected lowercase SHA-256")
    return digest


def _resolve_referenced_artifact(
    raw_path: Any,
    *,
    summary_path: Path,
    expected_sha256: str,
    location: str,
) -> Path:
    text_path = _required_nonempty_text(raw_path, location=f"{location}.path")
    supplied = Path(text_path).expanduser()
    if supplied.is_absolute():
        candidates = [supplied]
    else:
        # Current artifacts were recorded relative to the repository root.  A
        # summary-local candidate also makes copied test bundles portable.
        candidates = [REPO_ROOT / supplied, summary_path.parent / supplied]
        cwd_candidate = Path.cwd() / supplied
        if cwd_candidate not in candidates:
            candidates.insert(1, cwd_candidate)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(normalized)

    existing = [candidate for candidate in unique_candidates if candidate.is_file()]
    if not existing:
        raise ValueError(
            f"{location}: referenced artifact does not exist: {text_path}"
        )

    matching = [candidate for candidate in existing if sha256_file(candidate) == expected_sha256]
    if not matching:
        actual = ", ".join(
            f"{candidate}={sha256_file(candidate)}" for candidate in existing
        )
        raise ValueError(
            f"{location}: artifact SHA-256 mismatch; expected {expected_sha256}; "
            f"observed {actual}"
        )
    # If the same relative reference resolves to multiple byte-identical files,
    # prefer the repository-root interpretation used by the producer.
    return matching[0]


def _validate_summary_case_shape(
    summary_path: Path,
    payload: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], int]:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{summary_path}: cases must be a non-empty list")

    summary_block = payload.get("summary")
    if not isinstance(summary_block, dict):
        raise ValueError(f"{summary_path}: missing summary object")
    repeat_count = _required_int(
        summary_block.get("judge_repeat_count"),
        location=f"{summary_path}: summary.judge_repeat_count",
        minimum=1,
    )

    order: list[str] = []
    cases_by_id: dict[str, dict[str, Any]] = {}
    answer_ids: set[str] = set()
    for index, case in enumerate(raw_cases):
        location = f"{summary_path}: cases[{index}]"
        if not isinstance(case, dict):
            raise ValueError(f"{location}: expected object")
        case_id = _required_nonempty_text(case.get("case_id"), location=f"{location}.case_id")
        if case_id in cases_by_id:
            raise ValueError(f"{summary_path}: duplicate case_id {case_id}")
        answer_id = _required_nonempty_text(
            case.get("answer_id"), location=f"{location}.answer_id"
        )
        if answer_id in answer_ids:
            raise ValueError(f"{summary_path}: duplicate answer_id {answer_id}")
        answer_ids.add(answer_id)

        values = case.get("grounded_fully_correct_values")
        if not isinstance(values, list):
            raise ValueError(f"{location}: grounded_fully_correct_values must be a list")
        if len(values) != repeat_count:
            raise ValueError(
                f"{location}: repeat count mismatch; expected {repeat_count}, "
                f"got {len(values)}"
            )
        if any(type(value) is not bool for value in values):
            raise ValueError(
                f"{location}: grounded_fully_correct_values must contain only booleans; "
                "null is not allowed"
            )

        true_count = sum(values)
        if case.get("gfc_true_count") != true_count:
            raise ValueError(f"{location}: gfc_true_count mismatch")
        expected_majority = strict_boolean_majority(true_count, repeat_count)
        if expected_majority is None:
            raise ValueError(
                f"{location}: gfc_majority is null because the repeated votes tie; "
                "binary sensitivity analysis is undefined"
            )
        if type(case.get("gfc_majority")) is not bool:
            raise ValueError(f"{location}: gfc_majority must be boolean, not null")
        if case["gfc_majority"] is not expected_majority:
            raise ValueError(f"{location}: gfc_majority mismatch")

        order.append(case_id)
        cases_by_id[case_id] = case

    question_count = _required_int(
        summary_block.get("question_count"),
        location=f"{summary_path}: summary.question_count",
        minimum=1,
    )
    effective_n = _required_int(
        summary_block.get("effective_sample_n"),
        location=f"{summary_path}: summary.effective_sample_n",
        minimum=1,
    )
    judgment_record_count = _required_int(
        summary_block.get("judgment_record_count"),
        location=f"{summary_path}: summary.judgment_record_count",
        minimum=1,
    )
    if question_count != len(order) or effective_n != len(order):
        raise ValueError(
            f"{summary_path}: question/effective sample count does not match cases"
        )
    if judgment_record_count != len(order) * repeat_count:
        raise ValueError(f"{summary_path}: judgment_record_count mismatch")

    methodology = payload.get("methodology")
    if not isinstance(methodology, dict):
        raise ValueError(f"{summary_path}: missing methodology object")
    if methodology.get("sample_unit") != "question":
        raise ValueError(f"{summary_path}: sample_unit must be question")
    if methodology.get("judge_repeats_count_as_additional_samples") is not False:
        raise ValueError(
            f"{summary_path}: judge repeats must not count as additional samples"
        )
    if methodology.get("effective_sample_n") != len(order):
        raise ValueError(f"{summary_path}: methodology.effective_sample_n mismatch")
    return order, cases_by_id, repeat_count


def validate_repeat_summary(path: Path, *, expected_condition: str) -> ValidatedSummary:
    summary_path = path.resolve()
    summary_sha_before = sha256_file(summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{summary_path}: expected JSON object")
    if payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"{summary_path}: schema_version must be {SUMMARY_SCHEMA_VERSION!r}"
        )
    order, cases_by_id, repeat_count = _validate_summary_case_shape(summary_path, payload)

    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{summary_path}: missing inputs object")
    answer_meta = inputs.get("answers")
    if not isinstance(answer_meta, dict):
        raise ValueError(f"{summary_path}: missing inputs.answers object")
    answer_sha = _validate_sha256(
        answer_meta.get("sha256"), location=f"{summary_path}: inputs.answers.sha256"
    )
    answer_path = _resolve_referenced_artifact(
        answer_meta.get("path"),
        summary_path=summary_path,
        expected_sha256=answer_sha,
        location=f"{summary_path}: inputs.answers",
    )

    judgment_meta = inputs.get("judgments")
    if not isinstance(judgment_meta, list) or not judgment_meta:
        raise ValueError(f"{summary_path}: inputs.judgments must be a non-empty list")
    if len(judgment_meta) != repeat_count:
        raise ValueError(
            f"{summary_path}: judgment artifact count does not match repeat count"
        )

    judgment_paths: list[Path] = []
    seen_paths: set[Path] = set()
    seen_shas: set[str] = set()
    seen_run_ids: set[str] = set()
    for index, metadata in enumerate(judgment_meta):
        location = f"{summary_path}: inputs.judgments[{index}]"
        if not isinstance(metadata, dict):
            raise ValueError(f"{location}: expected object")
        artifact_sha = _validate_sha256(
            metadata.get("sha256"), location=f"{location}.sha256"
        )
        judge_run_id = _required_nonempty_text(
            metadata.get("judge_run_id"), location=f"{location}.judge_run_id"
        )
        if judge_run_id in seen_run_ids:
            raise ValueError(f"{summary_path}: duplicate judge_run_id {judge_run_id}")
        seen_run_ids.add(judge_run_id)
        judgment_path = _resolve_referenced_artifact(
            metadata.get("path"),
            summary_path=summary_path,
            expected_sha256=artifact_sha,
            location=location,
        )
        if judgment_path in seen_paths:
            raise ValueError(f"{summary_path}: duplicate judgment artifact path {judgment_path}")
        if artifact_sha in seen_shas:
            raise ValueError(f"{summary_path}: duplicate judgment artifact SHA-256 {artifact_sha}")
        seen_paths.add(judgment_path)
        seen_shas.add(artifact_sha)
        judgment_paths.append(judgment_path)

    # Rebuild from immutable source artifacts.  The aggregator validates answer
    # and judgment record hashes, one-to-one answer joins, and optional artifact
    # bindings embedded in judgments.
    rebuilt, rebuilt_cases = aggregate_repeats(
        answer_path=answer_path,
        judgment_paths=judgment_paths,
    )
    if payload.get("methodology") != rebuilt["methodology"]:
        raise ValueError(f"{summary_path}: methodology differs from rebuilt summary")
    if payload.get("summary") != rebuilt["summary"]:
        raise ValueError(f"{summary_path}: aggregate metrics differ from source artifacts")
    if payload.get("cases") != rebuilt_cases:
        raise ValueError(f"{summary_path}: case rows differ from source artifacts")

    rebuilt_answer_meta = rebuilt["inputs"]["answers"]
    if rebuilt_answer_meta["sha256"] != answer_sha:
        raise ValueError(f"{summary_path}: rebuilt answer artifact SHA-256 mismatch")
    rebuilt_judgments = rebuilt["inputs"]["judgments"]
    for index, (recorded, actual) in enumerate(zip(judgment_meta, rebuilt_judgments)):
        location = f"{summary_path}: inputs.judgments[{index}]"
        for field in ("sha256", "judge_run_id", "judge_config_sha256"):
            if recorded.get(field) != actual.get(field):
                raise ValueError(f"{location}: {field} differs from source artifact")

    answer_order, answers_by_case = load_answers(answer_path)
    if answer_order != order:
        raise ValueError(f"{summary_path}: answer artifact case order mismatch")
    condition_values = {str(answer["condition_id"]) for answer in answers_by_case.values()}
    experiment_values = {str(answer["experiment_id"]) for answer in answers_by_case.values()}
    generation_values = {
        str(answer["generation_run_id"]) for answer in answers_by_case.values()
    }
    if len(condition_values) != 1 or len(experiment_values) != 1 or len(generation_values) != 1:
        raise ValueError(f"{summary_path}: answer artifact mixes identity fields")
    condition_id = next(iter(condition_values))
    if condition_id != expected_condition:
        raise ValueError(
            f"{summary_path}: expected condition_id {expected_condition!r}, "
            f"got {condition_id!r}"
        )

    for case_id, answer in answers_by_case.items():
        summary_case = cases_by_id[case_id]
        if summary_case["answer_id"] != answer["answer_id"]:
            raise ValueError(f"{summary_path}: answer_id binding mismatch for {case_id}")

    if sha256_file(summary_path) != summary_sha_before:
        raise ValueError(f"{summary_path}: summary changed during validation")
    return ValidatedSummary(
        path=summary_path,
        artifact_sha256=summary_sha_before,
        condition_id=condition_id,
        experiment_id=next(iter(experiment_values)),
        generation_run_id=next(iter(generation_values)),
        repeat_count=repeat_count,
        order=order,
        cases_by_id=cases_by_id,
        answer_artifact={"path": str(answer_path), "sha256": answer_sha},
        judgment_artifacts=[
            {
                "path": str(path),
                "sha256": metadata["sha256"],
                "judge_run_id": metadata["judge_run_id"],
                "judge_config_sha256": metadata.get("judge_config_sha256"),
            }
            for path, metadata in zip(judgment_paths, judgment_meta)
        ],
    )


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    deltas: list[float], *, iterations: int, seed: int
) -> dict[str, Any]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one pair")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    rng = random.Random(seed)
    size = len(deltas)
    samples = [
        mean(deltas[rng.randrange(size)] for _ in range(size))
        for _ in range(iterations)
    ]
    return {
        "method": "question-level paired nonparametric bootstrap percentile CI",
        "iterations": iterations,
        "seed": seed,
        "confidence_level": 0.95,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
    }


def paired_sign_flip_test(
    vote_count_deltas: list[int],
    *,
    repeat_count: int,
    monte_carlo_iterations: int,
    seed: int,
    exact_max_nonzero: int = EXACT_SIGN_FLIP_MAX_NONZERO,
) -> dict[str, Any]:
    """Two-sided randomization test of the paired mean GFC-rate difference."""

    if not vote_count_deltas:
        raise ValueError("paired sign-flip test requires at least one pair")
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    if monte_carlo_iterations <= 0:
        raise ValueError("sign-flip iterations must be positive")
    nonzero = [value for value in vote_count_deltas if value != 0]
    observed_abs_sum = abs(sum(vote_count_deltas))

    if len(nonzero) <= exact_max_nonzero:
        extreme = 0
        total = 0
        for signs in itertools.product((-1, 1), repeat=len(nonzero)):
            permuted_abs_sum = abs(sum(sign * value for sign, value in zip(signs, nonzero)))
            extreme += permuted_abs_sum >= observed_abs_sum
            total += 1
        p_value = extreme / total
        return {
            "method": "exact paired sign-flip randomization test",
            "two_sided": True,
            "statistic": mean(vote_count_deltas) / repeat_count,
            "nonzero_pair_count": len(nonzero),
            "enumerated_assignments": total,
            "extreme_assignments": extreme,
            "p_value": p_value,
            "monte_carlo_iterations": None,
            "seed": None,
        }

    rng = random.Random(seed)
    extreme = 0
    for _ in range(monte_carlo_iterations):
        permuted_abs_sum = abs(
            sum((1 if rng.getrandbits(1) else -1) * value for value in nonzero)
        )
        extreme += permuted_abs_sum >= observed_abs_sum
    # The +1 correction prevents a zero Monte Carlo p-value and includes the
    # observed arrangement in the reference distribution.
    p_value = (extreme + 1) / (monte_carlo_iterations + 1)
    return {
        "method": "deterministic Monte Carlo paired sign-flip randomization test",
        "two_sided": True,
        "statistic": mean(vote_count_deltas) / repeat_count,
        "nonzero_pair_count": len(nonzero),
        "enumerated_assignments": None,
        "extreme_random_draws": extreme,
        "p_value": p_value,
        "monte_carlo_iterations": monte_carlo_iterations,
        "seed": seed,
        "p_value_correction": "(extreme + 1) / (iterations + 1)",
    }


def exact_mcnemar(*, c0_only: int, c1_only: int) -> dict[str, Any]:
    if c0_only < 0 or c1_only < 0:
        raise ValueError("McNemar cell counts cannot be negative")
    discordant = c0_only + c1_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(c0_only, c1_only)
        lower_tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "method": "exact two-sided McNemar test (binomial discordant-pair test)",
        "discordant_pair_count": discordant,
        "p_value": p_value,
    }


def compare_summaries(
    c0: ValidatedSummary,
    c1: ValidatedSummary,
    *,
    bootstrap_iterations: int,
    sign_flip_iterations: int,
    seed: int,
    condition_a: str = "c0",
    condition_b: str = "c1",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    condition_a = _required_nonempty_text(condition_a, location="condition_a")
    condition_b = _required_nonempty_text(condition_b, location="condition_b")
    if condition_a == condition_b:
        raise ValueError("condition A and condition B must be different")
    if c0.condition_id != condition_a or c1.condition_id != condition_b:
        raise ValueError(
            "condition binding must be "
            f"{condition_a!r} versus {condition_b!r}; got "
            f"{c0.condition_id!r} versus {c1.condition_id!r}"
        )
    if c0.experiment_id != c1.experiment_id:
        raise ValueError(
            "cross-condition experiment_id mismatch: "
            f"{c0.experiment_id!r} != {c1.experiment_id!r}"
        )
    if c0.repeat_count != c1.repeat_count:
        raise ValueError(
            "cross-condition judge repeat count mismatch: "
            f"{c0.repeat_count} != {c1.repeat_count}"
        )
    c0_ids = set(c0.cases_by_id)
    c1_ids = set(c1.cases_by_id)
    if c0_ids != c1_ids:
        raise ValueError(
            "cross-condition case_id set mismatch; "
            f"missing_in_c1={sorted(c0_ids - c1_ids) or 'none'}, "
            f"extra_in_c1={sorted(c1_ids - c0_ids) or 'none'}"
        )

    rows: list[dict[str, Any]] = []
    c0_rates: list[float] = []
    c1_rates: list[float] = []
    vote_count_deltas: list[int] = []
    wins = ties = losses = 0
    both_true = c0_only = c1_only = both_false = 0
    for case_id in c0.order:
        case0 = c0.cases_by_id[case_id]
        case1 = c1.cases_by_id[case_id]
        true0 = int(case0["gfc_true_count"])
        true1 = int(case1["gfc_true_count"])
        rate0 = true0 / c0.repeat_count
        rate1 = true1 / c1.repeat_count
        delta = rate1 - rate0
        c0_rates.append(rate0)
        c1_rates.append(rate1)
        vote_count_deltas.append(true1 - true0)
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1

        majority0 = case0["gfc_majority"]
        majority1 = case1["gfc_majority"]
        if majority0 and majority1:
            both_true += 1
            transition = "both_gfc"
        elif majority0 and not majority1:
            c0_only += 1
            transition = "c0_only"
        elif not majority0 and majority1:
            c1_only += 1
            transition = "c1_only"
        else:
            both_false += 1
            transition = "neither_gfc"

        rows.append(
            {
                "case_id": case_id,
                "c0_answer_id": case0["answer_id"],
                "c1_answer_id": case1["answer_id"],
                "judge_repeat_count": c0.repeat_count,
                "c0_grounded_fully_correct_values": case0["grounded_fully_correct_values"],
                "c1_grounded_fully_correct_values": case1["grounded_fully_correct_values"],
                "c0_gfc_rate": rate0,
                "c1_gfc_rate": rate1,
                "delta_c1_minus_c0": delta,
                "c0_gfc_majority": majority0,
                "c1_gfc_majority": majority1,
                "majority_transition": transition,
            }
        )

    deltas = [row["delta_c1_minus_c0"] for row in rows]
    bootstrap = paired_bootstrap(deltas, iterations=bootstrap_iterations, seed=seed)
    sign_flip = paired_sign_flip_test(
        vote_count_deltas,
        repeat_count=c0.repeat_count,
        monte_carlo_iterations=sign_flip_iterations,
        seed=seed,
    )
    mcnemar = exact_mcnemar(c0_only=c0_only, c1_only=c1_only)
    question_count = len(rows)
    sensitivity_c0_true = both_true + c0_only
    sensitivity_c1_true = both_true + c1_only

    payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "inputs": {
            "c0": {
                "summary": {"path": str(c0.path), "sha256": c0.artifact_sha256},
                "condition_id": c0.condition_id,
                "experiment_id": c0.experiment_id,
                "generation_run_id": c0.generation_run_id,
                "answers": c0.answer_artifact,
                "judgments": c0.judgment_artifacts,
            },
            "c1": {
                "summary": {"path": str(c1.path), "sha256": c1.artifact_sha256},
                "condition_id": c1.condition_id,
                "experiment_id": c1.experiment_id,
                "generation_run_id": c1.generation_run_id,
                "answers": c1.answer_artifact,
                "judgments": c1.judgment_artifacts,
            },
        },
        "integrity_verification": {
            "referenced_answer_artifacts_rehashed": True,
            "referenced_judgment_artifacts_rehashed": True,
            "answer_and_judgment_record_hashes_revalidated": True,
            "case_answer_condition_bindings_revalidated": True,
            "summaries_rebuilt_from_source_artifacts": True,
            "cross_condition_pair_key": "case_id",
            "cross_condition_answer_id_equality_required": False,
        },
        "condition_binding": {
            "a": condition_a,
            "b": condition_b,
            "legacy_v1_field_aliases": {"c0": "a", "c1": "b"},
        },
        "methodology": {
            "sample_unit": "paired question",
            "effective_sample_n": question_count,
            "judge_repeat_count_per_condition": c0.repeat_count,
            "judge_repeats_count_as_additional_samples": False,
            "primary_estimand": (
                "mean over questions of (condition B GFC repeat-success "
                "proportion - condition A GFC repeat-success proportion)"
            ),
            "uncertainty": {
                "paired_bootstrap": bootstrap,
                "paired_sign_flip": sign_flip,
            },
            "sensitivity_estimand": "strict-majority GFC per question (at least 2/3 when repeats=3)",
            "sensitivity_test": "exact two-sided McNemar",
        },
        "primary": {
            "question_count": question_count,
            "effective_sample_n": question_count,
            "c0_mean_gfc_rate": mean(c0_rates),
            "c1_mean_gfc_rate": mean(c1_rates),
            "mean_delta_c1_minus_c0": mean(deltas),
            "c1_higher_count": wins,
            "equal_count": ties,
            "c1_lower_count": losses,
            "paired_bootstrap_95ci": bootstrap,
            "paired_sign_flip_two_sided": sign_flip,
        },
        "sensitivity": {
            "definition": "strict majority of repeated GFC judgments",
            "c0_majority_gfc_true_count": sensitivity_c0_true,
            "c0_majority_gfc_rate": sensitivity_c0_true / question_count,
            "c1_majority_gfc_true_count": sensitivity_c1_true,
            "c1_majority_gfc_rate": sensitivity_c1_true / question_count,
            "rate_delta_c1_minus_c0": (sensitivity_c1_true - sensitivity_c0_true) / question_count,
            "paired_2x2": {
                "both_gfc": both_true,
                "c0_only": c0_only,
                "c1_only": c1_only,
                "neither_gfc": both_false,
            },
            "exact_mcnemar_two_sided": mcnemar,
        },
        "cases": rows,
    }
    return payload, rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write empty paired case CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        for row in materialized:
            serialized = dict(row)
            for field in (
                "c0_grounded_fully_correct_values",
                "c1_grounded_fully_correct_values",
            ):
                serialized[field] = json.dumps(serialized[field], separators=(",", ":"))
            writer.writerow(serialized)


def _reject_input_output_aliases(
    *, c0_summary: Path, c1_summary: Path, json_out: Path, csv_out: Path
) -> None:
    inputs = {c0_summary.resolve(), c1_summary.resolve()}
    outputs = [json_out.resolve(), csv_out.resolve()]
    if outputs[0] == outputs[1]:
        raise ValueError("--json-out and --csv-out must be different files")
    if any(output in inputs for output in outputs):
        raise ValueError("output paths must not overwrite input summaries")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-summary", type=Path, required=True)
    parser.add_argument("--c1-summary", type=Path, required=True)
    parser.add_argument(
        "--condition-a",
        default="c0",
        help="expected condition_id for --c0-summary (default: c0)",
    )
    parser.add_argument(
        "--condition-b",
        default="c1",
        help="expected condition_id for --c1-summary (default: c1)",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument(
        "--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS
    )
    parser.add_argument(
        "--sign-flip-iterations", type=int, default=DEFAULT_SIGN_FLIP_ITERATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap <= 0:
        raise SystemExit("--bootstrap must be positive")
    if args.sign_flip_iterations <= 0:
        raise SystemExit("--sign-flip-iterations must be positive")
    try:
        _reject_input_output_aliases(
            c0_summary=args.c0_summary,
            c1_summary=args.c1_summary,
            json_out=args.json_out,
            csv_out=args.csv_out,
        )
        c0 = validate_repeat_summary(
            args.c0_summary, expected_condition=args.condition_a
        )
        c1 = validate_repeat_summary(
            args.c1_summary, expected_condition=args.condition_b
        )
        payload, rows = compare_summaries(
            c0,
            c1,
            bootstrap_iterations=args.bootstrap,
            sign_flip_iterations=args.sign_flip_iterations,
            seed=args.seed,
            condition_a=args.condition_a,
            condition_b=args.condition_b,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.csv_out, rows)

    primary = payload["primary"]
    sensitivity = payload["sensitivity"]
    ci_low, ci_high = primary["paired_bootstrap_95ci"]["ci95"]
    print(
        f"questions={primary['question_count']} "
        f"judge_repeats={payload['methodology']['judge_repeat_count_per_condition']} "
        f"effective_sample_n={primary['effective_sample_n']}"
    )
    print(
        f"{args.condition_a}={primary['c0_mean_gfc_rate']:.4f} "
        f"{args.condition_b}={primary['c1_mean_gfc_rate']:.4f} "
        f"delta={primary['mean_delta_c1_minus_c0']:+.4f} "
        f"paired-bootstrap-95CI=[{ci_low:+.4f},{ci_high:+.4f}] "
        f"sign-flip-p={primary['paired_sign_flip_two_sided']['p_value']:.6g}"
    )
    table = sensitivity["paired_2x2"]
    print(
        f"majority both/c0-only/c1-only/neither="
        f"{table['both_gfc']}/{table['c0_only']}/{table['c1_only']}/"
        f"{table['neither_gfc']} "
        f"McNemar-p={sensitivity['exact_mcnemar_two_sided']['p_value']:.6g}"
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
