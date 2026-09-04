#!/usr/bin/env python3
"""Analyze final C0/C1 GFC across three independent generation runs.

Each question remains one statistical sample.  The three generation outputs
are within-question repeats; Judge retries/rejudgments are never additional
samples.  The LLM-Judge result is headline-eligible only when the supplied
Judge-human calibration artifact passes the frozen protocol gate.  On gate
failure, explicit human-label input is required and adjudicated run-1 C0/C1
GFC becomes the headline fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_judge_human_calibration import (  # noqa: E402
    GATE_THRESHOLDS,
    HUMAN_LABEL_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION as CALIBRATION_SCHEMA_VERSION,
    _judge_human_metrics,
    analyze_calibration,
    load_human_labels,
)
from final_generation_slots import (  # noqa: E402
    answer_is_judge_eligible as validate_slot_disposition,
)
from immutable_outputs import (  # noqa: E402
    publish_immutable_texts,
    reject_symlink_inputs,
    require_new_outputs,
)
from service_eval_artifacts import (  # noqa: E402
    load_unique_jsonl,
    sha256_json,
    validate_answer_record,
    validate_final_generation_provenance,
    validate_judgment_record,
)


OUTPUT_SCHEMA_VERSION = "pnu.final-generation-gfc-analysis.v1"
RUN_IDS = ("run1", "run2", "run3")
CONDITION_IDS = ("c0", "c1")
CORE_SPLIT = "holdout-core"
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_SIGN_FLIP_ITERATIONS = 100_000
DEFAULT_SEED = 20260914
EXACT_SIGN_FLIP_MAX_NONZERO = 20


@dataclass(frozen=True)
class GenerationRun:
    condition_id: str
    generation_run_id: str
    experiment_id: str
    answer_path: Path
    answer_artifact_sha256: str
    judgment_path: Path
    judgment_artifact_sha256: str
    judge_run_id: str | None
    judge_config_sha256: str | None
    order: list[str]
    answers: dict[str, dict[str, Any]]
    judgments: dict[str, dict[str, Any]]


def _terminal_service_error(
    record: Mapping[str, Any], *, path: Path
) -> bool:
    return not validate_slot_disposition(
        record,
        source=path,
        allow_legacy=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a non-empty string")
    return value.strip()


def _valid_sha256(value: Any, label: str) -> str:
    digest = _required_text(value, label)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label}: expected lowercase SHA-256")
    return digest


def _load_json_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = sha256_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must be a JSON object")
    if sha256_file(path) != before:
        raise ValueError(f"{path}: artifact changed during validation")
    return value, before


def load_core_cases(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    artifact_sha_before = sha256_file(path)
    records = load_unique_jsonl(path, key="id")
    if not records:
        raise ValueError(f"{path}: cases artifact is empty")
    core = [record for record in records if record.get("split") == CORE_SPLIT]
    if not core:
        raise ValueError(f"{path}: no {CORE_SPLIT} cases")
    order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for record in core:
        case_id = _required_text(record.get("id"), f"{path}: case id")
        family_id = _required_text(
            record.get("family_id"), f"{path}: {case_id}.family_id"
        )
        if case_id in by_id:
            raise ValueError(f"{path}: duplicate Core case_id {case_id}")
        if not family_id:
            raise AssertionError("unreachable")
        order.append(case_id)
        by_id[case_id] = record
    if sha256_file(path) != artifact_sha_before:
        raise ValueError(f"{path}: cases artifact changed during validation")
    return order, by_id, {
        "path": str(path.resolve()),
        "sha256": artifact_sha_before,
        "canonical_sha256": sha256_json(records),
        "core_case_count": len(order),
        "core_case_ids_sha256": sha256_json(order),
        "core_family_count": len({str(row["family_id"]) for row in core}),
    }


def _load_answer_artifact(
    path: Path,
    *,
    expected_condition: str,
    core_order: list[str],
    core_cases: Mapping[str, Mapping[str, Any]],
    cases_artifact_sha256: str,
    cases_canonical_sha256: str,
) -> tuple[str, str, str, dict[str, dict[str, Any]], str]:
    artifact_sha_before = sha256_file(path)
    records = load_unique_jsonl(path, key="answer_id")
    if not records:
        raise ValueError(f"{path}: answer artifact is empty")
    observed_order: list[str] = []
    by_case: dict[str, dict[str, Any]] = {}
    collector_config_hashes: set[str] = set()
    selected_hash = sha256_json(core_order)
    for record in records:
        validate_answer_record(record)
        case_id = str(record["case_id"])
        if record.get("error") is not None:
            raise ValueError(f"{path}: answer {case_id} has error: {record['error']}")
        if not str(record.get("answer") or "").strip():
            raise ValueError(f"{path}: answer {case_id} is empty")
        _terminal_service_error(record, path=path)
        if record.get("condition_id") != expected_condition:
            raise ValueError(f"{path}: condition_id must be {expected_condition}")
        case = core_cases.get(case_id)
        if case is None:
            raise ValueError(f"{path}: non-Core or unknown case_id {case_id}")
        if record.get("case_sha256") != sha256_json(case):
            raise ValueError(f"{path}: case_sha256 mismatch for {case_id}")
        collector = record.get("collector_config")
        if not isinstance(collector, dict):
            raise ValueError(f"{path}: missing collector_config for {case_id}")
        if record.get("collector_config_sha256") != sha256_json(collector):
            raise ValueError(f"{path}: collector_config_sha256 mismatch for {case_id}")
        collector_config_hashes.add(str(record["collector_config_sha256"]))
        generation_provider = _required_text(
            collector.get("provider"), f"{path}: collector provider"
        )
        generation_model = _required_text(
            collector.get("model"), f"{path}: collector model"
        )
        max_output_tokens = collector.get(
            "expected_generation_max_output_tokens"
        )
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError(
                f"{path}: invalid expected_generation_max_output_tokens"
            )
        try:
            validate_final_generation_provenance(
                record,
                provider=generation_provider,
                model=generation_model,
                max_output_tokens=max_output_tokens,
                require_collection_attempts=True,
            )
        except ValueError as exc:
            raise ValueError(
                f"{path}: invalid generation provenance for {case_id}: {exc}"
            ) from exc
        if collector.get("cases_sha256") != cases_artifact_sha256:
            raise ValueError(
                f"{path}: raw cases artifact SHA-256 binding mismatch for {case_id}"
            )
        if collector.get("cases_canonical_sha256") != cases_canonical_sha256:
            raise ValueError(
                f"{path}: canonical cases SHA-256 binding mismatch for {case_id}"
            )
        if collector.get("selected_case_ids_sha256") != selected_hash:
            raise ValueError(
                f"{path}: selected Core case IDs hash mismatch for {case_id}"
            )
        if case_id in by_case:
            raise ValueError(f"{path}: duplicate case_id {case_id}")
        observed_order.append(case_id)
        by_case[case_id] = record
    if observed_order != core_order:
        raise ValueError(
            f"{path}: answer cases must exactly cover Core in frozen order"
        )
    if len(collector_config_hashes) != 1:
        raise ValueError(f"{path}: answer artifact mixes collector configs")
    experiments = {str(record["experiment_id"]) for record in records}
    runs = {str(record["generation_run_id"]) for record in records}
    if len(experiments) != 1 or len(runs) != 1:
        raise ValueError(f"{path}: answer artifact mixes experiment or generation run")
    run_id = next(iter(runs))
    if run_id not in RUN_IDS:
        raise ValueError(f"{path}: generation_run_id must be one of {RUN_IDS}")
    if sha256_file(path) != artifact_sha_before:
        raise ValueError(f"{path}: answer artifact changed during validation")
    return (
        next(iter(experiments)),
        run_id,
        artifact_sha_before,
        by_case,
        str(records[0]["condition_id"]),
    )


def _attach_judgment(
    answer_data: tuple[str, str, str, dict[str, dict[str, Any]], str],
    *,
    answer_path: Path,
    judgment_path: Path,
    core_order: list[str],
) -> GenerationRun:
    experiment_id, run_id, answer_sha, answers, condition_id = answer_data
    judgment_sha_before = sha256_file(judgment_path)
    records = load_unique_jsonl(judgment_path, key="judgment_id")
    terminal_case_ids = {
        case_id
        for case_id, answer in answers.items()
        if _terminal_service_error(answer, path=answer_path)
    }
    successful_order = [
        case_id for case_id in core_order if case_id not in terminal_case_ids
    ]
    observed_judgment_order = [
        str(record.get("case_id") or "") for record in records
    ]
    judged_terminal_cases = sorted(
        set(observed_judgment_order).intersection(terminal_case_ids)
    )
    if judged_terminal_cases:
        raise ValueError(
            f"{judgment_path}: terminal service-error slot(s) were judged: "
            f"{judged_terminal_cases}"
        )
    if observed_judgment_order != successful_order:
        raise ValueError(
            f"{judgment_path}: judgments must exactly cover successful Core "
            "slots in answer order; terminal slots must not be judged"
        )
    by_answer: dict[str, dict[str, Any]] = {}
    all_answer_ids = {
        str(answer["answer_id"]): answer for answer in answers.values()
    }
    eligible_answer_ids = {
        str(answers[case_id]["answer_id"]): answers[case_id]
        for case_id in successful_order
    }
    judge_run_ids: set[str] = set()
    config_hashes: set[str] = set()
    for record in records:
        validate_judgment_record(record)
        case_id = str(record["case_id"])
        answer_id = str(record["answer_id"])
        answer = all_answer_ids.get(answer_id)
        if answer is None:
            raise ValueError(
                f"{judgment_path}: judgment references unknown answer_id {answer_id}"
            )
        if answer_id not in eligible_answer_ids:
            raise ValueError(
                f"{judgment_path}: terminal service-error slot was judged: {case_id}"
            )
        for field, expected in (
            ("experiment_id", experiment_id),
            ("condition_id", condition_id),
            ("generation_run_id", run_id),
            ("case_id", answer["case_id"]),
            ("answer_sha256", answer["answer_sha256"]),
            ("answer_record_sha256", sha256_json(answer)),
            ("answers_artifact_sha256", answer_sha),
        ):
            if record.get(field) != expected:
                raise ValueError(
                    f"{judgment_path}: {field} mismatch for answer_id {answer_id}"
                )
        if record.get("error") is not None:
            raise ValueError(
                f"{judgment_path}: judgment {case_id} has error: {record['error']}"
            )
        if record.get("judge_repeat_selection") is not None:
            raise ValueError(
                f"{judgment_path}: stability-repeat judgment cannot be used "
                f"as the definitive judgment for {case_id}"
            )
        judge_config = record.get("judge_config")
        if not isinstance(judge_config, dict) or not judge_config:
            raise ValueError(
                f"{judgment_path}: missing or invalid judge_config for {case_id}"
            )
        judge_config_identity = dict(judge_config)
        judge_config_identity.pop("judge_config_sha256", None)
        if record.get("judge_config_sha256") != sha256_json(
            judge_config_identity
        ):
            raise ValueError(
                f"{judgment_path}: missing or invalid judge_config for {case_id}"
            )
        judge = record.get("judge")
        if not isinstance(judge, dict):
            raise ValueError(f"{judgment_path}: missing judge object for {case_id}")
        score = judge.get("score")
        gfc = judge.get("grounded_fully_correct")
        if type(score) is not int or score not in (0, 1, 2):
            raise ValueError(f"{judgment_path}: invalid score for {case_id}")
        if type(gfc) is not bool:
            raise ValueError(f"{judgment_path}: invalid GFC for {case_id}")
        if gfc and score != 2:
            raise ValueError(f"{judgment_path}: GFC=true requires score=2 for {case_id}")
        if answer_id in by_answer:
            raise ValueError(f"{judgment_path}: multiple judgments for {answer_id}")
        by_answer[answer_id] = record
        judge_run_ids.add(_required_text(record.get("judge_run_id"), "judge_run_id"))
        config_hashes.add(
            _valid_sha256(record.get("judge_config_sha256"), "judge_config_sha256")
        )
    missing = sorted(set(eligible_answer_ids) - set(by_answer))
    if missing:
        raise ValueError(f"{judgment_path}: missing judgments for {missing}")
    if records and (len(judge_run_ids) != 1 or len(config_hashes) != 1):
        raise ValueError(f"{judgment_path}: mixes Judge run or config")
    if not records and successful_order:
        raise ValueError(f"{judgment_path}: successful slots have no judgments")
    if sha256_file(judgment_path) != judgment_sha_before:
        raise ValueError(f"{judgment_path}: judgment artifact changed during validation")
    return GenerationRun(
        condition_id=condition_id,
        generation_run_id=run_id,
        experiment_id=experiment_id,
        answer_path=answer_path.resolve(),
        answer_artifact_sha256=answer_sha,
        judgment_path=judgment_path.resolve(),
        judgment_artifact_sha256=judgment_sha_before,
        judge_run_id=next(iter(judge_run_ids)) if judge_run_ids else None,
        judge_config_sha256=next(iter(config_hashes)) if config_hashes else None,
        order=list(core_order),
        answers=answers,
        judgments=by_answer,
    )


def load_condition_runs(
    *,
    condition_id: str,
    answer_paths: Sequence[Path],
    judgment_paths: Sequence[Path],
    core_order: list[str],
    core_cases: Mapping[str, Mapping[str, Any]],
    cases_artifact_sha256: str,
    cases_canonical_sha256: str,
) -> dict[str, GenerationRun]:
    if len(answer_paths) != 3 or len(judgment_paths) != 3:
        raise ValueError(f"{condition_id}: exactly three answer and judgment files required")
    answers_by_run: dict[
        str,
        tuple[Path, tuple[str, str, str, dict[str, dict[str, Any]], str]],
    ] = {}
    for expected_run_id, path in zip(RUN_IDS, answer_paths):
        data = _load_answer_artifact(
            path,
            expected_condition=condition_id,
            core_order=core_order,
            core_cases=core_cases,
            cases_artifact_sha256=cases_artifact_sha256,
            cases_canonical_sha256=cases_canonical_sha256,
        )
        run_id = data[1]
        if run_id != expected_run_id:
            raise ValueError(
                f"{condition_id}: answer inputs must be ordered run1, run2, run3; "
                f"position {expected_run_id} contains {run_id}"
            )
        if run_id in answers_by_run:
            raise ValueError(f"{condition_id}: duplicate generation run {run_id}")
        answers_by_run[run_id] = (path, data)
    if set(answers_by_run) != set(RUN_IDS):
        raise ValueError(
            f"{condition_id}: generation run set must be {list(RUN_IDS)}"
        )

    return {
        run_id: _attach_judgment(
            answers_by_run[run_id][1],
            answer_path=answers_by_run[run_id][0],
            judgment_path=judgment_path,
            core_order=core_order,
        )
        for run_id, judgment_path in zip(RUN_IDS, judgment_paths)
    }


def _resolve_reference(raw_path: Any, *, owner: Path, expected_sha: str) -> Path:
    value = Path(_required_text(raw_path, f"{owner}: referenced path")).expanduser()
    candidates = (
        [value]
        if value.is_absolute()
        else [REPO_ROOT / value, owner.parent / value, Path.cwd() / value]
    )
    resolved: list[Path] = []
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized not in resolved and normalized.is_file():
            resolved.append(normalized)
    matches = [path for path in resolved if sha256_file(path) == expected_sha]
    if not matches:
        raise ValueError(f"{owner}: referenced artifact missing or SHA-256 mismatch: {value}")
    return matches[0]


def _validate_calibration(
    path: Path,
    *,
    run1_by_condition: Mapping[str, GenerationRun],
    core_order: Sequence[str],
) -> tuple[bool, dict[str, Any], dict[str, dict[str, Any]], list[dict[str, str]]]:
    payload, artifact_sha = _load_json_object(path, "calibration artifact")
    if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(f"{path}: invalid calibration schema_version")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{path}: calibration inputs are missing")
    resolved_inputs: dict[str, list[dict[str, str]]] = {}
    for key in ("judgments", "human_labels"):
        entries = inputs.get(key)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{path}: calibration inputs.{key} must be non-empty")
        seen_paths: set[Path] = set()
        resolved_entries: list[dict[str, str]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: inputs.{key}[{index}] must be an object")
            expected_sha = _valid_sha256(
                entry.get("sha256"), f"{path}: inputs.{key}[{index}].sha256"
            )
            resolved = _resolve_reference(
                entry.get("path"), owner=path.resolve(), expected_sha=expected_sha
            )
            if resolved in seen_paths:
                raise ValueError(f"{path}: duplicate referenced {key} path")
            seen_paths.add(resolved)
            resolved_entries.append({"path": str(resolved), "sha256": expected_sha})
        resolved_inputs[key] = resolved_entries

    rebuilt, _ = analyze_calibration(
        judgment_paths=[Path(entry["path"]) for entry in resolved_inputs["judgments"]],
        human_label_paths=[Path(entry["path"]) for entry in resolved_inputs["human_labels"]],
    )
    for field in ("human_label_schema", "methodology", "summary", "cases"):
        if payload.get(field) != rebuilt.get(field):
            raise ValueError(
                f"{path}: calibration {field} differs from referenced source artifacts"
            )

    required_judgment_shas = {
        run1_by_condition[condition].judgment_artifact_sha256
        for condition in CONDITION_IDS
        if run1_by_condition[condition].judgments
    }
    calibration_judgment_shas = {
        entry["sha256"] for entry in resolved_inputs["judgments"]
    }
    if not required_judgment_shas.issubset(calibration_judgment_shas):
        raise ValueError(
            f"{path}: calibration is not bound to every nonempty definitive "
            "run1 judgment artifact"
        )

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{path}: calibration cases must be non-empty")
    if any(not isinstance(row, dict) for row in raw_cases):
        raise ValueError(f"{path}: every calibration case must be an object")
    all_answer_ids = {
        str(answer["answer_id"])
        for run in run1_by_condition.values()
        for answer in run.answers.values()
        if not _terminal_service_error(answer, path=run.answer_path)
    }
    if not all_answer_ids:
        raise ValueError(
            f"{path}: calibration cannot be evaluated because run1 has no "
            "successful answers"
        )
    core_rows = [row for row in raw_cases if row.get("split") == CORE_SPLIT]
    rows_by_answer: dict[str, dict[str, Any]] = {}
    blind_ids: set[str] = set()
    for row in core_rows:
        answer_id = _required_text(row.get("answer_id"), f"{path}: calibration answer_id")
        if answer_id in rows_by_answer:
            raise ValueError(f"{path}: duplicate calibration answer_id {answer_id}")
        blind_id = _required_text(row.get("blind_item_id"), f"{path}: blind_item_id")
        if blind_id in blind_ids:
            raise ValueError(f"{path}: duplicate calibration blind_item_id {blind_id}")
        blind_ids.add(blind_id)
        rows_by_answer[answer_id] = row
    if set(rows_by_answer) != all_answer_ids:
        raise ValueError(
            f"{path}: calibration Core answer set is not exactly C0/C1 run1"
        )

    for condition in CONDITION_IDS:
        run = run1_by_condition[condition]
        for case_id in core_order:
            answer = run.answers[case_id]
            if _terminal_service_error(answer, path=run.answer_path):
                continue
            answer_id = str(answer["answer_id"])
            judgment = run.judgments[answer_id]
            row = rows_by_answer[answer_id]
            if row.get("case_id") != case_id:
                raise ValueError(f"{path}: calibration case_id mismatch for {answer_id}")
            recorded_judge = row.get("judge")
            if not isinstance(recorded_judge, dict) or recorded_judge != {
                "judgment_id": judgment["judgment_id"],
                "score": judgment["judge"]["score"],
                "grounded_fully_correct": judgment["judge"]["grounded_fully_correct"],
            }:
                raise ValueError(f"{path}: calibration Judge binding mismatch for {answer_id}")
            adjudicated = row.get("adjudicated")
            if not isinstance(adjudicated, dict):
                raise ValueError(f"{path}: calibration adjudication missing for {answer_id}")
            score = adjudicated.get("score")
            gfc = adjudicated.get("grounded_fully_correct")
            if type(score) is not int or score not in (0, 1, 2) or type(gfc) is not bool:
                raise ValueError(f"{path}: invalid adjudication for {answer_id}")
            if gfc and score != 2:
                raise ValueError(f"{path}: adjudicated GFC=true requires score=2")

    recomputed = _judge_human_metrics(list(rows_by_answer.values()))
    expected_observed = {
        "raw_agreement": recomputed["raw_agreement"],
        "balanced_accuracy": recomputed["balanced_accuracy"],
        "cohen_kappa": recomputed["cohen_kappa"],
        "macro_f1": recomputed["macro_f1"],
    }
    expected_checks = {
        key: bool(expected_observed[key] is not None and expected_observed[key] >= threshold)
        for key, threshold in GATE_THRESHOLDS.items()
    }
    expected_passed = all(expected_checks.values())
    summary = payload.get("summary")
    gate = summary.get("protocol_gate") if isinstance(summary, dict) else None
    if not isinstance(gate, dict):
        raise ValueError(f"{path}: calibration protocol_gate is missing")
    if (
        gate.get("scope") != "core_split_only"
        or gate.get("answer_count") != len(all_answer_ids)
        or gate.get("eligible") is not True
        or gate.get("thresholds") != GATE_THRESHOLDS
        or gate.get("observed") != expected_observed
        or gate.get("checks") != expected_checks
        or gate.get("passed") is not expected_passed
        or gate.get("undefined_kappa_passes") is not False
    ):
        raise ValueError(f"{path}: calibration protocol gate is internally inconsistent")
    return expected_passed, {
        "path": str(path.resolve()),
        "sha256": artifact_sha,
        "protocol_gate": gate,
        "referenced_inputs_rehashed": True,
        "resolved_inputs": resolved_inputs,
    }, rows_by_answer, resolved_inputs["human_labels"]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def family_cluster_bootstrap(
    deltas: Mapping[str, float],
    families: Mapping[str, str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0 or not deltas:
        raise ValueError("family bootstrap requires positive iterations and pairs")
    clusters: dict[str, list[str]] = defaultdict(list)
    for case_id in deltas:
        clusters[_required_text(families.get(case_id), f"{case_id}: family_id")].append(case_id)
    family_ids = sorted(clusters)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_cases: list[str] = []
        for _slot in family_ids:
            sampled_cases.extend(clusters[rng.choice(family_ids)])
        samples.append(mean(deltas[case_id] for case_id in sampled_cases))
    return {
        "method": "paired cluster bootstrap by frozen family_id",
        "unit": "question family",
        "clusters": len(family_ids),
        "questions": len(deltas),
        "iterations": iterations,
        "seed": seed,
        "confidence_level": 0.95,
        "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
    }


def paired_sign_flip(
    deltas: Sequence[float], *, iterations: int, seed: int
) -> dict[str, Any]:
    if iterations <= 0 or not deltas:
        raise ValueError("sign-flip requires positive iterations and pairs")
    nonzero = [float(value) for value in deltas if value != 0]
    observed = abs(sum(nonzero))
    if len(nonzero) <= EXACT_SIGN_FLIP_MAX_NONZERO:
        extreme = 0
        total = 0
        for signs in itertools.product((-1, 1), repeat=len(nonzero)):
            extreme += abs(sum(sign * value for sign, value in zip(signs, nonzero))) >= observed
            total += 1
        return {
            "method": "exact paired sign-flip randomization test",
            "two_sided": True,
            "nonzero_pair_count": len(nonzero),
            "enumerated_assignments": total,
            "monte_carlo_iterations": None,
            "seed": None,
            "p_value": extreme / total,
        }
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = abs(sum((1 if rng.getrandbits(1) else -1) * value for value in nonzero))
        extreme += permuted >= observed
    return {
        "method": "deterministic Monte Carlo paired sign-flip randomization test",
        "two_sided": True,
        "nonzero_pair_count": len(nonzero),
        "enumerated_assignments": None,
        "monte_carlo_iterations": iterations,
        "seed": seed,
        "p_value": (extreme + 1) / (iterations + 1),
        "p_value_correction": "(extreme + 1) / (iterations + 1)",
    }


def exact_mcnemar(c0: Sequence[bool], c1: Sequence[bool]) -> dict[str, Any]:
    if len(c0) != len(c1) or not c0:
        raise ValueError("McNemar requires non-empty paired values")
    both = sum(left and right for left, right in zip(c0, c1))
    c0_only = sum(left and not right for left, right in zip(c0, c1))
    c1_only = sum((not left) and right for left, right in zip(c0, c1))
    neither = len(c0) - both - c0_only - c1_only
    discordant = c0_only + c1_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(
            math.comb(discordant, k)
            for k in range(min(c0_only, c1_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "method": "exact two-sided McNemar (binomial discordant-pair test)",
        "both_gfc": both,
        "c0_only": c0_only,
        "c1_only": c1_only,
        "neither_gfc": neither,
        "discordant_pair_count": discordant,
        "p_value": p_value,
    }


def _paired_statistics(
    *,
    order: Sequence[str],
    c0_values: Mapping[str, float],
    c1_values: Mapping[str, float],
    families: Mapping[str, str],
    c0_binary: Mapping[str, bool],
    c1_binary: Mapping[str, bool],
    bootstrap_iterations: int,
    sign_flip_iterations: int,
    seed: int,
) -> dict[str, Any]:
    deltas = {case_id: c1_values[case_id] - c0_values[case_id] for case_id in order}
    return {
        "question_count": len(order),
        "effective_sample_n": len(order),
        "family_cluster_count": len({families[case_id] for case_id in order}),
        "c0_mean_gfc_rate": mean(c0_values[case_id] for case_id in order),
        "c1_mean_gfc_rate": mean(c1_values[case_id] for case_id in order),
        "mean_delta_c1_minus_c0": mean(deltas.values()),
        "gained_case_ids": [case_id for case_id in order if deltas[case_id] > 0],
        "lost_case_ids": [case_id for case_id in order if deltas[case_id] < 0],
        "tied_case_ids": [case_id for case_id in order if deltas[case_id] == 0],
        "paired_family_bootstrap_95ci": family_cluster_bootstrap(
            deltas, families, iterations=bootstrap_iterations, seed=seed
        ),
        "paired_sign_flip_two_sided": paired_sign_flip(
            list(deltas.values()), iterations=sign_flip_iterations, seed=seed
        ),
        "majority_sensitivity": {
            "definition": "at least two GFC successes across three generation runs",
            "c0_success_count": sum(c0_binary.values()),
            "c1_success_count": sum(c1_binary.values()),
            "exact_mcnemar_two_sided": exact_mcnemar(
                [c0_binary[case_id] for case_id in order],
                [c1_binary[case_id] for case_id in order],
            ),
        },
    }


def _slot_observation(run: GenerationRun, case_id: str) -> dict[str, Any]:
    answer = run.answers[case_id]
    answer_id = str(answer["answer_id"])
    service_error = _terminal_service_error(answer, path=run.answer_path)
    judgment = run.judgments.get(answer_id)
    if service_error:
        if judgment is not None:
            raise ValueError(f"terminal service-error slot was judged: {answer_id}")
        return {
            "answer_id": answer_id,
            "judgment_id": None,
            "judge_score": None,
            "gfc": False,
            "service_error": True,
        }
    if judgment is None:
        raise ValueError(f"successful answer has no definitive judgment: {answer_id}")
    return {
        "answer_id": answer_id,
        "judgment_id": str(judgment["judgment_id"]),
        "judge_score": int(judgment["judge"]["score"]),
        "gfc": bool(judgment["judge"]["grounded_fully_correct"]),
        "service_error": False,
    }


def build_judge_analysis(
    runs: Mapping[str, Mapping[str, GenerationRun]],
    *,
    order: Sequence[str],
    families: Mapping[str, str],
    bootstrap_iterations: int,
    sign_flip_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_run: dict[str, Any] = {}
    values: dict[str, dict[str, float]] = {condition: {} for condition in CONDITION_IDS}
    binary: dict[str, dict[str, bool]] = {condition: {} for condition in CONDITION_IDS}
    for run_id in RUN_IDS:
        run_values: dict[str, dict[str, list[Any]]] = {
            condition: {"scores": [], "gfc": [], "service_errors": []}
            for condition in CONDITION_IDS
        }
        gained: list[str] = []
        lost: list[str] = []
        tied: list[str] = []
        for case_id in order:
            observed: dict[str, bool] = {}
            for condition in CONDITION_IDS:
                run = runs[condition][run_id]
                slot = _slot_observation(run, case_id)
                if slot["judge_score"] is not None:
                    run_values[condition]["scores"].append(slot["judge_score"])
                run_values[condition]["gfc"].append(slot["gfc"])
                run_values[condition]["service_errors"].append(
                    slot["service_error"]
                )
                observed[condition] = bool(slot["gfc"])
            if observed["c1"] and not observed["c0"]:
                gained.append(case_id)
            elif observed["c0"] and not observed["c1"]:
                lost.append(case_id)
            else:
                tied.append(case_id)
        by_run[run_id] = {
            condition: {
                "logical_slot_count": len(order),
                "successful_answer_count": len(
                    run_values[condition]["scores"]
                ),
                "judgment_count": len(run_values[condition]["scores"]),
                "conditional_judged_score_mean": (
                    mean(run_values[condition]["scores"])
                    if run_values[condition]["scores"]
                    else None
                ),
                "judged_score_sum": sum(run_values[condition]["scores"]),
                "gfc_count": sum(run_values[condition]["gfc"]),
                "gfc_rate": sum(run_values[condition]["gfc"]) / len(order),
                "service_error_count": sum(
                    run_values[condition]["service_errors"]
                ),
                "service_error_rate": sum(
                    run_values[condition]["service_errors"]
                )
                / len(order),
                "service_error_case_ids": [
                    case_id
                    for case_id, is_error in zip(
                        order, run_values[condition]["service_errors"]
                    )
                    if is_error
                ],
            }
            for condition in CONDITION_IDS
        }
        c0_conditional_score = by_run[run_id]["c0"][
            "conditional_judged_score_mean"
        ]
        c1_conditional_score = by_run[run_id]["c1"][
            "conditional_judged_score_mean"
        ]
        by_run[run_id]["paired"] = {
            "conditional_judged_mean_score_delta_c1_minus_c0": (
                c1_conditional_score - c0_conditional_score
                if c0_conditional_score is not None
                and c1_conditional_score is not None
                else None
            ),
            "gfc_rate_delta_c1_minus_c0": (
                by_run[run_id]["c1"]["gfc_rate"]
                - by_run[run_id]["c0"]["gfc_rate"]
            ),
            "gained_case_ids": gained,
            "lost_case_ids": lost,
            "tied_case_ids": tied,
        }

    for case_id in order:
        condition_data: dict[str, Any] = {}
        for condition in CONDITION_IDS:
            answer_ids: list[str] = []
            judgment_ids: list[str | None] = []
            service_errors: list[bool] = []
            scores: list[int | None] = []
            gfc_values: list[bool] = []
            for run_id in RUN_IDS:
                run = runs[condition][run_id]
                slot = _slot_observation(run, case_id)
                answer_ids.append(slot["answer_id"])
                judgment_ids.append(slot["judgment_id"])
                scores.append(slot["judge_score"])
                gfc_values.append(slot["gfc"])
                service_errors.append(slot["service_error"])
            rate = sum(gfc_values) / len(RUN_IDS)
            values[condition][case_id] = rate
            binary[condition][case_id] = sum(gfc_values) >= 2
            condition_data[condition] = {
                "answer_ids_by_run": dict(zip(RUN_IDS, answer_ids)),
                "judgment_ids_by_run": dict(zip(RUN_IDS, judgment_ids)),
                "judge_scores_by_run": dict(zip(RUN_IDS, scores)),
                "gfc_by_run": dict(zip(RUN_IDS, gfc_values)),
                "service_errors_by_run": dict(zip(RUN_IDS, service_errors)),
                "conditional_judged_score_mean": (
                    mean(score for score in scores if score is not None)
                    if any(score is not None for score in scores)
                    else None
                ),
                "gfc_success_count": sum(gfc_values),
                "gfc_success_proportion": rate,
                "majority_gfc": sum(gfc_values) >= 2,
                "service_error_count": sum(service_errors),
            }
        delta = values["c1"][case_id] - values["c0"][case_id]
        rows.append(
            {
                "case_id": case_id,
                "family_id": families[case_id],
                **condition_data,
                "delta_c1_minus_c0": delta,
                "rate_transition": (
                    "gained" if delta > 0 else "lost" if delta < 0 else "tied"
                ),
            }
        )

    primary = _paired_statistics(
        order=order,
        c0_values=values["c0"],
        c1_values=values["c1"],
        families=families,
        c0_binary=binary["c0"],
        c1_binary=binary["c1"],
        bootstrap_iterations=bootstrap_iterations,
        sign_flip_iterations=sign_flip_iterations,
        seed=seed,
    )
    return {
        "generation_repeat_count_per_question_condition": len(RUN_IDS),
        "judge_repeats_count_as_generation_repeats": False,
        "judgments_used_per_judge_eligible_answer": 1,
        "judgments_used_per_terminal_service_error": 0,
        "by_generation_run": by_run,
        "paired_question_analysis": primary,
        "total_logical_generation_slots": len(order)
        * len(CONDITION_IDS)
        * len(RUN_IDS),
        "total_answer_records": len(order) * len(CONDITION_IDS) * len(RUN_IDS),
        "total_successful_answers": sum(
            not _terminal_service_error(
                answer, path=runs[condition][run_id].answer_path
            )
            for condition in CONDITION_IDS
            for run_id in RUN_IDS
            for answer in runs[condition][run_id].answers.values()
        ),
        "total_definitive_judgments": sum(
            len(runs[condition][run_id].judgments)
            for condition in CONDITION_IDS
            for run_id in RUN_IDS
        ),
        "total_terminal_service_error_count": sum(
            _terminal_service_error(answer, path=runs[condition][run_id].answer_path)
            for condition in CONDITION_IDS
            for run_id in RUN_IDS
            for answer in runs[condition][run_id].answers.values()
        ),
    }, rows


def _load_human_fallback(
    paths: Sequence[Path],
    *,
    calibration_rows: Mapping[str, Mapping[str, Any]],
    calibration_human_inputs: Sequence[Mapping[str, str]],
    run1_by_condition: Mapping[str, GenerationRun],
    order: Sequence[str],
) -> tuple[dict[str, dict[str, bool]], list[dict[str, str]]]:
    if not paths:
        raise ValueError(
            "calibration gate failed; --human-labels with adjudicated run1 labels is required"
        )
    allowed = {(entry["path"], entry["sha256"]) for entry in calibration_human_inputs}
    for path in paths:
        identity = (str(path.resolve()), sha256_file(path))
        if identity not in allowed:
            raise ValueError(
                f"{path}: human-label input is not bound to the calibration artifact"
            )
    metadata, records = load_human_labels(paths)
    any_labeled_answer_ids = {str(record["answer_id"]) for record in records}
    by_answer: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["label_kind"] != "adjudicated":
            continue
        answer_id = str(record["answer_id"])
        if answer_id in by_answer:
            raise ValueError(f"duplicate adjudicated label for answer_id {answer_id}")
        by_answer[answer_id] = record

    result = {condition: {} for condition in CONDITION_IDS}
    for condition in CONDITION_IDS:
        run = run1_by_condition[condition]
        for case_id in order:
            answer_id = str(run.answers[case_id]["answer_id"])
            if _terminal_service_error(
                run.answers[case_id], path=run.answer_path
            ):
                if answer_id in any_labeled_answer_ids:
                    raise ValueError(
                        "terminal service-error slot must not have any human "
                        f"label: {answer_id}"
                    )
                result[condition][case_id] = False
                continue
            label = by_answer.get(answer_id)
            if label is None:
                raise ValueError(f"missing adjudicated run1 human label for {answer_id}")
            if label.get("case_id") != case_id:
                raise ValueError(f"human label case_id mismatch for {answer_id}")
            calibration_adjudicated = calibration_rows[answer_id]["adjudicated"]
            if (
                label.get("score") != calibration_adjudicated.get("score")
                or label.get("grounded_fully_correct")
                is not calibration_adjudicated.get("grounded_fully_correct")
            ):
                raise ValueError(f"human label differs from calibration for {answer_id}")
            result[condition][case_id] = bool(label["grounded_fully_correct"])
    return result, metadata


def build_analysis(
    *,
    cases_path: Path,
    calibration_path: Path,
    c0_answer_paths: Sequence[Path],
    c0_judgment_paths: Sequence[Path],
    c1_answer_paths: Sequence[Path],
    c1_judgment_paths: Sequence[Path],
    human_label_paths: Sequence[Path],
    bootstrap_iterations: int,
    sign_flip_iterations: int,
    seed: int,
    expected_core_count: int = 27,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    order, cases, cases_meta = load_core_cases(cases_path)
    if expected_core_count <= 0:
        raise ValueError("expected Core count must be positive")
    if len(order) != expected_core_count:
        raise ValueError(
            f"Core case count {len(order)} != expected {expected_core_count}"
        )
    families = {case_id: str(cases[case_id]["family_id"]) for case_id in order}
    if len(set(families.values())) != len(order):
        raise ValueError(
            "final Core requires one unique frozen family_id per question"
        )
    runs = {
        "c0": load_condition_runs(
            condition_id="c0",
            answer_paths=c0_answer_paths,
            judgment_paths=c0_judgment_paths,
            core_order=order,
            core_cases=cases,
            cases_artifact_sha256=cases_meta["sha256"],
            cases_canonical_sha256=cases_meta["canonical_sha256"],
        ),
        "c1": load_condition_runs(
            condition_id="c1",
            answer_paths=c1_answer_paths,
            judgment_paths=c1_judgment_paths,
            core_order=order,
            core_cases=cases,
            cases_artifact_sha256=cases_meta["sha256"],
            cases_canonical_sha256=cases_meta["canonical_sha256"],
        ),
    }
    all_runs = [runs[condition][run_id] for condition in CONDITION_IDS for run_id in RUN_IDS]
    experiments = {run.experiment_id for run in all_runs}
    configs = {
        run.judge_config_sha256
        for run in all_runs
        if run.judge_config_sha256 is not None
    }
    answer_ids = [
        str(answer["answer_id"])
        for run in all_runs
        for answer in run.answers.values()
    ]
    if len(experiments) != 1:
        raise ValueError("C0/C1 generation artifacts mix experiment_id")
    if len(configs) != 1:
        raise ValueError("definitive judgments mix judge_config_sha256")
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError(
            "answer_id reuse detected; Judge repeats must not be supplied as generation runs"
        )
    if len({run.answer_artifact_sha256 for run in all_runs}) != len(all_runs):
        raise ValueError("duplicate answer artifact supplied across generation runs")

    run1 = {condition: runs[condition]["run1"] for condition in CONDITION_IDS}
    (
        calibration_threshold_passed,
        calibration_meta,
        calibration_rows,
        calibration_humans,
    ) = _validate_calibration(
        calibration_path,
        run1_by_condition=run1,
        core_order=order,
    )
    run1_terminal_case_ids = {
        condition: [
            case_id
            for case_id in order
            if _terminal_service_error(
                run1[condition].answers[case_id], path=run1[condition].answer_path
            )
        ]
        for condition in CONDITION_IDS
    }
    required_run1_complete = not any(run1_terminal_case_ids.values())
    gate_passed = calibration_threshold_passed and required_run1_complete
    calibration_meta["threshold_gate_passed"] = calibration_threshold_passed
    calibration_meta["required_run1_complete"] = required_run1_complete
    calibration_meta["run1_terminal_service_error_case_ids"] = (
        run1_terminal_case_ids
    )
    calibration_meta["effective_headline_gate_passed"] = gate_passed
    judge_analysis, rows = build_judge_analysis(
        runs,
        order=order,
        families=families,
        bootstrap_iterations=bootstrap_iterations,
        sign_flip_iterations=sign_flip_iterations,
        seed=seed,
    )

    fallback: dict[str, Any] | None = None
    human_metadata: list[dict[str, str]] = []
    if gate_passed:
        if human_label_paths:
            raise ValueError(
                "--human-labels is only accepted when the calibration gate fails"
            )
        headline_source = "llm_judge_three_independent_generation_runs"
        headline = judge_analysis["paired_question_analysis"]
    else:
        human_values, human_metadata = _load_human_fallback(
            human_label_paths,
            calibration_rows=calibration_rows,
            calibration_human_inputs=calibration_humans,
            run1_by_condition=run1,
            order=order,
        )
        fallback = _paired_statistics(
            order=order,
            c0_values={case_id: float(human_values["c0"][case_id]) for case_id in order},
            c1_values={case_id: float(human_values["c1"][case_id]) for case_id in order},
            families=families,
            c0_binary=human_values["c0"],
            c1_binary=human_values["c1"],
            bootstrap_iterations=bootstrap_iterations,
            sign_flip_iterations=sign_flip_iterations,
            seed=seed,
        )
        fallback["generation_run_id"] = "run1"
        fallback["source"] = "adjudicated_human"
        fallback["terminal_service_error_case_ids_by_condition"] = (
            run1_terminal_case_ids
        )
        fallback["terminal_service_error_count"] = sum(
            len(case_ids) for case_ids in run1_terminal_case_ids.values()
        )
        fallback["terminal_service_error_automatic_gfc_value"] = False
        fallback["majority_sensitivity"]["definition"] = (
            "adjudicated human binary GFC on generation run1"
        )
        headline_source = "adjudicated_human_generation_run1_fallback"
        headline = fallback

    artifact_inputs = {
        condition: {
            run_id: {
                "answers": {
                    "path": str(runs[condition][run_id].answer_path),
                    "sha256": runs[condition][run_id].answer_artifact_sha256,
                },
                "judgments": {
                    "path": str(runs[condition][run_id].judgment_path),
                    "sha256": runs[condition][run_id].judgment_artifact_sha256,
                    "judge_run_id": runs[condition][run_id].judge_run_id,
                },
            }
            for run_id in RUN_IDS
        }
        for condition in CONDITION_IDS
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "inputs": {
            "cases": cases_meta,
            "generation_and_definitive_judgments": artifact_inputs,
            "calibration": calibration_meta,
            "fallback_human_labels": human_metadata,
        },
        "integrity_verification": {
            "all_source_artifacts_rehashed": True,
            "answer_and_judgment_record_identities_revalidated": True,
            "case_condition_generation_run_bindings_revalidated": True,
            "exactly_one_definitive_judgment_per_judge_eligible_answer": True,
            "terminal_service_error_slots_have_no_judgment": True,
            "every_core_logical_slot_has_answer_xor_terminal_disposition": True,
            "generation_run_ids": list(RUN_IDS),
            "judge_repeats_count_as_samples": False,
        },
        "methodology": {
            "sample_unit": "paired Core question",
            "effective_sample_n": len(order),
            "expected_core_question_count": expected_core_count,
            "within_question_generation_repeats": 3,
            "judge_repeats_count_as_generation_repeats": False,
            "family_cluster_bootstrap_iterations": bootstrap_iterations,
            "sign_flip_iterations": sign_flip_iterations,
            "seed": seed,
            "headline_gate": (
                "LLM-Judge n=3 only when frozen Judge-human calibration passes; "
                "otherwise adjudicated-human run1 paired GFC"
            ),
            "terminal_service_error_policy": (
                "A terminal service-error generation slot has no Judge score or "
                "judgment, contributes GFC=0, and remains inside paired n. Any "
                "run1 terminal slot forces the Judge headline gate to fail."
            ),
        },
        "calibration_gate_passed": gate_passed,
        "calibration_threshold_gate_passed": calibration_threshold_passed,
        "required_run1_complete": required_run1_complete,
        "headline": {**headline, "source": headline_source},
        "judge_three_generation_runs": {
            "reporting_status": "headline" if gate_passed else "exploratory",
            **judge_analysis,
        },
        "human_run1_fallback": fallback,
        "cases": rows,
    }
    return payload, rows


def render_csv(rows: Iterable[dict[str, Any]]) -> str:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty case CSV")
    flattened: list[dict[str, Any]] = []
    for row in materialized:
        flattened.append(
            {
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "c0_judge_scores_by_run": json.dumps(
                    row["c0"]["judge_scores_by_run"], separators=(",", ":")
                ),
                "c1_judge_scores_by_run": json.dumps(
                    row["c1"]["judge_scores_by_run"], separators=(",", ":")
                ),
                "c0_gfc_by_run": json.dumps(row["c0"]["gfc_by_run"], separators=(",", ":")),
                "c1_gfc_by_run": json.dumps(row["c1"]["gfc_by_run"], separators=(",", ":")),
                "c0_service_errors_by_run": json.dumps(
                    row["c0"]["service_errors_by_run"], separators=(",", ":")
                ),
                "c1_service_errors_by_run": json.dumps(
                    row["c1"]["service_errors_by_run"], separators=(",", ":")
                ),
                "c0_gfc_success_proportion": row["c0"]["gfc_success_proportion"],
                "c1_gfc_success_proportion": row["c1"]["gfc_success_proportion"],
                "delta_c1_minus_c0": row["delta_c1_minus_c0"],
                "rate_transition": row["rate_transition"],
            }
        )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
    writer.writeheader()
    writer.writerows(flattened)
    return handle.getvalue()


def _reject_aliases(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    def aliases(left: Path, right: Path) -> bool:
        if left.resolve() == right.resolve():
            return True
        try:
            return left.exists() and right.exists() and left.samefile(right)
        except OSError:
            return False

    if any(
        aliases(left, right)
        for index, left in enumerate(inputs)
        for right in inputs[index + 1 :]
    ):
        raise ValueError("the same input path was supplied more than once")
    if any(
        aliases(left, right)
        for index, left in enumerate(outputs)
        for right in outputs[index + 1 :]
    ):
        raise ValueError("--json-out and --csv-out must be different files")
    if any(aliases(source, target) for source in inputs for target in outputs):
        raise ValueError("output paths must not overwrite input artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--c0-answers", type=Path, nargs=3, required=True)
    parser.add_argument("--c0-judgments", type=Path, nargs=3, required=True)
    parser.add_argument("--c1-answers", type=Path, nargs=3, required=True)
    parser.add_argument("--c1-judgments", type=Path, nargs=3, required=True)
    parser.add_argument(
        "--human-labels",
        type=Path,
        nargs="+",
        default=[],
        help="required on calibration FAIL; must contain adjudicated run1 labels",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument(
        "--sign-flip-iterations", type=int, default=DEFAULT_SIGN_FLIP_ITERATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-core-count",
        type=int,
        default=27,
        help="frozen final Core question count (default: 27)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap <= 0 or args.sign_flip_iterations <= 0:
        raise SystemExit("--bootstrap and --sign-flip-iterations must be positive")
    inputs = [
        args.cases,
        args.calibration,
        *args.c0_answers,
        *args.c0_judgments,
        *args.c1_answers,
        *args.c1_judgments,
        *args.human_labels,
    ]
    try:
        reject_symlink_inputs(inputs)
        _reject_aliases(inputs, [args.json_out, args.csv_out])
        require_new_outputs([args.json_out, args.csv_out])
        payload, rows = build_analysis(
            cases_path=args.cases,
            calibration_path=args.calibration,
            c0_answer_paths=args.c0_answers,
            c0_judgment_paths=args.c0_judgments,
            c1_answer_paths=args.c1_answers,
            c1_judgment_paths=args.c1_judgments,
            human_label_paths=args.human_labels,
            bootstrap_iterations=args.bootstrap,
            sign_flip_iterations=args.sign_flip_iterations,
            seed=args.seed,
            expected_core_count=args.expected_core_count,
        )
        csv_text = render_csv(rows)
        payload["output_publication"] = {
            "authoritative_completion_artifact": "json",
            "authoritative_path": str(args.json_out.resolve()),
            "required_companion_artifacts": [
                {
                    "kind": "case_csv",
                    "path": str(args.csv_out.resolve()),
                    "sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
                    "bytes": len(csv_text.encode("utf-8")),
                }
            ],
            "contract": (
                "The authoritative JSON is published only after every required "
                "companion is durably published; all output paths are immutable."
            ),
        }
        json_text = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        publish_immutable_texts(
            {args.csv_out: csv_text, args.json_out: json_text},
            authoritative_path=args.json_out,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"validation error: {exc}") from exc
    headline = payload["headline"]
    ci_low, ci_high = headline["paired_family_bootstrap_95ci"]["ci95"]
    print(
        f"headline={headline['source']} n={headline['effective_sample_n']} "
        f"C0={headline['c0_mean_gfc_rate']:.4f} "
        f"C1={headline['c1_mean_gfc_rate']:.4f} "
        f"delta={headline['mean_delta_c1_minus_c0']:+.4f} "
        f"family-bootstrap-95CI=[{ci_low:+.4f},{ci_high:+.4f}] "
        f"sign-flip-p={headline['paired_sign_flip_two_sided']['p_value']:.6g}"
    )
    print(f"calibration_gate_passed={str(payload['calibration_gate_passed']).lower()}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
