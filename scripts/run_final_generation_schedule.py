#!/usr/bin/env python3
"""Create, audit, and execute the frozen final generation call schedule.

The final holdout is deliberately collected one call at a time so Core C0/C1
requests follow a case -> generation-run -> condition AB/BA order.  This
orchestrator is the only supported path for using ``evaluate_service_answers``
with a holdout subset.  It validates the complete holdout and sign-off before
the first service request and audits every append-only artifact before and
after each call.

This schedule contains the 189 direct service calls only (Core 162 + Challenge
27).  The 9 oracle-context diagnostic calls are a separate collector and are
why the full runbook's combined generation count is 198.

Schedule creation and ``--check`` are local-only.  ``--run --dry-run`` performs
the same frozen-holdout gates but never invokes the service collector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_service_answers.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from holdout_gold import (  # noqa: E402
    CHALLENGE_SPLIT,
    CORE_SPLIT,
    collect_holdout_validation_errors,
    load_jsonl,
    sha256_file,
)
from immutable_outputs import exclusive_run_lock  # noqa: E402
import build_holdout_signoff as signoff_artifacts  # noqa: E402
from service_eval_artifacts import (  # noqa: E402
    _issue_exact_final_authorization,
    expected_answer_id,
    load_unique_jsonl,
    sha256_json,
    validate_answer_record,
    validate_final_generation_provenance,
)
from validate_service_holdout import (  # noqa: E402
    _read_repo_review_artifact,
    validate_paths,
)


SCHEDULE_SCHEMA_VERSION = "pnu.final-generation-schedule.v1"
AUTHORIZATION_SCHEMA_VERSION = "pnu.final-generation-authorization.v1"
GATE_INPUT_SCHEMA_VERSION = "pnu.final-holdout-gate-inputs.v1"
ERROR_SCHEMA_VERSION = "pnu.service-answer-attempt.v1"
RUN_IDS = ("run1", "run2", "run3")
CONDITION_IDS = ("c0", "c1")
EXPECTED_CORE_COUNT = 27
EXPECTED_CHALLENGE_COUNT = 9
EXPECTED_CORE_PAIR_COUNT = 81
EXPECTED_CALL_COUNT = 189
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_C0_API_BASE = "http://127.0.0.1:8101"
DEFAULT_C1_API_BASE = "http://127.0.0.1:8100"
ORDER_SEED = 20260914
ORDER_ASSIGNMENT_ALGORITHM = "sha256-category-stratified-balanced-v1"


class ScheduleError(ValueError):
    """The schedule, authorization, or an existing artifact is unsafe."""


def _git_identity(repo_root: Path) -> dict[str, Any]:
    """Return the exact local code revision used by the final collector."""

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.SubprocessError) as exc:
        raise ScheduleError(f"cannot verify Git freeze: {exc}") from exc
    return {"commit": commit, "clean": status == ""}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_jsonl_snapshot(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Read and parse one exact byte snapshot, avoiding hash/read TOCTOU."""

    data = path.read_bytes()
    byte_sha = _sha256_bytes(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScheduleError(f"{path}: JSONL must be UTF-8: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScheduleError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ScheduleError(f"{path}:{line_number}: expected JSON object")
        records.append(record)
    return byte_sha, records


def _file_input(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ScheduleError(f"gate input is not a file: {resolved}")
    data = resolved.read_bytes()
    return {
        "path": str(resolved),
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _payload_input(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _signoff_dependency_binding(
    signoff_path: Path,
    *,
    repo_root: Path,
    protected_paths: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    """Strictly snapshot the sign-off and its packet/A/B dependencies."""

    try:
        signoff_payload = signoff_artifacts._read_regular_bytes(
            signoff_path, label="sign-off manifest"
        )
        payload = signoff_artifacts._load_json_bytes(
            signoff_payload, source=str(signoff_path)
        )
    except (OSError, ValueError) as exc:
        raise ScheduleError(f"cannot bind sign-off manifest: {exc}") from exc
    expected_manifest_keys = {
        "schema_version",
        "cases_sha256",
        "review_packet_path",
        "review_packet_sha256",
        "review_response_sha256s",
        "case_signoffs",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_manifest_keys:
        raise ScheduleError("sign-off manifest schema is not exact")
    if payload.get("schema_version") != signoff_artifacts.SIGNOFF_SCHEMA_VERSION:
        raise ScheduleError("sign-off manifest schema_version is not supported")
    if not isinstance(payload.get("case_signoffs"), list):
        raise ScheduleError("sign-off manifest case_signoffs must be a list")

    try:
        packet_path, packet_payload = _read_repo_review_artifact(
            repo_root,
            payload.get("review_packet_path"),
            label="sign-off review packet",
        )
    except (OSError, ValueError) as exc:
        raise ScheduleError(f"cannot bind sign-off review packet: {exc}") from exc
    if not packet_payload:
        raise ScheduleError("sign-off review packet must not be empty")
    packet_snapshot = _payload_input(packet_path, packet_payload)
    if payload.get("review_packet_sha256") != packet_snapshot["sha256"]:
        raise ScheduleError("sign-off review packet SHA-256 mismatch")

    raw_responses = payload.get("review_response_sha256s")
    if not isinstance(raw_responses, list) or len(raw_responses) != 2:
        raise ScheduleError("sign-off must bind exactly two review responses")
    response_snapshots: list[dict[str, Any]] = []
    review_paths: list[tuple[str, Path]] = [
        ("sign-off", signoff_path.resolve()),
        ("review packet", packet_path),
    ]
    for index, expected_slot in enumerate(("A", "B")):
        record = raw_responses[index]
        expected_record_keys = {"reviewer_slot", "reviewer_id", "sha256", "path"}
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise ScheduleError(
                f"sign-off Reviewer {expected_slot} provenance schema is not exact"
            )
        if record.get("reviewer_slot") != expected_slot:
            raise ScheduleError("sign-off response slots must be exactly A then B")
        reviewer_id = record.get("reviewer_id")
        if (
            not isinstance(reviewer_id, str)
            or not reviewer_id
            or reviewer_id != reviewer_id.strip()
        ):
            raise ScheduleError(
                f"sign-off Reviewer {expected_slot} reviewer_id is invalid"
            )
        try:
            response_path, response_payload = _read_repo_review_artifact(
                repo_root,
                record.get("path"),
                label=f"sign-off Reviewer {expected_slot} response",
            )
        except (OSError, ValueError) as exc:
            raise ScheduleError(
                f"cannot bind Reviewer {expected_slot} response: {exc}"
            ) from exc
        response_snapshot = {
            "reviewer_slot": expected_slot,
            "reviewer_id": reviewer_id,
            **_payload_input(response_path, response_payload),
        }
        if record.get("sha256") != response_snapshot["sha256"]:
            raise ScheduleError(
                f"sign-off Reviewer {expected_slot} response SHA-256 mismatch"
            )
        response_snapshots.append(response_snapshot)
        review_paths.append((f"Reviewer {expected_slot} response", response_path))

    for index, (left_label, left_path) in enumerate(review_paths):
        for right_label, right_path in review_paths[index + 1 :]:
            if signoff_artifacts.paths_alias(left_path, right_path):
                raise ScheduleError(
                    f"{left_label} aliases {right_label}: {left_path}"
                )
        for protected_label, protected_path in protected_paths:
            if signoff_artifacts.paths_alias(left_path, protected_path):
                raise ScheduleError(
                    f"{left_label} aliases {protected_label}: {left_path}"
                )

    return {
        "signoff": _payload_input(signoff_path, signoff_payload),
        "review_packet": packet_snapshot,
        "review_responses": response_snapshots,
    }


def build_gate_input_binding(
    schedule: Mapping[str, Any],
    *,
    dev_paths: Sequence[Path],
    corpus_index: Path,
    signoff_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Bind the four holdout gates to exact input-file byte snapshots.

    The SQLite snapshot hash is SHA-256 over the complete main ``.sqlite`` file
    bytes.  A live WAL/journal (or SHM paired with WAL mode) would make those
    bytes an incomplete database snapshot, so such sidecars are rejected.
    """

    cases_path = Path(schedule["cases_path"])
    cases_input = _file_input(cases_path)
    if cases_input["sha256"] != schedule["cases_sha256"]:
        raise ScheduleError("gate binding cases SHA-256 mismatch")
    if not dev_paths:
        raise ScheduleError("gate binding requires at least one DEV manifest")
    normalized_dev_paths = sorted({path.resolve() for path in dev_paths}, key=str)
    resolved_repo_root = repo_root.resolve()

    resolved_corpus = corpus_index.resolve()
    sidecars = [
        Path(str(resolved_corpus) + suffix)
        for suffix in ("-wal", "-journal", "-shm")
    ]
    present_sidecars = [str(path) for path in sidecars if path.exists()]
    if present_sidecars:
        raise ScheduleError(
            "SQLite content snapshot requires no WAL/journal/SHM sidecars: "
            + ", ".join(present_sidecars)
        )
    corpus_input = _file_input(resolved_corpus)
    if corpus_input["sha256"] != schedule["controls"]["shared"][
        "expected_index_sha256"
    ]:
        raise ScheduleError("corpus index byte SHA-256 does not match schedule")
    signoff_dependencies = _signoff_dependency_binding(
        signoff_path,
        repo_root=resolved_repo_root,
        protected_paths=[
            ("cases", cases_path),
            *[("DEV manifest", path) for path in normalized_dev_paths],
            ("corpus index", resolved_corpus),
        ],
    )
    return {
        "schema_version": GATE_INPUT_SCHEMA_VERSION,
        "repo_root": str(resolved_repo_root),
        "cases": cases_input,
        "dev_manifests": [_file_input(path) for path in normalized_dev_paths],
        "signoff": signoff_dependencies["signoff"],
        "review_packet": signoff_dependencies["review_packet"],
        "review_responses": signoff_dependencies["review_responses"],
        "corpus_index": {
            **corpus_input,
            "snapshot_definition": (
                "sha256-complete-main-sqlite-file-bytes-no-wal-journal-shm-v1"
            ),
        },
    }


def validate_gate_input_binding(
    schedule: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    """Re-hash every gate input and require the exact frozen binding."""

    if binding.get("schema_version") != GATE_INPUT_SCHEMA_VERSION:
        raise ScheduleError("invalid gate input binding schema_version")
    dev_inputs = binding.get("dev_manifests")
    signoff = binding.get("signoff")
    corpus = binding.get("corpus_index")
    repo_root = binding.get("repo_root")
    review_packet = binding.get("review_packet")
    review_responses = binding.get("review_responses")
    if (
        not isinstance(dev_inputs, list)
        or not dev_inputs
        or any(not isinstance(value, Mapping) for value in dev_inputs)
        or not isinstance(signoff, Mapping)
        or not isinstance(corpus, Mapping)
        or not isinstance(repo_root, str)
        or not Path(repo_root).is_absolute()
        or not isinstance(review_packet, Mapping)
        or not isinstance(review_responses, list)
        or len(review_responses) != 2
        or any(not isinstance(value, Mapping) for value in review_responses)
    ):
        raise ScheduleError("gate input binding is incomplete")
    try:
        rebuilt = build_gate_input_binding(
            schedule,
            dev_paths=[Path(str(value.get("path") or "")) for value in dev_inputs],
            corpus_index=Path(str(corpus.get("path") or "")),
            signoff_path=Path(str(signoff.get("path") or "")),
            repo_root=Path(repo_root),
        )
    except (OSError, ValueError) as exc:
        raise ScheduleError(
            "one or more frozen gate input byte SHA values changed"
        ) from exc
    if dict(binding) != rebuilt:
        raise ScheduleError("one or more frozen gate input byte SHA values changed")


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ScheduleError(f"{label} must be a non-empty string")
    return text


def _schedule_digest(schedule: Mapping[str, Any]) -> str:
    payload = dict(schedule)
    payload.pop("schedule_id", None)
    payload.pop("schedule_sha256", None)
    return sha256_json(payload)


def _artifact_id(condition_id: str, split_label: str, run_id: str) -> str:
    lane = "core" if split_label == CORE_SPLIT else "challenge"
    return f"{condition_id}-{lane}-{run_id}"


def _balanced_core_assignments(
    core_cases: Sequence[Mapping[str, Any]],
    *,
    seed: int = ORDER_SEED,
) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]]]:
    """Assign AB/BA by seeded hashes, balanced within every category."""

    by_category: dict[str, list[tuple[str, str]]] = {}
    for case in core_cases:
        category = _nonempty(case.get("category"), "Core category")
        for run_id in RUN_IDS:
            by_category.setdefault(category, []).append((str(case["id"]), run_id))
    if len(by_category) != 9 or any(len(pairs) != 9 for pairs in by_category.values()):
        raise ScheduleError(
            "seeded AB/BA assignment requires 9 categories with 9 case-run pairs each"
        )

    category_order = sorted(
        by_category,
        key=lambda category: hashlib.sha256(
            f"{seed}:category:{category}".encode("utf-8")
        ).hexdigest(),
    )
    ab_major_categories = set(category_order[:5])
    assignments: dict[tuple[str, str], str] = {}
    for category, pairs in by_category.items():
        ranked = sorted(
            pairs,
            key=lambda pair: hashlib.sha256(
                f"{seed}:pair:{category}:{pair[0]}:{pair[1]}".encode("utf-8")
            ).hexdigest(),
        )
        ab_target = 5 if category in ab_major_categories else 4
        for index, pair in enumerate(ranked):
            assignments[pair] = "AB" if index < ab_target else "BA"

    assignment_rows = [
        {
            "case_id": str(case["id"]),
            "generation_run_id": run_id,
            "pair_label": assignments[(str(case["id"]), run_id)],
        }
        for case in core_cases
        for run_id in RUN_IDS
    ]
    return assignments, assignment_rows


def _build_artifacts(
    output_root: Path,
    core_ids: Sequence[str],
    challenge_ids: Sequence[str],
) -> list[dict[str, Any]]:
    generation_root = (output_root / "generation").resolve()
    artifacts: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        for condition_id, split_label, case_ids in (
            ("c0", CORE_SPLIT, core_ids),
            ("c1", CORE_SPLIT, core_ids),
            ("c1", CHALLENGE_SPLIT, challenge_ids),
        ):
            artifact_id = _artifact_id(condition_id, split_label, run_id)
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "condition_id": condition_id,
                    "generation_run_id": run_id,
                    "split": split_label,
                    "expected_case_count": len(case_ids),
                    "expected_case_ids": list(case_ids),
                    "expected_case_ids_sha256": sha256_json(list(case_ids)),
                    "answers_path": str(
                        generation_root / f"{artifact_id}.answers.jsonl"
                    ),
                    "errors_path": str(
                        generation_root / f"{artifact_id}.errors.jsonl"
                    ),
                    "journal_path": str(
                        generation_root / f"{artifact_id}.attempt-journal.jsonl"
                    ),
                }
            )
    return artifacts


def build_schedule(
    cases_path: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    output_root: Path,
    expected_corpus_revision: str,
    expected_git_commit: str,
    expected_index_sha256: str,
    expected_source_manifest_sha256: str,
    model: str = DEFAULT_MODEL,
    c0_api_base: str = DEFAULT_C0_API_BASE,
    c1_api_base: str = DEFAULT_C1_API_BASE,
) -> dict[str, Any]:
    """Build the one canonical 189-call schedule for a 36-case holdout."""

    schema_errors = collect_holdout_validation_errors(cases)
    if schema_errors:
        raise ScheduleError(
            "holdout schema/composition failed: " + "; ".join(schema_errors)
        )
    experiment_id = _nonempty(experiment_id, "experiment_id")
    expected_corpus_revision = _nonempty(
        expected_corpus_revision, "expected_corpus_revision"
    )
    expected_git_commit = _nonempty(expected_git_commit, "expected_git_commit")
    expected_index_sha256 = _nonempty(
        expected_index_sha256, "expected_index_sha256"
    ).casefold()
    expected_source_manifest_sha256 = _nonempty(
        expected_source_manifest_sha256,
        "expected_source_manifest_sha256",
    ).casefold()
    if len(expected_git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_git_commit
    ):
        raise ScheduleError("expected_git_commit must be a lowercase SHA-1")
    if len(expected_index_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_index_sha256
    ):
        raise ScheduleError("expected_index_sha256 must be a lowercase SHA-256")
    if len(expected_source_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_source_manifest_sha256
    ):
        raise ScheduleError(
            "expected_source_manifest_sha256 must be a lowercase SHA-256"
        )
    model = _nonempty(model, "model")
    if model != DEFAULT_MODEL:
        raise ScheduleError(
            f"final schedule model is frozen to {DEFAULT_MODEL!r}"
        )
    c0_api_base = _nonempty(c0_api_base, "c0_api_base").rstrip("/")
    c1_api_base = _nonempty(c1_api_base, "c1_api_base").rstrip("/")

    core_cases = [case for case in cases if case.get("split") == CORE_SPLIT]
    challenge_cases = [
        case for case in cases if case.get("split") == CHALLENGE_SPLIT
    ]
    if len(core_cases) != EXPECTED_CORE_COUNT:
        raise ScheduleError(
            f"expected {EXPECTED_CORE_COUNT} Core cases, got {len(core_cases)}"
        )
    if len(challenge_cases) != EXPECTED_CHALLENGE_COUNT:
        raise ScheduleError(
            "expected "
            f"{EXPECTED_CHALLENGE_COUNT} Challenge cases, got {len(challenge_cases)}"
        )

    core_ids = [str(case["id"]) for case in core_cases]
    challenge_ids = [str(case["id"]) for case in challenge_cases]
    artifacts = _build_artifacts(output_root.resolve(), core_ids, challenge_ids)
    artifact_lookup = {row["artifact_id"]: row for row in artifacts}
    assignments, assignment_rows = _balanced_core_assignments(core_cases)

    calls: list[dict[str, Any]] = []
    call_order = 0
    pair_order = 0
    ab_count = 0
    ba_count = 0
    for case_position, case in enumerate(core_cases, 1):
        for run_number, run_id in enumerate(RUN_IDS, 1):
            pair_order += 1
            pair_label = assignments[(str(case["id"]), run_id)]
            if pair_label == "AB":
                conditions = ("c0", "c1")
                ab_count += 1
            else:
                conditions = ("c1", "c0")
                ba_count += 1
            for condition_position, condition_id in enumerate(conditions, 1):
                call_order += 1
                artifact_id = _artifact_id(condition_id, CORE_SPLIT, run_id)
                assert artifact_id in artifact_lookup
                calls.append(
                    {
                        "call_order": call_order,
                        "phase": "core",
                        "split": CORE_SPLIT,
                        "case_position": case_position,
                        "case_id": str(case["id"]),
                        "case_sha256": sha256_json(case),
                        "generation_run_id": run_id,
                        "run_number": run_number,
                        "condition_id": condition_id,
                        "condition_position": condition_position,
                        "pair_order": pair_order,
                        "pair_label": pair_label,
                        "artifact_id": artifact_id,
                    }
                )

    for case_position, case in enumerate(challenge_cases, 1):
        for run_number, run_id in enumerate(RUN_IDS, 1):
            call_order += 1
            artifact_id = _artifact_id("c1", CHALLENGE_SPLIT, run_id)
            assert artifact_id in artifact_lookup
            calls.append(
                {
                    "call_order": call_order,
                    "phase": "challenge",
                    "split": CHALLENGE_SPLIT,
                    "case_position": case_position,
                    "case_id": str(case["id"]),
                    "case_sha256": sha256_json(case),
                    "generation_run_id": run_id,
                    "run_number": run_number,
                    "condition_id": "c1",
                    "condition_position": 1,
                    "pair_order": None,
                    "pair_label": None,
                    "artifact_id": artifact_id,
                }
            )

    unsigned: dict[str, Any] = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": sha256_file(cases_path),
        "cases_canonical_sha256": sha256_json(list(cases)),
        "output_root": str(output_root.resolve()),
        "generation_run_ids": list(RUN_IDS),
        "order_assignment": {
            "seed": ORDER_SEED,
            "algorithm": ORDER_ASSIGNMENT_ALGORITHM,
            "assignment_sha256": sha256_json(assignment_rows),
            "assignments": assignment_rows,
        },
        "controls": {
            "shared": {
                "provider": "frontier",
                "model": model,
                "institution": None,
                "context_k": 8,
                "max_context_chars": 24_000,
                "max_output_tokens": 900,
                "sampling_parameters": [],
                "parser_profile": "cascade",
                "retrieval_mode": "bm25",
                "expected_corpus_revision": expected_corpus_revision,
                "expected_git_commit": expected_git_commit,
                "expected_index_sha256": expected_index_sha256,
                "expected_source_manifest_sha256": (
                    expected_source_manifest_sha256
                ),
                "expected_context_chunks_per_document": 2,
                "eval_trace": True,
                "allow_unpinned": False,
                "sleep_seconds": 2.0,
                "timeout_seconds": 180.0,
                "max_attempts": 3,
                "retry_backoff_seconds": 2.0,
            },
            "conditions": {
                "c0": {
                    "api_base": c0_api_base,
                    "expected_retrieval_tuning": False,
                },
                "c1": {
                    "api_base": c1_api_base,
                    "expected_retrieval_tuning": True,
                },
            },
        },
        "artifacts": artifacts,
        "calls": calls,
        "summary": {
            "core_case_count": len(core_ids),
            "challenge_case_count": len(challenge_ids),
            "generation_run_count": len(RUN_IDS),
            "core_pair_count": pair_order,
            "core_ab_pair_count": ab_count,
            "core_ba_pair_count": ba_count,
            "core_call_count": pair_order * 2,
            "challenge_c1_call_count": len(challenge_ids) * len(RUN_IDS),
            "total_call_count": len(calls),
        },
    }
    digest = _schedule_digest(unsigned)
    schedule = dict(unsigned)
    schedule["schedule_id"] = f"final_generation_{digest[:24]}"
    schedule["schedule_sha256"] = digest
    return schedule


def write_new_schedule(path: Path, schedule: Mapping[str, Any]) -> None:
    """Create a schedule once; never overwrite a possibly used schedule."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        schedule, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
    except FileExistsError as exc:
        raise ScheduleError(
            f"schedule already exists and will not be overwritten: {path}"
        ) from exc


def load_schedule(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScheduleError(f"cannot load schedule {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScheduleError("schedule must be a JSON object")
    return value


def validate_schedule_mapping(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild an in-memory schedule and require canonical semantic equality."""

    if schedule.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ScheduleError("unsupported schedule schema_version")
    cases_path = Path(_nonempty(schedule.get("cases_path"), "cases_path"))
    try:
        cases = load_jsonl(cases_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ScheduleError(f"cannot load scheduled cases: {exc}") from exc
    if sha256_file(cases_path) != schedule.get("cases_sha256"):
        raise ScheduleError("scheduled cases SHA-256 no longer matches")
    if sha256_json(cases) != schedule.get("cases_canonical_sha256"):
        raise ScheduleError("scheduled canonical cases SHA-256 no longer matches")

    controls = schedule.get("controls")
    if not isinstance(controls, Mapping):
        raise ScheduleError("schedule controls must be an object")
    shared = controls.get("shared")
    conditions = controls.get("conditions")
    if not isinstance(shared, Mapping) or not isinstance(conditions, Mapping):
        raise ScheduleError("schedule controls are incomplete")
    c0 = conditions.get("c0")
    c1 = conditions.get("c1")
    if not isinstance(c0, Mapping) or not isinstance(c1, Mapping):
        raise ScheduleError("schedule condition controls are incomplete")

    expected = build_schedule(
        cases_path,
        cases,
        experiment_id=_nonempty(schedule.get("experiment_id"), "experiment_id"),
        output_root=Path(
            _nonempty(schedule.get("output_root"), "output_root")
        ),
        expected_corpus_revision=_nonempty(
            shared.get("expected_corpus_revision"),
            "controls.shared.expected_corpus_revision",
        ),
        expected_git_commit=_nonempty(
            shared.get("expected_git_commit"),
            "controls.shared.expected_git_commit",
        ),
        expected_index_sha256=_nonempty(
            shared.get("expected_index_sha256"),
            "controls.shared.expected_index_sha256",
        ),
        expected_source_manifest_sha256=_nonempty(
            shared.get("expected_source_manifest_sha256"),
            "controls.shared.expected_source_manifest_sha256",
        ),
        model=_nonempty(shared.get("model"), "controls.shared.model"),
        c0_api_base=_nonempty(c0.get("api_base"), "controls.conditions.c0.api_base"),
        c1_api_base=_nonempty(c1.get("api_base"), "controls.conditions.c1.api_base"),
    )
    if schedule != expected:
        raise ScheduleError(
            "schedule content is not the canonical case/run/condition plan"
        )
    if schedule.get("schedule_sha256") != _schedule_digest(schedule):
        raise ScheduleError("schedule_sha256 mismatch")
    return dict(schedule)


def validate_schedule(path: Path) -> dict[str, Any]:
    """Load and validate a frozen schedule from disk."""

    return validate_schedule_mapping(load_schedule(path))


def _artifact_lookup(schedule: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = schedule.get("artifacts")
    if not isinstance(artifacts, list):
        raise ScheduleError("schedule artifacts must be a list")
    result: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ScheduleError("schedule artifact must be an object")
        artifact_id = _nonempty(artifact.get("artifact_id"), "artifact_id")
        if artifact_id in result:
            raise ScheduleError(f"duplicate artifact_id {artifact_id!r}")
        result[artifact_id] = artifact
    return result


def _expected_authorization_metadata(
    schedule: Mapping[str, Any],
    artifact: Mapping[str, Any],
    gate_summary_sha256: str,
    gate_input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "collection_purpose": "final_generation",
        "schedule_id": schedule["schedule_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "cases_sha256": schedule["cases_sha256"],
        "cases_canonical_sha256": schedule["cases_canonical_sha256"],
        "gate_summary_sha256": gate_summary_sha256,
        "gate_input_binding": dict(gate_input_binding),
        "gate_input_binding_sha256": sha256_json(gate_input_binding),
        "artifact_id": artifact["artifact_id"],
        "split": artifact["split"],
        "expected_case_count": artifact["expected_case_count"],
        "expected_case_ids_sha256": artifact["expected_case_ids_sha256"],
        "source_manifest_sha256": schedule["controls"]["shared"][
            "expected_source_manifest_sha256"
        ],
    }


def _expected_record_schedule_metadata(
    schedule: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schedule_id": schedule["schedule_id"],
        "collection_purpose": "final_generation",
        "schedule_sha256": schedule["schedule_sha256"],
        "call_order": entry["call_order"],
        "phase": entry["phase"],
        "split": entry["split"],
        "artifact_id": entry["artifact_id"],
        "pair_order": entry["pair_order"],
        "pair_label": entry["pair_label"],
        "condition_position": entry["condition_position"],
    }


def _validate_collector_config(
    config: Any,
    *,
    schedule: Mapping[str, Any],
    artifact: Mapping[str, Any],
    gate_hashes: set[str],
    gate_input_bindings: dict[str, Mapping[str, Any]],
) -> None:
    if not isinstance(config, Mapping):
        raise ScheduleError("answer collector_config must be an object")
    shared = schedule["controls"]["shared"]
    condition = schedule["controls"]["conditions"][artifact["condition_id"]]
    expected = {
        "api_base": condition["api_base"],
        "cases_path": schedule["cases_path"],
        "cases_sha256": schedule["cases_sha256"],
        "cases_canonical_sha256": schedule["cases_canonical_sha256"],
        "selected_case_ids_sha256": artifact["expected_case_ids_sha256"],
        "provider": shared["provider"],
        "model": shared["model"],
        "institution": shared["institution"],
        "context_k": shared["context_k"],
        "expected_generation_max_context_chars": shared["max_context_chars"],
        "expected_generation_max_output_tokens": shared["max_output_tokens"],
        "expected_generation_sampling_parameters": shared[
            "sampling_parameters"
        ],
        "parser_profile": shared["parser_profile"],
        "retrieval_mode": shared["retrieval_mode"],
        "expected_corpus_revision": shared["expected_corpus_revision"],
        "expected_source_manifest_sha256": shared[
            "expected_source_manifest_sha256"
        ],
        "expected_retrieval_tuning": condition[
            "expected_retrieval_tuning"
        ],
        "expected_context_chunks_per_document": shared[
            "expected_context_chunks_per_document"
        ],
        "eval_trace": shared["eval_trace"],
        "allow_unpinned": shared["allow_unpinned"],
        "max_attempts": shared["max_attempts"],
        "retry_backoff_seconds": shared["retry_backoff_seconds"],
        "request_timeout_seconds": shared["timeout_seconds"],
        "inter_call_sleep_seconds": shared["sleep_seconds"],
        "errors_path": artifact["errors_path"],
        "attempt_journal_path": artifact["journal_path"],
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ScheduleError(
                f"collector control mismatch for {artifact['artifact_id']}: "
                f"{key} expected {expected_value!r}, got {config.get(key)!r}"
            )
    authorization = config.get("final_authorization")
    if not isinstance(authorization, Mapping):
        raise ScheduleError("scheduled answer omitted final_authorization")
    gate_hash = _nonempty(
        authorization.get("gate_summary_sha256"), "gate_summary_sha256"
    )
    if len(gate_hash) != 64 or any(
        character not in "0123456789abcdef" for character in gate_hash
    ):
        raise ScheduleError("invalid gate_summary_sha256")
    gate_hashes.add(gate_hash)
    gate_input_binding = authorization.get("gate_input_binding")
    if not isinstance(gate_input_binding, Mapping):
        raise ScheduleError("final authorization omitted gate input binding")
    binding_sha = sha256_json(gate_input_binding)
    if authorization.get("gate_input_binding_sha256") != binding_sha:
        raise ScheduleError("gate input binding SHA-256 mismatch")
    gate_input_bindings[binding_sha] = gate_input_binding
    expected_authorization = _expected_authorization_metadata(
        schedule, artifact, gate_hash, gate_input_binding
    )
    if dict(authorization) != expected_authorization:
        raise ScheduleError("final_authorization metadata mismatch")

    server_config = config.get("server_config")
    if not isinstance(server_config, Mapping):
        raise ScheduleError("collector server_config must be an object")
    server_expected = {
        "parser_profile": shared["parser_profile"],
        "retrieval_mode": shared["retrieval_mode"],
        "corpus_revision": shared["expected_corpus_revision"],
        "retrieval_tuning": condition["expected_retrieval_tuning"],
        "context_chunks_per_document": shared[
            "expected_context_chunks_per_document"
        ],
        "evaluation_trace_enabled": True,
        "generation": {
            "provider": "frontier",
            "configured": True,
            "model": shared["model"],
            "max_context_chars": shared["max_context_chars"],
            "max_output_tokens": shared["max_output_tokens"],
            "sampling_parameters": shared["sampling_parameters"],
        },
        "profile_index": {
            "sha256": shared["expected_index_sha256"],
            "size_bytes": gate_input_binding["corpus_index"]["size_bytes"],
            "source_manifest_sha256": shared[
                "expected_source_manifest_sha256"
            ],
        },
    }
    freeze = server_config.get("freeze")
    if not isinstance(freeze, Mapping):
        raise ScheduleError("recorded server freeze metadata is missing")
    if (
        freeze.get("startup_git_commit") != shared["expected_git_commit"]
        or freeze.get("startup_worktree_clean") is not True
        or not isinstance(freeze.get("startup_code_sha256"), str)
        or len(freeze.get("startup_code_sha256", "")) != 64
        or not isinstance(freeze.get("process_started_at"), str)
        or not freeze.get("process_started_at")
    ):
        raise ScheduleError("recorded server startup identity mismatch")
    comparable = dict(server_config)
    comparable.pop("freeze", None)
    if comparable != server_expected:
        raise ScheduleError("recorded server controls do not match schedule")


def _validate_error_record(
    record: dict[str, Any],
    *,
    schedule: Mapping[str, Any],
    artifact: Mapping[str, Any],
    entry_by_case: Mapping[str, Mapping[str, Any]],
    gate_hashes: set[str],
    gate_input_bindings: dict[str, Mapping[str, Any]],
) -> int:
    if record.get("schema_version") != ERROR_SCHEMA_VERSION:
        raise ScheduleError("invalid scheduled error schema_version")
    if record.get("record_type") != "answer_collection_error":
        raise ScheduleError("invalid scheduled error record_type")
    attempt_number = record.get("attempt_number")
    if not isinstance(attempt_number, int) or attempt_number <= 0:
        raise ScheduleError("invalid scheduled error attempt_number")
    payload = dict(record)
    actual_sha = payload.pop("record_sha256", None)
    if actual_sha != sha256_json(payload):
        raise ScheduleError("scheduled error record_sha256 mismatch")
    case_id = _nonempty(record.get("case_id"), "error case_id")
    entry = entry_by_case.get(case_id)
    if entry is None:
        raise ScheduleError(
            f"error artifact contains unexpected case_id {case_id!r}"
        )
    for key in ("experiment_id", "condition_id", "generation_run_id"):
        expected = (
            schedule["experiment_id"]
            if key == "experiment_id"
            else artifact[key]
        )
        if record.get(key) != expected:
            raise ScheduleError(f"scheduled error {key} mismatch")
    if record.get("case_sha256") != entry["case_sha256"]:
        raise ScheduleError("scheduled error case_sha256 mismatch")
    if record.get("call_order") != entry["call_order"]:
        raise ScheduleError("scheduled error call_order mismatch")
    if record.get("collection_schedule") != _expected_record_schedule_metadata(
        schedule, entry
    ):
        raise ScheduleError("scheduled error collection metadata mismatch")
    _validate_collector_config(
        record.get("collector_config"),
        schedule=schedule,
        artifact=artifact,
        gate_hashes=gate_hashes,
        gate_input_bindings=gate_input_bindings,
    )
    collector_config = record["collector_config"]
    collector_config_sha = sha256_json(collector_config)
    if record.get("collector_config_sha256") != collector_config_sha:
        raise ScheduleError("scheduled error collector_config_sha256 mismatch")
    expected_logical_id = expected_answer_id(
        {
            "experiment_id": schedule["experiment_id"],
            "condition_id": artifact["condition_id"],
            "generation_run_id": artifact["generation_run_id"],
            "case_id": case_id,
        }
    )
    if record.get("logical_answer_id") != expected_logical_id:
        raise ScheduleError("scheduled error logical_answer_id mismatch")
    attempt_identity = {
        "logical_answer_id": expected_logical_id,
        "collector_config_sha256": collector_config_sha,
        "attempt_number": attempt_number,
    }
    expected_attempt_id = f"attempt_{sha256_json(attempt_identity)[:24]}"
    if record.get("attempt_id") != expected_attempt_id:
        raise ScheduleError("scheduled error attempt_id mismatch")
    return int(entry["call_order"])


def audit_schedule_artifacts(
    schedule: Mapping[str, Any], *, verify_gate_inputs: bool = True
) -> dict[str, Any]:
    """Validate artifacts and require completed calls to be a global prefix."""

    artifacts = _artifact_lookup(schedule)
    shared = schedule["controls"]["shared"]
    calls = schedule["calls"]
    calls_by_artifact: dict[str, list[dict[str, Any]]] = {
        artifact_id: [] for artifact_id in artifacts
    }
    for entry in calls:
        calls_by_artifact[entry["artifact_id"]].append(entry)

    generation_root = Path(schedule["output_root"]) / "generation"
    expected_paths = {
        Path(artifact[key]).resolve()
        for artifact in artifacts.values()
        for key in ("answers_path", "errors_path", "journal_path")
    }
    if generation_root.exists():
        observed_paths = {
            path.resolve()
            for pattern in (
                "*.answers.jsonl", "*.errors.jsonl", "*.attempt-journal.jsonl"
            )
            for path in generation_root.rglob(pattern)
        }
        unexpected = sorted(str(path) for path in observed_paths - expected_paths)
        if unexpected:
            raise ScheduleError(
                "unexpected generation artifact(s): " + ", ".join(unexpected)
            )

    completed_orders: set[int] = set()
    error_orders: set[int] = set()
    gate_hashes: set[str] = set()
    gate_input_bindings: dict[str, Mapping[str, Any]] = {}
    answer_count = 0
    error_count = 0
    for artifact_id, artifact in artifacts.items():
        expected_entries = calls_by_artifact[artifact_id]
        entry_by_case = {entry["case_id"]: entry for entry in expected_entries}
        answers_path = Path(artifact["answers_path"])
        if answers_path.is_symlink():
            raise ScheduleError(f"answer artifact must not be a symlink: {answers_path}")
        answers = (
            load_unique_jsonl(answers_path, key="answer_id")
            if answers_path.exists()
            else []
        )
        journal_path = Path(artifact["journal_path"])
        if journal_path.is_symlink():
            raise ScheduleError(f"journal artifact must not be a symlink: {journal_path}")
        journal_records = (
            load_unique_jsonl(journal_path, key="journal_event_id")
            if journal_path.exists()
            else []
        )
        journal_by_order: dict[int, list[dict[str, Any]]] = {}
        journal_entry_by_order = {
            int(value["call_order"]): value for value in expected_entries
        }
        for event in journal_records:
            payload = dict(event)
            record_sha = payload.pop("record_sha256", None)
            event_id = payload.pop("journal_event_id", None)
            if event.get("schema_version") != "pnu.final-generation-slot-journal.v1":
                raise ScheduleError("invalid slot journal schema")
            if record_sha != sha256_json(payload):
                raise ScheduleError("slot journal record SHA-256 mismatch")
            if event_id != "journal_" + sha256_json({**payload, "record_sha256": record_sha})[:24]:
                raise ScheduleError("slot journal event id mismatch")
            order = event.get("call_order")
            if not isinstance(order, int) or order not in journal_entry_by_order:
                raise ScheduleError("slot journal call_order is invalid")
            expected_entry = journal_entry_by_order[order]
            expected_identity = {
                "schedule_id": schedule["schedule_id"],
                "schedule_sha256": schedule["schedule_sha256"],
                "artifact_id": artifact_id,
                "case_id": expected_entry["case_id"],
                "case_sha256": expected_entry["case_sha256"],
                "max_chat_attempts": shared["max_attempts"],
            }
            if any(event.get(key) != value for key, value in expected_identity.items()):
                raise ScheduleError("slot journal identity/control mismatch")
            if event.get("event") not in {
                "slot_started", "slot_completed", "slot_poisoned"
            }:
                raise ScheduleError("invalid slot journal event")
            if event.get("event") == "slot_started" and event.get("outcome") is not None:
                raise ScheduleError("slot_started journal outcome must be null")
            journal_by_order.setdefault(order, []).append(event)
        if len(answers) > len(expected_entries):
            raise ScheduleError(f"too many answers in {artifact_id}")
        answer_outcomes: dict[int, str] = {}
        answer_by_order: dict[int, dict[str, Any]] = {}
        for local_index, record in enumerate(answers):
            entry = expected_entries[local_index]
            try:
                validate_answer_record(record)
            except ValueError as exc:
                raise ScheduleError(
                    f"invalid answer in {artifact_id}: {exc}"
                ) from exc
            expected_fields = {
                "experiment_id": schedule["experiment_id"],
                "condition_id": artifact["condition_id"],
                "generation_run_id": artifact["generation_run_id"],
                "case_id": entry["case_id"],
                "case_sha256": entry["case_sha256"],
                "call_order": entry["call_order"],
            }
            for key, expected_value in expected_fields.items():
                if record.get(key) != expected_value:
                    raise ScheduleError(
                        f"answer {artifact_id} {key} mismatch at local row "
                        f"{local_index + 1}"
                    )
            if record.get(
                "collection_schedule"
            ) != _expected_record_schedule_metadata(schedule, entry):
                raise ScheduleError("answer collection_schedule mismatch")
            _validate_collector_config(
                record.get("collector_config"),
                schedule=schedule,
                artifact=artifact,
                gate_hashes=gate_hashes,
                gate_input_bindings=gate_input_bindings,
            )
            try:
                validate_final_generation_provenance(
                    record,
                    provider=shared["provider"],
                    model=shared["model"],
                    max_output_tokens=shared["max_output_tokens"],
                    require_collection_attempts=True,
                )
            except ValueError as exc:
                raise ScheduleError(
                    f"invalid generation provenance in {artifact_id}: {exc}"
                ) from exc
            if record.get("collector_config_sha256") != sha256_json(
                record["collector_config"]
            ):
                raise ScheduleError("answer collector_config_sha256 mismatch")
            completed_orders.add(int(entry["call_order"]))
            answer_outcomes[int(entry["call_order"])] = str(
                record.get("slot_outcome")
            )
            answer_by_order[int(entry["call_order"])] = record
            answer_count += 1

        for entry in expected_entries:
            events = journal_by_order.get(int(entry["call_order"]), [])
            event_names = [event.get("event") for event in events]
            answer_present = int(entry["call_order"]) in completed_orders
            if len(event_names) != len(set(event_names)):
                raise ScheduleError("duplicate logical-slot journal event")
            if event_names and event_names[0] != "slot_started":
                raise ScheduleError("slot journal must begin with slot_started")
            if "slot_poisoned" in event_names:
                if event_names != ["slot_started", "slot_poisoned"]:
                    raise ScheduleError("invalid poisoned slot journal transition")
                poisoned = events[-1].get("outcome")
                if poisoned not in {
                    "nonretryable_transport", "response_decode",
                    "response_contract", "response_control",
                }:
                    raise ScheduleError("invalid slot poison reason")
                raise ScheduleError("generation schedule is poisoned; use a new experiment")
            if "slot_started" in event_names and not answer_present:
                raise ScheduleError(
                    "uncertain started logical slot; automatic retry forbidden"
                )
            if answer_present:
                if event_names != ["slot_started", "slot_completed"]:
                    raise ScheduleError(
                        "completed answer requires exact started/completed slot journal"
                    )
                if events[-1].get("outcome") not in {"answer", "service_error"}:
                    raise ScheduleError("invalid completed slot outcome")
                if events[-1].get("outcome") != answer_outcomes.get(
                    int(entry["call_order"])
                ):
                    raise ScheduleError("slot journal/answer outcome mismatch")

        errors_path = Path(artifact["errors_path"])
        if errors_path.is_symlink():
            raise ScheduleError(f"error artifact must not be a symlink: {errors_path}")
        errors = (
            load_unique_jsonl(errors_path, key="attempt_id")
            if errors_path.exists()
            else []
        )
        attempts_by_case: dict[str, int] = {}
        errors_by_order: dict[int, list[dict[str, Any]]] = {}
        for record in errors:
            call_order = _validate_error_record(
                record,
                schedule=schedule,
                artifact=artifact,
                entry_by_case=entry_by_case,
                gate_hashes=gate_hashes,
                gate_input_bindings=gate_input_bindings,
            )
            case_id = str(record["case_id"])
            expected_attempt = attempts_by_case.get(case_id, 0) + 1
            if record.get("attempt_number") != expected_attempt:
                raise ScheduleError(
                    f"non-contiguous attempts for {case_id}: expected "
                    f"{expected_attempt}, got {record.get('attempt_number')!r}"
                )
            attempts_by_case[case_id] = expected_attempt
            errors_by_order.setdefault(call_order, []).append(record)
            error_orders.add(call_order)
            error_count += 1

        for entry in expected_entries:
            order = int(entry["call_order"])
            answer = answer_by_order.get(order)
            slot_errors = errors_by_order.get(order, [])
            if answer is None:
                if slot_errors:
                    # An error sidecar proves the request started. It can never
                    # be treated as an untouched slot eligible for replay.
                    raise ScheduleError(
                        "error-sidecar-only logical slot is uncertain/poisoned"
                    )
                continue
            if answer.get("slot_outcome") == "answer":
                if slot_errors:
                    raise ScheduleError("successful answer has terminal error sidecar")
                continue
            if answer.get("slot_outcome") != "service_error":
                raise ScheduleError("invalid answer slot outcome")
            if len(slot_errors) != 1:
                raise ScheduleError(
                    "terminal service_error requires exactly one error sidecar"
                )
            sidecar = slot_errors[0]
            detail = answer.get("service_error")
            error_detail = sidecar.get("error")
            if not isinstance(detail, Mapping) or not isinstance(error_detail, Mapping):
                raise ScheduleError("terminal service_error detail is invalid")
            expected_terminal = {
                "stage": sidecar.get("stage"),
                "type": error_detail.get("type"),
                "message": error_detail.get("message"),
                "retryable": error_detail.get("retryable"),
                "http_status": error_detail.get("http_status"),
                "request_attempts": sidecar.get("request_attempts"),
            }
            if dict(detail) != expected_terminal:
                raise ScheduleError("terminal answer/error sidecar mismatch")
            if (
                sidecar.get("logical_answer_id") != answer.get("answer_id")
                or sidecar.get("collector_config_sha256")
                != answer.get("collector_config_sha256")
                or sidecar.get("case_id") != answer.get("case_id")
                or sidecar.get("call_order") != answer.get("call_order")
            ):
                raise ScheduleError("terminal answer/error identity mismatch")

    if len(gate_hashes) > 1:
        raise ScheduleError("artifacts were collected under different holdout gates")
    if len(gate_input_bindings) > 1:
        raise ScheduleError("artifacts bind different gate input byte snapshots")
    if verify_gate_inputs and gate_input_bindings:
        validate_gate_input_binding(
            schedule, next(iter(gate_input_bindings.values()))
        )
    completed_count = len(completed_orders)
    expected_prefix = set(range(1, completed_count + 1))
    if completed_orders != expected_prefix:
        missing = sorted(expected_prefix - completed_orders)
        out_of_order = sorted(completed_orders - expected_prefix)
        raise ScheduleError(
            "completed answers are not a global call_order prefix; "
            f"missing={missing}, out_of_order={out_of_order}"
        )
    next_call_order = completed_count + 1
    future_errors = sorted(order for order in error_orders if order > next_call_order)
    if future_errors:
        raise ScheduleError(
            f"error log contains future out-of-order call(s): {future_errors}"
        )
    total = len(calls)
    return {
        "ok": True,
        "schedule_id": schedule["schedule_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "total_call_count": total,
        "completed_call_count": completed_count,
        "remaining_call_count": total - completed_count,
        "next_call_order": next_call_order if completed_count < total else None,
        "answer_record_count": answer_count,
        "error_record_count": error_count,
        "gate_summary_sha256": next(iter(gate_hashes), None),
        "complete": completed_count == total,
    }


def require_frozen_holdout(
    schedule: Mapping[str, Any],
    *,
    dev_paths: Sequence[Path],
    corpus_index: Path,
    signoff_path: Path,
    repo_root: Path = REPO_ROOT,
    validator: Callable[..., dict[str, Any]] = validate_paths,
    git_identity: Callable[[Path], Mapping[str, Any]] = _git_identity,
) -> dict[str, Any]:
    """Require a non-draft holdout and all four validation gates."""

    cases_path = Path(schedule["cases_path"]).resolve()
    dev_paths = [Path(path).resolve() for path in dev_paths]
    corpus_index = Path(corpus_index).resolve()
    signoff_path = Path(signoff_path).resolve()
    repo_root = Path(repo_root).resolve()
    if "draft" in cases_path.name.casefold():
        raise ScheduleError(
            "final generation is forbidden for a draft holdout path"
        )
    if not dev_paths:
        raise ScheduleError("at least one DEV source manifest is required")
    git = dict(git_identity(repo_root))
    expected_git = {
        "commit": schedule["controls"]["shared"]["expected_git_commit"],
        "clean": True,
    }
    if git != expected_git:
        raise ScheduleError(
            "final generation requires the clean frozen local Git commit"
        )
    gate_input_binding_before = build_gate_input_binding(
        schedule,
        dev_paths=dev_paths,
        corpus_index=corpus_index,
        signoff_path=signoff_path,
        repo_root=repo_root,
    )
    summary = validator(
        cases_path,
        list(dev_paths),
        corpus_index=corpus_index,
        signoff_path=signoff_path,
        repo_root=repo_root,
    )
    required_gates = {"schema", "dev", "corpus", "signoff"}
    gates = summary.get("gates")
    if (
        summary.get("ok") is not True
        or not isinstance(gates, Mapping)
        or set(gates) != required_gates
        or any(
            not isinstance(gates[name], Mapping)
            or gates[name].get("ok") is not True
            for name in required_gates
        )
    ):
        errors = summary.get("errors") or ["unknown holdout gate failure"]
        raise ScheduleError(
            "frozen holdout preflight failed: " + "; ".join(map(str, errors))
        )
    if summary.get("cases_sha256") != schedule["cases_sha256"]:
        raise ScheduleError("holdout gate summary cases_sha256 mismatch")
    try:
        gate_input_binding = build_gate_input_binding(
            schedule,
            dev_paths=dev_paths,
            corpus_index=corpus_index,
            signoff_path=signoff_path,
            repo_root=repo_root,
        )
    except (OSError, ValueError) as exc:
        raise ScheduleError(
            "gate inputs changed while holdout validation was running"
        ) from exc
    if gate_input_binding != gate_input_binding_before:
        raise ScheduleError(
            "gate inputs changed while holdout validation was running"
        )
    result = dict(summary)
    result["git"] = git
    result["gate_input_binding"] = gate_input_binding
    result["gate_input_binding_sha256"] = sha256_json(gate_input_binding)
    return result


@dataclass(frozen=True)
class _GenerationAuthorizationValidator:
    """In-process authorization passed to the existing collector."""

    schedule: Mapping[str, Any]
    entry: Mapping[str, Any]
    artifact: Mapping[str, Any]
    gate_summary_sha256: str
    gate_input_binding: Mapping[str, Any]
    def authorize_evaluator_invocation(
        self,
        *,
        cases_path: Path,
        output_path: Path,
        errors_path: Path,
        journal_path: Path,
        experiment_id: str,
        condition_id: str,
        generation_run_id: str,
        selected_case_ids: Sequence[str],
        collector_controls: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate every CLI value and return immutable record metadata."""

        schedule = self.schedule
        entry = self.entry
        artifact = self.artifact
        if len(self.gate_summary_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.gate_summary_sha256
        ):
            raise ScheduleError("invalid authorization gate summary SHA-256")
        validate_gate_input_binding(schedule, self.gate_input_binding)
        cases_byte_sha, authorized_all_cases = _load_jsonl_snapshot(cases_path)
        if cases_byte_sha != schedule["cases_sha256"]:
            raise ScheduleError("final collector cases SHA-256 changed")
        if sha256_json(authorized_all_cases) != schedule["cases_canonical_sha256"]:
            raise ScheduleError("final collector canonical cases SHA-256 changed")
        expected = {
            "cases_path": Path(schedule["cases_path"]).resolve(),
            "output_path": Path(artifact["answers_path"]).resolve(),
            "errors_path": Path(artifact["errors_path"]).resolve(),
            "journal_path": Path(artifact["journal_path"]).resolve(),
            "experiment_id": schedule["experiment_id"],
            "condition_id": entry["condition_id"],
            "generation_run_id": entry["generation_run_id"],
            "selected_case_ids": [entry["case_id"]],
        }
        observed = {
            "cases_path": cases_path.resolve(),
            "output_path": output_path.resolve(),
            "errors_path": errors_path.resolve(),
            "journal_path": journal_path.resolve(),
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "generation_run_id": generation_run_id,
            "selected_case_ids": list(selected_case_ids),
        }
        for key, expected_value in expected.items():
            if observed[key] != expected_value:
                raise ScheduleError(
                    f"final collector invocation {key} mismatch: expected "
                    f"{expected_value!r}, got {observed[key]!r}"
                )

        shared = schedule["controls"]["shared"]
        condition = schedule["controls"]["conditions"][condition_id]
        expected_controls = {
            "api_base": condition["api_base"],
            "provider": shared["provider"],
            "model": shared["model"],
            "institution": shared["institution"],
            "context_k": shared["context_k"],
            "expected_generation_max_context_chars": shared[
                "max_context_chars"
            ],
            "expected_generation_max_output_tokens": shared[
                "max_output_tokens"
            ],
            "expected_generation_sampling_parameters": shared[
                "sampling_parameters"
            ],
            "parser_profile": shared["parser_profile"],
            "retrieval_mode": shared["retrieval_mode"],
            "expected_corpus_revision": shared["expected_corpus_revision"],
            "expected_retrieval_tuning": condition[
                "expected_retrieval_tuning"
            ],
            "expected_context_chunks_per_document": shared[
                "expected_context_chunks_per_document"
            ],
            "expected_git_commit": shared["expected_git_commit"],
            "expected_index_sha256": shared["expected_index_sha256"],
            "expected_source_manifest_sha256": shared[
                "expected_source_manifest_sha256"
            ],
            "eval_trace": shared["eval_trace"],
            "allow_unpinned": shared["allow_unpinned"],
            "max_attempts": shared["max_attempts"],
            "retry_backoff_seconds": shared["retry_backoff_seconds"],
            "request_timeout_seconds": shared["timeout_seconds"],
            "inter_call_sleep_seconds": shared["sleep_seconds"],
        }
        for key, expected_value in expected_controls.items():
            if collector_controls.get(key) != expected_value:
                raise ScheduleError(
                    f"final collector control {key} mismatch: expected "
                    f"{expected_value!r}, got {collector_controls.get(key)!r}"
                )

        authorized_cases = [
            case
            for case in authorized_all_cases
            if str(case.get("id") or "") == entry["case_id"]
        ]
        if len(authorized_cases) != 1:
            raise ScheduleError("authorized case payload is missing or duplicated")
        authorized_case = authorized_cases[0]
        if sha256_json(authorized_case) != entry["case_sha256"]:
            raise ScheduleError("authorized case payload SHA-256 mismatch")
        return {
            "collector": _expected_authorization_metadata(
                schedule,
                artifact,
                self.gate_summary_sha256,
                self.gate_input_binding,
            ),
            "record": _expected_record_schedule_metadata(schedule, entry),
            "artifact_expected_case_ids": list(artifact["expected_case_ids"]),
            "authorized_all_cases": authorized_all_cases,
            "authorized_case": authorized_case,
        }


def _issue_generation_authorization(**payload: Any) -> object:
    """Issue the exact shared, data-only generation authorization."""

    return _issue_exact_final_authorization("generation", dict(payload))


def validate_generation_authorization(
    authorization_payload: Mapping[str, Any], **observed: Any
) -> dict[str, Any]:
    """Trusted evaluator entrypoint; caller-supplied methods are never run."""
    validator = _GenerationAuthorizationValidator(**authorization_payload)
    schedule = validate_schedule_mapping(validator.schedule)
    binding = validator.gate_input_binding
    authentic_gate = require_frozen_holdout(
        schedule,
        dev_paths=[Path(item["path"]) for item in binding["dev_manifests"]],
        corpus_index=Path(binding["corpus_index"]["path"]),
        signoff_path=Path(binding["signoff"]["path"]),
    )
    authentic_gate_hash = sha256_json(authentic_gate)
    if validator.gate_summary_sha256 != authentic_gate_hash:
        raise ScheduleError(
            "generation authorization gate summary is not validator-authentic"
        )
    if dict(authentic_gate.get("gate_input_binding") or {}) != dict(binding):
        raise ScheduleError(
            "generation authorization gate input binding is not validator-authentic"
        )

    call_order = validator.entry.get("call_order")
    matching_entries = [
        entry for entry in schedule["calls"] if entry.get("call_order") == call_order
    ]
    if len(matching_entries) != 1 or dict(matching_entries[0]) != dict(
        validator.entry
    ):
        raise ScheduleError("generation authorization entry is not canonical")
    artifacts = _artifact_lookup(schedule)
    artifact_id = str(validator.artifact.get("artifact_id") or "")
    if artifact_id not in artifacts or artifacts[artifact_id] != dict(
        validator.artifact
    ):
        raise ScheduleError("generation authorization artifact is not canonical")

    progress = audit_schedule_artifacts(schedule, verify_gate_inputs=False)
    if progress.get("complete") is True or progress.get("next_call_order") != call_order:
        raise ScheduleError(
            "generation authorization is not for the next scheduled call"
        )
    existing_gate_hash = progress.get("gate_summary_sha256")
    if existing_gate_hash not in (None, authentic_gate_hash):
        raise ScheduleError(
            "generation artifacts are bound to a different holdout gate"
        )
    return validator.authorize_evaluator_invocation(**observed)


def build_collector_argv(
    schedule: Mapping[str, Any], entry: Mapping[str, Any]
) -> list[str]:
    artifacts = _artifact_lookup(schedule)
    artifact = artifacts[entry["artifact_id"]]
    shared = schedule["controls"]["shared"]
    condition = schedule["controls"]["conditions"][entry["condition_id"]]
    return [
        "--cases",
        schedule["cases_path"],
        "--api-base",
        condition["api_base"],
        "--out",
        artifact["answers_path"],
        "--errors-out",
        artifact["errors_path"],
        "--attempt-journal",
        artifact["journal_path"],
        "--experiment-id",
        schedule["experiment_id"],
        "--condition-id",
        entry["condition_id"],
        "--generation-run-id",
        entry["generation_run_id"],
        "--provider",
        shared["provider"],
        "--model",
        shared["model"],
        "--institution",
        "none",
        "--context-k",
        str(shared["context_k"]),
        "--parser-profile",
        shared["parser_profile"],
        "--retrieval-mode",
        shared["retrieval_mode"],
        "--expected-corpus-revision",
        shared["expected_corpus_revision"],
        "--expected-retrieval-tuning",
        "on" if condition["expected_retrieval_tuning"] else "off",
        "--expected-context-chunks-per-document",
        str(shared["expected_context_chunks_per_document"]),
        "--expected-generation-max-context-chars",
        str(shared["max_context_chars"]),
        "--expected-generation-max-output-tokens",
        str(shared["max_output_tokens"]),
        "--expected-generation-sampling",
        "absent",
        "--expected-git-commit",
        shared["expected_git_commit"],
        "--expected-index-sha256",
        shared["expected_index_sha256"],
        "--expected-source-manifest-sha256",
        shared["expected_source_manifest_sha256"],
        "--eval-trace",
        "--only",
        entry["case_id"],
        "--allow-partial",
        "--sleep",
        str(shared["sleep_seconds"]),
        "--timeout",
        str(shared["timeout_seconds"]),
        "--max-attempts",
        str(shared["max_attempts"]),
        "--retry-backoff",
        str(shared["retry_backoff_seconds"]),
    ]


def _revalidate_execute_gates(
    schedule: Mapping[str, Any], dev_paths: Sequence[Path],
    corpus_index: Path, signoff_path: Path,
) -> dict[str, Any]:
    return require_frozen_holdout(
        schedule, dev_paths=dev_paths, corpus_index=corpus_index,
        signoff_path=signoff_path,
    )


def execute_schedule(
    schedule: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
    *,
    dry_run: bool,
    collector_main: Callable[..., Any] | None = None,
    audit_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    dev_paths: Sequence[Path] | None = None,
    corpus_index: Path | None = None,
    signoff_path: Path | None = None,
) -> dict[str, Any]:
    """Resume a schedule, allowing exactly the next global call each time."""

    if collector_main is None and not dry_run:
        from evaluate_service_answers import main as collector_main  # type: ignore
    if audit_fn is None:
        audit_fn = lambda value: audit_schedule_artifacts(
            value, verify_gate_inputs=False
        )

    if not dev_paths or corpus_index is None or signoff_path is None:
        raise ScheduleError("execute_schedule requires gate input paths")
    verified_summary = _revalidate_execute_gates(
        schedule, dev_paths, corpus_index, signoff_path,
    )
    if dict(verified_summary) != dict(gate_summary):
        raise ScheduleError("programmatic gate summary is not validator-authentic")
    required_gates = {"schema", "dev", "corpus", "signoff"}
    gates = gate_summary.get("gates")
    if (
        gate_summary.get("ok") is not True
        or not isinstance(gates, Mapping)
        or set(gates) != required_gates
        or any(
            not isinstance(gates[name], Mapping)
            or gates[name].get("ok") is not True
            for name in required_gates
        )
        or gate_summary.get("cases_sha256") != schedule["cases_sha256"]
    ):
        raise ScheduleError("execute_schedule requires exact four-gate PASS")
    gate_hash = sha256_json(gate_summary)
    gate_input_binding = gate_summary.get("gate_input_binding")
    if not isinstance(gate_input_binding, Mapping):
        raise ScheduleError("gate summary omitted byte-level input binding")
    if gate_summary.get("gate_input_binding_sha256") != sha256_json(
        gate_input_binding
    ):
        raise ScheduleError("gate summary input binding SHA-256 mismatch")
    validate_gate_input_binding(schedule, gate_input_binding)
    artifacts = _artifact_lookup(schedule)
    calls = schedule["calls"]
    if dry_run:
        audit = audit_fn(schedule)
        existing_gate_hash = audit.get("gate_summary_sha256")
        if existing_gate_hash not in (None, gate_hash):
            raise ScheduleError(
                "existing artifacts are bound to a different holdout gate"
            )
        start = audit["completed_call_count"]
        for entry in calls[start:]:
            command = ["python3", "-B", str(SCRIPT_PATH)] + build_collector_argv(
                schedule, entry
            )
            print(
                json.dumps(
                    {
                        "call_order": entry["call_order"],
                        "case_id": entry["case_id"],
                        "condition_id": entry["condition_id"],
                        "generation_run_id": entry["generation_run_id"],
                        "collector_argv_preview": shlex.join(command),
                        "standalone_executable": False,
                        "internal_final_authorization_required": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return audit

    assert collector_main is not None
    lock_path = Path(schedule["output_root"]) / ".final-generation-run.lock"
    with exclusive_run_lock(
        lock_path,
        {
            "kind": "final_generation",
            "schedule_id": schedule["schedule_id"],
            "schedule_sha256": schedule["schedule_sha256"],
        },
    ):
        while True:
            before = audit_fn(schedule)
            existing_gate_hash = before.get("gate_summary_sha256")
            if existing_gate_hash not in (None, gate_hash):
                raise ScheduleError(
                    "existing artifacts are bound to a different holdout gate"
                )
            if before["complete"]:
                return before
            call_order = before["next_call_order"]
            entry = calls[call_order - 1]
            if entry["call_order"] != call_order:
                raise ScheduleError("schedule call_order is not contiguous")
            artifact = artifacts[entry["artifact_id"]]
            authorization = _issue_generation_authorization(
                schedule=schedule,
                entry=entry,
                artifact=artifact,
                gate_summary_sha256=gate_hash,
                gate_input_binding=gate_input_binding,
            )
            argv = build_collector_argv(schedule, entry)
            try:
                collector_main(argv, final_authorization=authorization)
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise ScheduleError(
                        f"collector stopped at call_order {call_order} "
                        f"with exit code {exc.code}"
                    ) from exc
            after = audit_fn(schedule)
            if (
                after["completed_call_count"]
                != before["completed_call_count"] + 1
            ):
                raise ScheduleError(
                    f"collector did not append exactly call_order {call_order}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--expected-corpus-revision")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--model")
    parser.add_argument("--c0-api-base")
    parser.add_argument("--c1-api-base")
    parser.add_argument("--dev-manifest", type=Path, action="append")
    parser.add_argument("--corpus-index", type=Path)
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--authorize-final-collection",
        action="store_true",
        help="required acknowledgement before any service/API call",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.create:
            required = {
                "--cases": args.cases,
                "--experiment-id": args.experiment_id,
                "--output-root": args.output_root,
                "--expected-corpus-revision": args.expected_corpus_revision,
                "--expected-git-commit": args.expected_git_commit,
                "--expected-index-sha256": args.expected_index_sha256,
                "--expected-source-manifest-sha256": (
                    args.expected_source_manifest_sha256
                ),
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                parser.error("--create requires " + ", ".join(missing))
            if args.dry_run or args.authorize_final_collection:
                parser.error(
                    "--dry-run/--authorize-final-collection are not valid with --create"
                )
            if args.dev_manifest or args.corpus_index or args.signoff:
                parser.error(
                    "holdout gate inputs are only valid with --run"
                )
            cases = load_jsonl(args.cases)
            schedule = build_schedule(
                args.cases,
                cases,
                experiment_id=args.experiment_id,
                output_root=args.output_root,
                expected_corpus_revision=args.expected_corpus_revision,
                expected_git_commit=args.expected_git_commit,
                expected_index_sha256=args.expected_index_sha256,
                expected_source_manifest_sha256=(
                    args.expected_source_manifest_sha256
                ),
                model=args.model or DEFAULT_MODEL,
                c0_api_base=args.c0_api_base or DEFAULT_C0_API_BASE,
                c1_api_base=args.c1_api_base or DEFAULT_C1_API_BASE,
            )
            write_new_schedule(args.schedule, schedule)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "schedule_path": str(args.schedule),
                        "schedule_id": schedule["schedule_id"],
                        "schedule_sha256": schedule["schedule_sha256"],
                        **schedule["summary"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if (
            args.cases
            or args.experiment_id
            or args.output_root
            or args.expected_corpus_revision
            or args.expected_git_commit
            or args.expected_index_sha256
            or args.expected_source_manifest_sha256
            or args.model
            or args.c0_api_base
            or args.c1_api_base
        ):
            parser.error(
                "case, experiment, output, and service controls are read from "
                "an existing schedule"
            )
        schedule = validate_schedule(args.schedule)
        if args.check:
            if args.dry_run or args.authorize_final_collection:
                parser.error(
                    "--dry-run/--authorize-final-collection are not valid with --check"
                )
            if args.dev_manifest or args.corpus_index or args.signoff:
                parser.error("holdout gate inputs are only valid with --run")
            audit = audit_schedule_artifacts(schedule)
            print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
            return 0

        if not args.dev_manifest or args.corpus_index is None or args.signoff is None:
            parser.error(
                "--run requires --dev-manifest, --corpus-index, and --signoff"
            )
        if not args.dry_run and not args.authorize_final_collection:
            parser.error(
                "live --run requires explicit --authorize-final-collection"
            )
        gate_summary = require_frozen_holdout(
            schedule,
            dev_paths=args.dev_manifest,
            corpus_index=args.corpus_index,
            signoff_path=args.signoff,
        )
        result = execute_schedule(
            schedule, gate_summary, dry_run=args.dry_run,
            dev_paths=args.dev_manifest,
            corpus_index=args.corpus_index,
            signoff_path=args.signoff,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, ScheduleError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
