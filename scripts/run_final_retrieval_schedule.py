#!/usr/bin/env python3
"""Create, gate, audit, and execute the 108-call final retrieval schedule.

The four frozen lanes are PAR-B, PAR-CH, C0, and C1.  Collection is always
case-major, one authorized holdout Core case at a time, with the local
``extractive`` provider.  Creating, checking, and dry-running a schedule never
calls the service.  Live execution still requires an explicit authorization
flag and the shared collector capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = REPO_ROOT / "scripts" / "evaluate_service_answers.py"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from holdout_gold import CORE_SPLIT, load_jsonl, sha256_file  # noqa: E402
from immutable_outputs import exclusive_run_lock  # noqa: E402
import build_holdout_signoff as signoff_artifacts  # noqa: E402
from service_eval_artifacts import (  # noqa: E402
    _issue_exact_final_authorization,
    load_unique_jsonl,
    sha256_json,
    validate_answer_record,
    validate_final_retrieval_provenance,
)
from validate_service_holdout import (  # noqa: E402
    _read_repo_review_artifact,
    validate_paths,
)

SCHEDULE_SCHEMA_VERSION = "pnu.final-retrieval-schedule.v1"
AUTHORIZATION_SCHEMA_VERSION = "pnu.final-retrieval-authorization.v1"
GATE_SCHEMA_VERSION = "pnu.final-retrieval-gate-inputs.v1"
LANE_IDS = ("par-b", "par-ch", "c0", "c1")
LANE_PROFILES = {
    "par-b": "baseline",
    "par-ch": "challenger",
    "c0": "cascade",
    "c1": "cascade",
}
LANE_TUNING = {"par-b": True, "par-ch": True, "c0": False, "c1": True}
EXPECTED_CORE_COUNT = 27
EXPECTED_CALL_COUNT = 108
RUN_ID = "retrieval-once"


class RetrievalScheduleError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonempty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RetrievalScheduleError(f"{label} must be non-empty")
    return text


def _file_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RetrievalScheduleError(f"not a file: {resolved}")
    data = resolved.read_bytes()
    return {"path": str(resolved), "size_bytes": len(data), "sha256": _sha(data)}


def _payload_snapshot(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": _sha(payload),
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
        raise RetrievalScheduleError(
            f"cannot bind sign-off manifest: {exc}"
        ) from exc
    expected_manifest_keys = {
        "schema_version",
        "cases_sha256",
        "review_packet_path",
        "review_packet_sha256",
        "review_response_sha256s",
        "case_signoffs",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_manifest_keys:
        raise RetrievalScheduleError("sign-off manifest schema is not exact")
    if payload.get("schema_version") != signoff_artifacts.SIGNOFF_SCHEMA_VERSION:
        raise RetrievalScheduleError(
            "sign-off manifest schema_version is not supported"
        )
    if not isinstance(payload.get("case_signoffs"), list):
        raise RetrievalScheduleError(
            "sign-off manifest case_signoffs must be a list"
        )

    try:
        packet_path, packet_payload = _read_repo_review_artifact(
            repo_root,
            payload.get("review_packet_path"),
            label="sign-off review packet",
        )
    except (OSError, ValueError) as exc:
        raise RetrievalScheduleError(
            f"cannot bind sign-off review packet: {exc}"
        ) from exc
    if not packet_payload:
        raise RetrievalScheduleError("sign-off review packet must not be empty")
    packet_snapshot = _payload_snapshot(packet_path, packet_payload)
    if payload.get("review_packet_sha256") != packet_snapshot["sha256"]:
        raise RetrievalScheduleError("sign-off review packet SHA-256 mismatch")

    raw_responses = payload.get("review_response_sha256s")
    if not isinstance(raw_responses, list) or len(raw_responses) != 2:
        raise RetrievalScheduleError(
            "sign-off must bind exactly two review responses"
        )
    response_snapshots: list[dict[str, Any]] = []
    review_paths: list[tuple[str, Path]] = [
        ("sign-off", signoff_path.resolve()),
        ("review packet", packet_path),
    ]
    for index, expected_slot in enumerate(("A", "B")):
        record = raw_responses[index]
        expected_record_keys = {"reviewer_slot", "reviewer_id", "sha256", "path"}
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise RetrievalScheduleError(
                f"sign-off Reviewer {expected_slot} provenance schema is not exact"
            )
        if record.get("reviewer_slot") != expected_slot:
            raise RetrievalScheduleError(
                "sign-off response slots must be exactly A then B"
            )
        reviewer_id = record.get("reviewer_id")
        if (
            not isinstance(reviewer_id, str)
            or not reviewer_id
            or reviewer_id != reviewer_id.strip()
        ):
            raise RetrievalScheduleError(
                f"sign-off Reviewer {expected_slot} reviewer_id is invalid"
            )
        try:
            response_path, response_payload = _read_repo_review_artifact(
                repo_root,
                record.get("path"),
                label=f"sign-off Reviewer {expected_slot} response",
            )
        except (OSError, ValueError) as exc:
            raise RetrievalScheduleError(
                f"cannot bind Reviewer {expected_slot} response: {exc}"
            ) from exc
        response_snapshot = {
            "reviewer_slot": expected_slot,
            "reviewer_id": reviewer_id,
            **_payload_snapshot(response_path, response_payload),
        }
        if record.get("sha256") != response_snapshot["sha256"]:
            raise RetrievalScheduleError(
                f"sign-off Reviewer {expected_slot} response SHA-256 mismatch"
            )
        response_snapshots.append(response_snapshot)
        review_paths.append((f"Reviewer {expected_slot} response", response_path))

    for index, (left_label, left_path) in enumerate(review_paths):
        for right_label, right_path in review_paths[index + 1 :]:
            if signoff_artifacts.paths_alias(left_path, right_path):
                raise RetrievalScheduleError(
                    f"{left_label} aliases {right_label}: {left_path}"
                )
        for protected_label, protected_path in protected_paths:
            if signoff_artifacts.paths_alias(left_path, protected_path):
                raise RetrievalScheduleError(
                    f"{left_label} aliases {protected_label}: {left_path}"
                )

    return {
        "signoff": _payload_snapshot(signoff_path, signoff_payload),
        "review_packet": packet_snapshot,
        "review_responses": response_snapshots,
    }


def _index_meta(path: Path) -> dict[str, str]:
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT key, value FROM index_meta").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise RetrievalScheduleError(f"cannot read index_meta from {path}: {exc}")
    return {str(key): str(value) for key, value in rows}


def _reject_sqlite_sidecars(path: Path) -> None:
    sidecars = [Path(str(path.resolve()) + suffix) for suffix in ("-wal", "-shm", "-journal")]
    present = [str(value) for value in sidecars if value.exists()]
    if present:
        raise RetrievalScheduleError("SQLite snapshot has live sidecars: " + ", ".join(present))


def _schedule_sha(schedule: Mapping[str, Any]) -> str:
    value = dict(schedule)
    value.pop("schedule_id", None)
    value.pop("schedule_sha256", None)
    return sha256_json(value)


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=normal")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RetrievalScheduleError(f"cannot verify Git freeze: {exc}") from exc
    return {"commit": commit, "clean": status == ""}


def build_schedule(
    cases_path: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    output_root: Path,
    expected_git_commit: str,
    expected_source_manifest_sha256: str,
    lane_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if "draft" in cases_path.name.casefold():
        raise RetrievalScheduleError("final retrieval forbids a draft holdout path")
    core = [dict(case) for case in cases if case.get("split") == CORE_SPLIT]
    if len(core) != EXPECTED_CORE_COUNT:
        raise RetrievalScheduleError("final retrieval requires exactly 27 Core cases")
    ids = [str(case.get("id") or "") for case in core]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RetrievalScheduleError("Core case IDs are empty or duplicated")
    source_sha = _nonempty(expected_source_manifest_sha256, "source manifest SHA-256")
    if len(source_sha) != 64:
        raise RetrievalScheduleError("source manifest SHA-256 must have length 64")

    lanes: dict[str, Any] = {}
    for lane_id in LANE_IDS:
        raw = lane_inputs.get(lane_id)
        if not isinstance(raw, Mapping):
            raise RetrievalScheduleError(f"missing lane {lane_id}")
        index_path = Path(_nonempty(raw.get("index_path"), f"{lane_id}.index_path"))
        _reject_sqlite_sidecars(index_path)
        snapshot = _file_snapshot(index_path)
        metadata = _index_meta(index_path)
        profile = LANE_PROFILES[lane_id]
        revision = _nonempty(raw.get("corpus_revision"), f"{lane_id}.corpus_revision")
        if metadata.get("profile") != profile:
            raise RetrievalScheduleError(f"{lane_id} index profile mismatch")
        if metadata.get("corpus_revision") != revision:
            raise RetrievalScheduleError(f"{lane_id} index revision mismatch")
        if metadata.get("source_manifest_sha256") != source_sha:
            raise RetrievalScheduleError(f"{lane_id} source manifest mismatch")
        lanes[lane_id] = {
            "condition_id": lane_id,
            "parser_profile": profile,
            "retrieval_mode": "bm25",
            "expected_retrieval_tuning": LANE_TUNING[lane_id],
            "api_base": _nonempty(raw.get("api_base"), f"{lane_id}.api_base").rstrip("/"),
            "corpus_revision": revision,
            "index": snapshot,
            "source_manifest_sha256": source_sha,
        }
    if lanes["c0"]["index"]["sha256"] != lanes["c1"]["index"]["sha256"]:
        raise RetrievalScheduleError(
            "C0/C1 must use the exact same cascade index snapshot"
        )

    artifacts = []
    for lane_id in LANE_IDS:
        prefix = output_root / "retrieval" / lane_id
        artifacts.append(
            {
                "artifact_id": lane_id,
                "condition_id": lane_id,
                "answers_path": str(prefix.with_suffix(".answers.jsonl")),
                "errors_path": str(prefix.with_suffix(".errors.jsonl")),
                "journal_path": str(prefix.with_suffix(".journal.jsonl")),
                "expected_case_ids": ids,
                "expected_case_ids_sha256": sha256_json(ids),
                "expected_case_count": len(ids),
            }
        )
    calls = []
    call_order = 0
    for case in core:
        for position, lane_id in enumerate(LANE_IDS, start=1):
            call_order += 1
            calls.append(
                {
                    "call_order": call_order,
                    "case_id": case["id"],
                    "case_sha256": sha256_json(case),
                    "family_id": case["family_id"],
                    "condition_id": lane_id,
                    "condition_position": position,
                    "artifact_id": lane_id,
                    "generation_run_id": RUN_ID,
                }
            )
    unsigned = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "experiment_id": _nonempty(experiment_id, "experiment_id"),
        "cases_path": str(cases_path.resolve()),
        "cases_sha256": sha256_file(cases_path),
        "cases_canonical_sha256": sha256_json(list(cases)),
        "selected_case_ids_sha256": sha256_json(ids),
        "output_root": str(output_root.resolve()),
        "expected_git_commit": _nonempty(expected_git_commit, "expected_git_commit"),
        "shared_source_manifest_sha256": source_sha,
        "controls": {
            "provider": "extractive",
            "model": None,
            "institution": None,
            "context_k": 8,
            "context_chunks_per_document": 2,
            "eval_trace": True,
            "allow_unpinned": False,
            "max_attempts": 3,
            "retry_backoff_seconds": 2.0,
            "timeout_seconds": 180.0,
            "sleep_seconds": 0.0,
        },
        "lanes": lanes,
        "artifacts": artifacts,
        "calls": calls,
        "summary": {"core_cases": len(core), "lanes": 4, "calls": len(calls)},
    }
    if len(calls) != EXPECTED_CALL_COUNT:
        raise RetrievalScheduleError("retrieval call count is not 108")
    digest = _schedule_sha(unsigned)
    return {**unsigned, "schedule_id": f"final_retrieval_{digest[:24]}", "schedule_sha256": digest}


def write_new_schedule(path: Path, schedule: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(schedule, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RetrievalScheduleError("schedule already exists; overwrite forbidden") from exc


def validate_schedule_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a retrieval schedule and reject non-canonical in-memory plans."""

    if not isinstance(value, dict) or value.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise RetrievalScheduleError("invalid retrieval schedule")
    cases_path = Path(_nonempty(value.get("cases_path"), "cases_path"))
    cases = load_jsonl(cases_path)
    if sha256_file(cases_path) != value.get("cases_sha256"):
        raise RetrievalScheduleError("scheduled cases byte SHA-256 changed")
    if sha256_json(cases) != value.get("cases_canonical_sha256"):
        raise RetrievalScheduleError("scheduled canonical cases SHA-256 changed")
    lanes = value.get("lanes")
    if not isinstance(lanes, Mapping) or set(lanes) != set(LANE_IDS):
        raise RetrievalScheduleError("retrieval schedule lanes are incomplete")
    lane_inputs: dict[str, dict[str, Any]] = {}
    for lane_id in LANE_IDS:
        lane = lanes.get(lane_id)
        if not isinstance(lane, Mapping):
            raise RetrievalScheduleError(f"invalid scheduled lane {lane_id}")
        index = lane.get("index")
        if not isinstance(index, Mapping):
            raise RetrievalScheduleError(f"scheduled lane {lane_id} index missing")
        lane_inputs[lane_id] = {
            "api_base": lane.get("api_base"),
            "index_path": index.get("path"),
            "corpus_revision": lane.get("corpus_revision"),
        }
    expected = build_schedule(
        cases_path,
        cases,
        experiment_id=_nonempty(value.get("experiment_id"), "experiment_id"),
        output_root=Path(_nonempty(value.get("output_root"), "output_root")),
        expected_git_commit=_nonempty(
            value.get("expected_git_commit"), "expected_git_commit"
        ),
        expected_source_manifest_sha256=_nonempty(
            value.get("shared_source_manifest_sha256"),
            "shared_source_manifest_sha256",
        ),
        lane_inputs=lane_inputs,
    )
    if dict(value) != expected:
        raise RetrievalScheduleError(
            "retrieval schedule is not the canonical 27x4 plan"
        )
    return dict(value)


def load_schedule(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetrievalScheduleError("retrieval schedule must be an object")
    return validate_schedule_mapping(value)


def build_gate_binding(
    schedule: Mapping[str, Any],
    *,
    dev_paths: Sequence[Path],
    signoff_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if not dev_paths:
        raise RetrievalScheduleError("at least one DEV manifest is required")
    indexes = {}
    for lane_id, lane in schedule["lanes"].items():
        index_path = Path(lane["index"]["path"])
        _reject_sqlite_sidecars(index_path)
        snapshot = _file_snapshot(index_path)
        if snapshot != lane["index"]:
            raise RetrievalScheduleError(f"{lane_id} index snapshot changed")
        metadata = _index_meta(index_path)
        for key, expected in {
            "profile": lane["parser_profile"],
            "corpus_revision": lane["corpus_revision"],
            "source_manifest_sha256": schedule["shared_source_manifest_sha256"],
        }.items():
            if metadata.get(key) != expected:
                raise RetrievalScheduleError(f"{lane_id} index metadata {key} changed")
        indexes[lane_id] = snapshot
    resolved_repo_root = repo_root.resolve()
    signoff_dependencies = _signoff_dependency_binding(
        signoff_path,
        repo_root=resolved_repo_root,
        protected_paths=[
            ("cases", Path(schedule["cases_path"])),
            *[("DEV manifest", path) for path in dev_paths],
            *[
                (f"{lane_id} index", Path(lane["index"]["path"]))
                for lane_id, lane in schedule["lanes"].items()
            ],
        ],
    )
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "repo_root": str(resolved_repo_root),
        "cases": _file_snapshot(Path(schedule["cases_path"])),
        "dev_manifests": [_file_snapshot(path) for path in sorted(dev_paths, key=str)],
        "signoff": signoff_dependencies["signoff"],
        "review_packet": signoff_dependencies["review_packet"],
        "review_responses": signoff_dependencies["review_responses"],
        "indexes": indexes,
    }


def require_final_gates(
    schedule: Mapping[str, Any],
    *,
    dev_paths: Sequence[Path],
    signoff_path: Path,
    repo_root: Path = REPO_ROOT,
    validator: Callable[..., Mapping[str, Any]] = validate_paths,
    git_identity: Callable[[Path], Mapping[str, Any]] = _git_identity,
) -> dict[str, Any]:
    dev_paths = [Path(path).resolve() for path in dev_paths]
    signoff_path = Path(signoff_path).resolve()
    repo_root = Path(repo_root).resolve()
    if "draft" in Path(schedule["cases_path"]).name.casefold():
        raise RetrievalScheduleError("final retrieval forbids a draft holdout path")
    before = build_gate_binding(
        schedule,
        dev_paths=dev_paths,
        signoff_path=signoff_path,
        repo_root=repo_root,
    )
    if before["cases"]["sha256"] != schedule["cases_sha256"]:
        raise RetrievalScheduleError("cases snapshot changed")
    git = dict(git_identity(repo_root))
    if git != {"commit": schedule["expected_git_commit"], "clean": True}:
        raise RetrievalScheduleError("final retrieval requires the clean frozen Git commit")
    gate_summaries = {}
    for profile in ("baseline", "challenger", "cascade"):
        lane_id = next(key for key, lane in schedule["lanes"].items() if lane["parser_profile"] == profile)
        summary = validator(
            Path(schedule["cases_path"]),
            list(dev_paths),
            corpus_index=Path(schedule["lanes"][lane_id]["index"]["path"]),
            signoff_path=signoff_path,
            repo_root=repo_root,
        )
        gates = summary.get("gates") if isinstance(summary, Mapping) else None
        required = {"schema", "dev", "corpus", "signoff"}
        if (
            summary.get("ok") is not True
            or not isinstance(gates, Mapping)
            or set(gates) != required
            or any(gates[name].get("ok") is not True for name in required)
        ):
            raise RetrievalScheduleError(f"{profile} four-gate validation failed")
        gate_summaries[profile] = dict(summary)
    try:
        after = build_gate_binding(
            schedule,
            dev_paths=dev_paths,
            signoff_path=signoff_path,
            repo_root=repo_root,
        )
    except (OSError, ValueError) as exc:
        raise RetrievalScheduleError(
            "gate inputs changed during validation (TOCTOU)"
        ) from exc
    if after != before:
        raise RetrievalScheduleError("gate inputs changed during validation (TOCTOU)")
    result = {"ok": True, "inputs": before, "profiles": gate_summaries, "git": git}
    result["gate_summary_sha256"] = sha256_json(result)
    return result


def validate_gate_binding(
    schedule: Mapping[str, Any], gate_inputs: Mapping[str, Any]
) -> None:
    """Re-hash every frozen gate input; call before every logical slot."""

    if gate_inputs.get("schema_version") != GATE_SCHEMA_VERSION:
        raise RetrievalScheduleError("invalid stored gate input binding")
    repo_root = gate_inputs.get("repo_root")
    cases = gate_inputs.get("cases")
    signoff = gate_inputs.get("signoff")
    packet = gate_inputs.get("review_packet")
    responses = gate_inputs.get("review_responses")
    dev_manifests = gate_inputs.get("dev_manifests")
    indexes = gate_inputs.get("indexes")
    if (
        not isinstance(repo_root, str)
        or not Path(repo_root).is_absolute()
        or not isinstance(cases, Mapping)
        or not isinstance(signoff, Mapping)
        or not isinstance(packet, Mapping)
        or not isinstance(responses, list)
        or len(responses) != 2
        or any(not isinstance(value, Mapping) for value in responses)
        or not isinstance(dev_manifests, list)
        or not dev_manifests
        or any(not isinstance(value, Mapping) for value in dev_manifests)
        or not isinstance(indexes, Mapping)
    ):
        raise RetrievalScheduleError("stored gate snapshot is malformed")
    if cases.get("sha256") != schedule["cases_sha256"]:
        raise RetrievalScheduleError("stored gate cases do not match schedule")
    if set(indexes) != set(schedule["lanes"]):
        raise RetrievalScheduleError("stored gate indexes do not match schedule")
    for lane_id, expected in indexes.items():
        if expected != schedule["lanes"][lane_id]["index"]:
            raise RetrievalScheduleError("stored gate index does not match schedule")
    try:
        rebuilt = build_gate_binding(
            schedule,
            dev_paths=[
                Path(str(value.get("path") or "")) for value in dev_manifests
            ],
            signoff_path=Path(str(signoff.get("path") or "")),
            repo_root=Path(repo_root),
        )
    except (OSError, ValueError) as exc:
        raise RetrievalScheduleError(
            "frozen gate input changed after sign-off"
        ) from exc
    if dict(gate_inputs) != rebuilt:
        raise RetrievalScheduleError("frozen gate input changed after sign-off")


def _artifact_map(schedule: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(value["artifact_id"]): value for value in schedule["artifacts"]}


def _authorization_metadata(
    schedule: Mapping[str, Any], artifact: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    binding = gate["inputs"]
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "collection_purpose": "final_retrieval",
        "schedule_id": schedule["schedule_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "cases_sha256": schedule["cases_sha256"],
        "cases_canonical_sha256": schedule["cases_canonical_sha256"],
        "gate_summary_sha256": gate["gate_summary_sha256"],
        "gate_input_binding": binding,
        "gate_input_binding_sha256": sha256_json(binding),
        "artifact_id": artifact["artifact_id"],
        "split": CORE_SPLIT,
        "expected_case_count": artifact["expected_case_count"],
        "expected_case_ids_sha256": artifact["expected_case_ids_sha256"],
        "shared_source_manifest_sha256": schedule["shared_source_manifest_sha256"],
    }


def _record_metadata(schedule: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "collection_purpose": "final_retrieval",
        "schedule_id": schedule["schedule_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "call_order": entry["call_order"],
        "phase": "final_retrieval",
        "split": CORE_SPLIT,
        "artifact_id": entry["artifact_id"],
        "family_id": entry["family_id"],
        "condition_position": entry["condition_position"],
    }


@dataclass(frozen=True)
class _RetrievalAuthorizationValidator:
    schedule: Mapping[str, Any]
    entry: Mapping[str, Any]
    artifact: Mapping[str, Any]
    gate: Mapping[str, Any]
    def authorize_evaluator_invocation(self, **observed: Any) -> dict[str, Any]:
        schedule, entry, artifact = self.schedule, self.entry, self.artifact
        validate_gate_binding(schedule, self.gate.get("inputs") or {})
        expected = {
            "cases_path": Path(schedule["cases_path"]).resolve(),
            "output_path": Path(artifact["answers_path"]).resolve(),
            "errors_path": Path(artifact["errors_path"]).resolve(),
            "journal_path": Path(artifact["journal_path"]).resolve(),
            "experiment_id": schedule["experiment_id"],
            "condition_id": entry["condition_id"],
            "generation_run_id": RUN_ID,
            "selected_case_ids": [entry["case_id"]],
        }
        for key, value in expected.items():
            actual = observed.get(key)
            if key.endswith("_path"):
                actual = Path(actual).resolve()
            if actual != value:
                raise RetrievalScheduleError(f"authorized invocation {key} mismatch")
        lane, controls = schedule["lanes"][entry["condition_id"]], schedule["controls"]
        expected_controls = {
            "api_base": lane["api_base"], "provider": "extractive", "model": None,
            "institution": None, "context_k": 8,
            "parser_profile": lane["parser_profile"], "retrieval_mode": "bm25",
            "expected_corpus_revision": lane["corpus_revision"],
            "expected_retrieval_tuning": lane["expected_retrieval_tuning"],
            "expected_context_chunks_per_document": 2,
            "expected_generation_max_context_chars": None,
            "expected_generation_max_output_tokens": None,
            "expected_generation_sampling_parameters": None,
            "expected_git_commit": schedule["expected_git_commit"],
            "expected_index_sha256": lane["index"]["sha256"],
            "expected_source_manifest_sha256": schedule["shared_source_manifest_sha256"],
            "eval_trace": True, "allow_unpinned": False,
            "max_attempts": controls["max_attempts"],
            "retry_backoff_seconds": controls["retry_backoff_seconds"],
            "request_timeout_seconds": controls["timeout_seconds"],
            "inter_call_sleep_seconds": controls["sleep_seconds"],
        }
        actual_controls = observed.get("collector_controls")
        if not isinstance(actual_controls, Mapping):
            raise RetrievalScheduleError("collector controls missing")
        for key, value in expected_controls.items():
            if actual_controls.get(key) != value:
                raise RetrievalScheduleError(f"retrieval collector control {key} mismatch")
        cases_bytes = Path(schedule["cases_path"]).read_bytes()
        all_cases = load_jsonl(Path(schedule["cases_path"]))
        if _sha(cases_bytes) != schedule["cases_sha256"] or sha256_json(all_cases) != schedule["cases_canonical_sha256"]:
            raise RetrievalScheduleError("authorized cases snapshot changed")
        selected = [case for case in all_cases if case.get("id") == entry["case_id"]]
        if len(selected) != 1 or sha256_json(selected[0]) != entry["case_sha256"]:
            raise RetrievalScheduleError("authorized case changed")
        return {
            "collector": _authorization_metadata(schedule, artifact, self.gate),
            "record": _record_metadata(schedule, entry),
            "artifact_expected_case_ids": list(artifact["expected_case_ids"]),
            "authorized_all_cases": all_cases,
            "authorized_case": selected[0],
        }


def _issue_retrieval_authorization(*args: Any, **payload: Any) -> object:
    if args:
        if len(args) != 4 or payload:
            raise TypeError("expected schedule, entry, artifact, gate")
        payload = dict(zip(("schedule", "entry", "artifact", "gate"), args))
    return _issue_exact_final_authorization("retrieval", dict(payload))


def validate_retrieval_authorization(
    authorization_payload: Mapping[str, Any], **observed: Any
) -> dict[str, Any]:
    """Trusted evaluator entrypoint; re-run the semantic gates before use.

    Re-hashing the stored inputs is necessary but not sufficient: a caller could
    otherwise issue an exact data authorization around a freshly hashed sign-off
    file whose contents were never approved by ``validate_paths``, or around a
    dirty Git checkout.  Recover the frozen input paths from the sealed payload,
    execute the canonical four-gate validator again, and require byte-for-byte
    equality with the gate summary bound into the schedule authorization.
    """

    validator = _RetrievalAuthorizationValidator(**authorization_payload)
    schedule = validate_schedule_mapping(validator.schedule)
    gate = validator.gate
    inputs = gate.get("inputs") if isinstance(gate, Mapping) else None
    if not isinstance(inputs, Mapping):
        raise RetrievalScheduleError("retrieval authorization gate inputs missing")
    dev_manifests = inputs.get("dev_manifests")
    signoff = inputs.get("signoff")
    if (
        not isinstance(dev_manifests, list)
        or not dev_manifests
        or any(
            not isinstance(item, Mapping) or not str(item.get("path") or "").strip()
            for item in dev_manifests
        )
        or not isinstance(signoff, Mapping)
        or not str(signoff.get("path") or "").strip()
    ):
        raise RetrievalScheduleError("retrieval authorization gate paths malformed")
    authentic_gate = require_final_gates(
        schedule,
        dev_paths=[Path(str(item["path"])) for item in dev_manifests],
        signoff_path=Path(str(signoff["path"])),
    )
    if dict(authentic_gate) != dict(gate):
        raise RetrievalScheduleError(
            "retrieval authorization gate is not validator-authentic"
        )
    call_order = validator.entry.get("call_order")
    matching_entries = [
        entry for entry in schedule["calls"] if entry.get("call_order") == call_order
    ]
    if len(matching_entries) != 1 or dict(matching_entries[0]) != dict(
        validator.entry
    ):
        raise RetrievalScheduleError("retrieval authorization entry is not canonical")
    artifacts = _artifact_map(schedule)
    artifact_id = str(validator.artifact.get("artifact_id") or "")
    if artifact_id not in artifacts or dict(artifacts[artifact_id]) != dict(
        validator.artifact
    ):
        raise RetrievalScheduleError("retrieval authorization artifact is not canonical")
    progress = audit_artifacts(schedule)
    if progress.get("complete") is True or progress.get("next_call_order") != call_order:
        raise RetrievalScheduleError(
            "retrieval authorization is not for the next scheduled call"
        )
    existing_gate_hash = progress.get("gate_summary_sha256")
    if existing_gate_hash not in (None, authentic_gate["gate_summary_sha256"]):
        raise RetrievalScheduleError(
            "retrieval artifacts are bound to a different final gate"
        )
    return validator.authorize_evaluator_invocation(**observed)


def build_collector_argv(schedule: Mapping[str, Any], entry: Mapping[str, Any]) -> list[str]:
    artifact = _artifact_map(schedule)[entry["artifact_id"]]
    lane, controls = schedule["lanes"][entry["condition_id"]], schedule["controls"]
    return [
        "--cases", schedule["cases_path"], "--api-base", lane["api_base"],
        "--out", artifact["answers_path"], "--errors-out", artifact["errors_path"],
        "--attempt-journal", artifact["journal_path"], "--experiment-id", schedule["experiment_id"],
        "--condition-id", entry["condition_id"], "--generation-run-id", RUN_ID,
        "--provider", "extractive", "--institution", "none", "--context-k", "8",
        "--parser-profile", lane["parser_profile"], "--retrieval-mode", "bm25",
        "--expected-corpus-revision", lane["corpus_revision"],
        "--expected-retrieval-tuning", "on" if lane["expected_retrieval_tuning"] else "off",
        "--expected-context-chunks-per-document", "2",
        "--expected-git-commit", schedule["expected_git_commit"],
        "--expected-index-sha256", lane["index"]["sha256"],
        "--expected-source-manifest-sha256", schedule["shared_source_manifest_sha256"],
        "--eval-trace", "--only", entry["case_id"], "--allow-partial",
        "--sleep", str(controls["sleep_seconds"]), "--timeout", str(controls["timeout_seconds"]),
        "--max-attempts", str(controls["max_attempts"]),
        "--retry-backoff", str(controls["retry_backoff_seconds"]),
    ]


def audit_artifacts(schedule: Mapping[str, Any]) -> dict[str, Any]:
    entries = {int(entry["call_order"]): entry for entry in schedule["calls"]}
    entries_by_artifact: dict[str, list[Mapping[str, Any]]] = {
        str(artifact["artifact_id"]): [] for artifact in schedule["artifacts"]
    }
    for entry in schedule["calls"]:
        entries_by_artifact[str(entry["artifact_id"])].append(entry)
    completed: set[int] = set()
    gate_hashes: set[str] = set()
    gate_bindings: dict[str, Mapping[str, Any]] = {}
    for artifact in schedule["artifacts"]:
        artifact_id = str(artifact["artifact_id"])
        expected_entries = entries_by_artifact[artifact_id]
        answer_path = Path(artifact["answers_path"])
        error_path = Path(artifact["errors_path"])
        journal_path = Path(artifact["journal_path"])
        for label, path in (
            ("answer", answer_path),
            ("error", error_path),
            ("journal", journal_path),
        ):
            if path.is_symlink():
                raise RetrievalScheduleError(
                    f"retrieval {label} artifact must not be a symlink: {path}"
                )
        if error_path.exists() and error_path.stat().st_size:
            raise RetrievalScheduleError("retrieval schedule is poisoned by an error artifact")
        records = load_unique_jsonl(answer_path, key="answer_id") if answer_path.exists() else []
        if len(records) > len(expected_entries):
            raise RetrievalScheduleError("too many retrieval answers for artifact")
        answer_orders: set[int] = set()
        for local_index, record in enumerate(records):
            try:
                validate_answer_record(record)
                validate_final_retrieval_provenance(
                    record, require_collection_attempts=True
                )
            except ValueError as exc:
                raise RetrievalScheduleError(
                    f"invalid retrieval answer provenance: {exc}"
                ) from exc
            entry = expected_entries[local_index]
            order = record.get("call_order")
            if order != entry["call_order"] or entries.get(order) != entry:
                raise RetrievalScheduleError("answer call_order/artifact mismatch")
            expected_fields = {
                "experiment_id": schedule["experiment_id"],
                "condition_id": artifact_id,
                "generation_run_id": RUN_ID,
                "case_id": entry["case_id"],
                "case_sha256": entry["case_sha256"],
                "slot_outcome": "answer",
                "answer_eligible_for_judge": True,
            }
            if any(record.get(key) != value for key, value in expected_fields.items()):
                raise RetrievalScheduleError("answer case binding mismatch")
            if record.get("collection_schedule") != _record_metadata(schedule, entry):
                raise RetrievalScheduleError("answer schedule metadata mismatch")
            config = record.get("collector_config")
            if not isinstance(config, Mapping) or record.get(
                "collector_config_sha256"
            ) != sha256_json(config):
                raise RetrievalScheduleError("retrieval collector config invalid")
            lane = schedule["lanes"][artifact_id]
            expected_config = {
                "cases_sha256": schedule["cases_sha256"],
                "cases_canonical_sha256": schedule["cases_canonical_sha256"],
                "selected_case_ids_sha256": artifact[
                    "expected_case_ids_sha256"
                ],
                "provider": "extractive",
                "model": None,
                "institution": None,
                "context_k": 8,
                "parser_profile": lane["parser_profile"],
                "retrieval_mode": "bm25",
                "expected_corpus_revision": lane["corpus_revision"],
                "expected_retrieval_tuning": lane[
                    "expected_retrieval_tuning"
                ],
                "expected_context_chunks_per_document": 2,
                "expected_git_commit": schedule["expected_git_commit"],
                "expected_index_sha256": lane["index"]["sha256"],
                "expected_source_manifest_sha256": schedule[
                    "shared_source_manifest_sha256"
                ],
                "eval_trace": True,
                "allow_unpinned": False,
                "max_attempts": schedule["controls"]["max_attempts"],
                "retry_backoff_seconds": schedule["controls"][
                    "retry_backoff_seconds"
                ],
                "request_timeout_seconds": schedule["controls"][
                    "timeout_seconds"
                ],
                "inter_call_sleep_seconds": schedule["controls"][
                    "sleep_seconds"
                ],
                "errors_path": artifact["errors_path"],
                "attempt_journal_path": artifact["journal_path"],
            }
            for key, expected in expected_config.items():
                if config.get(key) != expected:
                    raise RetrievalScheduleError(
                        f"retrieval collector config {key} mismatch"
                    )
            auth = config.get("final_authorization")
            if not isinstance(auth, Mapping) or auth.get("collection_purpose") != "final_retrieval":
                raise RetrievalScheduleError("answer final authorization missing")
            if auth.get("schedule_sha256") != schedule["schedule_sha256"]:
                raise RetrievalScheduleError("answer authorization schedule mismatch")
            gate_hash = _nonempty(auth.get("gate_summary_sha256"), "gate hash")
            if len(gate_hash) != 64 or any(char not in "0123456789abcdef" for char in gate_hash):
                raise RetrievalScheduleError("answer gate summary SHA-256 invalid")
            binding = auth.get("gate_input_binding")
            if not isinstance(binding, Mapping) or auth.get(
                "gate_input_binding_sha256"
            ) != sha256_json(binding):
                raise RetrievalScheduleError("answer gate input binding invalid")
            expected_auth = _authorization_metadata(
                schedule,
                artifact,
                {"inputs": binding, "gate_summary_sha256": gate_hash},
            )
            if dict(auth) != expected_auth:
                raise RetrievalScheduleError("answer final authorization metadata mismatch")
            gate_hashes.add(gate_hash)
            gate_bindings[sha256_json(binding)] = binding
            completed.add(int(order))
            answer_orders.add(int(order))

        journal_by_order: dict[int, list[Mapping[str, Any]]] = {}
        if journal_path.exists():
            events = load_unique_jsonl(journal_path, key="journal_event_id")
            for event in events:
                payload = dict(event)
                event_id = payload.pop("journal_event_id", None)
                stored = payload.pop("record_sha256", None)
                if stored != sha256_json(payload):
                    raise RetrievalScheduleError("journal record SHA-256 mismatch")
                if event_id != "journal_" + sha256_json(
                    {**payload, "record_sha256": stored}
                )[:24]:
                    raise RetrievalScheduleError("journal event ID mismatch")
                order = event.get("call_order")
                entry = entries.get(order)
                if entry is None or entry.get("artifact_id") != artifact_id:
                    raise RetrievalScheduleError("journal call_order/artifact mismatch")
                expected_identity = {
                    "schema_version": "pnu.final-generation-slot-journal.v1",
                    "schedule_id": schedule["schedule_id"],
                    "schedule_sha256": schedule["schedule_sha256"],
                    "artifact_id": artifact_id,
                    "case_id": entry["case_id"],
                    "case_sha256": entry["case_sha256"],
                    "max_chat_attempts": schedule["controls"]["max_attempts"],
                }
                if any(event.get(key) != value for key, value in expected_identity.items()):
                    raise RetrievalScheduleError("journal identity/control mismatch")
                journal_by_order.setdefault(int(order), []).append(event)

        for entry in expected_entries:
            order = int(entry["call_order"])
            events = journal_by_order.get(order, [])
            names = [str(event.get("event")) for event in events]
            if len(names) != len(set(names)):
                raise RetrievalScheduleError("duplicate retrieval slot journal event")
            if order in answer_orders:
                if names != ["slot_started", "slot_completed"]:
                    raise RetrievalScheduleError(
                        "completed retrieval answer requires started/completed WAL"
                    )
                if events[0].get("outcome") is not None or events[1].get(
                    "outcome"
                ) != "answer":
                    raise RetrievalScheduleError("retrieval WAL outcome mismatch")
            elif names:
                raise RetrievalScheduleError(
                    "uncertain/poisoned retrieval journal slot cannot resume"
                )
    expected_prefix = set(range(1, len(completed) + 1))
    if completed != expected_prefix:
        raise RetrievalScheduleError("completed answers are not a global call_order prefix")
    if len(gate_hashes) > 1:
        raise RetrievalScheduleError("answers use mixed final gate summaries")
    if len(gate_bindings) > 1:
        raise RetrievalScheduleError("answers use mixed final gate input bindings")
    if gate_bindings:
        validate_gate_binding(schedule, next(iter(gate_bindings.values())))
    return {
        "schedule_id": schedule["schedule_id"], "schedule_sha256": schedule["schedule_sha256"],
        "completed_call_count": len(completed), "total_call_count": EXPECTED_CALL_COUNT,
        "next_call_order": len(completed) + 1 if len(completed) < EXPECTED_CALL_COUNT else None,
        "complete": len(completed) == EXPECTED_CALL_COUNT,
        "gate_summary_sha256": next(iter(gate_hashes), None),
    }


def execute_schedule(
    schedule: Mapping[str, Any], gate: Mapping[str, Any], *, dry_run: bool,
    dev_paths: Sequence[Path], signoff_path: Path,
    collector_main: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    authentic_gate = require_final_gates(
        schedule, dev_paths=dev_paths, signoff_path=signoff_path
    )
    if dict(gate) != authentic_gate:
        raise RetrievalScheduleError("provided gate is not validator-authentic")
    validate_gate_binding(schedule, authentic_gate["inputs"])
    audit = audit_artifacts(schedule)
    if dry_run:
        return audit
    if collector_main is None:
        from evaluate_service_answers import main as collector_main
    artifacts = _artifact_map(schedule)
    lock_path = Path(schedule["output_root"]) / ".final-retrieval-run.lock"
    with exclusive_run_lock(
        lock_path,
        {
            "kind": "final_retrieval",
            "schedule_id": schedule["schedule_id"],
            "schedule_sha256": schedule["schedule_sha256"],
        },
    ):
        validate_gate_binding(schedule, authentic_gate["inputs"])
        audit = audit_artifacts(schedule)
        while not audit["complete"]:
            entry = schedule["calls"][audit["next_call_order"] - 1]
            authorization = _issue_retrieval_authorization(
                schedule, entry, artifacts[entry["artifact_id"]], gate
            )
            try:
                collector_main(
                    build_collector_argv(schedule, entry),
                    final_authorization=authorization,
                )
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RetrievalScheduleError(
                        f"collector failed at call {entry['call_order']}"
                    ) from exc
            validate_gate_binding(schedule, authentic_gate["inputs"])
            after = audit_artifacts(schedule)
            if (
                after["completed_call_count"]
                != audit["completed_call_count"] + 1
            ):
                raise RetrievalScheduleError(
                    "collector did not append exactly one logical slot"
                )
            audit = after
        return audit


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
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-source-manifest-sha256")
    for lane_id in LANE_IDS:
        prefix = lane_id.replace("-", "_")
        parser.add_argument(f"--{lane_id}-api-base", dest=f"{prefix}_api_base")
        parser.add_argument(f"--{lane_id}-index", dest=f"{prefix}_index", type=Path)
        parser.add_argument(f"--{lane_id}-revision", dest=f"{prefix}_revision")
    parser.add_argument("--dev-manifest", type=Path, action="append")
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize-final-collection", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser, args = build_parser(), None
    args = parser.parse_args(argv)
    try:
        if args.create:
            required = [args.cases, args.experiment_id, args.output_root, args.expected_git_commit, args.expected_source_manifest_sha256]
            lane_inputs = {}
            for lane_id in LANE_IDS:
                prefix = lane_id.replace("-", "_")
                values = {
                    "api_base": getattr(args, f"{prefix}_api_base"),
                    "index_path": getattr(args, f"{prefix}_index"),
                    "corpus_revision": getattr(args, f"{prefix}_revision"),
                }
                required.extend(values.values())
                lane_inputs[lane_id] = values
            if any(value is None for value in required):
                parser.error("--create requires all case/freeze/source/lane controls")
            schedule = build_schedule(
                args.cases, load_jsonl(args.cases), experiment_id=args.experiment_id,
                output_root=args.output_root, expected_git_commit=args.expected_git_commit,
                expected_source_manifest_sha256=args.expected_source_manifest_sha256,
                lane_inputs=lane_inputs,
            )
            write_new_schedule(args.schedule, schedule)
            print(json.dumps(schedule["summary"], sort_keys=True))
            return 0
        schedule = load_schedule(args.schedule)
        if args.check:
            print(json.dumps(audit_artifacts(schedule), sort_keys=True))
            return 0
        if not args.dev_manifest or args.signoff is None:
            parser.error("--run requires --dev-manifest and --signoff")
        if not args.dry_run and not args.authorize_final_collection:
            parser.error("live run requires --authorize-final-collection")
        gate = require_final_gates(schedule, dev_paths=args.dev_manifest, signoff_path=args.signoff)
        print(json.dumps(execute_schedule(
            schedule, gate, dry_run=args.dry_run,
            dev_paths=args.dev_manifest, signoff_path=args.signoff,
        ), sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, RetrievalScheduleError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
