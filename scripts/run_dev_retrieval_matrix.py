#!/usr/bin/env python3
"""Run the exploratory DEV45 retrieval matrix through the real ``/chat`` path.

The matrix is retrieval-only even though it uses ``POST /chat``: every request
pins the in-process ``extractive`` provider, so no external generation API is
called.  The saved answer artifacts include the complete evaluation trace and
the exact final top-8 contexts that the normal service pipeline selected.

The eight protocol conditions are intentionally not a full factorial:

* PAR-B / PAR-CH / C1 use service tuning ON;
* C0 and all Dense/Hybrid lanes use service tuning OFF; and
* Dense/Hybrid lanes use only the Cascade parser artifacts.

Outputs are append-only via ``evaluate_service_answers.py``.  This wrapper
refuses to reuse an output directory unless ``--resume`` is explicit, and it
never writes to the historical C0/C1 artifact directories.

Example (all conditions):

  python3 scripts/run_dev_retrieval_matrix.py \
    --output-dir processed/eval/dev45-matrix-20260902 \
    --experiment-id dev45-retrieval-matrix-20260902

Example (safe BM25 subset when dense runtime dependencies are unavailable):

  python3 scripts/run_dev_retrieval_matrix.py \
    --output-dir processed/eval/dev45-matrix-20260902 \
    --experiment-id dev45-retrieval-matrix-20260902 \
    --conditions PAR-B,PAR-CH,C0,C1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from analyze_service_retrieval_ab import (  # noqa: E402
    load_cases,
    load_complete_run,
    record_values,
    summarize_arm,
)


SCHEMA_VERSION = "pnu.dev-retrieval-matrix/v1"
COMPACT_SCHEMA_VERSION = "pnu.dev-retrieval-trace.compact/v1"
DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-eval.jsonl"
DEFAULT_INDEXES = {
    "baseline": REPO_ROOT
    / "processed/index/pnu-20260725-curated-baseline-v5-allow-suspect.sqlite",
    "challenger": REPO_ROOT
    / "processed/index/pnu-20260725-curated-challenger-v5-allow-suspect.sqlite",
    "cascade": REPO_ROOT
    / "processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite",
}
DEFAULT_LEARNED_ROOT = REPO_ROOT / "processed/index/learned-dense"
DEFAULT_CONTEXT_K = 8
DEFAULT_CONTEXT_CHUNKS_PER_DOCUMENT = 2
DENSE_MODULES = ("numpy", "torch", "sentence_transformers")


@dataclass(frozen=True)
class Condition:
    id: str
    parser_profile: str
    retrieval_mode: str
    service_tuning: bool
    purpose: str

    @property
    def artifact_name(self) -> str:
        return self.id.casefold().replace("-", "_") + ".answers.jsonl"


CONDITIONS = (
    Condition("PAR-B", "baseline", "bm25", True, "parser comparison"),
    Condition("PAR-CH", "challenger", "bm25", True, "parser comparison"),
    Condition("C0", "cascade", "bm25", False, "untuned causal control"),
    Condition("C1", "cascade", "bm25", True, "current tuned service"),
    Condition("D-K", "cascade", "kure_dense", False, "exploratory dense"),
    Condition("H-K", "cascade", "kure_hybrid", False, "exploratory hybrid"),
    Condition("D-S", "cascade", "snowflake_dense", False, "exploratory dense"),
    Condition("H-S", "cascade", "snowflake_hybrid", False, "exploratory hybrid"),
)
CONDITIONS_BY_ID = {condition.id: condition for condition in CONDITIONS}


class MatrixError(RuntimeError):
    """Raised when a matrix control or immutable artifact is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _index_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise MatrixError(f"missing BM25 index: {path}")
    connection = sqlite3.connect(str(path))
    try:
        rows = connection.execute("SELECT key, value FROM index_meta").fetchall()
        chunk_count = int(
            connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        )
    except sqlite3.DatabaseError as exc:
        raise MatrixError(f"invalid BM25 index {path}: {exc}") from exc
    finally:
        connection.close()
    result = {str(key): str(value) for key, value in rows}
    result["chunk_count"] = str(chunk_count)
    return result


def _artifact_dir(learned_root: Path, family: str) -> Path:
    names = {
        "kure": "kure-v1",
        "snowflake": "snowflake-arctic-l-v2-ko",
    }
    # The established Cascade artifacts use the legacy model-only layout.
    profile_path = learned_root / "cascade" / names[family]
    return profile_path if profile_path.is_dir() else learned_root / names[family]


def audit_inputs(
    *,
    cases_path: Path,
    indexes: Mapping[str, Path],
    learned_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    order, _ = load_cases(cases_path)
    if len(order) != 45:
        raise MatrixError(f"DEV matrix requires exactly 45 cases, found {len(order)}")

    index_rows: dict[str, dict[str, Any]] = {}
    source_manifest_values: set[str] = set()
    for profile in ("baseline", "challenger", "cascade"):
        path = Path(indexes[profile])
        metadata = _index_metadata(path)
        if metadata.get("profile") != profile:
            raise MatrixError(
                f"{profile} index profile mismatch: {metadata.get('profile')!r}"
            )
        revision = str(metadata.get("corpus_revision") or "")
        if not revision:
            raise MatrixError(f"{profile} index has no corpus_revision")
        source_manifest = str(metadata.get("source_manifest_sha256") or "")
        if not source_manifest:
            raise MatrixError(f"{profile} index has no source_manifest_sha256")
        source_manifest_values.add(source_manifest)
        index_rows[profile] = {
            "path": str(path),
            "profile": profile,
            "corpus_revision": revision,
            "source_manifest_sha256": source_manifest,
            "chunk_count": int(metadata["chunk_count"]),
        }
    if len(source_manifest_values) != 1:
        raise MatrixError(
            "parser indexes do not share one source manifest: "
            f"{sorted(source_manifest_values)}"
        )

    module_probe = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.util,json; "
                "print(json.dumps({n: bool(importlib.util.find_spec(n)) "
                f"for n in {DENSE_MODULES!r}}}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    modules = {name: False for name in DENSE_MODULES}
    if module_probe.returncode == 0:
        try:
            modules.update(json.loads(module_probe.stdout.strip()))
        except json.JSONDecodeError:
            pass

    dense_rows: dict[str, dict[str, Any]] = {}
    cascade_revision = index_rows["cascade"]["corpus_revision"]
    for family in ("kure", "snowflake"):
        artifact = _artifact_dir(learned_root, family)
        row: dict[str, Any] = {
            "path": str(artifact),
            "ready": False,
            "blockers": [],
        }
        manifest_path = artifact / "manifest.json"
        if not manifest_path.is_file():
            row["blockers"].append("manifest_missing")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                source = manifest.get("source") or {}
                vectors = manifest.get("vectors") or {}
                chunk_ids = manifest.get("chunk_ids") or {}
                row.update(
                    {
                        "model": (manifest.get("model") or {}).get("id"),
                        "model_revision": (manifest.get("model") or {}).get(
                            "revision"
                        ),
                        "corpus_revision": source.get("corpus_revision"),
                        "chunk_count": vectors.get("count"),
                        "dimensions": vectors.get("dimensions"),
                    }
                )
                if source.get("corpus_revision") != cascade_revision:
                    row["blockers"].append("corpus_revision_mismatch")
                for metadata, label in (
                    (vectors, "vectors"),
                    (chunk_ids, "chunk_ids"),
                ):
                    name = str(metadata.get("file") or "")
                    if not name or Path(name).name != name or not (artifact / name).is_file():
                        row["blockers"].append(f"{label}_missing")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                row["blockers"].append(
                    f"invalid_manifest:{type(exc).__name__}:{exc}"
                )
        missing_modules = [name for name, ready in modules.items() if not ready]
        if missing_modules:
            row["blockers"].append(
                "runtime_modules_missing:" + ",".join(missing_modules)
            )
        row["ready"] = not row["blockers"]
        dense_rows[family] = row

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "DEV45 exploratory retrieval-only input audit",
        "case_count": len(order),
        "cases_path": str(cases_path),
        "cases_sha256": _sha256_file(cases_path),
        "source_manifest_sha256": next(iter(source_manifest_values)),
        "python_executable": python_executable,
        "dense_runtime_modules": modules,
        "indexes": index_rows,
        "learned_dense": dense_rows,
    }


def parse_condition_ids(value: str) -> list[str]:
    requested = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not requested:
        raise MatrixError("at least one condition is required")
    unknown = sorted(set(requested) - set(CONDITIONS_BY_ID))
    if unknown:
        raise MatrixError(f"unknown condition(s): {', '.join(unknown)}")
    if len(requested) != len(set(requested)):
        raise MatrixError("condition IDs must be unique")
    requested_set = set(requested)
    return [condition.id for condition in CONDITIONS if condition.id in requested_set]


def build_server_command(
    *,
    python_executable: str,
    port: int,
    indexes: Mapping[str, Path],
    learned_root: Path,
    service_tuning: bool,
) -> list[str]:
    command = [
        python_executable,
        "-B",
        str(SCRIPTS_ROOT / "search_api.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--index",
        str(indexes["cascade"]),
        "--profile-index",
        f"baseline={indexes['baseline']}",
        "--profile-index",
        f"challenger={indexes['challenger']}",
        "--default-parser-profile",
        "cascade",
        "--learned-dense-root",
        str(learned_root),
        "--context-chunks-per-document",
        str(DEFAULT_CONTEXT_CHUNKS_PER_DOCUMENT),
        "--env-file",
        os.devnull,
    ]
    if not service_tuning:
        command.append("--no-retrieval-tuning")
    return command


def build_collector_command(
    condition: Condition,
    *,
    python_executable: str,
    cases_path: Path,
    output_path: Path,
    experiment_id: str,
    api_base: str,
    corpus_revision: str,
) -> list[str]:
    return [
        python_executable,
        "-B",
        str(SCRIPTS_ROOT / "evaluate_service_answers.py"),
        "--cases",
        str(cases_path),
        "--api-base",
        api_base,
        "--out",
        str(output_path),
        "--experiment-id",
        experiment_id,
        "--condition-id",
        condition.id,
        "--generation-run-id",
        "retrieval-only-run1",
        "--provider",
        "extractive",
        "--institution",
        "none",
        "--context-k",
        str(DEFAULT_CONTEXT_K),
        "--parser-profile",
        condition.parser_profile,
        "--retrieval-mode",
        condition.retrieval_mode,
        "--expected-corpus-revision",
        corpus_revision,
        "--expected-retrieval-tuning",
        "on" if condition.service_tuning else "off",
        "--expected-context-chunks-per-document",
        str(DEFAULT_CONTEXT_CHUNKS_PER_DOCUMENT),
        "--eval-trace",
        "--sleep",
        "0",
    ]


def _read_health(api_base: str, timeout: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{api_base}/health", timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise MatrixError("/health returned a non-object")
    return value


def wait_for_health(api_base: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            health = _read_health(api_base)
            if health.get("ready") is True:
                return health
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise MatrixError(f"server did not become ready: {last_error}")


def health_condition_blocker(
    health: Mapping[str, Any], condition: Condition
) -> str | None:
    service = health.get("service_config")
    if not isinstance(service, Mapping):
        return "health_missing_service_config"
    if service.get("retrieval_tuning") is not condition.service_tuning:
        return "health_retrieval_tuning_mismatch"
    if service.get("context_chunks_per_document") != DEFAULT_CONTEXT_CHUNKS_PER_DOCUMENT:
        return "health_context_cap_mismatch"
    if service.get("evaluation_trace_enabled") is not True:
        return "health_eval_trace_disabled"
    profiles = health.get("parser_profiles")
    profiles = profiles if isinstance(profiles, list) else []
    profile = next(
        (
            item
            for item in profiles
            if isinstance(item, Mapping)
            and item.get("id") == condition.parser_profile
        ),
        None,
    )
    if profile is None or profile.get("ready") is not True:
        return f"parser_profile_unavailable:{condition.parser_profile}"
    modes = profile.get("retrieval_modes")
    modes = modes if isinstance(modes, list) else []
    mode = next(
        (
            item
            for item in modes
            if isinstance(item, Mapping)
            and item.get("id") == condition.retrieval_mode
        ),
        None,
    )
    if mode is None:
        return f"retrieval_mode_not_reported:{condition.retrieval_mode}"
    if mode.get("ready") is not True:
        return (
            f"retrieval_mode_unavailable:{condition.retrieval_mode}:"
            f"{mode.get('reason')}"
        )
    return None


def _validate_record_controls(record: Mapping[str, Any], condition: Condition) -> None:
    case_id = str(record.get("case_id") or record.get("id") or "")
    prefix = f"{condition.id}/{case_id or '?'}"
    if record.get("condition_id") != condition.id:
        raise MatrixError(f"{prefix}: condition_id mismatch")
    config = record.get("collector_config")
    if not isinstance(config, Mapping):
        raise MatrixError(f"{prefix}: collector_config missing")
    expected = {
        "provider": "extractive",
        "context_k": DEFAULT_CONTEXT_K,
        "parser_profile": condition.parser_profile,
        "retrieval_mode": condition.retrieval_mode,
        "expected_retrieval_tuning": condition.service_tuning,
        "expected_context_chunks_per_document": DEFAULT_CONTEXT_CHUNKS_PER_DOCUMENT,
        "eval_trace": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise MatrixError(
                f"{prefix}: collector control {key} expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    response_config = record.get("response_config")
    if not isinstance(response_config, Mapping):
        raise MatrixError(f"{prefix}: response_config missing")
    if response_config.get("parser_profile") != condition.parser_profile:
        raise MatrixError(f"{prefix}: response parser profile mismatch")
    if response_config.get("retrieval_mode") != condition.retrieval_mode:
        raise MatrixError(f"{prefix}: response retrieval mode mismatch")
    if response_config.get("generation_used") != "extractive":
        raise MatrixError(f"{prefix}: external/non-extractive generation observed")
    trace = record.get("evaluation_trace")
    if not isinstance(trace, Mapping) or trace.get("schema_version") != 1:
        raise MatrixError(f"{prefix}: evaluation trace missing")
    stages = trace.get("retrieval_stages")
    if not isinstance(stages, Mapping):
        raise MatrixError(f"{prefix}: retrieval stages missing")
    final_contexts = stages.get("final_contexts")
    if not isinstance(final_contexts, list):
        raise MatrixError(f"{prefix}: final context trace missing")
    sources = record.get("sources")
    if not isinstance(sources, list) or len(sources) != len(final_contexts):
        raise MatrixError(f"{prefix}: final context/source count mismatch")
    if len(final_contexts) > DEFAULT_CONTEXT_K:
        raise MatrixError(f"{prefix}: more than top-8 final contexts")


def _compact_source(source: Mapping[str, Any], rank: int) -> dict[str, Any]:
    """Project a final context without copying preview/location payloads."""

    return {
        "rank": rank,
        "chunk_id": source.get("chunk_id"),
        "document_id": source.get("document_id") or source.get("doc_id"),
        "corpus_revision": source.get("corpus_revision"),
        "source_title": source.get("source_title"),
        "source_host": source.get("source_host"),
        "retrieval": source.get("retrieval") or {},
    }


def _compact_stage_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Keep ranking provenance and text integrity, but not duplicated text."""

    return {
        "rank": entry.get("rank"),
        "stage": entry.get("stage"),
        "chunk_id": entry.get("chunk_id"),
        "document_id": entry.get("document_id") or entry.get("doc_id"),
        "chunk_index": entry.get("chunk_index"),
        "corpus_revision": entry.get("corpus_revision"),
        "source_title": entry.get("source_title"),
        "source_host": entry.get("source_host"),
        "retrieval": entry.get("retrieval") or {},
        "text_sha256": entry.get("text_sha256"),
    }


def compact_record(
    record: Mapping[str, Any],
    condition: Condition,
    *,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    """Build a metric-complete retrieval projection from one answer record.

    The immutable answer artifact remains the authority for full context text.
    This derivative is sufficient to recompute document/chunk metrics and audit
    every stage's order, scores, identities, and source text hashes.
    """

    _validate_record_controls(record, condition)
    trace = record["evaluation_trace"]
    assert isinstance(trace, Mapping)
    stages = trace["retrieval_stages"]
    assert isinstance(stages, Mapping)
    compact_stages: dict[str, list[dict[str, Any]]] = {}
    for stage_name in (
        "raw_bm25",
        "post_retrieval_pool",
        "post_neighbor_expansion",
        "final_contexts",
    ):
        entries = stages.get(stage_name)
        if not isinstance(entries, list):
            raise MatrixError(
                f"{condition.id}/{record.get('case_id')}: missing {stage_name} trace"
            )
        compact_stages[stage_name] = [
            _compact_stage_entry(entry)
            for entry in entries
            if isinstance(entry, Mapping)
        ]
        if len(compact_stages[stage_name]) != len(entries):
            raise MatrixError(
                f"{condition.id}/{record.get('case_id')}: invalid {stage_name} row"
            )

    collector = record.get("collector_config")
    assert isinstance(collector, Mapping)
    sources = record.get("sources")
    assert isinstance(sources, list)
    compact: dict[str, Any] = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "source_artifact_sha256": source_artifact_sha256,
        "source_record_sha256": record.get("record_sha256"),
        "source_answer_id": record.get("answer_id"),
        "experiment_id": record.get("experiment_id"),
        "condition_id": condition.id,
        "generation_run_id": record.get("generation_run_id"),
        "case_id": record.get("case_id"),
        "case_sha256": record.get("case_sha256"),
        "query": record.get("query"),
        "controls": {
            "collector_config_sha256": record.get("collector_config_sha256"),
            "provider": collector.get("provider"),
            "context_k": collector.get("context_k"),
            "parser_profile": collector.get("parser_profile"),
            "retrieval_mode": collector.get("retrieval_mode"),
            "corpus_revision": collector.get("expected_corpus_revision"),
            "service_tuning": collector.get("expected_retrieval_tuning"),
            "context_chunks_per_document": collector.get(
                "expected_context_chunks_per_document"
            ),
            "eval_trace": collector.get("eval_trace"),
        },
        "latency_ms": record.get("latency_ms"),
        "retrieval_hit": record.get("retrieval_hit"),
        "evidence_at_k": record.get("evidence_at_k"),
        "retrieval": record.get("retrieval") or {},
        "sources": [
            _compact_source(source, rank)
            for rank, source in enumerate(sources, start=1)
            if isinstance(source, Mapping)
        ],
        "evaluation_trace": {
            "schema_version": trace.get("schema_version"),
            "retrieval_stages": compact_stages,
            "stage_counts": {
                name: len(entries) for name, entries in compact_stages.items()
            },
            "timing_ms": trace.get("timing_ms") or {},
        },
    }
    if len(compact["sources"]) != len(sources):
        raise MatrixError(
            f"{condition.id}/{record.get('case_id')}: invalid source row"
        )
    compact["compact_record_sha256"] = hashlib.sha256(
        _canonical_json(compact).encode("utf-8")
    ).hexdigest()
    return compact


def _verify_compact_metrics(
    *,
    cases_path: Path,
    source: Path,
    compact: Path,
) -> None:
    order, cases = load_cases(cases_path)
    expected_ids = set(order)
    raw_run = load_complete_run(source, expected_ids)
    compact_run = load_complete_run(compact, expected_ids)

    def metrics(run: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        per_case = {
            case_id: record_values(run[case_id], cases[case_id])
            for case_id in order
        }
        summary = summarize_arm(order, per_case)
        return {
            key: summary[key]
            for key in ("document", "gold_chunk", "latency_ms", "unique_documents")
        }

    if metrics(raw_run) != metrics(compact_run):
        raise MatrixError(f"compact retrieval metrics differ from source: {source}")


def write_compact_outputs(output_dir: Path, *, cases_path: Path) -> dict[str, Any]:
    """Write atomic compact derivatives and report their size reduction."""

    compact_dir = output_dir / "compact"
    compact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        source = output_dir / condition.artifact_name
        if not source.is_file():
            continue
        source_sha = _sha256_file(source)
        target = compact_dir / condition.artifact_name.replace(
            ".answers.jsonl", ".retrieval.jsonl"
        )
        temporary = target.with_suffix(target.suffix + ".tmp")
        count = 0
        with source.open("r", encoding="utf-8") as reader, temporary.open(
            "w", encoding="utf-8"
        ) as writer:
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MatrixError(
                        f"{source}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise MatrixError(f"{source}:{line_number}: expected object")
                projection = compact_record(
                    record,
                    condition,
                    source_artifact_sha256=source_sha,
                )
                writer.write(_canonical_json(projection) + "\n")
                count += 1
            writer.flush()
            os.fsync(writer.fileno())
        if count != 45:
            temporary.unlink(missing_ok=True)
            raise MatrixError(f"{source}: expected 45 records, found {count}")
        os.replace(temporary, target)
        _verify_compact_metrics(
            cases_path=cases_path,
            source=source,
            compact=target,
        )
        source_bytes = source.stat().st_size
        compact_bytes = target.stat().st_size
        artifacts.append(
            {
                "condition_id": condition.id,
                "source": str(source),
                "source_sha256": source_sha,
                "source_bytes": source_bytes,
                "compact": str(target),
                "compact_sha256": _sha256_file(target),
                "compact_bytes": compact_bytes,
                "size_reduction_fraction": 1.0 - compact_bytes / source_bytes,
                "records": count,
                "retrieval_metrics_verified_identical": True,
            }
        )
    source_total = sum(row["source_bytes"] for row in artifacts)
    compact_total = sum(row["compact_bytes"] for row in artifacts)
    report = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "scope": "metric-complete derivative; full immutable traces retained separately",
        "generated_at": _utc_now(),
        "source_total_bytes": source_total,
        "compact_total_bytes": compact_total,
        "size_reduction_fraction": (
            1.0 - compact_total / source_total if source_total else 0.0
        ),
        "omitted_payloads": [
            "answer/generation/postprocessing payloads",
            "raw/sanitized draft and generation prompt",
            "duplicated context text (text_sha256 retained)",
            "source preview and location arrays",
        ],
        "artifacts": artifacts,
    }
    (compact_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def summarize_outputs(
    *,
    cases_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    order, cases = load_cases(cases_path)
    expected_ids = set(order)
    condition_reports: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    per_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        artifact = output_dir / condition.artifact_name
        if not artifact.is_file():
            condition_reports.append(
                {**asdict(condition), "status": "pending", "artifact": str(artifact)}
            )
            continue
        run = load_complete_run(artifact, expected_ids)
        for record in run.values():
            _validate_record_controls(record, condition)
        per_case = {
            case_id: record_values(run[case_id], cases[case_id])
            for case_id in order
        }
        per_condition[condition.id] = per_case
        arm = summarize_arm(order, per_case)
        report: dict[str, Any] = {
            **asdict(condition),
            "status": "complete",
            "artifact": str(artifact),
            "artifact_sha256": _sha256_file(artifact),
            "questions": arm["questions"],
            "document": arm["document"],
            "latency_ms": arm["latency_ms"],
            "unique_documents": arm["unique_documents"],
        }
        # DEV gold chunk IDs are Cascade-specific.  Exposing zeros for the
        # parser lanes would be a false parser-quality comparison.
        if condition.parser_profile == "cascade":
            report["gold_chunk"] = arm["gold_chunk"]
        else:
            report["gold_chunk"] = {
                "available": False,
                "reason": "DEV gold chunk IDs are Cascade-specific",
            }
        condition_reports.append(report)

        row: dict[str, Any] = {
            "condition": condition.id,
            "parser_profile": condition.parser_profile,
            "retrieval_mode": condition.retrieval_mode,
            "service_tuning": condition.service_tuning,
            "questions": arm["questions"],
            "document_hit_at_1": arm["document"]["hit_at_1"]["rate"],
            "document_hit_at_3": arm["document"]["hit_at_3"]["rate"],
            "document_hit_at_5": arm["document"]["hit_at_5"]["rate"],
            "document_mrr_at_5": arm["document"]["mrr_at_5"],
            "latency_p50_ms": arm["latency_ms"]["p50"],
            "latency_p95_ms": arm["latency_ms"]["p95"],
            "mean_unique_documents": arm["unique_documents"]["mean"],
            "any_gold_chunk_at_5": None,
            "all_gold_chunks_at_5": None,
            "gold_chunk_recall_at_5": None,
            "any_gold_chunk_at_8": None,
            "all_gold_chunks_at_8": None,
            "gold_chunk_recall_at_8": None,
        }
        if condition.parser_profile == "cascade":
            for cutoff in (5, 8):
                values = arm["gold_chunk"][str(cutoff)]
                row[f"any_gold_chunk_at_{cutoff}"] = values["any_rate"]
                row[f"all_gold_chunks_at_{cutoff}"] = values["all_rate"]
                row[f"gold_chunk_recall_at_{cutoff}"] = values[
                    "mean_gold_chunk_recall"
                ]
        csv_rows.append(row)

    paired_against_c1: dict[str, Any] = {}
    reference = per_condition.get("C1")
    if reference is not None:
        for lane_id in ("C0", "D-K", "H-K", "D-S", "H-S"):
            lane = per_condition.get(lane_id)
            if lane is None:
                continue

            def transition(metric: str, cutoff: str) -> dict[str, Any]:
                if metric == "document":
                    getter = lambda value: bool(
                        value["document"]["hit_at_k"][cutoff]
                    )
                else:
                    getter = lambda value: bool(
                        value["evidence_at_k"][cutoff][metric]
                    )
                gained = [
                    case_id
                    for case_id in order
                    if getter(lane[case_id]) and not getter(reference[case_id])
                ]
                lost = [
                    case_id
                    for case_id in order
                    if getter(reference[case_id]) and not getter(lane[case_id])
                ]
                return {
                    "gained_count": len(gained),
                    "lost_count": len(lost),
                    "net_count": len(gained) - len(lost),
                    "gained_case_ids": gained,
                    "lost_case_ids": lost,
                }

            paired_against_c1[lane_id] = {
                "direction": f"{lane_id} minus C1",
                "document_hit_at_5": transition("document", "5"),
                "any_gold_chunk_at_8": transition("any_matched", "8"),
                "all_gold_chunks_at_8": transition("all_matched", "8"),
            }

    complete = sum(row.get("status") == "complete" for row in condition_reports)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": "DEV45 exploratory retrieval-only matrix; not HOLDOUT or generalization",
        "generated_at": _utc_now(),
        "cases_path": str(cases_path),
        "cases_sha256": _sha256_file(cases_path),
        "question_count": len(order),
        "context_k": DEFAULT_CONTEXT_K,
        "generation_provider": "extractive (no external LLM)",
        "matrix_status": "complete" if complete == len(CONDITIONS) else "partial",
        "completed_conditions": complete,
        "total_conditions": len(CONDITIONS),
        "metric_notes": {
            "document": "source-title document rank after folding repeated chunks",
            "parser_gold": (
                "PAR-B/PAR-CH use Source Hit/MRR only because DEV chunk gold is "
                "Cascade-specific"
            ),
            "cascade_gold": (
                "Any/All/Recall are legacy exact DEV gold-chunk coverage, not "
                "atomic Evidence Recall"
            ),
            "latency": "single sequential local run; diagnostic only",
        },
        "conditions": condition_reports,
        "paired_against_c1": paired_against_c1,
        "paired_analysis_note": (
            "Every compact row retains case_id and ordered final source IDs/titles, "
            "so all paired source/gold transitions can be recomputed without full text."
        ),
    }
    return summary, csv_rows


def write_summary(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "matrix-summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _safe_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "RAG_EVAL_TRACE": "1",
            "RAG_GENERATION_MODE": "extractive",
            "RAG_EMBEDDING_LOCAL_ONLY": "1",
            "RAG_API_TOKEN": "",
            "GEMINI_API_KEY": "",
            "OPENAI_API_KEY": "",
        }
    )
    return environment


def run_group(
    conditions: Sequence[Condition],
    *,
    args: argparse.Namespace,
    indexes: Mapping[str, Path],
    revisions: Mapping[str, str],
) -> dict[str, str]:
    if not conditions:
        return {}
    tuning = conditions[0].service_tuning
    if any(condition.service_tuning is not tuning for condition in conditions):
        raise MatrixError("one server group cannot mix retrieval tuning controls")
    api_base = f"http://127.0.0.1:{args.port}"
    label = "tuned" if tuning else "untuned"
    log_path = args.output_dir / f"server-{label}.log"
    statuses: dict[str, str] = {}
    command = build_server_command(
        python_executable=args.python,
        port=args.port,
        indexes=indexes,
        learned_root=args.learned_dense_root,
        service_tuning=tuning,
    )
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_safe_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            health = wait_for_health(api_base, args.startup_timeout)
            for condition in conditions:
                blocker = health_condition_blocker(health, condition)
                if blocker:
                    statuses[condition.id] = "blocked:" + blocker
                    if not args.allow_partial:
                        raise MatrixError(f"{condition.id}: {blocker}")
                    continue
                artifact = args.output_dir / condition.artifact_name
                if artifact.exists() and not args.resume:
                    raise MatrixError(
                        f"refusing to reuse existing artifact without --resume: {artifact}"
                    )
                collector = build_collector_command(
                    condition,
                    python_executable=args.python,
                    cases_path=args.cases,
                    output_path=artifact,
                    experiment_id=args.experiment_id,
                    api_base=api_base,
                    corpus_revision=revisions[condition.parser_profile],
                )
                completed = subprocess.run(
                    collector,
                    cwd=REPO_ROOT,
                    env=_safe_environment(),
                    check=False,
                )
                if completed.returncode != 0:
                    statuses[condition.id] = (
                        f"failed:collector_exit_{completed.returncode}"
                    )
                    if not args.allow_partial:
                        raise MatrixError(
                            f"{condition.id}: collector exited {completed.returncode}"
                        )
                else:
                    statuses[condition.id] = "complete"
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return statuses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--conditions",
        default=",".join(condition.id for condition in CONDITIONS),
        help="comma-separated condition IDs",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--learned-dense-root", type=Path, default=DEFAULT_LEARNED_ROOT)
    parser.add_argument("--baseline-index", type=Path, default=DEFAULT_INDEXES["baseline"])
    parser.add_argument(
        "--challenger-index", type=Path, default=DEFAULT_INDEXES["challenger"]
    )
    parser.add_argument("--cascade-index", type=Path, default=DEFAULT_INDEXES["cascade"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="record unavailable conditions and keep completed artifacts",
    )
    parser.add_argument(
        "--audit-only", action="store_true", help="validate inputs without binding a port"
    )
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="derive compact metric-complete traces from completed raw artifacts",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if args.audit_only and args.compact_only:
        parser.error("--audit-only and --compact-only are mutually exclusive")
    try:
        args.condition_ids = parse_condition_ids(args.conditions)
    except MatrixError as exc:
        parser.error(str(exc))
    args.output_dir = args.output_dir.resolve()
    args.cases = args.cases.resolve()
    args.learned_dense_root = args.learned_dense_root.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    indexes = {
        "baseline": args.baseline_index.resolve(),
        "challenger": args.challenger_index.resolve(),
        "cascade": args.cascade_index.resolve(),
    }
    if args.output_dir.exists() and not args.resume and any(args.output_dir.iterdir()):
        raise MatrixError(
            f"output directory is not empty; use --resume explicitly: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_inputs(
        cases_path=args.cases,
        indexes=indexes,
        learned_root=args.learned_dense_root,
        python_executable=args.python,
    )
    audit["audited_at"] = _utc_now()
    (args.output_dir / "input-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0

    if args.compact_only:
        compact = write_compact_outputs(args.output_dir, cases_path=args.cases)
        summary, rows = summarize_outputs(
            cases_path=args.cases,
            output_dir=args.output_dir,
        )
        summary["compact_projection"] = compact
        summary["input_audit"] = str(args.output_dir / "input-audit.json")
        write_summary(args.output_dir, summary, rows)
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return 0

    selected = [CONDITIONS_BY_ID[value] for value in args.condition_ids]
    revisions = {
        profile: str(row["corpus_revision"])
        for profile, row in audit["indexes"].items()
    }
    statuses: dict[str, str] = {}
    # Tuned and untuned conditions must run in separate server processes so the
    # global service-tuning control cannot leak between cells.
    for tuning in (True, False):
        group = [condition for condition in selected if condition.service_tuning is tuning]
        statuses.update(
            run_group(
                group,
                args=args,
                indexes=indexes,
                revisions=revisions,
            )
        )

    compact = write_compact_outputs(args.output_dir, cases_path=args.cases)
    summary, rows = summarize_outputs(cases_path=args.cases, output_dir=args.output_dir)
    summary["execution_status"] = statuses
    summary["input_audit"] = str(args.output_dir / "input-audit.json")
    summary["compact_projection"] = compact
    write_summary(args.output_dir, summary, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = [value for value in statuses.values() if value != "complete"]
    return 0 if not failed or args.allow_partial else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatrixError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
