#!/usr/bin/env python3
"""Run the post-freeze C2 quote-bound generation experiment on DEV answers.

C2 deliberately reuses the immutable C1 retrieval trace.  It changes only the
generation/grounding boundary, so a C1-versus-C2 comparison isolates whether
claim-level verbatim evidence binding improves grounded answer quality.

This script is DEV-only and fail-closed: paths containing ``holdout`` or
``draft`` are rejected, dry-run never loads credentials or writes files, live
collection requires an exact authorization phrase, and every output is
published without replacement.  It is not wired into the frozen service.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from immutable_outputs import (  # noqa: E402
    publish_immutable_texts,
    reject_symlink_inputs,
    require_new_outputs,
)
from rag.grounded_claims_v2 import (  # noqa: E402
    MAX_CLAIMS,
    MAX_EVIDENCE_PER_CLAIM,
    PROMPT_VERSION,
    SCHEMA_VERSION as GROUNDED_SCHEMA_VERSION,
    SYSTEM_INSTRUCTION,
    build_prompt,
    sha256_text,
    verify_response,
)
from service_eval_artifacts import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    build_answer_identity,
    canonical_json,
    sha256_json,
    validate_answer_record,
)


COLLECTOR_VERSION = "pnu.grounded-claims-c2-collector.v1"
AUTHORIZATION_PHRASE = "I_APPROVE_C2_DEV_GENERATION"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_OUTPUT_TOKENS = 1_200
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_RETRIES = 3
DEFAULT_SLEEP_SECONDS = 3.0

RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims", "unanswered"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": MAX_CLAIMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence"],
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_EVIDENCE_PER_CLAIM,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["source_number", "quote"],
                            "properties": {
                                "source_number": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                                "quote": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "unanswered": {
            "type": "array",
            "maxItems": MAX_CLAIMS,
            "items": {"type": "string"},
        },
    },
}

PostJsonFn = Callable[
    [str, bytes, str, float], tuple[dict[str, Any], Optional[int]]
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE rows without logging secret values."""

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _forbidden_dev_path(path: Path) -> bool:
    lowered = str(path).casefold()
    return "holdout" in lowered or "draft" in lowered


def _final_contexts(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("evaluation_trace")
    if not isinstance(trace, Mapping):
        return []
    stages = trace.get("retrieval_stages")
    if not isinstance(stages, Mapping):
        return []
    contexts = stages.get("final_contexts")
    if not isinstance(contexts, list):
        return []
    return [dict(value) for value in contexts if isinstance(value, Mapping)]


def load_dev_answers(
    path: Path,
    *,
    only: Sequence[str] = (),
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load a single immutable C1 DEV run and verify its identity/integrity."""

    if _forbidden_dev_path(path):
        raise ValueError("C2 experiment accepts DEV inputs only")
    reject_symlink_inputs([path])
    before = sha256_file(path)
    records: list[dict[str, Any]] = []
    seen_answer_ids: set[str] = set()
    seen_case_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        validate_answer_record(value)
        if value.get("condition_id") != "c1":
            raise ValueError("C2 source must be a frozen C1 answer artifact")
        answer_id = str(value["answer_id"])
        case_id = str(value["case_id"])
        if answer_id in seen_answer_ids or case_id in seen_case_ids:
            raise ValueError(f"duplicate answer/case identity: {case_id}")
        seen_answer_ids.add(answer_id)
        seen_case_ids.add(case_id)
        contexts = _final_contexts(value)
        if not contexts:
            raise ValueError(f"C1 answer {answer_id} has no final_contexts")
        source_numbers = [context.get("source_number") for context in contexts]
        if source_numbers != list(range(1, len(contexts) + 1)):
            raise ValueError(f"C1 answer {answer_id} has non-sequential sources")
        records.append(value)
    if sha256_file(path) != before:
        raise ValueError("C1 source artifact changed while it was being read")
    if not records:
        raise ValueError("C1 source artifact is empty")

    run_identity = {
        (
            str(record.get("experiment_id")),
            str(record.get("generation_run_id")),
        )
        for record in records
    }
    if len(run_identity) != 1:
        raise ValueError("C1 source artifact mixes experiment or run identities")

    if only:
        requested = set(only)
        unknown = sorted(requested - seen_case_ids)
        if unknown:
            raise ValueError(f"unknown --only case ids: {unknown}")
        records = [record for record in records if record["case_id"] in requested]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:limit]
    if not records:
        raise ValueError("C2 selection is empty")
    return records, before


def build_request_body(prompt: str, *, max_output_tokens: int) -> dict[str, Any]:
    return {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": RESPONSE_JSON_SCHEMA,
        },
    }


def _post_json(
    url: str,
    body: bytes,
    api_key: str,
    timeout: float,
) -> tuple[dict[str, Any], int | None]:
    request = urllib.request.Request(
        url,
        body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", None)
        return json.load(response), int(status) if status is not None else None


def _extract_response_text(payload: Mapping[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response omitted candidates")
    content = candidates[0].get("content")
    if not isinstance(content, Mapping):
        raise ValueError("Gemini response omitted candidate content")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Gemini response omitted content parts")
    texts = [part.get("text") for part in parts if isinstance(part, Mapping)]
    if not texts or any(not isinstance(value, str) for value in texts):
        raise ValueError("Gemini response omitted text output")
    return "".join(str(value) for value in texts)


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or 500 <= exc.code < 600
    return isinstance(
        exc,
        (
            TimeoutError,
            socket.timeout,
            socket.gaierror,
            ConnectionError,
            urllib.error.URLError,
        ),
    )


def call_gemini(
    *,
    prompt: str,
    api_key: str,
    model: str,
    max_output_tokens: int,
    timeout: float,
    retries: int,
    post_json: PostJsonFn = _post_json,
) -> tuple[str, list[dict[str, Any]], float]:
    """Call one pinned Gemini model; never fall back to another model."""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    encoded = json.dumps(
        build_request_body(prompt, max_output_tokens=max_output_tokens),
        ensure_ascii=False,
    ).encode("utf-8")
    attempts: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for attempt_number in range(1, retries + 1):
        started = time.perf_counter()
        try:
            payload, http_status = post_json(url, encoded, api_key, timeout)
            raw = _extract_response_text(payload)
            attempts.append(
                {
                    "attempt_number": attempt_number,
                    "status": "ok",
                    "http_status": http_status,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            return raw, attempts, round((time.perf_counter() - total_started) * 1000, 3)
        except Exception as exc:  # noqa: BLE001 - preserve terminal call failure
            should_retry = _retryable(exc) and attempt_number < retries
            attempts.append(
                {
                    "attempt_number": attempt_number,
                    "status": "error",
                    "http_status": (
                        int(exc.code) if isinstance(exc, urllib.error.HTTPError) else None
                    ),
                    "retryable": _retryable(exc),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            if should_retry:
                time.sleep(min(15.0, float(3 * attempt_number)))
                continue
            raise RuntimeError(
                f"Gemini C2 request failed after {attempt_number} attempt(s): {exc}"
            ) from exc
    raise AssertionError("unreachable")


def _role_perspective(record: Mapping[str, Any]) -> str | None:
    role = record.get("role")
    if isinstance(role, str) and role.strip():
        return role.strip()
    mapping = record.get("role_mapping")
    if isinstance(mapping, Mapping):
        for key in ("perspective", "role", "label"):
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def build_c2_answer(
    *,
    base: Mapping[str, Any],
    raw_response: str,
    prompt: str,
    verified: Any,
    experiment_id: str,
    generation_run_id: str,
    model: str,
    max_output_tokens: int,
    attempts: Sequence[Mapping[str, Any]],
    latency_ms: float,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    """Project a verified C2 result into the Judge-compatible answer schema."""

    record = copy.deepcopy(dict(base))
    for key in (
        "answer_id",
        "answer_sha256",
        "record_sha256",
        "slot_outcome",
        "answer_eligible_for_judge",
        "service_error",
        "collection_attempt_number",
        "request_attempts",
        "health_request_attempts",
    ):
        record.pop(key, None)

    prompt_sha = sha256_text(prompt)
    system_sha = sha256_text(SYSTEM_INSTRUCTION)
    request_config = {
        "provider": "gemini",
        "model_requested": model,
        "api_style": "generateContent",
        "generation_config": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema_sha256": sha256_json(RESPONSE_JSON_SCHEMA),
        },
        "prompt_used": True,
        "system_instruction_sha256": system_sha,
    }
    collector_config = {
        "collector_version": COLLECTOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "grounded_schema_version": GROUNDED_SCHEMA_VERSION,
        "condition": "c2_quote_bound_generation",
        "source_condition": "c1",
        "source_artifact_sha256": source_artifact_sha256,
        "model": model,
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "response_json_schema_sha256": sha256_json(RESPONSE_JSON_SCHEMA),
        "fallback_models": [],
    }
    trace = record.get("evaluation_trace")
    trace = copy.deepcopy(trace) if isinstance(trace, Mapping) else {}
    trace.update(
        {
            "schema_version": 1,
            "generation_input": {
                "user_prompt": prompt,
                "system_instruction": SYSTEM_INSTRUCTION,
                "prompt_sha256": prompt_sha,
                "system_instruction_sha256": system_sha,
            },
            "raw_draft": raw_response,
            "sanitized_draft": verified.answer,
        }
    )
    timing = trace.get("timing_ms")
    timing = dict(timing) if isinstance(timing, Mapping) else {}
    timing["c2_generation"] = latency_ms
    trace["timing_ms"] = timing

    record.update(
        {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "record_type": "answer",
            "experiment_id": experiment_id,
            "condition_id": "c2",
            "generation_run_id": generation_run_id,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "answer": verified.answer,
            "cited_answer": verified.cited_answer,
            "claims": [dict(value) for value in verified.claims],
            "citations": [dict(value) for value in verified.citations],
            "collector_config": collector_config,
            "collector_config_sha256": sha256_json(collector_config),
            "generator": "gemini-quote-bound-c2",
            "generation": {
                "requested": "gemini",
                "used": "gemini",
                "model": model,
                "fallback_reason": None,
                "attempts": [dict(value) for value in attempts],
                "prompt_sha256": prompt_sha,
                "system_instruction_sha256": system_sha,
                "request_config": request_config,
                "request_config_sha256": sha256_json(request_config),
                "implementation": COLLECTOR_VERSION,
            },
            "postprocessing": {
                "mode": "quote-bound-fail-closed",
                "accepted_claim_count": verified.accepted_count,
                "rejected_claim_count": verified.rejected_count,
                "unanswered_count": len(verified.unanswered),
            },
            "postprocessor_diagnostic": {
                "schema_version": GROUNDED_SCHEMA_VERSION,
                "accepted_claim_count": verified.accepted_count,
                "rejected_claim_count": verified.rejected_count,
                "rejection_reasons": [
                    claim.get("validation_reason")
                    for claim in verified.claims
                    if not claim.get("supported")
                ],
            },
            "evaluation_trace": trace,
        }
    )
    return build_answer_identity(record)


def plan_collection(
    records: Sequence[Mapping[str, Any]],
    *,
    model: str,
    max_output_tokens: int,
) -> tuple[list[tuple[Mapping[str, Any], str]], dict[str, Any]]:
    planned: list[tuple[Mapping[str, Any], str]] = []
    for record in records:
        prompt = build_prompt(
            str(record.get("query") or ""),
            _final_contexts(record),
            role_perspective=_role_perspective(record),
        )
        planned.append((record, prompt))
    summary = {
        "collector_version": COLLECTOR_VERSION,
        "mode": "dry-run",
        "external_calls": 0,
        "planned_successful_calls": len(planned),
        "model": model,
        "max_output_tokens": max_output_tokens,
        "prompt_version": PROMPT_VERSION,
        "system_instruction_sha256": sha256_text(SYSTEM_INSTRUCTION),
        "response_json_schema_sha256": sha256_json(RESPONSE_JSON_SCHEMA),
        "prompts": [
            {
                "case_id": record["case_id"],
                "prompt_sha256": sha256_text(prompt),
                "context_count": len(_final_contexts(record)),
            }
            for record, prompt in planned
        ],
    }
    return planned, summary


def collect_planned(
    planned: Sequence[tuple[Mapping[str, Any], str]],
    *,
    api_key: str,
    experiment_id: str,
    generation_run_id: str,
    model: str,
    max_output_tokens: int,
    timeout: float,
    retries: int,
    sleep_seconds: float,
    source_artifact_sha256: str,
    post_json: PostJsonFn = _post_json,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for index, (base, prompt) in enumerate(planned):
        raw, attempts, latency_ms = call_gemini(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            retries=retries,
            post_json=post_json,
        )
        verified = verify_response(raw, _final_contexts(base))
        answer = build_c2_answer(
            base=base,
            raw_response=raw,
            prompt=prompt,
            verified=verified,
            experiment_id=experiment_id,
            generation_run_id=generation_run_id,
            model=model,
            max_output_tokens=max_output_tokens,
            attempts=attempts,
            latency_ms=latency_ms,
            source_artifact_sha256=source_artifact_sha256,
        )
        validate_answer_record(answer)
        collected.append(answer)
        if sleep_seconds and index + 1 < len(planned):
            time.sleep(sleep_seconds)
    return collected


def load_projection_answers(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load prior C2 rows whose raw model responses will be re-verified."""

    if _forbidden_dev_path(path):
        raise ValueError("C2 reprojection accepts DEV inputs only")
    reject_symlink_inputs([path])
    before = sha256_file(path)
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        validate_answer_record(value)
        if value.get("condition_id") != "c2":
            raise ValueError("reprojection source must be a C2 answer artifact")
        case_id = str(value.get("case_id") or "")
        if not case_id or case_id in rows:
            raise ValueError(f"duplicate or empty reprojection case id: {case_id}")
        trace = value.get("evaluation_trace")
        if not isinstance(trace, Mapping) or not isinstance(trace.get("raw_draft"), str):
            raise ValueError(f"C2 answer {case_id} omitted raw_draft")
        rows[case_id] = value
    if sha256_file(path) != before:
        raise ValueError("C2 projection source changed while it was being read")
    if not rows:
        raise ValueError("C2 projection source is empty")
    return rows, before


def reproject_planned(
    planned: Sequence[tuple[Mapping[str, Any], str]],
    projection_rows: Mapping[str, Mapping[str, Any]],
    *,
    experiment_id: str,
    generation_run_id: str,
    model: str,
    max_output_tokens: int,
    source_artifact_sha256: str,
    projection_artifact_sha256: str,
) -> list[dict[str, Any]]:
    """Apply the current verifier to prior raw responses without an API call."""

    expected_cases = {str(base["case_id"]) for base, _ in planned}
    if set(projection_rows) != expected_cases:
        raise ValueError(
            "reprojection case set mismatch: "
            f"missing={sorted(expected_cases - set(projection_rows))} "
            f"extra={sorted(set(projection_rows) - expected_cases)}"
        )
    results: list[dict[str, Any]] = []
    for base, prompt in planned:
        case_id = str(base["case_id"])
        source = projection_rows[case_id]
        generation = source.get("generation")
        if not isinstance(generation, Mapping):
            raise ValueError(f"C2 answer {case_id} omitted generation metadata")
        if generation.get("prompt_sha256") != sha256_text(prompt):
            raise ValueError(f"C2 answer {case_id} prompt hash changed")
        if generation.get("model") != model:
            raise ValueError(f"C2 answer {case_id} model mismatch")
        trace = source["evaluation_trace"]
        raw_response = str(trace["raw_draft"])
        verified = verify_response(raw_response, _final_contexts(base))
        attempts = generation.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(f"C2 answer {case_id} omitted request attempts")
        answer = build_c2_answer(
            base=base,
            raw_response=raw_response,
            prompt=prompt,
            verified=verified,
            experiment_id=experiment_id,
            generation_run_id=generation_run_id,
            model=model,
            max_output_tokens=max_output_tokens,
            attempts=attempts,
            latency_ms=float(source.get("latency_ms") or 0.0),
            source_artifact_sha256=source_artifact_sha256,
        )
        for key in ("answer_id", "answer_sha256", "record_sha256"):
            answer.pop(key, None)
        reprojection = {
            "mode": "offline_raw_response_reprojection",
            "source_artifact_sha256": projection_artifact_sha256,
            "source_answer_id": source["answer_id"],
            "source_answer_sha256": source["answer_sha256"],
            "source_record_sha256": source["record_sha256"],
            "new_external_calls": 0,
        }
        answer["collector_config"]["reprojection"] = reprojection
        answer["collector_config_sha256"] = sha256_json(answer["collector_config"])
        answer["generation"]["response_reused_from"] = reprojection
        answer = build_answer_identity(answer)
        validate_answer_record(answer)
        results.append(answer)
    return results


def _jsonl(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def _companion_paths(out: Path) -> tuple[Path, Path, Path]:
    return (
        out.with_name(out.name + ".summary.json"),
        out.with_name(out.name + ".partial.jsonl"),
        out.with_name(out.name + ".error.json"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--generation-run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", help="comma-separated DEV case ids")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reproject-from",
        type=Path,
        help="re-verify prior C2 raw responses locally; performs zero API calls",
    )
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument(
        "--authorize-experimental-collection",
        help=f"live collection requires exact phrase {AUTHORIZATION_PHRASE}",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_output_tokens <= 0 or args.timeout <= 0 or args.retries <= 0:
        parser.error("token, timeout, and retry values must be positive")
    if args.sleep < 0:
        parser.error("--sleep must be non-negative")
    for label, value in (
        ("--experiment-id", args.experiment_id),
        ("--generation-run-id", args.generation_run_id),
        ("--model", args.model),
    ):
        if not str(value).strip():
            parser.error(f"{label} must be non-empty")
    if _forbidden_dev_path(args.out):
        parser.error("C2 experiment output must not use holdout/draft paths")
    if args.answers.resolve() == args.out.resolve():
        parser.error("--answers and --out must be different paths")
    summary_path, partial_path, error_path = _companion_paths(args.out)
    try:
        require_new_outputs([args.out, summary_path, partial_path, error_path])
        only = [value.strip() for value in (args.only or "").split(",") if value.strip()]
        records, source_sha = load_dev_answers(args.answers, only=only, limit=args.limit)
        planned, plan = plan_collection(
            records,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    plan.update(
        {
            "source_artifact": str(args.answers),
            "source_artifact_sha256": source_sha,
            "experiment_id": args.experiment_id,
            "generation_run_id": args.generation_run_id,
        }
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.reproject_from is not None:
        try:
            projection_rows, projection_sha = load_projection_answers(
                args.reproject_from
            )
            collected = reproject_planned(
                planned,
                projection_rows,
                experiment_id=args.experiment_id,
                generation_run_id=args.generation_run_id,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                source_artifact_sha256=source_sha,
                projection_artifact_sha256=projection_sha,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        output_text = _jsonl(collected)
        summary = {
            **{
                key: value
                for key, value in plan.items()
                if key not in {"prompts", "external_calls"}
            },
            "mode": "offline-raw-response-reprojection",
            "external_calls": 0,
            "completed_answers": len(collected),
            "projection_source": str(args.reproject_from),
            "projection_source_sha256": projection_sha,
            "accepted_claims": sum(
                int(record["postprocessing"]["accepted_claim_count"])
                for record in collected
            ),
            "rejected_claims": sum(
                int(record["postprocessing"]["rejected_claim_count"])
                for record in collected
            ),
            "answers_sha256": sha256_text(output_text),
        }
        publish_immutable_texts(
            {
                args.out: output_text,
                summary_path: json.dumps(
                    summary, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
            },
            authoritative_path=args.out,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.authorize_experimental_collection != AUTHORIZATION_PHRASE:
        parser.error(
            "live C2 collection requires --authorize-experimental-collection "
            f"{AUTHORIZATION_PHRASE}"
        )
    if args.env_file.exists():
        load_env(args.env_file)
    api_key = str(os.environ.get(args.api_key_env) or "").strip()
    if not api_key:
        parser.error(f"missing API key in {args.api_key_env}")

    collected: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        # Collect one row at a time so a terminal error can preserve completed
        # rows under a distinct, immutable diagnostic path.
        for index, item in enumerate(planned):
            batch = collect_planned(
                [item],
                api_key=api_key,
                experiment_id=args.experiment_id,
                generation_run_id=args.generation_run_id,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=0.0,
                source_artifact_sha256=source_sha,
            )
            collected.extend(batch)
            print(f"[{index + 1}/{len(planned)}] {item[0]['case_id']}", flush=True)
            if args.sleep and index + 1 < len(planned):
                time.sleep(args.sleep)
    except Exception as exc:  # noqa: BLE001 - persist an auditable failed batch
        error = {
            **{
                key: value
                for key, value in plan.items()
                if key not in {"prompts", "external_calls"}
            },
            "mode": "live-error",
            "external_calls": None,
            "external_calls_note": (
                "terminal failures may include retries not represented by completed rows"
            ),
            "verified_successful_external_calls": len(collected),
            "completed_calls": len(collected),
            "failed_case_id": planned[len(collected)][0]["case_id"],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        outputs = {
            partial_path: _jsonl(collected),
            error_path: json.dumps(error, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        }
        publish_immutable_texts(outputs, authoritative_path=error_path)
        raise

    output_text = _jsonl(collected)
    summary = {
        **{
            key: value
            for key, value in plan.items()
            if key not in {"prompts", "external_calls"}
        },
        "mode": "live-complete",
        "external_calls": sum(
            len(record["generation"]["attempts"]) for record in collected
        ),
        "successful_external_calls": sum(
            sum(
                attempt.get("status") == "ok"
                for attempt in record["generation"]["attempts"]
            )
            for record in collected
        ),
        "completed_calls": len(collected),
        "accepted_claims": sum(
            int(record["postprocessing"]["accepted_claim_count"])
            for record in collected
        ),
        "rejected_claims": sum(
            int(record["postprocessing"]["rejected_claim_count"])
            for record in collected
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "answers_sha256": sha256_text(output_text),
    }
    publish_immutable_texts(
        {
            args.out: output_text,
            summary_path: json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        },
        authoritative_path=args.out,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
