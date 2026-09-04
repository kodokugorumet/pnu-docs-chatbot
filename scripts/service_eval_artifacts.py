#!/usr/bin/env python3
"""Stable identities and integrity checks for service evaluation artifacts.

Answer collection and LLM judging are deliberately separate.  This module keeps
their append-only JSONL files linked without copying mutable judge output into the
raw answer artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


ANSWER_SCHEMA_VERSION = "pnu.service-answer.v2"
_ALLOWED_NETWORK_ERROR_TYPES = {
    "URLError",
    "TimeoutError",
    "ConnectionError",
    "BrokenPipeError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "IncompleteRead",
    "BadStatusLine",
    "RemoteDisconnected",
    "ContentTooShortError",
}


_FINAL_CAPABILITY_SEAL = object()


class ExactFinalCollectionAuthorization:
    """Data-only authorization; evaluator dispatches to trusted validators.

    This prevents accidental/duck-typed programmatic bypass. Arbitrary
    same-process monkeypatching or private-sentinel introspection is explicitly
    outside the threat model; the process running a final collection must be
    trusted.
    """

    __slots__ = ("kind", "payload", "_seal", "_frozen")

    def __init__(self, kind: str, payload: Any, seal: object) -> None:
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("exact final authorization is immutable")
        object.__setattr__(self, name, value)

    @property
    def schedule(self) -> Any:
        return self.payload["schedule"]

    @property
    def entry(self) -> Any:
        return self.payload["entry"]

    @property
    def artifact(self) -> Any:
        return self.payload["artifact"]


def _issue_exact_final_authorization(kind: str, payload: Any) -> object:
    """Internal factory used only by the two canonical schedule runners."""

    if kind not in {"generation", "retrieval"}:
        raise ValueError("invalid final authorization kind")
    return ExactFinalCollectionAuthorization(kind, payload, _FINAL_CAPABILITY_SEAL)


def is_exact_final_authorization(value: object) -> bool:
    return (
        type(value) is ExactFinalCollectionAuthorization
        and value._seal is _FINAL_CAPABILITY_SEAL  # type: ignore[attr-defined]
        and value.kind in {"generation", "retrieval"}  # type: ignore[attr-defined]
    )
JUDGMENT_SCHEMA_VERSION = "pnu.service-judgment.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def record_sha256(record: dict[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return sha256_json(payload)


def _required_text(record: dict[str, Any], key: str) -> str:
    value = str(record.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing {key}")
    return value


def expected_answer_id(record: dict[str, Any]) -> str:
    identity = {
        "experiment_id": _required_text(record, "experiment_id"),
        "condition_id": _required_text(record, "condition_id"),
        "generation_run_id": _required_text(record, "generation_run_id"),
        "case_id": _required_text(record, "case_id"),
    }
    return f"answer_{sha256_json(identity)[:24]}"


def build_answer_identity(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value.pop("record_sha256", None)
    answer = value.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    expected_id = expected_answer_id(value)
    existing_id = value.get("answer_id")
    if existing_id not in (None, expected_id):
        raise ValueError(
            f"answer_id mismatch: expected {expected_id}, got {existing_id}"
        )
    value["answer_id"] = expected_id
    value["answer_sha256"] = sha256_text(answer)
    value["record_sha256"] = record_sha256(value)
    return value


def validate_answer_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != ANSWER_SCHEMA_VERSION:
        raise ValueError(
            f"invalid answer schema_version {record.get('schema_version')!r}"
        )
    if record.get("record_type") != "answer":
        raise ValueError("record_type must be 'answer'")
    case_id = _required_text(record, "case_id")
    legacy_id = record.get("id")
    if legacy_id is not None and str(legacy_id) != case_id:
        raise ValueError(f"id/case_id mismatch for {case_id}")
    expected_id = expected_answer_id(record)
    if record.get("answer_id") != expected_id:
        raise ValueError(
            f"answer_id mismatch for {case_id}: expected {expected_id}"
        )
    answer = record.get("answer")
    if not isinstance(answer, str):
        raise ValueError(f"answer must be a string for {case_id}")
    expected_sha = sha256_text(answer)
    if record.get("answer_sha256") != expected_sha:
        raise ValueError(
            f"answer_sha256 mismatch for {case_id}: expected {expected_sha}"
        )
    expected_record_sha = record_sha256(record)
    if record.get("record_sha256") != expected_record_sha:
        raise ValueError(
            f"record_sha256 mismatch for {case_id}: "
            f"expected {expected_record_sha}"
        )


def validate_final_generation_provenance(
    record: dict[str, Any],
    *,
    provider: str,
    model: str,
    max_output_tokens: int,
    require_collection_attempts: bool = False,
) -> None:
    """Validate the exact final-model request and evaluation trace."""

    if record.get("slot_outcome") == "service_error":
        if record.get("answer_eligible_for_judge") is not False:
            raise ValueError("service_error slot must be ineligible for judge")
        if record.get("answer") != "[SERVICE_ERROR]":
            raise ValueError("service_error slot must use sentinel answer")
        error = record.get("service_error")
        if not isinstance(error, dict) or set(error) != {
            "stage", "type", "message", "retryable", "http_status",
            "request_attempts",
        }:
            raise ValueError("service_error detail is invalid")
        attempts = error.get("request_attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
            raise ValueError("service_error request attempts are invalid")
        if record.get("collection_attempt_number") != 1:
            raise ValueError("terminal collection_attempt_number must be 1")
        for ordinal, attempt in enumerate(attempts, 1):
            if (
                not isinstance(attempt, dict)
                or attempt.get("attempt_number") != ordinal
                or attempt.get("status") not in {"ok", "error"}
            ):
                raise ValueError("service_error request attempt ordinal/status invalid")
        if record.get("request_attempts") != attempts:
            raise ValueError("service_error request attempts mismatch")
        if error.get("stage") == "chat_transport":
            if error.get("retryable") is not True or len(attempts) != 3:
                raise ValueError("transport service_error must exhaust 3 retries")
            for attempt in attempts:
                _validate_retryable_error_attempt(attempt)
            if (
                error.get("type") != attempts[-1].get("error_type")
                or error.get("message") != attempts[-1].get("error")
                or error.get("http_status") != attempts[-1].get("http_status")
            ):
                raise ValueError("transport service_error detail/attempt mismatch")
            return
        if error.get("stage") != "response_validation":
            raise ValueError("unsupported terminal service_error stage")
        if (
            error.get("retryable") is not False
            or error.get("http_status") is not None
            or error.get("type") != "AnswerPayloadError"
            or error.get("message") not in {
                "response answer must be a string",
                "response answer must be a non-empty string",
            }
        ):
            raise ValueError("response_validation terminal detail is invalid")
        _validate_success_request_attempts(record)
        return
    if require_collection_attempts:
        if record.get("slot_outcome") != "answer":
            raise ValueError("final answer slot_outcome must be 'answer'")
        if record.get("answer_eligible_for_judge") is not True:
            raise ValueError("answer slot must be eligible for judge")
    elif "slot_outcome" in record or "answer_eligible_for_judge" in record:
        if record.get("slot_outcome") != "answer":
            raise ValueError("final answer slot_outcome must be 'answer'")
        if record.get("answer_eligible_for_judge") is not True:
            raise ValueError("answer slot must be eligible for judge")
    if require_collection_attempts:
        _validate_success_request_attempts(record)
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("final answer omitted generation metadata")
    if generation.get("requested") != provider or generation.get("used") != provider:
        raise ValueError("final answer generation provider mismatch")
    if generation.get("model") != model:
        raise ValueError("final answer generation model mismatch")
    request_config = generation.get("request_config")
    if not isinstance(request_config, dict):
        raise ValueError("final answer omitted generation request_config")
    expected_keys = {
        "provider",
        "model_requested",
        "api_style",
        "generation_config",
        "prompt_used",
        "system_instruction_sha256",
    }
    if set(request_config) != expected_keys:
        raise ValueError("generation request_config has unapproved keys")
    generation_config = request_config.get("generation_config")
    if generation_config != {"maxOutputTokens": max_output_tokens}:
        raise ValueError(
            "generation_config must contain only the pinned maxOutputTokens"
        )
    expected_config = {
        "provider": "gemini",
        "model_requested": model,
        "api_style": "generateContent",
        "generation_config": {"maxOutputTokens": max_output_tokens},
        "prompt_used": True,
        "system_instruction_sha256": generation.get(
            "system_instruction_sha256"
        ),
    }
    if request_config != expected_config:
        raise ValueError("generation request_config values mismatch")
    if generation.get("request_config_sha256") != sha256_json(request_config):
        raise ValueError("generation request_config_sha256 mismatch")
    for key in ("prompt_sha256", "system_instruction_sha256"):
        value = str(generation.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"generation {key} is invalid")

    _validate_final_evaluation_trace(record, generation)


def _validate_final_evaluation_trace(
    record: dict[str, Any], generation: dict[str, Any]
) -> None:
    """Validate content-addressed prompt inputs and retrieval trace."""

    trace = record.get("evaluation_trace")
    if not isinstance(trace, dict) or trace.get("schema_version") != 1:
        raise ValueError("final answer omitted evaluation trace schema v1")
    generation_input = trace.get("generation_input")
    if not isinstance(generation_input, dict):
        raise ValueError("evaluation trace omitted generation_input")
    if generation_input.get("prompt_sha256") != generation.get("prompt_sha256"):
        raise ValueError("generation trace prompt SHA mismatch")
    if generation_input.get("system_instruction_sha256") != generation.get(
        "system_instruction_sha256"
    ):
        raise ValueError("generation trace system instruction SHA mismatch")
    for key in ("user_prompt", "system_instruction"):
        if not isinstance(generation_input.get(key), str):
            raise ValueError(f"generation_input omitted {key}")
    if sha256_text(generation_input["user_prompt"]) != generation.get("prompt_sha256"):
        raise ValueError("generation user prompt content SHA mismatch")
    if sha256_text(generation_input["system_instruction"]) != generation.get(
        "system_instruction_sha256"
    ):
        raise ValueError("generation system instruction content SHA mismatch")
    for key in ("raw_draft", "sanitized_draft"):
        if not isinstance(trace.get(key), str):
            raise ValueError(f"evaluation trace omitted {key}")
    stages = trace.get("retrieval_stages")
    if not isinstance(stages, dict) or not isinstance(
        stages.get("final_contexts"), list
    ):
        raise ValueError("evaluation trace omitted final retrieval contexts")


def validate_final_retrieval_provenance(
    record: dict[str, Any], *, require_collection_attempts: bool = False
) -> None:
    """Require the offline extractive generator for retrieval-only collection."""

    if require_collection_attempts:
        if record.get("slot_outcome") != "answer":
            raise ValueError("final retrieval slot_outcome must be 'answer'")
        if record.get("answer_eligible_for_judge") is not True:
            raise ValueError("final retrieval answer slot must be eligible")
        _validate_success_request_attempts(record)
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("final retrieval response omitted generation metadata")
    if generation.get("requested") != "extractive":
        raise ValueError("final retrieval must request the extractive provider")
    if generation.get("used") != "extractive" or generation.get("model") is not None:
        raise ValueError("final retrieval used an external generation provider")
    request_config = generation.get("request_config")
    expected_config = {
        "provider": "extractive",
        "model_requested": None,
        "prompt_used": False,
    }
    if request_config != expected_config:
        raise ValueError("final retrieval extractive request_config mismatch")
    if generation.get("request_config_sha256") != sha256_json(request_config):
        raise ValueError("final retrieval request_config_sha256 mismatch")
    for key in ("prompt_sha256", "system_instruction_sha256"):
        value = str(generation.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"generation {key} is invalid")
    _validate_final_evaluation_trace(record, generation)


def _validate_success_request_attempts(record: Mapping[str, Any]) -> None:
    """Validate the bounded technical-retry trace for a successful slot."""

    if record.get("collection_attempt_number") != 1:
        raise ValueError("successful collection_attempt_number must be 1")
    attempts = record.get("request_attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
        raise ValueError("successful request attempts must contain 1..3 rows")
    for ordinal, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, Mapping) or attempt.get(
            "attempt_number"
        ) != ordinal:
            raise ValueError("successful request attempt ordinals are invalid")
        _validate_attempt_elapsed(attempt)
        if ordinal == len(attempts):
            if set(attempt) != {"attempt_number", "status", "elapsed_ms"}:
                raise ValueError("final successful request attempt has invalid fields")
            if attempt.get("status") != "ok":
                raise ValueError("final request attempt must be successful")
            continue
        _validate_retryable_error_attempt(attempt)


def _validate_attempt_elapsed(attempt: Mapping[str, Any]) -> None:
    elapsed = attempt.get("elapsed_ms")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        raise ValueError("request attempt elapsed_ms is invalid")


def _validate_retryable_error_attempt(attempt: Mapping[str, Any]) -> None:
    if set(attempt) != {
        "attempt_number", "status", "elapsed_ms", "error_type", "error",
        "retryable", "http_status",
    }:
        raise ValueError("retryable request attempt has invalid fields")
    _validate_attempt_elapsed(attempt)
    error_type = str(attempt.get("error_type") or "").strip()
    message = str(attempt.get("error") or "").strip()
    status = attempt.get("http_status")
    if attempt.get("status") != "error" or attempt.get("retryable") is not True:
        raise ValueError("request attempt is not a retryable error")
    if not error_type or not message:
        raise ValueError("retryable request attempt error detail is empty")
    if status is None:
        if error_type not in _ALLOWED_NETWORK_ERROR_TYPES:
            raise ValueError("retryable request attempt is not a network error")
    elif (
        type(status) is not int
        or not (status in {408, 429} or 500 <= status <= 599)
        or error_type != "HTTPError"
    ):
        raise ValueError("retryable request attempt HTTP status/type is invalid")


def expected_judgment_id(record: dict[str, Any]) -> str:
    identity = {
        "experiment_id": _required_text(record, "experiment_id"),
        "answer_id": _required_text(record, "answer_id"),
        "judge_run_id": _required_text(record, "judge_run_id"),
    }
    return f"judgment_{sha256_json(identity)[:24]}"


def build_judgment_identity(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value.pop("record_sha256", None)
    _required_text(value, "answer_sha256")
    _required_text(value, "case_id")
    expected_id = expected_judgment_id(value)
    existing_id = value.get("judgment_id")
    if existing_id not in (None, expected_id):
        raise ValueError(
            f"judgment_id mismatch: expected {expected_id}, got {existing_id}"
        )
    value["judgment_id"] = expected_id
    value["record_sha256"] = record_sha256(value)
    return value


def validate_judgment_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != JUDGMENT_SCHEMA_VERSION:
        raise ValueError(
            "invalid judgment schema_version "
            f"{record.get('schema_version')!r}"
        )
    if record.get("record_type") != "judgment":
        raise ValueError("record_type must be 'judgment'")
    case_id = _required_text(record, "case_id")
    expected_id = expected_judgment_id(record)
    if record.get("judgment_id") != expected_id:
        raise ValueError(
            f"judgment_id mismatch for {case_id}: expected {expected_id}"
        )
    answer_sha = _required_text(record, "answer_sha256")
    if len(answer_sha) != 64 or any(
        character not in "0123456789abcdef" for character in answer_sha
    ):
        raise ValueError(f"invalid answer_sha256 for {case_id}")
    judge_config = record.get("judge_config")
    if isinstance(judge_config, dict) and judge_config:
        config_payload = dict(judge_config)
        embedded_config_sha = config_payload.pop("judge_config_sha256", None)
        expected_config_sha = sha256_json(config_payload)
        if embedded_config_sha not in (None, expected_config_sha):
            raise ValueError(f"judge_config_sha256 mismatch for {case_id}")
        if record.get("judge_config_sha256") != expected_config_sha:
            raise ValueError(f"judge_config_sha256 mismatch for {case_id}")
    expected_record_sha = record_sha256(record)
    if record.get("record_sha256") != expected_record_sha:
        raise ValueError(f"record_sha256 mismatch for {case_id}")


def load_unique_jsonl(path: Path, *, key: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        stable_key = str(record.get(key) or "")
        if not stable_key:
            raise ValueError(f"{path}:{line_number}: missing {key}")
        if stable_key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate {key} {stable_key}")
        seen.add(stable_key)
        records.append(record)
    return records


def join_answers_and_judgments(
    answers_by_case: dict[str, dict[str, Any]],
    judgments: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    answer_by_id: dict[str, dict[str, Any]] = {}
    for case_id, answer in answers_by_case.items():
        validate_answer_record(answer)
        if case_id != answer["case_id"]:
            raise ValueError(f"answer map key mismatch for {case_id}")
        answer_id = str(answer["answer_id"])
        if answer_id in answer_by_id:
            raise ValueError(f"duplicate answer_id {answer_id}")
        answer_by_id[answer_id] = answer

    judgment_by_answer: dict[str, dict[str, Any]] = {}
    seen_judgments: set[str] = set()
    for judgment in judgments:
        validate_judgment_record(judgment)
        judgment_id = str(judgment["judgment_id"])
        if judgment_id in seen_judgments:
            raise ValueError(f"duplicate judgment_id {judgment_id}")
        seen_judgments.add(judgment_id)
        answer_id = str(judgment["answer_id"])
        if answer_id in judgment_by_answer:
            raise ValueError(f"multiple judgments for answer_id {answer_id}")
        if answer_id not in answer_by_id:
            raise ValueError(f"judgment references unknown answer_id {answer_id}")
        judgment_by_answer[answer_id] = judgment

    joined: dict[str, dict[str, Any]] = {}
    for answer_id, answer in answer_by_id.items():
        judgment = judgment_by_answer.get(answer_id)
        if judgment is None:
            raise ValueError(f"missing judgment for answer_id {answer_id}")
        case_id = str(answer["case_id"])
        if judgment["case_id"] != case_id:
            raise ValueError(f"case_id mismatch for answer_id {answer_id}")
        if judgment["experiment_id"] != answer["experiment_id"]:
            raise ValueError(f"experiment_id mismatch for answer_id {answer_id}")
        if judgment["answer_sha256"] != answer["answer_sha256"]:
            raise ValueError(f"answer_sha256 mismatch for answer_id {answer_id}")
        judge = judgment.get("judge")
        score = judge.get("score") if isinstance(judge, dict) else None
        if score not in (0, 1, 2):
            raise ValueError(
                f"invalid or missing judge score for answer_id {answer_id}"
            )
        merged = dict(answer)
        merged["judge"] = dict(judge)
        merged["judgment"] = {
            "judgment_id": judgment["judgment_id"],
            "judge_run_id": judgment["judge_run_id"],
            "judge_config": judgment.get("judge_config") or {},
        }
        joined[case_id] = merged
    return joined
