#!/usr/bin/env python3
"""Shared fail-closed disposition checks for final generation logical slots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SERVICE_ERROR_SENTINEL = "[SERVICE_ERROR]"
ALLOWED_RESPONSE_VALIDATION_MESSAGES = {
    "response answer must be a string",
    "response answer must be a non-empty string",
}
ALLOWED_NETWORK_ERROR_TYPES = {
    "URLError",
    "TimeoutError",
    "ConnectionError",
    "BrokenPipeError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "IncompleteRead",
    "BadStatusLine",
}


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _label(source: Path | str, case_id: str) -> str:
    return f"{source}: {case_id}"


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_http_status(value: Any, label: str) -> None:
    if value is not None and (
        type(value) is not int or value < 100 or value > 599
    ):
        raise ValueError(f"{label} must be null or an HTTP status integer")


def answer_is_judge_eligible(
    record: Mapping[str, Any],
    *,
    source: Path | str,
    allow_legacy: bool,
) -> bool:
    """Return whether a logical slot may be judged after validating its shape.

    Final rows must be either a successful answer or one of the two allowed
    terminal outcomes produced by the frozen collector. Preflight rows may
    record a later durable collection attempt after a failed prior run, while
    final-authorized rows remain single-dispatch. Legacy DEV rows can be
    accepted only when every disposition field is absent.
    """

    case_id = str(record.get("case_id") or "<missing>")
    label = _label(source, case_id)
    outcome = record.get("slot_outcome")
    eligible = record.get("answer_eligible_for_judge")
    service_error = record.get("service_error")
    if outcome is None and eligible is None and service_error is None:
        if allow_legacy:
            return True
        raise ValueError(f"{label}.slot_outcome is required for final generation")

    collector = record.get("collector_config")
    if not isinstance(collector, Mapping) or not collector:
        raise ValueError(f"{label}: final slot omits collector_config")
    if record.get("collector_config_sha256") != _sha256_json(collector):
        raise ValueError(f"{label}: collector_config_sha256 mismatch")

    if outcome == "answer":
        if eligible is not True:
            raise ValueError(f"{label}: successful answer must be Judge-eligible")
        if service_error is not None:
            raise ValueError(f"{label}: successful answer has service_error detail")
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{label}: successful answer is empty")
        if answer == SERVICE_ERROR_SENTINEL:
            raise ValueError(f"{label}: successful answer uses error sentinel")
        max_attempts = collector.get("max_attempts")
        final_authorized = isinstance(
            collector.get("final_authorization"), Mapping
        )
        if final_authorized and max_attempts != 3:
            raise ValueError(f"{label}: final successful max_attempts must equal 3")
        if (
            not final_authorized
            and (
                type(max_attempts) is not int
                or max_attempts <= 0
            )
        ):
            raise ValueError(
                f"{label}: DEV successful max_attempts must be positive"
            )
        attempts = record.get("request_attempts")
        if (
            not isinstance(attempts, list)
            or not 1 <= len(attempts) <= max_attempts
            or any(not isinstance(attempt, Mapping) for attempt in attempts)
        ):
            raise ValueError(
                f"{label}: successful answer must preserve 1..3 request_attempts"
            )
        collection_attempt_number = record.get("collection_attempt_number")
        if final_authorized and collection_attempt_number != 1:
            raise ValueError(f"{label}: collection_attempt_number must equal 1")
        if (
            not final_authorized
            and (
                type(collection_attempt_number) is not int
                or collection_attempt_number <= 0
            )
        ):
            raise ValueError(
                f"{label}: collection_attempt_number must be positive"
            )
        if [attempt.get("attempt_number") for attempt in attempts] != list(
            range(1, len(attempts) + 1)
        ):
            raise ValueError(f"{label}: request-attempt ordinals are invalid")
        for index, attempt in enumerate(attempts):
            elapsed = attempt.get("elapsed_ms")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or elapsed < 0
            ):
                raise ValueError(f"{label}: request attempt elapsed_ms is invalid")
            is_last = index == len(attempts) - 1
            if is_last:
                if set(attempt) != {"attempt_number", "status", "elapsed_ms"}:
                    raise ValueError(f"{label}: invalid successful request fields")
                if attempt.get("status") != "ok":
                    raise ValueError(f"{label}: final request attempt must be ok")
                continue
            expected_error_fields = {
                "attempt_number",
                "status",
                "elapsed_ms",
                "error_type",
                "error",
                "retryable",
                "http_status",
            }
            if set(attempt) != expected_error_fields:
                raise ValueError(f"{label}: invalid retried request fields")
            if attempt.get("status") != "error" or attempt.get("retryable") is not True:
                raise ValueError(f"{label}: only retryable errors may precede success")
            error_type = _nonempty_text(
                attempt.get("error_type"), f"{label}: attempt error_type"
            )
            _nonempty_text(attempt.get("error"), f"{label}: attempt error")
            _validate_http_status(
                attempt.get("http_status"), f"{label}: attempt http_status"
            )
            status = attempt.get("http_status")
            if status is None and error_type not in ALLOWED_NETWORK_ERROR_TYPES:
                raise ValueError(f"{label}: invalid retryable network error type")
            if status is not None and (
                error_type != "HTTPError"
                or not (status in {408, 429} or 500 <= status <= 599)
            ):
                raise ValueError(f"{label}: non-retryable HTTP status before success")
        return True

    if outcome != "service_error":
        raise ValueError(
            f"{label}.slot_outcome must be 'answer' or 'service_error'"
        )
    if eligible is not False:
        raise ValueError(f"{label}: terminal slot must not be Judge-eligible")
    if record.get("answer") != SERVICE_ERROR_SENTINEL:
        raise ValueError(f"{label}: terminal slot must use {SERVICE_ERROR_SENTINEL}")

    required = {
        "stage",
        "type",
        "message",
        "retryable",
        "http_status",
        "request_attempts",
    }
    if not isinstance(service_error, Mapping):
        raise ValueError(f"{label}: terminal slot has no service_error object")
    missing = sorted(required - service_error.keys())
    unknown = sorted(service_error.keys() - required)
    if missing or unknown:
        raise ValueError(
            f"{label}: invalid service_error fields; "
            f"missing={missing or 'none'}, unknown={unknown or 'none'}"
        )
    stage = _nonempty_text(service_error.get("stage"), f"{label}.stage")
    error_type = _nonempty_text(service_error.get("type"), f"{label}.type")
    message = _nonempty_text(service_error.get("message"), f"{label}.message")
    retryable = service_error.get("retryable")
    if type(retryable) is not bool:
        raise ValueError(f"{label}.retryable must be boolean")
    _validate_http_status(service_error.get("http_status"), f"{label}.http_status")

    attempts = service_error.get("request_attempts")
    if (
        not isinstance(attempts, list)
        or not attempts
        or any(not isinstance(attempt, Mapping) for attempt in attempts)
    ):
        raise ValueError(f"{label}: terminal slot must preserve request_attempts")
    if record.get("request_attempts") != attempts:
        raise ValueError(f"{label}: top-level and service_error attempts differ")
    if [attempt.get("attempt_number") for attempt in attempts] != list(
        range(1, len(attempts) + 1)
    ):
        raise ValueError(f"{label}: request-attempt ordinals are invalid")
    for attempt in attempts:
        elapsed = attempt.get("elapsed_ms")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError(f"{label}: request attempt elapsed_ms is invalid")
        _validate_http_status(attempt.get("http_status"), f"{label}: attempt http_status")

    max_attempts = collector.get("max_attempts")
    if max_attempts != 3:
        raise ValueError(f"{label}: final terminal max_attempts must equal 3")
    if record.get("collection_attempt_number") != 1:
        raise ValueError(f"{label}: collection_attempt_number must equal 1")

    if stage == "chat_transport":
        if retryable is not True or len(attempts) != max_attempts:
            raise ValueError(
                f"{label}: retryable transport did not exhaust its attempt budget"
            )
        for attempt in attempts:
            if attempt.get("status") != "error" or attempt.get("retryable") is not True:
                raise ValueError(f"{label}: invalid retryable transport attempt")
            expected_attempt_fields = {
                "attempt_number",
                "status",
                "elapsed_ms",
                "error_type",
                "error",
                "retryable",
                "http_status",
            }
            if set(attempt) != expected_attempt_fields:
                raise ValueError(f"{label}: invalid transport attempt fields")
            attempt_error_type = _nonempty_text(
                attempt.get("error_type"), f"{label}: attempt error_type"
            )
            _nonempty_text(attempt.get("error"), f"{label}: attempt error")
            status = attempt.get("http_status")
            if status is None and attempt_error_type not in ALLOWED_NETWORK_ERROR_TYPES:
                raise ValueError(f"{label}: invalid retryable network error type")
            if status is not None and (
                attempt_error_type != "HTTPError"
                or not (status in {408, 429} or 500 <= status <= 599)
            ):
                raise ValueError(f"{label}: non-retryable HTTP status in terminal")
        if service_error.get("http_status") != attempts[-1].get("http_status"):
            raise ValueError(f"{label}: terminal HTTP status differs from last attempt")
        if error_type != attempts[-1].get("error_type"):
            raise ValueError(f"{label}: terminal error type differs from last attempt")
        if message != attempts[-1].get("error"):
            raise ValueError(f"{label}: terminal error message differs from last attempt")
        return False

    if stage == "response_validation":
        if (
            error_type != "AnswerPayloadError"
            or message not in ALLOWED_RESPONSE_VALIDATION_MESSAGES
            or retryable is not False
            or service_error.get("http_status") is not None
            or len(attempts) != 1
            or attempts[0].get("status") != "ok"
            or set(attempts[0])
            != {"attempt_number", "status", "elapsed_ms"}
        ):
            raise ValueError(
                f"{label}: response-validation terminal is not an allowed "
                "one-request empty/missing-answer outcome"
            )
        return False

    raise ValueError(
        f"{label}: poisoned/control failure cannot be a terminal slot: {stage!r}"
    )
