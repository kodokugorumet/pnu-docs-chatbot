#!/usr/bin/env python3
"""부산대 서비스 평가셋의 답변을 실제 서비스 경로에서 수집한다.

연구비 벤치마크와 달리 검색·생성을 직접 조립하지 않고, 실행 중인
search_api의 POST /chat을 호출한다. 역할 매핑·프롬프트 조립·공급자
폴백·근거 검증까지 사용자가 실제로 받는 경로 그대로를 측정하기 위함이다.

이 스크립트는 성공한 raw answer artifact만 append한다. 실패는 answer_id와
충돌하지 않는 attempt_id를 가진 ``*.errors.jsonl`` sidecar에 append한다. LLM
judge는 완성된 answer JSONL의 hash를 고정한 다음 judge_service_answers.py로
별도 실행한다. 따라서 재채점이나 재개가 원본 답변·실패 기록을 수정하지 않는다.

append + stable answer_id 재개 방식. 사용 예:
  python3 scripts/evaluate_service_answers.py \
      --api-base http://127.0.0.1:8000 \
      --experiment-id final-20260909 --condition-id c1 --generation-run-id run1 \
      --provider frontier --model gemini-3.1-flash-lite \
      --parser-profile cascade --retrieval-mode bm25 \
      --expected-corpus-revision REVISION \
      --expected-retrieval-tuning on \
      --expected-context-chunks-per-document 2 --eval-trace \
      --out processed/eval/final-20260909/c1-run1.answers.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from service_eval_artifacts import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    build_answer_identity,
    expected_answer_id,
    load_unique_jsonl,
    sha256_json,
    validate_answer_record,
    validate_final_generation_provenance,
    validate_final_retrieval_provenance,
)
from holdout_gold import match_evidence  # noqa: E402

DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-eval.jsonl"
ERROR_SCHEMA_VERSION = "pnu.service-answer-attempt.v1"
ATOMIC_EVIDENCE_MATCHER_VERSION = "pnu.atomic-evidence-nfkc-fuzzy-table-v2"

T = TypeVar("T")


class RequestFailure(RuntimeError):
    """A terminal HTTP/network request failure with its immutable attempt log."""

    def __init__(
        self,
        cause: BaseException,
        *,
        attempts: list[dict[str, Any]],
        retryable: bool,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts
        self.retryable = retryable


class AnswerPayloadError(RuntimeError):
    """A non-retryable malformed/empty service answer."""


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


def retrieval_hit(results: list[dict], expected: dict, k: int) -> dict:
    """응답 results(청크 순위)를 문서 단위로 접어 기대 문서의 순위를 찾는다.

    적중 기준은 제목 부분일치. 동일 문서가 여러 호스트에 중복 수집된
    사본도 정답 문서이므로 호스트 불일치는 미적중이 아니라 진단 기록
    (host_match=False)으로만 남긴다.
    """
    want_title = _norm(expected.get("source_title_contains", ""))
    want_host = _norm(expected.get("source_host", ""))
    doc_rank = 0
    seen_docs: set[str] = set()
    for row in results:
        doc_key = str(row.get("document_id") or row.get("doc_id") or row.get("chunk_id"))
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        doc_rank += 1
        if doc_rank > k:
            break
        if want_title and want_title in _norm(row.get("source_title")):
            host_match = not want_host or want_host == _norm(row.get("source_host"))
            return {"matched": True, "rank": doc_rank, "host_match": host_match}
    return {"matched": False, "rank": None, "host_match": None}


def evidence_hit(results: list[dict], case: dict, *, k: int) -> dict:
    """정답 근거 청크가 상위 ``k``개 컨텍스트에 실제로 들어갔는지.

    문서 단위 적중(retrieval_hit)과 별개 지표: 문서는 맞는데 정답 사실이
    담긴 청크가 빠지면 생성이 회피·부분 답변으로 빠진다(과잉 회피 진단용).

    현재 DEV gold는 atomic claim이 아니라 chunk ID 목록이므로 ``matched``는
    Any-Required-Gold-Chunk@k, ``all_matched``는
    All-Required-Gold-Chunks@k다. ``required_for_answer=false``인 참고 근거는
    답변 완전성 및 retrieval recall의 분모에서 제외한다. 복수 필수 gold
    chunk의 부분 회수를 숨기지 않도록 gold-chunk recall도 함께 반환한다.
    Atomic Evidence Recall은 required_claims/evidence_options를 갖는 holdout
    v2에서만 별도로 계산한다.
    """
    cutoff = max(0, int(k))
    wanted: set[str] = set()
    for index, item in enumerate(case.get("evidence") or []):
        if not isinstance(item, dict):
            raise ValueError(f"evidence[{index}] must be an object")
        required = item.get("required_for_answer", True)
        if not isinstance(required, bool):
            raise ValueError(
                f"evidence[{index}].required_for_answer must be boolean"
            )
        if required and item.get("chunk_id"):
            wanted.add(str(item["chunk_id"]))
    returned = {
        str(row.get("chunk_id"))
        for row in results[:cutoff]
        if row.get("chunk_id")
    }
    found = sorted(wanted & returned)
    missing = sorted(wanted - returned)
    recall = len(found) / len(wanted) if wanted else 0.0
    return {
        "k": cutoff,
        "matched": bool(found),
        "all_matched": bool(wanted) and not missing,
        "recall": recall,
        "found": found,
        "missing": missing,
        "wanted": len(wanted),
    }


def validate_atomic_evidence_schema(case: dict[str, Any]) -> None:
    required_claims = case.get("required_claims")
    if required_claims in (None, []):
        return
    if not isinstance(required_claims, list):
        raise ValueError("required_claims must be a list")
    seen_claim_ids: set[str] = set()
    for claim_index, claim in enumerate(required_claims):
        if not isinstance(claim, dict):
            raise ValueError(f"required_claims[{claim_index}] must be an object")
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            raise ValueError(f"required_claims[{claim_index}] is missing claim_id")
        if claim_id in seen_claim_ids:
            raise ValueError(f"duplicate required claim_id {claim_id!r}")
        seen_claim_ids.add(claim_id)
        options = claim.get("evidence_options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"required claim {claim_id!r} has no evidence_options")
        for option_index, option in enumerate(options):
            if not isinstance(option, dict):
                raise ValueError(
                    f"required claim {claim_id!r} evidence option "
                    f"{option_index} must be an object"
                )
            if not str(option.get("document_id") or "").strip():
                raise ValueError(
                    f"required claim {claim_id!r} evidence option "
                    f"{option_index} is missing document_id"
                )
            if not str(option.get("quote") or "").strip():
                raise ValueError(
                    f"required claim {claim_id!r} evidence option "
                    f"{option_index} is missing quote"
                )


def atomic_evidence_hit(
    results: list[dict[str, Any]], case: dict[str, Any], *, k: int
) -> dict[str, Any]:
    """Compute required-claim recall using alternative evidence options.

    This interface is only active for holdout-v2 rows with ``required_claims``.
    DEV's flat ``evidence[]`` metrics remain unchanged and are still reported as
    Any/All-Gold-Chunk coverage rather than atomic evidence recall.
    """

    validate_atomic_evidence_schema(case)
    cutoff = max(0, int(k))
    required_claims = case.get("required_claims")
    if not isinstance(required_claims, list) or not required_claims:
        return {
            "available": False,
            "matcher_version": ATOMIC_EVIDENCE_MATCHER_VERSION,
            "k": cutoff,
            "matched": False,
            "all_matched": False,
            "recall": None,
            "found_claim_ids": [],
            "missing_claim_ids": [],
            "wanted": 0,
            "claims": [],
        }

    claim_results: list[dict[str, Any]] = []
    found: list[str] = []
    missing: list[str] = []
    contexts = [row for row in results[:cutoff] if isinstance(row, dict)]
    for claim in required_claims:
        assert isinstance(claim, dict)
        claim_id = str(claim.get("claim_id") or "").strip()
        options = claim.get("evidence_options")
        assert isinstance(options, list) and options

        matched_option: dict[str, Any] | None = None
        for result_index, result in enumerate(contexts):
            for option_index, option in enumerate(options):
                assert isinstance(option, dict)
                match = match_evidence(result, option, claim=claim)
                if match.get("matched") is True:
                    matched_option = {
                        "option_index": option_index,
                        "result_rank": result_index + 1,
                        "method": match.get("method"),
                        "reason": match.get("reason"),
                        "ratio": match.get("ratio"),
                        "document_id_match": match.get("document_id_match"),
                    }
                    break
            if matched_option is not None:
                break
        is_matched = matched_option is not None
        (found if is_matched else missing).append(claim_id)
        claim_results.append(
            {
                "claim_id": claim_id,
                "matched": is_matched,
                "match": matched_option,
            }
        )

    recall = len(found) / len(required_claims)
    return {
        "available": True,
        "matcher_version": ATOMIC_EVIDENCE_MATCHER_VERSION,
        "k": cutoff,
        "matched": bool(found),
        "all_matched": not missing,
        "recall": recall,
        "found_claim_ids": found,
        "missing_claim_ids": missing,
        "wanted": len(required_claims),
        "claims": claim_results,
    }


def final_evaluation_contexts(
    response: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Return the untruncated contexts actually supplied to generation.

    Public ``results`` intentionally contain previews, so they cannot support a
    quote-level atomic evidence metric. Final runs obtain full text from the
    eval-only trace. ``None`` means the required trace is unavailable; an empty
    list is a valid observation with zero retrieved contexts.
    """

    trace = response.get("evaluation_trace")
    if not isinstance(trace, dict) or trace.get("schema_version") != 1:
        return None
    stages = trace.get("retrieval_stages")
    if not isinstance(stages, dict):
        return None
    contexts = stages.get("final_contexts")
    if not isinstance(contexts, list):
        return None
    if not all(isinstance(context, dict) for context in contexts):
        return None
    public_results = response.get("results")
    if not isinstance(public_results, list) or len(public_results) != len(contexts):
        return None
    for index, (context, public_result) in enumerate(
        zip(contexts, public_results), start=1
    ):
        if not isinstance(public_result, dict):
            return None
        text = context.get("text")
        chunk_id = str(context.get("chunk_id") or "")
        document_id = str(
            context.get("document_id") or context.get("doc_id") or ""
        )
        if not isinstance(text, str) or not text.strip() or not chunk_id or not document_id:
            return None
        if context.get("rank") != index or context.get("stage") != "final_contexts":
            return None
        expected_text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if context.get("text_sha256") != expected_text_sha:
            return None
        public_chunk_id = str(public_result.get("chunk_id") or "")
        public_document_id = str(
            public_result.get("document_id") or public_result.get("doc_id") or ""
        )
        if chunk_id != public_chunk_id or document_id != public_document_id:
            return None
    return list(contexts)


def build_chat_body(
    case: dict,
    *,
    provider: str | None,
    model: str | None,
    top_k: int,
    parser_profile: str | None,
    retrieval_mode: str | None,
    institution: str | None = None,
    eval_trace: bool = False,
) -> dict:
    """Build an explicit, reproducible service request for one eval case."""
    body: dict = {
        "question": case["query"],
        "top_k": int(top_k),
        "institution": institution,
    }
    if case.get("role"):
        body["role"] = case["role"]
    if provider:
        body["provider"] = provider
    if model:
        body["model"] = model
    if parser_profile:
        body["parser_profile"] = parser_profile
    if retrieval_mode:
        body["retrieval_mode"] = retrieval_mode
    if eval_trace:
        body["eval_trace"] = True
    return body


def call_chat(
    api_base: str,
    case: dict,
    provider: str | None,
    model: str | None,
    timeout: float,
    *,
    top_k: int,
    parser_profile: str | None,
    retrieval_mode: str | None,
    institution: str | None,
    eval_trace: bool,
) -> dict:
    body = build_chat_body(
        case,
        provider=provider,
        model=model,
        top_k=top_k,
        parser_profile=parser_profile,
        retrieval_mode=retrieval_mode,
        institution=institution,
        eval_trace=eval_trace,
    )
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat",
        json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def call_health(api_base: str, timeout: float) -> dict:
    with urllib.request.urlopen(
        f"{api_base.rstrip('/')}/health", timeout=timeout
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("/health returned a non-object payload")
    return value


def _retryable_request_error(error: BaseException) -> tuple[bool, int | None]:
    if isinstance(error, urllib.error.HTTPError):
        status = int(error.code)
        return status in {408, 429} or 500 <= status <= 599, status
    if isinstance(
        error,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.BadStatusLine,
        ),
    ):
        return True, None
    return False, None


def _request_attempt(
    *,
    attempt_number: int,
    status: str,
    elapsed_ms: float,
    error: BaseException | None = None,
    retryable: bool | None = None,
    http_status: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "attempt_number": int(attempt_number),
        "status": status,
        "elapsed_ms": round(elapsed_ms, 3),
    }
    if error is not None:
        value.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "retryable": bool(retryable),
                "http_status": http_status,
            }
        )
    return value


def call_with_technical_retries(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    backoff_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[T, list[dict[str, Any]]]:
    """Run an HTTP operation with a narrow, reproducible retry policy.

    Only timeout/network errors and HTTP 408, 429, or 5xx are retried. HTTP
    400/401/403, JSON/response validation errors, and experiment-control
    mismatches escape immediately without another request.
    """

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be non-negative")
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            value = operation()
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.BadStatusLine,
        ) as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            retryable, http_status = _retryable_request_error(error)
            attempts.append(
                _request_attempt(
                    attempt_number=attempt_number,
                    status="error",
                    elapsed_ms=elapsed_ms,
                    error=error,
                    retryable=retryable,
                    http_status=http_status,
                )
            )
            if not retryable or attempt_number >= max_attempts:
                raise RequestFailure(
                    error,
                    attempts=attempts,
                    retryable=retryable,
                ) from error
            delay = backoff_seconds * (2 ** (attempt_number - 1))
            sleep_fn(delay)
        else:
            attempts.append(
                _request_attempt(
                    attempt_number=attempt_number,
                    status="ok",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
            )
            return value, attempts
    raise AssertionError("unreachable retry loop")


def validate_health_controls(
    health: dict,
    *,
    parser_profile: str,
    retrieval_mode: str,
    expected_corpus_revision: str | None,
    expected_retrieval_tuning: bool | None,
    expected_context_chunks_per_document: int | None,
    expected_generation_provider: str | None = None,
    expected_generation_model: str | None = None,
    expected_generation_max_context_chars: int | None = None,
    expected_generation_max_output_tokens: int | None = None,
    expected_generation_sampling_parameters: list[str] | None = None,
    expected_git_commit: str | None = None,
    expected_index_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    require_eval_trace: bool = False,
) -> dict:
    if health.get("ready") is not True:
        raise RuntimeError("service health is not ready")
    profiles = health.get("parser_profiles")
    profiles = profiles if isinstance(profiles, list) else []
    profile = next(
        (
            value
            for value in profiles
            if isinstance(value, dict) and value.get("id") == parser_profile
        ),
        None,
    )
    if profile is None or profile.get("ready") is not True:
        raise RuntimeError(f"parser profile {parser_profile!r} is not ready")
    corpus_revision = str(profile.get("corpus_revision") or "")
    if (
        expected_corpus_revision is not None
        and corpus_revision != expected_corpus_revision
    ):
        raise RuntimeError(
            "health corpus revision mismatch: "
            f"expected {expected_corpus_revision!r}, got {corpus_revision!r}"
        )
    modes = profile.get("retrieval_modes")
    modes = modes if isinstance(modes, list) else []
    mode = next(
        (
            value
            for value in modes
            if isinstance(value, dict) and value.get("id") == retrieval_mode
        ),
        None,
    )
    if mode is None or mode.get("ready") is not True:
        raise RuntimeError(f"retrieval mode {retrieval_mode!r} is not ready")

    service_config = health.get("service_config")
    if not isinstance(service_config, dict):
        raise RuntimeError("health omitted service_config")
    actual_tuning = service_config.get("retrieval_tuning")
    if (
        expected_retrieval_tuning is not None
        and actual_tuning is not expected_retrieval_tuning
    ):
        raise RuntimeError(
            "retrieval tuning mismatch: "
            f"expected {expected_retrieval_tuning}, got {actual_tuning!r}"
        )
    actual_cap = service_config.get("context_chunks_per_document")
    if (
        expected_context_chunks_per_document is not None
        and actual_cap != expected_context_chunks_per_document
    ):
        raise RuntimeError(
            "context chunk cap mismatch: "
            f"expected {expected_context_chunks_per_document}, got {actual_cap!r}"
        )
    if require_eval_trace and service_config.get("evaluation_trace_enabled") is not True:
        raise RuntimeError("server evaluation trace is not enabled")
    generation_config = service_config.get("generation")
    if not isinstance(generation_config, dict):
        if any(
            value is not None
            for value in (
                expected_generation_max_context_chars,
                expected_generation_max_output_tokens,
                expected_generation_sampling_parameters,
            )
        ):
            raise RuntimeError("health omitted effective generation controls")
        generation_config = {}
    if expected_generation_provider is not None:
        if generation_config.get("provider") != expected_generation_provider:
            raise RuntimeError("health generation provider mismatch")
        if generation_config.get("configured") is not True:
            raise RuntimeError("health generation provider is not configured")
    if expected_generation_model is not None:
        allowed = generation_config.get("allowed_models")
        models = generation_config.get("models")
        if (
            not isinstance(allowed, list)
            or expected_generation_model not in allowed
            or not isinstance(models, dict)
            or not isinstance(models.get(expected_generation_model), dict)
        ):
            raise RuntimeError("health requested generation model is not ready")
        effective_generation = models[expected_generation_model]
    else:
        effective_generation = generation_config
    generation_expectations = {
        "max_context_chars": expected_generation_max_context_chars,
        "max_output_tokens": expected_generation_max_output_tokens,
        "sampling_parameters": expected_generation_sampling_parameters,
    }
    for key, expected in generation_expectations.items():
        if expected is not None and effective_generation.get(key) != expected:
            raise RuntimeError(
                f"health generation {key} mismatch: expected {expected!r}, "
                f"got {effective_generation.get(key)!r}"
            )
    freeze = service_config.get("freeze")
    if expected_git_commit is not None or expected_index_sha256 is not None:
        if not isinstance(freeze, dict):
            raise RuntimeError("health omitted freeze metadata")
        if freeze.get("startup_worktree_clean") is not True:
            raise RuntimeError("health reports a dirty Git worktree")
        if (
            expected_git_commit is not None
            and freeze.get("startup_git_commit") != expected_git_commit
        ):
            raise RuntimeError("health Git commit mismatch")
        if (
            expected_index_sha256 is not None
            and profile.get("index_sha256") != expected_index_sha256
        ):
            raise RuntimeError("health index SHA-256 mismatch")
        if (
            expected_source_manifest_sha256 is not None
            and profile.get("source_manifest_sha256")
            != expected_source_manifest_sha256
        ):
            raise RuntimeError("health source manifest SHA-256 mismatch")
        for key in ("startup_code_sha256", "process_started_at"):
            if not isinstance(freeze.get(key), str) or not freeze[key]:
                raise RuntimeError(f"health omitted {key}")
    else:
        freeze = freeze if isinstance(freeze, dict) else {}
    return {
        "parser_profile": parser_profile,
        "retrieval_mode": retrieval_mode,
        "corpus_revision": corpus_revision or None,
        "retrieval_tuning": actual_tuning,
        "context_chunks_per_document": actual_cap,
        "evaluation_trace_enabled": service_config.get(
            "evaluation_trace_enabled"
        ),
        "generation": {
            "provider": generation_config.get("provider"),
            "configured": generation_config.get("configured"),
            "model": expected_generation_model,
            **dict(effective_generation),
        },
        "freeze": {
            key: freeze.get(key)
            for key in (
                "startup_git_commit", "startup_worktree_clean",
                "startup_code_sha256", "process_started_at",
            )
        },
        "profile_index": {
            "sha256": profile.get("index_sha256"),
            "size_bytes": profile.get("index_size_bytes"),
            "source_manifest_sha256": profile.get("source_manifest_sha256"),
        },
    }


def _public_provider_name(provider: str) -> str:
    return "frontier" if provider == "gemini" else provider


def validate_response_controls(
    response: dict,
    *,
    provider: str,
    model: str | None,
    parser_profile: str,
    retrieval_mode: str,
    expected_corpus_revision: str | None,
    institution: str | None = None,
    require_eval_trace: bool = False,
) -> None:
    """Fail closed when the live server did not run the pinned condition."""

    actual_profile = str(response.get("parser_profile") or "")
    if actual_profile != parser_profile:
        raise RuntimeError(
            "parser profile mismatch: "
            f"expected {parser_profile!r}, got {actual_profile!r}"
        )
    actual_mode = str(response.get("retrieval_mode") or "")
    if actual_mode != retrieval_mode:
        raise RuntimeError(
            "retrieval mode mismatch: "
            f"expected {retrieval_mode!r}, got {actual_mode!r}"
        )
    if response.get("institution") != institution:
        raise RuntimeError(
            "institution mismatch: "
            f"expected {institution!r}, got {response.get('institution')!r}"
        )

    generation = response.get("generation")
    if not isinstance(generation, dict):
        raise RuntimeError("response omitted generation metadata")
    expected_provider = _public_provider_name(provider)
    requested = str(generation.get("requested") or "")
    used = str(generation.get("used") or "")
    if requested != expected_provider:
        raise RuntimeError(
            "generation provider request mismatch: "
            f"expected {expected_provider!r}, got {requested!r}"
        )
    if provider != "auto" and used != expected_provider:
        raise RuntimeError(
            "generation provider fallback: "
            f"expected {expected_provider!r}, used {used!r}"
        )
    if model is not None and str(generation.get("model") or "") != model:
        raise RuntimeError(
            "generation model mismatch: "
            f"expected {model!r}, got {generation.get('model')!r}"
        )

    if expected_corpus_revision:
        results = response.get("results") or []
        actual_revisions = {
            str(result.get("corpus_revision") or "")
            for result in results
            if isinstance(result, dict)
        }
        if results and actual_revisions != {expected_corpus_revision}:
            raise RuntimeError(
                "corpus revision mismatch: "
                f"expected {expected_corpus_revision!r}, "
                f"got {sorted(actual_revisions)!r}"
            )

    if require_eval_trace:
        trace = response.get("evaluation_trace")
        if not isinstance(trace, dict):
            raise RuntimeError("response omitted evaluation trace")
        generation_input = trace.get("generation_input")
        if response.get("results") and not isinstance(generation_input, dict):
            raise RuntimeError("evaluation trace omitted generation input")
        if isinstance(generation_input, dict):
            trace_prompt_sha = str(
                generation_input.get("prompt_sha256") or ""
            )
            generation_prompt_sha = str(
                generation.get("prompt_sha256") or ""
            )
            if (
                len(trace_prompt_sha) != 64
                or trace_prompt_sha != generation_prompt_sha
            ):
                raise RuntimeError(
                    "generation prompt hash mismatch between trace and provider metadata"
                )
            request_config_sha = str(
                generation.get("request_config_sha256") or ""
            )
            if len(request_config_sha) != 64:
                raise RuntimeError("generation request config hash is missing")


def build_record_base(
    case: dict,
    *,
    experiment_id: str,
    condition_id: str,
    generation_run_id: str,
    collector_config: dict,
) -> dict:
    case_id = str(case["id"])
    record = {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "record_type": "answer",
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "generation_run_id": generation_run_id,
        "case_id": case_id,
        # Retained for the existing retrieval analyzers and review UI.
        "id": case_id,
        "case_sha256": sha256_json(case),
        "collector_config": collector_config,
        "collector_config_sha256": sha256_json(collector_config),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "category": case.get("category"),
        "role": case.get("role"),
        "query": case["query"],
    }
    record["answer_id"] = expected_answer_id(record)
    return record


def validate_answer_payload(response: Any) -> str:
    if not isinstance(response, dict):
        raise AnswerPayloadError("response payload must be an object")
    answer = response.get("answer")
    if not isinstance(answer, str):
        raise AnswerPayloadError("response answer must be a string")
    if not answer.strip():
        raise AnswerPayloadError("response answer must be a non-empty string")
    return answer


def build_collection_error_record(
    base_record: dict[str, Any],
    *,
    attempt_number: int,
    stage: str,
    error: BaseException,
    request_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an append-only failure event that cannot collide with answer_id."""

    if attempt_number <= 0:
        raise ValueError("attempt_number must be positive")
    logical_answer_id = str(base_record.get("answer_id") or "")
    if not logical_answer_id:
        raise ValueError("base record is missing answer_id")
    identity = {
        "logical_answer_id": logical_answer_id,
        "collector_config_sha256": base_record.get("collector_config_sha256"),
        "attempt_number": int(attempt_number),
    }
    attempt_id = f"attempt_{sha256_json(identity)[:24]}"
    retryable, http_status = _retryable_request_error(error)
    record: dict[str, Any] = {
        "schema_version": ERROR_SCHEMA_VERSION,
        "record_type": "answer_collection_error",
        "attempt_id": attempt_id,
        "attempt_number": int(attempt_number),
        "logical_answer_id": logical_answer_id,
        "experiment_id": base_record.get("experiment_id"),
        "condition_id": base_record.get("condition_id"),
        "generation_run_id": base_record.get("generation_run_id"),
        "case_id": base_record.get("case_id"),
        # Generic artifact tooling uses `id` for non-answer/non-judgment rows.
        "id": attempt_id,
        "case_sha256": base_record.get("case_sha256"),
        "collector_config": base_record.get("collector_config"),
        "collector_config_sha256": base_record.get("collector_config_sha256"),
        "health_request_attempts": base_record.get("health_request_attempts") or [],
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "retryable": retryable,
            "http_status": http_status,
        },
        "request_attempts": request_attempts,
    }
    if base_record.get("call_order") is not None:
        record["call_order"] = base_record["call_order"]
    if isinstance(base_record.get("collection_schedule"), dict):
        record["collection_schedule"] = dict(
            base_record["collection_schedule"]
        )
    record["record_sha256"] = sha256_json(record)
    return record


def _validate_collection_error_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != ERROR_SCHEMA_VERSION:
        raise ValueError("invalid collection error schema_version")
    if record.get("record_type") != "answer_collection_error":
        raise ValueError("invalid collection error record_type")
    attempt_number = record.get("attempt_number")
    if not isinstance(attempt_number, int) or attempt_number <= 0:
        raise ValueError("invalid collection error attempt_number")
    logical_answer_id = str(record.get("logical_answer_id") or "")
    if not logical_answer_id:
        raise ValueError("collection error is missing logical_answer_id")
    identity = {
        "logical_answer_id": logical_answer_id,
        "collector_config_sha256": record.get("collector_config_sha256"),
        "attempt_number": attempt_number,
    }
    expected_attempt_id = f"attempt_{sha256_json(identity)[:24]}"
    if record.get("attempt_id") != expected_attempt_id:
        raise ValueError("collection error attempt_id mismatch")
    payload = dict(record)
    actual_sha = payload.pop("record_sha256", None)
    if actual_sha != sha256_json(payload):
        raise ValueError("collection error record_sha256 mismatch")


def load_error_attempt_counts(
    path: Path,
    *,
    experiment_id: str,
    condition_id: str,
    generation_run_id: str,
    collector_config_sha256: str,
) -> dict[str, int]:
    """Validate an error sidecar and return the last attempt per case."""

    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for record in load_unique_jsonl(path, key="attempt_id"):
        _validate_collection_error_record(record)
        expected_fields = {
            "experiment_id": experiment_id,
            "condition_id": condition_id,
            "generation_run_id": generation_run_id,
            "collector_config_sha256": collector_config_sha256,
        }
        for key, expected in expected_fields.items():
            if record.get(key) != expected:
                raise ValueError(f"existing error log has a different {key}")
        case_id = str(record.get("case_id") or "")
        if not case_id:
            raise ValueError("collection error is missing case_id")
        attempt_number = int(record["attempt_number"])
        previous = counts.get(case_id, 0)
        if attempt_number != previous + 1:
            raise ValueError(
                f"non-contiguous collection attempts for {case_id}: "
                f"expected {previous + 1}, got {attempt_number}"
            )
        counts[case_id] = attempt_number
    return counts


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())


def main(
    argv: list[str] | None = None,
    *,
    final_authorization: Any | None = None,
) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--condition-id", required=True)
    ap.add_argument("--generation-run-id", "--run-id", required=True)
    ap.add_argument(
        "--provider",
        required=True,
        choices=("auto", "local", "frontier", "gemini", "extractive"),
    )
    ap.add_argument("--model", help="서비스에 명시적으로 요청할 생성 모델")
    ap.add_argument(
        "--institution",
        default="none",
        help="요청 institution; UI 기본 경로는 none(null)으로 고정",
    )
    ap.add_argument(
        "--context-k",
        type=int,
        default=8,
        help="/chat 생성 컨텍스트 개수. 최종 평가는 8로 고정한다.",
    )
    ap.add_argument("--parser-profile", required=True)
    ap.add_argument("--retrieval-mode", required=True)
    ap.add_argument(
        "--expected-corpus-revision",
        help="모든 반환 context에서 요구할 corpus revision",
    )
    ap.add_argument(
        "--expected-retrieval-tuning",
        choices=("on", "off"),
        help="/health에서 요구할 서비스 BM25 tuning 상태",
    )
    ap.add_argument(
        "--expected-context-chunks-per-document",
        type=int,
        help="/health에서 요구할 문서당 context chunk cap",
    )
    ap.add_argument("--expected-generation-max-context-chars", type=int)
    ap.add_argument("--expected-generation-max-output-tokens", type=int)
    ap.add_argument(
        "--expected-generation-sampling",
        choices=("absent",),
        help="final model sampling parameters must be absent/not sent",
    )
    ap.add_argument("--expected-git-commit")
    ap.add_argument("--expected-index-sha256")
    ap.add_argument("--expected-source-manifest-sha256")
    ap.add_argument(
        "--eval-trace",
        action="store_true",
        help="RAG_EVAL_TRACE=1 서버에서 raw draft·context·timing trace를 수집",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="쉼표로 구분한 문항 id만 실행")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="DEV smoke에서만 --limit/--only를 허용",
    )
    ap.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="DEV smoke에서만 model/corpus/eval trace 고정을 완화",
    )
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="timeout/network/HTTP 408·429·5xx 요청의 최대 시도 횟수",
    )
    ap.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="기술적 재시도 지수 backoff의 최초 대기 초",
    )
    ap.add_argument(
        "--errors-out",
        type=Path,
        help="append-only 수집 실패 로그(기본: OUT의 .errors.jsonl sidecar)",
    )
    ap.add_argument(
        "--attempt-journal",
        type=Path,
        help="internal final-schedule write-ahead logical-slot journal",
    )
    args = ap.parse_args(argv)

    if args.context_k <= 0:
        ap.error("--context-k must be positive")
    if args.max_attempts <= 0:
        ap.error("--max-attempts must be positive")
    if args.retry_backoff < 0:
        ap.error("--retry-backoff must be non-negative")
    if args.sleep < 0:
        ap.error("--sleep must be non-negative")
    if (args.limit is not None or args.only) and not args.allow_partial:
        ap.error("--limit/--only requires --allow-partial")
    if not args.allow_unpinned:
        if args.provider == "auto":
            ap.error("final runs must pin --provider instead of auto")
        if args.provider not in {"extractive"} and not args.model:
            ap.error("non-extractive final runs must pin --model")
        if not args.expected_corpus_revision:
            ap.error("final runs must pin --expected-corpus-revision")
        if args.expected_retrieval_tuning is None:
            ap.error("final runs must pin --expected-retrieval-tuning")
        if args.expected_context_chunks_per_document is None:
            ap.error(
                "final runs must pin --expected-context-chunks-per-document"
            )
        if not args.eval_trace:
            ap.error("final runs require --eval-trace")
    if (
        args.expected_context_chunks_per_document is not None
        and not 1 <= args.expected_context_chunks_per_document <= 8
    ):
        ap.error("--expected-context-chunks-per-document must be between 1 and 8")

    expected_tuning = (
        args.expected_retrieval_tuning == "on"
        if args.expected_retrieval_tuning is not None
        else None
    )
    institution = (
        None
        if args.institution.strip().casefold() in {"", "none", "null"}
        else args.institution.strip()
    )

    # Validate and select the input before touching /health.  In particular,
    # a final holdout must never reach the service through the public
    # ``--allow-partial`` escape hatch: the schedule orchestrator supplies a
    # validated in-process authorization for exactly one global call_order.
    try:
        all_cases = load_cases(args.cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ap.error(f"cannot load cases: {exc}")
    case_ids = [str(case.get("id") or "") for case in all_cases]
    if any(not case_id for case_id in case_ids):
        ap.error("every case must have an id")
    if len(case_ids) != len(set(case_ids)):
        ap.error("case ids must be unique")
    try:
        for case in all_cases:
            validate_atomic_evidence_schema(case)
    except ValueError as exc:
        ap.error(f"invalid atomic evidence schema: {exc}")
    cases = list(all_cases)
    if args.only:
        wanted = {token.strip() for token in args.only.split(",") if token.strip()}
        cases = [case for case in cases if case["id"] in wanted]
        missing_only = wanted - {case["id"] for case in cases}
        if missing_only:
            ap.error(f"unknown --only ids: {sorted(missing_only)}")
    if args.limit is not None:
        if args.limit <= 0:
            ap.error("--limit must be positive")
        cases = cases[: args.limit]
    if not cases:
        ap.error("selected evaluation case set must be non-empty")

    contains_final_holdout = any(
        case.get("split") in {"holdout-core", "holdout-challenge"}
        for case in all_cases
    )
    if contains_final_holdout and final_authorization is None:
        ap.error(
            "final holdout collection requires "
            "scripts/run_final_generation_schedule.py; --allow-partial is "
            "not a final authorization"
        )

    errors_out = args.errors_out
    if errors_out is None:
        errors_out = (
            args.out.with_suffix(".errors.jsonl")
            if args.out.suffix
            else Path(str(args.out) + ".errors.jsonl")
        )
    if errors_out.resolve() == args.out.resolve():
        ap.error("--errors-out must differ from --out")
    if args.attempt_journal is not None and args.attempt_journal.resolve() in {
        args.out.resolve(), errors_out.resolve()
    }:
        ap.error("--attempt-journal must differ from answer/error artifacts")

    final_metadata: dict[str, Any] | None = None
    if final_authorization is not None:
        from service_eval_artifacts import is_exact_final_authorization

        if not is_exact_final_authorization(final_authorization):
            ap.error("invalid exact final collection authorization")
        if final_authorization.kind == "generation":
            from run_final_generation_schedule import validate_generation_authorization
            authorize = lambda **values: validate_generation_authorization(
                final_authorization.payload, **values
            )
        else:
            from run_final_retrieval_schedule import validate_retrieval_authorization
            authorize = lambda **values: validate_retrieval_authorization(
                final_authorization.payload, **values
            )
        if args.attempt_journal is None:
            ap.error("scheduled final collection requires --attempt-journal")
        if args.limit is not None or not args.only or len(cases) != 1:
            ap.error(
                "scheduled final collection requires exactly one --only case "
                "and forbids --limit"
            )
        if not args.allow_partial:
            ap.error("scheduled final collection requires explicit --allow-partial")
        if args.allow_unpinned:
            ap.error("scheduled final collection forbids --allow-unpinned")
        collector_controls = {
            "api_base": args.api_base.rstrip("/"),
            "provider": args.provider,
            "model": args.model,
            "institution": institution,
            "context_k": args.context_k,
            "parser_profile": args.parser_profile,
            "retrieval_mode": args.retrieval_mode,
            "expected_corpus_revision": args.expected_corpus_revision,
            "expected_retrieval_tuning": expected_tuning,
            "expected_context_chunks_per_document": (
                args.expected_context_chunks_per_document
            ),
            "expected_generation_max_context_chars": (
                args.expected_generation_max_context_chars
            ),
            "expected_generation_max_output_tokens": (
                args.expected_generation_max_output_tokens
            ),
            "expected_generation_sampling_parameters": (
                [] if args.expected_generation_sampling == "absent" else None
            ),
            "expected_git_commit": args.expected_git_commit,
            "expected_index_sha256": args.expected_index_sha256,
            "expected_source_manifest_sha256": (
                args.expected_source_manifest_sha256
            ),
            "eval_trace": args.eval_trace,
            "allow_unpinned": args.allow_unpinned,
            "max_attempts": args.max_attempts,
            "retry_backoff_seconds": args.retry_backoff,
            "request_timeout_seconds": args.timeout,
            "inter_call_sleep_seconds": args.sleep,
        }
        try:
            final_metadata = authorize(
                cases_path=args.cases,
                output_path=args.out,
                errors_path=errors_out,
                journal_path=args.attempt_journal,
                experiment_id=args.experiment_id,
                condition_id=args.condition_id,
                generation_run_id=args.generation_run_id,
                selected_case_ids=[str(case["id"]) for case in cases],
                collector_controls=collector_controls,
            )
        except (OSError, ValueError, TypeError) as exc:
            ap.error(f"final collection authorization failed: {exc}")
        if not isinstance(final_metadata, dict):
            ap.error("final collection authorization returned invalid metadata")
        if not isinstance(final_metadata.get("collector"), dict):
            ap.error("final collection authorization omitted collector metadata")
        if not isinstance(final_metadata.get("record"), dict):
            ap.error("final collection authorization omitted record metadata")
        authorized_case_ids = final_metadata.get(
            "artifact_expected_case_ids"
        )
        if (
            not isinstance(authorized_case_ids, list)
            or not authorized_case_ids
            or any(
                not isinstance(case_id, str) or not case_id
                for case_id in authorized_case_ids
            )
        ):
            ap.error(
                "final collection authorization omitted artifact case ids"
            )
        call_order = final_metadata["record"].get("call_order")
        if not isinstance(call_order, int) or call_order <= 0:
            ap.error("final collection authorization has invalid call_order")
        authorized_all_cases = final_metadata.get("authorized_all_cases")
        authorized_case = final_metadata.get("authorized_case")
        if (
            not isinstance(authorized_all_cases, list)
            or any(not isinstance(case, dict) for case in authorized_all_cases)
            or not isinstance(authorized_case, dict)
        ):
            ap.error("final collection authorization omitted case snapshot")
        authorized_ids = [
            str(case.get("id") or "") for case in authorized_all_cases
        ]
        if (
            any(not case_id for case_id in authorized_ids)
            or len(authorized_ids) != len(set(authorized_ids))
            or authorized_case.get("id") != cases[0].get("id")
        ):
            ap.error("final collection authorization case snapshot mismatch")
        try:
            for case in authorized_all_cases:
                validate_atomic_evidence_schema(case)
        except ValueError as exc:
            ap.error(f"authorized atomic evidence schema is invalid: {exc}")
        # Only the byte snapshot verified inside the authorization is allowed
        # to reach call_chat.  Never retain the pre-authorization object: the
        # file may have changed between the initial CLI selection and gate
        # verification (TOCTOU).
        all_cases = authorized_all_cases
        cases = [authorized_case]

    try:
        collection_purpose = (
            final_metadata.get("collector", {}).get("collection_purpose")
            if final_metadata is not None
            else None
        )
        if final_metadata is not None and collection_purpose not in {
            "final_generation", "final_retrieval"
        }:
            raise ValueError("unsupported final collection purpose")
        retrieval_only = collection_purpose == "final_retrieval"
        health, health_attempts = call_with_technical_retries(
            lambda: call_health(args.api_base, args.timeout),
            max_attempts=args.max_attempts,
            backoff_seconds=args.retry_backoff,
        )
        server_config = validate_health_controls(
            health,
            parser_profile=args.parser_profile,
            retrieval_mode=args.retrieval_mode,
            expected_corpus_revision=args.expected_corpus_revision,
            expected_retrieval_tuning=expected_tuning,
            expected_context_chunks_per_document=(
                args.expected_context_chunks_per_document
            ),
            expected_generation_provider=(
                args.provider
                if final_metadata is not None and not retrieval_only
                else None
            ),
            # 모델별 생성 한도(max_context_chars 등)는 /health의
            # service_config.generation.models[<model>]에만 있다. DEV 실행이라도
            # 그 한도 검사를 요청했다면 모델명을 넘겨 모델별 항목을 대조한다.
            # (2026-09-04: 상위 객체를 대조해 항상 None mismatch가 나던 결함 수정)
            expected_generation_model=(
                args.model
                if args.model
                and not retrieval_only
                and (
                    final_metadata is not None
                    or args.expected_generation_max_context_chars is not None
                    or args.expected_generation_max_output_tokens is not None
                    or args.expected_generation_sampling == "absent"
                )
                else None
            ),
            expected_generation_max_context_chars=(
                args.expected_generation_max_context_chars
                if not retrieval_only else None
            ),
            expected_generation_max_output_tokens=(
                args.expected_generation_max_output_tokens
                if not retrieval_only else None
            ),
            expected_generation_sampling_parameters=(
                []
                if args.expected_generation_sampling == "absent" and not retrieval_only
                else None
            ),
            expected_git_commit=args.expected_git_commit,
            expected_index_sha256=args.expected_index_sha256,
            expected_source_manifest_sha256=(
                args.expected_source_manifest_sha256
            ),
            require_eval_trace=args.eval_trace,
        )
    except (RequestFailure, RuntimeError, ValueError) as exc:
        ap.error(f"service preflight failed: {exc}")

    collector_config = {
        "api_base": args.api_base.rstrip("/"),
        "cases_path": str(args.cases),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "cases_canonical_sha256": sha256_json(all_cases),
        "selected_case_ids_sha256": sha256_json(
            final_metadata["artifact_expected_case_ids"]
            if final_metadata is not None
            else [str(case["id"]) for case in cases]
        ),
        "provider": args.provider,
        "model": args.model,
        "institution": institution,
        "context_k": args.context_k,
        "parser_profile": args.parser_profile,
        "retrieval_mode": args.retrieval_mode,
        "expected_corpus_revision": args.expected_corpus_revision,
        "expected_retrieval_tuning": expected_tuning,
        "expected_context_chunks_per_document": (
            args.expected_context_chunks_per_document
        ),
        "expected_generation_max_context_chars": (
            args.expected_generation_max_context_chars
        ),
        "expected_generation_max_output_tokens": (
            args.expected_generation_max_output_tokens
        ),
        "expected_generation_sampling_parameters": (
            [] if args.expected_generation_sampling == "absent" else None
        ),
        "expected_git_commit": args.expected_git_commit,
        "expected_index_sha256": args.expected_index_sha256,
        "expected_source_manifest_sha256": args.expected_source_manifest_sha256,
        "eval_trace": args.eval_trace,
        "allow_unpinned": args.allow_unpinned,
        "max_attempts": args.max_attempts,
        "retry_backoff_seconds": args.retry_backoff,
        "request_timeout_seconds": args.timeout,
        "inter_call_sleep_seconds": args.sleep,
        "errors_path": str(errors_out),
        "attempt_journal_path": (
            str(args.attempt_journal) if args.attempt_journal is not None else None
        ),
        "server_config": server_config,
    }
    if final_metadata is not None:
        collector_config["final_authorization"] = final_metadata["collector"]
    collector_config_sha = sha256_json(collector_config)

    journal_events: list[dict[str, Any]] = []
    if args.attempt_journal is not None and args.attempt_journal.exists():
        try:
            journal_events = load_unique_jsonl(
                args.attempt_journal, key="journal_event_id"
            )
        except ValueError as exc:
            ap.error(f"invalid attempt journal: {exc}")
        current_journal_events = (
            [
                event for event in journal_events
                if event.get("call_order") == final_metadata["record"]["call_order"]
            ]
            if final_metadata is not None
            else []
        )
        if current_journal_events:
            ap.error(
                "logical slot already has a durable journal event; automatic "
                "retry is forbidden (manual experiment invalidation required)"
            )

    done_ids: set[str] = set()
    legacy_error_cases: list[str] = []
    if args.out.exists():
        try:
            existing_answers = load_unique_jsonl(args.out, key="answer_id")
        except ValueError as exc:
            ap.error(f"invalid existing answer artifact: {exc}")
        for record in existing_answers:
            if record.get("experiment_id") != args.experiment_id:
                ap.error("existing output has a different experiment_id")
            if record.get("condition_id") != args.condition_id:
                ap.error("existing output has a different condition_id")
            if record.get("generation_run_id") != args.generation_run_id:
                ap.error("existing output has a different generation_run_id")
            if record.get("error"):
                legacy_error_cases.append(str(record.get("case_id") or ""))
                continue
            if record.get("collector_config_sha256") != collector_config_sha:
                ap.error("existing output has a different collector config")
            try:
                validate_answer_record(record)
            except ValueError as exc:
                ap.error(f"invalid existing answer artifact: {exc}")
            if not str(record.get("answer") or "").strip():
                ap.error("existing answer artifact contains an empty answer")
            done_ids.add(str(record.get("case_id") or ""))
    if legacy_error_cases:
        ap.error(
            "existing --out contains legacy inline error row(s) for "
            f"{sorted(legacy_error_cases)}; they are not completed, but cannot "
            "be retried without duplicating immutable answer_id. Use a new "
            "--generation-run-id and --out; new failures are written to the "
            ".errors.jsonl sidecar."
        )

    try:
        error_attempt_counts = load_error_attempt_counts(
            errors_out,
            experiment_id=args.experiment_id,
            condition_id=args.condition_id,
            generation_run_id=args.generation_run_id,
            collector_config_sha256=collector_config_sha,
        )
    except ValueError as exc:
        ap.error(f"invalid existing collection error log: {exc}")

    todo = [case for case in cases if case["id"] not in done_ids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed_case_ids: set[str] = set()
    with args.out.open("a", encoding="utf-8") as sink:
        for pos, case in enumerate(todo, 1):
            rec = build_record_base(
                case,
                experiment_id=args.experiment_id,
                condition_id=args.condition_id,
                generation_run_id=args.generation_run_id,
                collector_config=collector_config,
            )
            if final_metadata is not None:
                rec["call_order"] = final_metadata["record"]["call_order"]
                rec["collection_schedule"] = dict(final_metadata["record"])
            rec["health_request_attempts"] = health_attempts
            def journal(event: str, outcome: str | None = None) -> None:
                if args.attempt_journal is None:
                    return
                payload = {
                    "schema_version": "pnu.final-generation-slot-journal.v1",
                    "event": event,
                    "schedule_id": final_metadata["record"]["schedule_id"],
                    "schedule_sha256": final_metadata["record"]["schedule_sha256"],
                    "artifact_id": final_metadata["record"]["artifact_id"],
                    "call_order": final_metadata["record"]["call_order"],
                    "case_id": case["id"],
                    "case_sha256": rec["case_sha256"],
                    "max_chat_attempts": args.max_attempts,
                    "outcome": outcome,
                }
                payload["record_sha256"] = sha256_json(payload)
                payload["journal_event_id"] = (
                    "journal_" + sha256_json(payload)[:24]
                )
                append_jsonl_record(args.attempt_journal, payload)

            # Health and all authorization checks have succeeded. From this
            # fsync onward a crash is an uncertain sent request and must never
            # be automatically replayed.
            if final_metadata is not None:
                journal("slot_started")
            request_attempts: list[dict[str, Any]] = []
            started = time.perf_counter()
            try:
                resp, request_attempts = call_with_technical_retries(
                    lambda: call_chat(
                        args.api_base,
                        case,
                        args.provider,
                        args.model,
                        args.timeout,
                        top_k=args.context_k,
                        parser_profile=args.parser_profile,
                        retrieval_mode=args.retrieval_mode,
                        institution=institution,
                        eval_trace=args.eval_trace,
                    ),
                    max_attempts=args.max_attempts,
                    backoff_seconds=args.retry_backoff,
                )
            except RequestFailure as exc:
                attempt_number = error_attempt_counts.get(case["id"], 0) + 1
                error_record = build_collection_error_record(
                    rec,
                    attempt_number=attempt_number,
                    stage="chat_transport",
                    error=exc.cause,
                    request_attempts=exc.attempts,
                )
                append_jsonl_record(errors_out, error_record)
                error_attempt_counts[case["id"]] = attempt_number
                failed_case_ids.add(case["id"])
                print(
                    f"[{pos}/{len(todo)}] {case['id']} chat_error: {exc}",
                    flush=True,
                )
                if not exc.retryable:
                    if final_metadata is not None:
                        journal("slot_poisoned", "nonretryable_transport")
                    # A terminal HTTP/client-contract error is condition-wide;
                    # do not turn subsequent cases into implicit retries.
                    break
                if final_metadata is None:
                    continue
                rec.update(
                    {
                        "slot_outcome": "service_error",
                        "answer_eligible_for_judge": False,
                        "answer": "[SERVICE_ERROR]",
                        "service_error": {
                            "stage": "chat_transport",
                            "type": type(exc.cause).__name__,
                            "message": str(exc.cause),
                            "retryable": True,
                            "http_status": getattr(exc.cause, "status", None),
                            "request_attempts": exc.attempts,
                        },
                        "request_attempts": exc.attempts,
                        "collection_attempt_number": 1,
                    }
                )
                rec = build_answer_identity(rec)
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                journal("slot_completed", "service_error")
                failed_case_ids.discard(case["id"])
                continue
            except (ValueError, TypeError) as exc:
                attempt_number = error_attempt_counts.get(case["id"], 0) + 1
                error_record = build_collection_error_record(
                    rec,
                    attempt_number=attempt_number,
                    stage="chat_response_decode",
                    error=exc,
                    request_attempts=request_attempts,
                )
                append_jsonl_record(errors_out, error_record)
                error_attempt_counts[case["id"]] = attempt_number
                failed_case_ids.add(case["id"])
                if final_metadata is not None:
                    journal("slot_poisoned", "response_decode")
                print(
                    f"[{pos}/{len(todo)}] {case['id']} response_error: {exc}",
                    flush=True,
                )
                # Malformed JSON/payload encoding is a service-contract error,
                # not a transient per-case generation failure.
                break

            if not isinstance(resp, dict):
                exc = AnswerPayloadError("response payload must be an object")
                attempt_number = error_attempt_counts.get(case["id"], 0) + 1
                error_record = build_collection_error_record(
                    rec,
                    attempt_number=attempt_number,
                    stage="response_validation",
                    error=exc,
                    request_attempts=request_attempts,
                )
                append_jsonl_record(errors_out, error_record)
                error_attempt_counts[case["id"]] = attempt_number
                failed_case_ids.add(case["id"])
                if final_metadata is not None:
                    journal("slot_poisoned", "response_contract")
                print(
                    f"[{pos}/{len(todo)}] {case['id']} response_error: {exc}",
                    flush=True,
                )
                break

            atomic_contexts: list[dict[str, Any]] | None = None
            try:
                if args.eval_trace and not isinstance(
                    resp.get("evaluation_trace"), dict
                ):
                    raise RuntimeError(
                        "eval trace was requested but the response omitted it"
                    )
                validate_response_controls(
                    resp,
                    provider=args.provider,
                    model=args.model,
                    parser_profile=args.parser_profile,
                    retrieval_mode=args.retrieval_mode,
                    expected_corpus_revision=args.expected_corpus_revision,
                    institution=institution,
                    require_eval_trace=args.eval_trace,
                )
                if final_metadata is not None:
                    if collection_purpose == "final_retrieval":
                        validate_final_retrieval_provenance(resp)
                    else:
                        validate_final_generation_provenance(
                            resp,
                            provider=args.provider,
                            model=str(args.model),
                            max_output_tokens=int(
                                args.expected_generation_max_output_tokens
                            ),
                        )
                if case.get("required_claims"):
                    atomic_contexts = final_evaluation_contexts(resp)
                    if atomic_contexts is None:
                        raise RuntimeError(
                            "atomic evidence evaluation requires "
                            "evaluation_trace.retrieval_stages.final_contexts"
                        )
            except (RuntimeError, ValueError) as exc:
                attempt_number = error_attempt_counts.get(case["id"], 0) + 1
                error_record = build_collection_error_record(
                    rec,
                    attempt_number=attempt_number,
                    stage="response_control",
                    error=exc,
                    request_attempts=request_attempts,
                )
                append_jsonl_record(errors_out, error_record)
                error_attempt_counts[case["id"]] = attempt_number
                failed_case_ids.add(case["id"])
                if final_metadata is not None:
                    journal("slot_poisoned", "response_control")
                print(
                    f"[{pos}/{len(todo)}] {case['id']} control_error: {exc}",
                    flush=True,
                )
                # A pinned-control mismatch invalidates the condition. Stop
                # before issuing more potentially contaminated requests.
                break

            try:
                answer = validate_answer_payload(resp)
            except AnswerPayloadError as exc:
                attempt_number = error_attempt_counts.get(case["id"], 0) + 1
                error_record = build_collection_error_record(
                    rec,
                    attempt_number=attempt_number,
                    stage="response_validation",
                    error=exc,
                    request_attempts=request_attempts,
                )
                append_jsonl_record(errors_out, error_record)
                error_attempt_counts[case["id"]] = attempt_number
                failed_case_ids.add(case["id"])
                print(
                    f"[{pos}/{len(todo)}] {case['id']} response_error: {exc}",
                    flush=True,
                )
                if final_metadata is None:
                    continue
                rec.update(
                    {
                        "slot_outcome": "service_error",
                        "answer_eligible_for_judge": False,
                        "answer": "[SERVICE_ERROR]",
                        "service_error": {
                            "stage": "response_validation",
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "retryable": False,
                            "http_status": None,
                            "request_attempts": request_attempts,
                        },
                        "request_attempts": request_attempts,
                        "collection_attempt_number": 1,
                    }
                )
                rec = build_answer_identity(rec)
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                journal("slot_completed", "service_error")
                failed_case_ids.discard(case["id"])
                continue
            rec["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            rec["answer"] = answer
            rec["slot_outcome"] = "answer"
            rec["answer_eligible_for_judge"] = True
            rec["request_attempts"] = request_attempts
            rec["collection_attempt_number"] = (
                error_attempt_counts.get(case["id"], 0) + 1
            )
            rec["cited_answer"] = resp.get("cited_answer", "")
            rec["claims"] = resp.get("claims") or []
            rec["citations"] = resp.get("citations") or []
            rec["postprocessing"] = resp.get("postprocessing") or {}
            rec["generator"] = resp.get("generator")
            rec["generation"] = resp.get("generation")
            rec["role_mapping"] = resp.get("role")
            rec["request"] = build_chat_body(
                case,
                provider=args.provider,
                model=args.model,
                top_k=args.context_k,
                parser_profile=args.parser_profile,
                retrieval_mode=args.retrieval_mode,
                institution=institution,
                eval_trace=args.eval_trace,
            )
            rec["response_config"] = {
                "institution": resp.get("institution"),
                "parser_profile": resp.get("parser_profile"),
                "retrieval_mode": resp.get("retrieval_mode"),
                "generation_requested": (resp.get("generation") or {}).get("requested"),
                "generation_used": (resp.get("generation") or {}).get("used"),
                "generation_model": (resp.get("generation") or {}).get("model"),
            }
            rec["retrieval"] = resp.get("retrieval") or {}
            if args.eval_trace:
                rec["evaluation_trace"] = resp["evaluation_trace"]
            source_k = int(case.get("k", 5))
            rec["retrieval_hit"] = retrieval_hit(
                resp.get("results") or [], case.get("expected") or {}, source_k
            )
            evidence_ks = sorted({source_k, args.context_k})
            rec["evidence_at_k"] = {
                str(k): evidence_hit(resp.get("results") or [], case, k=k)
                for k in evidence_ks
            }
            if case.get("required_claims"):
                assert atomic_contexts is not None
                rec["atomic_evidence_at_k"] = {
                    str(k): atomic_evidence_hit(
                        atomic_contexts, case, k=k
                    )
                    for k in evidence_ks
                }
            # Backward-compatible alias. Its scope is now explicit and always
            # equals the actual generation context cutoff, never an implicit @5.
            rec["evidence_hit"] = rec["evidence_at_k"][str(args.context_k)]
            rec["sources"] = [
                {
                    "source_title": row.get("source_title"),
                    "source_host": row.get("source_host"),
                    "chunk_id": row.get("chunk_id"),
                    "document_id": row.get("document_id") or row.get("doc_id"),
                    "corpus_revision": row.get("corpus_revision"),
                    "preview": row.get("preview"),
                    "locations": row.get("locations") or [],
                    "retrieval": row.get("retrieval") or {},
                }
                for row in (resp.get("results") or [])
            ]
            rec = build_answer_identity(rec)
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            if final_metadata is not None:
                journal("slot_completed", "answer")
            hit = rec["retrieval_hit"]["matched"]
            ev_source = rec["evidence_at_k"][str(source_k)]["matched"]
            ev_context = rec["evidence_at_k"][str(args.context_k)]["matched"]
            print(
                f"[{pos}/{len(todo)}] {case['id']} answer_id={rec['answer_id']} hit={hit} "
                f"gold-chunk@{source_k}={ev_source} gold-chunk@{args.context_k}={ev_context}",
                flush=True,
            )
            time.sleep(args.sleep)

    all_records = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()]
    summarize(all_records)
    completed_case_ids = {
        str(record.get("case_id") or "")
        for record in all_records
        if not record.get("error")
    }
    unresolved = [
        str(case["id"])
        for case in cases
        if str(case["id"]) not in completed_case_ids
    ]
    if unresolved:
        print(
            f"collection incomplete: {len(unresolved)}/{len(cases)} "
            f"case(s) unresolved; failures={sorted(failed_case_ids)}; "
            f"error_log={errors_out}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)


def summarize(records: list[dict]) -> None:
    errors = [r for r in records if r.get("error")]
    hits = [r for r in records if (r.get("retrieval_hit") or {}).get("matched")]
    with_hit_info = [r for r in records if r.get("retrieval_hit") is not None]
    print(f"answer records={len(records)} errors={len(errors)}")
    if with_hit_info:
        print(f"retrieval hit@k: {len(hits)}/{len(with_hit_info)} = {len(hits) / len(with_hit_info):.3f}")
        rr = [1.0 / r["retrieval_hit"]["rank"] if r["retrieval_hit"].get("rank") else 0.0
              for r in with_hit_info]
        print(f"retrieval MRR: {sum(rr) / len(rr):.3f}")
        evidence_ks = sorted(
            {
                int(k)
                for record in records
                for k in (record.get("evidence_at_k") or {})
            }
        )
        if evidence_ks:
            for k in evidence_ks:
                values = [
                    record["evidence_at_k"][str(k)]
                    for record in records
                    if str(k) in (record.get("evidence_at_k") or {})
                ]
                any_hits = sum(bool(value.get("matched")) for value in values)
                all_hits = sum(bool(value.get("all_matched")) for value in values)
                mean_recall = sum(float(value.get("recall", 0.0)) for value in values) / len(values)
                print(
                    f"gold-chunk@{k}: any={any_hits}/{len(values)} "
                    f"all={all_hits}/{len(values)} mean_recall={mean_recall:.3f}"
                )
        else:
            # Old artifacts did not record a cutoff. Keep the diagnostic but
            # deliberately avoid attaching an @k label to it.
            ev_records = [r for r in records if r.get("evidence_hit") is not None]
            if ev_records:
                ev_hits = [r for r in ev_records if r["evidence_hit"]["matched"]]
                print(
                    "legacy evidence context hit (cutoff unknown): "
                    f"{len(ev_hits)}/{len(ev_records)} = "
                    f"{len(ev_hits) / len(ev_records):.3f}"
                )

        atomic_ks = sorted(
            {
                int(k)
                for record in records
                for k in (record.get("atomic_evidence_at_k") or {})
            }
        )
        for k in atomic_ks:
            values = [
                record["atomic_evidence_at_k"][str(k)]
                for record in records
                if str(k) in (record.get("atomic_evidence_at_k") or {})
                and record["atomic_evidence_at_k"][str(k)].get("available")
            ]
            if not values:
                continue
            found_claims = sum(
                len(value.get("found_claim_ids") or []) for value in values
            )
            wanted_claims = sum(int(value.get("wanted") or 0) for value in values)
            micro_recall = (
                found_claims / wanted_claims if wanted_claims else 0.0
            )
            macro_recall = sum(float(value["recall"]) for value in values) / len(
                values
            )
            all_hits = sum(bool(value.get("all_matched")) for value in values)
            print(
                f"atomic-evidence@{k}: all={all_hits}/{len(values)} "
                f"micro_recall={micro_recall:.3f} "
                f"macro_recall={macro_recall:.3f}"
            )


if __name__ == "__main__":
    main()
