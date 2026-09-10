#!/usr/bin/env python3
"""Serve the local BM25 index over a tiny HTTP API.

This is a development bridge for the React frontend. It intentionally uses only
the Python standard library plus the existing bm25_search module.

Examples:
  python scripts/search_api.py
  python scripts/search_api.py --port 8001
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

try:
    from .bm25_search import (
        DEFAULT_INDEX,
        DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
        SERVICE_QUERY_STOP_TOKENS,
        demote_generic_candidates,
        load_chunks_by_ids,
        search_bm25_candidates,
        search_index,
        tokenize,
    )
    from .rag.generators import (
        GenerationError,
        SUPPORTED_PROVIDERS,
        SYSTEM_INSTRUCTION as GENERATION_SYSTEM_INSTRUCTION,
        build_prompt as build_generation_prompt,
        generate,
        generation_runtime_controls,
    )
    from .local_model_runtime import (
        ExternalLocalModelRuntime,
        LocalModelBusyError,
        LocalModelRuntimeError,
        LocalModelUnmanagedError,
        build_local_model_runtime,
    )
    from .rag.retrieval import DenseIndex, HybridRetriever
    from .rag.learned_dense import LearnedDenseIndex
    from .rag import role_router
    from .rag.security import enforce_output, evaluate_contexts
except ImportError:  # Direct CLI execution: python scripts/search_api.py
    from bm25_search import (
        DEFAULT_INDEX,
        DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
        SERVICE_QUERY_STOP_TOKENS,
        demote_generic_candidates,
        load_chunks_by_ids,
        search_bm25_candidates,
        search_index,
        tokenize,
    )
    from rag.generators import (
        GenerationError,
        SUPPORTED_PROVIDERS,
        SYSTEM_INSTRUCTION as GENERATION_SYSTEM_INSTRUCTION,
        build_prompt as build_generation_prompt,
        generate,
        generation_runtime_controls,
    )
    from local_model_runtime import (
        ExternalLocalModelRuntime,
        LocalModelBusyError,
        LocalModelRuntimeError,
        LocalModelUnmanagedError,
        build_local_model_runtime,
    )
    from rag.retrieval import DenseIndex, HybridRetriever
    from rag.learned_dense import LearnedDenseIndex
    from rag import role_router
    from rag.security import enforce_output, evaluate_contexts


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_TOP_K = 8
DEFAULT_MAX_TOP_K = 20
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024
DEFAULT_MAX_QUESTION_CHARS = 1000
DEFAULT_PREVIEW_CHARS = 700
DEFAULT_SOURCE_CHARS = 1800
DEFAULT_MAX_CITATION_LOCATIONS = 100
DEFAULT_MAX_CONCURRENT_GENERATIONS = 2
DEFAULT_ALLOWED_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
MAX_CLAIMS = 8
MAX_CHAT_CANDIDATES = 80
CHAT_CANDIDATE_MULTIPLIER = 4
CLAIM_SUPPORT_THRESHOLD = 0.40
MIN_CLAIM_ANCHORS = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_INDEX_IDENTITY_CACHE: dict[str, tuple[int, int, int, str]] = {}
_INDEX_IDENTITY_LOCK = threading.Lock()


def _cached_index_identity(path: Path) -> tuple[str, int]:
    """Hash once per unchanged inode/size/mtime snapshot."""

    resolved = path.resolve()
    stat = resolved.stat()
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    key = str(resolved)
    with _INDEX_IDENTITY_LOCK:
        cached = _INDEX_IDENTITY_CACHE.get(key)
        if cached is not None and cached[:3] == signature:
            return cached[3], stat.st_size
        digest = _sha256_file(resolved)
        after = resolved.stat()
        if (after.st_ino, after.st_size, after.st_mtime_ns) != signature:
            raise RuntimeError("index changed while computing health identity")
        _INDEX_IDENTITY_CACHE[key] = (
            signature[0], signature[1], signature[2], digest
        )
        return digest, stat.st_size


def _startup_runtime_metadata() -> dict[str, Any]:
    """Capture code identity once; later checkout changes cannot rewrite it."""

    def git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "startup_git_commit": git("rev-parse", "HEAD") or None,
        "startup_worktree_clean": status == "",
        "startup_code_sha256": _sha256_file(Path(__file__).resolve()),
        "process_started_at": datetime.now(timezone.utc).isoformat(),
    }


STARTUP_RUNTIME_METADATA = MappingProxyType(_startup_runtime_metadata())


def freeze_runtime_metadata(index_path: Path) -> dict[str, Any]:
    """Expose immutable process identity plus the selected index snapshot."""

    index_sha256, index_size_bytes = _cached_index_identity(index_path)
    return {
        **STARTUP_RUNTIME_METADATA,
        "index_sha256": index_sha256,
        "index_size_bytes": index_size_bytes,
    }

def final_generation_health() -> dict[str, Any]:
    """Expose final provider/model controls without credentials."""

    models = gemini_model_candidates()
    return {
        "provider": "frontier",
        "configured": bool(gemini_api_key()),
        "allowed_models": models,
        "models": {
            model: generation_runtime_controls(model) for model in models
        },
    }
RETRIEVAL_REQUEST_TERMS = frozenset(
    {
        "관련",
        "규정",
        "규정을",
        "규정은",
        "문서",
        "문서를",
        "내용",
        "내용을",
        "검색",
        "검색해줘",
        "검색해주세요",
        "검색해",
        "찾아줘",
        "찾아주세요",
        "알려줘",
        "알려주세요",
        "설명해줘",
        "설명해주세요",
        "요약해줘",
        "요약해주세요",
    }
)
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_DENSE_INDEX = Path("processed/index/dense.sqlite")
DEFAULT_LEARNED_DENSE_ROOT = Path("processed/index/learned-dense")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite",
)
REQUEST_MODEL_MISSING = object()
PARSER_PROFILE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
PARSER_PROFILE_LABELS = {
    "baseline": "Baseline · 기본 파서",
    "challenger": "Challenger · 대체 파서",
    "cascade": "Cascade · 품질 기반 선택",
}
PARSER_PROFILE_ORDER = ("baseline", "challenger", "cascade")
RETRIEVAL_MODE_LABELS = {
    "bm25": "BM25 · 키워드 기준",
    "kure_dense": "KURE Dense · 의미 기준",
    "kure_hybrid": "BM25 + KURE · 하이브리드",
    "snowflake_dense": "Snowflake Dense · 의미 기준",
    "snowflake_hybrid": "BM25 + Snowflake · 하이브리드",
}
RETRIEVAL_MODE_ORDER = tuple(RETRIEVAL_MODE_LABELS)
LEARNED_DENSE_ARTIFACTS = {
    "kure": "kure-v1",
    "snowflake": "snowflake-arctic-l-v2-ko",
}


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, error: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.error = error
        self.message = message


@dataclass(frozen=True)
class RetrievalModeState:
    id: str
    label: str
    retriever: HybridRetriever | None
    warning: str | None = None
    learned_index: LearnedDenseIndex | None = None

    @property
    def ready(self) -> bool:
        return self.warning is None and (
            self.id == "bm25" or self.retriever is not None
        )


@dataclass(frozen=True)
class ParserIndexTarget:
    profile: str
    index_path: Path
    retriever: HybridRetriever | None
    warning: str | None
    retrieval_modes: Mapping[str, RetrievalModeState] = field(
        default_factory=dict
    )


def load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def env_list(name: str, default: tuple[str, ...]) -> list[str]:
    value = os.environ.get(name, "")
    if not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def evaluation_trace_enabled() -> bool:
    return os.environ.get("RAG_EVAL_TRACE", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def resolve_evaluation_trace_request(value: Any) -> bool:
    """Allow full evaluation traces only after an explicit server opt-in."""
    if value is None or value is False:
        return False
    if value is not True:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_eval_trace",
            "eval_trace must be a JSON boolean.",
        )
    if not evaluation_trace_enabled():
        raise ApiError(
            HTTPStatus.FORBIDDEN,
            "eval_trace_disabled",
            "evaluation trace is disabled; start the server with RAG_EVAL_TRACE=1.",
        )
    return True


def allowed_origins() -> list[str]:
    return env_list("RAG_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)


def cors_origin_for(origin: str | None) -> str | None:
    origins = allowed_origins()
    if "*" in origins:
        return "*"
    if origin and origin in origins:
        return origin
    return None


def api_token() -> str:
    return os.environ.get("RAG_API_TOKEN", "").strip()


def request_max_bytes() -> int:
    return max(1024, get_env_int("RAG_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))


def question_max_chars() -> int:
    return max(100, get_env_int("RAG_MAX_QUESTION_CHARS", DEFAULT_MAX_QUESTION_CHARS))


def max_top_k() -> int:
    return max(1, get_env_int("RAG_MAX_TOP_K", DEFAULT_MAX_TOP_K))


def preview_chars() -> int:
    return max(100, get_env_int("RAG_PREVIEW_CHARS", DEFAULT_PREVIEW_CHARS))


def source_chars() -> int:
    return max(300, get_env_int("RAG_SOURCE_CHARS", DEFAULT_SOURCE_CHARS))


def max_citation_locations() -> int:
    return max(
        1,
        min(
            1000,
            get_env_int(
                "RAG_MAX_CITATION_LOCATIONS",
                DEFAULT_MAX_CITATION_LOCATIONS,
            ),
        ),
    )


def parse_top_k(value: Any, default: int = DEFAULT_TOP_K) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(1, parsed), max_top_k())


def validate_question(value: Any) -> str:
    question = str(value or "").strip()
    if not question:
        raise ApiError(HTTPStatus.BAD_REQUEST, "question_required", "질문을 입력해 주세요.")
    if len(question) > question_max_chars():
        raise ApiError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "question_too_long",
            f"질문은 {question_max_chars()}자 이내로 입력해 주세요.",
        )
    return question


def normalize_retrieval_query(
    question: str,
    institution: str | None = None,
) -> str:
    """Keep content-bearing terms and remove UI-style request boilerplate."""

    value = str(question or "").strip()
    if institution:
        value = re.sub(re.escape(institution), " ", value, flags=re.IGNORECASE)
    original_terms = tokenize(value, include_ngrams=False)
    content_terms = [
        term for term in original_terms if term not in RETRIEVAL_REQUEST_TERMS
    ]
    # A query consisting only of a generic word such as "규정" is still valid.
    resolved = content_terms or original_terms
    return " ".join(resolved).strip() or str(question or "").strip()


_OUTSOURCED_SERVICE_QUERY_RE = re.compile(r"학생증|증명서")
_OUTSOURCING_INTENT_RE = re.compile(r"외부\s*기관|위탁")
_ADMISSION_YEAR_RE = re.compile(r"(?:19|20)\d{2}\s*학년도")
_ADMISSION_INTAKE_RE = re.compile(r"신입생|신입학")
_ADMISSION_COUNT_RE = re.compile(r"총\s*몇|몇\s*명|모집\s*인원")
_ADMISSION_CHANGE_RE = re.compile(r"전년\s*대비|달라|변경")
_D2_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])D\s*[-‐‑‒–—]?\s*2(?!\d)",
    re.IGNORECASE,
)
_GROUP_VISA_INTENT_RE = re.compile(r"비자|체류")
_FOREIGN_STUDENT_RE = re.compile(r"외국인")
_UNDERGRADUATE_RE = re.compile(r"학부")
_NEW_ADMISSION_RE = re.compile(r"신입학|신입생")
_ELIGIBILITY_INTENT_RE = re.compile(r"자격|조건|요건")
_EXCHANGE_STUDENT_RE = re.compile(r"교환학생|해외\s*파견")
_SELECTION_SCALE_RE = re.compile(r"선발\s*규모")
_APPLICATION_SCHEDULE_RE = re.compile(r"지원\s*일정|접수\s*일정")
_ASSISTIVE_DEVICE_RE = re.compile(r"정보통신\s*보조기기")
_COPAY_RE = re.compile(r"자부담금|개인부담금")


def _is_foreign_undergraduate_admission_eligibility_query(
    question: str,
) -> bool:
    value = str(question or "")
    return all(
        pattern.search(value)
        for pattern in (
            _FOREIGN_STUDENT_RE,
            _UNDERGRADUATE_RE,
            _NEW_ADMISSION_RE,
            _ELIGIBILITY_INTENT_RE,
        )
    )


def _is_exchange_selection_query(question: str) -> bool:
    value = str(question or "")
    return bool(
        _EXCHANGE_STUDENT_RE.search(value)
        and _SELECTION_SCALE_RE.search(value)
        and _APPLICATION_SCHEDULE_RE.search(value)
    )


def _is_third_party_reporting_query(question: str) -> bool:
    value = str(question or "")
    return bool(
        "신고" in value
        and re.search(r"피해자가?\s*아닌|대신\s*신고|제\s*3\s*자|목격", value)
    )


def _is_group_visa_application_query(question: str) -> bool:
    value = str(question or "")
    return bool(
        _D2_IDENTIFIER_RE.search(value)
        and _GROUP_VISA_INTENT_RE.search(value)
        and "연장" in value
        and "단체" in value
    )


def _is_assistive_device_copay_support_query(question: str) -> bool:
    value = str(question or "")
    return bool(
        _ASSISTIVE_DEVICE_RE.search(value)
        and _COPAY_RE.search(value)
        and "지원" in value
    )


def _service_retrieval_query_expansions(question: str) -> tuple[str, ...]:
    """Return narrow document-language expansions for explicit service intents.

    The terms name an official document type or administrative vocabulary; no
    answer value is injected.  This is deliberately limited to intents where
    colloquial user wording and the corpus title/heading vocabulary diverge.
    """

    value = str(question or "")
    expansions: list[str] = []
    if (
        _OUTSOURCED_SERVICE_QUERY_RE.search(value)
        and _OUTSOURCING_INTENT_RE.search(value)
    ):
        expansions.extend(("개인정보처리", "위탁현황", "수탁기관"))
    if (
        _ADMISSION_YEAR_RE.search(value)
        and _ADMISSION_INTAKE_RE.search(value)
        and _ADMISSION_COUNT_RE.search(value)
        and _ADMISSION_CHANGE_RE.search(value)
    ):
        expansions.extend(
            ("대학입학전형", "기본계획", "모집인원", "주요", "변경사항")
        )
    if _is_group_visa_application_query(value):
        expansions.extend(
            (
                "비자",
                "단체접수",
                "체류기간",
                "연장",
                "사전",
                "예약",
                "제출서류",
                "수수료",
            )
        )
    if _is_foreign_undergraduate_admission_eligibility_query(value):
        expansions.extend(
            (
                "외국인",
                "특별전형",
                "모집요강",
                "지원자격",
                "국적",
                "언어능력",
                "학력",
            )
        )
    if _is_exchange_selection_query(value):
        expansions.extend(
            (
                "교환",
                "교비",
                "프로그램",
                "1차",
                "선발요강",
                "선발규모",
                "온라인지원",
                "합격자발표",
            )
        )
    if (
        "여름방학" in value
        and "자격증" in value
        and re.search(r"강의|특강|프로그램|대비", value)
    ):
        expansions.extend(
            (
                "하계방학",
                "취업역량",
                "강화",
                "비교과",
                "프로그램",
                "자격증",
                "대비반",
            )
        )
    if (
        "장애" in value
        and "지원" in value
        and re.search(r"수업|시험", value)
    ):
        expansions.extend(
            (
                "장애학생지원센터",
                "학생지원기관",
                "교수학습",
                "강의지원",
                "교재지원",
            )
        )
    if _is_third_party_reporting_query(value):
        expansions.extend(("센터이용", "Q&A", "제3자", "신고"))
    return tuple(dict.fromkeys(expansions))


def expand_service_retrieval_query(
    question: str,
    institution: str | None = None,
) -> str:
    """Expand a normalized query only for the opt-in service-tuning lane."""

    normalized = normalize_retrieval_query(question, institution)
    # FTS keeps a bounded number of whole-token + Korean-bigram terms. Put the
    # small, high-information expansion first so long conversational wording
    # cannot consume that budget before the document vocabulary is reached.
    terms = list(_service_retrieval_query_expansions(question))
    seen = set(terms)
    for term in normalized.split():
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return " ".join(terms)


def is_authorized(headers: Any) -> bool:
    token = api_token()
    if not token:
        return True

    provided = headers.get("X-RAG-API-Key", "").strip()
    authorization = headers.get("Authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    return bool(provided) and hmac.compare_digest(provided, token)


def gemini_api_key() -> str:
    key = (
        os.environ.get("RAG_GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    )
    if key.lower() in {"your_api_key_here", "replace_me", "changeme"}:
        return ""
    return key.strip()


def gemini_model() -> str:
    return (
        os.environ.get("RAG_GEMINI_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()


def gemini_model_candidates() -> list[str]:
    fallback_value = (
        os.environ.get("RAG_GEMINI_FALLBACK_MODELS")
        or os.environ.get("GEMINI_FALLBACK_MODELS")
        or ""
    )
    fallbacks = [
        item.strip()
        for item in fallback_value.split(",")
        if item.strip()
    ] or list(DEFAULT_GEMINI_FALLBACK_MODELS)
    candidates = [gemini_model(), *fallbacks]
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def generation_mode() -> str:
    value = os.environ.get("RAG_GENERATION_MODE", "auto").strip().lower()
    if value not in SUPPORTED_PROVIDERS:
        return "auto"
    return value


def gemini_enabled() -> bool:
    mode = generation_mode()
    if mode == "extractive":
        return False
    return bool(gemini_api_key())


def validate_provider(value: Any) -> str:
    provider = str(value or generation_mode()).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_provider",
            "provider는 auto, local, frontier, gemini, extractive 중 하나를 선택해 주세요.",
        )
    return provider


def public_provider_name(provider: str) -> str:
    """Map an internal generation provider to the product-facing provider."""

    return "frontier" if provider == "gemini" else provider


def resolve_generation_provider(provider: str) -> str:
    """Resolve the product-facing Frontier AI option to its Gemini adapter."""

    return "gemini" if provider == "frontier" else provider


def public_generation_metadata(
    metadata: dict[str, Any],
    *,
    requested_provider: str,
) -> dict[str, Any]:
    """Keep internal adapter names out of the public generation contract."""

    public = dict(metadata)
    public["requested"] = public_provider_name(requested_provider)

    internal_used = str(public.get("used") or "").strip()
    if internal_used:
        public_used = public_provider_name(internal_used)
        public["used"] = public_used
        if public_used != internal_used:
            public["implementation"] = internal_used

    attempts = public.get("attempts")
    if isinstance(attempts, list):
        public_attempts: list[Any] = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                public_attempts.append(attempt)
                continue
            public_attempt = dict(attempt)
            internal_provider = str(public_attempt.get("provider") or "").strip()
            if internal_provider:
                public_attempt["provider"] = public_provider_name(internal_provider)
            public_attempts.append(public_attempt)
        public["attempts"] = public_attempts

    return public


def local_models() -> list[str]:
    """Return the configured local-model allowlist with a stable default first."""

    configured_default = os.environ.get("RAG_LOCAL_MODEL", "").strip()
    configured_models = [
        value.strip()
        for value in os.environ.get("RAG_LOCAL_MODELS", "").split(",")
        if value.strip()
    ]
    candidates = ([configured_default] if configured_default else []) + configured_models
    models: list[str] = []
    for model in candidates:
        if model not in models:
            models.append(model)
    return models


def local_default_model() -> str | None:
    configured_default = os.environ.get("RAG_LOCAL_MODEL", "").strip()
    if configured_default:
        return configured_default
    models = local_models()
    return models[0] if models else None


def local_model_label(model: str) -> str:
    normalized = model.rstrip("/")
    return normalized.rsplit("/", 1)[-1] or model


def gemini_model_label(model: str) -> str:
    labels = {
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
        "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    }
    return labels.get(model, model)


def is_loopback_bind_host(host: str) -> bool:
    normalized = str(host or "").strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def generation_may_use_local(provider: str) -> bool:
    if provider == "local":
        return True
    if provider != "auto":
        return False
    configured = [
        value.strip().lower()
        for value in os.environ.get(
            "RAG_AUTO_PROVIDER_ORDER",
            "local,frontier,gemini,extractive",
        ).split(",")
    ]
    providers = [
        value
        for value in configured
        if value in SUPPORTED_PROVIDERS and value != "auto"
    ]
    # The generator falls back to its default sequence when the configured
    # order contains no valid provider. That default starts with local.
    return "local" in providers if providers else True


def local_attempt_model(
    attempts: Any,
    fallback_model: str | None = None,
) -> str | None:
    for attempt in reversed(tuple(attempts or ())):
        provider = (
            attempt.get("provider")
            if isinstance(attempt, Mapping)
            else getattr(attempt, "provider", None)
        )
        if provider != "local":
            continue
        status = (
            attempt.get("status")
            if isinstance(attempt, Mapping)
            else getattr(attempt, "status", None)
        )
        if status != "success":
            continue
        model = (
            attempt.get("model")
            if isinstance(attempt, Mapping)
            else getattr(attempt, "model", None)
        )
        return str(model or fallback_model or "").strip() or None
    return None


def with_local_runtime_failure(
    metadata: Mapping[str, Any],
    *,
    code: str | None,
    model: str | None,
) -> dict[str, Any]:
    public = dict(metadata)
    normalized_code = str(code or "").strip()
    if not normalized_code:
        return public
    failure_reason = f"local:{normalized_code}"
    existing_reason = str(public.get("fallback_reason") or "").strip()
    public["fallback_reason"] = (
        f"{failure_reason},{existing_reason}"
        if existing_reason
        else failure_reason
    )
    attempts = list(public.get("attempts") or [])
    attempts.insert(
        0,
        {
            "provider": "local",
            "model": model,
            "status": "error",
            "error": normalized_code,
            "elapsed_ms": 0,
        },
    )
    public["attempts"] = attempts
    return public


def validate_requested_model(
    provider: str,
    value: Any = REQUEST_MODEL_MISSING,
) -> str | None:
    """Resolve a provider model against its server-configured allowlist."""

    if value is REQUEST_MODEL_MISSING:
        return local_default_model() if provider == "local" else None
    if provider == "local":
        if not isinstance(value, str) or not value or value not in local_models():
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_local_model",
                "허용된 로컬 모델을 선택해 주세요.",
            )
        return value
    if provider in {"frontier", "gemini"}:
        if (
            not isinstance(value, str)
            or not value
            or value not in gemini_model_candidates()
        ):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_gemini_model",
                "허용된 Gemini 모델을 선택해 주세요.",
            )
        return value
    raise ApiError(
        HTTPStatus.BAD_REQUEST,
        "model_not_allowed_for_provider",
        "model은 로컬 LLM 또는 프론티어 AI를 직접 선택했을 때만 지정할 수 있습니다.",
    )


def generation_provider_status() -> dict[str, dict[str, Any]]:
    local_model = local_default_model()
    configured_local_models = local_models()
    return {
        "auto": {
            "configured": True,
            "label": "자동 선택",
        },
        "local": {
            "configured": bool(local_model),
            "label": "로컬 LLM",
            "model": local_model,
            "default_model": local_model,
            "models": [
                {
                    "id": model,
                    "label": local_model_label(model),
                    "available": True,
                }
                for model in configured_local_models
            ],
            "api_style": os.environ.get(
                "RAG_LOCAL_API_STYLE", "chat_completions"
            ).strip(),
        },
        "frontier": {
            "configured": bool(gemini_api_key()),
            "label": "프론티어 AI (Gemini)",
            "model": gemini_model(),
            "default_model": gemini_model(),
            "models": [
                {
                    "id": model,
                    "label": gemini_model_label(model),
                    "available": True,
                }
                for model in gemini_model_candidates()
            ],
            "implementation": "gemini",
        },
        "gemini": {
            "configured": bool(gemini_api_key()),
            "label": "Gemini API",
            "model": gemini_model(),
        },
        "extractive": {
            "configured": True,
            "label": "추출형 안전 응답",
            "model": None,
        },
    }


def index_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = connection.execute("SELECT key, value FROM index_meta").fetchall()
    except sqlite3.DatabaseError:
        return {}
    return {str(key): str(value) for key, value in rows}


def parser_profile_label(profile: str) -> str:
    return PARSER_PROFILE_LABELS.get(profile, profile)


def parser_profile_from_index(index_path: Path) -> str | None:
    if not index_path.is_file():
        return None
    connection = sqlite3.connect(str(index_path))
    try:
        profile = index_metadata(connection).get("profile", "").strip()
    finally:
        connection.close()
    return profile or None


def parse_profile_index_spec(value: str) -> tuple[str, Path]:
    profile, separator, raw_path = str(value or "").partition("=")
    profile = profile.strip()
    raw_path = raw_path.strip()
    if (
        not separator
        or not PARSER_PROFILE_PATTERN.fullmatch(profile)
        or not raw_path
    ):
        raise ValueError(
            "--profile-index는 profile=/path/to/index.sqlite 형식이어야 합니다."
        )
    return profile, Path(raw_path)


def build_parser_index_registry(
    default_index: Path,
    profile_specs: list[str],
    requested_default: str | None = None,
) -> tuple[dict[str, Path], str]:
    inferred_default = parser_profile_from_index(default_index) or "default"
    default_profile = (requested_default or inferred_default).strip()
    if not PARSER_PROFILE_PATTERN.fullmatch(default_profile):
        raise ValueError("기본 parser profile 이름이 올바르지 않습니다.")

    registry: dict[str, Path] = {}
    if requested_default and requested_default != inferred_default:
        registry[inferred_default] = default_index
    else:
        registry[default_profile] = default_index

    for spec in profile_specs:
        profile, index_path = parse_profile_index_spec(spec)
        existing = registry.get(profile)
        if existing is not None and existing != index_path:
            raise ValueError(f"{profile} parser profile이 두 인덱스에 연결됐습니다.")
        registry[profile] = index_path

    if default_profile not in registry:
        if requested_default and inferred_default == default_profile:
            registry[default_profile] = default_index
        else:
            raise ValueError(
                f"기본 parser profile '{default_profile}' 인덱스가 없습니다."
            )

    for profile, index_path in registry.items():
        configured_profile = parser_profile_from_index(index_path)
        if configured_profile and configured_profile != profile:
            raise ValueError(
                f"{profile}에 연결한 인덱스의 실제 profile은 "
                f"{configured_profile}입니다: {index_path}"
            )

    ordered = {
        profile: registry[profile]
        for profile in (
            *PARSER_PROFILE_ORDER,
            *sorted(set(registry) - set(PARSER_PROFILE_ORDER)),
        )
        if profile in registry
    }
    return ordered, default_profile


def resolve_parser_target(
    value: Any,
    targets: Mapping[str, ParserIndexTarget],
    default_profile: str,
    *,
    require_ready: bool = True,
) -> ParserIndexTarget:
    if value is None or (isinstance(value, str) and not value.strip()):
        profile = default_profile
    elif isinstance(value, str):
        profile = value.strip()
    else:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_parser_profile",
            "parser_profile 형식이 올바르지 않습니다.",
        )

    target = targets.get(profile)
    if target is None:
        choices = ", ".join(targets)
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_parser_profile",
            f"parser_profile은 {choices} 중 하나를 선택해 주세요.",
        )
    if require_ready and not target.index_path.is_file():
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "parser_profile_unavailable",
            f"{parser_profile_label(profile)} 검색 인덱스가 준비되지 않았습니다.",
        )
    return target


def list_institutions(index_path: Path) -> list[str]:
    connection = sqlite3.connect(str(index_path))
    rows = connection.execute(
        """
        SELECT institution
        FROM chunks
        WHERE institution IS NOT NULL AND institution != ''
        GROUP BY institution
        ORDER BY institution
        """
    ).fetchall()
    connection.close()
    return [row[0] for row in rows]


def dense_index_path() -> Path:
    value = os.environ.get("RAG_DENSE_INDEX", "").strip()
    return Path(value) if value else DEFAULT_DENSE_INDEX


def learned_dense_root_path() -> Path:
    value = os.environ.get("RAG_LEARNED_DENSE_ROOT", "").strip()
    return Path(value) if value else DEFAULT_LEARNED_DENSE_ROOT


def learned_dense_artifact_path(
    learned_root: Path,
    profile: str,
    directory_name: str,
) -> Path:
    """Resolve one model artifact without mixing parser-profile corpora.

    New multi-profile artifacts live below ``<root>/<profile>/``.  The original
    single-profile layout remains a Cascade-only fallback so existing builds do
    not need to be copied or rebuilt.
    """

    profile_path = learned_root / profile / directory_name
    if profile_path.is_dir():
        return profile_path
    if profile == "cascade":
        legacy_path = learned_root / directory_name
        if legacy_path.is_dir():
            return legacy_path
    return profile_path


def share_learned_dense_embedder(
    learned_index: LearnedDenseIndex,
    cache: dict[tuple[Any, ...], Any],
) -> None:
    """Reuse one lazy query model across parser-profile vector matrices."""

    key = (
        learned_index.model_id,
        learned_index.model_revision,
        learned_index.query_prefix,
        learned_index.embedder.max_sequence_length,
    )
    learned_index.embedder = cache.setdefault(key, learned_index.embedder)


def configured_retrieval_mode() -> str:
    value = os.environ.get("RAG_RETRIEVAL_MODE", "bm25").strip().lower()
    return value if value in RETRIEVAL_MODE_LABELS else "bm25"


def dense_index_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    connection = sqlite3.connect(str(path))
    try:
        rows = connection.execute(
            "SELECT key, value FROM dense_meta"
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    finally:
        connection.close()
    return {str(key): str(value) for key, value in rows}


def index_stats(
    index_path: Path,
    dense_path: Path | None = None,
) -> dict[str, Any]:
    if not index_path.exists():
        return {
            "ok": False,
            "ready": False,
            "status": "not_ready",
            "index": str(index_path),
            "default_provider": public_provider_name(generation_mode()),
            "providers": generation_provider_status(),
            "pipeline": {
                "parser": {"status": "external"},
                "corpus_gate": {"status": "not_ready"},
                "chunk": {"status": "not_ready"},
                "bm25": {"status": "not_ready"},
                "dense": {"status": "not_ready"},
                "rrf": {"status": "not_ready"},
                "reranker": {"status": "not_ready"},
                "generation": {"status": "not_ready"},
                "citation": {"status": "not_ready"},
            },
        }

    index_sha256, index_size_bytes = _cached_index_identity(index_path)

    connection = sqlite3.connect(str(index_path))
    chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    document_count = connection.execute(
        "SELECT COUNT(DISTINCT document_id) FROM chunks"
    ).fetchone()[0]
    institution_count = connection.execute(
        "SELECT COUNT(DISTINCT institution) FROM chunks WHERE institution != ''"
    ).fetchone()[0]
    metadata = index_metadata(connection)
    connection.close()
    dense_path = dense_path or dense_index_path()
    dense_metadata = dense_index_metadata(dense_path)
    dense_revision = dense_metadata.get("corpus_revision")
    dense_ready = bool(
        dense_metadata
        and dense_revision
        and dense_revision == metadata.get("corpus_revision")
    )
    dense_reason = None
    if dense_path.is_file() and not dense_metadata:
        dense_reason = "invalid_dense_index"
    elif dense_path.is_file() and not dense_ready:
        dense_reason = "corpus_revision_mismatch"
    providers = generation_provider_status()
    generation_ready = any(
        provider.get("configured") is True for provider in providers.values()
    )
    status = "ready" if generation_ready else "degraded"
    return {
        "index_sha256": index_sha256,
        "index_size_bytes": index_size_bytes,
        "ok": True,
        "ready": True,
        "status": status,
        "index": str(index_path),
        "chunk_count": chunk_count,
        "document_count": document_count,
        "institution_count": institution_count,
        "corpus_revision": metadata.get("corpus_revision"),
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "run_id": metadata.get("run_id") or None,
        "profile": metadata.get("profile") or None,
        "generation_mode": generation_mode(),
        "gemini_configured": bool(gemini_api_key()),
        "gemini_model": gemini_model(),
        "gemini_model_candidates": gemini_model_candidates(),
        "default_provider": public_provider_name(generation_mode()),
        "providers": providers,
        "dense_index": str(dense_path),
        "dense_ready": dense_ready,
        "dense_corpus_revision": dense_revision,
        "dense_reason": dense_reason,
        "pipeline": {
            "parser": {
                "status": "ready",
                "mode": "verified_external_artifact",
            },
            "corpus_gate": {
                "status": "ready" if metadata.get("corpus_revision") else "legacy",
                "corpus_revision": metadata.get("corpus_revision"),
            },
            "chunk": {"status": "ready", "count": chunk_count},
            "bm25": {"status": "ready", "count": chunk_count},
            "dense": {
                "status": "ready" if dense_ready else "disabled",
                "index": str(dense_path),
                "reason": dense_reason,
            },
            "rrf": {"status": "ready" if dense_ready else "single_lane"},
            "reranker": {
                "status": "ready",
                "kind": (
                    "lexical_fallback"
                    if dense_ready
                    else "bm25_document_diverse"
                ),
            },
            "generation": {
                "status": "ready" if generation_ready else "fallback",
                "default_provider": public_provider_name(generation_mode()),
                "extractive_fallback": True,
            },
            "citation": {"status": "ready", "schema": "CitationV1"},
        },
        "max_top_k": max_top_k(),
        "max_question_chars": question_max_chars(),
        "api_token_required": bool(api_token()),
    }


def parser_profile_summaries(
    targets: Mapping[str, ParserIndexTarget],
    dense_path: Path,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for profile, target in targets.items():
        stats = index_stats(target.index_path, dense_path)
        retrieval_modes = retrieval_mode_summaries(target)
        learned_dense_ready = any(
            item.get("ready") is True and item.get("id") != "bm25"
            for item in retrieval_modes
        )
        summaries.append(
            {
                "id": profile,
                "label": parser_profile_label(profile),
                "ready": stats.get("ready") is True,
                "chunk_count": stats.get("chunk_count"),
                "document_count": stats.get("document_count"),
                "run_id": stats.get("run_id"),
                "profile": stats.get("profile"),
                "corpus_revision": stats.get("corpus_revision"),
                "source_manifest_sha256": stats.get("source_manifest_sha256"),
                "index_sha256": stats.get("index_sha256"),
                "index_size_bytes": stats.get("index_size_bytes"),
                "dense_ready": (
                    stats.get("dense_ready") is True or learned_dense_ready
                ),
                "default_retrieval_mode": default_retrieval_mode(target),
                "retrieval_modes": retrieval_modes,
                "retriever_warning": target.warning,
                "reason": (
                    None
                    if stats.get("ready") is True
                    else stats.get("status") or "index_not_ready"
                ),
            }
        )
    return summaries


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"([\[\(]page \d+[\]\)])", "", value, flags=re.IGNORECASE)
    return value.strip()


def result_source_text(result: dict[str, Any]) -> str:
    return str(result.get("text") or result.get("preview") or "")


def evaluation_context_trace(
    results: list[dict[str, Any]],
    *,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    """Return an eval-only, whitelisted copy of retrieved context text."""
    trace: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        source_text = result_source_text(result)
        metadata = (
            dict(result.get("metadata") or {})
            if isinstance(result.get("metadata"), dict)
            else {}
        )
        trace.append(
            {
                "rank": rank,
                "stage": stage,
                "source_number": result.get("source_number"),
                "chunk_id": result.get("chunk_id"),
                "document_id": result.get("document_id") or result.get("doc_id"),
                "chunk_index": result.get("chunk_index"),
                "corpus_revision": result.get("corpus_revision"),
                "institution": result.get("institution"),
                "file_name": result.get("file_name"),
                "source_title": result.get("source_title"),
                "source_host": result.get("source_host"),
                "source_url": result.get("source_url"),
                "published_at": result.get("published_at"),
                "section_path": result.get("section_path") or metadata.get("section_path"),
                "table_ids": result.get("table_ids") or metadata.get("table_ids") or [],
                "block_ids": result.get("block_ids") or metadata.get("block_ids") or [],
                "locations": result.get("locations") or [],
                "retrieval": dict(result.get("retrieval") or {}),
                "text": source_text,
                "text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            }
        )
    return trace


def generation_input_trace(
    question: str,
    results: list[dict[str, Any]],
    *,
    role_perspective: str | None = None,
) -> dict[str, Any]:
    """Capture the exact provider-neutral generation input for eval runs."""

    prompt = build_generation_prompt(
        question,
        results,
        role_perspective=role_perspective,
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    system_sha = hashlib.sha256(
        GENERATION_SYSTEM_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    envelope = json.dumps(
        {
            "system_instruction": GENERATION_SYSTEM_INSTRUCTION,
            "user_prompt": prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "user_prompt": prompt,
        "prompt_sha256": prompt_sha,
        "system_instruction": GENERATION_SYSTEM_INSTRUCTION,
        "system_instruction_sha256": system_sha,
        "input_sha256": hashlib.sha256(envelope.encode("utf-8")).hexdigest(),
    }


def chat_candidate_limit(top_k: int) -> int:
    """Overfetch chat candidates so duplicate evidence does not consume slots."""

    requested = max(1, int(top_k))
    return min(
        MAX_CHAT_CANDIDATES,
        max(requested, requested * CHAT_CANDIDATE_MULTIPLIER),
    )


def _canonical_context_text(result: dict[str, Any]) -> str:
    value = unicodedata.normalize("NFKC", result_source_text(result))
    return normalize_text(value).casefold()


def select_distinct_contexts(
    rows: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep ranked contexts, removing only normalized exact copies."""

    limit = max(0, int(top_k))
    selected: list[dict[str, Any]] = []
    exact_keys: dict[tuple[str, str], str] = {}
    removed: list[dict[str, Any]] = []
    exact_removed = 0
    overflow_unique_count = 0

    for value in rows:
        row = dict(value)
        canonical = _canonical_context_text(row)
        institution = str(row.get("institution") or "").strip().casefold()
        chunk_id = str(row.get("chunk_id") or "")
        duplicate_of = None
        reason = None

        if canonical:
            duplicate_of = exact_keys.get((institution, canonical))
            if duplicate_of is not None:
                reason = "normalized_exact"
                exact_removed += 1

        if reason is not None:
            removed.append(
                {
                    "chunk_id": chunk_id,
                    "duplicate_of_chunk_id": duplicate_of,
                    "reason": reason,
                }
            )
            continue

        if canonical:
            exact_keys[(institution, canonical)] = chunk_id

        if len(selected) >= limit:
            overflow_unique_count += 1
            continue

        retrieval = dict(row.get("retrieval") or {})
        previous_rank = retrieval.get("final_rank")
        if previous_rank is not None:
            retrieval["pre_dedupe_rank"] = previous_rank
        retrieval["final_rank"] = len(selected) + 1
        row["retrieval"] = retrieval
        selected.append(row)

    diagnostics = {
        "requested_count": limit,
        "candidate_count": len(rows),
        "scanned_count": len(rows),
        "unscanned_count": 0,
        "kept_count": len(selected),
        "overflow_unique_count": overflow_unique_count,
        "removed_count": len(removed),
        "exact_removed": exact_removed,
        "near_removed": 0,
        "removed": removed,
    }
    return selected, diagnostics


def result_source_excerpt(result: dict[str, Any]) -> str:
    return normalize_text(result_source_text(result))[:source_chars()]


def claim_support_excerpt(
    claim: str,
    result: dict[str, Any],
) -> tuple[str, int, int]:
    """Return a bounded source window centered on the claim's support."""

    source = normalize_text(result_source_text(result))
    limit = source_chars()
    if len(source) <= limit:
        return source, 0, len(source)

    normalized_claim = normalize_text(claim)
    exact_anchor = normalized_claim[:80]
    exact_position = source.find(exact_anchor) if exact_anchor else -1
    if exact_position >= 0:
        start = max(
            0,
            min(
                exact_position - max(0, (limit - len(exact_anchor)) // 2),
                len(source) - limit,
            ),
        )
        end = min(len(source), start + limit)
        return source[start:end], start, end

    claim_terms = {
        term
        for term in tokenize(normalized_claim, include_ngrams=False)
        if not term.isdigit()
    }
    claim_values = extract_critical_values(normalized_claim)
    positions: set[int] = {0, max(0, len(source) - limit)}
    folded_source = source.casefold()
    for term in claim_terms:
        position = folded_source.find(term.casefold())
        if position >= 0:
            positions.add(max(0, min(position - limit // 3, len(source) - limit)))

    best: tuple[tuple[int, int, int, int], int, str] | None = None
    for start in sorted(positions):
        window = source[start : start + limit]
        window_terms = set(tokenize(window, include_ngrams=False))
        matched_terms = sum(
            _claim_term_supported(term, window_terms)
            for term in claim_terms
        )
        window_values = extract_critical_values(window)
        matched_values = len(claim_values.intersection(window_values))
        score = (
            int(bool(claim_values) and claim_values.issubset(window_values)),
            matched_values,
            matched_terms,
            -start,
        )
        candidate = (score, start, window)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return source[:limit], 0, limit
    _, start, window = best
    return window, start, start + len(window)


def public_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for result in results:
        row = {key: value for key, value in result.items() if key != "text"}
        row["document_id"] = result.get("document_id") or result.get("doc_id")
        # Temporary alias for clients built against the original MVP contract.
        row["doc_id"] = row["document_id"]
        locations = result.get("locations")
        if isinstance(locations, list):
            limit = max_citation_locations()
            row["location_count"] = len(locations)
            row["locations_truncated"] = len(locations) > limit
            row["locations"] = locations[:limit]
        public.append(row)
    return public


def citation_for_result(
    result: dict[str, Any],
    claim: str | None = None,
) -> dict[str, Any]:
    locations = result.get("locations")
    locations = locations if isinstance(locations, list) else []
    limit = max_citation_locations()
    if claim is None:
        excerpt = result_source_excerpt(result)
        excerpt_start = 0
        excerpt_end = len(excerpt)
        claim_sha = None
    else:
        excerpt, excerpt_start, excerpt_end = claim_support_excerpt(
            claim, result
        )
        claim_sha = hashlib.sha256(claim.encode("utf-8")).hexdigest()
    return {
        "citation_id": (
            f"citation:{result.get('chunk_id', '')}"
            + (f":{claim_sha[:12]}" if claim_sha else "")
        ),
        "source_number": result.get("source_number"),
        "chunk_id": result.get("chunk_id"),
        "document_id": result.get("document_id") or result.get("doc_id"),
        "corpus_revision": result.get("corpus_revision"),
        "excerpt": excerpt,
        "excerpt_start": excerpt_start,
        "excerpt_end": excerpt_end,
        "excerpt_sha256": hashlib.sha256(
            excerpt.encode("utf-8")
        ).hexdigest(),
        "claim_sha256": claim_sha,
        "locations": locations[:limit],
        "location_count": len(locations),
        "locations_truncated": len(locations) > limit,
        "institution": result.get("institution"),
        "file_name": result.get("file_name"),
        "source_path": result.get("source_path"),
        "relative_path": result.get("relative_path"),
        "source_title": result.get("source_title"),
        "source_url": result.get("source_url"),
        "download_url": result.get("download_url"),
        "fetched_at": result.get("fetched_at"),
        "published_at": result.get("published_at"),
    }


def claim_citations_for_result(
    claim: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten location metadata to match the public ClaimCitation contract."""

    citation = citation_for_result(result, claim=claim)
    locations = citation.pop("locations", [])
    if not locations:
        return [citation]
    return [
        {**citation, **location}
        for location in locations
        if isinstance(location, dict)
    ] or [citation]


_NAVIGATION_NOISE_RE = re.compile(
    r"^[\s\-*•·▶▷☞#]*"
    r"(?:공지\s*사항|자세한\s*내용|상세\s*내용|첨부\s*파일|원문|메뉴|목록|홈)"
    r"(?:\s*(?:바로가기|더\s*보기|보기|이동|열기|다운로드))?"
    r"[\s.!?。！？]*$",
    re.IGNORECASE,
)


def is_low_quality_candidate(text: str, *, minimum_chars: int = 20) -> bool:
    if _NAVIGATION_NOISE_RE.fullmatch(normalize_text(text)):
        return True
    terms = tokenize(text, include_ngrams=False)
    if len(terms) >= 18 and len(set(terms)) / len(terms) < 0.42:
        return True
    if len(re.findall(r"[-_=]{4,}", text)) >= 1:
        return True
    if len(text) < minimum_chars:
        return True
    return False


_DATE_DOT_SENTINEL = "\ue000"
_DOTTED_FULL_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>[‘’']?\d{2}|(?:19|20)\d{2})\.\s*"
    r"(?P<month>\d{1,2})\.\s*(?P<day>\d{1,2})"
    r"(?P<terminal>\.)?(?!\d)"
)
_DOTTED_MONTH_DAY_PATTERN = re.compile(
    rf"(?<![\d.{_DATE_DOT_SENTINEL}])(?P<month>\d{{1,2}})\.\s*"
    r"(?P<day>\d{1,2})(?P<terminal>\.)?(?!\d)"
)
_DOTTED_DATE_CONTINUATION_RE = re.compile(
    r"^\s*(?:\((?:월|화|수|목|금|토|일)(?:요일)?\)|"
    r"[~∼～\-–—]|및(?:\s|$)|또는(?:\s|$)|이거나(?:\s|$))"
)


def _protect_dotted_date_match(
    match: re.Match[str],
    *,
    internal_periods: int,
) -> str:
    month = int(match.group("month"))
    day = int(match.group("day"))
    try:
        date(2000, month, day)
    except ValueError:
        return match.group(0)

    value = match.group(0)
    period_positions = [
        index for index, character in enumerate(value) if character == "."
    ]
    protected_positions = set(period_positions[:internal_periods])
    if (
        match.group("terminal")
        and _DOTTED_DATE_CONTINUATION_RE.match(
            match.string[match.end() :]
        )
    ):
        protected_positions.add(period_positions[-1])
    return "".join(
        _DATE_DOT_SENTINEL if index in protected_positions else character
        for index, character in enumerate(value)
    )


def _protect_dotted_dates(text: str) -> str:
    """Protect periods inside Korean dotted dates before sentence splitting.

    Administrative answers commonly contain values such as ``2026. 8. 24.(월)``.
    Treating every period followed by whitespace as a sentence boundary turns those
    dates into short fragments, which are then discarded by the quality filter.
    """

    protected = _DOTTED_FULL_DATE_PATTERN.sub(
        lambda match: _protect_dotted_date_match(
            match,
            internal_periods=2,
        ),
        text,
    )
    return _DOTTED_MONTH_DAY_PATTERN.sub(
        lambda match: _protect_dotted_date_match(
            match,
            internal_periods=1,
        ),
        protected,
    )


def split_candidate_sentences(text: str, *, minimum_chars: int = 20) -> list[str]:
    text = re.sub(r"([\[\(]page \d+[\]\)])", "", text, flags=re.IGNORECASE)
    text = _protect_dotted_dates(text)
    rough_parts = re.split(
        r"\n+|(?<=[.!?。！？])\s+|(?<=다\.)\s+|(?<=임\.)\s+|(?<=음\.)\s+",
        text,
    )
    candidates: list[str] = []
    for part in rough_parts:
        subparts = re.split(r"\s+(?=[①②③④⑤⑥⑦⑧⑨⑩□ㅇ◦])", part)
        for subpart in subparts:
            subpart = subpart.replace(_DATE_DOT_SENTINEL, ".")
            subpart = normalize_text(subpart)
            if len(subpart) > 280:
                subpart = subpart[:280].rsplit(" ", 1)[0].strip()
            if subpart and not is_low_quality_candidate(
                subpart, minimum_chars=minimum_chars
            ):
                candidates.append(subpart)
    return candidates


def overlap_score(question_terms: set[str], text: str) -> float:
    terms = set(tokenize(text, include_ngrams=False))
    if not terms:
        return 0.0
    matched = 0
    for question_term in question_terms:
        if question_term in terms:
            matched += 1
            continue
        if re.fullmatch(r"[가-힣]{2,}", question_term) and any(
            question_term in term for term in terms
        ):
            # Korean particles and verb endings are commonly attached to the
            # content word (휴학 → 휴학할, 휴학기간은).
            matched += 1
    return matched / max(1, len(question_terms))


TEMPORAL_QUERY_RE = re.compile(
    r"언제|기간|일정|날짜|마감|몇\s*시"
)
_TEMPORAL_DATE_QUERY_RE = re.compile(r"언제|기간|일정|날짜|마감")
_TEMPORAL_TIME_QUERY_RE = re.compile(r"몇\s*시")
PROCEDURE_QUERY_RE = re.compile(
    r"절차|방법|단계|어떻게|신청|제출|업로드|"
    r"(?:납부|접수|등록)\s*(?:방법|절차)"
)
_TEMPORAL_DATE_RE = re.compile(
    r"(?<!\d)(?:(?:19|20)?\d{2}\s*[.\-/년]\s*)?"
    r"\d{1,2}\s*[.\-/월]\s*\d{1,2}\s*(?:[.일]|\([월화수목금토일]\))?"
)
_TEMPORAL_TIME_RE = re.compile(
    r"(?<!\d)(?:[01]?\d|2[0-3])\s*:\s*[0-5]\d(?!\d)|"
    r"(?:오전|오후)\s*\d{1,2}\s*시"
)
_AMOUNT_QUERY_RE = re.compile(
    r"금액|지원액|지원\s*금액|한도|"
    r"얼마나(?!\s*(?:걸리|소요))|얼마(?!나|\s*(?:걸리|소요))|"
    r"비용|수수료|요금|"
    r"납부액"
)
_RATE_QUERY_RE = re.compile(r"금리|이율")
_ELIGIBILITY_QUERY_RE = re.compile(
    r"자격|대상|누가|소득\s*분위|지원\s*구간|선발\s*조건|"
    r"신청\s*조건|지원\s*조건|(?:어떤|어느)\s*학과"
)
_ACADEMIC_DEPARTMENT_QUERY_RE = re.compile(r"(?:어떤|어느)\s*학과")
# Merely naming an application/registration does not ask for a method.  Requiring
# an explicit interrogative keeps the new marginal-facet path byte-stable for
# single date/amount questions such as "신청 기간은 언제인가요?".
_METHOD_QUERY_RE = re.compile(
    r"방법|절차|단계|"
    r"어떻게\s*(?:(?:직접|온라인으로?|오프라인으로?)\s*)?"
    r"(?:신청|접수|제출|납부|등록|이용|진행)|"
    r"(?:어디서|어디로)\s*(?:신청|접수|제출|납부|등록)|"
    r"(?:신청|접수|제출|납부|등록)\s*(?:하는\s*)?(?:법|방법|절차)"
)
_DESTINATION_QUERY_RE = re.compile(
    r"통장|계좌|"
    r"(?:어디|어느\s*곳)(?:로|에)?\s*(?:입금|지급|상환)|"
    r"(?:입금|지급|상환).{0,20}(?:어디|통장|계좌)"
)
_BUDGET_AMOUNT_QUERY_RE = re.compile(
    r"예산.{0,20}(?:규모|총액|전체)|"
    r"(?:전체|총|총액|규모).{0,20}예산"
)
_BUDGET_SHARE_QUERY_RE = re.compile(
    r"(?:영역별|분야별).{0,20}(?:비중|구성비)|"
    r"(?:비중|구성비).{0,20}(?:영역별|분야별)"
)
_DOCUMENT_SUBMISSION_DEADLINE_QUERY_RE = re.compile(
    r"(?:성적표|서류|증빙|신청서|보고서|논문).{0,30}"
    r"(?:언제까지|기한|마감).{0,20}(?:내|제출)|"
    r"(?:언제까지|기한|마감).{0,30}"
    r"(?:성적표|서류|증빙|신청서|보고서|논문).{0,20}(?:내|제출)"
)
_DATE_RELATION_RE = re.compile(
    r"(?:신청|접수|제출|납부|등록|운영|시험|심사|모집|공고)\s*"
    r"(?:기간|일정|일시|마감|마감일)|"
    r"(?:기간|일정|일시|마감일?)\s*[:：]"
)
_AMOUNT_RELATION_RE = re.compile(
    r"금액|지원액|지원금|한도|비용|수수료|심사료|요금|납부액|등록금|"
    r"원금|이자"
)
_RATE_RELATION_RE = re.compile(r"금리|이율")
_ELIGIBILITY_RELATION_RE = re.compile(
    r"신청\s*자격|지원\s*자격|응시\s*자격|선발\s*대상|신청\s*대상|"
    r"지원\s*대상|응시\s*대상|대상자|자격\s*요건|지원\s*구간|"
    r"소득\s*분위|대출\s*대상|대상\s*학과"
)
_FOREIGN_ADMISSION_NATIONALITY_RE = re.compile(
    r"(?:지원자.{0,40}부모|부모.{0,40}지원자).{0,80}외국\s*국적|"
    r"이중국적.{0,40}(?:지원\s*불가|제한|이탈|상실)"
)
_FOREIGN_ADMISSION_LANGUAGE_RE = re.compile(
    r"(?:언어능력|TOPIK|TOEFL|IELTS|언어교육원).{0,180}"
    r"(?:기준|급|점|수료|성적|지원\s*가능)"
)
_FOREIGN_ADMISSION_EDUCATION_RE = re.compile(
    r"(?:학력|고등학교|교육과정).{0,140}"
    r"(?:졸업|수료|인정|요건|지원\s*가능)"
)
_EXCHANGE_SELECTION_SCALE_RELATION_RE = re.compile(
    r"선발\s*규모.{0,180}(?:명|인원|개\s*(?:언어|국|대학))"
)
_EXCHANGE_SELECTION_SCHEDULE_RELATION_RE = re.compile(
    r"(?:온라인\s*지원|On\s*-?\s*Line\s*지원|온라인\s*접수)"
    r".{0,100}(?:(?:19|20)\d{2}\s*[.\-/년]\s*)?"
    r"\d{1,2}\s*[.\-/월]\s*\d{1,2}",
    re.IGNORECASE,
)
_THIRD_PARTY_REPORT_PERMISSION_RE = re.compile(
    r"피해자가?\s*아닌\s*제\s*3\s*자.{0,80}(?:신고|할\s*수)"
)
_THIRD_PARTY_VICTIM_CONSENT_RE = re.compile(
    r"피해자\s*본인.{0,60}의사.{0,80}(?:상담|조사|결정)"
)
_THIRD_PARTY_VICTIM_IDENTIFICATION_RE = re.compile(
    r"피해자\s*본인.{0,60}인적사항.{0,80}(?:알려|제공|필요)"
)
_GROUP_VISA_RESERVATION_RELATION_RE = re.compile(
    r"단체접수.{0,180}출입국.{0,180}(?:서비스|제출)|"
    r"사전\s*예약.{0,100}(?:불가능|반드시\s*예약)"
)
_GROUP_VISA_SCHEDULE_RELATION_RE = re.compile(
    r"회차.{0,100}접수기간.{0,100}접수대상",
    re.DOTALL,
)
_GROUP_VISA_FEE_RELATION_RE = re.compile(
    r"D\s*[-‐‑‒–—]?\s*2\s*체류기간\s*연장.{0,500}"
    r"수수료\s*\(\s*\d[\d,]*\s*원\s*\).{0,160}현금",
    re.IGNORECASE | re.DOTALL,
)
_ASSISTIVE_COPAY_SUMMARY_RELATION_RE = re.compile(
    r"정보통신\s*보조기기.{0,220}(?:선정자|최종\s*선정).{0,180}"
    r"(?:자부담금|개인부담금).{0,80}전액\s*지원",
    re.DOTALL,
)
_ASSISTIVE_COPAY_PROCEDURE_RELATION_RE = re.compile(
    r"(?:자부담금|개인부담금).{0,100}납부.{0,180}"
    r"(?:납부\s*증빙자료|증빙자료\s*확인).{0,160}지원",
    re.DOTALL,
)
_ASSISTIVE_COPAY_DOCUMENTS_RELATION_RE = re.compile(
    r"제출\s*서류.{0,500}(?:자부담금|개인부담금)\s*납부\s*증빙자료"
    r".{0,500}(?:제출\s*방법|구글\s*폼|Google\s*Form)",
    re.IGNORECASE | re.DOTALL,
)
_METHOD_RELATION_RE = re.compile(
    r"신청\s*방법|접수\s*방법|제출\s*방법|납부\s*방법|등록\s*방법|"
    r"이용\s*방법|진행\s*절차|신청\s*절차|접수\s*절차|"
    r"온라인\s*(?:신청|접수|제출)|학생지원시스템"
)
_DESTINATION_RELATION_RE = re.compile(
    r"(?:입금|지급|상환)[^\n]{0,100}(?:계좌|통장)|"
    r"(?:계좌|통장)[^\n]{0,100}(?:입금|지급|상환)"
)
_METHOD_DETAIL_RE = re.compile(
    r"온라인|오프라인|이메일|전자우편|우편|방문|홈페이지|누리집|"
    r"모바일\s*(?:앱|애플리케이션)|애플리케이션|학생지원시스템|"
    r"로그인|업로드|첨부|제출처|접수처|[→>]"
)
_METHOD_ACTIONS = (
    "신청",
    "접수",
    "제출",
    "납부",
    "등록",
    "이용",
    "진행",
)
_QUERY_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9-]{2,}"
)
_GENERIC_QUERY_IDENTIFIERS = frozenset(
    {"APP", "FAQ", "HWP", "HTML", "PDF", "PNU", "PPT", "PPTX"}
)
_QUERY_ANCHOR_STOP_TERMS = frozenset(
    {
        "이번",
        "어느",
        "언제",
        "언제까지",
        "어떻게",
        "있나요",
        "되나요",
        "내야",
        "하고",
        "방법",
        "절차",
        "단계",
        "신청",
        "제출",
        "이미",
        "냈는데",
        "해요",
        "수도",
        "알려",
        "주세요",
        "하나요",
    }
)
_KOREAN_QUERY_SUFFIXES = (
    "으려면",
    "하려면",
    "이라면",
    "이라서",
    "인데",
    "해야",
    "하려고",
    "에서는",
    "으로는",
    "에게서",
    "까지는",
    "부터는",
    "인가요",
    "에서",
    "으로",
    "하고",
    "까지",
    "부터",
    "이며",
    "나요",
    "할",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "도",
    "로",
    "와",
    "과",
)
_PROCEDURE_CUES = (
    "로그인",
    "접속",
    "클릭",
    "선택",
    "입력",
    "업로드",
    "제출",
    "신청",
    "이체",
    "송금",
    "결제",
    "계좌",
    "인증",
    "otp",
    "다음단계",
    "완료",
)
_MAX_FACET_SIBLING_COMPLETIONS = 2
_NUMERIC_QUERY_SCOPE_TERMS = (
    "교원",
    "교수",
    "직원",
    "조교",
    "학부생",
    "대학원생",
    "재학생",
    "휴학생",
    "수료생",
    "신입생",
    "편입생",
    "학부",
    "대학원",
    "원금상환",
    "이자지원",
    "원금",
    "이자",
)


def _is_table_context(result: dict[str, Any]) -> bool:
    table_ids = result.get("table_ids")
    if not isinstance(table_ids, list) or not table_ids:
        return False
    return bool(re.search(r"\d", result_source_text(result)))


def _document_id(result: dict[str, Any]) -> str:
    return str(
        result.get("document_id")
        or result.get("doc_id")
        or ""
    )


def _source_title_text(result: dict[str, Any]) -> str:
    metadata = (
        result.get("metadata")
        if isinstance(result.get("metadata"), dict)
        else {}
    )
    return " ".join(
        str(value or "")
        for value in (
            result.get("source_title"),
            result.get("file_name"),
            metadata.get("source_title"),
            metadata.get("file_name"),
        )
    )


def _query_anchor_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in tokenize(query, include_ngrams=False):
        term = raw_term.casefold()
        if term.isdigit():
            continue
        if re.fullmatch(r"[가-힣]+", term):
            for suffix in _KOREAN_QUERY_SUFFIXES:
                if term.endswith(suffix) and len(term) - len(suffix) >= 2:
                    term = term[: -len(suffix)]
                    break
        if (
            len(term) < 2
            or term in _QUERY_ANCHOR_STOP_TERMS
            or term in seen
        ):
            continue
        seen.add(term)
        terms.append(term)
    return tuple(terms)


def _term_in_text(term: str, text: str) -> bool:
    folded = text.casefold()
    if re.fullmatch(r"[a-z][a-z0-9-]*", term):
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                folded,
            )
        )
    compact = re.sub(r"\s+", "", folded)
    if term in compact:
        return True
    # Administrative titles commonly insert ``청구`` between the user's
    # broader term and ``논문``.  Treating the compound as unrelated can make
    # a generic graduate-school schedule outrank the exact thesis-review
    # document before sibling completion gets a chance to inspect it.
    if term == "학위논문" and "학위청구논문" in compact:
        return True
    return False


def _query_identifiers(query: str) -> frozenset[str]:
    return frozenset(
        match.group(0).upper()
        for match in _QUERY_IDENTIFIER_RE.finditer(query)
        if match.group(0).upper() not in _GENERIC_QUERY_IDENTIFIERS
    )


def _strong_anchor_score(
    result: dict[str, Any],
    *,
    query_terms: tuple[str, ...],
    query_identifiers: frozenset[str],
) -> tuple[int, int, int] | None:
    title = _source_title_text(result)
    body = result_source_text(result)
    identity_text = f"{title} {body}".upper()
    identifier_matches = sum(
        bool(
            re.search(
                rf"(?<![A-Z0-9]){re.escape(identifier)}(?![A-Z0-9])",
                identity_text,
            )
        )
        for identifier in query_identifiers
    )
    title_matches = sum(_term_in_text(term, title) for term in query_terms)
    body_matches = sum(_term_in_text(term, body) for term in query_terms)
    distinctive_title_matches = sum(
        len(term) >= 4 and _term_in_text(term, title)
        for term in query_terms
    )
    if (
        identifier_matches == 0
        and title_matches < 2
        and not (title_matches >= 1 and body_matches >= 2)
        and not (
            distinctive_title_matches >= 1
            and body_matches >= 1
        )
        and body_matches < 3
    ):
        return None
    return identifier_matches, title_matches, body_matches


def _temporal_facets(text: str) -> frozenset[str]:
    facets: set[str] = set()
    if _TEMPORAL_DATE_RE.search(text):
        facets.add("date")
    if _TEMPORAL_TIME_RE.search(text):
        facets.add("time")
    return frozenset(facets)


_DOCUMENT_DEADLINE_OBJECTS = (
    "성적표",
    "서류",
    "증빙",
    "신청서",
    "보고서",
    "논문",
)


def _supports_requested_deadline(text: str, query: str) -> bool:
    """Keep a deadline only when it governs the document named by the user."""

    requested_objects = tuple(
        term for term in _DOCUMENT_DEADLINE_OBJECTS if term in query
    )
    if not requested_objects:
        return False
    for raw_unit in str(text or "").splitlines():
        unit = normalize_text(raw_unit)
        if not unit:
            continue
        if not any(term in unit for term in requested_objects):
            continue
        if not re.search(r"제출|신청|접수|납부|등록", unit):
            continue
        if not re.search(r"까지|이전|기한|마감", unit):
            continue
        if not (_TEMPORAL_DATE_RE.search(unit) or _TEMPORAL_TIME_RE.search(unit)):
            continue
        return True
    return False


def _answer_temporal_facets(
    text: str,
    *,
    query: str | None = None,
) -> frozenset[str]:
    """Return answer-bearing temporal values, excluding outline/table codes."""

    facets = set(_temporal_facets(text))
    if "date" in facets:
        date_supported = bool(_DATE_RELATION_RE.search(text))
        if (
            query is not None
            and _DOCUMENT_SUBMISSION_DEADLINE_QUERY_RE.search(query)
        ):
            date_supported = _supports_requested_deadline(text, query)
        if not date_supported:
            facets.remove("date")
    return frozenset(facets)


def _required_temporal_facets(query: str) -> frozenset[str]:
    facets: set[str] = set()
    if _TEMPORAL_DATE_QUERY_RE.search(query):
        facets.add("date")
    if _TEMPORAL_TIME_QUERY_RE.search(query):
        facets.add("time")
    return frozenset(facets)


def _required_query_facets(query: str) -> frozenset[str]:
    """Return only facets the user explicitly asks the answer to cover."""

    facets: set[str] = set()
    if _TEMPORAL_DATE_QUERY_RE.search(query):
        facets.add("date")
    if _AMOUNT_QUERY_RE.search(query):
        facets.add("amount")
    if _RATE_QUERY_RE.search(query):
        facets.add("rate")
    if _ELIGIBILITY_QUERY_RE.search(query):
        facets.add("eligibility")
    if _METHOD_QUERY_RE.search(query):
        facets.add("method")
    if _DESTINATION_QUERY_RE.search(query):
        facets.add("destination")
    if _BUDGET_AMOUNT_QUERY_RE.search(query):
        facets.add("budget_amount")
    if _BUDGET_SHARE_QUERY_RE.search(query):
        facets.add("budget_share")
    return frozenset(facets)


def _required_method_actions(query: str) -> frozenset[str]:
    normalized = normalize_text(query)
    return frozenset(
        action
        for action in _METHOD_ACTIONS
        if (
            re.search(r"등록(?!금)", normalized)
            if action == "등록"
            else action in normalized
        )
    )


def _supports_requested_method(text: str, query: str) -> bool:
    """Require an answer-bearing method unit for the action in the query.

    Long administrative chunks often contain an unrelated ``확인 방법`` or end
    with a section heading such as ``신청방법 및 지급``.  Neither is enough to
    answer a request for the actual application path.  A qualifying unit must
    name the requested action and include an operational detail such as the
    channel, system, destination, or concrete step.
    """

    requested_actions = _required_method_actions(query)
    if not requested_actions:
        return bool(
            _METHOD_RELATION_RE.search(text)
            or len(_procedure_cues(text)) >= 2
        )
    # Do not concatenate adjacent lines here.  Separate table/outline rows can
    # otherwise lend the action word from one row to an unrelated UI path in
    # the next row (for example, scholarship application vs tuition lookup).
    units = [
        normalize_text(raw_line)
        for raw_line in str(text or "").splitlines()
        if normalize_text(raw_line)
    ]
    if not units:
        units = [normalize_text(text)]
    for unit in units:
        normalized = normalize_text(unit)
        if not any(action in normalized for action in requested_actions):
            continue
        if not _METHOD_DETAIL_RE.search(unit):
            continue
        if _METHOD_RELATION_RE.search(unit) or len(_procedure_cues(unit)) >= 2:
            return True
    return False


def _budget_breakdown_facets(text: str) -> frozenset[str]:
    """Recognize an answer-bearing budget table, not a budget disclaimer."""

    raw_text = str(text or "")
    compact = re.sub(r"\s+", "", raw_text)
    comma_numbers = re.findall(
        r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)", raw_text
    )
    decimal_shares = re.findall(
        r"(?<!\d)\d{1,2}\.\d{1,2}(?!\d)", raw_text
    )
    area_count = sum(
        label in compact
        for label in ("교육영역", "연구영역", "학생지도영역")
    )
    facets: set[str] = set()
    if "예산액" in compact and "합계" in compact and len(comma_numbers) >= 2:
        facets.add("budget_amount")
    if "구성비" in compact and area_count >= 2 and len(decimal_shares) >= 2:
        facets.add("budget_share")
    return frozenset(facets)


def _budget_breakdown_claim_facets(text: str) -> frozenset[str]:
    """Return the budget-table facets explicitly asserted by one claim."""

    compact = re.sub(r"\s+", "", str(text or ""))
    facets: set[str] = set()
    if re.search(r"예산(?:액|규모)", compact):
        facets.add("budget_amount")
    if "구성비" in compact or "비중" in compact:
        facets.add("budget_share")
    return frozenset(facets)


def _supported_query_facets(
    text: str,
    *,
    query: str | None = None,
) -> frozenset[str]:
    """Find relation-scoped answer facets in one context chunk.

    A bare date is intentionally insufficient: dates describing an interest
    accrual window previously masked a missing application-period chunk.
    """

    facets: set[str] = set()
    date_supported = bool(
        _TEMPORAL_DATE_RE.search(text) and _DATE_RELATION_RE.search(text)
    )
    if (
        query is not None
        and _DOCUMENT_SUBMISSION_DEADLINE_QUERY_RE.search(query)
    ):
        date_supported = _supports_requested_deadline(text, query)
    if date_supported:
        facets.add("date")

    critical_values = extract_critical_values(text)
    has_currency_amount = any(
        value.startswith(("amount_krw:", "amount_krw_raw:"))
        for value in critical_values
    )
    has_percent_amount = any(
        re.search(r"\d{1,3}(?:\.\d+)?\s*%", unit)
        and re.search(
            r"금액|지원액|지원금|한도|납부액|등록금|원금|이자",
            unit,
        )
        for unit in str(text or "").splitlines()
    )
    has_amount = has_currency_amount or has_percent_amount
    if has_amount and _AMOUNT_RELATION_RE.search(text):
        facets.add("amount")
    if (
        any(value.startswith("percent:") for value in critical_values)
        and _RATE_RELATION_RE.search(text)
    ):
        facets.add("rate")
    if _ELIGIBILITY_RELATION_RE.search(text):
        facets.add("eligibility")
    if _DESTINATION_RELATION_RE.search(text):
        facets.add("destination")
    method_supported = (
        _supports_requested_method(text, query)
        if query is not None and _METHOD_QUERY_RE.search(query)
        else bool(
            _METHOD_RELATION_RE.search(text)
            or len(_procedure_cues(text)) >= 2
        )
    )
    if method_supported:
        facets.add("method")
    facets.update(_budget_breakdown_facets(text))
    return frozenset(facets)


def _foreign_admission_eligibility_facets(text: str) -> frozenset[str]:
    """Return the three independent eligibility facts a general answer needs."""

    facets: set[str] = set()
    if _FOREIGN_ADMISSION_NATIONALITY_RE.search(text):
        facets.add("nationality")
    if _FOREIGN_ADMISSION_LANGUAGE_RE.search(text):
        facets.add("language")
    if _FOREIGN_ADMISSION_EDUCATION_RE.search(text):
        facets.add("education")
    return frozenset(facets)


def _intent_specific_supported_facets(
    text: str,
    *,
    exchange_selection: bool,
    foreign_admission_eligibility: bool,
    group_visa_application: bool,
    assistive_device_copay_support: bool,
    third_party_reporting: bool,
) -> frozenset[str]:
    facets: set[str] = set()
    if foreign_admission_eligibility:
        facets.update(_foreign_admission_eligibility_facets(text))
    if exchange_selection and _EXCHANGE_SELECTION_SCALE_RELATION_RE.search(text):
        facets.add("selection_scale")
    if exchange_selection and _EXCHANGE_SELECTION_SCHEDULE_RELATION_RE.search(text):
        facets.add("selection_schedule")
    if group_visa_application:
        if _GROUP_VISA_RESERVATION_RELATION_RE.search(text):
            facets.add("group_visa_reservation")
        if _GROUP_VISA_SCHEDULE_RELATION_RE.search(text):
            facets.add("group_visa_schedule")
        if _GROUP_VISA_FEE_RELATION_RE.search(text):
            facets.add("group_visa_fee")
    if assistive_device_copay_support:
        if _ASSISTIVE_COPAY_SUMMARY_RELATION_RE.search(text):
            facets.add("assistive_copay_summary")
        if _ASSISTIVE_COPAY_PROCEDURE_RELATION_RE.search(text):
            facets.add("assistive_copay_procedure")
        if _ASSISTIVE_COPAY_DOCUMENTS_RELATION_RE.search(text):
            facets.add("assistive_copay_documents")
    if third_party_reporting:
        if _THIRD_PARTY_REPORT_PERMISSION_RE.search(text):
            facets.add("third_party_permission")
        if _THIRD_PARTY_VICTIM_CONSENT_RE.search(text):
            facets.add("victim_consent")
        if _THIRD_PARTY_VICTIM_IDENTIFICATION_RE.search(text):
            facets.add("victim_identification")
    return frozenset(facets)


def _query_candidate_scope_conflicts(
    query: str,
    candidate: dict[str, Any],
) -> tuple[str, ...]:
    """Reject explicit year/semester contradictions, while allowing omissions."""

    query_values = extract_critical_values(query)
    body_values = extract_critical_values(result_source_text(candidate))
    title_values = extract_critical_values(_source_title_text(candidate))
    conflicts: list[str] = []
    for label, prefixes in (
        ("year", ("academic_year:", "calendar_year:")),
        ("semester", ("semester:",)),
    ):
        expected = {
            value
            for value in query_values
            if value.startswith(prefixes)
        }
        observed_body = {
            value for value in body_values if value.startswith(prefixes)
        }
        observed_title = {
            value for value in title_values if value.startswith(prefixes)
        }
        observed = observed_body or observed_title
        # Academic-year and calendar-year prefixes represent the same explicit
        # year for retrieval scoping, so compare their numeric payloads.
        expected_payloads = {value.split(":", 1)[1] for value in expected}
        observed_payloads = {value.split(":", 1)[1] for value in observed}
        title_payloads = {
            value.split(":", 1)[1] for value in observed_title
        }
        expected_academic_years = {
            int(value.split(":", 1)[1])
            for value in expected
            if value.startswith("academic_year:")
        }
        if label == "year" and expected_academic_years:
            allowed_calendar_years = {
                str(year)
                for expected_year in expected_academic_years
                for year in (expected_year, expected_year + 1)
            }
            if (
                expected_payloads & title_payloads
                and observed_payloads <= allowed_calendar_years
            ):
                continue
        if (
            expected_payloads
            and observed_payloads
            and expected_payloads.isdisjoint(observed_payloads)
        ):
            conflicts.append(label)
    return tuple(conflicts)


def _procedure_cues(text: str) -> frozenset[str]:
    compact = re.sub(r"\s+", "", text).casefold()
    return frozenset(cue for cue in _PROCEDURE_CUES if cue in compact)


def _decoded_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _neighbor_result(
    seed: dict[str, Any],
    row: sqlite3.Row,
) -> dict[str, Any]:
    stored = dict(row)
    text = str(stored.get("text") or "")
    locations = _decoded_json(stored.pop("locations_json", None), [])
    section_path = _decoded_json(
        stored.pop("section_path_json", None),
        None,
    )
    table_ids = _decoded_json(stored.pop("table_ids_json", None), [])
    block_ids = _decoded_json(stored.pop("block_ids_json", None), [])
    source_aliases = _decoded_json(
        stored.pop("source_aliases_json", None),
        [],
    )
    page_start = stored.pop("page_start", None)
    page_end = stored.pop("page_end", None)

    neighbor = dict(seed)
    neighbor.update(stored)
    neighbor.update(
        {
            "text": text,
            "preview": text[:preview_chars()],
            "locations": locations,
            "section_path": section_path,
            "table_ids": table_ids,
            "block_ids": block_ids,
            "source_aliases": source_aliases,
            "location": {
                "page_start": page_start,
                "page_end": page_end,
                "section_path": section_path,
                "table_ids": table_ids,
                "block_ids": block_ids,
            },
            "retrieval": {
                "bm25": None,
                "dense": None,
                "rrf": None,
                "reranker": None,
                "neighbor": {
                    "seed_chunk_id": seed.get("chunk_id"),
                    "distance": abs(
                        int(stored.get("chunk_index") or 0)
                        - int(seed.get("chunk_index") or 0)
                    ),
                },
                "final_rank": None,
            },
            "score": seed.get("score"),
        }
    )
    neighbor.pop("source_number", None)
    metadata = dict(seed.get("metadata") or {})
    metadata.update(
        {
            "corpus_revision": neighbor.get("corpus_revision"),
            "source_title": neighbor.get("source_title"),
            "source_url": neighbor.get("source_url"),
            "download_url": neighbor.get("download_url"),
            "source_host": neighbor.get("source_host"),
            "fetched_at": neighbor.get("fetched_at"),
            "published_at": neighbor.get("published_at"),
            "category": neighbor.get("category"),
            "include_reason": neighbor.get("include_reason"),
            "crawl_storage_path": neighbor.get("crawl_storage_path"),
            "source_aliases": source_aliases,
            "page_start": page_start,
            "page_end": page_end,
            "section_path": section_path,
            "table_ids": table_ids,
            "block_ids": block_ids,
        }
    )
    neighbor["metadata"] = metadata
    return neighbor


def _complete_with_adjacent_facets(
    index_path: Path,
    rows: list[dict[str, Any]],
    query: str,
    *,
    max_chunks_per_document: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Complete explicit answer facets from nearby same-document chunks.

    The first qualifying retrieved row is a sibling-search seed, not protected
    context.  A multi-facet marginal replacement may evict that seed when the
    other selected row preserves every facet already covered.
    """

    temporal_query = bool(TEMPORAL_QUERY_RE.search(query))
    procedure_query = bool(PROCEDURE_QUERY_RE.search(query))
    multipart_query = query.count("?") >= 2
    required_temporal = _required_temporal_facets(query)
    required_query_facets = _required_query_facets(query)
    foreign_admission_eligibility = (
        _is_foreign_undergraduate_admission_eligibility_query(query)
    )
    exchange_selection = _is_exchange_selection_query(query)
    group_visa_application = _is_group_visa_application_query(query)
    assistive_device_copay_support = (
        _is_assistive_device_copay_support_query(query)
    )
    third_party_reporting = _is_third_party_reporting_query(query)
    if foreign_admission_eligibility:
        required_query_facets = required_query_facets | frozenset(
            {"nationality", "language", "education"}
        )
    if exchange_selection:
        required_query_facets = required_query_facets | frozenset(
            {"selection_scale", "selection_schedule"}
        )
    if group_visa_application:
        required_query_facets = (
            required_query_facets - frozenset({"method"})
        ) | frozenset(
            {
                "group_visa_reservation",
                "group_visa_schedule",
                "group_visa_fee",
            }
        )
    if assistive_device_copay_support:
        required_query_facets = (
            required_query_facets - frozenset({"method"})
        ) | frozenset(
            {
                "assistive_copay_summary",
                "assistive_copay_procedure",
                "assistive_copay_documents",
            }
        )
    if third_party_reporting:
        required_query_facets = required_query_facets | frozenset(
            {
                "third_party_permission",
                "victim_consent",
                "victim_identification",
            }
        )
    explicit_multifacet = len(required_query_facets) >= 2
    single_numeric_facet = bool(
        len(required_query_facets) == 1
        and required_query_facets
        <= {"amount", "rate", "budget_amount", "budget_share"}
    )
    diagnostics: dict[str, Any] = {
        "mode": "facet_sibling_completion",
        "enabled": (
            temporal_query
            or procedure_query
            or multipart_query
            or explicit_multifacet
            or single_numeric_facet
        ),
        "query_facets": [
            facet
            for facet, enabled in (
                ("temporal", temporal_query),
                ("procedure", procedure_query),
                ("multipart", multipart_query),
            )
            if enabled
        ],
        "required_temporal_facets": sorted(required_temporal),
        "required_query_facets": sorted(required_query_facets),
        "explicit_multifacet": explicit_multifacet,
        "foreign_admission_eligibility": foreign_admission_eligibility,
        "exchange_selection": exchange_selection,
        "group_visa_application": group_visa_application,
        "assistive_device_copay_support": assistive_device_copay_support,
        "third_party_reporting": third_party_reporting,
        "considered_count": 0,
        "completed_count": 0,
        "skipped_noise_count": 0,
        "completions": [],
        "marginal_replacement_count": 0,
        "marginal_replacements": [],
        "scope_conflict_skip_count": 0,
        "scope_conflict_skips": [],
        # Compatibility fields for existing evaluation readers. Required-facet
        # replacements have dedicated diagnostics below and do not overload
        # the legacy adjacent-replacement counters.
        "replaced_count": 0,
        "replacements": [],
    }
    if not diagnostics["enabled"] or not rows or not index_path.is_file():
        return [dict(row) for row in rows], diagnostics

    configured_per_document_cap = max(1, int(max_chunks_per_document))
    # A few explicit administrative intents are intentionally spread across
    # three parts of one official document. Permit exactly one additional
    # same-document context only for those three-part intents; all other
    # queries retain the configured diversity cap.
    three_part_same_document_intent = bool(
        group_visa_application or assistive_device_copay_support
    )
    per_document_cap = (
        max(3, configured_per_document_cap)
        if three_part_same_document_intent
        else configured_per_document_cap
    )
    diagnostics["configured_per_document_cap"] = configured_per_document_cap
    diagnostics["effective_per_document_cap"] = per_document_cap
    diagnostics["per_document_cap_exception"] = (
        three_part_same_document_intent
    )
    query_terms = _query_anchor_terms(query)
    query_identifiers = _query_identifiers(query)

    def completion_anchor_strength(
        row: dict[str, Any],
    ) -> tuple[int, int, int] | None:
        strength = _strong_anchor_score(
            row,
            query_terms=query_terms,
            query_identifiers=query_identifiers,
        )
        if strength is not None:
            return strength
        supported = _intent_specific_supported_facets(
            f"{_result_scope_text(row)}\n{result_source_text(row)}",
            exchange_selection=exchange_selection,
            foreign_admission_eligibility=foreign_admission_eligibility,
            group_visa_application=group_visa_application,
            assistive_device_copay_support=(
                assistive_device_copay_support
            ),
            third_party_reporting=third_party_reporting,
        )
        if supported:
            return 0, 0, len(supported)
        return None

    completed = [dict(row) for row in rows]
    original_ids = {
        str(row.get("chunk_id") or "")
        for row in rows
        if str(row.get("chunk_id") or "")
    }
    inserted_ids: set[str] = set()

    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    try:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(chunks)"
            ).fetchall()
        }
        document_column = (
            "document_id" if "document_id" in columns else "doc_id"
        )

        # A document is considered at most once.  Prefer the document with the
        # strongest query anchors before lower-ranked partial matches.  The old
        # rank-only order could spend the completion budget on a generic
        # registration document before reaching the scholarship notice that
        # actually named the user's subject.
        document_first_position: dict[str, int] = {}
        document_strength: dict[str, tuple[int, int, int]] = {}
        for position, row in enumerate(rows):
            document_id = _document_id(row)
            if not document_id:
                continue
            document_first_position.setdefault(document_id, position)
            strength = completion_anchor_strength(row)
            if strength is not None:
                document_strength[document_id] = max(
                    strength,
                    document_strength.get(document_id, strength),
                )
        document_order = sorted(
            document_first_position,
            key=lambda document_id: (
                tuple(-value for value in document_strength.get(
                    document_id, (0, 0, 0)
                )),
                document_first_position[document_id],
            ),
        )

        for document_id in document_order:
            if diagnostics["completed_count"] >= _MAX_FACET_SIBLING_COMPLETIONS:
                break
            document_rows = [
                (position, row)
                for position, row in enumerate(completed)
                if _document_id(row) == document_id
            ]
            if not document_rows:
                continue

            strong_anchors: list[tuple[int, dict[str, Any]]] = []
            for position, row in document_rows:
                strength = completion_anchor_strength(row)
                if strength is not None:
                    strong_anchors.append((position, row))
            if not strong_anchors:
                continue
            # Retrieval rank remains authoritative among qualifying chunks.
            # Re-scoring here previously selected a later card-payment chunk
            # over the earlier bank-list seed for a two-facet question.  The
            # seed locates siblings; the marginal path may still replace it.
            if _ACADEMIC_DEPARTMENT_QUERY_RE.search(query):
                # Eligibility tables are commonly near the front of a long
                # notice, while duplicate amount/rate summaries occur near
                # the end.  For an explicit "which department" question,
                # anchor the sibling window at the earliest retrieved chunk
                # in the document so the nearby eligibility row is reachable.
                _, anchor = min(
                    strong_anchors,
                    key=lambda item: (
                        int(item[1].get("chunk_index") or 0),
                        item[0],
                    ),
                )
            else:
                _, anchor = min(strong_anchors, key=lambda item: item[0])
            anchor_id = str(anchor.get("chunk_id") or "")
            chunk_index = anchor.get("chunk_index")
            if not anchor_id or not isinstance(chunk_index, int):
                continue
            anchor_scope_conflicts = _query_candidate_scope_conflicts(
                query,
                anchor,
            )
            if anchor_scope_conflicts:
                diagnostics["scope_conflict_skips"].append(
                    {
                        "chunk_id": anchor_id,
                        "conflicts": list(anchor_scope_conflicts),
                        "reason": "anchor_query_scope_conflict",
                    }
                )
                diagnostics["scope_conflict_skip_count"] += 1
                continue

            current_text = "\n".join(
                result_source_text(row) for _, row in document_rows
            )
            current_temporal = _answer_temporal_facets(
                current_text,
                query=query,
            )
            current_procedure = _procedure_cues(current_text)
            current_required_by_position = {
                position: (
                    (
                        _supported_query_facets(
                            result_source_text(row),
                            query=query,
                        )
                        | _intent_specific_supported_facets(
                            (
                                f"{_result_scope_text(row)}\n"
                                f"{result_source_text(row)}"
                            ),
                            exchange_selection=exchange_selection,
                            foreign_admission_eligibility=(
                                foreign_admission_eligibility
                            ),
                            group_visa_application=group_visa_application,
                            assistive_device_copay_support=(
                                assistive_device_copay_support
                            ),
                            third_party_reporting=third_party_reporting,
                        )
                    )
                    & required_query_facets
                )
                for position, row in document_rows
            }
            current_required = frozenset().union(
                *current_required_by_position.values()
            )
            missing_required = required_query_facets - current_required
            empty_distinctive_seed = bool(
                (explicit_multifacet or single_numeric_facet)
                and not current_required
                and (
                    single_numeric_facet
                    or (
                        exchange_selection
                        and "교환" in _source_title_text(anchor)
                        and "선발" in _source_title_text(anchor)
                    )
                    or any(
                        len(term) >= 4
                        and _term_in_text(term, _source_title_text(anchor))
                        for term in query_terms
                    )
                )
                and any(
                    _term_in_text(term, result_source_text(anchor))
                    for term in query_terms
                )
            )
            distant_department_seed = bool(
                explicit_multifacet
                and current_required
                and "eligibility" in missing_required
                and _ACADEMIC_DEPARTMENT_QUERY_RE.search(query)
                and any(
                    len(term) >= 4
                    and _term_in_text(term, _source_title_text(anchor))
                    for term in query_terms
                )
            )
            distant_submission_deadline_seed = bool(
                "date" in missing_required
                and _DOCUMENT_SUBMISSION_DEADLINE_QUERY_RE.search(query)
                and any(
                    len(term) >= 4
                    and _term_in_text(term, _source_title_text(anchor))
                    for term in query_terms
                )
            )
            # Sibling completion is intentionally document-local.  A method
            # found in an unrelated result must not mask a missing method in
            # the primary notice, and once the strongest relevant document is
            # already self-contained there is no reason to expand distractors.
            if (
                (explicit_multifacet or single_numeric_facet)
                and not missing_required
            ):
                break
            need_marginal_completion = bool(
                (explicit_multifacet or single_numeric_facet)
                and missing_required
                and (current_required or empty_distinctive_seed)
            )
            faq_document = bool(
                re.search(
                    r"FAQ|자주\s*하는\s*질문|질문\s*[/·-]?\s*답변",
                    _source_title_text(anchor),
                    re.IGNORECASE,
                )
            )
            missing_temporal = required_temporal - current_temporal
            # The explicit multi-facet path uses document-local coverage.
            # Running the older temporal/procedure heuristics as well can add
            # a plausible-looking but question-irrelevant sibling.
            need_temporal = (
                temporal_query
                and not explicit_multifacet
                and bool(missing_temporal)
            )
            need_procedure = procedure_query and not explicit_multifacet
            need_multipart_faq = multipart_query and faq_document
            if (
                not need_temporal
                and not need_procedure
                and not need_multipart_faq
                and not need_marginal_completion
            ):
                continue

            candidate_radius = (
                24
                if distant_submission_deadline_seed
                or (single_numeric_facet and empty_distinctive_seed)
                or group_visa_application
                else (
                    8
                    if (
                        empty_distinctive_seed
                        or distant_department_seed
                        or exchange_selection
                        or assistive_device_copay_support
                    )
                    else (
                        4
                        if foreign_admission_eligibility
                        or third_party_reporting
                        else 2
                    )
                )
            )
            candidates = connection.execute(
                f"""
                SELECT *
                FROM chunks
                WHERE {document_column} = ?
                  AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
                """,
                (
                    document_id,
                    chunk_index - candidate_radius,
                    chunk_index + candidate_radius,
                ),
            ).fetchall()
            scored: list[
                tuple[
                    tuple[int, int, int, int, float, int, int],
                    dict[str, Any],
                    str,
                    int,
                ]
            ] = []
            marginal_scored: list[
                tuple[
                    tuple[int, int, int, int, int, float, float, int, int],
                    dict[str, Any],
                    int,
                    int | None,
                    frozenset[str],
                    frozenset[str],
                ]
            ] = []
            diagnostics["considered_count"] += 1
            for candidate in candidates:
                neighbor = _neighbor_result(anchor, candidate)
                neighbor_id = str(neighbor.get("chunk_id") or "")
                if (
                    not neighbor_id
                    or neighbor_id in original_ids
                    or neighbor_id in inserted_ids
                ):
                    continue
                neighbor_text = result_source_text(neighbor)
                if is_low_quality_candidate(
                    neighbor_text,
                    minimum_chars=30,
                ):
                    diagnostics["skipped_noise_count"] += 1
                    continue

                scope_conflicts = (
                    _query_candidate_scope_conflicts(query, neighbor)
                    if need_marginal_completion
                    else ()
                )
                if scope_conflicts:
                    diagnostics["scope_conflict_skips"].append(
                        {
                            "chunk_id": neighbor_id,
                            "conflicts": list(scope_conflicts),
                            "reason": "query_scope_conflict",
                        }
                    )
                    diagnostics["scope_conflict_skip_count"] += 1
                    continue

                query_overlap = overlap_score(
                    set(query_terms),
                    neighbor_text,
                )
                if need_marginal_completion:
                    candidate_required = (
                        (
                            _supported_query_facets(
                                neighbor_text,
                                query=query,
                            )
                            | _intent_specific_supported_facets(
                                (
                                    f"{_result_scope_text(neighbor)}\n"
                                    f"{neighbor_text}"
                                ),
                                exchange_selection=exchange_selection,
                                foreign_admission_eligibility=(
                                    foreign_admission_eligibility
                                ),
                                group_visa_application=group_visa_application,
                                assistive_device_copay_support=(
                                    assistive_device_copay_support
                                ),
                                third_party_reporting=third_party_reporting,
                            )
                        )
                        & required_query_facets
                    )
                    added_required = candidate_required & missing_required
                    destination_bridge = "destination" in added_required
                    intent_bridge = bool(
                        three_part_same_document_intent and added_required
                    )
                    if added_required and (
                        query_overlap > 0
                        or destination_bridge
                        or intent_bridge
                    ):
                        numeric_scope_matches = (
                            sum(
                                term in query and term in neighbor_text
                                for term in _NUMERIC_QUERY_SCOPE_TERMS
                            )
                            if single_numeric_facet
                            else 0
                        )
                        removal_options: list[
                            tuple[int | None, frozenset[str], float]
                        ] = []
                        if len(document_rows) < per_document_cap:
                            removal_options.append((None, frozenset(), 0.0))
                        else:
                            for remove_position, remove_row in document_rows:
                                remaining_required = frozenset().union(
                                    *(
                                        facets
                                        for position, facets
                                        in current_required_by_position.items()
                                        if position != remove_position
                                    )
                                )
                                after_required = (
                                    remaining_required | candidate_required
                                )
                                if not current_required <= after_required:
                                    continue
                                removed_marginal = (
                                    current_required_by_position[remove_position]
                                    - remaining_required
                                )
                                removal_options.append(
                                    (
                                        remove_position,
                                        removed_marginal,
                                        overlap_score(
                                            set(query_terms),
                                            result_source_text(remove_row),
                                        ),
                                    )
                                )
                        distance = abs(
                            int(neighbor.get("chunk_index") or 0)
                            - chunk_index
                        )
                        for (
                            remove_position,
                            removed_marginal,
                            removed_overlap,
                        ) in removal_options:
                            marginal_score = (
                                len(added_required),
                                len(candidate_required),
                                numeric_scope_matches,
                                -len(removed_marginal),
                                (
                                    remove_position
                                    if remove_position is not None
                                    else len(completed)
                                ),
                                -removed_overlap,
                                query_overlap,
                                -distance,
                                -int(neighbor.get("chunk_index") or 0),
                            )
                            marginal_scored.append(
                                (
                                    marginal_score,
                                    neighbor,
                                    distance,
                                    remove_position,
                                    added_required,
                                    removed_marginal,
                                )
                            )

                temporal_added = (
                    _answer_temporal_facets(
                        neighbor_text,
                        query=query,
                    )
                    & missing_temporal
                    if need_temporal
                    else frozenset()
                )
                procedure_added = (
                    _procedure_cues(neighbor_text) - current_procedure
                    if need_procedure
                    else frozenset()
                )
                multipart_term_matches = (
                    sum(
                        _term_in_text(term, neighbor_text)
                        for term in query_terms
                    )
                    if need_multipart_faq
                    else 0
                )
                if temporal_added and query_overlap <= 0:
                    temporal_added = frozenset()
                multipart_added = multipart_term_matches >= 2
                if (
                    not temporal_added
                    and len(procedure_added) < 2
                    and not multipart_added
                ):
                    continue

                kind = (
                    "temporal_sibling_completion"
                    if temporal_added
                    else (
                        "procedure_sibling_completion"
                        if len(procedure_added) >= 2
                        else "multipart_faq_sibling_completion"
                    )
                )
                distance = abs(
                    int(neighbor.get("chunk_index") or 0) - chunk_index
                )
                score = (
                    int(bool(temporal_added)),
                    len(temporal_added),
                    len(procedure_added),
                    int(multipart_added),
                    query_overlap,
                    -distance,
                    -int(neighbor.get("chunk_index") or 0),
                )
                scored.append((score, neighbor, kind, distance))

            if marginal_scored:
                marginal_scored.sort(key=lambda item: item[0], reverse=True)
                (
                    _,
                    neighbor,
                    distance,
                    remove_position,
                    added_required,
                    removed_marginal,
                ) = marginal_scored[0]
                neighbor_id = str(neighbor.get("chunk_id") or "")
                removed_chunk_id: str | None = None
                if remove_position is None:
                    anchor_position = next(
                        (
                            position
                            for position, row in enumerate(completed)
                            if str(row.get("chunk_id") or "") == anchor_id
                        ),
                        None,
                    )
                    if anchor_position is None:
                        continue
                    completed.insert(anchor_position + 1, neighbor)
                else:
                    removed = completed[remove_position]
                    removed_chunk_id = (
                        str(removed.get("chunk_id") or "") or None
                    )
                    # Replacement in the same slot prevents unrelated context
                    # ranks from shifting when the per-document cap is full.
                    completed[remove_position] = neighbor

                retrieval = dict(neighbor.get("retrieval") or {})
                retrieval["context_expansion"] = {
                    "kind": "required_facet_sibling_completion",
                    "seed_chunk_id": anchor_id,
                    "seed_replaced": removed_chunk_id == anchor_id,
                    "distance": distance,
                    "removed_chunk_id": removed_chunk_id,
                    "reason": "missing_query_facet",
                    "added_facets": sorted(added_required),
                }
                neighbor["retrieval"] = retrieval
                if remove_position is not None:
                    completed[remove_position] = neighbor
                inserted_ids.add(neighbor_id)
                action = {
                    "kind": "required_facet_sibling_completion",
                    "seed_chunk_id": anchor_id,
                    "seed_replaced": removed_chunk_id == anchor_id,
                    "chunk_id": neighbor_id,
                    "distance": distance,
                    "removed_chunk_id": removed_chunk_id,
                    "reason": "missing_query_facet",
                    "required_facets": sorted(required_query_facets),
                    "missing_facets_before": sorted(missing_required),
                    "added_facets": sorted(added_required),
                    "removed_marginal_facets": sorted(removed_marginal),
                }
                diagnostics["completions"].append(action)
                diagnostics["completed_count"] += 1
                if removed_chunk_id is not None:
                    diagnostics["marginal_replacements"].append(action)
                    diagnostics["marginal_replacement_count"] += 1
                # Do not run the legacy heuristic again for the same document;
                # one explicit required-facet action is the stability boundary.
                # ``candidate_required`` belongs to the final candidate
                # visited by the scan, not necessarily the candidate selected
                # above.  ``added_required`` is carried with the selected
                # marginal row and is therefore the authoritative coverage
                # delta for deciding whether document expansion is complete.
                if required_query_facets <= (
                    current_required | added_required
                ):
                    break
                if (
                    three_part_same_document_intent
                    and diagnostics["completed_count"]
                    < _MAX_FACET_SIBLING_COMPLETIONS
                ):
                    # Revisit this one strongly anchored document once. The
                    # first pass may add one missing answer part; the second
                    # fills the other while keeping three fact-bearing chunks
                    # under the narrow effective cap above.
                    document_order.append(document_id)
                continue

            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            _, neighbor, kind, distance = scored[0]
            neighbor_id = str(neighbor.get("chunk_id") or "")

            # Keep the anchor. If the document already uses its cap, evict the
            # least useful non-anchor sibling (lowest query overlap, then latest
            # rank) before inserting the completed context directly after it.
            removed_chunk_id: str | None = None
            document_rows = [
                (position, row)
                for position, row in enumerate(completed)
                if _document_id(row) == document_id
            ]
            if len(document_rows) >= per_document_cap:
                removable = [
                    (
                        overlap_score(
                            set(query_terms),
                            result_source_text(row),
                        ),
                        -position,
                        position,
                        row,
                    )
                    for position, row in document_rows
                    if str(row.get("chunk_id") or "") != anchor_id
                ]
                if not removable:
                    continue
                removable.sort(key=lambda item: (item[0], item[1]))
                _, _, remove_position, removed = removable[0]
                removed_chunk_id = str(removed.get("chunk_id") or "") or None
                completed.pop(remove_position)

            anchor_position = next(
                (
                    position
                    for position, row in enumerate(completed)
                    if str(row.get("chunk_id") or "") == anchor_id
                ),
                None,
            )
            if anchor_position is None:
                continue
            retrieval = dict(neighbor.get("retrieval") or {})
            retrieval["context_expansion"] = {
                "kind": kind,
                "anchor_chunk_id": anchor_id,
                "distance": distance,
                "removed_chunk_id": removed_chunk_id,
            }
            neighbor["retrieval"] = retrieval
            completed.insert(anchor_position + 1, neighbor)
            inserted_ids.add(neighbor_id)
            diagnostics["completions"].append(
                {
                    "kind": kind,
                    "anchor_chunk_id": anchor_id,
                    "chunk_id": neighbor_id,
                    "distance": distance,
                    "removed_chunk_id": removed_chunk_id,
                }
            )
            diagnostics["completed_count"] += 1
    finally:
        connection.close()

    return completed, diagnostics


def replace_with_adjacent_temporal_contexts(
    index_path: Path,
    rows: list[dict[str, Any]],
    query: str,
    *,
    facet_completion: bool = False,
    max_chunks_per_document: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Complete tuned facet context, or preserve legacy temporal replacement.

    ``facet_completion=False`` is the frozen control behavior. The tuned path
    uses a strongly relevant retrieved row as a sibling-search seed and finds
    at most one substantive sibling per document. For an explicit multi-facet
    query it may replace that seed or another zero-marginal same-document chunk
    in place, while respecting the configured cap.
    """

    if facet_completion:
        return _complete_with_adjacent_facets(
            index_path,
            rows,
            query,
            max_chunks_per_document=max_chunks_per_document,
        )

    diagnostics: dict[str, Any] = {
        "mode": "adjacent_table_replacement",
        "enabled": bool(TEMPORAL_QUERY_RE.search(query)),
        "considered_count": 0,
        "replaced_count": 0,
        "replacements": [],
    }
    if not diagnostics["enabled"] or not rows or not index_path.is_file():
        return list(rows), diagnostics

    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
    }
    document_column = (
        "document_id" if "document_id" in columns else "doc_id"
    )
    original_ids = {
        str(row.get("chunk_id") or "")
        for row in rows
    }
    replacement_ids: set[str] = set()
    query_terms = set(tokenize(query, include_ngrams=False))
    expanded: list[dict[str, Any]] = []

    try:
        for seed in rows:
            seed_id = str(seed.get("chunk_id") or "")
            document_id = str(
                seed.get("document_id")
                or seed.get("doc_id")
                or ""
            )
            chunk_index = seed.get("chunk_index")
            if not document_id or not isinstance(chunk_index, int):
                expanded.append(seed)
                continue
            candidates = connection.execute(
                f"""
                SELECT *
                FROM chunks
                WHERE {document_column} = ?
                  AND chunk_index IN (?, ?)
                ORDER BY chunk_index
                """,
                (document_id, chunk_index - 1, chunk_index + 1),
            ).fetchall()
            scored_neighbors: list[
                tuple[float, int, dict[str, Any]]
            ] = []
            diagnostics["considered_count"] += 1
            for candidate in candidates:
                neighbor = _neighbor_result(seed, candidate)
                neighbor_id = str(neighbor.get("chunk_id") or "")
                if (
                    not neighbor_id
                    or neighbor_id in original_ids
                    or neighbor_id in replacement_ids
                    or not _is_table_context(neighbor)
                ):
                    continue
                relevance = overlap_score(
                    query_terms,
                    result_source_text(neighbor),
                )
                if relevance <= 0:
                    continue
                distance = abs(
                    int(neighbor.get("chunk_index") or 0) - chunk_index
                )
                scored_neighbors.append(
                    (relevance, -distance, neighbor)
                )
            if not scored_neighbors:
                expanded.append(seed)
                continue

            scored_neighbors.sort(
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
            neighbor_relevance, _, neighbor = scored_neighbors[0]
            seed_relevance = overlap_score(
                query_terms,
                result_source_text(seed),
            )
            if neighbor_relevance <= seed_relevance:
                expanded.append(seed)
                continue

            neighbor_id = str(neighbor.get("chunk_id") or "")
            retrieval = dict(seed.get("retrieval") or {})
            retrieval["context_expansion"] = {
                "kind": "adjacent_table_replacement",
                "anchor_chunk_id": seed_id,
                "anchor_relevance": round(seed_relevance, 4),
                "replacement_relevance": round(
                    neighbor_relevance,
                    4,
                ),
            }
            neighbor["retrieval"] = retrieval
            expanded.append(neighbor)
            replacement_ids.add(neighbor_id)
            diagnostics["replacements"].append(
                {
                    "anchor_chunk_id": seed_id,
                    "chunk_id": neighbor_id,
                }
            )
    finally:
        connection.close()

    diagnostics["replaced_count"] = len(
        diagnostics["replacements"]
    )
    return expanded, diagnostics


def _select_budget_breakdown_claims(
    question: str,
    results: list[dict[str, Any]],
) -> list[str]:
    """Render the requested year from a budget amount/share table."""

    required = _required_query_facets(question)
    if not {"budget_amount", "budget_share"} <= required:
        return []
    year_match = ACADEMIC_YEAR_RE.search(question)
    if year_match is None:
        return []
    year = year_match.group(1)
    year_re = re.compile(rf"{re.escape(year)}\s*학년도")
    requested_labels = ("교육영역", "연구영역", "학생지도영역", "합계")

    for result in results:
        text = result_source_text(result)
        if not {
            "budget_amount",
            "budget_share",
        } <= _budget_breakdown_facets(text):
            continue
        lines = [line for line in text.splitlines() if "\t" in line]
        header_fields: list[str] | None = None
        amount_index: int | None = None
        share_index: int | None = None
        for line in lines:
            fields = [normalize_text(field) for field in line.split("\t")]
            for index, field in enumerate(fields):
                if (
                    year_re.search(field)
                    and "예산액" in field
                    and "최종예산액" not in field
                ):
                    amount_index = index
            if amount_index is not None:
                share_index = next(
                    (
                        index
                        for index, field in enumerate(fields)
                        if index > amount_index and "구성비" in field
                    ),
                    None,
                )
            if amount_index is not None and share_index is not None:
                header_fields = fields
                break
        if header_fields is None or amount_index is None or share_index is None:
            continue

        parsed: dict[str, tuple[str, str]] = {}
        for line in lines:
            fields = [normalize_text(field) for field in line.split("\t")]
            if len(fields) <= max(amount_index, share_index):
                continue
            label = re.sub(r"[\s.]", "", fields[0])
            if label not in requested_labels:
                continue
            amount = fields[amount_index]
            share = fields[share_index]
            if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", amount):
                continue
            if not re.fullmatch(r"(?:100|\d{1,2}(?:\.\d{1,2})?)", share):
                continue
            parsed[label] = (amount, share)

        if not all(label in parsed for label in requested_labels):
            continue
        total_amount, total_share = parsed["합계"]
        area_parts = [
            f"{label} {parsed[label][0]}({parsed[label][1]}%)"
            for label in requested_labels[:3]
        ]
        return [
            (
                f"{year}학년도 전체 예산액은 {total_amount}이며, "
                f"구성비 합계는 {total_share}%입니다."
            ),
            f"영역별 예산액과 구성비는 {', '.join(area_parts)}입니다.",
        ]
    return []


def select_answer_claims(question: str, results: list[dict[str, Any]]) -> list[str]:
    budget_claims = _select_budget_breakdown_claims(question, results)
    if budget_claims:
        return budget_claims[:MAX_CLAIMS]

    question_terms = set(tokenize(question, include_ngrams=False))
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()

    for rank, result in enumerate(results, start=1):
        rank_bonus = 1 / (rank + 2)
        for sentence in split_candidate_sentences(result_source_text(result)):
            key = sentence[:80]
            if key in seen:
                continue
            seen.add(key)
            score = overlap_score(question_terms, sentence) + rank_bonus
            scored.append((score, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    claims = [sentence for score, sentence in scored if score > 0][:MAX_CLAIMS]

    if claims:
        return claims

    fallback = [
        normalize_text(result_source_text(result))[:240].strip()
        for result in results[: min(2, len(results))]
    ]
    return [item for item in fallback if item]


def extractive_fallback_answer(
    question: str,
    contexts: Any,
) -> str:
    results = [
        dict(item)
        for item in contexts
        if isinstance(item, dict)
    ]
    institutions = {
        str(item.get("institution") or "").strip()
        for item in results
        if str(item.get("institution") or "").strip()
    }
    institution = next(iter(institutions)) if len(institutions) == 1 else None
    answer_query = normalize_retrieval_query(question, institution)
    return "\n".join(select_answer_claims(answer_query, results))


def strip_untrusted_citation_markers(value: str) -> str:
    """Citation numbers are assigned only after server-side attribution."""

    return re.sub(r"\s*\[(?:\d+\s*,?\s*)+\]", "", value).strip()


def build_gemini_prompt(question: str, results: list[dict[str, Any]]) -> str:
    """Build the legacy Gemini prompt from the active provider-neutral prompt."""

    return build_generation_prompt(question, results)


def extract_gemini_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def generate_answer_with_gemini_model(
    question: str,
    results: list[dict[str, Any]],
    model: str,
) -> str:
    key = gemini_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {
            "parts": [
                {
                    "text": GENERATION_SYSTEM_INSTRUCTION
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": build_gemini_prompt(question, results)}]}],
        "generationConfig": {
            "temperature": get_env_float("GEMINI_TEMPERATURE", 0.2),
            "maxOutputTokens": get_env_int("GEMINI_MAX_OUTPUT_TOKENS", 900),
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=get_env_int("GEMINI_TIMEOUT_SECONDS", 30)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail[:500]}") from exc

    text = extract_gemini_text(payload)
    if not text:
        raise RuntimeError("Gemini API returned no text")
    return text


def generate_answer_with_gemini(question: str, results: list[dict[str, Any]]) -> tuple[str, str]:
    errors: list[str] = []
    for model in gemini_model_candidates():
        try:
            return generate_answer_with_gemini_model(question, results, model), model
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:240]}")
            continue
    raise RuntimeError("All Gemini models failed. " + " | ".join(errors))


ACADEMIC_YEAR_RE = re.compile(r"(?<!\d)(\d{4})\s*학년도")
ACADEMIC_YEAR_DO_RE = re.compile(r"(?<!\d)(\d{4})\s*년도")
CALENDAR_YEAR_RE = re.compile(
    r"(?<!\d)(\d{4})\s*년(?!도|\s*\d{1,2}\s*월)"
)
SEMESTER_RE = re.compile(r"(?<!\d)([12])\s*학기")
ROUND_RE = re.compile(r"(?<!\d)(\d+)\s*차(?!원)")
COORDINATED_ROUND_RE = re.compile(
    r"(?<!\d)((?:\d+\s*(?:[·ㆍ‧・]|[,，])\s*)+)(\d+)\s*차(?!원)"
)
FULL_DATE_RE = re.compile(
    r"(?<!\d)[‘’']?(\d{2,4})\s*(?:년|[./-])\s*"
    r"(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})\s*일?"
)
KOREAN_MONTH_DAY_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일"
)
DOTTED_MONTH_DAY_RE = re.compile(
    r"(?<![\d.])(\d{1,2})\s*\.\s*(\d{1,2})\s*\."
    r"(?=\s*(?:\(|[~∼～]|$|,))"
)
SLASH_MONTH_DAY_RE = re.compile(
    r"(?<![\d/])(\d{1,2})\s*/\s*(\d{1,2})(?!\s*/\s*\d)"
)
ABBREVIATED_SAME_MONTH_RANGE_RE = re.compile(
    r"(?<!\d)(\d{2,4})\s*[./-]\s*(\d{1,2})\s*[./-]\s*"
    r"(\d{1,2})\s*\.?\s*(?:\([^)]{1,4}\))?\s*[~∼～]\s*"
    r"(\d{1,2})(?!\s*[./-]\s*\d)\s*(?:일)?\s*"
    r"(?:\([^)]{1,4}\))?"
)
COLON_TIME_RE = re.compile(
    r"(?<!\d)([01]?\d|2[0-4]):([0-5]\d)(?!\d)"
)
KOREAN_TIME_RE = re.compile(
    r"(?:(오전|오후)\s*)?(\d{1,2})\s*시"
    r"(?!\s*(?:간|점))"
    r"(?:\s*(\d{1,2})\s*분)?"
)
MONEY_EXPR_RE = re.compile(
    r"(?<![\d,조억만천백])"
    r"([+\-−△▲]?\s*"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:조|억|만|천|백)\s*\d[\d,]*(?:\.\d+)?)*"
    r"\s*(?:(?:천|백)\s*만|조|억|만|천|백)?)"
    r"\s*원(?!칙)"
)
MONEY_PART_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*((?:조|억|만|천|백)*)"
)
PERCENT_RE = re.compile(
    r"(?<!\d)([+\-−△▲]?)\s*(\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:%|％|퍼센트)(?!\s*(?:포인트|[pP]))"
)
QUANTITY_RE = re.compile(
    r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(부(?!터)|명|개월|학점|개(?!월)|회)"
)
SHARED_UNIT_QUANTITY_RANGE_RE = re.compile(
    r"(?<![\d.,])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:[~∼～\-–—]|(?:에서|부터))\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*"
    r"(부(?!터)|명|개월|학점|개(?!월)|회)"
)
PHONE_RE = re.compile(r"(?<!\d)(0\d{1,2})[-\s](\d{3,4})[-\s](\d{4})(?!\d)")
ABSTENTION_RE = re.compile(
    r"(?:제공된\s*)?(?:문서|검색\s*근거|검색\s*결과|근거)"
    r"(?:에서|에서는|만으로는|가|는)?\s*.{0,50}?"
    r"(?:확인할\s*수\s*없|확인되지\s*않|찾지\s*못|알\s*수\s*없|부족)"
)
PERMISSION_DENIED_RE = re.compile(
    r"(?:할|될|받을|낼|볼|둘|갈|올)?\s*수\s*없"
    r"|불가능|불가(?!피)|금지"
    r"|(?:허용|가능)되지(?:는)?\s*않|가능하지(?:는)?\s*않"
    r"|가능성(?:이|은|는)?\s*없"
)
PERMISSION_DOUBLE_NEGATION_RE = re.compile(
    r"(?:금지|불가능|불가)\s*(?:"
    r"(?:되|하)?지(?:는)?\s*않"
    r"|(?:된|한)\s*(?:것은|게)?\s*(?:아니|아닙)"
    r")"
)
PERMISSION_ALLOWED_RE = re.compile(
    r"(?:할|될|받을|낼|볼|둘|갈|올)?\s*수\s*있"
    r"|가능(?!성)|허용"
    r"|(?:신청|접수)(?:이|가)\s*원칙"
    r"|(?:신청|접수)\s*순서로\s*진행"
)
COMPARATOR_VALUE_RE = re.compile(
    r"(?<![\d,])((?:"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:조|억|만|천|백)\s*\d[\d,]*(?:\.\d+)?)*"
    r"\s*(?:(?:천|백)\s*만|조|억|만|천|백)?\s*원"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:퍼센트|%|％)"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:학점|개월|점|명|개|회|세|학기)"
    r"|\d[\d,]*(?:\.\d+)?"
    r"))\s*(이상|이하|초과|미만|이내|미달)"
)
MINMAX_VALUE_RE = re.compile(
    r"(최소|최대)\s*((?:"
    r"\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:조|억|만|천|백)\s*\d[\d,]*(?:\.\d+)?)*"
    r"\s*(?:(?:천|백)\s*만|조|억|만|천|백)?\s*원"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:퍼센트|%|％)"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:학점|개월|점|명|개|회|세|학기)"
    r"|\d[\d,]*(?:\.\d+)?"
    r"))"
)
COORDINATED_TERM_LIMIT_TRIGGER_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*학기\s*까지\s*지원\s*가능"
)
COORDINATED_TERM_LIMIT_ITEM_RE = re.compile(
    r"(?P<scope>[^\d,，()]{1,60}?)\s*"
    r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*학기"
)
COMPARATOR_CANONICAL = {
    "이내": "이하",
    "미달": "미만",
    "최소": "이상",
    "최대": "이하",
}
COMPARATOR_COMPLEMENT = {
    "이상": "미만",
    "이하": "초과",
    "초과": "이하",
    "미만": "이상",
}
COMPARATOR_NEGATION_RE = re.compile(
    r"^\s*(?:이|가|은|는)?\s*(?:아니(?!면|어도)|아님|아닙)"
)
DIRECTION_NEGATION_RE = re.compile(
    r"^\s*(?:(?:되|하)지(?:는)?\s*않|"
    r"(?:된|한)\s*(?:것은|게)?\s*(?:아니|아닙)|"
    r"(?:은|는)\s*없)"
)
RELATION_DIRECTION_PATTERNS = {
    "increase": re.compile(r"인상(?!적)|증가|상승|올랐|오름"),
    # ``로그인하여`` and ``확인하여`` contain the character sequence
    # ``인하`` but do not express a price/rate decrease.  A Korean-letter
    # boundary keeps those UI instructions out of the high-risk direction
    # relation guard while preserving ordinary forms such as ``금리 인하``.
    "decrease": re.compile(r"(?<![가-힣])인하|감소|하락|축소|내렸|내림"),
    "frozen": re.compile(r"동결|변동\s*없"),
}
RELATION_UNDETERMINED_RE = re.compile(
    r"여부|검토|논의|협의|고려|미정|"
    r"(?:확정|결정)되지(?:는)?\s*않"
)
RELATION_PLANNED_RE = re.compile(r"예정|계획")
RELATION_POSSIBLE_RE = re.compile(r"(?:될|할)\s*수(?:도)?\s*있|가능성")
TEMPORAL_ACTION_AVAILABILITY_RE = re.compile(
    r"(?:신청|접수|제출|납부|등록|수강|출력|확인|조회|열람|발급|다운로드)"
    r"\s*(?:할|될)\s*수(?:도)?\s*있"
)
RELATION_CANCELLED_RE = re.compile(r"취소|철회|백지화")
RELATION_NEGATED_ASSERTION_RE = re.compile(
    r"(?:된|한|일)?\s*(?:것은|게)?\s*(?:아니|아닙)"
    r"|(?:은|는)\s*없"
    r"|필요(?:가|는)?\s*없"
)
TEMPORAL_BOUNDARY_RE = re.compile(
    r"((?:"
    r"(?:\d{2,4}\s*년\s*)?\d{1,2}\s*월\s*\d{1,2}\s*일"
    r"|(?:\d{2,4}\s*[./-]\s*)?"
    r"\d{1,2}\s*[./-]\s*\d{1,2}\s*\.?)"
    r"(?:\s*\([월화수목금토일]\))?"
    r"(?:\s*(?:(?:오전|오후)\s*)?"
    r"(?:[01]?\d|2[0-4])(?::[0-5]\d|\s*시(?:\s*\d{1,2}\s*분)?))?"
    r"|(?:(?:오전|오후)\s*)?(?:[01]?\d|2[0-4]):[0-5]\d"
    r")\s*(부터|까지|이후|이전)"
)
TEMPORAL_BOUNDARY_OPERATORS = {
    "부터": "start_inclusive",
    "이후": "start_strict",
    "까지": "end_inclusive",
    "이전": "end_strict",
}
RELATION_CLAUSE_SPLIT_RE = re.compile(
    r"(?:(?<=[가-힣])[.!?](?=\s|[가-힣A-Z]|$)\s*|"
    r";\s*|※\s*|(?<!\d),(?!\d)\s*|"
    r"(?<=가능)하나\s*|반면(?:에)?\s*|"
    r"(?:이고|이며|되(?:었|였)?고|됐고|되며|하(?:였)?고|했고|하며|으며|"
    r"(?<!이)거나|지만)\s*)"
)
RELATION_TERM_STEMS = (
    "가능",
    "불가",
    "금지",
    "허용",
    "이상",
    "이하",
    "초과",
    "미만",
    "이내",
    "미달",
    "최소",
    "최대",
    "부터",
    "까지",
    "이전",
    "이후",
    "인상",
    "인하",
    "증가",
    "감소",
    "상승",
    "하락",
    "축소",
    "동결",
)
SEMANTIC_SCOPE_PREDICATE_NOISE = {
    "입니다",
    "이어야",
    "합니다",
    "됩니다",
    "되었습니다",
    "아닙니다",
}
PERMISSION_GENERIC_SCOPE_STEMS = ("신청", "접수", "진행")
KOREAN_TERM_SUFFIXES = tuple(
    sorted(
        {
            "으로부터",
            "에서부터",
            "에게서",
            "에서는",
            "으로는",
            "입니다",
            "됩니다",
            "합니다",
            "였습니다",
            "습니다",
            "에서",
            "으로",
            "에게",
            "까지",
            "부터",
            "처럼",
            "보다",
            "하며",
            "하여",
            "해서",
            "해야",
            "하려면",
            "으려면",
            "하고",
            "이며",
            "에서",
            "으로",
            "로는",
            "에는",
            "에게",
            "한테",
            "께서",
            "마다",
            "조차",
            "마저",
            "밖에",
            "부터",
            "까지",
            "만큼",
            "라고",
            "이라는",
            "으로서",
            "으로써",
            "와",
            "과",
            "을",
            "를",
            "은",
            "는",
            "이",
            "가",
            "의",
            "에",
            "도",
            "만",
            "할",
        },
        key=len,
        reverse=True,
    )
)


def _normalized_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    rendered = format(normalized, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _normalized_year(value: str) -> int:
    year = int(value)
    if year < 100:
        return 2000 + year if year <= 69 else 1900 + year
    return year


def _is_valid_month_day(month: int, day: int) -> bool:
    try:
        date(2000, month, day)
    except ValueError:
        return False
    return True


MONEY_UNIT_FACTORS = {
    "조": Decimal(1_000_000_000_000),
    "억": Decimal(100_000_000),
    "만": Decimal(10_000),
    "천": Decimal(1_000),
    "백": Decimal(100),
}


def _parse_money_expression(value: str) -> Decimal | None:
    expression = value.strip()
    sign = Decimal(1)
    if expression[:1] in {"-", "−", "△"}:
        sign = Decimal(-1)
        expression = expression[1:].strip()
    elif expression[:1] in {"+", "▲"}:
        expression = expression[1:].strip()

    total = Decimal(0)
    cursor = 0
    previous_multiplier: Decimal | None = None
    found = False
    for match in MONEY_PART_RE.finditer(expression):
        if expression[cursor : match.start()].strip():
            return None
        try:
            number = Decimal(match.group(1).replace(",", ""))
        except InvalidOperation:
            return None
        units = match.group(2)
        multiplier = Decimal(1)
        for unit in units:
            multiplier *= MONEY_UNIT_FACTORS[unit]
        if (
            previous_multiplier is not None
            and multiplier >= previous_multiplier
        ):
            return None
        total += number * multiplier
        previous_multiplier = multiplier
        cursor = match.end()
        found = True

    if not found or expression[cursor:].strip():
        return None
    return sign * total


def extract_critical_values(value: str) -> frozenset[str]:
    """Extract normalized values that must be present in supporting evidence."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    facts: set[str] = set()
    full_years: set[int] = set()
    month_days: set[tuple[int, int]] = set()

    for match in ACADEMIC_YEAR_RE.finditer(normalized):
        facts.add(f"academic_year:{int(match.group(1)):04d}")
    # The PNU academic-calendar table labels terms as ``2026년도 2학기``
    # while user questions commonly say ``2026학년도``.  In the presence of
    # a semester marker these are the same academic-year scope.
    for match in ACADEMIC_YEAR_DO_RE.finditer(normalized):
        tail = normalized[match.end() : match.end() + 16]
        if SEMESTER_RE.search(tail):
            facts.add(f"academic_year:{int(match.group(1)):04d}")
    for match in CALENDAR_YEAR_RE.finditer(normalized):
        facts.add(f"calendar_year:{int(match.group(1)):04d}")
    for match in SEMESTER_RE.finditer(normalized):
        facts.add(f"semester:{int(match.group(1))}")
    for match in ROUND_RE.finditer(normalized):
        facts.add(f"round:{int(match.group(1))}")
    # Administrative tables often elide the counter on all but the last
    # coordinated value (``1 ・ 4차``).  ROUND_RE sees only ``4차``; recover
    # the preceding values without treating unrelated bare numbers as rounds.
    for match in COORDINATED_ROUND_RE.finditer(normalized):
        for value in re.findall(r"\d+", match.group(1)):
            facts.add(f"round:{int(value)}")
        facts.add(f"round:{int(match.group(2))}")
    for match in FULL_DATE_RE.finditer(normalized):
        year = _normalized_year(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        if not _is_valid_month_day(month, day):
            continue
        full_years.add(year)
        month_days.add((month, day))
        facts.add(f"date:{year:04d}-{month:02d}-{day:02d}")
        facts.add(f"month_day:{month:02d}-{day:02d}")
    for pattern in (
        KOREAN_MONTH_DAY_RE,
        DOTTED_MONTH_DAY_RE,
        SLASH_MONTH_DAY_RE,
    ):
        for match in pattern.finditer(normalized):
            month = int(match.group(1))
            day = int(match.group(2))
            if not _is_valid_month_day(month, day):
                continue
            month_days.add((month, day))
            facts.add(f"month_day:{month:02d}-{day:02d}")
    # Notices often abbreviate a same-month range as ``26.7.8(수)~22(수)``.
    # The bare endpoint is meaningful only because the start pins both year
    # and month, so infer it locally rather than treating arbitrary bare
    # numbers as dates.
    for match in ABBREVIATED_SAME_MONTH_RANGE_RE.finditer(normalized):
        year = _normalized_year(match.group(1))
        month = int(match.group(2))
        end_day = int(match.group(4))
        if not _is_valid_month_day(month, end_day):
            continue
        full_years.add(year)
        month_days.add((month, end_day))
        facts.add(f"date:{year:04d}-{month:02d}-{end_day:02d}")
        facts.add(f"month_day:{month:02d}-{end_day:02d}")
    if len(full_years) == 1:
        year = next(iter(full_years))
        for month, day in month_days:
            facts.add(f"date:{year:04d}-{month:02d}-{day:02d}")
    for match in COLON_TIME_RE.finditer(normalized):
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour == 24 and minute != 0:
            continue
        minutes = hour * 60 + minute
        facts.add(f"time_minutes:{minutes}")
    for match in KOREAN_TIME_RE.finditer(normalized):
        meridiem = match.group(1)
        hour = int(match.group(2))
        minute = int(match.group(3) or 0)
        if minute >= 60:
            continue
        if meridiem and not 1 <= hour <= 12:
            continue
        if not meridiem and not 0 <= hour <= 24:
            continue
        if meridiem == "오후" and hour < 12:
            hour += 12
        elif meridiem == "오전" and hour == 12:
            hour = 0
        facts.add(f"time_minutes:{hour * 60 + minute}")
    for match in MONEY_EXPR_RE.finditer(normalized):
        expression = match.group(1)
        amount = _parse_money_expression(expression)
        if amount is None:
            raw = re.sub(r"\s+", "", expression)
            facts.add(f"amount_krw_raw:{raw}")
            continue
        facts.add(f"amount_krw:{_normalized_decimal(amount)}")
    for match in PERCENT_RE.finditer(normalized):
        try:
            percent = Decimal(match.group(2).replace(",", ""))
        except InvalidOperation:
            continue
        explicit_sign = match.group(1)
        suffix = normalized[match.end() : match.end() + 12]
        if explicit_sign in {"-", "−", "△"} or (
            not explicit_sign
            and re.search(r"인하|감소|하락|축소|내림", suffix)
        ):
            percent = -abs(percent)
        else:
            percent = abs(percent)
        facts.add(f"percent:{_normalized_decimal(percent)}")
    # Parsed tables often put ``(%)`` only in the column header, leaving body
    # cells as bare decimals.  Preserve their percentage semantics so a
    # synthesized claim such as ``구성비는 21.39%`` can be attributed to the
    # exact table rather than rejected as a critical-value mismatch.
    if re.search(r"구성비\s*\(\s*%\s*\)", normalized):
        for line in normalized.splitlines():
            if "\t" not in line:
                continue
            for field in line.split("\t")[1:]:
                candidate = normalize_text(field)
                if not re.fullmatch(r"(?:100|\d{1,2}(?:\.\d{1,2})?)", candidate):
                    continue
                try:
                    percent = Decimal(candidate)
                except InvalidOperation:
                    continue
                facts.add(f"percent:{_normalized_decimal(percent)}")
    # Korean notices commonly elide the unit on the first endpoint of a
    # range (``3~4개월``).  QUANTITY_RE sees only ``4개월`` in that form,
    # which makes a faithful ``3개월에서 4개월`` paraphrase look as if it
    # invented the lower endpoint.  Recover the first value only when both
    # endpoints are joined locally and share one explicit supported unit.
    for match in SHARED_UNIT_QUANTITY_RANGE_RE.finditer(normalized):
        unit = match.group(3)
        for raw_quantity in match.group(1, 2):
            try:
                quantity = Decimal(raw_quantity.replace(",", ""))
            except InvalidOperation:
                continue
            facts.add(
                f"quantity:{_normalized_decimal(quantity)}:{unit}"
            )
    for match in QUANTITY_RE.finditer(normalized):
        try:
            quantity = Decimal(match.group(1).replace(",", ""))
        except InvalidOperation:
            continue
        facts.add(
            f"quantity:{_normalized_decimal(quantity)}:{match.group(2)}"
        )
    for match in PHONE_RE.finditer(normalized):
        facts.add(f"phone:{''.join(match.groups())}")

    return frozenset(facts)


def _term_variants(value: str) -> set[str]:
    variants = {value}
    if not re.fullmatch(r"[가-힣]+", value):
        return variants
    for suffix in KOREAN_TERM_SUFFIXES:
        if value.endswith(suffix) and len(value) - len(suffix) >= 2:
            variants.add(value[: -len(suffix)])
    return variants


def _claim_term_supported(
    claim_term: str,
    source_terms: set[str],
) -> bool:
    claim_variants = _term_variants(claim_term)
    for source_term in source_terms:
        source_variants = _term_variants(source_term)
        for claim_value in claim_variants:
            for source_value in source_variants:
                if claim_value == source_value:
                    return True
                if min(len(claim_value), len(source_value)) < 2:
                    continue
                if claim_value in source_value or source_value in claim_value:
                    return True
    return False


def _scope_conflicts(
    left: frozenset[str],
    right: frozenset[str],
) -> bool:
    for prefix in (
        "round:",
        "academic_year:",
        "calendar_year:",
        "semester:",
    ):
        left_values = {
            value for value in left if value.startswith(prefix)
        }
        right_values = {
            value for value in right if value.startswith(prefix)
        }
        if (
            left_values
            and right_values
            and left_values != right_values
        ):
            return True
    return False


def _evidence_units(source: str) -> list[str]:
    lines: list[str] = []
    for raw_line in str(source or "").splitlines():
        line = normalize_text(raw_line)
        if line:
            lines.append(line)
    if not lines:
        fallback = normalize_text(str(source or ""))
        return [fallback] if fallback else []

    units = list(lines)
    for left, right in zip(lines, lines[1:]):
        if len(left) + len(right) + 1 > 500:
            continue
        left_values = extract_critical_values(left)
        right_values = extract_critical_values(right)
        if _scope_conflicts(left_values, right_values):
            continue
        if (
            sum(value.startswith("date:") for value in left_values) >= 2
            and sum(value.startswith("date:") for value in right_values) >= 2
        ):
            # Adjacent academic-calendar rows are independent facts.  They
            # may only be combined later by the same-subject aggregation
            # guard, not merely because they are neighbors in one chunk.
            continue
        units.append(f"{left} {right}")
    return list(dict.fromkeys(units))


def _relation_clauses(value: str) -> list[str]:
    """Split rows and connective clauses without merging neighboring rows."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    clauses: list[str] = []
    # PDF/table extraction commonly represents independent rows as tabs or
    # Markdown cells. Preserve those boundaries before normalize_text folds
    # all whitespace into spaces, otherwise values from adjacent subjects can
    # be assigned to the same relation atom.
    for raw_line in re.split(
        r"[\r\n\t]+|\s*\|\s*|\s+/\s+",
        normalized,
    ):
        line = normalize_text(raw_line)
        if not line:
            continue
        parts = RELATION_CLAUSE_SPLIT_RE.split(line)
        for part in (
            normalize_text(item).strip(" .") for item in parts
        ):
            if not part:
                continue
            parallel = re.split(r"\s+및\s+", part)
            if len(parallel) > 1 and sum(
                any(
                    pattern.search(item)
                    for pattern in RELATION_DIRECTION_PATTERNS.values()
                )
                for item in parallel
            ) >= 2:
                clauses.extend(item for item in parallel if item)
            else:
                clauses.append(part)
    if clauses:
        return clauses
    fallback = normalize_text(normalized).strip(" .")
    return [fallback] if fallback else []


def _semantic_scope_terms(value: str) -> frozenset[str]:
    terms: set[str] = set()
    # Year/semester/date expressions are compared structurally through
    # scope_values. Remove their surface forms so tokens such as "학년도" do
    # not become an extra lexical anchor that a title-scoped body row cannot
    # satisfy.
    lexical_value = str(value or "")
    for pattern in (
        ACADEMIC_YEAR_RE,
        ACADEMIC_YEAR_DO_RE,
        CALENDAR_YEAR_RE,
        SEMESTER_RE,
        ROUND_RE,
        FULL_DATE_RE,
        KOREAN_MONTH_DAY_RE,
        DOTTED_MONTH_DAY_RE,
        SLASH_MONTH_DAY_RE,
    ):
        lexical_value = pattern.sub(" ", lexical_value)
    for term in tokenize(lexical_value, include_ngrams=False):
        if term.isdigit():
            continue
        variants = _term_variants(term)
        if any(
            stem in variant
            for stem in RELATION_TERM_STEMS
            for variant in variants
        ):
            continue
        if any(
            variant in SEMANTIC_SCOPE_PREDICATE_NOISE
            for variant in variants
        ):
            continue
        terms.add(term)
    return frozenset(terms)


def _direction_scope_terms(value: str) -> frozenset[str]:
    """Drop a generic object when a direction has a specific subject.

    Korean administrative tables commonly abbreviate ``경영대학원 등록금``
    to ``경영대학원`` in the result row.  Treating ``등록금`` as a second
    mandatory subject anchor rejects that faithful abbreviation.  Keep it
    when it is the only scope term so an unscoped ``등록금 인상`` claim does
    not become universally compatible.
    """

    terms = _semantic_scope_terms(value)
    generic_scope = {"등록금", "부산대학교", "부산대"}
    specific = frozenset(
        term
        for term in terms
        if not _term_variants(term).intersection(generic_scope)
    )
    return specific or terms


SEMANTIC_SCOPE_VALUE_PREFIXES = (
    "academic_year:",
    "calendar_year:",
    "semester:",
    "round:",
    "date:",
    "month_day:",
)


def _semantic_scope_values(value: str) -> frozenset[str]:
    return frozenset(
        fact
        for fact in extract_critical_values(value)
        if fact.startswith(SEMANTIC_SCOPE_VALUE_PREFIXES)
    )


def _semantic_payload_values(value: str) -> frozenset[str]:
    payload: set[str] = set()
    for fact in extract_critical_values(value):
        if fact.startswith(SEMANTIC_SCOPE_VALUE_PREFIXES):
            continue
        if fact.startswith(("percent:", "amount_krw:")):
            prefix, raw_number = fact.split(":", 1)
            try:
                magnitude = _normalized_decimal(abs(Decimal(raw_number)))
            except InvalidOperation:
                payload.add(fact)
            else:
                payload.add(f"{prefix}:{magnitude}")
            continue
        payload.add(fact)
    return frozenset(payload)


def _comparator_value_key(raw_value: str) -> str | None:
    facts = sorted(_semantic_payload_values(raw_value))
    if facts:
        return "|".join(facts)
    match = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*"
        r"(학점|개월|퍼센트|점|명|개|회|세|학기|%|％)?",
        raw_value,
    )
    if match is None:
        return None
    try:
        number = _normalized_decimal(
            Decimal(match.group(1).replace(",", ""))
        )
    except InvalidOperation:
        return None
    unit = match.group(2) or ""
    if unit in {"퍼센트", "%", "％"}:
        unit = "percent"
    return f"number:{number}:{unit}"


def _semantic_relation_modality_text(local: str, *, kind: str) -> str:
    if RELATION_CANCELLED_RE.search(local):
        return "cancelled"
    if RELATION_NEGATED_ASSERTION_RE.search(local):
        return "negated"
    if RELATION_UNDETERMINED_RE.search(local):
        return "undetermined"
    if RELATION_PLANNED_RE.search(local):
        return "planned"
    if kind == "temporal_boundary":
        possibility_scope = TEMPORAL_ACTION_AVAILABILITY_RE.sub(" ", local)
    elif kind == "comparator":
        # A threshold followed by a different capability predicate describes
        # an asserted rule: ``70점 이상을 취득해야 이수할 수 있다``. Do not let
        # the later ``할 수 있다`` turn the preceding comparator into a
        # merely possible threshold.
        possibility_scope = PERMISSION_ALLOWED_RE.sub(" ", local)
    else:
        possibility_scope = local
    if kind in {"direction", "comparator", "temporal_boundary"} and (
        RELATION_POSSIBLE_RE.search(possibility_scope)
    ):
        return "possible"
    return "asserted"


def _semantic_relation_modality(
    clause: str,
    match: re.Match[str],
    *,
    kind: str,
    local_end: int | None = None,
) -> str:
    """Classify whether a matched relation is asserted or merely discussed."""

    local = clause[
        match.start() : (
            local_end if local_end is not None else match.end() + 56
        )
    ]
    return _semantic_relation_modality_text(local, kind=kind)


def _temporal_boundary_value_key(raw_value: str) -> str | None:
    facts = extract_critical_values(raw_value)
    values = sorted(
        fact
        for fact in facts
        if fact.startswith(("month_day:", "time_minutes:"))
    )
    return "|".join(values) if values else None


def _temporal_scope_values(prefix: str, raw_value: str) -> frozenset[str]:
    hierarchical = {
        value
        for value in _semantic_scope_values(prefix)
        if value.startswith(
            (
                "academic_year:",
                "calendar_year:",
                "semester:",
                "round:",
            )
        )
    }
    local_date = {
        value
        for value in _semantic_scope_values(raw_value)
        if value.startswith(("calendar_year:", "date:", "month_day:"))
    }
    month_days = {
        value.removeprefix("month_day:")
        for value in local_date
        if value.startswith("month_day:")
    }
    if (
        not any(value.startswith("date:") for value in local_date)
        and len(month_days) == 1
    ):
        prefix_values = _semantic_scope_values(prefix)
        years = {
            value.split(":", 1)[1]
            for value in prefix_values
            if value.startswith("calendar_year:")
        }
        years.update(
            value.split(":", 1)[1].split("-", 1)[0]
            for value in prefix_values
            if value.startswith("date:")
        )
        if len(years) == 1:
            local_date.add(
                f"date:{next(iter(years))}-{next(iter(month_days))}"
            )
    return frozenset(hierarchical | local_date)


def _matches_overlap(left: re.Match[str], right: re.Match[str]) -> bool:
    return left.start() < right.end() and right.start() < left.end()


def _permission_relation_matches(
    clause: str,
) -> list[tuple[re.Match[str], str, bool]]:
    """Return every non-overlapping permission predicate in text order."""

    matches: list[tuple[re.Match[str], str, bool]] = []
    double_negatives = list(PERMISSION_DOUBLE_NEGATION_RE.finditer(clause))
    matches.extend((match, "allowed", True) for match in double_negatives)

    denied = [
        match
        for match in PERMISSION_DENIED_RE.finditer(clause)
        if not any(
            _matches_overlap(match, double_negative)
            for double_negative in double_negatives
        )
    ]
    matches.extend((match, "denied", False) for match in denied)

    blocked = double_negatives + denied
    allowed = [
        match
        for match in PERMISSION_ALLOWED_RE.finditer(clause)
        if not any(_matches_overlap(match, item) for item in blocked)
    ]
    matches.extend((match, "allowed", False) for match in allowed)
    return sorted(matches, key=lambda item: (item[0].start(), item[0].end()))


def _semantic_clause_has_relation(clause: str) -> bool:
    return bool(
        _permission_relation_matches(clause)
        or COMPARATOR_VALUE_RE.search(clause)
        or MINMAX_VALUE_RE.search(clause)
        or TEMPORAL_BOUNDARY_RE.search(clause)
        or any(
            pattern.search(clause)
            for pattern in RELATION_DIRECTION_PATTERNS.values()
        )
    )


def _salient_permission_scope(
    scope_terms: frozenset[str],
) -> frozenset[str]:
    return frozenset(
        term
        for term in scope_terms
        if not any(
            stem in variant
            for stem in PERMISSION_GENERIC_SCOPE_STEMS
            for variant in _term_variants(term)
        )
    )


def _semantic_relation_atoms(value: str) -> list[dict[str, Any]]:
    """Extract narrow relation atoms used only as an attribution veto.

    These atoms are deliberately not a general entailment model.  They cover
    high-impact relation reversals common in university notices while leaving
    the existing lexical and critical-value checks in charge of acceptance.
    """

    atoms: list[dict[str, Any]] = []
    normalized_value = unicodedata.normalize("NFKC", str(value or ""))
    # A common scholarship-table shorthand applies the final ``까지 지원
    # 가능`` to every comma-separated programme limit, e.g. ``학사 8학기,
    # 석사 4학기, 박사 6학기까지 지원 가능``.  Materialize each limit as an
    # upper-bound atom so a faithful ``석사 4학기 이하`` paraphrase is not
    # discarded merely because the counter phrase is elided in that cell.
    for raw_line in normalized_value.splitlines():
        if not COORDINATED_TERM_LIMIT_TRIGGER_RE.search(raw_line):
            continue
        for segment in re.split(r"[,，]", raw_line):
            matches = list(COORDINATED_TERM_LIMIT_ITEM_RE.finditer(segment))
            if not matches:
                continue
            # The first segment can also contain ``2026년 2학기 등록 예정자``;
            # the programme limit is the last ``N학기`` in that segment.
            match = matches[-1]
            value_key = _comparator_value_key(
                f"{match.group('value')}학기"
            )
            if value_key is None:
                continue
            raw_scope_terms = _semantic_scope_terms(match.group("scope"))
            degree_terms = {
                "학사" if "학부" in term else term
                for term in raw_scope_terms
                if any(
                    level in term
                    for level in ("학부", "학사", "석사", "박사")
                )
            }
            scope_terms = frozenset(
                degree_terms
                | {
                    term
                    for term in raw_scope_terms
                    if term in {"국내", "국내외", "해외"}
                }
            )
            atoms.append(
                {
                    "kind": "comparator",
                    "operator": "이하",
                    "scope_terms": scope_terms or raw_scope_terms,
                    "scope_values": frozenset(),
                    "value_key": value_key,
                    "payload_values": frozenset(),
                    "modality": "asserted",
                }
            )
    previous_neutral_scope_terms: frozenset[str] = frozenset()
    for clause in _relation_clauses(value):
        permission_matches = _permission_relation_matches(clause)
        previous_end = 0
        for index, (permission_match, operator, double_negative) in enumerate(
            permission_matches
        ):
            prefix = clause[previous_end : permission_match.start()]
            scope_terms = _semantic_scope_terms(prefix)
            if (
                index == 0
                and not _salient_permission_scope(scope_terms)
                and previous_neutral_scope_terms
            ):
                scope_terms = frozenset(
                    set(scope_terms) | set(previous_neutral_scope_terms)
                )
            next_start = (
                permission_matches[index + 1][0].start()
                if index + 1 < len(permission_matches)
                else len(clause)
            )
            atoms.append(
                {
                    "kind": "permission",
                    "operator": operator,
                    "scope_terms": scope_terms,
                    "scope_values": _semantic_scope_values(prefix),
                    "value_key": None,
                    "payload_values": frozenset(),
                    "modality": (
                        "asserted"
                        if double_negative
                        else _semantic_relation_modality(
                            clause,
                            permission_match,
                            kind="permission",
                            local_end=next_start,
                        )
                    ),
                }
            )
            previous_end = permission_match.end()

        for match in COMPARATOR_VALUE_RE.finditer(clause):
            value_key = _comparator_value_key(match.group(1))
            if value_key is None:
                continue
            prefix = clause[: match.start()]
            suffix = clause[match.end() :]
            scope_terms = _semantic_scope_terms(f"{prefix} {suffix}")
            if not scope_terms:
                previous_comparator = next(
                    (
                        atom
                        for atom in reversed(atoms)
                        if atom["kind"] == "comparator"
                    ),
                    None,
                )
                if previous_comparator is not None:
                    scope_terms = previous_comparator["scope_terms"]
            operator = COMPARATOR_CANONICAL.get(match.group(2), match.group(2))
            complement_match = COMPARATOR_NEGATION_RE.search(
                clause[match.end() :]
            )
            if complement_match is not None:
                operator = COMPARATOR_COMPLEMENT[operator]
                local_end = min(len(clause), match.end() + 56)
                consumed_end = match.end() + complement_match.end()
                modality = _semantic_relation_modality_text(
                    (
                        clause[match.start() : match.end()]
                        + " "
                        + clause[consumed_end:local_end]
                    ),
                    kind="comparator",
                )
            else:
                modality = _semantic_relation_modality(
                    clause,
                    match,
                    kind="comparator",
                )
            atoms.append(
                {
                    "kind": "comparator",
                    "operator": operator,
                    "scope_terms": scope_terms,
                    "scope_values": _semantic_scope_values(
                        f"{prefix} {suffix}"
                    ),
                    "value_key": value_key,
                    "payload_values": frozenset(),
                    "modality": modality,
                }
            )

        for match in MINMAX_VALUE_RE.finditer(clause):
            value_key = _comparator_value_key(match.group(2))
            if value_key is None:
                continue
            prefix = clause[: match.start()]
            atoms.append(
                {
                    "kind": "comparator",
                    "operator": COMPARATOR_CANONICAL[match.group(1)],
                    "scope_terms": _semantic_scope_terms(prefix),
                    "scope_values": _semantic_scope_values(prefix),
                    "value_key": value_key,
                    "payload_values": frozenset(),
                    "modality": _semantic_relation_modality(
                        clause,
                        match,
                        kind="comparator",
                    ),
                }
            )

        for match in TEMPORAL_BOUNDARY_RE.finditer(clause):
            value_key = _temporal_boundary_value_key(match.group(1))
            if value_key is None:
                continue
            prefix = clause[: match.start()]
            atoms.append(
                {
                    "kind": "temporal_boundary",
                    "operator": TEMPORAL_BOUNDARY_OPERATORS[match.group(2)],
                    "scope_terms": _semantic_scope_terms(prefix),
                    "scope_values": _temporal_scope_values(
                        prefix,
                        match.group(1),
                    ),
                    "value_key": value_key,
                    "payload_values": frozenset(),
                    "modality": _semantic_relation_modality(
                        clause,
                        match,
                        kind="temporal_boundary",
                    ),
                }
            )

        abbreviated_range = ABBREVIATED_SAME_MONTH_RANGE_RE.search(clause)
        if abbreviated_range is not None:
            year = _normalized_year(abbreviated_range.group(1))
            month = int(abbreviated_range.group(2))
            start_day = int(abbreviated_range.group(3))
            end_day = int(abbreviated_range.group(4))
            start_text = f"{year:04d}.{month:02d}.{start_day:02d}"
            end_text = f"{year:04d}.{month:02d}.{end_day:02d}"
            range_scope_terms = _semantic_scope_terms(
                clause[: abbreviated_range.start()]
            )
            range_modality = _semantic_relation_modality_text(
                clause,
                kind="temporal_boundary",
            )
            atoms.extend(
                [
                    {
                        "kind": "temporal_boundary",
                        "operator": "start_inclusive",
                        "scope_terms": range_scope_terms,
                        "scope_values": _temporal_scope_values(
                            "", start_text
                        ),
                        "value_key": _temporal_boundary_value_key(start_text),
                        "payload_values": frozenset(),
                        "modality": range_modality,
                    },
                    {
                        "kind": "temporal_boundary",
                        "operator": "end_inclusive",
                        "scope_terms": range_scope_terms,
                        "scope_values": _temporal_scope_values(
                            start_text, end_text
                        ),
                        "value_key": _temporal_boundary_value_key(end_text),
                        "payload_values": frozenset(),
                        "modality": range_modality,
                    },
                ]
            )
            range_separators: list[re.Match[str]] = []
        else:
            range_separators = list(
                re.finditer(r"[~∼～]|\s[-–—]\s", clause)
            )
        if len(range_separators) == 1:
            separator = range_separators[0]
            left = clause[: separator.start()]
            right = clause[separator.end() :]
            left_key = _temporal_boundary_value_key(left)
            right_key = _temporal_boundary_value_key(right)
            if left_key and right_key:
                range_scope_terms = _semantic_scope_terms(left)
                range_modality = _semantic_relation_modality_text(
                    clause,
                    kind="temporal_boundary",
                )
                atoms.extend(
                    [
                        {
                            "kind": "temporal_boundary",
                            "operator": "start_inclusive",
                            "scope_terms": range_scope_terms,
                            "scope_values": _temporal_scope_values("", left),
                            "value_key": left_key,
                            "payload_values": frozenset(),
                            "modality": range_modality,
                        },
                        {
                            "kind": "temporal_boundary",
                            "operator": "end_inclusive",
                            "scope_terms": range_scope_terms,
                            "scope_values": _temporal_scope_values(left, right),
                            "value_key": right_key,
                            "payload_values": frozenset(),
                            "modality": range_modality,
                        },
                    ]
                )

        for operator, pattern in RELATION_DIRECTION_PATTERNS.items():
            for match in pattern.finditer(clause):
                prefix = clause[: match.start()]
                relation_operator = operator
                if DIRECTION_NEGATION_RE.search(clause[match.end() :]):
                    relation_operator = f"not_{operator}"
                atoms.append(
                    {
                        "kind": "direction",
                        "operator": relation_operator,
                        "scope_terms": _direction_scope_terms(prefix),
                        "scope_values": _semantic_scope_values(prefix),
                        "value_key": None,
                        # Rates and amounts often follow the relation noun
                        # ("인상률은 3.95%") or sit in parentheses after the
                        # predicate. Clause boundaries keep neighboring rows
                        # separate, so use the complete local clause.
                        "payload_values": _semantic_payload_values(clause),
                        "modality": _semantic_relation_modality(
                            clause,
                            match,
                            kind="direction",
                        ),
                    }
                )
        previous_neutral_scope_terms = (
            _semantic_scope_terms(clause)
            if not _semantic_clause_has_relation(clause)
            else frozenset()
        )
    return atoms


def _semantic_scopes_match(
    claim_scope: frozenset[str],
    source_scope: frozenset[str],
) -> bool:
    if not claim_scope:
        return not source_scope
    matched = {
        claim_term
        for claim_term in claim_scope
        if _claim_term_supported(claim_term, set(source_scope))
    }
    required = min(MIN_CLAIM_ANCHORS, len(claim_scope))
    return (
        len(matched) >= required
        and len(matched) / len(claim_scope) >= CLAIM_SUPPORT_THRESHOLD
    )


SEMANTIC_EXCLUSIVE_SCOPE_GROUPS = (
    ("학부", "대학원"),
    ("학부생", "대학원생"),
    ("학생", "학과"),
    ("신입생", "재학생"),
)


def _semantic_scope_markers(
    scope: frozenset[str],
    markers: tuple[str, ...],
) -> frozenset[str]:
    return frozenset(
        marker
        for marker in markers
        if any(
            marker in variant
            for term in scope
            for variant in _term_variants(term)
        )
    )


def _semantic_scopes_conflict(
    claim_scope: frozenset[str],
    source_scope: frozenset[str],
) -> bool:
    for group in SEMANTIC_EXCLUSIVE_SCOPE_GROUPS:
        claim_markers = _semantic_scope_markers(claim_scope, group)
        source_markers = _semantic_scope_markers(source_scope, group)
        if (
            claim_markers
            and source_markers
            and claim_markers.isdisjoint(source_markers)
        ):
            return True
    return False


def _exact_semantic_scope_overlap(
    claim_scope: frozenset[str],
    source_scope: frozenset[str],
) -> int:
    """Count morphology-normalized exact scope anchors, without substrings."""

    return sum(
        any(
            _term_variants(claim_term).intersection(
                _term_variants(source_term)
            )
            for source_term in source_scope
        )
        for claim_term in claim_scope
    )


def _semantic_permission_scopes_match(
    claim_scope: frozenset[str],
    source_scope: frozenset[str],
) -> bool:
    if _semantic_scopes_conflict(claim_scope, source_scope):
        return False
    if _semantic_scopes_match(claim_scope, source_scope):
        return True
    # Long generated paraphrases can add UI wording that is absent from a
    # concise policy sentence (for example "온라인 신청이 원칙"). Permit a
    # one-anchor fallback only for long, non-generic scopes; short policy
    # claims still require the stricter subject match above.
    salient_claim = _salient_permission_scope(claim_scope)
    salient_source = _salient_permission_scope(source_scope)
    if len(salient_claim) < 4:
        return False
    fallback_noise = {
        "경우",
        "관련",
        "사항",
        "내용",
        "처리",
        "등록금",
        "납부",
        "학생",
        "대상",
        "방법",
        "절차",
    }
    # The fallback is intentionally exact at the normalized-morpheme level.
    # Substring matching would make a specific subject such as
    # ``분할납부자`` match the unrelated generic token ``납부`` and could
    # attach a permission predicate from a different FAQ item.
    return any(
        not _term_variants(term).intersection(fallback_noise)
        and any(
            _term_variants(term).intersection(_term_variants(source_term))
            for source_term in salient_source
        )
        for term in salient_claim
    )


def _semantic_temporal_scopes_match(
    claim_atom: dict[str, Any],
    source_atom: dict[str, Any],
) -> bool:
    claim_scope = claim_atom["scope_terms"]
    source_scope = source_atom["scope_terms"]
    if _semantic_scopes_conflict(claim_scope, source_scope):
        return False
    if not claim_scope:
        return True
    if _semantic_scopes_match(claim_scope, source_scope):
        return True
    hierarchical_prefixes = (
        "academic_year:",
        "calendar_year:",
        "semester:",
        "round:",
    )
    has_hierarchical_scope = any(
        value.startswith(hierarchical_prefixes)
        for value in claim_atom["scope_values"]
    )
    if not has_hierarchical_scope:
        return False
    if not source_scope:
        return True
    # Exact dates plus a pinned year/semester/round already provide strong
    # row identity.  In that setting one matching local label (for example
    # 본등록 or 신청) is sufficient even when the generated claim also says
    # the generic word 기간.
    return any(
        _claim_term_supported(term, set(source_scope))
        for term in claim_scope
    )


def _semantic_atom_compatible(
    claim_atom: dict[str, Any],
    source_atom: dict[str, Any],
    *,
    require_same_operator: bool,
) -> bool:
    if claim_atom["kind"] != source_atom["kind"]:
        return False
    scopes_match = (
        _semantic_temporal_scopes_match(claim_atom, source_atom)
        if claim_atom["kind"] == "temporal_boundary"
        else (
            _semantic_permission_scopes_match(
                claim_atom["scope_terms"], source_atom["scope_terms"]
            )
            if claim_atom["kind"] == "permission"
            else (
                not _semantic_scopes_conflict(
                    claim_atom["scope_terms"], source_atom["scope_terms"]
                )
                and _semantic_scopes_match(
                    claim_atom["scope_terms"], source_atom["scope_terms"]
                )
            )
        )
    )
    if not scopes_match:
        return False
    if not claim_atom["scope_values"].issubset(
        source_atom["scope_values"]
    ):
        return False
    if claim_atom["kind"] == "comparator" and (
        claim_atom["value_key"] != source_atom["value_key"]
    ):
        return False
    if claim_atom["kind"] == "temporal_boundary":
        claim_values = set(str(claim_atom["value_key"]).split("|"))
        source_values = set(str(source_atom["value_key"]).split("|"))
        if not claim_values.issubset(source_values):
            return False
    if require_same_operator and (
        claim_atom["operator"] != source_atom["operator"]
    ):
        return False
    if require_same_operator and (
        claim_atom.get("modality", "asserted")
        != source_atom.get("modality", "asserted")
    ):
        return False
    if require_same_operator and claim_atom["kind"] == "direction":
        claim_values = claim_atom["payload_values"]
        source_values = source_atom["payload_values"]
        if claim_values and not claim_values.issubset(source_values):
            return False
    return True


def _semantic_operators_conflict(
    kind: str,
    claim_operator: str,
    source_operator: str,
) -> bool:
    if claim_operator == source_operator:
        return False
    if kind != "direction":
        return True
    if claim_operator.startswith("not_"):
        return source_operator == claim_operator.removeprefix("not_")
    if source_operator.startswith("not_"):
        return claim_operator == source_operator.removeprefix("not_")
    return True


def _coordinated_direction_supported(
    claim_atom: dict[str, Any],
    source_atoms: list[dict[str, Any]],
) -> bool:
    """Require every coordinated exclusive subject to have its own row.

    A claim such as ``학부 및 대학원 모두 동결`` is represented by one
    atom, while a parsed table usually represents it as two atoms.  A partial
    match is unsafe, so this recovery succeeds only when every coordinated
    subject has a same-operator, same-scope source atom and none has a
    conflicting operator.
    """

    if claim_atom.get("kind") != "direction":
        return False
    for group in SEMANTIC_EXCLUSIVE_SCOPE_GROUPS:
        markers = _semantic_scope_markers(claim_atom["scope_terms"], group)
        if len(markers) < 2:
            continue
        for marker in markers:
            marker_atom = dict(claim_atom)
            marker_atom["scope_terms"] = frozenset({marker})
            bound_atoms = [
                source_atom
                for source_atom in source_atoms
                if _semantic_atom_compatible(
                    marker_atom,
                    source_atom,
                    require_same_operator=False,
                )
            ]
            if any(
                _semantic_operators_conflict(
                    "direction",
                    str(marker_atom["operator"]),
                    str(source_atom["operator"]),
                )
                for source_atom in bound_atoms
            ):
                return False
            if not any(
                _semantic_atom_compatible(
                    marker_atom,
                    source_atom,
                    require_same_operator=True,
                )
                for source_atom in bound_atoms
            ):
                break
        else:
            return True
    return False


def _semantic_atom_with_context(
    atom: dict[str, Any],
    context_scope_values: frozenset[str],
    context_scope_terms: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Fill absent year/semester scope from source-level metadata.

    A notice title often owns the year while its body rows omit it. Explicit
    row scope always wins: title metadata only fills a missing scope category
    and never overwrites a year or semester stated in the row itself.
    """

    if not context_scope_values:
        return atom
    scope_values = set(atom["scope_values"])
    # Inherit only hierarchical notice scope and only when metadata agrees.
    # Dates in titles often describe publication or multiple deadlines, so
    # they must not be attached to every body row.
    for prefix in (
        "academic_year:",
        "calendar_year:",
        "semester:",
        "round:",
    ):
        if any(value.startswith(prefix) for value in scope_values):
            continue
        candidates = {
            value
            for value in context_scope_values
            if value.startswith(prefix)
        }
        if len(candidates) == 1:
            scope_values.update(candidates)
    enriched = dict(atom)
    enriched["scope_values"] = frozenset(scope_values)
    if atom.get("kind") == "temporal_boundary" and context_scope_terms:
        # Notice titles commonly carry the audience (for example 재학생)
        # while an individual schedule row carries only the round name
        # (본등록).  Both jointly scope the row; inheriting title terms avoids
        # rejecting a correct dated claim without borrowing values from a
        # neighboring schedule row.
        enriched["scope_terms"] = frozenset(
            set(atom.get("scope_terms", frozenset())) | set(context_scope_terms)
        )
    return enriched


_CATEGORICAL_AVAILABILITY_RE = re.compile(
    r"가능한\s+[^,.!?]{1,24}?(?:은|는|이|가)\s*(?P<items>[^.!?]+)"
)
_CATEGORICAL_LOCATION_AVAILABILITY_RE = re.compile(
    r"(?:은|는|에는|중)\s*(?P<items>[^.!?]{3,160}?)\s*"
    r"(?:전국\s*지점)?에서\s*(?:등록금을?\s*)?"
    r"(?:납부|신청|결제|송금|이용)\s*할\s*수"
)
_CATEGORICAL_ITEM_SPLIT_RE = re.compile(
    r"[,，·・/]\s*|\s+(?:및|또는|과|와)\s+"
)
_CATEGORICAL_ITEM_NOISE = frozenset(
    {"가능", "전국", "전국지점", "은행", "입니다", "지점"}
)
_NAMED_PROCEDURE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9.-]{3,}(?![A-Za-z0-9])")
_NAMED_PROCEDURE_TOKEN_NOISE = frozenset(
    {"app", "html", "http", "https", "korea", "korean", "mobile", "online", "smba"}
)
_PROCEDURE_AVAILABILITY_CLAIM_RE = re.compile(
    r"(?:납부|신청|결제|송금|이용)[^.!?]{0,36}?(?:할|될)\s*수\s*있"
)
_PROCEDURE_METHOD_RE = re.compile(
    r"(?:을|를)?\s*통한\s+[^.!?\n]{0,60}?(?:방법|안내|절차)"
    r"|(?:납부|신청|결제|송금|이용)\s*(?:방법|안내|절차)"
)
_PROCEDURE_NEGATION_RE = re.compile(
    r"(?:불가|금지|지원하지\s*않|할\s*수\s*없|이용할\s*수\s*없)"
)
_VERIFICATION_AVAILABILITY_CLAIM_RE = re.compile(
    r"(?P<action>확인|조회|열람|출력|발급|다운로드)"
    r"\s*(?:할|될)\s*수\s*있"
)
_VERIFICATION_UI_MARKER_RE = re.compile(
    r"→|(?:시스템|포털|홈페이지|웹사이트|사이트|누리집|"
    r"애플리케이션|앱|메뉴)"
)
_VERIFICATION_UNAVAILABLE_RE = re.compile(
    r"(?:확인|조회|열람|출력|발급|다운로드)[^.!?\n]{0,24}?"
    r"(?:할\s*수\s*없|불가|금지|중단|지원하지\s*않|되지\s*않|못함|못합니다)"
)
_VERIFICATION_SCOPE_NOISE = frozenset(
    {
        "확인",
        "조회",
        "열람",
        "출력",
        "발급",
        "다운로드",
        "직접",
        "직접적으로",
        "등록금",
        "메뉴",
        "원",
        "원으로",
    }
)
_UI_APPLICATION_CLAIM_RE = re.compile(
    r"(?P<action>납부|신청|결제|송금|이용|제출|등록)"
    r"[^.!?\n]{0,32}?(?:할|될)\s*수\s*있"
)
_UI_APPLICATION_PATH_NOISE = frozenset(
    {
        "로그인",
        "메뉴",
        "메뉴에서",
        "온라인",
        "온라인으로",
        "신청",
        "신청할",
        "있습니다",
        "부산대학교",
        "학생지원시스템",
        "학생지원시스템에",
    }
)
_ELIGIBILITY_AVAILABILITY_CLAIM_RE = re.compile(
    r"(?:(?:신청|지원)[^.!?\n]{0,24}?(?:할|될)|받을)\s*수\s*있"
)
_ELIGIBILITY_LIST_HEADING_RE = re.compile(
    r"(?:선발|신청|지원)\s*대상[^\n]{0,100}?"
    r"(?:신청|지원)\s*가능"
)
_ELIGIBILITY_SECTION_HEADING_RE = re.compile(
    r"(?:지원|신청)\s*자격|(?:대출|모집|선발|신청|지원)\s*대상"
)
_ELIGIBILITY_NEGATIVE_LINE_RE = re.compile(
    r"(?:지원\s*제한|제외|불가|할\s*수\s*없|자격\s*없)"
)
_ELIGIBILITY_ROSTER_HEADING_RE = re.compile(
    r"(?:모집|선발|신청|지원)\s*대상"
)
_BENEFIT_SECTION_HEADING_RE = re.compile(r"참\s*여\s*혜\s*택")
_BENEFIT_NEGATIVE_LINE_RE = re.compile(
    r"(?:지급\s*없|부여\s*없|제외|불가|받을\s*수\s*없)"
)
_CONTACT_CLAIM_RE = re.compile(r"(?:문의|연락처|전화)")
_CONTACT_UNAVAILABLE_RE = re.compile(
    r"(?:폐지|사용\s*중단|연결\s*불가|연락할\s*수\s*없)"
)
_ELIGIBILITY_SCOPE_NOISE = frozenset(
    {
        "신청",
        "신청할",
        "지원",
        "가능",
        "있습니다",
        "대상",
        "경우",
        "중인",
        "또는",
    }
)
_DOCUMENTED_CAPABILITY_CLAIM_RE = re.compile(
    r"(?P<action>탐색|경험|체험|확인|이수|준비)\s*(?:할|될)\s*수\s*있"
)
_DOCUMENTED_CAPABILITY_NEGATION_RE = re.compile(
    r"(?:탐색|경험|체험|확인|이수|준비)[^.!?\n]{0,32}?"
    r"(?:할\s*수\s*없|불가|금지|중단|제외|"
    r"(?:제공|운영|진행|지원|허용)하지(?:는)?\s*않)"
)
_DOCUMENTED_CAPABILITY_SCOPE_NOISE = frozenset(
    {
        "있는",
        "있습니다",
        "통해",
        "후에는",
        "여러",
    }
)
_ASSISTIVE_COPAY_PROCEDURE_RE = re.compile(
    r"(?:자부담금|개인부담금)[^.!?\n]{0,80}?납부"
    r"[^.!?\n]{0,80}?(?:증빙(?:자료)?[^.!?\n]{0,32}?확인|"
    r"확인[^.!?\n]{0,32}?증빙(?:자료)?)"
    r"[^.!?\n]{0,80}?지원"
)
_ASSISTIVE_COPAY_PROCEDURE_NEGATION_RE = re.compile(
    r"(?:납부|증빙|확인|지원)[^.!?\n]{0,24}?"
    r"(?:하지(?:는)?\s*않|할\s*수\s*없|불가|금지|제외|불필요)"
)
_GROUP_VISA_D2_RE = re.compile(
    r"(?<![A-Za-z0-9])D\s*[-‐‑‒–—]?\s*2(?!\d)",
    re.IGNORECASE,
)
_GROUP_VISA_SERVICE_NEGATION_RE = re.compile(
    r"(?:단체접수|서비스)[^.!?\n]{0,40}?"
    r"(?:제공하지(?:는)?\s*않|중단|폐지|이용할\s*수\s*없)"
)
_GROUP_VISA_FEE_NEGATION_RE = re.compile(
    r"현금[^.!?\n]{0,24}?(?:불가|가능하지(?:는)?\s*않)"
    r"|만원권[^.!?\n]{0,16}?미만"
    r"|정부초청장학생[^.!?\n]{0,40}?면제되지(?:는)?\s*않"
)
_LOCAL_RELATION_SCOPE_NOISE = frozenset(
    {
        "금액",
        "금액은",
        "분야",
        "분야의",
        "분야는",
        "지원",
        "지원액",
        "지원액은",
        "인당",
        "기준",
        "경우",
        "모두",
        "대상",
        "기간",
        "기간은",
        "신청",
        "신청기간",
        "내에서",
        "등록금",
    }
)


def _categorical_availability_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
    *,
    source_scope: str,
) -> bool:
    """Accept a positive availability paraphrase backed by an explicit list.

    Models commonly turn a categorical heading such as ``수납은행: A·B·C``
    into ``납부 가능한 은행은 A, B, C``.  The latter contains the token
    ``가능`` and therefore looks like a permission claim even though the
    source expresses the same fact as a list.  This narrow bridge requires at
    least three enumerated items, 75% item preservation, and a shared scope
    anchor; it does not relax ordinary permission claims such as
    ``신용카드 납부가 가능합니다``.
    """

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
    ):
        return False
    normalized_claim = normalize_text(claim)
    match = _CATEGORICAL_AVAILABILITY_RE.search(normalized_claim)
    if match is None:
        match = _CATEGORICAL_LOCATION_AVAILABILITY_RE.search(normalized_claim)
    if match is None:
        return False
    raw_parts = [
        part.strip()
        for part in _CATEGORICAL_ITEM_SPLIT_RE.split(match.group("items"))
        if part.strip()
    ]
    item_terms: list[str] = []
    for part in raw_parts:
        candidates = [
            term
            for term in tokenize(part, include_ngrams=False)
            if not term.isdigit() and term not in _CATEGORICAL_ITEM_NOISE
        ]
        if candidates:
            item_terms.append(candidates[0])
    item_terms = list(dict.fromkeys(item_terms))
    if len(item_terms) < 3:
        return False

    source_terms = set(
        tokenize(f"{source_scope} {source}", include_ngrams=False)
    )
    matched_items = sum(
        _claim_term_supported(term, source_terms) for term in item_terms
    )
    if matched_items * 4 < len(item_terms) * 3:
        return False
    semantic_source_terms = _semantic_scope_terms(
        f"{source_scope} {source}"
    )
    return any(
        _claim_term_supported(term, set(semantic_source_terms))
        for term in claim_atom.get("scope_terms", frozenset())
    )


def _named_procedure_availability_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
    *,
    source_scope: str,
) -> bool:
    """Accept availability implied by a named official procedure document.

    A guide headed ``BIDV ... 등록금 납부 방법`` directly entails that the
    named channel can be used for that procedure, even if it never repeats
    the modal phrase ``납부할 수 있습니다``.  Keep the bridge narrow: the
    claim must assert procedural availability, the evidence must contain a
    method/guide construction, and both sides must share a distinctive Latin
    product token such as ``BIDV``.  Explicit negative wording always wins.
    """

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
    ):
        return False
    normalized_claim = normalize_text(claim)
    normalized_source = normalize_text(f"{source_scope} {source}")
    if not _PROCEDURE_AVAILABILITY_CLAIM_RE.search(normalized_claim):
        return False
    if not _PROCEDURE_METHOD_RE.search(normalized_source):
        return False
    if _PROCEDURE_NEGATION_RE.search(normalized_source):
        return False

    def named_tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in _NAMED_PROCEDURE_TOKEN_RE.findall(value)
            if token.casefold() not in _NAMED_PROCEDURE_TOKEN_NOISE
        }

    shared_named_tokens = named_tokens(claim).intersection(
        named_tokens(f"{source_scope} {source}")
    )
    if not shared_named_tokens:
        return False
    source_terms = set(
        tokenize(f"{source_scope} {source}", include_ngrams=False)
    )
    action_terms = {
        term
        for term in tokenize(claim, include_ngrams=False)
        if any(marker in term for marker in ("납부", "신청", "결제", "송금", "이용"))
    }
    return any(_claim_term_supported(term, source_terms) for term in action_terms)


def _verification_availability_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
) -> bool:
    """Accept an official UI lookup instruction as availability evidence.

    University FAQs often express an available lookup as a terse path such as
    ``학생지원시스템 → 등록 → 납부확인 ... 확인``.  A generated paraphrase uses
    ``확인할 수 있습니다``, which looks like a permission relation even though
    the source has no modal verb.  Keep this bridge narrow: both sides need a
    UI marker, the same lookup action, every distinctive claim object, and all
    critical values in one local source clause.  Explicit unavailability still
    wins.
    """

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
        or not _VERIFICATION_UI_MARKER_RE.search(claim)
    ):
        return False

    action_matches = list(_VERIFICATION_AVAILABILITY_CLAIM_RE.finditer(claim))
    claim_scope = claim_atom.get("scope_terms", frozenset())
    actions = {
        match.group("action")
        for match in action_matches
        if _claim_term_supported(match.group("action"), set(claim_scope))
    }
    if not actions:
        return False

    def is_generic_scope_term(term: str) -> bool:
        variants = _term_variants(term)
        if variants.intersection(_VERIFICATION_SCOPE_NOISE):
            return True
        return any(
            variant.endswith(
                ("시스템", "포털", "홈페이지", "웹사이트", "사이트", "앱")
            )
            for variant in variants
        )

    distinctive_claim_terms = {
        term for term in claim_scope if not is_generic_scope_term(term)
    }
    if not distinctive_claim_terms:
        return False

    claim_values = extract_critical_values(claim)
    for clause in _relation_clauses(source):
        if (
            not _VERIFICATION_UI_MARKER_RE.search(clause)
            or not any(action in clause for action in actions)
            or _VERIFICATION_UNAVAILABLE_RE.search(clause)
        ):
            continue
        if any(
            operator == "denied"
            for _, operator, _ in _permission_relation_matches(clause)
        ):
            continue
        source_scope = _semantic_scope_terms(clause)
        if not _semantic_permission_scopes_match(claim_scope, source_scope):
            continue
        source_terms = set(source_scope)
        if not all(
            any(
                _term_variants(term).intersection(_term_variants(source_term))
                for source_term in source_terms
            )
            for term in distinctive_claim_terms
        ):
            continue
        if not claim_values.issubset(extract_critical_values(clause)):
            continue
        return True

    # Official notices often put the dated action label and its exact menu
    # path on adjacent lines: ``고지서출력: DATE`` followed by
    # ``학생지원시스템 → ... → 고지서출력``.  Bind only within this one
    # retrieved chunk, require the exact dated values and object on the dated
    # line, and independently require the same object/action on a UI-path
    # line.  This does not borrow a date from a different action.
    if not claim_values or _VERIFICATION_UNAVAILABLE_RE.search(source):
        return False

    def supports_distinctive_terms(line_scope: frozenset[str]) -> bool:
        """Match the claimed object, without reducing it to its action.

        ``고지서는`` may legitimately match the compound ``고지서출력`` in
        an official menu path.  The reverse containment is unsafe, however:
        a generic source action such as ``확인`` must not satisfy the more
        specific claimed object ``납부확인``.  Keep containment directional
        from the claim's object stem into the source term.
        """

        for claim_term in distinctive_claim_terms:
            claim_variants = _term_variants(claim_term)
            matched = False
            for source_term in line_scope:
                source_variants = _term_variants(source_term)
                if claim_variants.intersection(source_variants):
                    matched = True
                    break
                if any(
                    len(claim_variant) >= 2
                    and claim_variant in source_variant
                    for claim_variant in claim_variants
                    for source_variant in source_variants
                ):
                    matched = True
                    break
            if not matched:
                return False
        return True

    dated_action = False
    ui_path = False
    for raw_line in str(source or "").splitlines():
        line = normalize_text(raw_line)
        if not line or _VERIFICATION_UNAVAILABLE_RE.search(line):
            continue
        line_scope = _semantic_scope_terms(line)
        if not supports_distinctive_terms(line_scope):
            continue
        has_action = any(action in line for action in actions)
        if has_action and claim_values.issubset(extract_critical_values(line)):
            dated_action = True
        if has_action and _VERIFICATION_UI_MARKER_RE.search(line):
            ui_path = True
    return dated_action and ui_path


def _official_ui_application_path_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
) -> bool:
    """Accept a positive application claim backed by an exact UI path.

    A menu breadcrumb ending in an action (``... → 장학금신청 → ...에서
    신청``) is an imperative procedure even when it does not repeat the
    modal phrase ``신청할 수 있습니다``.  The bridge requires two exact,
    non-generic path labels and refuses any locally negated instruction.
    """

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
        or not _VERIFICATION_UI_MARKER_RE.search(claim)
    ):
        return False
    action_match = _UI_APPLICATION_CLAIM_RE.search(normalize_text(claim))
    if action_match is None:
        return False
    action = action_match.group("action")

    claim_terms = _semantic_scope_terms(claim)
    path_terms = {
        term
        for term in claim_terms
        if term not in _UI_APPLICATION_PATH_NOISE
        and len(term) >= 4
        and not any(
            term.endswith(suffix)
            for suffix in ("시스템", "포털", "홈페이지", "사이트", "메뉴")
        )
    }
    if len(path_terms) < 2:
        return False

    for raw_line in str(source or "").splitlines():
        line = normalize_text(raw_line)
        if (
            not line
            or not _VERIFICATION_UI_MARKER_RE.search(line)
            or action not in line
            or _PROCEDURE_NEGATION_RE.search(line)
        ):
            continue
        if any(
            operator == "denied"
            for _, operator, _ in _permission_relation_matches(line)
        ):
            continue
        source_terms = _semantic_scope_terms(line)
        exact_path_matches = {
            claim_term
            for claim_term in path_terms
            if any(
                _term_variants(claim_term).intersection(
                    _term_variants(source_term)
                )
                for source_term in source_terms
            )
        }
        if len(exact_path_matches) >= 2:
            return True
    return False


def _enumerated_eligibility_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
) -> bool:
    """Accept a paraphrase of an explicit enumerated eligibility section."""

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
        or not _ELIGIBILITY_AVAILABILITY_CLAIM_RE.search(
            normalize_text(claim)
        )
    ):
        return False

    normalized_source = str(source or "")
    roster_lines = [
        normalize_text(line)
        for line in normalized_source.splitlines()
        if _ELIGIBILITY_ROSTER_HEADING_RE.search(line)
    ]
    salient_permission = {
        term
        for term in claim_atom.get("scope_terms", frozenset())
        if term not in _ELIGIBILITY_SCOPE_NOISE and len(term) >= 2
    }
    for line in roster_lines:
        # Parenthetical/footnote text after a roster commonly lists explicit
        # exclusions. It must not satisfy a positive applicant scope merely
        # because the excluded noun occurs in the same line.
        positive_line = line
        if "제외" in line:
            exclusion_end = line.find("제외")
            cut_points = [
                index
                for marker in ("(", "※")
                if (index := line.rfind(marker, 0, exclusion_end)) >= 0
            ]
            if cut_points:
                positive_line = line[: min(cut_points)]
        positive_terms = _semantic_scope_terms(positive_line)
        matched_permission = {
            term
            for term in salient_permission
            if any(
                _term_variants(term).intersection(
                    _term_variants(source_term)
                )
                for source_term in positive_terms
            )
        }
        if (
            len(salient_permission) >= 2
            and len(matched_permission) >= 2
            and len(matched_permission) / len(salient_permission) >= 0.75
        ):
            return True

    salient_claim = {
        term
        for term in _semantic_scope_terms(claim)
        if term not in _ELIGIBILITY_SCOPE_NOISE and len(term) >= 2
    }
    if len(salient_claim) < 3:
        return False
    if _ELIGIBILITY_LIST_HEADING_RE.search(normalized_source):
        source_terms = set(tokenize(source, include_ngrams=False))
        matched = {
            term
            for term in salient_claim
            if _claim_term_supported(term, source_terms)
        }
        if len(matched) >= 3 and (
            len(matched) / len(salient_claim) >= 0.6
        ):
            return True

    # Some official notices express eligibility as a label followed by noun
    # phrases rather than repeating a modal predicate, for example
    # ``(대출대상) ... 대상학과 학부생`` or ``지원자격은 아래와 같습니다``.
    # Treat only nearby, non-negative lines as the positive roster.  This
    # preserves a natural ``지원할/받을 수 있습니다`` paraphrase without
    # allowing an exclusion or support-restriction line to prove eligibility.
    section_lines_remaining = 0
    for raw_line in str(source or "").splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if _ELIGIBILITY_SECTION_HEADING_RE.search(line):
            section_lines_remaining = 12
        elif section_lines_remaining <= 0:
            continue

        section_lines_remaining -= 1
        positive_line = line
        negative_match = _ELIGIBILITY_NEGATIVE_LINE_RE.search(line)
        if negative_match is not None:
            cut_points = [negative_match.start()]
            cut_points.extend(
                index
                for marker in ("(", "※")
                if (
                    index := line.rfind(
                        marker,
                        0,
                        negative_match.start(),
                    )
                )
                >= 0
            )
            positive_line = line[: min(cut_points)]
        positive_terms = set(
            tokenize(positive_line, include_ngrams=False)
        )
        matched = {
            term
            for term in salient_claim
            if _claim_term_supported(term, positive_terms)
        }
        if len(matched) >= 4 and (
            len(matched) / len(salient_claim) >= 0.7
        ):
            return True
    return False


def _labeled_participation_benefit_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
) -> bool:
    """Accept a faithful paraphrase of one explicit participation-benefit row."""

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
        or not re.search(r"(?:부여)?받을\s*수\s*있", claim)
        or _BENEFIT_SECTION_HEADING_RE.search(source) is None
    ):
        return False

    claim_terms = {
        term
        for term in _salient_permission_scope(
            _semantic_scope_terms(claim)
        )
        if len(term) >= 2
        and term not in {"참여자", "참여자는", "받을", "부여받을"}
    }
    if len(claim_terms) < 4:
        return False
    claim_numbers = set(
        re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", claim)
    )

    section_lines_remaining = 0
    for raw_line in str(source or "").splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if _BENEFIT_SECTION_HEADING_RE.search(line):
            section_lines_remaining = 8
        elif section_lines_remaining <= 0:
            continue
        section_lines_remaining -= 1
        if _BENEFIT_NEGATIVE_LINE_RE.search(line):
            continue
        line_numbers = set(
            re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", line)
        )
        if not claim_numbers.issubset(line_numbers):
            continue
        line_terms = set(tokenize(line, include_ngrams=False))
        matched = {
            term
            for term in claim_terms
            if _claim_term_supported(term, line_terms)
        }
        if len(matched) >= 4 and len(matched) / len(claim_terms) >= 0.6:
            return True
    return False


def _contact_phone_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
) -> bool:
    """Accept contact availability only from the exact owner/phone row."""

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
        or _CONTACT_CLAIM_RE.search(claim) is None
    ):
        return False
    claim_phones = {
        value
        for value in extract_critical_values(claim)
        if value.startswith("phone:")
    }
    if not claim_phones:
        return False
    owner_terms = {
        term
        for term in _semantic_scope_terms(claim)
        if any(marker in term for marker in ("학과", "학부", "대학원", "센터"))
    }
    for raw_line in str(source or "").splitlines():
        line = normalize_text(raw_line)
        if not line or _CONTACT_UNAVAILABLE_RE.search(line):
            continue
        if not claim_phones.issubset(extract_critical_values(line)):
            continue
        line_terms = _semantic_scope_terms(line)
        if owner_terms and not all(
            any(
                _term_variants(owner).intersection(_term_variants(source_term))
                for source_term in line_terms
            )
            for owner in owner_terms
        ):
            continue
        return True
    return False


def _documented_capability_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
    *,
    source_scope: str,
) -> bool:
    """Accept a strongly matched program activity stated as a noun phrase.

    Official programme notices often say ``현직자 피드백을 통한 직무체험``
    or ``진로포트폴리오 ... 직무 확인``. A natural answer turns that into
    ``체험할 수 있습니다`` or ``확인할 수 있습니다``. These are descriptive
    capabilities, not policy permissions. Keep the bridge fail-closed: only
    low-risk activity verbs qualify, explicit local negation wins, and at
    least 75% of a four-term claim scope must match exactly after Korean
    suffix normalization.
    """

    capability_match = _DOCUMENTED_CAPABILITY_CLAIM_RE.search(claim)
    if capability_match is None or _DOCUMENTED_CAPABILITY_NEGATION_RE.search(
        source
    ):
        return False

    action = capability_match.group("action")
    if action == "준비":
        compact_source = re.sub(r"\s+", "", source)
        if not all(
            marker in compact_source
            for marker in ("특강", "운영", "자격증", "취득", "지원")
        ):
            return False

    if action == "이수":
        if re.search(r"이수(?:기준|됨|자|하여|해야)", source) is None:
            return False
        claim_comparators = [
            atom
            for atom in _semantic_relation_atoms(claim)
            if atom.get("kind") == "comparator"
        ]
        source_comparators = [
            atom
            for atom in _semantic_relation_atoms(source)
            if atom.get("kind") == "comparator"
        ]
        if claim_atom.get("kind") == "comparator":
            return any(
                _local_relation_atom_matches(claim_atom, source_comparator)
                for source_comparator in source_comparators
            )
        if (
            claim_atom.get("kind") != "permission"
            or claim_atom.get("operator") != "allowed"
        ):
            return False
        return bool(claim_comparators) and all(
            any(
                _local_relation_atom_matches(
                    claim_comparator,
                    source_comparator,
                )
                for source_comparator in source_comparators
            )
            for claim_comparator in claim_comparators
        )

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
    ):
        return False

    claim_terms = {
        term
        for term in _semantic_scope_terms(claim)
        if term not in _DOCUMENTED_CAPABILITY_SCOPE_NOISE
    }
    if len(claim_terms) < 4:
        return False
    source_terms = _semantic_scope_terms(f"{source_scope} {source}")
    exact_matches = {
        claim_term
        for claim_term in claim_terms
        if any(
            _term_variants(claim_term).intersection(
                _term_variants(source_term)
            )
            for source_term in source_terms
        )
    }
    return (
        len(exact_matches) >= 4
        and len(exact_matches) / len(claim_terms)
        >= (0.65 if action == "준비" else 0.75)
    )


def _assistive_copay_support_procedure_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
    *,
    source_scope: str,
) -> bool:
    """Accept the notice's explicit pay-proof-support procedure paraphrase.

    ``지원받을 수 있다`` is parsed as a permission, while the notice states
    the same fact procedurally (pay the co-pay, verify proof, then provide the
    support).  Keep this bridge specific to the assistive-device programme and
    require the same ordered relation in both claim and one source sentence.
    """

    normalized_claim = normalize_text(claim)
    normalized_scope = normalize_text(f"{source_scope} {source}")
    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
        or "정보통신보조기기" not in normalized_claim.replace(" ", "")
        or "정보통신보조기기" not in normalized_scope.replace(" ", "")
        or _ASSISTIVE_COPAY_PROCEDURE_RE.search(normalized_claim) is None
        or _ASSISTIVE_COPAY_PROCEDURE_NEGATION_RE.search(normalized_claim)
    ):
        return False

    protected_source = _protect_dotted_dates(str(source or ""))
    for raw_segment in re.split(r"(?<=[.!?。！？])\s+|\n+", protected_source):
        segment = normalize_text(raw_segment.replace(_DATE_DOT_SENTINEL, "."))
        if (
            segment
            and _ASSISTIVE_COPAY_PROCEDURE_RE.search(segment)
            and not _ASSISTIVE_COPAY_PROCEDURE_NEGATION_RE.search(segment)
        ):
            return True
    return False


def _is_group_visa_scope(value: str) -> bool:
    normalized = normalize_text(value)
    compact = re.sub(r"\s+", "", normalized)
    return bool(
        _GROUP_VISA_D2_RE.search(normalized)
        or ("비자" in compact and "단체접수" in compact)
    )


def _group_visa_reservation_service_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
    *,
    source_scope: str,
) -> bool:
    """Accept the D-2 group-service definition plus mandatory reservation."""

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
    ):
        return False
    normalized_claim = normalize_text(claim)
    normalized_source = normalize_text(source)
    normalized_scope = normalize_text(f"{source_scope} {source}")
    compact_claim = re.sub(r"\s+", "", normalized_claim)
    compact_source = re.sub(r"\s+", "", normalized_source)
    if (
        _GROUP_VISA_D2_RE.search(normalized_claim) is None
        or not _is_group_visa_scope(normalized_scope)
        or "단체접수" not in compact_claim
        or "단체접수" not in compact_source
        or "출입국을대신방문" not in compact_claim
        or "서류를제출" not in compact_claim
        or "사전예약후이용할수있" not in compact_claim
        or "출입국을대신방문" not in compact_source
        or "서류를제출" not in compact_source
        or re.search(r"사전예약없이[^.!?]{0,32}불가능", compact_source)
        is None
        or re.search(r"반드시예약후[^.!?]{0,20}방문", compact_source)
        is None
        or _GROUP_VISA_SERVICE_NEGATION_RE.search(normalized_source)
    ):
        return False
    return True


def _group_visa_fee_payment_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
    *,
    source_scope: str,
) -> bool:
    """Accept the exact D-2 fee, cash denomination, and GKS waiver row."""

    if (
        claim_atom.get("kind") != "permission"
        or claim_atom.get("operator") != "allowed"
    ):
        return False
    normalized_claim = normalize_text(claim)
    normalized_source = normalize_text(source)
    normalized_scope = normalize_text(f"{source_scope} {source}")
    compact_claim = re.sub(r"\s+", "", normalized_claim)
    compact_source = re.sub(r"\s+", "", normalized_source)
    if (
        not _is_group_visa_scope(normalized_scope)
        or "수수료" not in compact_claim
        or "현금" not in compact_claim
        or "만원권" not in compact_claim
        or "정부초청장학생" not in compact_claim
        or "장학증서" not in compact_claim
        or "면제" not in compact_claim
        or re.search(r"현금(?:만)?가능", compact_source) is None
        or re.search(r"만원권이상[^.!?]{0,16}준비", compact_source) is None
        or re.search(
            r"정부초청장학생[^.!?]{0,32}장학증서[^.!?]{0,24}면제",
            compact_source,
        )
        is None
        or _GROUP_VISA_FEE_NEGATION_RE.search(normalized_source)
    ):
        return False
    return True


def _group_visa_multi_round_audience_supported(
    claim: str,
    source: str,
    *,
    source_scope: str,
) -> bool:
    """Accept an exact 1·2차 versus 3차 audience summary of the visa table."""

    if not _is_group_visa_scope(f"{source_scope} {source}"):
        return False
    compact_claim = re.sub(r"\s+", "", normalize_text(claim))
    if re.search(
        r"1차와2차접수대상[^.!?]{0,48}?신입생[^.!?]{0,24}?재학생"
        r"[^.!?]{0,24}?수료생[^.!?]{0,24}?제한(?:이)?없",
        compact_claim,
    ) is None or re.search(
        r"3차접수대상[^.!?]{0,32}?재학생[^.!?]{0,24}?수료생",
        compact_claim,
    ) is None:
        return False

    rows_by_round: dict[int, str] = {}
    for raw_line in str(source or "").splitlines():
        line = normalize_text(raw_line)
        rounds = {
            int(value.split(":", 1)[1])
            for value in extract_critical_values(line)
            if value.startswith("round:")
        }
        if len(rounds) == 1:
            rows_by_round[next(iter(rounds))] = re.sub(r"\s+", "", line)
    if not all(round_number in rows_by_round for round_number in (1, 2, 3)):
        return False
    for round_number in (1, 2):
        row = rows_by_round[round_number]
        if not all(
            marker in row
            for marker in ("제한없음", "신입생", "재학생", "수료생")
        ):
            return False
    third_row = rows_by_round[3]
    return (
        "재학생" in third_row
        and "수료생" in third_row
        and "신입생" not in third_row
    )


def _local_relation_atom_matches(
    claim_atom: dict[str, Any],
    source_atom: dict[str, Any],
) -> bool:
    if (
        claim_atom.get("kind") != source_atom.get("kind")
        or claim_atom.get("operator") != source_atom.get("operator")
        or claim_atom.get("modality", "asserted")
        != source_atom.get("modality", "asserted")
    ):
        return False
    if claim_atom.get("kind") == "comparator":
        return claim_atom.get("value_key") == source_atom.get("value_key")
    if claim_atom.get("kind") == "temporal_boundary":
        claim_values = set(str(claim_atom.get("value_key") or "").split("|"))
        source_values = set(
            str(source_atom.get("value_key") or "").split("|")
        )
        return bool(claim_values) and claim_values.issubset(source_values)
    return False


def _local_relation_atoms_match(
    claim_atom: dict[str, Any],
    source_atoms: list[dict[str, Any]],
    *,
    row_values: frozenset[str] = frozenset(),
) -> bool:
    """Match one row, allowing a deadline's date and time to be split.

    Parsed schedule rows often yield ``8월 31일까지`` and ``18:00까지`` as
    separate end-boundary atoms even though both values belong to the same
    row.  Aggregate only same-row atoms with the same operator and modality;
    comparator values remain single-atom matches so neighboring table values
    cannot be swapped.
    """

    if any(
        _local_relation_atom_matches(claim_atom, source_atom)
        for source_atom in source_atoms
    ):
        return True
    if claim_atom.get("kind") != "temporal_boundary":
        return False
    claim_values = set(str(claim_atom.get("value_key") or "").split("|"))
    if not claim_values:
        return False
    source_values: set[str] = set()
    for source_atom in source_atoms:
        if (
            source_atom.get("kind") == "temporal_boundary"
            and source_atom.get("operator") == claim_atom.get("operator")
            and source_atom.get("modality", "asserted")
            == claim_atom.get("modality", "asserted")
        ):
            source_values.update(
                str(source_atom.get("value_key") or "").split("|")
            )
    if source_values:
        source_values.update(
            value
            for value in row_values
            if value.startswith(("month_day:", "time_minutes:"))
        )
    return claim_values.issubset(source_values)


def _explicit_local_relation_supported(
    claim: str,
    source: str,
    claim_atom: dict[str, Any],
    *,
    source_scope: str,
) -> bool:
    """Recover relations whose subject and value occupy separate table cells.

    This is intentionally row-local for the operator and value.  It may use
    the document only to confirm the audience of a dated procedure, never to
    borrow a number from another row.
    """

    kind = claim_atom.get("kind")
    if kind not in {"comparator", "temporal_boundary"}:
        return False
    claim_full_scope = _semantic_scope_terms(claim)
    source_full_scope = _semantic_scope_terms(f"{source_scope} {source}")

    local_rows: list[tuple[str, frozenset[str]]] = []
    active_rounds: frozenset[str] = frozenset()
    for raw_line in str(source or "").splitlines():
        protected = _protect_dotted_dates(raw_line)
        for segment in re.split(r"(?<=[.!?。！？])\s+", protected):
            line = normalize_text(segment.replace(_DATE_DOT_SENTINEL, "."))
            if line:
                explicit_rounds = frozenset(
                    value
                    for value in _semantic_scope_values(line)
                    if value.startswith("round:")
                )
                if explicit_rounds:
                    active_rounds = explicit_rounds
                local_rows.append((line, active_rounds))

    claim_rounds = frozenset(
        value
        for value in claim_atom.get("scope_values", frozenset())
        if value.startswith("round:")
    )
    for line, local_rounds in local_rows:
        if (
            claim_rounds
            and local_rounds
            and claim_rounds.isdisjoint(local_rounds)
        ):
            continue
        local_values = _semantic_scope_values(f"{source_scope} {line}")
        # Date/month-day values are already bound by the row-local relation
        # atom below. Requiring them again as scope breaks abbreviated table
        # endpoints such as ``2026. 7. 17. ~ 7. 26.`` because only the first
        # endpoint carries the year. Keep only hierarchy here; the explicit
        # active-round check above prevents borrowing a 2차 row for 1차.
        hierarchical_claim_values = frozenset(
            value
            for value in claim_atom.get("scope_values", frozenset())
            if value.startswith(
                ("academic_year:", "calendar_year:", "semester:", "round:")
            )
        )
        if not hierarchical_claim_values.issubset(local_values):
            continue
        if not _local_relation_atoms_match(
            claim_atom,
            _semantic_relation_atoms(line),
            row_values=extract_critical_values(line),
        ):
            continue

        line_scope = _semantic_scope_terms(line)
        if _semantic_scopes_conflict(claim_full_scope, line_scope):
            continue
        if kind == "comparator":
            salient_claim = {
                term
                for term in claim_atom.get("scope_terms", frozenset())
                if term not in _LOCAL_RELATION_SCOPE_NOISE
            }
            if salient_claim and not all(
                _claim_term_supported(term, set(line_scope))
                for term in salient_claim
            ):
                continue
            if _semantic_scopes_match(
                claim_atom.get("scope_terms", frozenset()), line_scope
            ):
                return True
            continue

        # A dated procedure must have its action in the same row.  Audience
        # terms may come from the rest of the same notice (for example the
        # heading says 국가유공자 while the row says 신청기간).
        action_terms = (
            "신청",
            "지원",
            "접수",
            "납부",
            "등록",
            "수강",
            "휴학",
            "복학",
            "출력",
        )
        if not any(action in claim and action in line for action in action_terms):
            continue
        if _semantic_scopes_match(claim_full_scope, line_scope):
            return True
        distinctive_claim = {
            term
            for term in claim_full_scope
            if term not in _LOCAL_RELATION_SCOPE_NOISE
        }
        matched_context = {
            term
            for term in distinctive_claim
            if _claim_term_supported(term, set(source_full_scope))
        }
        if distinctive_claim and len(matched_context) >= min(
            2, len(distinctive_claim)
        ):
            return True
    return False


def _semantic_relation_mismatch(
    claim: str,
    source: str,
    *,
    source_scope: str = "",
) -> bool:
    """Return true unless each high-risk relation is explicitly supported."""

    claim_atoms = _semantic_relation_atoms(claim)
    if not claim_atoms:
        return False
    # Table extraction can place a unique round/year/semester label in an
    # adjacent cell. Source-wide values are used only through the unique-value
    # inheritance rule in _semantic_atom_with_context.
    context_scope_values = _semantic_scope_values(
        f"{source_scope} {source}"
    )
    context_scope_terms = _semantic_scope_terms(source_scope)
    source_atoms = [
        _semantic_atom_with_context(
            atom,
            context_scope_values,
            context_scope_terms,
        )
        for atom in _semantic_relation_atoms(source)
    ]
    for claim_atom in claim_atoms:
        if _coordinated_direction_supported(claim_atom, source_atoms):
            continue
        bound_atoms = [
            source_atom
            for source_atom in source_atoms
            if _semantic_atom_compatible(
                claim_atom,
                source_atom,
                require_same_operator=False,
            )
        ]
        # Inspect all same-scope statements before accepting a compatible one.
        # This prevents an unscoped answer from citing an old rule while the
        # same chunk also states a conflicting current rule.
        conflicting_atoms = [
            source_atom
            for source_atom in bound_atoms
            if (
            _semantic_operators_conflict(
                claim_atom["kind"],
                claim_atom["operator"],
                source_atom["operator"],
            )
            and not (
                claim_atom["kind"] == "temporal_boundary"
                and len(
                    set(str(source_atom.get("value_key") or "").split("|"))
                )
                > len(
                    set(str(claim_atom.get("value_key") or "").split("|"))
                )
            )
            )
        ]
        if conflicting_atoms:
            # Hyphen-range table rows are parsed as a date cell plus a label
            # cell, so the raw atom can be unscoped.  Prefer an exact
            # same-row/same-subject match over an opposite operator from a
            # different unscoped row.  Explicitly scoped conflicts still win.
            table_row_disambiguated = bool(
                claim_atom["kind"] == "temporal_boundary"
                and all(
                    not source_atom.get("scope_terms")
                    for source_atom in conflicting_atoms
                )
                and _explicit_local_relation_supported(
                    claim,
                    source,
                    claim_atom,
                    source_scope=source_scope,
                )
            )
            permission_scope_disambiguated = False
            if claim_atom["kind"] == "permission":
                same_operator_atoms = [
                    source_atom
                    for source_atom in bound_atoms
                    if source_atom["operator"] == claim_atom["operator"]
                ]
                permission_scope_disambiguated = bool(
                    any(
                        _exact_semantic_scope_overlap(
                            claim_atom["scope_terms"],
                            source_atom["scope_terms"],
                        )
                        >= MIN_CLAIM_ANCHORS
                        for source_atom in same_operator_atoms
                    )
                    and all(
                        _exact_semantic_scope_overlap(
                            claim_atom["scope_terms"],
                            source_atom["scope_terms"],
                        )
                        < MIN_CLAIM_ANCHORS
                        for source_atom in conflicting_atoms
                    )
                )
            group_visa_reservation_disambiguated = (
                _group_visa_reservation_service_supported(
                    claim,
                    source,
                    claim_atom,
                    source_scope=source_scope,
                )
            )
            if not (
                table_row_disambiguated
                or permission_scope_disambiguated
                or group_visa_reservation_disambiguated
            ):
                return True
        compatible = any(
            _semantic_atom_compatible(
                claim_atom,
                source_atom,
                require_same_operator=True,
            )
            for source_atom in source_atoms
        )
        if compatible:
            continue
        if _categorical_availability_supported(
            claim,
            source,
            claim_atom,
            source_scope=source_scope,
        ):
            continue
        if _named_procedure_availability_supported(
            claim,
            source,
            claim_atom,
            source_scope=source_scope,
        ):
            continue
        if _verification_availability_supported(claim, source, claim_atom):
            continue
        if _official_ui_application_path_supported(
            claim,
            source,
            claim_atom,
        ):
            continue
        if _enumerated_eligibility_supported(claim, source, claim_atom):
            continue
        if _labeled_participation_benefit_supported(
            claim,
            source,
            claim_atom,
        ):
            continue
        if _contact_phone_supported(claim, source, claim_atom):
            continue
        if _group_visa_reservation_service_supported(
            claim,
            source,
            claim_atom,
            source_scope=source_scope,
        ):
            continue
        if _group_visa_fee_payment_supported(
            claim,
            source,
            claim_atom,
            source_scope=source_scope,
        ):
            continue
        if _assistive_copay_support_procedure_supported(
            claim,
            source,
            claim_atom,
            source_scope=source_scope,
        ):
            continue
        if _documented_capability_supported(
            claim,
            source,
            claim_atom,
            source_scope=source_scope,
        ):
            continue
        if _explicit_local_relation_supported(
            claim,
            source,
            claim_atom,
            source_scope=source_scope,
        ):
            continue
        # Permission, comparator, and direction claims are high-risk because
        # lexical overlap cannot distinguish allowance from prohibition,
        # boundary reversal, or an unrelated/negated direction. Require an
        # explicit same-operator atom with matching local payload.
        return True
    return False


def _result_scope_text(result: dict[str, Any]) -> str:
    """Collect document and section metadata that scopes a retrieved chunk."""

    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    values: list[str] = []

    def append_value(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                append_value(item)
            return
        normalized = normalize_text(str(value or ""))
        if normalized:
            values.append(normalized)

    for value in (
        result.get("source_title"),
        result.get("file_name"),
        result.get("section_path"),
        metadata.get("source_title"),
        metadata.get("file_name"),
        metadata.get("section_path"),
    ):
        append_value(value)

    for location_group in (
        result.get("locations"),
        metadata.get("locations"),
    ):
        if not isinstance(location_group, list):
            continue
        for location in location_group:
            if not isinstance(location, dict):
                continue
            append_value(location.get("section_path"))

    return " ".join(dict.fromkeys(values))


def _critical_value_support(
    claim: str,
    claim_values: frozenset[str],
    result: dict[str, Any],
) -> tuple[bool, set[str]]:
    if not claim_values:
        return True, set()

    source = result_source_text(result)
    scope_text = _result_scope_text(result)
    scope_values = {
        value
        for value in extract_critical_values(scope_text)
        if value.startswith(
            ("academic_year:", "calendar_year:", "semester:", "round:")
        )
    }
    if _group_visa_multi_round_audience_supported(
        claim,
        source,
        source_scope=scope_text,
    ):
        table_values = frozenset(
            set(extract_critical_values(source)) | scope_values
        )
        if claim_values.issubset(table_values):
            return True, set()
    budget_claim_facets = _budget_breakdown_claim_facets(claim)
    if (
        budget_claim_facets
        and budget_claim_facets <= _budget_breakdown_facets(source)
    ):
        table_values = frozenset(
            set(extract_critical_values(source)) | scope_values
        )
        if claim_values.issubset(table_values):
            return True, set()
    best_values: frozenset[str] = frozenset()
    best_intersection = -1
    for unit in _evidence_units(source):
        unit_values = frozenset(
            set(extract_critical_values(unit)) | scope_values
        )
        if claim_values.issubset(unit_values):
            return True, set()
        intersection = len(claim_values.intersection(unit_values))
        if (
            intersection > best_intersection
            or (
                intersection == best_intersection
                and len(unit_values) < len(best_values)
            )
        ):
            best_values = unit_values
            best_intersection = intersection

    # A single answer sentence may faithfully summarize two separate rows of
    # the same academic-calendar item.  Combine only multi-range date claims,
    # and only rows whose subject scope independently matches the claim.  This
    # keeps a 휴·복학 row from borrowing dates from 수강신청 or another table
    # row merely because both live in one chunk.
    full_dates = {
        value for value in claim_values if value.startswith("date:")
    }
    if len(full_dates) >= 3:
        claim_scope = _semantic_scope_terms(claim)
        salient_claim_scope = {
            term
            for term in claim_scope
            if term not in _LOCAL_RELATION_SCOPE_NOISE
            and term not in {"또는", "이거나"}
        }
        aggregated_values = set(scope_values)
        for raw_line in str(source or "").splitlines():
            line = normalize_text(raw_line)
            if not line:
                continue
            line_values = frozenset(
                set(extract_critical_values(line)) | scope_values
            )
            if not full_dates.intersection(line_values):
                continue
            if _scope_conflicts(claim_values, line_values):
                continue
            line_scope = _semantic_scope_terms(line)
            if _semantic_scopes_conflict(claim_scope, line_scope):
                continue
            if not _semantic_scopes_match(claim_scope, line_scope):
                continue
            if salient_claim_scope and not all(
                _claim_term_supported(term, set(line_scope))
                for term in salient_claim_scope
            ):
                continue
            aggregated_values.update(line_values)
        aggregated = frozenset(aggregated_values)
        if claim_values.issubset(aggregated):
            return True, set()
        intersection = len(claim_values.intersection(aggregated))
        if intersection > best_intersection:
            best_values = aggregated

        # A model may compact identical schedules for explicitly enumerated
        # audiences (학부, 대학원, 타대생) into one sentence.  Accept that
        # compaction only when every named audience independently has every
        # claimed date; a date present for just one audience is insufficient.
        audience_markers = ("학부", "대학원", "타대생")
        claimed_audiences = _semantic_scope_markers(
            claim_scope,
            audience_markers,
        )
        if len(claimed_audiences) >= 2:
            base_scope = {
                term
                for term in salient_claim_scope
                if not _semantic_scope_markers(
                    frozenset({term}),
                    audience_markers,
                )
            }
            values_by_audience: dict[str, set[str]] = {
                audience: set(scope_values)
                for audience in claimed_audiences
            }
            for raw_line in str(source or "").splitlines():
                line = normalize_text(raw_line)
                if not line:
                    continue
                line_scope = _semantic_scope_terms(line)
                line_audiences = _semantic_scope_markers(
                    line_scope,
                    audience_markers,
                ).intersection(claimed_audiences)
                if not line_audiences:
                    continue
                if base_scope and not all(
                    _claim_term_supported(term, set(line_scope))
                    for term in base_scope
                ):
                    continue
                line_values = extract_critical_values(line)
                for audience in line_audiences:
                    values_by_audience[audience].update(line_values)
            if all(
                claim_values.issubset(values_by_audience[audience])
                for audience in claimed_audiences
            ):
                return True, set()
    return False, set(claim_values - best_values)


def _is_abstention_claim(claim: str) -> bool:
    normalized = normalize_text(claim)
    return bool(ABSTENTION_RE.search(normalized))


def attribute_claim(
    claim: str,
    results: list[dict[str, Any]],
    *,
    semantic_guard: bool = True,
) -> dict[str, Any]:
    claim_terms = {
        term
        for term in tokenize(claim, include_ngrams=False)
        if not term.isdigit()
    }
    claim_values = extract_critical_values(claim)
    matches: list[tuple[float, int, dict[str, Any]]] = []
    best_score = -1.0
    best_missing_values: set[str] = set(claim_values)
    saw_semantic_mismatch = False
    normalized_claim = normalize_text(claim)

    if _is_abstention_claim(claim):
        return {
            "text": claim,
            "supported": False,
            "confidence": 0.0,
            "source_ids": [],
            "source_numbers": [],
            "citations": [],
            "validation_reason": "model_abstention",
            "missing_critical_values": [],
            "best_score": 0.0,
        }

    for source_number, result in enumerate(results, start=1):
        source = result_source_text(result)
        source_scope = _result_scope_text(result)
        source_terms = set(
            tokenize(f"{source_scope} {source}", include_ngrams=False)
        )
        matched_terms = {
            claim_term
            for claim_term in claim_terms
            if _claim_term_supported(claim_term, source_terms)
        }
        lexical_score = (
            len(matched_terms) / len(claim_terms)
            if claim_terms
            else 0.0
        )
        normalized_source = normalize_text(source)
        exact = bool(
            normalized_claim
            and normalized_claim[:80] in normalized_source
        )
        support_score = 1.0 if exact else lexical_score
        rank_bonus = 0.05 / source_number
        ranked_score = support_score + rank_bonus
        facts_supported, missing_values = _critical_value_support(
            claim,
            claim_values,
            result,
        )
        required_anchors = (
            1
            if claim_values
            else min(MIN_CLAIM_ANCHORS, len(claim_terms))
        )
        temporal_relation = any(
            atom.get("kind") == "temporal_boundary"
            for atom in _semantic_relation_atoms(claim)
        )
        dense_temporal_values = sum(
            value.startswith(
                ("date:", "month_day:", "round:", "time_minutes:")
            )
            for value in claim_values
        ) >= 4
        temporal_paraphrase = bool(
            facts_supported
            and temporal_relation
            and (
                (len(matched_terms) >= 3 and lexical_score >= 0.30)
                or (
                    dense_temporal_values
                    and len(matched_terms) >= 2
                    and lexical_score >= 0.20
                )
            )
        )
        contact_phone_paraphrase = bool(
            facts_supported
            and any(
                _contact_phone_supported(
                    claim,
                    source,
                    claim_atom,
                )
                for claim_atom in _semantic_relation_atoms(claim)
            )
        )
        lexical_supported = (
            exact
            or (not claim_terms and facts_supported)
            or temporal_paraphrase
            or contact_phone_paraphrase
            or (
                lexical_score >= CLAIM_SUPPORT_THRESHOLD
                and len(matched_terms) >= required_anchors
            )
        )

        if (
            support_score > best_score
            or (
                support_score == best_score
                and len(missing_values) < len(best_missing_values)
            )
        ):
            best_score = support_score
            best_missing_values = missing_values
        semantic_mismatch = bool(
            semantic_guard
            and facts_supported
            and _semantic_relation_mismatch(
                claim,
                source,
                source_scope=source_scope,
            )
        )
        if semantic_mismatch:
            saw_semantic_mismatch = True
        elif lexical_supported and facts_supported:
            matches.append((ranked_score, source_number, result))

    matches.sort(key=lambda item: item[0], reverse=True)
    top_matches = matches[:2]
    validation_reason = "supported" if top_matches else "low_lexical_overlap"
    if not top_matches and saw_semantic_mismatch:
        validation_reason = "semantic_relation_mismatch"
    elif not top_matches and best_missing_values:
        validation_reason = "critical_value_mismatch"
    return {
        "text": claim,
        "supported": bool(top_matches),
        "confidence": round(min(0.99, top_matches[0][0]) if top_matches else 0.0, 3),
        "source_ids": [match[2]["chunk_id"] for match in top_matches],
        "source_numbers": [match[1] for match in top_matches],
        "citations": [
            citation
            for match in top_matches
            for citation in claim_citations_for_result(claim, match[2])
        ],
        "validation_reason": validation_reason,
        "missing_critical_values": (
            [] if top_matches else sorted(best_missing_values)
        ),
        "best_score": round(max(0.0, min(1.0, best_score)), 3),
    }


def split_draft_claims(draft_answer: str) -> list[str]:
    claims: list[str] = []
    for raw_line in str(draft_answer or "").splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if _is_abstention_claim(line):
            claims.append(line)
        else:
            # Generated drafts may contain concise but complete facts (for example,
            # "학부 등록금은 동결되었습니다."). The 20-character extractive noise
            # filter must not erase those claims before attribution.
            claims.extend(split_candidate_sentences(line, minimum_chars=8))
    return claims


def build_rag_response(
    question: str,
    results: list[dict[str, Any]],
    draft_answer: str | None = None,
    generator: str = "extractive",
) -> dict[str, Any]:
    if not results:
        return {
            "answer": "검색된 근거 문서가 없습니다. 기관 범위나 질문 표현을 바꿔 다시 검색해 주세요.",
            "cited_answer": "검색된 근거 문서가 없습니다. 기관 범위나 질문 표현을 바꿔 다시 검색해 주세요.",
            "claims": [],
            "citations": [],
            "generator": generator,
            "postprocessing": {
                "claim_limit": MAX_CLAIMS,
                "input_claim_count": 0,
                "processed_claim_count": 0,
                "truncated_claim_count": 0,
                "claims_truncated": False,
            },
        }

    input_claim_texts = split_draft_claims(draft_answer) if draft_answer else []
    if not input_claim_texts:
        input_claim_texts = select_answer_claims(question, results)
    claim_texts = input_claim_texts[:MAX_CLAIMS]
    truncated_claim_count = max(0, len(input_claim_texts) - len(claim_texts))
    postprocessing = {
        "claim_limit": MAX_CLAIMS,
        "input_claim_count": len(input_claim_texts),
        "processed_claim_count": len(claim_texts),
        "truncated_claim_count": truncated_claim_count,
        "claims_truncated": truncated_claim_count > 0,
    }

    claims = [attribute_claim(claim, results) for claim in claim_texts]
    supported_claims = [claim for claim in claims if claim["supported"]]

    if not supported_claims:
        if claims and all(
            claim.get("validation_reason") == "model_abstention"
            for claim in claims
        ):
            message = (
                "제공된 검색 근거에서 질문에 답할 내용을 "
                "확인할 수 없습니다."
            )
            return {
                "answer": message,
                "cited_answer": message,
                "claims": claims,
                "citations": [],
                "draft_answer": draft_answer,
                "generator": generator,
                "postprocessing": postprocessing,
            }
        return {
            "answer": "검색 결과는 있으나 답변 문장을 지지하는 근거를 충분히 확인하지 못했습니다.",
            "cited_answer": "검색 결과는 있으나 답변 문장을 지지하는 근거를 충분히 확인하지 못했습니다.",
            "claims": claims,
            "citations": [],
            "draft_answer": draft_answer,
            "generator": generator,
            "postprocessing": postprocessing,
        }

    answer_lines = [claim["text"] for claim in supported_claims]
    cited_lines = [
        f"{claim['text']} "
        f"{' '.join(f'[{source_number}]' for source_number in claim['source_numbers'])}"
        for claim in supported_claims
    ]

    return {
        "answer": "\n".join(f"- {line}" for line in answer_lines),
        "cited_answer": "\n".join(f"- {line}" for line in cited_lines),
        "claims": claims,
        "citations": [
            {
                **citation,
                "claim_index": claim_index,
                "claim_text": claim["text"],
            }
            for claim_index, claim in enumerate(supported_claims)
            for citation in claim.get("citations") or []
        ],
        "draft_answer": draft_answer,
        "generator": generator,
        "postprocessing": postprocessing,
    }


def number_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**result, "source_number": index} for index, result in enumerate(results, start=1)]


def create_hybrid_retriever(
    index_path: Path,
    dense_path: Path,
) -> tuple[HybridRetriever | None, str | None]:
    """Load a revision-matched dense lane and always retain BM25 fallback."""

    dense = None
    warning = None
    if dense_path.is_file():
        try:
            candidate = DenseIndex.load(dense_path)
            connection = sqlite3.connect(str(index_path))
            try:
                revision = index_metadata(connection).get("corpus_revision")
            finally:
                connection.close()
            if not revision or candidate.corpus_revision != revision:
                warning = "dense_corpus_revision_mismatch"
            else:
                dense = candidate
        except Exception as exc:
            warning = f"dense_load_failed:{type(exc).__name__}"
    else:
        warning = "dense_index_missing"

    # A missing or stale dense lane should use the same BM25 path as the
    # document benchmark.  A single-lane HybridRetriever otherwise changes
    # candidate depth and reranking without providing hybrid retrieval.
    if dense is None:
        return None, warning

    def bm25_lane(
        *,
        query: str,
        top_k: int,
        institution: str | None,
    ) -> list[dict[str, Any]]:
        return search_bm25_candidates(
            index_path,
            query,
            top_k,
            institution,
            preview_chars=source_chars(),
            include_text=True,
        )

    return (
        HybridRetriever(
            bm25_search=bm25_lane,
            dense_index=dense,
        ),
        warning,
    )


def create_retrieval_modes(
    index_path: Path,
    learned_root: Path,
    profile: str,
    *,
    query_embedder_cache: dict[tuple[Any, ...], Any] | None = None,
) -> Mapping[str, RetrievalModeState]:
    """Build independently selectable BM25, dense, and hybrid lanes."""

    modes: dict[str, RetrievalModeState] = {
        "bm25": RetrievalModeState(
            id="bm25",
            label=RETRIEVAL_MODE_LABELS["bm25"],
            retriever=None,
        )
    }

    def bm25_lane(
        *,
        query: str,
        top_k: int,
        institution: str | None,
    ) -> list[dict[str, Any]]:
        return search_bm25_candidates(
            index_path,
            query,
            top_k,
            institution,
            preview_chars=source_chars(),
            include_text=True,
        )

    for family, directory_name in LEARNED_DENSE_ARTIFACTS.items():
        artifact_dir = learned_dense_artifact_path(
            learned_root,
            profile,
            directory_name,
        )
        learned_index = None
        warning = None
        if not artifact_dir.is_dir():
            warning = "learned_dense_index_missing"
        else:
            try:
                learned_index = LearnedDenseIndex(
                    artifact_dir,
                    source_index=index_path,
                    row_loader=lambda chunk_ids, source=index_path: (
                        load_chunks_by_ids(source, chunk_ids)
                    ),
                )
                if query_embedder_cache is not None:
                    share_learned_dense_embedder(
                        learned_index,
                        query_embedder_cache,
                    )
            except Exception as exc:
                warning = f"learned_dense_load_failed:{type(exc).__name__}:{exc}"

        dense_id = f"{family}_dense"
        hybrid_id = f"{family}_hybrid"
        dense_retriever = (
            HybridRetriever(
                bm25_search=None,
                dense_index=learned_index,
                candidate_multiplier=1,
                reranker=None,
            )
            if learned_index is not None
            else None
        )
        hybrid_retriever = (
            HybridRetriever(
                bm25_search=bm25_lane,
                dense_index=learned_index,
            )
            if learned_index is not None
            else None
        )
        modes[dense_id] = RetrievalModeState(
            id=dense_id,
            label=RETRIEVAL_MODE_LABELS[dense_id],
            retriever=dense_retriever,
            warning=warning,
            learned_index=learned_index,
        )
        modes[hybrid_id] = RetrievalModeState(
            id=hybrid_id,
            label=RETRIEVAL_MODE_LABELS[hybrid_id],
            retriever=hybrid_retriever,
            warning=warning,
            learned_index=learned_index,
        )
    return MappingProxyType(modes)


def default_retrieval_mode(target: ParserIndexTarget) -> str:
    requested = configured_retrieval_mode()
    state = target.retrieval_modes.get(requested)
    if state is not None and state.ready:
        return requested
    return "bm25"


def resolve_retrieval_mode(
    target: ParserIndexTarget,
    value: Any = None,
) -> RetrievalModeState:
    """Validate a request mode and fail closed when its artifact is stale."""

    if not target.retrieval_modes:
        if value not in (None, "", "bm25"):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_retrieval_mode",
                "이 서버에는 선택형 검색 방식이 구성되지 않았습니다.",
            )
        return RetrievalModeState(
            id="legacy_hybrid" if target.retriever is not None else "bm25",
            label=(
                "Legacy hybrid"
                if target.retriever is not None
                else RETRIEVAL_MODE_LABELS["bm25"]
            ),
            retriever=target.retriever,
            warning=target.warning,
        )

    if value is None or (isinstance(value, str) and not value.strip()):
        mode = default_retrieval_mode(target)
    elif isinstance(value, str):
        mode = value.strip().lower()
    else:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_retrieval_mode",
            "retrieval_mode 형식이 올바르지 않습니다.",
        )
    state = target.retrieval_modes.get(mode)
    if state is None:
        choices = ", ".join(RETRIEVAL_MODE_ORDER)
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "invalid_retrieval_mode",
            f"retrieval_mode은 {choices} 중 하나를 선택해 주세요.",
        )
    if not state.ready:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "retrieval_mode_unavailable",
            f"{state.label} 검색 인덱스를 사용할 수 없습니다. ({state.warning})",
        )
    return state


def retrieval_mode_summaries(
    target: ParserIndexTarget,
) -> list[dict[str, Any]]:
    modes = target.retrieval_modes
    if not modes:
        modes = {
            "bm25": RetrievalModeState(
                id="bm25",
                label=RETRIEVAL_MODE_LABELS["bm25"],
                retriever=None,
            )
        }
    summaries: list[dict[str, Any]] = []
    for mode in RETRIEVAL_MODE_ORDER:
        state = modes.get(mode)
        if state is None:
            summaries.append(
                {
                    "id": mode,
                    "label": RETRIEVAL_MODE_LABELS[mode],
                    "ready": False,
                    "reason": "not_configured",
                }
            )
            continue
        summary = {
            "id": state.id,
            "label": state.label,
            "ready": state.ready,
            "reason": state.warning,
        }
        if state.learned_index is not None:
            summary.update(state.learned_index.public_status())
        summaries.append(summary)
    return summaries


def _stage_score(row: dict[str, Any]) -> float | None:
    retrieval = row.get("retrieval")
    if not isinstance(retrieval, dict):
        return None
    for stage in ("reranker", "rrf", "dense", "bm25"):
        value = retrieval.get(stage)
        if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
            return float(value["score"])
    return None


def search_pipeline(
    index_path: Path,
    retriever: HybridRetriever | None,
    question: str,
    top_k: int,
    institution: str | None,
    *,
    include_text: bool,
    retrieval_mode: str | None = None,
    service_tuning: bool = False,
    capture_candidates: bool = False,
    max_chunks_per_document: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return backward-compatible rows plus an explicit retrieval trace."""

    normalized_retrieval_query = normalize_retrieval_query(
        question,
        institution,
    )
    query_expansions = (
        _service_retrieval_query_expansions(question)
        if service_tuning
        else ()
    )
    retrieval_query = (
        expand_service_retrieval_query(question, institution)
        if service_tuning
        else normalized_retrieval_query
    )
    if retriever is None:
        evaluation_candidates: dict[str, list[dict[str, Any]]] = {}
        if capture_candidates:
            raw_limit = max(
                20,
                top_k,
                top_k * DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
            )
            raw_rows = search_bm25_candidates(
                index_path,
                retrieval_query,
                raw_limit,
                institution,
                preview_chars=preview_chars(),
                include_text=include_text,
                drop_query_tokens=(
                    SERVICE_QUERY_STOP_TOKENS if service_tuning else None
                ),
            )
            if service_tuning:
                raw_rows = demote_generic_candidates(
                    retrieval_query,
                    raw_rows,
                )
            evaluation_candidates["raw_bm25"] = raw_rows
        rows = search_index(
            index_path,
            retrieval_query,
            top_k,
            institution,
            preview_chars=preview_chars(),
            include_text=include_text,
            service_tuning=service_tuning,
            max_chunks_per_document=max_chunks_per_document,
        )
        trace = {
            "strategy": "BM25 (document-diverse)",
            "mode": retrieval_mode or "bm25",
            "retrieval_query": retrieval_query,
            "normalized_retrieval_query": normalized_retrieval_query,
            "query_expansions": list(query_expansions),
            "service_tuning": service_tuning,
            "max_chunks_per_document": max_chunks_per_document,
            "result_count": len(rows),
            "lanes": {
                "bm25": {"status": "ok", "count": len(rows)},
                "dense": {"status": "disabled", "count": 0},
            },
            "fusion": {"status": "single_lane", "kind": "rrf"},
            "reranker": {"status": "disabled"},
        }
        if capture_candidates:
            trace["evaluation_candidates"] = evaluation_candidates
        return rows, trace

    result = retriever.search(
        retrieval_query,
        top_k=top_k,
        institution=institution,
    )
    rows = []
    for hit in result.hits:
        row = hit.to_dict()
        text = str(row.get("text") or "")
        row["preview"] = text[:preview_chars()]
        if not include_text:
            row.pop("text", None)
        score = _stage_score(row)
        row["score"] = score
        row["scores"] = {
            stage: (
                value.get("score")
                if isinstance(value, dict)
                else None
            )
            for stage, value in row["retrieval"].items()
        }
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            row["corpus_revision"] = metadata.get("corpus_revision")
        row["location"] = row["locations"][0] if row["locations"] else None
        rows.append(row)

    trace = dict(result.trace)
    trace["retrieval_query"] = retrieval_query
    trace["normalized_retrieval_query"] = normalized_retrieval_query
    trace["query_expansions"] = list(query_expansions)
    dense_status = (
        trace.get("lanes", {}).get("dense", {}).get("status")
        if isinstance(trace.get("lanes"), dict)
        else None
    )
    strategies = {
        "kure_dense": "KURE Dense (cosine)",
        "kure_hybrid": "BM25 + KURE + RRF",
        "snowflake_dense": "Snowflake Dense (cosine)",
        "snowflake_hybrid": "BM25 + Snowflake + RRF",
    }
    trace.update(
        {
            "strategy": (
                strategies.get(retrieval_mode)
                or (
                    "BM25 + Dense + RRF"
                    if dense_status == "ok"
                    else "BM25 single-lane + RRF"
                )
            ),
            "mode": retrieval_mode or "legacy_hybrid",
            "result_count": len(rows),
        }
    )
    return rows, trace


class SearchHandler(BaseHTTPRequestHandler):
    index_path: Path = DEFAULT_INDEX
    dense_path: Path = DEFAULT_DENSE_INDEX
    # 서비스 검색 조정 (worklog S7: 구어체 토큰 제거 + 연도·상용구 강등).
    # 벤치마크 스크립트는 search_index를 기본값(꺼짐)으로 직접 호출하므로
    # 이 플래그는 search_api 경로에만 작용한다.
    retrieval_tuning: bool = True
    context_chunks_per_document: int = 2
    retriever: HybridRetriever | None = None
    retriever_warning: str | None = None
    default_parser_profile: str = "default"
    parser_targets: Mapping[str, ParserIndexTarget] = MappingProxyType({})
    generation_semaphore: threading.BoundedSemaphore = threading.BoundedSemaphore(DEFAULT_MAX_CONCURRENT_GENERATIONS)
    local_generation_semaphore: threading.BoundedSemaphore = threading.BoundedSemaphore(1)
    local_model_runtime: Any = ExternalLocalModelRuntime()

    @classmethod
    def parser_target(
        cls,
        value: Any = None,
        *,
        require_ready: bool = True,
    ) -> ParserIndexTarget:
        if cls.parser_targets:
            return resolve_parser_target(
                value,
                cls.parser_targets,
                cls.default_parser_profile,
                require_ready=require_ready,
            )
        fallback = ParserIndexTarget(
            profile=cls.default_parser_profile,
            index_path=cls.index_path,
            retriever=cls.retriever,
            warning=cls.retriever_warning,
        )
        return resolve_parser_target(
            value,
            {cls.default_parser_profile: fallback},
            cls.default_parser_profile,
            # Legacy single-index tests and imports configure index_path
            # directly. Startup-created parser_targets use the strict branch
            # above and still fail closed for missing configured files.
            require_ready=False,
        )

    def end_headers(self) -> None:
        origin = self.headers.get("Origin")
        allowed_origin = cors_origin_for(origin)
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-RAG-API-Key, Authorization")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def write_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_error(self, error: str, message: str, status: HTTPStatus) -> None:
        self.write_json({"error": error, "message": message}, status)

    def request_origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin and cors_origin_for(origin) is None:
            self.write_error("origin_not_allowed", "허용되지 않은 Origin입니다.", HTTPStatus.FORBIDDEN)
            return False
        return True

    def request_authorized(self) -> bool:
        if is_authorized(self.headers):
            return True
        self.write_error("unauthorized", "API 인증 토큰이 필요합니다.", HTTPStatus.UNAUTHORIZED)
        return False

    def request_allowed(self, *, protected: bool = False) -> bool:
        if not self.request_origin_allowed():
            return False
        if protected and not self.request_authorized():
            return False
        return True

    def read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_content_length", "Content-Length가 올바르지 않습니다.") from exc
        if length <= 0:
            return {}
        if length > request_max_bytes():
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                f"요청 본문은 {request_max_bytes()} bytes 이내여야 합니다.",
            )
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_json", "JSON 형식이 올바르지 않습니다.") from exc
        if not isinstance(payload, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_json_body", "JSON object를 보내 주세요.")
        return payload

    def do_OPTIONS(self) -> None:
        if not self.request_origin_allowed():
            return
        self.write_json({"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        try:
            if parsed.path == "/health":
                if not self.request_allowed():
                    return
                target = self.parser_target(require_ready=False)
                payload = index_stats(target.index_path, self.dense_path)
                providers = payload.get("providers")
                if isinstance(providers, dict):
                    local_provider = providers.get("local")
                    if isinstance(local_provider, dict):
                        local_provider.update(
                            self.local_model_runtime.public_status()
                        )
                payload["retriever_warning"] = target.warning
                payload["default_parser_profile"] = self.default_parser_profile
                mode_summaries = retrieval_mode_summaries(target)
                payload["default_retrieval_mode"] = default_retrieval_mode(
                    target
                )
                payload["retrieval_modes"] = mode_summaries
                payload["service_config"] = {
                    "retrieval_tuning": bool(self.retrieval_tuning),
                    "context_chunks_per_document": int(
                        self.context_chunks_per_document
                    ),
                    "default_context_top_k": DEFAULT_TOP_K,
                    "evaluation_trace_enabled": evaluation_trace_enabled(),
                    "generation": final_generation_health(),
                    "freeze": freeze_runtime_metadata(target.index_path),
                }
                learned_ready = any(
                    item.get("ready") is True and item.get("id") != "bm25"
                    for item in mode_summaries
                )
                pipeline = payload.get("pipeline")
                if learned_ready and isinstance(pipeline, dict):
                    pipeline["dense"] = {
                        "status": "ready",
                        "modes": [
                            item["id"]
                            for item in mode_summaries
                            if item.get("ready") is True
                            and str(item.get("id", "")).endswith("_dense")
                        ],
                    }
                    pipeline["rrf"] = {"status": "ready"}
                payload["parser_profiles"] = parser_profile_summaries(
                    self.parser_targets
                    or {self.default_parser_profile: target},
                    self.dense_path,
                )
                payload["roles"] = [
                    {"id": profile.id, "label": profile.label}
                    for profile in role_router.PRESET_ROLES
                ]
                self.write_json(payload)
                return

            if parsed.path == "/institutions":
                if not self.request_allowed():
                    return
                target = self.parser_target(
                    query.get("parser_profile", [None])[0]
                )
                self.write_json(
                    {
                        "parser_profile": target.profile,
                        "institutions": list_institutions(target.index_path),
                    }
                )
                return

            if parsed.path == "/search":
                if not self.request_allowed(protected=True):
                    return
                question = query.get("q", [""])[0].strip()
                institution = query.get("institution", [""])[0].strip() or None
                top_k = parse_top_k(query.get("top_k", [DEFAULT_TOP_K])[0])
                target = self.parser_target(
                    query.get("parser_profile", [None])[0]
                )
                retrieval_state = resolve_retrieval_mode(
                    target,
                    query.get("retrieval_mode", [None])[0],
                )
                results, retrieval = search_pipeline(
                    target.index_path,
                    retrieval_state.retriever,
                    question,
                    top_k,
                    institution,
                    include_text=False,
                    retrieval_mode=retrieval_state.id,
                    service_tuning=self.retrieval_tuning,
                )
                retrieval["parser_profile"] = target.profile
                retrieval["retriever_warning"] = retrieval_state.warning
                self.write_json(
                    {
                        "query": question,
                        "institution": institution,
                        "parser_profile": target.profile,
                        "retrieval_mode": retrieval_state.id,
                        "retrieval": retrieval,
                        "results": public_results(results),
                    }
                )
                return

            self.write_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except ApiError as exc:
            self.write_error(exc.error, exc.message, exc.status)
        except Exception as exc:
            print(f"GET {parsed.path} failed: {type(exc).__name__}: {exc}", flush=True)
            self.write_error("internal_error", "요청 처리 중 문제가 발생했습니다.", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/local-model/unload":
                if not self.request_allowed(protected=True):
                    return
                self.read_json_body()
                try:
                    payload = self.local_model_runtime.unload()
                except LocalModelBusyError:
                    self.write_error(
                        "local_model_busy",
                        "로컬 모델이 답변을 생성 중이라 지금은 내릴 수 없습니다.",
                        HTTPStatus.CONFLICT,
                    )
                    return
                except LocalModelUnmanagedError:
                    self.write_error(
                        "local_model_unload_not_supported",
                        "현재 로컬 모델 서버는 이 앱이 관리하지 않아 메모리에서 내릴 수 없습니다.",
                        HTTPStatus.NOT_IMPLEMENTED,
                    )
                    return
                except LocalModelRuntimeError:
                    self.write_error(
                        "local_model_unavailable",
                        "로컬 모델을 메모리에서 내리는 중 문제가 발생했습니다.",
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
                self.write_json(payload)
                return

            if parsed.path != "/chat":
                self.write_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                return
            if not self.request_allowed(protected=True):
                return

            request_started = time.perf_counter()
            body = self.read_json_body()
            include_evaluation_trace = resolve_evaluation_trace_request(
                body.get("eval_trace", False)
            )
            question = validate_question(body.get("question", ""))
            raw_institution = body.get("institution")
            institution = (
                str(raw_institution).strip() or None
                if raw_institution is not None
                else None
            )
            # 역할 자유 입력은 매핑에만 쓰고 원문은 프롬프트에 넣지 않는다.
            # 매핑 판정 비용이 길이에 비례하므로 상한만 자른다.
            requested_role = str(body.get("role", "")).strip()[:120] or None
            role_profile = role_router.resolve(requested_role)
            role_perspective = (
                role_profile.perspective
                if role_profile.id != "general"
                else None
            )
            top_k = parse_top_k(body.get("top_k", DEFAULT_TOP_K))
            target = self.parser_target(body.get("parser_profile"))
            retrieval_state = resolve_retrieval_mode(
                target, body.get("retrieval_mode")
            )
            requested_provider = public_provider_name(
                validate_provider(body.get("provider"))
            )
            generation_provider = resolve_generation_provider(requested_provider)
            requested_model = validate_requested_model(
                requested_provider,
                body["model"] if "model" in body else REQUEST_MODEL_MISSING,
            )

            candidate_limit = chat_candidate_limit(top_k)
            retrieval_started = time.perf_counter()
            results, retrieval = search_pipeline(
                target.index_path,
                retrieval_state.retriever,
                question,
                candidate_limit,
                institution,
                include_text=True,
                retrieval_mode=retrieval_state.id,
                service_tuning=self.retrieval_tuning,
                capture_candidates=include_evaluation_trace,
                max_chunks_per_document=self.context_chunks_per_document,
            )
            evaluation_candidate_stages = retrieval.pop(
                "evaluation_candidates",
                {},
            )
            retrieved_candidate_trace = (
                evaluation_context_trace(
                    results,
                    stage="post_retrieval_pool",
                )
                if include_evaluation_trace
                else []
            )
            # 역할 소프트 우선순위: top-k 선정(인접 확장·중복 제거) 이전의
            # 넓은 후보 풀에서 우선 기관 문서를 앞으로 보낸다. 기관 필터가
            # 명시된 요청에는 이미 스코프가 고정되어 있으므로 건드리지
            # 않는다.
            if institution is None:
                results = role_router.prioritize_hits(results, role_profile)
            results, neighbor_expansion = (
                replace_with_adjacent_temporal_contexts(
                    target.index_path,
                    results,
                    (
                        question
                        if self.retrieval_tuning
                        else str(
                            retrieval.get("retrieval_query")
                            or question
                        )
                    ),
                    facet_completion=self.retrieval_tuning,
                    max_chunks_per_document=(
                        self.context_chunks_per_document
                    ),
                )
            )
            post_expansion_trace = (
                evaluation_context_trace(
                    results,
                    stage="post_neighbor_expansion",
                )
                if include_evaluation_trace
                else []
            )
            results, context_deduplication = select_distinct_contexts(
                results,
                top_k,
            )
            retrieval["result_count"] = len(results)
            retrieval["neighbor_expansion"] = neighbor_expansion
            retrieval["context_deduplication"] = {
                **context_deduplication,
                "candidate_limit": candidate_limit,
            }
            retrieval["parser_profile"] = target.profile
            retrieval["retriever_warning"] = retrieval_state.warning
            retrieval["role"] = role_router.public_role(
                role_profile, requested_role
            )
            context_security = evaluate_contexts(results)
            results = list(context_security.original_contexts)
            generation_results = list(context_security.generation_contexts)
            retrieval["result_count"] = len(results)
            retrieval["security_gate"] = context_security.summary()
            retrieval_elapsed_ms = (
                time.perf_counter() - retrieval_started
            ) * 1000
            numbered_results = number_sources(results)
            numbered_generation_results = number_sources(generation_results)
            generation_input = (
                generation_input_trace(
                    question,
                    numbered_generation_results,
                    role_perspective=role_perspective,
                )
                if include_evaluation_trace and numbered_generation_results
                else None
            )
            draft_answer = None
            raw_draft_answer = None
            generator = "extractive"
            generation_elapsed_ms = 0.0
            generation = {
                "requested": requested_provider,
                "used": "none",
                "model": None,
                "fallback_reason": "no_results" if not numbered_results else None,
                "attempts": [],
            }
            if numbered_results:
                generation_started = time.perf_counter()
                generation_semaphore = None
                if generation_provider == "local":
                    generation_semaphore = self.local_generation_semaphore
                elif generation_provider != "extractive":
                    generation_semaphore = self.generation_semaphore
                acquired = (
                    generation_semaphore.acquire(blocking=False)
                    if generation_semaphore is not None
                    else True
                )
                if not acquired:
                    self.write_error(
                        "server_busy",
                        "답변 생성 요청이 많습니다. 잠시 후 다시 시도해 주세요.",
                        HTTPStatus.TOO_MANY_REQUESTS,
                    )
                    return
                runtime_generation = None
                runtime_generation_entered = False
                runtime_model = requested_model or local_default_model()
                runtime_failure_code = None
                excluded_providers: tuple[str, ...] = ()
                try:
                    if generation_may_use_local(generation_provider):
                        try:
                            runtime_generation = self.local_model_runtime.generation(
                                runtime_model
                            )
                            runtime_generation.__enter__()
                            runtime_generation_entered = True
                        except LocalModelRuntimeError as exc:
                            # Managed mode is fail-closed: never send retrieved
                            # document context to an unexpected process that
                            # happens to occupy the configured local port.
                            runtime_generation = None
                            runtime_failure_code = exc.code
                            excluded_providers = ("local",)
                    try:
                        generated = generate(
                            question,
                            numbered_generation_results,
                            requested=generation_provider,
                            extractive_fallback=extractive_fallback_answer,
                            requested_model=requested_model,
                            excluded_providers=excluded_providers,
                            # 일반 사용자(매핑 실패 포함)는 역할 블록을 아예
                            # 넣지 않아 역할 없는 요청과 프롬프트가 같다.
                            role_perspective=role_perspective,
                        )
                        raw_draft_answer = generated.text
                        draft_answer = strip_untrusted_citation_markers(
                            raw_draft_answer
                        )
                        generation = public_generation_metadata(
                            with_local_runtime_failure(
                                generated.metadata(),
                                code=runtime_failure_code,
                                model=runtime_model,
                            ),
                            requested_provider=requested_provider,
                        )
                        loaded_model = local_attempt_model(
                            getattr(generated, "attempts", ()),
                            runtime_model,
                        )
                        if loaded_model:
                            self.local_model_runtime.note_loaded(loaded_model)
                        generator = (
                            f"{generated.used}:{generated.model}"
                            if generated.model
                            else generated.used
                        )
                    except GenerationError as exc:
                        loaded_model = local_attempt_model(
                            exc.attempts,
                            runtime_model,
                        )
                        if loaded_model:
                            self.local_model_runtime.note_loaded(loaded_model)
                        # A deadline can expire before the normal extractive route.
                        # Keep the service useful with an in-process safe fallback.
                        draft_answer = extractive_fallback_answer(
                            question, numbered_generation_results
                        )
                        raw_draft_answer = draft_answer
                        generation = public_generation_metadata(
                            with_local_runtime_failure(
                                {
                                    "requested": generation_provider,
                                    "used": "extractive",
                                    "model": None,
                                    "fallback_reason": exc.code,
                                    "attempts": [
                                        attempt.to_dict()
                                        for attempt in exc.attempts
                                    ],
                                },
                                code=runtime_failure_code,
                                model=runtime_model,
                            ),
                            requested_provider=requested_provider,
                        )
                        generator = "extractive"
                finally:
                    try:
                        if (
                            runtime_generation is not None
                            and runtime_generation_entered
                        ):
                            runtime_generation.__exit__(None, None, None)
                    finally:
                        if generation_semaphore is not None:
                            generation_semaphore.release()
                generation_elapsed_ms = (
                    time.perf_counter() - generation_started
                ) * 1000

            postprocess_started = time.perf_counter()
            rag = build_rag_response(question, numbered_results, draft_answer, generator)
            output_security = enforce_output(rag, numbered_results)
            rag = output_security.response
            postprocess_elapsed_ms = (
                time.perf_counter() - postprocess_started
            ) * 1000
            payload = {
                "question": question,
                "institution": institution,
                "role": role_router.public_role(role_profile, requested_role),
                "parser_profile": target.profile,
                "retrieval_mode": retrieval_state.id,
                "answer": rag["answer"],
                "cited_answer": rag["cited_answer"],
                "claims": rag["claims"],
                "citations": rag.get("citations", []),
                "postprocessing": rag["postprocessing"],
                "generator": rag["generator"],
                "generation": generation,
                "retrieval": retrieval,
                "security": {
                    "context_gate": context_security.summary(),
                    "output_gate": output_security.summary(),
                },
                "results": public_results(numbered_results),
            }
            if include_evaluation_trace:
                payload["evaluation_trace"] = {
                    "schema_version": 1,
                    "retrieval_stages": {
                        "raw_bm25": evaluation_context_trace(
                            evaluation_candidate_stages.get("raw_bm25", []),
                            stage="raw_bm25",
                        ),
                        "post_retrieval_pool": retrieved_candidate_trace,
                        "post_neighbor_expansion": post_expansion_trace,
                        "final_contexts": evaluation_context_trace(
                            numbered_results,
                            stage="final_contexts",
                        ),
                    },
                    "raw_draft": raw_draft_answer,
                    "sanitized_draft": draft_answer,
                    "generation_input": generation_input,
                    "timing_ms": {
                        "retrieval": round(retrieval_elapsed_ms, 3),
                        "generation": round(generation_elapsed_ms, 3),
                        "postprocess": round(postprocess_elapsed_ms, 3),
                        "e2e": round(
                            (time.perf_counter() - request_started) * 1000,
                            3,
                        ),
                    },
                }
            self.write_json(payload)
        except ApiError as exc:
            self.write_error(exc.error, exc.message, exc.status)
        except Exception as exc:
            print(f"POST {parsed.path} failed: {type(exc).__name__}: {exc}", flush=True)
            self.write_error("internal_error", "요청 처리 중 문제가 발생했습니다.", HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--profile-index",
        action="append",
        default=[],
        metavar="PROFILE=PATH",
        help="Register an additional fixed parser-profile BM25 index.",
    )
    parser.add_argument(
        "--default-parser-profile",
        help="Default profile when a request omits parser_profile.",
    )
    parser.add_argument("--dense-index", type=Path)
    parser.add_argument(
        "--learned-dense-root",
        type=Path,
        help=(
            "Directory containing profile/model learned-dense artifacts; "
            "legacy model-only artifacts are used for Cascade."
        ),
    )
    parser.add_argument(
        "--no-retrieval-tuning",
        action="store_true",
        help="서비스 검색 조정(구어체 토큰 제거·연도/상용구 강등)을 끕니다. A/B 재현용.",
    )
    parser.add_argument(
        "--context-chunks-per-document",
        type=int,
        default=2,
        help="BM25 chat 후보 풀에서 문서당 허용할 최대 청크 수 (1~8).",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--override-env", action="store_true", help="Allow values from --env-file to override existing environment variables.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file, override=args.override_env)
    try:
        parser_indexes, default_parser_profile = build_parser_index_registry(
            args.index,
            args.profile_index,
            args.default_parser_profile,
        )
    except ValueError as exc:
        raise SystemExit(f"Parser index configuration error: {exc}") from exc

    SearchHandler.retrieval_tuning = not args.no_retrieval_tuning
    if not 1 <= args.context_chunks_per_document <= 8:
        raise SystemExit("--context-chunks-per-document must be between 1 and 8")
    SearchHandler.context_chunks_per_document = args.context_chunks_per_document
    SearchHandler.dense_path = args.dense_index or dense_index_path()
    learned_root = args.learned_dense_root or learned_dense_root_path()
    query_embedder_cache: dict[tuple[Any, ...], Any] = {}
    targets: dict[str, ParserIndexTarget] = {}
    for profile, index_path in parser_indexes.items():
        retriever, warning = create_hybrid_retriever(
            index_path,
            SearchHandler.dense_path,
        )
        targets[profile] = ParserIndexTarget(
            profile=profile,
            index_path=index_path,
            retriever=retriever,
            warning=warning,
            retrieval_modes=create_retrieval_modes(
                index_path,
                learned_root,
                profile,
                query_embedder_cache=query_embedder_cache,
            ),
        )

    SearchHandler.default_parser_profile = default_parser_profile
    SearchHandler.parser_targets = MappingProxyType(dict(targets))
    default_target = targets[default_parser_profile]
    SearchHandler.index_path = default_target.index_path
    SearchHandler.retriever = default_target.retriever
    SearchHandler.retriever_warning = default_target.warning
    SearchHandler.generation_semaphore = threading.BoundedSemaphore(
        max(1, get_env_int("RAG_MAX_CONCURRENT_GENERATIONS", DEFAULT_MAX_CONCURRENT_GENERATIONS))
    )
    SearchHandler.local_generation_semaphore = threading.BoundedSemaphore(1)
    SearchHandler.local_model_runtime = build_local_model_runtime()
    if (
        getattr(SearchHandler.local_model_runtime, "managed", False)
        and not is_loopback_bind_host(args.host)
        and not api_token()
    ):
        raise SystemExit(
            "RAG_API_TOKEN is required when a managed local runtime is "
            "served on a non-loopback host."
        )
    server = ThreadingHTTPServer((args.host, args.port), SearchHandler)
    print(f"Search API listening on http://{args.host}:{args.port}")
    print(f"Default parser profile: {default_parser_profile}")
    for profile, target in targets.items():
        print(
            f"Parser index [{profile}]: {target.index_path.resolve()} "
            f"(retriever warning: {target.warning})"
        )
    print(f"Dense index: {SearchHandler.dense_path.resolve()}")
    print(f"Learned dense root: {learned_root.resolve()}")
    print(
        "Default retrieval mode: "
        f"{default_retrieval_mode(default_target)}"
    )
    for mode in retrieval_mode_summaries(default_target):
        print(
            f"Retrieval mode [{mode['id']}]: "
            f"{'ready' if mode.get('ready') else mode.get('reason')}"
        )
    print(f"Generation mode: {generation_mode()}")
    print(f"Gemini configured: {bool(gemini_api_key())}")
    print(f"Gemini model: {gemini_model()}")
    print(f"Gemini candidates: {', '.join(gemini_model_candidates())}")
    print(f"Allowed origins: {', '.join(allowed_origins())}")
    print(f"API token required: {bool(api_token())}")
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def stop_on_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        SearchHandler.local_model_runtime.close()
        server.server_close()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
