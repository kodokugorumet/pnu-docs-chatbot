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
import hmac
import ipaddress
import json
import os
import re
import signal
import sqlite3
import threading
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
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
except ImportError:  # Direct CLI execution: python scripts/search_api.py
    from bm25_search import (
        DEFAULT_INDEX,
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
MAX_CLAIMS = 5
MAX_CHAT_CANDIDATES = 80
CHAT_CANDIDATE_MULTIPLIER = 4
CLAIM_SUPPORT_THRESHOLD = 0.40
MIN_CLAIM_ANCHORS = 2
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
        "ok": True,
        "ready": True,
        "status": status,
        "index": str(index_path),
        "chunk_count": chunk_count,
        "document_count": document_count,
        "institution_count": institution_count,
        "corpus_revision": metadata.get("corpus_revision"),
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


def citation_for_result(result: dict[str, Any]) -> dict[str, Any]:
    locations = result.get("locations")
    locations = locations if isinstance(locations, list) else []
    limit = max_citation_locations()
    return {
        "citation_id": f"citation:{result.get('chunk_id', '')}",
        "source_number": result.get("source_number"),
        "chunk_id": result.get("chunk_id"),
        "document_id": result.get("document_id") or result.get("doc_id"),
        "corpus_revision": result.get("corpus_revision"),
        "excerpt": result_source_excerpt(result),
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


def is_low_quality_candidate(text: str) -> bool:
    terms = tokenize(text, include_ngrams=False)
    if len(terms) >= 18 and len(set(terms)) / len(terms) < 0.42:
        return True
    if len(re.findall(r"[-_=]{4,}", text)) >= 1:
        return True
    if len(text) < 20:
        return True
    return False


def split_candidate_sentences(text: str) -> list[str]:
    text = re.sub(r"([\[\(]page \d+[\]\)])", "", text, flags=re.IGNORECASE)
    rough_parts = re.split(
        r"\n+|(?<=[.!?。！？])\s+|(?<=다\.)\s+|(?<=임\.)\s+|(?<=음\.)\s+",
        text,
    )
    candidates: list[str] = []
    for part in rough_parts:
        subparts = re.split(r"\s+(?=[①②③④⑤⑥⑦⑧⑨⑩□ㅇ◦])", part)
        for subpart in subparts:
            subpart = normalize_text(subpart)
            if len(subpart) > 280:
                subpart = subpart[:280].rsplit(" ", 1)[0].strip()
            if subpart and not is_low_quality_candidate(subpart):
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


def _is_table_context(result: dict[str, Any]) -> bool:
    table_ids = result.get("table_ids")
    if not isinstance(table_ids, list) or not table_ids:
        return False
    return bool(re.search(r"\d", result_source_text(result)))


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


def replace_with_adjacent_temporal_contexts(
    index_path: Path,
    rows: list[dict[str, Any]],
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace a temporal hit with a more relevant adjacent table chunk."""

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


def select_answer_claims(question: str, results: list[dict[str, Any]]) -> list[str]:
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
SEMESTER_RE = re.compile(r"(?<!\d)([1-9])\s*학기")
ROUND_RE = re.compile(r"(?<!\d)(\d+)\s*차(?!원)")
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
    r"\s*(?:조|억|만|천|백)?)"
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
    r"(?<!\d)(\d[\d,]*)\s*(부(?!터)|명|개|회|학점|개월)"
)
PHONE_RE = re.compile(r"(?<!\d)(0\d{1,2})[-\s](\d{3,4})[-\s](\d{4})(?!\d)")
ABSTENTION_RE = re.compile(
    r"(?:제공된\s*)?(?:문서|검색\s*근거|검색\s*결과|근거)"
    r"(?:에서|에서는|만으로는|가|는)?\s*.{0,50}?"
    r"(?:확인할\s*수\s*없|확인되지\s*않|찾지\s*못|알\s*수\s*없|부족)"
)
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
    for match in SEMESTER_RE.finditer(normalized):
        facts.add(f"semester:{int(match.group(1))}")
    for match in ROUND_RE.finditer(normalized):
        facts.add(f"round:{int(match.group(1))}")
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
    for pattern in (KOREAN_MONTH_DAY_RE, DOTTED_MONTH_DAY_RE):
        for match in pattern.finditer(normalized):
            month = int(match.group(1))
            day = int(match.group(2))
            if not _is_valid_month_day(month, day):
                continue
            month_days.add((month, day))
            facts.add(f"month_day:{month:02d}-{day:02d}")
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
            and re.search(r"감소|하락|축소", suffix)
        ):
            percent = -abs(percent)
        else:
            percent = abs(percent)
        facts.add(f"percent:{_normalized_decimal(percent)}")
    for match in QUANTITY_RE.finditer(normalized):
        quantity = int(match.group(1).replace(",", ""))
        facts.add(f"quantity:{quantity}:{match.group(2)}")
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
    for prefix in ("round:", "academic_year:", "semester:"):
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
        units.append(f"{left} {right}")
    return list(dict.fromkeys(units))


def _critical_value_support(
    claim_values: frozenset[str],
    result: dict[str, Any],
) -> tuple[bool, set[str]]:
    if not claim_values:
        return True, set()

    source = result_source_text(result)
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    scope_text = " ".join(
        str(value)
        for value in (
            result.get("source_title"),
            result.get("file_name"),
            metadata.get("source_title"),
        )
        if value
    )
    scope_values = {
        value
        for value in extract_critical_values(scope_text)
        if value.startswith(("academic_year:", "semester:"))
    }
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
    return False, set(claim_values - best_values)


def _is_abstention_claim(claim: str) -> bool:
    normalized = normalize_text(claim)
    return bool(ABSTENTION_RE.search(normalized))


def attribute_claim(claim: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    claim_terms = {
        term
        for term in tokenize(claim, include_ngrams=False)
        if not term.isdigit()
    }
    claim_values = extract_critical_values(claim)
    matches: list[tuple[float, int, dict[str, Any]]] = []
    best_score = -1.0
    best_missing_values: set[str] = set(claim_values)
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
        source_terms = set(tokenize(source, include_ngrams=False))
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
            claim_values,
            result,
        )
        required_anchors = (
            1
            if claim_values
            else min(MIN_CLAIM_ANCHORS, len(claim_terms))
        )
        lexical_supported = (
            exact
            or (not claim_terms and facts_supported)
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
        if lexical_supported and facts_supported:
            matches.append((ranked_score, source_number, result))

    matches.sort(key=lambda item: item[0], reverse=True)
    top_matches = matches[:2]
    validation_reason = "supported" if top_matches else "low_lexical_overlap"
    if not top_matches and best_missing_values:
        validation_reason = "critical_value_mismatch"
    return {
        "text": claim,
        "supported": bool(top_matches),
        "confidence": round(min(0.99, top_matches[0][0]) if top_matches else 0.0, 3),
        "source_ids": [match[2]["chunk_id"] for match in top_matches],
        "source_numbers": [match[1] for match in top_matches],
        "citations": [citation_for_result(match[2]) for match in top_matches],
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
            claims.extend(split_candidate_sentences(line))
    return claims[:MAX_CLAIMS]


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
        }

    claim_texts = split_draft_claims(draft_answer) if draft_answer else []
    if not claim_texts:
        claim_texts = select_answer_claims(question, results)

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
            }
        return {
            "answer": "검색 결과는 있으나 답변 문장을 지지하는 근거를 충분히 확인하지 못했습니다.",
            "cited_answer": "검색 결과는 있으나 답변 문장을 지지하는 근거를 충분히 확인하지 못했습니다.",
            "claims": claims,
            "citations": [],
            "draft_answer": draft_answer,
            "generator": generator,
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
            citation_for_result(result)
            for result in results
            if result.get("chunk_id")
            in {
                source_id
                for claim in supported_claims
                for source_id in claim["source_ids"]
            }
        ],
        "draft_answer": draft_answer,
        "generator": generator,
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return backward-compatible rows plus an explicit retrieval trace."""

    retrieval_query = normalize_retrieval_query(question, institution)
    if retriever is None:
        rows = search_index(
            index_path,
            retrieval_query,
            top_k,
            institution,
            preview_chars=preview_chars(),
            include_text=include_text,
        )
        return rows, {
            "strategy": "BM25 (document-diverse)",
            "mode": retrieval_mode or "bm25",
            "retrieval_query": retrieval_query,
            "result_count": len(rows),
            "lanes": {
                "bm25": {"status": "ok", "count": len(rows)},
                "dense": {"status": "disabled", "count": 0},
            },
            "fusion": {"status": "single_lane", "kind": "rrf"},
            "reranker": {"status": "disabled"},
        }

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

            body = self.read_json_body()
            question = validate_question(body.get("question", ""))
            institution = str(body.get("institution", "")).strip() or None
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
            results, retrieval = search_pipeline(
                target.index_path,
                retrieval_state.retriever,
                question,
                candidate_limit,
                institution,
                include_text=True,
                retrieval_mode=retrieval_state.id,
            )
            results, neighbor_expansion = (
                replace_with_adjacent_temporal_contexts(
                    target.index_path,
                    results,
                    str(
                        retrieval.get("retrieval_query")
                        or question
                    ),
                )
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
            numbered_results = number_sources(results)
            draft_answer = None
            generator = "extractive"
            generation = {
                "requested": requested_provider,
                "used": "none",
                "model": None,
                "fallback_reason": "no_results" if not numbered_results else None,
                "attempts": [],
            }
            if numbered_results:
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
                            numbered_results,
                            requested=generation_provider,
                            extractive_fallback=extractive_fallback_answer,
                            requested_model=requested_model,
                            excluded_providers=excluded_providers,
                        )
                        draft_answer = strip_untrusted_citation_markers(generated.text)
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
                            question, numbered_results
                        )
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

            rag = build_rag_response(question, numbered_results, draft_answer, generator)
            self.write_json(
                {
                    "question": question,
                    "institution": institution,
                    "parser_profile": target.profile,
                    "retrieval_mode": retrieval_state.id,
                    "answer": rag["answer"],
                    "cited_answer": rag["cited_answer"],
                    "claims": rag["claims"],
                    "citations": rag.get("citations", []),
                    "generator": rag["generator"],
                    "generation": generation,
                    "retrieval": retrieval,
                    "results": public_results(numbered_results),
                }
            )
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
