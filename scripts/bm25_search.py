#!/usr/bin/env python3
"""Build and query a local BM25 search index for parsed RAG chunks.

The index uses SQLite FTS5 so it stays local and dependency-free. For Korean
documents, the indexed text includes simple Hangul bigrams in addition to exact
tokens, which makes early MVP searches more forgiving.

Examples:
  python scripts/bm25_search.py build
  python scripts/bm25_search.py search "신탁 수탁고 현황" --top-k 5
  python scripts/bm25_search.py search "부산대 휴학 신청" --institution 부산대학교
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Direct CLI execution puts scripts/, not the repository root, on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .rag.corpus import inspect_corpus
except ImportError:  # Direct CLI execution: python scripts/bm25_search.py
    from rag.corpus import inspect_corpus


DEFAULT_CHUNKS = Path("processed/current/chunks.jsonl")
DEFAULT_INDEX = Path("processed/index/bm25.sqlite")
DEFAULT_DENSE_INDEX = Path("processed/index/dense.sqlite")
DEFAULT_RERANK_CANDIDATE_MULTIPLIER = 4
TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+|\d+")
HANGUL_RE = re.compile(r"^[가-힣]+$")


def tokenize(value: str, *, include_ngrams: bool = True) -> list[str]:
    terms: list[str] = []
    for match in TOKEN_RE.finditer(value.lower()):
        token = match.group(0)
        if len(token) <= 1 and not token.isdigit():
            continue
        terms.append(token)
        if include_ngrams and HANGUL_RE.match(token) and len(token) >= 2:
            terms.extend(token[index : index + 2] for index in range(len(token) - 1))
    return terms


def dedupe_keep_order(values: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if limit and len(result) >= limit:
            break
    return result


def make_search_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    fields = [
        metadata.get("institution", ""),
        metadata.get("file_name", ""),
        metadata.get("relative_path", ""),
        metadata.get("extension", ""),
        metadata.get("source_title", ""),
        metadata.get("source_host", ""),
        metadata.get("category", ""),
        chunk.get("text", ""),
    ]
    tokens = tokenize("\n".join(str(field) for field in fields if field))
    return " ".join(tokens)


def open_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    # The database is built at a private temporary path and published only
    # after close, so WAL adds no reader benefit.  DELETE mode leaves one
    # self-contained file that can be opened with SQLite mode=ro immediately;
    # a WAL-mode main file without its transient -shm sidecar cannot be.
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS index_meta;
        DROP TABLE IF EXISTS chunk_fts;

        CREATE TABLE chunks (
          chunk_id TEXT PRIMARY KEY,
          doc_id TEXT NOT NULL,
          document_id TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          institution TEXT,
          source_path TEXT,
          relative_path TEXT,
          file_name TEXT,
          extension TEXT,
          parser TEXT,
          source_title TEXT,
          source_url TEXT,
          download_url TEXT,
          source_host TEXT,
          fetched_at TEXT,
          published_at TEXT,
          category TEXT,
          include_reason TEXT,
          crawl_storage_path TEXT,
          source_aliases_json TEXT NOT NULL DEFAULT '[]',
          char_count INTEGER,
          page_start INTEGER,
          page_end INTEGER,
          section_path_json TEXT,
          table_ids_json TEXT NOT NULL DEFAULT '[]',
          block_ids_json TEXT NOT NULL DEFAULT '[]',
          locations_json TEXT NOT NULL DEFAULT '[]',
          corpus_revision TEXT NOT NULL,
          text TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunk_fts USING fts5(
          chunk_id UNINDEXED,
          search_text,
          tokenize = 'unicode61'
        );

        CREATE TABLE index_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE INDEX idx_chunks_institution ON chunks(institution);
        CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX idx_chunks_document_id ON chunks(document_id);
        """
    )


def iter_chunks(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def _json_text(value: Any, default: Any) -> str:
    normalized = value if value is not None else default
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _chunks_paths(value: Path | Sequence[Path]) -> list[Path]:
    paths = [value] if isinstance(value, Path) else list(value)
    if not paths:
        raise ValueError("At least one chunks file is required")
    normalized = [Path(path) for path in paths]
    missing = [path for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing chunks file: {missing[0]}")
    return normalized


def _collection_revision(revisions: Sequence[str]) -> str:
    if len(revisions) == 1:
        return revisions[0]
    payload = json.dumps(
        sorted(revisions),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"collection:{hashlib.sha256(payload).hexdigest()[:24]}"


def build_index(
    chunks_path: Path | Sequence[Path],
    index_path: Path,
    batch_size: int,
    *,
    allow_suspect: bool = False,
    require_manifest: bool = False,
) -> dict[str, Any]:
    """Build a gated index and atomically publish it when complete."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks_paths = _chunks_paths(chunks_path)
    sources = [
        (
            path,
            inspect_corpus(
                path,
                allow_suspect=allow_suspect,
                require_manifest=require_manifest,
            ),
        )
        for path in chunks_paths
    ]
    corpus_revision = _collection_revision(
        [gate.corpus_revision for _, gate in sources]
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_path.with_name(f".{index_path.name}.{uuid.uuid4().hex}.tmp")
    connection: sqlite3.Connection | None = None

    try:
        connection = open_index(temporary_index)
        ensure_schema(connection)

        chunk_rows: list[tuple[Any, ...]] = []
        fts_rows: list[tuple[str, str]] = []
        institutions: Counter[str] = Counter()
        seen_chunk_ids: set[str] = set()
        total = 0

        def flush() -> None:
            nonlocal chunk_rows, fts_rows
            if not chunk_rows:
                return
            assert connection is not None
            connection.executemany(
                """
                INSERT INTO chunks (
                  chunk_id, doc_id, document_id, chunk_index, institution,
                  source_path, relative_path, file_name, extension, parser,
                  source_title, source_url, download_url, source_host,
                  fetched_at, published_at, category, include_reason,
                  crawl_storage_path, source_aliases_json,
                  char_count, page_start, page_end, section_path_json,
                  table_ids_json, block_ids_json, locations_json,
                  corpus_revision, text
                ) VALUES (
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                chunk_rows,
            )
            connection.executemany(
                "INSERT INTO chunk_fts (chunk_id, search_text) VALUES (?, ?)",
                fts_rows,
            )
            connection.commit()
            chunk_rows = []
            fts_rows = []

        for source_path, gate in sources:
            for chunk in iter_chunks(source_path):
                if not gate.allows_chunk(chunk):
                    continue
                metadata = chunk.get("metadata") or {}
                chunk_id = str(chunk["chunk_id"])
                if chunk_id in seen_chunk_ids:
                    raise RuntimeError(
                        f"Duplicate chunk_id across corpus sources: {chunk_id}"
                    )
                seen_chunk_ids.add(chunk_id)
                document_id = str(
                    chunk.get("document_id") or chunk.get("doc_id") or ""
                )
                if not document_id:
                    raise RuntimeError(
                        f"Chunk {chunk_id} is missing document_id/doc_id"
                    )
                institution = str(metadata.get("institution") or "")
                text = str(chunk.get("text") or "")
                block_ids = (
                    metadata.get("block_ids")
                    if isinstance(metadata.get("block_ids"), list)
                    else []
                )
                table_ids = (
                    metadata.get("table_ids")
                    if isinstance(metadata.get("table_ids"), list)
                    else []
                )
                chunk_rows.append(
                    (
                        chunk_id,
                        document_id,
                        document_id,
                        int(chunk.get("chunk_index", 0)),
                        institution,
                        metadata.get("source_path", ""),
                        metadata.get("relative_path", ""),
                        metadata.get("file_name", ""),
                        metadata.get("extension", ""),
                        metadata.get("parser", ""),
                        metadata.get("source_title", ""),
                        metadata.get("source_url", ""),
                        metadata.get("download_url", ""),
                        metadata.get("source_host", ""),
                        metadata.get("fetched_at", ""),
                        metadata.get("published_at", ""),
                        metadata.get("category", ""),
                        metadata.get("include_reason", ""),
                        metadata.get("crawl_storage_path", ""),
                        _json_text(metadata.get("source_aliases"), []),
                        int(chunk.get("char_count") or len(text)),
                        metadata.get("page_start"),
                        metadata.get("page_end"),
                        _json_text(metadata.get("section_path"), None),
                        _json_text(table_ids, []),
                        _json_text(block_ids, []),
                        _json_text(gate.locations_for_chunk(chunk), []),
                        gate.corpus_revision,
                        text,
                    )
                )
                fts_rows.append((chunk_id, make_search_text(chunk)))
                institutions[institution] += 1
                total += 1

                if total % batch_size == 0:
                    flush()
                    print(f"Indexed {total} chunks")

        flush()
        if total == 0:
            raise RuntimeError("Corpus gate produced no non-empty chunks")

        created_at = datetime.now(timezone.utc).isoformat()
        single_gate = sources[0][1] if len(sources) == 1 else None
        source_descriptors = [
            {
                "chunks_path": str(path),
                "corpus_revision": gate.corpus_revision,
                "run_id": gate.run_id,
                "profile": gate.profile,
                "manifest_sha256": gate.manifest_sha256,
                "source_manifest_sha256": gate.source_manifest_sha256,
                "selection_counts": gate.selection_counts,
            }
            for path, gate in sources
        ]
        source_manifest_sha256s = sorted(
            {
                gate.source_manifest_sha256
                for _, gate in sources
                if gate.source_manifest_sha256 is not None
            }
        )
        shared_source_manifest_sha256 = (
            source_manifest_sha256s[0]
            if len(source_manifest_sha256s) == 1
            and all(
                gate.source_manifest_sha256 is not None
                for _, gate in sources
            )
            else ""
        )
        metadata_rows = {
            "chunks_path": str(chunks_paths[0]) if len(chunks_paths) == 1 else "",
            "chunks_paths": json.dumps(
                [str(path) for path in chunks_paths],
                ensure_ascii=False,
            ),
            "sources": json.dumps(source_descriptors, ensure_ascii=False),
            "source_count": str(len(sources)),
            "chunk_count": str(total),
            "created_at": created_at,
            "institutions": json.dumps(institutions, ensure_ascii=False),
            "corpus_revision": corpus_revision,
            "run_id": single_gate.run_id or "" if single_gate else "",
            "profile": single_gate.profile or "" if single_gate else "",
            "manifest_sha256": (
                single_gate.manifest_sha256 if single_gate else ""
            ),
            "source_manifest_sha256": shared_source_manifest_sha256,
            "source_manifest_sha256s": json.dumps(
                source_manifest_sha256s,
                ensure_ascii=False,
            ),
            "selection_counts": json.dumps(
                single_gate.selection_counts if single_gate else {},
                ensure_ascii=False,
            ),
            "excluded_document_count": str(
                sum(len(gate.excluded_document_ids) for _, gate in sources)
            ),
            "excluded_chunk_count": str(
                sum(len(gate.excluded_chunk_ids) for _, gate in sources)
            ),
        }
        connection.executemany(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            metadata_rows.items(),
        )
        connection.commit()
        connection.close()
        connection = None

        # The previous index stays intact until this point.
        os.replace(temporary_index, index_path)
        return {
            "index": str(index_path),
            "chunks_path": (
                str(chunks_paths[0]) if len(chunks_paths) == 1 else None
            ),
            "chunks_paths": [str(path) for path in chunks_paths],
            "source_count": len(sources),
            "chunk_count": total,
            "institutions": dict(institutions),
            "created_at": created_at,
            "corpus_revision": corpus_revision,
            "verified_run": all(gate.is_verified_run for _, gate in sources),
            "source_manifest_sha256": (
                shared_source_manifest_sha256 or None
            ),
            "source_manifest_sha256s": source_manifest_sha256s,
            "selection_counts": (
                dict(single_gate.selection_counts) if single_gate else {}
            ),
            "excluded_document_count": sum(
                len(gate.excluded_document_ids) for _, gate in sources
            ),
            "excluded_chunk_count": sum(
                len(gate.excluded_chunk_ids) for _, gate in sources
            ),
        }
    finally:
        if connection is not None:
            connection.close()
        temporary_index.unlink(missing_ok=True)


def build_dense_index(
    source_index: Path,
    index_path: Path,
    *,
    dimensions: int = 256,
) -> dict[str, Any]:
    """Build the deterministic offline dense lane from a BM25 index."""

    try:
        from .rag.retrieval import DenseIndex, HashingEmbedder
    except ImportError:  # Direct CLI execution.
        from rag.retrieval import DenseIndex, HashingEmbedder

    dense = DenseIndex.from_bm25_index(
        source_index,
        embedder=HashingEmbedder(dimensions=dimensions),
        index_path=index_path,
    )
    return {
        "index": str(index_path),
        "source_index": str(source_index),
        "chunk_count": len(dense),
        "embedding_kind": dense.embedding_kind,
        "dimensions": dense.dimensions,
        "corpus_revision": dense.corpus_revision,
    }


# 서비스 질의 전용 구어체 어미·의문사 토큰 (2026-08-31 진단, worklog S7).
# tokenize()는 어절 통짜 + 한글 바이그램을 만들기 때문에 "있나요"가
# 있나요·있나·나요 3토큰으로 같은 표면 문자열에 중복 매칭되고, 이 토큰들은
# FAQ·상담 가이드북 계열에 집중되어 있어 무관 질의의 상위를 점령한다.
# 여기서 제거하는 것은 질의 측 OR 항뿐이다 — 내용어의 통짜 토큰과 나머지
# 바이그램은 남으므로 정보 손실이 없다. 벤치마크 경로는 이 상수를 쓰지
# 않는다 (service_tuning 기본 꺼짐).
SERVICE_QUERY_STOP_TOKENS = frozenset({
    "나요", "있나", "있나요", "되나", "되나요", "됐나", "됐나요",
    "하나요", "한가요", "인가요", "할까", "할까요", "까요", "가요",
    "어떻", "떻게", "어떻게", "어떤", "해요", "세요", "궁금", "금해",
    "궁금해", "니까", "습니까", "합니까",
})

# 내비게이션·상용구 제목 (정규화: 공백 제거 + 소문자). 일반적으로는
# 강등하되, 본문·섹션이 질의의 강한 내용어를 충분히 덮는 실제 내용
# 페이지는 아래 content-aware 규칙으로 구제한다.
GENERIC_SOURCE_TITLES = frozenset({
    "faqs", "부산대학교", "국문(korean)", "pdf", "hwp", "[다운로드]",
    "사이트맵", "100%크게보기",
    "pnu포커스>뉴스>홍보센터|부산대학교",
    "카드뉴스내용>뉴스>홍보센터|부산대학교",
    "공지사항내용>공지사항>공지/참여|부산대학교",
    # 회의록공개 셸: 본문 없는 목록 페이지인데 연도가 없어 연도 강등을
    # 피해 상위로 유입 (S8 A/B에서 svc_reg_05 오염 실증). 브레드크럼
    # 제목 일반(453문서)에는 실제 내용 페이지가 많아 일괄 강등은 금지 —
    # 정확 일치로만 추가한다.
    "등록금심의위원회회의록공개>행정서비스>학교소개|부산대학교",
    "등록금심의위원회회의록공개내용>행정서비스>학교소개|부산대학교",
})
_GENERIC_DOWNLOAD_TITLE_RE = re.compile(
    r"^(?:pdf|hwp|hwpx|doc|docx|xls|xlsx|ppt|pptx)(?:파일)?다운로드$"
)

_YEAR_TOKEN_RE = re.compile(r"(?:19|20)\d{2}")
_LATIN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]{2,}(?![A-Za-z0-9])"
)

# 파일 형식·기관명·일반 앱 표현은 많은 문서 제목에 반복되므로 희소 식별자
# 부스트로 쓰지 않는다. BIDV/TOPIK처럼 질의와 제목에 정확히 함께 등장하는
# 토큰만 후보를 앞당긴다.
_GENERIC_LATIN_TITLE_TOKENS = frozenset({
    "app", "apps", "application", "applications", "com", "doc", "docx",
    "download", "file", "hwp", "hwpx", "html", "http", "https", "jpeg",
    "jpg", "kr", "pdf", "png", "pnu", "ppt", "pptx", "use", "www",
    "xls", "xlsx",
})

_QUERY_CONTENT_STOP_TERMS = frozenset({
    "그거", "내야", "누가", "뭐가", "무엇", "부산대", "부산대학교", "수",
    "알려줘", "알려주세요", "어느", "어디", "어떤", "어떻게", "언제",
    "언제까지", "이번", "있나요", "있어", "있어요", "지금", "현재", "하고",
    "하나요", "해당",
}) | SERVICE_QUERY_STOP_TOKENS

_KOREAN_TERM_SUFFIXES = (
    "으로부터", "에게서는", "에서부터", "으로는", "에서는", "에서도",
    "에게서", "까지는", "부터는", "하려고", "하면서", "하는", "해서",
    "에서", "에게", "한테", "으로", "까지", "부터", "처럼", "보다",
    "하며", "하면", "하고", "하려", "할", "은", "는", "이", "가", "을",
    "를", "에", "도", "만",
)

_PROCEDURE_QUERY_TERMS = frozenset({
    "app", "application", "단계", "방법", "신청", "앱", "어떻게", "업로드",
    "입력", "절차", "제출", "로그인", "선택",
})
_MINUTES_DECISION_TERMS = frozenset({
    "결과", "동결", "심의", "위원회", "의결", "인상", "책정", "회의록",
})


def fts_query(user_query: str, *, drop_tokens: frozenset[str] | None = None) -> str:
    terms = dedupe_keep_order(tokenize(user_query), limit=32)
    if drop_tokens:
        kept = [term for term in terms if term not in drop_tokens]
        terms = kept or terms
    return " OR ".join(terms)


def _normalize_content_term(value: str) -> str:
    term = str(value or "").lower()
    if not HANGUL_RE.fullmatch(term):
        return term
    for suffix in _KOREAN_TERM_SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 2:
            return term[: -len(suffix)]
    return term


def _content_terms(value: str) -> set[str]:
    return {
        normalized
        for token in tokenize(value, include_ngrams=False)
        if (normalized := _normalize_content_term(token))
    }


def _flatten_scope_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _flatten_scope_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _flatten_scope_strings(nested)


def _row_body_and_section_text(row: dict[str, Any]) -> str:
    scopes: list[Any] = [
        row.get("text") or row.get("preview") or "",
        row.get("section_path"),
    ]
    location = row.get("location")
    if isinstance(location, dict):
        scopes.append(location.get("section_path"))
    locations = row.get("locations")
    if isinstance(locations, list):
        scopes.extend(
            item.get("section_path")
            for item in locations
            if isinstance(item, dict)
        )
    return "\n".join(
        text
        for scope in scopes
        for text in _flatten_scope_strings(scope)
        if text
    )


def _has_strong_content_coverage(query: str, row: dict[str, Any]) -> bool:
    query_terms = {
        term
        for term in _content_terms(query)
        if term not in _QUERY_CONTENT_STOP_TERMS
        and (len(term) >= 2 or term.isdigit())
    }
    if len(query_terms) < 3:
        return False
    candidate_terms = _content_terms(_row_body_and_section_text(row))
    matched = query_terms & candidate_terms
    # 숫자(연도·학기·금액)는 의미를 뒤집을 수 있어 비율 계산만으로
    # 누락을 허용하지 않는다.
    numeric_terms = {term for term in query_terms if term.isdigit()}
    return (
        numeric_terms <= candidate_terms
        and len(matched) >= 3
        and len(matched) / len(query_terms) >= 0.85
    )


def _rare_query_title_match(query: str, row: dict[str, Any]) -> bool:
    query_tokens = {
        token.lower()
        for token in _LATIN_TOKEN_RE.findall(str(query or ""))
        if token.lower() not in _GENERIC_LATIN_TITLE_TOKENS
    }
    if not query_tokens:
        return False
    title_text = " ".join(
        str(row.get(field) or "") for field in ("source_title", "file_name")
    )
    title_tokens = {
        token.lower() for token in _LATIN_TOKEN_RE.findall(title_text)
    }
    return bool(query_tokens & title_tokens)


def _is_procedure_minutes_noise(query: str, row: dict[str, Any]) -> bool:
    raw_query_terms = {
        token.lower() for token in tokenize(query, include_ngrams=False)
    }
    query_terms = raw_query_terms | {
        _normalize_content_term(token) for token in raw_query_terms
    }
    if not (query_terms & _PROCEDURE_QUERY_TERMS):
        return False
    if query_terms & _MINUTES_DECISION_TERMS:
        return False
    candidate_label = re.sub(
        r"\s+",
        "",
        " ".join(
            str(row.get(field) or "") for field in ("source_title", "file_name")
        ).lower(),
    )
    return "회의록" in candidate_label and "등록금" in candidate_label


_SPECIALIZED_FOREIGN_ADMISSION_TITLE_TERMS = (
    "추가모집",
    "추천트랙",
    "현지",
    "글로벌자유전공학부",
)


def _foreign_admission_title_priority(
    query: str,
    row: dict[str, Any],
) -> str | None:
    """Classify general vs specialized guides for an unscoped admission query."""

    query_key = re.sub(r"\s+", "", str(query or ""))
    intent_terms = ("외국인", "학부", "신입", "자격")
    if not all(term in query_key for term in intent_terms):
        return None
    # A user who explicitly asks for a country/local/recommendation/additional
    # track should keep ordinary BM25 ordering for that requested scope.
    if any(term in query_key for term in _SPECIALIZED_FOREIGN_ADMISSION_TITLE_TERMS):
        return None

    title_key = re.sub(
        r"\s+",
        "",
        " ".join(
            str(row.get(field) or "")
            for field in ("source_title", "file_name")
        ),
    )
    if "대학원" in title_key and "학부" not in title_key:
        return "scope_mismatch"
    if not all(
        term in title_key
        for term in ("학부", "외국인", "특별전형", "모집요강")
    ):
        return None
    if any(
        term in title_key
        for term in _SPECIALIZED_FOREIGN_ADMISSION_TITLE_TERMS
    ):
        return "specialized"
    return "general"


def demote_generic_candidates(
    query: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """서비스 전용 후보 안정 분할: 버킷 안의 순서·점수는 보존한다.

    우선 대상:
    - 상용구 제목이지만 본문·섹션이 질의 내용어를 85% 이상(최소 3개,
      숫자는 전부) 덮는 후보
    - 질의의 희소 영문 토큰(BIDV/TOPIK 등)이 제목·파일명에 정확히 있는 후보

    강등 대상 (role_router.prioritize_hits와 같은 분할 방식):
    - 제목이 내비게이션·상용구(GENERIC_SOURCE_TITLES)인 후보
    - 질의에 연도가 있는데 제목·파일명에 다른 연도만 있는 후보
      (연도 토큰은 DF가 커서 IDF≈0 — BM25가 연도 사본을 구분하지 못하는
      공백을 순위 밖에서 메운다. 제목에 연도가 없으면 강등하지 않고,
      질의에 연도가 없으면 규칙 자체가 발동하지 않는다.)
    - 절차·앱 질의에서 등록금심의위원회 회의록인 후보. 단 심의 결과·동결·
      인상 등을 직접 묻는 질의는 회의록이 정답이므로 제외한다.
    """
    query_years = set(_YEAR_TOKEN_RE.findall(str(query or "")))
    promoted: list[dict[str, Any]] = []
    preferred: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    scope_conflicted: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.get("source_title") or "")
        title_key = re.sub(r"\s+", "", title).lower()
        is_generic = (
            title_key in GENERIC_SOURCE_TITLES
            or bool(_GENERIC_DOWNLOAD_TITLE_RE.fullmatch(title_key))
        )
        generic_content_match = is_generic and _has_strong_content_coverage(
            query, row
        )
        year_mismatch = False
        if query_years and not is_generic:
            candidate_years = set(_YEAR_TOKEN_RE.findall(title)) | set(
                _YEAR_TOKEN_RE.findall(str(row.get("file_name") or ""))
            )
            year_mismatch = bool(candidate_years) and not (
                candidate_years & query_years
            )
        foreign_admission_priority = _foreign_admission_title_priority(
            query,
            row,
        )
        should_demote = (
            (is_generic and not generic_content_match)
            or year_mismatch
            or _is_procedure_minutes_noise(query, row)
            or foreign_admission_priority == "specialized"
        )
        if foreign_admission_priority == "scope_mismatch":
            scope_conflicted.append(row)
        elif should_demote:
            demoted.append(row)
        elif (
            generic_content_match
            or _rare_query_title_match(query, row)
            or foreign_admission_priority == "general"
        ):
            promoted.append(row)
        else:
            preferred.append(row)
    return promoted + preferred + demoted + scope_conflicted


def normalize_for_rank(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def lexical_rerank_score(query: str, row: dict[str, Any], bm25_rank: int) -> float:
    query_terms = set(dedupe_keep_order(tokenize(query, include_ngrams=False), limit=24))
    if not query_terms:
        return 0.0

    text = str(row.get("text") or row.get("preview") or "")
    metadata_text = " ".join(
        str(row.get(field) or "")
        for field in (
            "institution",
            "file_name",
            "relative_path",
            "source_title",
            "category",
        )
    )
    body_terms = set(tokenize(text, include_ngrams=False))
    metadata_terms = set(tokenize(metadata_text, include_ngrams=False))

    body_coverage = len(query_terms & body_terms) / len(query_terms)
    metadata_coverage = len(query_terms & metadata_terms) / len(query_terms)

    normalized_query = normalize_for_rank(query)
    normalized_text = normalize_for_rank(text)
    phrase_bonus = 1.0 if normalized_query and normalized_query in normalized_text else 0.0

    # SQLite FTS bm25 scores are useful but hard to compare across queries. Rank
    # position keeps that signal stable while allowing domain-specific boosts.
    bm25_rank_signal = 1 / max(1, bm25_rank)
    return (
        body_coverage * 0.52
        + metadata_coverage * 0.16
        + phrase_bonus * 0.18
        + bm25_rank_signal * 0.14
    )


def rerank_results(query: str, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scored = [
        (lexical_rerank_score(query, row, rank), rank, dict(row))
        for rank, row in enumerate(rows, start=1)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    reranked: list[dict[str, Any]] = []
    for final_rank, (rerank_score, _, row) in enumerate(scored[:top_k], start=1):
        retrieval = dict(row.get("retrieval") or {})
        retrieval["reranker"] = {
            "rank": final_rank,
            "score": round(rerank_score, 8),
            "kind": "lexical_fallback",
        }
        retrieval["final_rank"] = final_rank
        row["retrieval"] = retrieval
        row["rerank_score"] = round(rerank_score, 8)
        # Backward-compatible alias.  Unlike the old response, score now means
        # the final comparable relevance score, not SQLite's negative BM25.
        row["score"] = row["rerank_score"]
        reranked.append(row)
    return reranked


def select_document_diverse_results(
    rows: Sequence[dict[str, Any]],
    top_k: int,
    *,
    max_chunks_per_document: int = 2,
    preserve_chunk_id: str | None = None,
) -> list[dict[str, Any]]:
    """Keep BM25 order while preventing one document from crowding the result.

    The lexical fallback remains available as an explicit experiment through
    ``rerank_results``.  The production BM25 path is deliberately rank-safe:
    exact title/path matches that FTS5 already places highly must not be
    demoted by a second heuristic score.

    ``preserve_chunk_id`` marks one chunk (the BM25 anchor) that the per-document
    cap may not evict.  When the cap would drop it, it replaces the lowest-ranked
    already-selected chunk of the same document instead, inheriting that slot's
    position.  Rationale (grant_031, 2026-08-11): the CE reranker preferred two
    surface-similar chunks of the same document, the cap kept only those two, and
    the BM25 top-1 chunk holding the actual answer never reached the context.
    """

    limit = max(0, int(top_k))
    if limit == 0:
        return []
    per_document = max(1, int(max_chunks_per_document))
    document_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    preserved_pending: dict[str, Any] | None = None

    for value in rows:
        row = dict(value)
        document_id = str(
            row.get("document_id")
            or row.get("doc_id")
            or row.get("chunk_id")
            or ""
        )
        if document_counts[document_id] >= per_document:
            if preserve_chunk_id and row.get("chunk_id") == preserve_chunk_id:
                preserved_pending = row
            continue
        document_counts[document_id] += 1
        row["score"] = row.get("bm25_score", row.get("score"))
        selected.append(row)
        if len(selected) >= limit:
            break

    if preserved_pending is not None:
        anchor_document = str(
            preserved_pending.get("document_id")
            or preserved_pending.get("doc_id")
            or preserved_pending.get("chunk_id")
            or ""
        )
        for position in range(len(selected) - 1, -1, -1):
            candidate_document = str(
                selected[position].get("document_id")
                or selected[position].get("doc_id")
                or selected[position].get("chunk_id")
                or ""
            )
            if candidate_document == anchor_document:
                preserved_pending["score"] = preserved_pending.get(
                    "bm25_score", preserved_pending.get("score")
                )
                selected[position] = preserved_pending
                break

    for final_rank, row in enumerate(selected, start=1):
        retrieval = dict(row.get("retrieval") or {})
        retrieval["reranker"] = None
        retrieval["final_rank"] = final_rank
        row["retrieval"] = retrieval

    return selected


def _chunk_columns(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(chunks)").fetchall()
    }


def _json_column(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default
    return decoded


def search_bm25_candidates(
    index_path: Path,
    query: str,
    limit: int,
    institution: str | None,
    *,
    preview_chars: int = 700,
    include_text: bool = False,
    drop_query_tokens: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Return raw BM25 candidates for fusion without applying the reranker."""

    if not index_path.exists():
        raise FileNotFoundError(f"Missing index DB: {index_path}")

    match_query = fts_query(query, drop_tokens=drop_query_tokens)
    if not match_query:
        return []

    preview_chars = max(1, min(int(preview_chars), 5000))
    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    columns = _chunk_columns(connection)
    document_id_select = (
        "c.document_id AS document_id"
        if "document_id" in columns
        else "c.doc_id AS document_id"
    )

    optional_columns = {
        "page_start": "c.page_start",
        "page_end": "c.page_end",
        "section_path_json": "c.section_path_json",
        "table_ids_json": "c.table_ids_json",
        "block_ids_json": "c.block_ids_json",
        "locations_json": "c.locations_json",
        "corpus_revision": "c.corpus_revision",
        "source_title": "c.source_title",
        "source_url": "c.source_url",
        "download_url": "c.download_url",
        "source_host": "c.source_host",
        "fetched_at": "c.fetched_at",
        "published_at": "c.published_at",
        "category": "c.category",
        "include_reason": "c.include_reason",
        "crawl_storage_path": "c.crawl_storage_path",
        "source_aliases_json": "c.source_aliases_json",
    }
    optional_selects = [
        f"{expression} AS {name}" if name in columns else f"NULL AS {name}"
        for name, expression in optional_columns.items()
    ]
    if include_text:
        optional_selects.append("c.text AS text")
    optional_sql = ",\n          ".join(optional_selects)

    params: list[Any] = [preview_chars, match_query]
    where = "chunk_fts MATCH ?"
    if institution:
        where += " AND c.institution = ?"
        params.append(institution)
    params.append(max(1, int(limit)))

    rows = connection.execute(
        f"""
        SELECT
          c.chunk_id,
          c.doc_id,
          {document_id_select},
          c.chunk_index,
          c.institution,
          c.file_name,
          c.source_path,
          c.relative_path,
          c.char_count,
          bm25(chunk_fts) AS bm25_score,
          substr(c.text, 1, ?) AS preview,
          {optional_sql}
        FROM chunk_fts
        JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
        WHERE {where}
        ORDER BY bm25_score, c.chunk_id
        LIMIT ?
        """,
        params,
    ).fetchall()
    connection.close()

    candidates: list[dict[str, Any]] = []
    for bm25_rank, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        row["section_path"] = _json_column(row.pop("section_path_json", None), None)
        row["table_ids"] = _json_column(row.pop("table_ids_json", None), [])
        row["block_ids"] = _json_column(row.pop("block_ids_json", None), [])
        row["locations"] = _json_column(row.pop("locations_json", None), [])
        row["source_aliases"] = _json_column(
            row.pop("source_aliases_json", None), []
        )
        row["location"] = {
            "page_start": row.pop("page_start", None),
            "page_end": row.pop("page_end", None),
            "section_path": row["section_path"],
            "table_ids": row["table_ids"],
            "block_ids": row["block_ids"],
        }
        row["metadata"] = {
            "corpus_revision": row.get("corpus_revision"),
            "source_title": row.get("source_title"),
            "source_url": row.get("source_url"),
            "download_url": row.get("download_url"),
            "source_host": row.get("source_host"),
            "fetched_at": row.get("fetched_at"),
            "published_at": row.get("published_at"),
            "category": row.get("category"),
            "include_reason": row.get("include_reason"),
            "crawl_storage_path": row.get("crawl_storage_path"),
            "source_aliases": row["source_aliases"],
            **row["location"],
        }
        row["score"] = row["bm25_score"]
        row["retrieval"] = {
            "bm25": {"rank": bm25_rank, "score": row["bm25_score"]},
            "dense": None,
            "rrf": None,
            "reranker": None,
            "final_rank": None,
        }
        candidates.append(row)
    return candidates


def load_chunks_by_ids(
    index_path: Path,
    chunk_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Load complete chunk rows in caller order for a secondary retriever.

    Learned-dense indexes deliberately store only vectors and stable chunk IDs.
    This keeps document text and citation metadata authoritative in the BM25
    SQLite corpus instead of duplicating it in every embedding artifact.
    """

    ordered_ids = dedupe_keep_order(
        str(value).strip() for value in chunk_ids if str(value).strip()
    )
    if not ordered_ids:
        return []
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index DB: {index_path}")

    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in ordered_ids)
        rows = connection.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            ordered_ids,
        ).fetchall()
    finally:
        connection.close()

    found: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        row["section_path"] = _json_column(
            row.pop("section_path_json", None), None
        )
        row["table_ids"] = _json_column(
            row.pop("table_ids_json", None), []
        )
        row["block_ids"] = _json_column(
            row.pop("block_ids_json", None), []
        )
        row["locations"] = _json_column(
            row.pop("locations_json", None), []
        )
        row["source_aliases"] = _json_column(
            row.pop("source_aliases_json", None), []
        )
        row["preview"] = str(row.get("text") or "")
        row["metadata"] = {
            "corpus_revision": row.get("corpus_revision"),
            "source_title": row.get("source_title"),
            "source_url": row.get("source_url"),
            "download_url": row.get("download_url"),
            "source_host": row.get("source_host"),
            "fetched_at": row.get("fetched_at"),
            "published_at": row.get("published_at"),
            "category": row.get("category"),
            "include_reason": row.get("include_reason"),
            "crawl_storage_path": row.get("crawl_storage_path"),
            "source_aliases": row["source_aliases"],
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "section_path": row["section_path"],
            "table_ids": row["table_ids"],
            "block_ids": row["block_ids"],
        }
        found[str(row["chunk_id"])] = row
    return [found[chunk_id] for chunk_id in ordered_ids if chunk_id in found]


def search_index(
    index_path: Path,
    query: str,
    top_k: int,
    institution: str | None,
    *,
    preview_chars: int = 700,
    include_text: bool = False,
    candidate_multiplier: int = DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
    service_tuning: bool = False,
    max_chunks_per_document: int = 2,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    candidate_limit = max(
        20,
        top_k,
        top_k * max(1, int(candidate_multiplier)),
    )
    rows = search_bm25_candidates(
        index_path,
        query,
        candidate_limit,
        institution,
        preview_chars=preview_chars,
        include_text=include_text,
        drop_query_tokens=SERVICE_QUERY_STOP_TOKENS if service_tuning else None,
    )
    if service_tuning:
        rows = demote_generic_candidates(query, rows)
    return select_document_diverse_results(
        rows,
        top_k,
        max_chunks_per_document=max_chunks_per_document,
    )


def print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No results.")
        return
    for index, row in enumerate(results, start=1):
        print(f"\n[{index}] score={row['score']:.4f} institution={row['institution']}")
        print(f"file={row['file_name']} chunk={row['chunk_index']} chars={row['char_count']}")
        print(f"path={row['source_path']}")
        preview = re.sub(r"\s+", " ", row["preview"]).strip()
        print(preview[:500])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the SQLite BM25 index.")
    build.add_argument(
        "--chunks",
        type=Path,
        action="append",
        help=(
            "Chunks JSONL to include. Repeat for a verified multi-source "
            f"collection (default: {DEFAULT_CHUNKS})."
        ),
    )
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build.add_argument("--batch-size", type=int, default=1000)
    build.add_argument(
        "--allow-suspect",
        action="store_true",
        help="Index suspect documents as well as quality=pass documents.",
    )
    build.add_argument(
        "--require-manifest",
        action="store_true",
        help="Reject legacy chunks that are not part of a verified parser run.",
    )

    dense = subparsers.add_parser(
        "build-dense",
        help="Build the local hashing dense index from the BM25 store.",
    )
    dense.add_argument("--source-index", type=Path, default=DEFAULT_INDEX)
    dense.add_argument("--index", type=Path, default=DEFAULT_DENSE_INDEX)
    dense.add_argument("--dimensions", type=int, default=256)

    search = subparsers.add_parser("search", help="Search the BM25 index.")
    search.add_argument("query")
    search.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--institution")
    search.add_argument("--json", action="store_true", help="Print JSON instead of a readable list.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        summary = build_index(
            args.chunks or [DEFAULT_CHUNKS],
            args.index,
            args.batch_size,
            allow_suspect=args.allow_suspect,
            require_manifest=args.require_manifest,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "build-dense":
        summary = build_dense_index(
            args.source_index,
            args.index,
            dimensions=args.dimensions,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        results = search_index(args.index, args.query, args.top_k, args.institution)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print_results(results)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
