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
        chunk.get("text", ""),
    ]
    tokens = tokenize("\n".join(str(field) for field in fields if field))
    return " ".join(tokens)


def open_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
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
                  char_count, page_start, page_end, section_path_json,
                  table_ids_json, block_ids_json, locations_json,
                  corpus_revision, text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            }
            for path, gate in sources
        ]
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


def fts_query(user_query: str) -> str:
    terms = dedupe_keep_order(tokenize(user_query), limit=32)
    return " OR ".join(terms)


def normalize_for_rank(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def lexical_rerank_score(query: str, row: dict[str, Any], bm25_rank: int) -> float:
    query_terms = set(dedupe_keep_order(tokenize(query, include_ngrams=False), limit=24))
    if not query_terms:
        return 0.0

    text = str(row.get("text") or row.get("preview") or "")
    metadata_text = " ".join(
        str(row.get(field) or "")
        for field in ("institution", "file_name", "relative_path")
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
) -> list[dict[str, Any]]:
    """Return raw BM25 candidates for fusion without applying the reranker."""

    if not index_path.exists():
        raise FileNotFoundError(f"Missing index DB: {index_path}")

    match_query = fts_query(query)
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
        row["location"] = {
            "page_start": row.pop("page_start", None),
            "page_end": row.pop("page_end", None),
            "section_path": row["section_path"],
            "table_ids": row["table_ids"],
            "block_ids": row["block_ids"],
        }
        row["metadata"] = {
            "corpus_revision": row.get("corpus_revision"),
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


def search_index(
    index_path: Path,
    query: str,
    top_k: int,
    institution: str | None,
    *,
    preview_chars: int = 700,
    include_text: bool = False,
    candidate_multiplier: int = DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
) -> list[dict[str, Any]]:
    candidate_limit = max(top_k, top_k * max(1, int(candidate_multiplier)))
    rows = search_bm25_candidates(
        index_path,
        query,
        candidate_limit,
        institution,
        preview_chars=preview_chars,
        include_text=include_text,
    )
    return rerank_results(query, rows, top_k)


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
