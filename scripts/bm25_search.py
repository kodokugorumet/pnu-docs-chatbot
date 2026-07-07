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
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CHUNKS = Path("processed/current/chunks.jsonl")
DEFAULT_INDEX = Path("processed/index/bm25.sqlite")
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
          chunk_index INTEGER NOT NULL,
          institution TEXT,
          source_path TEXT,
          relative_path TEXT,
          file_name TEXT,
          extension TEXT,
          parser TEXT,
          char_count INTEGER,
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


def build_index(chunks_path: Path, index_path: Path, batch_size: int) -> dict[str, Any]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    connection = open_index(index_path)
    ensure_schema(connection)

    chunk_rows: list[tuple[Any, ...]] = []
    fts_rows: list[tuple[str, str]] = []
    institutions: Counter[str] = Counter()
    total = 0

    def flush() -> None:
        nonlocal chunk_rows, fts_rows
        if not chunk_rows:
            return
        connection.executemany(
            """
            INSERT INTO chunks (
              chunk_id, doc_id, chunk_index, institution, source_path, relative_path,
              file_name, extension, parser, char_count, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    for chunk in iter_chunks(chunks_path):
        metadata = chunk.get("metadata") or {}
        chunk_id = chunk["chunk_id"]
        institution = metadata.get("institution", "")
        text = chunk.get("text", "")
        chunk_rows.append(
            (
                chunk_id,
                chunk.get("doc_id", ""),
                int(chunk.get("chunk_index", 0)),
                institution,
                metadata.get("source_path", ""),
                metadata.get("relative_path", ""),
                metadata.get("file_name", ""),
                metadata.get("extension", ""),
                metadata.get("parser", ""),
                int(chunk.get("char_count") or len(text)),
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
    created_at = datetime.now(timezone.utc).isoformat()
    metadata_rows = {
        "chunks_path": str(chunks_path),
        "chunk_count": str(total),
        "created_at": created_at,
        "institutions": json.dumps(institutions, ensure_ascii=False),
    }
    connection.executemany(
        "INSERT INTO index_meta (key, value) VALUES (?, ?)",
        metadata_rows.items(),
    )
    connection.commit()
    connection.close()
    return {
        "index": str(index_path),
        "chunks_path": str(chunks_path),
        "chunk_count": total,
        "institutions": dict(institutions),
        "created_at": created_at,
    }


def fts_query(user_query: str) -> str:
    terms = dedupe_keep_order(tokenize(user_query), limit=32)
    return " OR ".join(terms)


def search_index(
    index_path: Path,
    query: str,
    top_k: int,
    institution: str | None,
    *,
    preview_chars: int = 700,
    include_text: bool = False,
) -> list[dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index DB: {index_path}")

    match_query = fts_query(query)
    if not match_query:
        return []

    preview_chars = max(1, min(int(preview_chars), 5000))
    select_full_text = ",\n          c.text AS text" if include_text else ""
    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    params: list[Any] = [preview_chars, match_query]
    where = "chunk_fts MATCH ?"
    if institution:
        where += " AND c.institution = ?"
        params.append(institution)
    params.append(top_k)

    rows = connection.execute(
        f"""
        SELECT
          c.chunk_id,
          c.doc_id,
          c.chunk_index,
          c.institution,
          c.file_name,
          c.source_path,
          c.relative_path,
          c.char_count,
          bm25(chunk_fts) AS score,
          substr(c.text, 1, ?) AS preview
          {select_full_text}
        FROM chunk_fts
        JOIN chunks c ON c.chunk_id = chunk_fts.chunk_id
        WHERE {where}
        ORDER BY score
        LIMIT ?
        """,
        params,
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


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
    build.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    build.add_argument("--batch-size", type=int, default=1000)

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
        summary = build_index(args.chunks, args.index, args.batch_size)
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
