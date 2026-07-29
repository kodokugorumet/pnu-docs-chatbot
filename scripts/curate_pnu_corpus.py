#!/usr/bin/env python3
"""Build an auditable, zero-copy PNU parser-input manifest.

The crawler's SQLite frontier is the source of truth.  This tool never edits
the database or raw crawl files; it validates candidate files in place and
publishes a new manifest directory atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .crawl_pnu_site import (
        CrawlScope,
        LinkParser,
        canonicalize_url,
        decode_html,
        load_crawl_scope,
    )
except ImportError:  # Direct CLI execution with scripts/ on sys.path.
    from crawl_pnu_site import (
        CrawlScope,
        LinkParser,
        canonicalize_url,
        decode_html,
        load_crawl_scope,
    )


SCHEMA_VERSION = 1
MANIFEST_FIELDS = (
    "input_relative_path",
    "sha256",
    "size_bytes",
    "source_title",
    "source_url",
    "download_url",
    "source_host",
    "fetched_at",
    "published_at",
    "crawl_storage_path",
    "source_aliases",
    "category",
    "include_reason",
)

# These are exact host names, not suffix patterns.
DEFAULT_CORE_HOSTS = (
    "pusan.ac.kr",
    "www.pusan.ac.kr",
    "dorm.pusan.ac.kr",
    "graduate.pusan.ac.kr",
    "go.pusan.ac.kr",
    "international.pusan.ac.kr",
    "job.pusan.ac.kr",
    "lib.pusan.ac.kr",
    "equality.pusan.ac.kr",
    "sedu-support.pusan.ac.kr",
)

ALLOWED_EXTENSIONS = frozenset(
    {
        ".html",
        ".htm",
        ".xhtml",
        ".pdf",
        ".hwp",
        ".hwpx",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".txt",
        ".csv",
        ".rtf",
        ".odt",
    }
)
PAGE_EXTENSIONS = frozenset({".html", ".htm", ".xhtml"})
HARD_REJECT_EXTENSIONS = frozenset(
    {
        ".7z",
        ".apk",
        ".avi",
        ".bat",
        ".bin",
        ".bmp",
        ".bz",
        ".bz2",
        ".cmd",
        ".com",
        ".dmg",
        ".dll",
        ".exe",
        ".flac",
        ".gif",
        ".gz",
        ".ico",
        ".iso",
        ".jar",
        ".jpeg",
        ".jpg",
        ".js",
        ".json",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".msi",
        ".ogg",
        ".png",
        ".rar",
        ".sh",
        ".svg",
        ".tar",
        ".tif",
        ".tiff",
        ".ts",
        ".wav",
        ".webm",
        ".webp",
        ".xml",
        ".xz",
        ".zip",
    }
)
HARD_REJECT_MIME_PREFIXES = (
    "audio/",
    "image/",
    "video/",
)
HARD_REJECT_MIME_TYPES = frozenset(
    {
        "application/gzip",
        "application/java-archive",
        "application/json",
        "application/vnd.android.package-archive",
        "application/x-7z-compressed",
        "application/x-bzip",
        "application/x-bzip2",
        "application/x-dosexec",
        "application/x-executable",
        "application/x-rar-compressed",
        "application/x-sh",
        "application/x-tar",
        "application/x-zip-compressed",
        "application/zip",
        "text/javascript",
        "text/xml",
    }
)

# Order is significant: specific categories beat the generic notice category.
CATEGORY_TERMS = (
    (
        "graduation",
        (
            "졸업",
            "학위",
            "학위청구",
            "수료",
            "논문",
            "graduation",
            "thesis",
        ),
    ),
    (
        "registration",
        (
            "등록금",
            "등록",
            "납부",
            "분할납부",
            "tuition",
            "registration",
        ),
    ),
    (
        "scholarship",
        (
            "장학",
            "학자금",
            "근로장학생",
            "scholarship",
        ),
    ),
    (
        "academic",
        (
            "학사",
            "수업",
            "수강",
            "교육과정",
            "성적",
            "시험",
            "학적",
            "휴학",
            "복학",
            "전과",
            "복수전공",
            "부전공",
            "계절학기",
            "course",
            "curriculum",
        ),
    ),
    (
        "employment",
        (
            "취업",
            "채용",
            "진로",
            "인턴",
            "현장실습",
            "job",
            "career",
            "internship",
        ),
    ),
    (
        "international",
        (
            "국제",
            "교환학생",
            "교류",
            "파견",
            "유학",
            "외국인",
            "international",
            "exchange",
        ),
    ),
    (
        "student_support",
        (
            "학생지원",
            "기숙사",
            "생활관",
            "상담",
            "복지",
            "장애학생",
            "인권",
            "student support",
            "dormitory",
        ),
    ),
    (
        "notice",
        (
            "공지",
            "notice",
        ),
    ),
)

CORE_HOST_CATEGORIES = {
    "job.pusan.ac.kr": "employment",
    "dorm.pusan.ac.kr": "student_support",
    "graduate.pusan.ac.kr": "graduate_studies",
    "go.pusan.ac.kr": "admissions",
    "international.pusan.ac.kr": "international",
    "lib.pusan.ac.kr": "library",
    "equality.pusan.ac.kr": "student_support",
    "sedu-support.pusan.ac.kr": "student_support",
}

HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_MAX_DOCUMENTS = 6_000
DEFAULT_MAX_TOTAL_BYTES = int(1.5 * 1024 * 1024 * 1024)


class CurationError(RuntimeError):
    """Raised when a curated corpus cannot be published safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _content_type_only(value: Any) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def _normalize_host(value: str) -> str:
    host = str(value).strip().rstrip(".").lower()
    if (
        not host
        or not HOST_RE.fullmatch(host)
        or ".." in host
        or not (host == "pusan.ac.kr" or host.endswith(".pusan.ac.kr"))
    ):
        raise CurationError(
            "host must be an exact pusan.ac.kr host name: {!r}".format(value)
        )
    return host


def _host_from_url(value: Any) -> str:
    try:
        return (urllib.parse.urlsplit(str(value or "")).hostname or "").lower()
    except ValueError:
        return ""


def _safe_storage_path(
    crawl_root: Path,
    input_root: Path,
    storage_path: Any,
) -> Tuple[Path, str, str]:
    if not isinstance(storage_path, str) or not storage_path.strip():
        raise CurationError("missing_storage_path")
    rendered = storage_path.replace("\\", "/")
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise CurationError("unsafe_storage_path")
    crawl_storage_path = pure.as_posix()
    candidate = crawl_root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise CurationError("symlink_storage_path")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise CurationError("missing_file")
    except OSError as exc:
        raise CurationError(
            "unreadable_file:{}:{}".format(type(exc).__name__, exc)
        )
    try:
        resolved.relative_to(crawl_root)
    except ValueError:
        raise CurationError("outside_crawl_root")
    if not resolved.is_file():
        raise CurationError("not_a_file")
    try:
        input_relative = resolved.relative_to(input_root).as_posix()
    except ValueError:
        raise CurationError("outside_input_root")
    if not input_relative or input_relative.startswith("../"):
        raise CurationError("unsafe_input_relative_path")
    return resolved, input_relative, crawl_storage_path


def _type_rejection(
    extension: str,
    content_type: str,
    allowed_extensions: Iterable[str] = ALLOWED_EXTENSIONS,
) -> Optional[str]:
    if extension in HARD_REJECT_EXTENSIONS:
        return "hard_rejected_extension"
    if content_type in HARD_REJECT_MIME_TYPES or any(
        content_type.startswith(prefix)
        for prefix in HARD_REJECT_MIME_PREFIXES
    ):
        return "hard_rejected_content_type"
    if extension not in frozenset(allowed_extensions):
        return "unsupported_extension"
    return None


def _searchable_text(row: Mapping[str, Any]) -> str:
    values = (
        row.get("url"),
        row.get("final_url"),
        row.get("title"),
        row.get("anchor_text"),
        row.get("parent_title"),
    )
    return " ".join(
        urllib.parse.unquote(str(value))
        for value in values
        if value is not None
    ).casefold()


def _category_match(row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    searchable = _searchable_text(row)
    for category, terms in CATEGORY_TERMS:
        for term in terms:
            normalized_term = term.casefold()
            if normalized_term.isascii():
                matched = re.search(
                    r"(?<![a-z0-9]){}(?![a-z0-9])".format(
                        re.escape(normalized_term)
                    ),
                    searchable,
                )
            else:
                matched = normalized_term in searchable
            if matched:
                return category, term
    return None, None


def _source_values(row: Mapping[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    kind = str(row.get("kind") or "").lower()
    requested_url = str(row.get("url") or "").strip()
    final_url = str(row.get("final_url") or "").strip()
    parent_url = str(row.get("parent_url") or "").strip()
    if kind == "attachment":
        download_url = final_url or requested_url or None
        source_url = (
            str(row.get("parent_canonical_url") or "").strip()
            or parent_url
            or download_url
        )
        title = (
            str(row.get("anchor_text") or "").strip()
            or str(row.get("title") or "").strip()
            or str(row.get("parent_title") or "").strip()
        )
    else:
        source_url = final_url or requested_url or None
        download_url = None
        title = (
            str(row.get("title") or "").strip()
            or str(row.get("anchor_text") or "").strip()
        )
    return title, source_url, download_url


def _canonical_page_url(
    row: Mapping[str, Any],
    path: Path,
    content_type: str,
) -> Optional[str]:
    if str(row.get("kind") or "").lower() != "page":
        return None

    base_url = str(row.get("final_url") or row.get("url") or "").strip()
    declared = str(row.get("canonical_url") or "").strip()
    if not declared and path.suffix.lower() in PAGE_EXTENSIONS:
        try:
            parser = LinkParser()
            parser.feed(decode_html(path.read_bytes(), content_type))
            declared = str(parser.canonical_href or "").strip()
        except (OSError, UnicodeError, ValueError):
            return None
    canonical = canonicalize_url(declared, base_url) if declared else None
    if not canonical or not _valid_http_url(canonical):
        return None
    canonical_host = _normalized_pnu_url_host(canonical)
    source_host = _normalized_pnu_url_host(base_url)
    if (
        canonical_host is None
        or source_host is None
        or canonical_host != source_host
    ):
        return None
    return canonical


def _valid_http_url(value: Optional[str]) -> bool:
    if value is None:
        return True
    if not value or value.strip() != value or any(
        character.isspace() for character in value
    ):
        return False
    try:
        parts = urllib.parse.urlsplit(value)
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme.lower() in {"http", "https"}
        and bool(parts.hostname)
        and not parts.username
        and not parts.password
        and (port is None or 1 <= port <= 65535)
    )


def _normalized_pnu_url_host(value: Any) -> Optional[str]:
    rendered = str(value or "").strip()
    if not _valid_http_url(rendered):
        return None
    try:
        return _normalize_host(_host_from_url(rendered))
    except CurationError:
        return None


def _valid_pnu_http_url(value: Optional[str]) -> bool:
    return value is None or _normalized_pnu_url_host(value) is not None


def _candidate_rank(candidate: Mapping[str, Any]) -> Tuple[Any, ...]:
    source_url = candidate.get("source_url") or ""
    return (
        0 if candidate.get("_scope") == "core" else 1,
        0 if source_url else 1,
        0 if str(source_url).startswith("https://") else 1,
        str(candidate["input_relative_path"]).casefold(),
        str(candidate.get("download_url") or "").casefold(),
    )


def _quota_rank(candidate: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Prefer substantive attachments when a legacy host exceeds its scope cap."""

    row = candidate.get("_row") or {}
    return (
        str(candidate.get("source_host") or ""),
        0 if str(row.get("kind") or "").lower() == "attachment" else 1,
        _candidate_rank(candidate),
    )


def _reject(
    rejected: List[Dict[str, Any]],
    row: Mapping[str, Any],
    reason: str,
    detail: Optional[str] = None,
    **extra: Any
) -> None:
    actual_url = row.get("final_url") or row.get("url")
    item: Dict[str, Any] = {
        "reason": reason,
        "source_host": _host_from_url(actual_url),
        "source_url": str(row.get("url") or ""),
        "crawl_storage_path": row.get("storage_path"),
        "sha256": row.get("sha256"),
    }
    if detail:
        item["detail"] = detail
    item.update(extra)
    rejected.append(item)


def _database_rows(database: Path) -> List[Dict[str, Any]]:
    uri = "file:{}?mode=ro".format(
        urllib.parse.quote(str(database.resolve()), safe="/:")
    )
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise CurationError(
            "cannot open crawl database read-only: {}".format(exc)
        )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(frontier)")
        }
        required = {
            "url",
            "host",
            "parent_url",
            "anchor_text",
            "status",
            "kind",
            "fetched_at",
            "content_type",
            "final_url",
            "title",
            "storage_path",
            "sha256",
            "size_bytes",
        }
        missing = sorted(required - columns)
        if missing:
            raise CurationError(
                "crawl database frontier is missing columns: {}".format(
                    ", ".join(missing)
                )
            )
        canonical_select = (
            "f.canonical_url AS canonical_url"
            if "canonical_url" in columns
            else "NULL AS canonical_url"
        )
        parent_canonical_select = (
            "p.canonical_url AS parent_canonical_url"
            if "canonical_url" in columns
            else "NULL AS parent_canonical_url"
        )
        query = """
            SELECT
                f.url,
                f.host,
                f.parent_url,
                f.anchor_text,
                f.kind,
                f.fetched_at,
                f.content_type,
                f.final_url,
                f.title,
                f.storage_path,
                f.sha256,
                f.size_bytes,
                {canonical_select},
                {parent_canonical_select},
                p.title AS parent_title
            FROM frontier AS f
            LEFT JOIN frontier AS p ON p.url = f.parent_url
            WHERE f.status = 'done'
            ORDER BY
                lower(f.host),
                f.url,
                coalesce(f.storage_path, '')
        """.format(
            canonical_select=canonical_select,
            parent_canonical_select=parent_canonical_select,
        )
        return [dict(row) for row in connection.execute(query)]
    except sqlite3.Error as exc:
        raise CurationError("cannot read crawl database: {}".format(exc))
    finally:
        connection.close()


def _load_host_file(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CurationError(
            "cannot read academic host file {}: {}".format(path, exc)
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        values = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        if isinstance(parsed, dict):
            host_values = parsed.get("academic_hosts")
            if host_values is None:
                tiers = parsed.get("tiers")
                department = (
                    tiers.get("department")
                    if isinstance(tiers, dict)
                    else None
                )
                host_values = (
                    department.get("hosts")
                    if isinstance(department, dict)
                    else None
                )
            parsed = host_values
        if not isinstance(parsed, list) or any(
            not isinstance(value, str) for value in parsed
        ):
            raise CurationError(
                "academic host JSON must be a string list or "
                "{'academic_hosts': [...]}"
            )
        values = parsed
    return values


def _load_scope_file(path: Path) -> CrawlScope:
    try:
        return load_crawl_scope(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise CurationError("cannot read scope file {}: {}".format(path, exc))


def normalize_hosts(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted({_normalize_host(value) for value in values}))


def curate_corpus(
    crawl_root: Path,
    database: Path,
    output_dir: Path,
    input_root: Optional[Path] = None,
    core_hosts: Sequence[str] = DEFAULT_CORE_HOSTS,
    academic_hosts: Sequence[str] = (),
    max_documents: Optional[int] = DEFAULT_MAX_DOCUMENTS,
    max_total_bytes: Optional[int] = DEFAULT_MAX_TOTAL_BYTES,
    max_pages: Optional[int] = None,
    max_files: Optional[int] = None,
    max_items_per_host: Optional[Mapping[str, int]] = None,
    allowed_extensions: Iterable[str] = ALLOWED_EXTENSIONS,
) -> Dict[str, Any]:
    """Validate, filter, deduplicate, and atomically publish a PNU manifest."""

    crawl_root = Path(crawl_root).resolve()
    database = Path(database).resolve()
    input_root = Path(input_root or (crawl_root / "content")).resolve()
    output_dir = Path(output_dir)
    output_resolved = output_dir.resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise CurationError("output directory already exists: {}".format(output_dir))
    if not crawl_root.is_dir():
        raise CurationError("crawl root is not a directory: {}".format(crawl_root))
    if not database.is_file():
        raise CurationError("crawl database is not a file: {}".format(database))
    if not input_root.is_dir():
        raise CurationError("input root is not a directory: {}".format(input_root))
    try:
        input_root.relative_to(crawl_root)
    except ValueError:
        raise CurationError("input root must be contained by crawl root")
    if max_documents is not None and max_documents <= 0:
        raise CurationError("max_documents must be positive or null")
    if max_total_bytes is not None and max_total_bytes <= 0:
        raise CurationError("max_total_bytes must be positive or null")
    if max_pages is not None and max_pages <= 0:
        raise CurationError("max_pages must be positive or null")
    if max_files is not None and max_files <= 0:
        raise CurationError("max_files must be positive or null")

    exact_core_hosts = frozenset(normalize_hosts(core_hosts))
    exact_academic_hosts = frozenset(normalize_hosts(academic_hosts))
    overlap = sorted(exact_core_hosts & exact_academic_hosts)
    if overlap:
        raise CurationError(
            "hosts cannot be both core and academic: {}".format(", ".join(overlap))
        )
    normalized_allowed_extensions = frozenset(
        extension
        if extension.startswith(".")
        else ".{}".format(extension)
        for extension in (
            str(value).strip().lower()
            for value in allowed_extensions
        )
        if extension and extension != "."
    )
    if not normalized_allowed_extensions:
        raise CurationError("allowed_extensions cannot be empty")
    normalized_host_limits: Dict[str, int] = {}
    for raw_host, raw_limit in (max_items_per_host or {}).items():
        host = _normalize_host(raw_host)
        if (
            isinstance(raw_limit, bool)
            or not isinstance(raw_limit, int)
            or raw_limit <= 0
        ):
            raise CurationError(
                "max_items_per_host must map hosts to positive integers"
            )
        if host not in exact_core_hosts and host not in exact_academic_hosts:
            raise CurationError(
                "max_items_per_host contains an out-of-scope host: {}".format(host)
            )
        normalized_host_limits[host] = raw_limit

    rows = _database_rows(database)
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    hash_cache: Dict[Path, Tuple[int, str]] = {}

    for row in rows:
        actual_url = row.get("final_url") or row.get("url")
        host = _normalized_pnu_url_host(actual_url)
        if host is None:
            _reject(rejected, row, "host_out_of_scope")
            continue
        row["host"] = host
        if host in exact_core_hosts:
            scope = "core"
        elif host in exact_academic_hosts:
            scope = "academic"
        else:
            _reject(rejected, row, "host_out_of_scope")
            continue

        storage_path = row.get("storage_path")
        if not isinstance(storage_path, str) or not storage_path:
            _reject(rejected, row, "missing_storage_path")
            continue
        extension = Path(storage_path).suffix.lower()
        content_type = _content_type_only(row.get("content_type"))
        type_reason = _type_rejection(
            extension,
            content_type,
            normalized_allowed_extensions,
        )
        if type_reason:
            _reject(
                rejected,
                row,
                type_reason,
                detail="extension={} content_type={}".format(
                    extension or "<none>",
                    content_type or "<none>",
                ),
            )
            continue

        category, matched_term = _category_match(row)
        if scope == "academic" and category is None:
            _reject(rejected, row, "academic_irrelevant")
            continue
        if category is None:
            category = CORE_HOST_CATEGORIES.get(host, "core")
        if scope == "core":
            include_reason = "core_host:{}".format(host)
        else:
            include_reason = "academic_relevance:{}:{}".format(
                category,
                matched_term,
            )

        try:
            path, input_relative, crawl_storage_path = _safe_storage_path(
                crawl_root,
                input_root,
                storage_path,
            )
        except CurationError as exc:
            _reject(rejected, row, str(exc))
            continue

        expected_sha = str(row.get("sha256") or "").strip().lower()
        if not SHA256_RE.fullmatch(expected_sha):
            _reject(rejected, row, "invalid_database_sha256")
            continue
        expected_size = row.get("size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            _reject(rejected, row, "invalid_database_size")
            continue
        try:
            actual_size, actual_sha = hash_cache[path]
        except KeyError:
            try:
                actual_size = path.stat().st_size
                actual_sha = _sha256_path(path)
            except OSError as exc:
                _reject(
                    rejected,
                    row,
                    "unreadable_file",
                    detail="{}:{}".format(type(exc).__name__, exc),
                )
                continue
            hash_cache[path] = (actual_size, actual_sha)
        if actual_size != expected_size:
            _reject(
                rejected,
                row,
                "size_mismatch",
                detail="database={} actual={}".format(expected_size, actual_size),
            )
            continue
        if actual_sha != expected_sha:
            _reject(
                rejected,
                row,
                "sha256_mismatch",
                detail="database={} actual={}".format(expected_sha, actual_sha),
            )
            continue

        title, source_url, download_url = _source_values(row)
        canonical_url = _canonical_page_url(row, path, content_type)
        source_aliases = set()
        if canonical_url is not None:
            for alias in (
                source_url,
                str(row.get("url") or "").strip() or None,
                str(row.get("final_url") or "").strip() or None,
            ):
                if (
                    alias
                    and alias != canonical_url
                    and _valid_pnu_http_url(alias)
                ):
                    source_aliases.add(alias)
            source_url = canonical_url
        if not _valid_pnu_http_url(
            source_url
        ) or not _valid_pnu_http_url(download_url):
            _reject(rejected, row, "invalid_source_url")
            continue
        candidate: Dict[str, Any] = {
            "input_relative_path": input_relative,
            "sha256": actual_sha,
            "size_bytes": actual_size,
            "source_title": title or None,
            "source_url": source_url,
            "download_url": download_url,
            "source_host": host,
            "fetched_at": row.get("fetched_at"),
            "published_at": None,
            "crawl_storage_path": crawl_storage_path,
            "source_aliases": sorted(source_aliases),
            "category": category,
            "include_reason": include_reason,
            "_scope": scope,
            "_canonical_url": canonical_url,
            "_row": row,
        }
        candidates.append(candidate)

    canonical_groups: Dict[str, List[Dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        canonical_url = candidate.get("_canonical_url")
        key = (
            "canonical:{}".format(canonical_url)
            if canonical_url
            else "candidate:{}".format(index)
        )
        canonical_groups.setdefault(key, []).append(candidate)

    canonical_candidates: List[Dict[str, Any]] = []
    for key in sorted(canonical_groups):
        group = sorted(canonical_groups[key], key=_candidate_rank)
        canonical = group[0]
        if canonical.get("_canonical_url"):
            canonical_urls = {
                value
                for value in (
                    canonical.get("source_url"),
                    canonical.get("download_url"),
                )
                if isinstance(value, str) and value
            }
            all_urls = {
                value
                for item in group
                for value in (
                    item.get("source_url"),
                    item.get("download_url"),
                    *(item.get("source_aliases") or []),
                )
                if isinstance(value, str) and value
            }
            canonical["source_aliases"] = sorted(all_urls - canonical_urls)
        canonical_candidates.append(canonical)
        for duplicate in group[1:]:
            _reject(
                rejected,
                duplicate["_row"],
                "duplicate_canonical_url",
                canonical_input_relative_path=canonical["input_relative_path"],
                canonical_source_url=canonical.get("source_url"),
            )

    by_sha: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in canonical_candidates:
        by_sha.setdefault(str(candidate["sha256"]), []).append(candidate)

    selected_candidates: List[Dict[str, Any]] = []
    for digest in sorted(by_sha):
        group = sorted(by_sha[digest], key=_candidate_rank)
        canonical = group[0]
        canonical_urls = {
            value
            for value in (
                canonical.get("source_url"),
                canonical.get("download_url"),
            )
            if isinstance(value, str) and value
        }
        all_urls = {
            value
            for item in group
            for value in (
                item.get("source_url"),
                item.get("download_url"),
                *(item.get("source_aliases") or []),
            )
            if isinstance(value, str) and value
        }
        canonical["source_aliases"] = sorted(all_urls - canonical_urls)
        selected_candidates.append(canonical)
        for duplicate in group[1:]:
            _reject(
                rejected,
                duplicate["_row"],
                "duplicate_sha256",
                canonical_input_relative_path=canonical["input_relative_path"],
                canonical_source_url=canonical.get("source_url"),
            )

    selected_candidates.sort(key=_quota_rank)
    selected: List[Dict[str, Any]] = []
    selected_per_host: Counter[str] = Counter()
    selected_kinds: Counter[str] = Counter()
    for candidate in selected_candidates:
        host = str(candidate["source_host"])
        host_limit = normalized_host_limits.get(host)
        if (
            host_limit is not None
            and selected_per_host[host] >= host_limit
        ):
            _reject(
                rejected,
                candidate["_row"],
                "max_items_per_host",
                detail="limit={}".format(host_limit),
            )
            continue
        selected_per_host[host] += 1
        selected_kinds[
            "attachment"
            if str(candidate["_row"].get("kind") or "").lower()
            == "attachment"
            else "page"
        ] += 1
        selected.append(
            {field: candidate.get(field) for field in MANIFEST_FIELDS}
        )
    selected.sort(key=lambda item: str(item["input_relative_path"]))
    rejected.sort(
        key=lambda item: (
            str(item.get("reason") or ""),
            str(item.get("source_host") or ""),
            str(item.get("source_url") or ""),
            str(item.get("crawl_storage_path") or ""),
        )
    )
    total_bytes = sum(int(item["size_bytes"]) for item in selected)
    if max_pages is not None and selected_kinds["page"] > max_pages:
        raise CurationError(
            "curated corpus exceeds page budget: {} > {}".format(
                selected_kinds["page"],
                max_pages,
            )
        )
    if max_files is not None and selected_kinds["attachment"] > max_files:
        raise CurationError(
            "curated corpus exceeds file budget: {} > {}".format(
                selected_kinds["attachment"],
                max_files,
            )
        )
    if max_documents is not None and len(selected) > max_documents:
        raise CurationError(
            "curated corpus exceeds document budget: {} > {}".format(
                len(selected),
                max_documents,
            )
        )
    if max_total_bytes is not None and total_bytes > max_total_bytes:
        raise CurationError(
            "curated corpus exceeds byte budget: {} > {}".format(
                total_bytes,
                max_total_bytes,
            )
        )
    if not selected:
        raise CurationError("curated corpus is empty")

    parent = output_resolved.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".{}.".format(output_resolved.name), dir=str(parent))
    )
    try:
        manifest_path = temporary / "curated-manifest.jsonl"
        rejected_path = temporary / "rejected.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in selected:
                handle.write(_canonical_json(item) + "\n")
        with rejected_path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in rejected:
                handle.write(_canonical_json(item) + "\n")

        reason_counts = Counter(str(item["reason"]) for item in rejected)
        host_counts = Counter(str(item["source_host"]) for item in selected)
        category_counts = Counter(str(item["category"]) for item in selected)
        extension_counts = Counter(
            Path(str(item["input_relative_path"])).suffix.lower()
            for item in selected
        )
        summary: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "crawl_root": str(crawl_root),
            "database": str(database),
            "input_root": str(input_root),
            "output_dir": str(output_resolved),
            "core_hosts": sorted(exact_core_hosts),
            "academic_hosts": sorted(exact_academic_hosts),
            "source_done_rows": len(rows),
            "selected_documents": len(selected),
            "selected_bytes": total_bytes,
            "rejected_rows": len(rejected),
            "duplicate_rows": reason_counts.get("duplicate_sha256", 0),
            "max_documents": max_documents,
            "max_total_bytes": max_total_bytes,
            "max_pages": max_pages,
            "max_files": max_files,
            "selected_pages": selected_kinds["page"],
            "selected_files": selected_kinds["attachment"],
            "max_items_per_host": dict(sorted(normalized_host_limits.items())),
            "allowed_extensions": sorted(normalized_allowed_extensions),
            "categories": dict(sorted(category_counts.items())),
            "hosts": dict(sorted(host_counts.items())),
            "extensions": dict(sorted(extension_counts.items())),
            "rejection_reasons": dict(sorted(reason_counts.items())),
            "manifest_sha256": _sha256_path(manifest_path),
            "rejected_sha256": _sha256_path(rejected_path),
        }
        (temporary / "summary.json").write_text(
            _canonical_json(summary) + "\n",
            encoding="utf-8",
        )
        if output_resolved.exists() or output_resolved.is_symlink():
            raise CurationError(
                "output directory appeared during publication: {}".format(
                    output_resolved
                )
            )
        os.rename(str(temporary), str(output_resolved))
        return summary
    except BaseException:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crawl-root",
        type=Path,
        default=Path("downloads/pnu-web-crawl"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="defaults to CRAWL_ROOT/state/crawl.sqlite3",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="defaults to CRAWL_ROOT/content",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--academic-host",
        action="append",
        default=[],
        help="exact department/college host; may be repeated",
    )
    parser.add_argument(
        "--academic-hosts-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON string list, newline-delimited exact hosts, or a scope "
            "JSON containing tiers.department.hosts"
        ),
    )
    parser.add_argument(
        "--scope-file",
        type=Path,
        help=(
            "crawl scope JSON; replaces default core hosts, adds "
            "tiers.department.hosts, and supplies budgets"
        ),
    )
    parser.add_argument(
        "--core-host",
        action="append",
        default=[],
        help="add an exact host to the active default or scope-file core set",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-total-bytes",
        type=int,
        default=None,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    crawl_root = args.crawl_root
    database = args.database or (crawl_root / "state" / "crawl.sqlite3")
    input_root = args.input_root or (crawl_root / "content")
    academic_hosts: List[str] = list(args.academic_host)
    try:
        scope_core_hosts: Optional[List[str]] = None
        scope_max_documents: Optional[int] = None
        scope_max_total_bytes: Optional[int] = None
        scope_max_pages: Optional[int] = None
        scope_max_files: Optional[int] = None
        scope_host_limits: Dict[str, int] = {}
        scope_allowed_extensions = ALLOWED_EXTENSIONS
        core_host_limit: Optional[int] = None
        academic_host_limit: Optional[int] = None
        if args.scope_file is not None:
            scope = _load_scope_file(args.scope_file)
            tiers = {tier.name: tier for tier in scope.tiers}
            if "core" not in tiers or "department" not in tiers:
                raise CurationError(
                    "scope file must define core and department tiers"
                )
            core_tier = tiers["core"]
            academic_tier = tiers["department"]
            scope_core_hosts = list(core_tier.hosts)
            scope_academic_hosts = list(academic_tier.hosts)
            academic_hosts.extend(scope_academic_hosts)
            core_host_limit = core_tier.max_items_per_host
            academic_host_limit = academic_tier.max_items_per_host
            scope_host_limits.update(
                {
                    host: tier.max_items_per_host
                    for tier in scope.tiers
                    for host in tier.hosts
                }
            )
            scope_allowed_extensions = (
                PAGE_EXTENSIONS
                | frozenset(scope.allowed_document_extensions)
            )
            scope_max_documents = scope.max_pages + scope.max_files
            scope_max_total_bytes = scope.max_bytes
            scope_max_pages = scope.max_pages
            scope_max_files = scope.max_files
        for path in args.academic_hosts_file:
            academic_hosts.extend(_load_host_file(path))
        core_hosts = tuple(
            scope_core_hosts
            if scope_core_hosts is not None
            else DEFAULT_CORE_HOSTS
        ) + tuple(args.core_host)
        if core_host_limit is not None:
            for host in core_hosts:
                scope_host_limits.setdefault(
                    _normalize_host(host),
                    core_host_limit,
                )
        if academic_host_limit is not None:
            for host in academic_hosts:
                scope_host_limits.setdefault(
                    _normalize_host(host),
                    academic_host_limit,
                )
        summary = curate_corpus(
            crawl_root=crawl_root,
            database=database,
            input_root=input_root,
            output_dir=args.output_dir,
            core_hosts=core_hosts,
            academic_hosts=academic_hosts,
            max_pages=scope_max_pages,
            max_files=scope_max_files,
            max_items_per_host=scope_host_limits,
            allowed_extensions=scope_allowed_extensions,
            max_documents=(
                args.max_documents
                if args.max_documents is not None
                else (
                    scope_max_documents
                    if scope_max_documents is not None
                    else DEFAULT_MAX_DOCUMENTS
                )
            ),
            max_total_bytes=(
                args.max_total_bytes
                if args.max_total_bytes is not None
                else (
                    scope_max_total_bytes
                    if scope_max_total_bytes is not None
                    else DEFAULT_MAX_TOTAL_BYTES
                )
            ),
        )
    except CurationError as exc:
        parser.exit(2, "error: {}\n".format(exc))
    sys.stdout.write(_canonical_json(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
