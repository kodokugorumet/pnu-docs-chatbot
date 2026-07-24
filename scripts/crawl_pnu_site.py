#!/usr/bin/env python3
"""Crawl public Pusan National University web pages and attachments.

The crawler starts from official PNU pages, follows links across public
``*.pusan.ac.kr`` department and institution sites, and stores both HTML pages
and document attachments for the parser pipeline.

It is deliberately conservative:

* GET requests only; forms are never submitted.
* robots.txt is honored and each host is rate limited.
* login/admin/search traps and static assets are excluded.
* a SQLite frontier makes interrupted crawls resumable.
* content and crawl state are stored separately so parser input stays clean.

Examples:
  python3 scripts/crawl_pnu_site.py --max-pages 100 --max-depth 3
  python3 scripts/crawl_pnu_site.py --max-pages 5000 --delay 1.5
  python3 scripts/crawl_pnu_site.py --status
"""

from __future__ import annotations

import argparse
import hashlib
import html
import ipaddress
import json
import mimetypes
import os
import posixpath
import re
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


DEFAULT_SEEDS = (
    "https://www.pusan.ac.kr/kor/Main.do",
    "https://www.pusan.ac.kr/kor/CMS/Contents/Contents.do?mCode=MN003",
    "https://www.pusan.ac.kr/kor/CMS/Board/Board.do?mCode=MN095",
    "https://www.pusan.ac.kr/kor/CMS/Haksailjung/view.do?mCode=MN076",
)
DEFAULT_ALLOWED_DOMAINS = ("pusan.ac.kr",)
USER_AGENT = (
    "PNU-Graduation-Project-Crawler/1.0 "
    "(public academic information crawler; contact: local-research-project)"
)
HTML_CONTENT_TYPES = {
    "application/xhtml+xml",
    "text/html",
}
DOCUMENT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".hwp",
    ".hwpx",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
    ".txt",
    ".xls",
    ".xlsx",
}
STATIC_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".css",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".m4a",
    ".map",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".ogg",
    ".png",
    ".rar",
    ".rss",
    ".svg",
    ".tar",
    ".tif",
    ".tiff",
    ".ts",
    ".ttf",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xml",
    ".zip",
}
DOWNLOAD_PATH_HINTS = (
    "/download",
    "download.do",
    "filedown",
    "file_down",
    "filedownload",
    "atchfile",
    "attachfile",
)
BLOCKED_HOST_LABELS = {
    "mail",
    "my",
    "one",
    "onestop",
    "plato",
    "sso",
    "webmail",
}
BLOCKED_PATH_PARTS = (
    "/admin",
    "/cmsmanager",
    "/join",
    "/lgn/",
    "/login",
    "/logout",
    "/member/",
    "/mypage",
    "/search/",
    "/sso/",
)
TRACKING_QUERY_KEYS = {
    "_",
    "fbclid",
    "gclid",
    "layout",
    "sessionid",
    "sid",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}
PAGE_QUERY_KEYS = {
    "page",
    "pageindex",
    "pageno",
    "pageunit",
}
HIGH_VALUE_TERMS = (
    "학사",
    "수업",
    "수강",
    "교육과정",
    "졸업",
    "학적",
    "등록",
    "등록금",
    "장학",
    "학생지원",
    "취업",
    "진로",
    "휴학",
    "복학",
    "전과",
    "복수전공",
    "부전공",
    "공지",
    "notice",
    "scholar",
    "graduate",
    "curriculum",
)
LOW_VALUE_TERMS = (
    "갤러리",
    "동문",
    "포토",
    "홍보영상",
    "facebook",
    "instagram",
    "youtube",
)
FILENAME_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
MULTISLASH = re.compile(r"/{2,}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_type_only(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def safe_filename(value: str, fallback: str = "download") -> str:
    value = html.unescape(value).strip().replace("\u00a0", " ")
    value = FILENAME_BAD_CHARS.sub("_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        value = fallback
    if len(value) > 160:
        stem, extension = os.path.splitext(value)
        value = f"{stem[:140]}{extension[:20]}"
    return value


def host_matches(host: str, allowed_domains: Sequence[str]) -> bool:
    host = host.rstrip(".").lower()
    return any(
        host == domain.rstrip(".").lower()
        or host.endswith(f".{domain.rstrip('.').lower()}")
        for domain in allowed_domains
    )


def canonicalize_url(url: str, base_url: str | None = None) -> str | None:
    """Return a stable HTTP(S) URL or None for unsupported links."""
    value = html.unescape(url).strip()
    if not value or value.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
        return None
    if base_url:
        value = urllib.parse.urljoin(base_url, value)
    try:
        parts = urllib.parse.urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    host = parts.hostname.rstrip(".").lower()
    if ":" in host and not host.startswith("["):
        rendered_host = f"[{host}]"
    else:
        rendered_host = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{rendered_host}:{port}"
    else:
        netloc = rendered_host
    path = urllib.parse.unquote(parts.path or "/", errors="replace")
    path = MULTISLASH.sub("/", path)
    normalized = posixpath.normpath(path)
    if path.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    path = urllib.parse.quote(normalized, safe="/:@-._~!$&'()*+,;=")
    query_items: list[tuple[str, str]] = []
    for key, query_value in urllib.parse.parse_qsl(parts.query, keep_blank_values=False):
        normalized_key = key.strip()
        if normalized_key.lower() in TRACKING_QUERY_KEYS:
            continue
        if normalized_key.lower() in PAGE_QUERY_KEYS:
            try:
                if int(query_value) > 500:
                    continue
            except ValueError:
                continue
        if len(normalized_key) > 80 or len(query_value) > 1200:
            continue
        query_items.append((normalized_key, query_value))
    query_items.sort(key=lambda item: (item[0].lower(), item[1]))
    query = urllib.parse.urlencode(query_items, doseq=True)
    result = urllib.parse.urlunsplit((scheme, netloc, path, query, ""))
    return result if len(result) <= 4096 else None


def is_document_url(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    extension = Path(urllib.parse.unquote(parts.path)).suffix.lower()
    lowered = parts.path.lower()
    return extension in DOCUMENT_EXTENSIONS or any(hint in lowered for hint in DOWNLOAD_PATH_HINTS)


def should_skip_url(url: str) -> tuple[bool, str | None]:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    first_label = host.split(".", 1)[0]
    lowered_path = urllib.parse.unquote(parts.path).lower()
    extension = Path(lowered_path).suffix.lower()
    if first_label in BLOCKED_HOST_LABELS:
        return True, "blocked_host"
    if any(part in lowered_path for part in BLOCKED_PATH_PARTS):
        return True, "blocked_path"
    if extension in STATIC_EXTENSIONS:
        return True, "static_asset"
    if re.search(r"/(write|edit|delete|insert|update|save)(?:[./]|$)", lowered_path):
        return True, "mutating_route"
    return False, None


def link_priority(url: str, text: str, *, seed: bool = False) -> int:
    if seed:
        return 1000
    searchable = f"{urllib.parse.unquote(url)} {text}".lower()
    score = 0
    score += sum(12 for term in HIGH_VALUE_TERMS if term.lower() in searchable)
    score -= sum(8 for term in LOW_VALUE_TERMS if term.lower() in searchable)
    if "/bbs/" in searchable or "board" in searchable:
        score += 8
    if "artclview" in searchable or "view.do" in searchable:
        score += 6
    if is_document_url(url):
        score += 16
    if urllib.parse.urlsplit(url).hostname != "www.pusan.ac.kr":
        score += 2
    return max(-100, min(score, 999))


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.title_parts: list[str] = []
        self.base_href: str | None = None
        self.canonical_href: str | None = None
        self._current_href: str | None = None
        self._current_text: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value for key, value in attrs}
        lowered = tag.lower()
        if lowered == "a" and attributes.get("href"):
            self._current_href = attributes["href"]
            self._current_text = []
        elif lowered == "base" and attributes.get("href") and not self.base_href:
            self.base_href = attributes["href"]
        elif lowered == "link":
            rel = (attributes.get("rel") or "").lower().split()
            if "canonical" in rel and attributes.get("href"):
                self.canonical_href = attributes["href"]
        elif lowered == "title":
            self._in_title = True

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)
        if self._in_title:
            self.title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "a" and self._current_href is not None:
            text = re.sub(r"\s+", " ", " ".join(self._current_text)).strip()
            self.links.append((self._current_href, text))
            self._current_href = None
            self._current_text = []
        elif lowered == "title":
            self._in_title = False

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()


def decode_html(value: bytes, content_type: str | None) -> str:
    candidates: list[str] = []
    if content_type:
        message = Message()
        message["content-type"] = content_type
        charset = message.get_content_charset()
        if charset:
            candidates.append(charset)
    head = value[:8192].decode("ascii", errors="ignore")
    match = re.search(
        r"""charset\s*=\s*["']?\s*([a-zA-Z0-9._-]+)""",
        head,
        re.IGNORECASE,
    )
    if match:
        candidates.append(match.group(1))
    candidates.extend(("utf-8", "euc-kr", "cp949"))
    for encoding in dict.fromkeys(candidates):
        try:
            return value.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return value.decode("utf-8", errors="replace")


def filename_from_headers(
    headers: Message,
    url: str,
    content_type: str,
) -> str:
    disposition = headers.get("Content-Disposition", "")
    filename: str | None = None
    if disposition:
        message = Message()
        message["content-disposition"] = disposition
        filename = message.get_filename()
        if filename:
            if "%" in filename or "+" in filename:
                filename = urllib.parse.unquote_plus(filename)
            try:
                filename = filename.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                for encoding in ("cp949", "euc-kr"):
                    try:
                        filename = filename.encode("latin-1").decode(encoding)
                        break
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        continue
    if not filename:
        filename = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name
    if not filename or filename.lower().endswith(
        (".do", ".jsp", ".php", ".asp", ".aspx")
    ):
        extension = mimetypes.guess_extension(content_type) or ""
        filename = f"attachment{extension}"
    return safe_filename(filename)


def response_is_document(url: str, headers: Message, content_type: str) -> bool:
    disposition = (headers.get("Content-Disposition") or "").lower()
    extension = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).suffix.lower()
    if "attachment" in disposition or "filename=" in disposition or "filename*=" in disposition:
        return True
    if extension in DOCUMENT_EXTENSIONS:
        return True
    if content_type in HTML_CONTENT_TYPES or content_type.startswith("text/html"):
        return False
    return (
        content_type.startswith("application/")
        or content_type.startswith("text/plain")
        or is_document_url(url)
    )


def is_public_host(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    if not addresses:
        return False
    for address in addresses:
        value = ipaddress.ip_address(address[4][0])
        if (
            value.is_private
            or value.is_loopback
            or value.is_link_local
            or value.is_multicast
            or value.is_reserved
            or value.is_unspecified
        ):
            return False
    return True


class GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], bool]) -> None:
        super().__init__()
        self.validator = validator

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: Message,
        new_url: str,
    ) -> urllib.request.Request:
        target = canonicalize_url(new_url, request.full_url)
        if not target or not self.validator(target):
            raise urllib.error.HTTPError(
                request.full_url,
                code,
                f"redirect blocked: {new_url}",
                headers,
                file_pointer,
            )
        redirected = super().redirect_request(
            request, file_pointer, code, message, headers, target
        )
        if redirected is None:
            raise urllib.error.HTTPError(
                request.full_url, code, "redirect rejected", headers, file_pointer
            )
        return redirected


@dataclass
class CrawlConfig:
    output: Path
    seeds: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    max_pages: int
    max_files: int
    max_depth: int
    max_pages_per_host: int
    max_response_bytes: int
    timeout: float
    delay: float
    retries: int
    obey_robots: bool
    allow_private: bool
    refresh: bool
    dry_run: bool


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    headers: Message
    body: bytes
    content_type: str


class CrawlStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS frontier (
                url TEXT PRIMARY KEY,
                host TEXT NOT NULL,
                depth INTEGER NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                parent_url TEXT,
                anchor_text TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                kind TEXT NOT NULL DEFAULT 'page',
                attempts INTEGER NOT NULL DEFAULT 0,
                discovered_at TEXT NOT NULL,
                fetched_at TEXT,
                http_status INTEGER,
                content_type TEXT,
                final_url TEXT,
                title TEXT,
                storage_path TEXT,
                sha256 TEXT,
                size_bytes INTEGER,
                error TEXT
            );
            CREATE INDEX IF NOT EXISTS frontier_status_priority
                ON frontier(status, priority DESC, depth, discovered_at);
            CREATE INDEX IF NOT EXISTS frontier_host_status
                ON frontier(host, status);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                event TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            "UPDATE frontier SET status = 'queued' WHERE status = 'fetching'"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def enqueue(
        self,
        url: str,
        depth: int,
        priority: int,
        parent_url: str | None,
        anchor_text: str,
        kind: str,
    ) -> bool:
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO frontier(
                url, host, depth, priority, parent_url, anchor_text,
                status, kind, discovered_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                url,
                urllib.parse.urlsplit(url).hostname or "",
                depth,
                priority,
                parent_url,
                anchor_text[:500],
                kind,
                utc_now(),
            ),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def claim_next(self, kind: str | None = None) -> sqlite3.Row | None:
        kind_filter = "AND kind = ?" if kind else ""
        parameters: tuple[object, ...] = (kind,) if kind else ()
        row = self.connection.execute(
            f"""
            SELECT * FROM frontier
            WHERE status = 'queued'
            {kind_filter}
            ORDER BY priority DESC, depth ASC, discovered_at ASC, url ASC
            LIMIT 1
            """,
            parameters,
        ).fetchone()
        if row is None:
            return None
        self.connection.execute(
            """
            UPDATE frontier
            SET status = 'fetching', attempts = attempts + 1
            WHERE url = ?
            """,
            (row["url"],),
        )
        self.connection.commit()
        return row

    def finish(self, url: str, **values: object) -> None:
        assignments = ["status = 'done'", "fetched_at = ?"]
        parameters: list[object] = [utc_now()]
        for key in (
            "http_status",
            "content_type",
            "final_url",
            "title",
            "storage_path",
            "sha256",
            "size_bytes",
            "kind",
        ):
            if key in values:
                assignments.append(f"{key} = ?")
                parameters.append(values[key])
        assignments.append("error = NULL")
        parameters.append(url)
        self.connection.execute(
            f"UPDATE frontier SET {', '.join(assignments)} WHERE url = ?",
            parameters,
        )
        self.connection.commit()

    def mark(self, url: str, status: str, error: str) -> None:
        self.connection.execute(
            """
            UPDATE frontier
            SET status = ?, error = ?, fetched_at = ?
            WHERE url = ?
            """,
            (status, error[:1000], utc_now(), url),
        )
        self.connection.commit()

    def log(self, url: str | None, event: str, detail: str = "") -> None:
        self.connection.execute(
            "INSERT INTO events(url, event, detail, created_at) VALUES (?, ?, ?, ?)",
            (url, event, detail[:2000], utc_now()),
        )
        self.connection.commit()

    def host_done_count(self, host: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM frontier WHERE host = ? AND status = 'done'",
            (host,),
        ).fetchone()
        return int(row["count"])

    def reset_for_refresh(self) -> None:
        self.connection.execute(
            """
            UPDATE frontier
            SET status = 'queued', error = NULL
            WHERE status IN ('done', 'error')
            """
        )
        self.connection.commit()

    def counts(self) -> dict[str, int]:
        result = {
            row["status"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT status, COUNT(*) AS count FROM frontier GROUP BY status"
            )
        }
        result["total"] = sum(result.values())
        result["pages"] = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM frontier WHERE status = 'done' AND kind = 'page'"
            ).fetchone()[0]
        )
        result["files"] = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM frontier WHERE status = 'done' AND kind = 'attachment'"
            ).fetchone()[0]
        )
        result["bytes"] = int(
            self.connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM frontier WHERE status = 'done'"
            ).fetchone()[0]
        )
        result["hosts"] = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT host) FROM frontier WHERE status = 'done'"
            ).fetchone()[0]
        )
        return result

    def iter_rows(self, kind: str) -> Iterator[sqlite3.Row]:
        yield from self.connection.execute(
            """
            SELECT * FROM frontier
            WHERE status = 'done' AND kind = ?
            ORDER BY fetched_at, url
            """,
            (kind,),
        )


class PnuCrawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.state_dir = config.output / "state"
        self.content_dir = config.output / "content" / "부산대학교"
        self.store = CrawlStore(self.state_dir / "crawl.sqlite3")
        self.last_request_at: dict[str, float] = {}
        self.public_host_cache: dict[str, bool] = {}
        self.robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.opener = urllib.request.build_opener(
            GuardedRedirectHandler(self.url_is_fetchable)
        )
        if config.refresh:
            self.store.reset_for_refresh()

    def close(self) -> None:
        self.store.close()

    def url_is_fetchable(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme not in {"http", "https"}:
            return False
        if not host_matches(host, self.config.allowed_domains):
            return False
        skipped, _ = should_skip_url(url)
        if skipped:
            return False
        if self.config.allow_private:
            return True
        if host not in self.public_host_cache:
            self.public_host_cache[host] = is_public_host(host)
        return self.public_host_cache[host]

    def seed(self) -> int:
        inserted = 0
        for raw_url in self.config.seeds:
            url = canonicalize_url(raw_url)
            if not url:
                self.store.log(None, "seed_rejected", raw_url)
                continue
            if not self.url_is_fetchable(url):
                self.store.log(url, "seed_rejected", "outside allowed public scope")
                continue
            inserted += int(
                self.store.enqueue(
                    url=url,
                    depth=0,
                    priority=link_priority(url, "", seed=True),
                    parent_url=None,
                    anchor_text="seed",
                    kind="attachment" if is_document_url(url) else "page",
                )
            )
        return inserted

    def wait_for_host(self, host: str) -> None:
        previous = self.last_request_at.get(host)
        if previous is not None:
            remaining = self.config.delay - (time.monotonic() - previous)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at[host] = time.monotonic()

    def robots_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        if not self.config.obey_robots:
            return None
        parts = urllib.parse.urlsplit(url)
        host_key = f"{parts.scheme}://{parts.netloc}"
        if host_key in self.robots_cache:
            return self.robots_cache[host_key]
        robots_url = f"{host_key}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            result = self.fetch(robots_url, check_robots=False, max_bytes=1024 * 1024)
        except (OSError, urllib.error.URLError, ValueError) as error:
            self.store.log(robots_url, "robots_unavailable", str(error))
            self.robots_cache[host_key] = None
            return None
        if result.status == 200:
            parser.parse(decode_html(result.body, result.headers.get("Content-Type")).splitlines())
            self.robots_cache[host_key] = parser
            return parser
        self.robots_cache[host_key] = None
        return None

    def robots_allows(self, url: str) -> bool:
        parser = self.robots_for(url)
        return parser is None or parser.can_fetch(USER_AGENT, url)

    def fetch(
        self,
        url: str,
        *,
        check_robots: bool = True,
        max_bytes: int | None = None,
    ) -> FetchResult:
        if not self.url_is_fetchable(url):
            raise ValueError("URL is outside allowed public scope")
        if check_robots and not self.robots_allows(url):
            raise PermissionError("blocked by robots.txt")
        limit = max_bytes or self.config.max_response_bytes
        host = urllib.parse.urlsplit(url).hostname or ""
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            self.wait_for_host(host)
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": (
                        "text/html,application/xhtml+xml,application/pdf,"
                        "application/octet-stream;q=0.9,*/*;q=0.5"
                    ),
                    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
                    "User-Agent": USER_AGENT,
                },
                method="GET",
            )
            try:
                with self.opener.open(request, timeout=self.config.timeout) as response:
                    declared_length = response.headers.get("Content-Length")
                    if declared_length and int(declared_length) > limit:
                        raise ValueError(
                            f"response too large: {declared_length} > {limit} bytes"
                        )
                    body = response.read(limit + 1)
                    if len(body) > limit:
                        raise ValueError(f"response exceeded {limit} bytes")
                    final_url = canonicalize_url(response.geturl()) or url
                    if not self.url_is_fetchable(final_url):
                        raise ValueError("final URL is outside allowed public scope")
                    return FetchResult(
                        requested_url=url,
                        final_url=final_url,
                        status=int(response.status),
                        headers=response.headers,
                        body=body,
                        content_type=content_type_only(
                            response.headers.get("Content-Type")
                        ),
                    )
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt >= self.config.retries:
                    raise
                retry_after = error.headers.get("Retry-After")
                try:
                    pause = float(retry_after) if retry_after else 2**attempt
                except ValueError:
                    pause = 2**attempt
                time.sleep(min(max(pause, self.config.delay), 30.0))
                last_error = error
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
                if attempt >= self.config.retries:
                    raise
                time.sleep(min(max(2**attempt, self.config.delay), 30.0))
        if last_error:
            raise last_error
        raise RuntimeError("fetch failed without an error")

    def html_storage_path(self, url: str) -> Path:
        parts = urllib.parse.urlsplit(url)
        host = safe_filename(parts.hostname or "unknown")
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        path_name = Path(urllib.parse.unquote(parts.path)).stem
        page_name = safe_filename(path_name, fallback="index")[:60]
        return self.content_dir / "웹페이지" / host / f"{page_name}-{digest}.html"

    def attachment_storage_path(
        self,
        url: str,
        headers: Message,
        content_type: str,
        digest: str,
    ) -> Path:
        host = safe_filename(urllib.parse.urlsplit(url).hostname or "unknown")
        filename = filename_from_headers(headers, url, content_type)
        return self.content_dir / "첨부파일" / host / f"{digest[:16]}-{filename}"

    def process_html(self, row: sqlite3.Row, result: FetchResult) -> None:
        decoded = decode_html(result.body, result.headers.get("Content-Type"))
        parser = LinkParser()
        try:
            parser.feed(decoded)
        except Exception as error:
            self.store.log(row["url"], "html_parse_warning", str(error))
        storage_path = self.html_storage_path(result.final_url)
        digest = sha256_bytes(result.body)
        if not self.config.dry_run:
            atomic_write_bytes(storage_path, result.body)
        relative_path = (
            str(storage_path.relative_to(self.config.output))
            if not self.config.dry_run
            else None
        )
        self.store.finish(
            row["url"],
            http_status=result.status,
            content_type=result.content_type,
            final_url=result.final_url,
            title=parser.title[:500],
            storage_path=relative_path,
            sha256=digest,
            size_bytes=len(result.body),
            kind="page",
        )
        if int(row["depth"]) >= self.config.max_depth:
            return
        link_base = canonicalize_url(parser.base_href or "", result.final_url)
        base_url = link_base or result.final_url
        discovered: set[str] = set()
        for raw_link, anchor_text in parser.links:
            url = canonicalize_url(raw_link, base_url)
            if not url or url in discovered:
                continue
            discovered.add(url)
            if not self.url_is_fetchable(url):
                continue
            skipped, reason = should_skip_url(url)
            if skipped:
                self.store.log(url, "link_skipped", reason or "filtered")
                continue
            kind = "attachment" if is_document_url(url) else "page"
            self.store.enqueue(
                url=url,
                depth=int(row["depth"]) + 1,
                priority=link_priority(url, anchor_text),
                parent_url=result.final_url,
                anchor_text=anchor_text,
                kind=kind,
            )

    def process_attachment(self, row: sqlite3.Row, result: FetchResult) -> None:
        digest = sha256_bytes(result.body)
        storage_path = self.attachment_storage_path(
            result.final_url, result.headers, result.content_type, digest
        )
        if not self.config.dry_run:
            atomic_write_bytes(storage_path, result.body)
        relative_path = (
            str(storage_path.relative_to(self.config.output))
            if not self.config.dry_run
            else None
        )
        self.store.finish(
            row["url"],
            http_status=result.status,
            content_type=result.content_type,
            final_url=result.final_url,
            title=storage_path.name,
            storage_path=relative_path,
            sha256=digest,
            size_bytes=len(result.body),
            kind="attachment",
        )

    def run(self) -> dict[str, object]:
        started_at = utc_now()
        self.seed()
        if self.store.counts()["total"] == 0:
            raise RuntimeError(
                "수집 가능한 seed URL이 없습니다. 네트워크/DNS와 --seed, "
                "--allow-domain 설정을 확인하세요."
            )
        pages_this_run = 0
        files_this_run = 0
        errors_this_run = 0
        skipped_this_run = 0
        while True:
            if (
                pages_this_run >= self.config.max_pages
                and files_this_run >= self.config.max_files
            ):
                break
            if pages_this_run >= self.config.max_pages:
                next_kind = "attachment"
            elif files_this_run >= self.config.max_files:
                next_kind = "page"
            else:
                next_kind = None
            row = self.store.claim_next(next_kind)
            if row is None:
                break
            url = str(row["url"])
            host = str(row["host"])
            if int(row["depth"]) > self.config.max_depth:
                self.store.mark(url, "skipped", "max_depth")
                skipped_this_run += 1
                continue
            if self.store.host_done_count(host) >= self.config.max_pages_per_host:
                self.store.mark(url, "skipped", "max_pages_per_host")
                skipped_this_run += 1
                continue
            try:
                result = self.fetch(url)
                if response_is_document(
                    result.final_url, result.headers, result.content_type
                ):
                    self.process_attachment(row, result)
                    files_this_run += 1
                    print(f"[file] {result.final_url}", flush=True)
                elif (
                    result.content_type in HTML_CONTENT_TYPES
                    or result.content_type.startswith("text/html")
                    or not result.content_type
                ):
                    self.process_html(row, result)
                    pages_this_run += 1
                    print(
                        f"[page {pages_this_run}/{self.config.max_pages}] "
                        f"{result.final_url}",
                        flush=True,
                    )
                else:
                    self.store.mark(
                        url,
                        "skipped",
                        f"unsupported content type: {result.content_type}",
                    )
                    skipped_this_run += 1
            except PermissionError as error:
                self.store.mark(url, "skipped", str(error))
                skipped_this_run += 1
            except (OSError, ValueError, urllib.error.URLError) as error:
                self.store.mark(url, "error", f"{type(error).__name__}: {error}")
                errors_this_run += 1
                print(f"[error] {url}: {error}", file=sys.stderr, flush=True)
        summary: dict[str, object] = {
            "started_at": started_at,
            "finished_at": utc_now(),
            "output": str(self.config.output),
            "dry_run": self.config.dry_run,
            "this_run": {
                "pages": pages_this_run,
                "files": files_this_run,
                "errors": errors_this_run,
                "skipped": skipped_this_run,
            },
            "database": self.store.counts(),
            "config": {
                **asdict(self.config),
                "output": str(self.config.output),
            },
        }
        self.export_manifests(summary)
        return summary

    def export_manifests(self, summary: dict[str, object]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        for kind, filename in (
            ("page", "pages.jsonl"),
            ("attachment", "attachments.jsonl"),
        ):
            lines = []
            for row in self.store.iter_rows(kind):
                record = dict(row)
                record["source_url"] = record.pop("final_url") or record["url"]
                lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
            atomic_write_text(
                self.state_dir / filename,
                "\n".join(lines) + ("\n" if lines else ""),
            )
        atomic_write_text(
            self.config.output / "summary.json",
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl public PNU pages and attachments across official "
            "*.pusan.ac.kr sites."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("downloads/pnu-web-crawl"),
        help="crawl root (default: downloads/pnu-web-crawl)",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="additional seed URL; repeatable",
    )
    parser.add_argument(
        "--replace-default-seeds",
        action="store_true",
        help="use only URLs passed with --seed",
    )
    parser.add_argument(
        "--allow-domain",
        action="append",
        default=[],
        help="additional allowed domain suffix; repeatable",
    )
    parser.add_argument("--max-pages", type=positive_int, default=1000)
    parser.add_argument("--max-files", type=positive_int, default=1000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-pages-per-host", type=positive_int, default=300)
    parser.add_argument(
        "--max-response-mb",
        type=positive_int,
        default=50,
        help="maximum response body size in MiB",
    )
    parser.add_argument("--timeout", type=non_negative_float, default=20.0)
    parser.add_argument(
        "--delay",
        type=non_negative_float,
        default=1.0,
        help="minimum delay between requests to the same host",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="ignore robots.txt (not recommended)",
    )
    parser.add_argument(
        "--allow-private",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch completed/error URLs while retaining the frontier",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="crawl and record metadata without saving response bodies",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="show the existing crawl database summary and exit",
    )
    return parser


def resolve_seeds(arguments: argparse.Namespace) -> tuple[str, ...]:
    if arguments.replace_default_seeds:
        if not arguments.seed:
            raise ValueError("--replace-default-seeds requires at least one --seed")
        return tuple(arguments.seed)
    return tuple(dict.fromkeys((*DEFAULT_SEEDS, *arguments.seed)))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    database_path = output / "state" / "crawl.sqlite3"
    if arguments.status:
        if not database_path.exists():
            print(json.dumps({"output": str(output), "status": "not_started"}, ensure_ascii=False))
            return 0
        store = CrawlStore(database_path)
        try:
            print(json.dumps(store.counts(), ensure_ascii=False, indent=2, sort_keys=True))
        finally:
            store.close()
        return 0
    if arguments.max_depth < 0:
        print("error: --max-depth must be zero or greater", file=sys.stderr)
        return 2
    if arguments.retries < 0:
        print("error: --retries must be zero or greater", file=sys.stderr)
        return 2
    try:
        seeds = resolve_seeds(arguments)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    allowed_domains = tuple(
        dict.fromkeys(
            domain.strip().lower().rstrip(".")
            for domain in (*DEFAULT_ALLOWED_DOMAINS, *arguments.allow_domain)
            if domain.strip()
        )
    )
    config = CrawlConfig(
        output=output,
        seeds=seeds,
        allowed_domains=allowed_domains,
        max_pages=arguments.max_pages,
        max_files=arguments.max_files,
        max_depth=arguments.max_depth,
        max_pages_per_host=arguments.max_pages_per_host,
        max_response_bytes=arguments.max_response_mb * 1024 * 1024,
        timeout=arguments.timeout,
        delay=arguments.delay,
        retries=arguments.retries,
        obey_robots=not arguments.ignore_robots,
        allow_private=arguments.allow_private,
        refresh=arguments.refresh,
        dry_run=arguments.dry_run,
    )
    crawler = PnuCrawler(config)
    try:
        summary = crawler.run()
    except KeyboardInterrupt:
        print("\n중단되었습니다. 같은 명령으로 실행하면 이어서 진행합니다.", file=sys.stderr)
        return 130
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        crawler.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
