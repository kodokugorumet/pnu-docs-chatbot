#!/usr/bin/env python3
"""Crawl public Pusan National University web pages and attachments.

The crawler starts from official PNU pages, follows links only across exact
hosts listed in a versioned scope file, and stores both HTML pages and document
attachments for the parser pipeline.

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
import fcntl
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
import threading
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
from typing import Callable, Iterable, Iterator, Sequence, TextIO


DEFAULT_SCOPE_PATH = Path(__file__).resolve().parents[1] / "config" / "pnu-crawl-scope.json"
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
ALLOWED_DOCUMENT_CONTENT_TYPES = {
    "application/haansofthwp",
    "application/hwp",
    "application/msword",
    "application/oxps",
    "application/pdf",
    "application/rtf",
    "application/vnd.hancom.hwp",
    "application/vnd.hancom.hwpx",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/x-hwp",
    "application/x-rtf",
    "text/csv",
    "text/plain",
}
DOCUMENT_CONTENT_TYPE_EXTENSIONS = {
    "application/haansofthwp": frozenset({".hwp"}),
    "application/hwp": frozenset({".hwp"}),
    "application/msword": frozenset({".doc"}),
    "application/pdf": frozenset({".pdf"}),
    "application/rtf": frozenset({".rtf"}),
    "application/vnd.hancom.hwp": frozenset({".hwp"}),
    "application/vnd.hancom.hwpx": frozenset({".hwpx"}),
    "application/vnd.ms-excel": frozenset({".xls"}),
    "application/vnd.ms-powerpoint": frozenset({".ppt"}),
    "application/vnd.oasis.opendocument.text": frozenset({".odt"}),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": frozenset(
        {".pptx"}
    ),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": frozenset(
        {".xlsx"}
    ),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": frozenset(
        {".docx"}
    ),
    "application/x-hwp": frozenset({".hwp"}),
    "application/x-rtf": frozenset({".rtf"}),
    "text/csv": frozenset({".csv"}),
    "text/plain": frozenset({".txt"}),
}
DYNAMIC_PAGE_EXTENSIONS = frozenset({".do", ".jsp", ".php", ".asp", ".aspx"})
HTML_PAGE_EXTENSIONS = frozenset({".htm", ".html", ".xhtml"})
GENERIC_BINARY_CONTENT_TYPES = {
    "application/binary",
    "application/download",
    "application/force-download",
    "application/octet-stream",
    "application/x-download",
    "application/x-msdownload",
}
BANNED_CONTENT_TYPE_PREFIXES = (
    "audio/",
    "image/",
    "video/",
)
BANNED_CONTENT_TYPES = {
    "application/gzip",
    "application/java-archive",
    "application/json",
    "application/vnd.android.package-archive",
    "application/x-7z-compressed",
    "application/x-apple-diskimage",
    "application/x-bzip",
    "application/x-bzip2",
    "application/x-dosexec",
    "application/x-executable",
    "application/x-gzip",
    "application/x-rar-compressed",
    "application/x-sharedlib",
    "application/x-tar",
    "application/x-zip-compressed",
    "application/zip",
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


class ResponseRejectedError(ValueError):
    """Raised before a response body is read when its headers are out of scope."""


class ResponseBudgetError(ValueError):
    """Raised before a response body is read when its kind has no budget left."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"{kind} budget reached")
        self.kind = kind


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


def provisional_kind_for_url(url: str) -> str:
    """Classify only URL shapes that unambiguously identify their response."""
    if is_document_url(url):
        return "attachment"
    extension = Path(
        urllib.parse.unquote(urllib.parse.urlsplit(url).path)
    ).suffix.lower()
    if extension in HTML_PAGE_EXTENSIONS:
        return "page"
    return "unknown"


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


def declared_filename(headers: Message, url: str) -> str:
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
    return safe_filename(filename, fallback="")


def filename_from_headers(
    headers: Message,
    url: str,
    content_type: str,
) -> str:
    filename = declared_filename(headers, url)
    normalized_type = content_type_only(content_type)
    mime_extensions = DOCUMENT_CONTENT_TYPE_EXTENSIONS.get(
        normalized_type,
        frozenset(),
    )
    inferred_extension = (
        sorted(mime_extensions)[0]
        if mime_extensions
        else mimetypes.guess_extension(normalized_type) or ""
    )
    existing_extension = Path(filename).suffix.lower()
    if not filename:
        filename = f"attachment{inferred_extension}"
    elif existing_extension in DYNAMIC_PAGE_EXTENSIONS:
        stem = Path(filename).stem or "attachment"
        filename = f"{stem}{inferred_extension}"
    elif not existing_extension and inferred_extension:
        filename = f"{filename}{inferred_extension}"
    return safe_filename(filename)


def classify_response_headers(
    url: str,
    headers: Message,
    content_type: str,
    allowed_extensions: Iterable[str] = DOCUMENT_EXTENSIONS,
) -> str:
    """Classify a response as page/attachment without consuming its body."""
    normalized_type = content_type_only(content_type)
    if (
        normalized_type in HTML_CONTENT_TYPES
        or normalized_type.startswith("text/html")
    ):
        return "page"

    allowed = {extension.lower() for extension in allowed_extensions}
    filename = declared_filename(headers, url)
    extension = Path(filename).suffix.lower()
    if extension in STATIC_EXTENSIONS:
        raise ResponseRejectedError(f"blocked attachment extension: {extension}")
    if (
        normalized_type in BANNED_CONTENT_TYPES
        or normalized_type.endswith("+json")
        or any(
            normalized_type.startswith(prefix)
            for prefix in BANNED_CONTENT_TYPE_PREFIXES
        )
    ):
        raise ResponseRejectedError(
            f"blocked response content type: {normalized_type or 'unknown'}"
        )
    if normalized_type in GENERIC_BINARY_CONTENT_TYPES:
        if extension in allowed:
            return "attachment"
        raise ResponseRejectedError(
            "generic binary response requires an allowed attachment filename"
        )
    if extension in DOCUMENT_EXTENSIONS:
        if extension in allowed:
            return "attachment"
        raise ResponseRejectedError(
            f"attachment extension is outside scope: {extension}"
        )
    if extension and extension not in DYNAMIC_PAGE_EXTENSIONS:
        raise ResponseRejectedError(
            f"unsupported attachment extension: {extension}"
        )
    mime_extensions = DOCUMENT_CONTENT_TYPE_EXTENSIONS.get(
        normalized_type,
        frozenset(),
    )
    if mime_extensions.intersection(allowed):
        return "attachment"
    if normalized_type in ALLOWED_DOCUMENT_CONTENT_TYPES:
        raise ResponseRejectedError(
            "response document type is outside the configured extension scope"
        )
    if not normalized_type and not is_document_url(url):
        return "page"
    raise ResponseRejectedError(
        f"unsupported response content type: {normalized_type or 'unknown'}"
    )


def response_is_document(url: str, headers: Message, content_type: str) -> bool:
    try:
        return classify_response_headers(url, headers, content_type) == "attachment"
    except ResponseRejectedError:
        return False


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
    def __init__(
        self,
        validator: Callable[[str], bool],
        before_redirect: Callable[[str, bool], None] | None = None,
    ) -> None:
        super().__init__()
        self.validator = validator
        self.before_redirect = before_redirect

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
        check_robots = bool(
            getattr(request, "_pnu_check_robots", True)
        )
        if self.before_redirect is not None:
            self.before_redirect(target, check_robots)
        redirected = super().redirect_request(
            request, file_pointer, code, message, headers, target
        )
        if redirected is None:
            raise urllib.error.HTTPError(
                request.full_url, code, "redirect rejected", headers, file_pointer
            )
        setattr(redirected, "_pnu_check_robots", check_robots)
        return redirected


@dataclass(frozen=True)
class ScopeTier:
    name: str
    hosts: tuple[str, ...]
    max_items_per_host: int


@dataclass(frozen=True)
class CrawlScope:
    version: int
    tiers: tuple[ScopeTier, ...]
    max_pages: int
    max_files: int
    max_bytes: int
    allowed_document_extensions: tuple[str, ...]
    sha256: str

    def tier_for_host(self, host: str) -> ScopeTier | None:
        normalized = host.rstrip(".").lower()
        return next(
            (tier for tier in self.tiers if normalized in tier.hosts),
            None,
        )

    @property
    def hosts(self) -> tuple[str, ...]:
        return tuple(host for tier in self.tiers for host in tier.hosts)


def normalized_scope_host(value: object) -> str:
    host = str(value).strip().lower().rstrip(".")
    if (
        not host
        or "://" in host
        or "/" in host
        or ":" in host
        or not re.fullmatch(r"[a-z0-9.-]+", host)
    ):
        raise ValueError(f"scope host must be an exact hostname: {value!r}")
    return host


def parse_crawl_scope(
    value: object,
    *,
    extra_hosts: Sequence[str] = (),
) -> CrawlScope:
    if not isinstance(value, dict):
        raise ValueError("scope root must be a JSON object")
    version = int(value.get("version", 0))
    if version != 1:
        raise ValueError(f"unsupported scope version: {version}")
    budgets = value.get("budgets")
    tiers_value = value.get("tiers")
    extensions_value = value.get("allowed_document_extensions")
    if not isinstance(budgets, dict):
        raise ValueError("scope budgets must be a JSON object")
    if not isinstance(tiers_value, dict) or not tiers_value:
        raise ValueError("scope tiers must be a non-empty JSON object")
    if not isinstance(extensions_value, list) or not extensions_value:
        raise ValueError("scope allowed_document_extensions must be a non-empty list")

    tiers: list[ScopeTier] = []
    seen_hosts: set[str] = set()
    for name, tier_value in sorted(tiers_value.items()):
        if not isinstance(tier_value, dict):
            raise ValueError(f"scope tier {name!r} must be a JSON object")
        hosts_value = tier_value.get("hosts")
        if not isinstance(hosts_value, list) or not hosts_value:
            raise ValueError(f"scope tier {name!r} must list exact hosts")
        host_limit = int(tier_value.get("max_items_per_host", 0))
        if host_limit <= 0:
            raise ValueError(
                f"scope tier {name!r} max_items_per_host must be positive"
            )
        hosts = tuple(
            sorted(dict.fromkeys(normalized_scope_host(host) for host in hosts_value))
        )
        overlap = seen_hosts.intersection(hosts)
        if overlap:
            raise ValueError(
                f"scope hosts may belong to only one tier: {sorted(overlap)!r}"
            )
        seen_hosts.update(hosts)
        tiers.append(
            ScopeTier(
                name=str(name),
                hosts=hosts,
                max_items_per_host=host_limit,
            )
        )

    normalized_extras = tuple(
        sorted(dict.fromkeys(normalized_scope_host(host) for host in extra_hosts))
    )
    extras = tuple(host for host in normalized_extras if host not in seen_hosts)
    if extras:
        core_limit = next(
            (
                tier.max_items_per_host
                for tier in tiers
                if tier.name == "core"
            ),
            min(tier.max_items_per_host for tier in tiers),
        )
        tiers.append(
            ScopeTier(
                name="extra",
                hosts=extras,
                max_items_per_host=core_limit,
            )
        )

    extensions = tuple(
        sorted(
            dict.fromkeys(
                (
                    str(extension).strip().lower()
                    if str(extension).strip().startswith(".")
                    else f".{str(extension).strip().lower()}"
                )
                for extension in extensions_value
            )
        )
    )
    unsupported_extensions = set(extensions).difference(DOCUMENT_EXTENSIONS)
    if unsupported_extensions:
        raise ValueError(
            "scope contains unsupported document extensions: "
            f"{sorted(unsupported_extensions)!r}"
        )
    max_pages = int(budgets.get("max_pages", 0))
    max_files = int(budgets.get("max_files", 0))
    max_bytes = int(budgets.get("max_bytes", 0))
    if min(max_pages, max_files, max_bytes) <= 0:
        raise ValueError("scope budgets must all be positive")

    canonical = {
        "allowed_document_extensions": list(extensions),
        "budgets": {
            "max_bytes": max_bytes,
            "max_files": max_files,
            "max_pages": max_pages,
        },
        "tiers": {
            tier.name: {
                "hosts": list(tier.hosts),
                "max_items_per_host": tier.max_items_per_host,
            }
            for tier in sorted(tiers, key=lambda item: item.name)
        },
        "version": version,
    }
    digest = sha256_bytes(
        json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return CrawlScope(
        version=version,
        tiers=tuple(sorted(tiers, key=lambda item: item.name)),
        max_pages=max_pages,
        max_files=max_files,
        max_bytes=max_bytes,
        allowed_document_extensions=extensions,
        sha256=digest,
    )


def load_crawl_scope(
    path: Path,
    *,
    extra_hosts: Sequence[str] = (),
) -> CrawlScope:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read scope file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid scope JSON in {path}: {error}") from error
    return parse_crawl_scope(value, extra_hosts=extra_hosts)


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
    scope: CrawlScope | None = None


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status: int
    headers: Message
    body: bytes
    content_type: str
    kind: str


class CrawlOutputLock:
    """Hold one crawler process per output directory.

    The lock file remains on disk intentionally. Removing it after unlock can
    create an inode race where two processes lock different files at the same
    path. The OS releases the advisory lock automatically if a process exits.
    """

    _registry_guard = threading.Lock()
    _held_paths: set[str] = set()

    def __init__(self, path: Path) -> None:
        self.path = path
        self._registry_key = str(path.resolve())
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._registry_guard:
            if self._registry_key in self._held_paths:
                raise RuntimeError(
                    f"another crawler is already using output state: {self.path.parent}"
                )
            self._held_paths.add(self._registry_key)

        handle: TextIO | None = None
        try:
            handle = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.seek(0)
                owner = handle.read().strip()
                detail = f" ({owner})" if owner else ""
                raise RuntimeError(
                    "another crawler is already using output state: "
                    f"{self.path.parent}{detail}"
                ) from error
            handle.seek(0)
            handle.truncate()
            json.dump(
                {"pid": os.getpid(), "acquired_at": utc_now()},
                handle,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            self._handle = handle
        except BaseException:
            if handle is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                handle.close()
            with self._registry_guard:
                self._held_paths.discard(self._registry_key)
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            with self._registry_guard:
                self._held_paths.discard(self._registry_key)


def preflight_existing_store_identity(
    path: Path,
    *,
    scope_hash: str,
    dry_run: bool,
) -> None:
    """Reject incompatible existing state before running mutable migrations."""
    if not path.exists():
        return
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        raise RuntimeError(
            f"cannot open existing crawl database read-only: {error}"
        ) from error
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        tables = {
            str(row["name"])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
        if not tables:
            return
        if "frontier" not in tables:
            raise RuntimeError(
                "existing crawl database is not a crawler state database; "
                "use a new --output directory"
            )
        frontier_count = int(
            connection.execute("SELECT COUNT(*) FROM frontier").fetchone()[0]
        )
        metadata: dict[str, str] = {}
        if "metadata" in tables:
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }

        stored_scope = metadata.get("scope_sha256")
        if stored_scope is None:
            if frontier_count:
                raise RuntimeError(
                    "existing crawl database has no scope identity; preserve it "
                    "and use a new --output directory for scoped crawling"
                )
        elif stored_scope != scope_hash:
            raise RuntimeError(
                "crawl scope does not match the scope stored in this database; "
                "use a new --output directory"
            )

        expected_storage = "metadata_only" if dry_run else "content"
        stored_storage = metadata.get("storage_mode")
        if stored_storage is None:
            if frontier_count:
                raise RuntimeError(
                    "existing crawl database has no storage mode identity; "
                    "preserve it and use a new --output directory"
                )
        elif stored_storage != expected_storage:
            raise RuntimeError(
                "crawl database storage mode does not match this run "
                f"({stored_storage} != {expected_storage}); "
                "use a new --output directory"
            )
    except sqlite3.Error as error:
        raise RuntimeError(
            f"cannot inspect existing crawl database read-only: {error}"
        ) from error
    finally:
        connection.close()


class CrawlStore:
    def __init__(
        self,
        path: Path,
        *,
        read_only: bool = False,
        recover_fetching: bool = False,
    ) -> None:
        if read_only:
            uri = f"{path.resolve().as_uri()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA query_only=ON")
            return

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
                canonical_url TEXT,
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
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        frontier_columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(frontier)")
        }
        if "canonical_url" not in frontier_columns:
            self.connection.execute(
                "ALTER TABLE frontier ADD COLUMN canonical_url TEXT"
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS frontier_canonical_status "
            "ON frontier(canonical_url, status)"
        )
        if recover_fetching:
            self.connection.execute(
                "UPDATE frontier SET status = 'queued' WHERE status = 'fetching'"
            )
        kind_version = self.connection.execute(
            """
            SELECT value
            FROM metadata
            WHERE key = 'provisional_kind_version'
            """
        ).fetchone()
        if kind_version is None or str(kind_version["value"]) != "1":
            provisional_updates = []
            for row in self.connection.execute(
                """
                SELECT url
                FROM frontier
                WHERE status IN ('queued', 'error') AND kind = 'page'
                """
            ):
                url = str(row["url"])
                provisional_kind = provisional_kind_for_url(url)
                if provisional_kind != "page":
                    provisional_updates.append((provisional_kind, url))
            if provisional_updates:
                self.connection.executemany(
                    """
                    UPDATE frontier
                    SET kind = ?
                    WHERE url = ?
                      AND status IN ('queued', 'error')
                    """,
                    provisional_updates,
                )
            self.connection.execute(
                """
                INSERT OR REPLACE INTO metadata(key, value)
                VALUES ('provisional_kind_version', '1')
                """
            )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def bind_scope(self, scope_hash: str) -> None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'scope_sha256'"
        ).fetchone()
        if row is not None:
            if str(row["value"]) != scope_hash:
                raise RuntimeError(
                    "crawl scope does not match the scope stored in this database; "
                    "use a new --output directory"
                )
            return
        frontier_count = int(
            self.connection.execute("SELECT COUNT(*) FROM frontier").fetchone()[0]
        )
        if frontier_count:
            raise RuntimeError(
                "existing crawl database has no scope identity; preserve it and "
                "use a new --output directory for scoped crawling"
            )
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('scope_sha256', ?)",
            (scope_hash,),
        )
        self.connection.commit()

    def bind_storage_mode(self, dry_run: bool) -> None:
        expected = "metadata_only" if dry_run else "content"
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'storage_mode'"
        ).fetchone()
        if row is not None:
            if str(row["value"]) != expected:
                raise RuntimeError(
                    "crawl database storage mode does not match this run "
                    f"({row['value']} != {expected}); use a new --output directory"
                )
            return
        frontier_count = int(
            self.connection.execute("SELECT COUNT(*) FROM frontier").fetchone()[0]
        )
        if frontier_count:
            raise RuntimeError(
                "existing crawl database has no storage mode identity; preserve "
                "it and use a new --output directory"
            )
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('storage_mode', ?)",
            (expected,),
        )
        self.connection.commit()

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

    def claim_next(
        self,
        kinds: str | Iterable[str] | None = None,
    ) -> sqlite3.Row | None:
        if isinstance(kinds, str):
            normalized_kinds = (kinds,)
        elif kinds is None:
            normalized_kinds = ()
        else:
            normalized_kinds = tuple(dict.fromkeys(kinds))
        kind_filter = ""
        parameters: tuple[object, ...] = ()
        if normalized_kinds:
            placeholders = ",".join("?" for _ in normalized_kinds)
            kind_filter = f"AND kind IN ({placeholders})"
            parameters = tuple(normalized_kinds)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
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
                self.connection.commit()
                return None
            cursor = self.connection.execute(
                """
                UPDATE frontier
                SET status = 'fetching', attempts = attempts + 1
                WHERE url = ? AND status = 'queued'
                """,
                (row["url"],),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"failed to claim queued crawl item atomically: {row['url']}"
                )
            self.connection.commit()
            return row
        except BaseException:
            self.connection.rollback()
            raise

    def finish(self, url: str, **values: object) -> None:
        assignments = ["status = 'done'", "fetched_at = ?"]
        parameters: list[object] = [utc_now()]
        for key in (
            "http_status",
            "content_type",
            "final_url",
            "canonical_url",
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

    def done_page_for_canonical(
        self,
        canonical_url: str,
        *,
        exclude_url: str,
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT *
            FROM frontier
            WHERE status = 'done'
              AND kind = 'page'
              AND url <> ?
              AND coalesce(canonical_url, final_url, url) = ?
            ORDER BY fetched_at, url
            LIMIT 1
            """,
            (exclude_url, canonical_url),
        ).fetchone()

    def finish_canonical_duplicate(
        self,
        url: str,
        *,
        canonical_url: str,
        duplicate_of: str,
        result: FetchResult,
        title: str,
        digest: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE frontier
            SET status = 'skipped',
                fetched_at = ?,
                http_status = ?,
                content_type = ?,
                final_url = ?,
                canonical_url = ?,
                title = ?,
                storage_path = NULL,
                sha256 = ?,
                size_bytes = ?,
                kind = 'page',
                error = ?
            WHERE url = ?
            """,
            (
                utc_now(),
                result.status,
                result.content_type,
                result.final_url,
                canonical_url,
                title,
                digest,
                len(result.body),
                "duplicate_canonical:{}".format(duplicate_of)[:1000],
                url,
            ),
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

    def requeue(self, url: str, kind: str, error: str) -> None:
        self.connection.execute(
            """
            UPDATE frontier
            SET status = 'queued', kind = ?, error = ?
            WHERE url = ?
            """,
            (kind, error[:1000], url),
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
        self.output_lock = CrawlOutputLock(self.state_dir / "crawl.lock")
        self.output_lock.acquire()
        self._closed = False
        database_path = self.state_dir / "crawl.sqlite3"
        try:
            if config.scope is not None:
                preflight_existing_store_identity(
                    database_path,
                    scope_hash=config.scope.sha256,
                    dry_run=config.dry_run,
                )
            self.store = CrawlStore(
                database_path,
                recover_fetching=True,
            )
            if config.scope is not None:
                self.store.bind_scope(config.scope.sha256)
                self.store.bind_storage_mode(config.dry_run)
            if config.refresh:
                self.store.reset_for_refresh()
        except BaseException:
            if hasattr(self, "store"):
                self.store.close()
            self.output_lock.release()
            raise
        self.last_request_at: dict[str, float] = {}
        self.public_host_cache: dict[str, bool] = {}
        self.robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.opener = urllib.request.build_opener(
            GuardedRedirectHandler(
                self.url_is_fetchable,
                self.prepare_redirect,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.store.close()
        finally:
            self.output_lock.release()

    def url_is_fetchable(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.scheme not in {"http", "https"}:
            return False
        if self.config.scope is not None:
            if self.config.scope.tier_for_host(host) is None:
                return False
        elif not host_matches(host, self.config.allowed_domains):
            return False
        skipped, _ = should_skip_url(url)
        if skipped:
            return False
        if self.config.allow_private:
            return True
        if host not in self.public_host_cache:
            self.public_host_cache[host] = is_public_host(host)
        return self.public_host_cache[host]

    def host_item_limit(self, host: str) -> int:
        limit = self.config.max_pages_per_host
        if self.config.scope is not None:
            tier = self.config.scope.tier_for_host(host)
            if tier is None:
                return 0
            limit = min(limit, tier.max_items_per_host)
        return limit

    def crawl_limits(self) -> tuple[int, int, int | None]:
        page_limit = self.config.max_pages
        file_limit = self.config.max_files
        byte_limit: int | None = None
        if self.config.scope is not None:
            page_limit = min(page_limit, self.config.scope.max_pages)
            file_limit = min(file_limit, self.config.scope.max_files)
            byte_limit = self.config.scope.max_bytes
        return page_limit, file_limit, byte_limit

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
                    kind=provisional_kind_for_url(url),
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
            result = self.fetch(
                robots_url,
                check_robots=False,
                max_bytes=1024 * 1024,
                classify_response=False,
            )
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

    def prepare_redirect(self, url: str, check_robots: bool) -> None:
        if check_robots and not self.robots_allows(url):
            raise PermissionError("redirect target blocked by robots.txt")
        host = urllib.parse.urlsplit(url).hostname or ""
        self.wait_for_host(host)

    def fetch(
        self,
        url: str,
        *,
        check_robots: bool = True,
        max_bytes: int | None = None,
        classify_response: bool = True,
        allowed_kinds: set[str] | None = None,
    ) -> FetchResult:
        if not self.url_is_fetchable(url):
            raise ValueError("URL is outside allowed public scope")
        if check_robots and not self.robots_allows(url):
            raise PermissionError("blocked by robots.txt")
        limit = (
            self.config.max_response_bytes
            if max_bytes is None
            else min(max_bytes, self.config.max_response_bytes)
        )
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
            setattr(request, "_pnu_check_robots", check_robots)
            try:
                with self.opener.open(request, timeout=self.config.timeout) as response:
                    final_url = canonicalize_url(response.geturl()) or url
                    if not self.url_is_fetchable(final_url):
                        raise ValueError("final URL is outside allowed public scope")
                    response_content_type = content_type_only(
                        response.headers.get("Content-Type")
                    )
                    if classify_response:
                        allowed_extensions = (
                            self.config.scope.allowed_document_extensions
                            if self.config.scope is not None
                            else tuple(DOCUMENT_EXTENSIONS)
                        )
                        kind = classify_response_headers(
                            final_url,
                            response.headers,
                            response_content_type,
                            allowed_extensions,
                        )
                        if allowed_kinds is not None and kind not in allowed_kinds:
                            raise ResponseBudgetError(kind)
                    else:
                        kind = "raw"
                    declared_length = response.headers.get("Content-Length")
                    if declared_length and int(declared_length) > limit:
                        raise ResponseRejectedError(
                            f"response too large: {declared_length} > {limit} bytes"
                        )
                    body = response.read(limit + 1)
                    if len(body) > limit:
                        raise ResponseRejectedError(
                            f"response exceeded {limit} bytes"
                        )
                    return FetchResult(
                        requested_url=url,
                        final_url=final_url,
                        status=int(response.status),
                        headers=response.headers,
                        body=body,
                        content_type=response_content_type,
                        kind=kind,
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
        declared_canonical = canonicalize_url(
            parser.canonical_href or "",
            result.final_url,
        )
        canonical_url = (
            declared_canonical
            if declared_canonical
            and self.url_is_fetchable(declared_canonical)
            else canonicalize_url(result.final_url) or result.final_url
        )
        digest = sha256_bytes(result.body)
        duplicate = self.store.done_page_for_canonical(
            canonical_url,
            exclude_url=str(row["url"]),
        )
        if duplicate is not None:
            self.store.finish_canonical_duplicate(
                str(row["url"]),
                canonical_url=canonical_url,
                duplicate_of=str(duplicate["url"]),
                result=result,
                title=parser.title[:500],
                digest=digest,
            )
            self.store.log(
                str(row["url"]),
                "canonical_duplicate",
                "{} -> {}".format(canonical_url, duplicate["url"]),
            )
            return

        storage_path = self.html_storage_path(canonical_url)
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
            canonical_url=canonical_url,
            title=parser.title[:500],
            storage_path=relative_path,
            sha256=digest,
            size_bytes=len(result.body),
            kind="page",
        )
        if int(row["depth"]) >= self.config.max_depth:
            return
        link_base = canonicalize_url(parser.base_href or "", canonical_url)
        base_url = link_base or canonical_url
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
            kind = provisional_kind_for_url(url)
            self.store.enqueue(
                url=url,
                depth=int(row["depth"]) + 1,
                priority=link_priority(url, anchor_text),
                parent_url=str(row["url"]),
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
        database_counts = self.store.counts()
        if database_counts["total"] == 0:
            raise RuntimeError(
                "수집 가능한 seed URL이 없습니다. 네트워크/DNS와 --seed, "
                "--allow-domain 설정을 확인하세요."
            )
        page_limit, file_limit, byte_limit = self.crawl_limits()
        done_pages = database_counts["pages"]
        done_files = database_counts["files"]
        done_bytes = database_counts["bytes"]
        pages_this_run = 0
        files_this_run = 0
        errors_this_run = 0
        skipped_this_run = 0
        while True:
            page_capacity = done_pages < page_limit
            file_capacity = done_files < file_limit
            if not page_capacity and not file_capacity:
                break
            if byte_limit is not None and done_bytes >= byte_limit:
                break
            allowed_kinds: set[str] = set()
            if page_capacity:
                allowed_kinds.add("page")
            if file_capacity:
                allowed_kinds.add("attachment")
            if not page_capacity:
                next_kinds: tuple[str, ...] | None = (
                    "attachment",
                    "unknown",
                )
            elif not file_capacity:
                next_kinds = ("page", "unknown")
            else:
                next_kinds = None
            row = self.store.claim_next(next_kinds)
            if row is None:
                break
            url = str(row["url"])
            host = str(row["host"])
            if int(row["depth"]) > self.config.max_depth:
                self.store.mark(url, "skipped", "max_depth")
                skipped_this_run += 1
                continue
            if self.store.host_done_count(host) >= self.host_item_limit(host):
                self.store.mark(url, "skipped", "max_items_per_host")
                skipped_this_run += 1
                continue
            try:
                response_limit = self.config.max_response_bytes
                if byte_limit is not None:
                    response_limit = min(response_limit, byte_limit - done_bytes)
                result = self.fetch(
                    url,
                    max_bytes=response_limit,
                    allowed_kinds=allowed_kinds,
                )
                if result.kind == "attachment":
                    self.process_attachment(row, result)
                    files_this_run += 1
                    done_files += 1
                    print(f"[file] {result.final_url}", flush=True)
                elif result.kind == "page":
                    self.process_html(row, result)
                    pages_this_run += 1
                    done_pages += 1
                    print(
                        f"[page {done_pages}/{page_limit}] "
                        f"{result.final_url}",
                        flush=True,
                    )
                else:
                    raise RuntimeError(f"unexpected response kind: {result.kind}")
                done_bytes += len(result.body)
            except ResponseBudgetError as error:
                self.store.requeue(url, error.kind, str(error))
            except ResponseRejectedError as error:
                self.store.mark(url, "skipped", str(error))
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
            "Crawl public PNU pages and attachments across exact hosts in a "
            "versioned scope."
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
        help="additional exact host to add to the scope; repeatable",
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=DEFAULT_SCOPE_PATH,
        help=(
            "exact-host crawl scope JSON "
            "(default: config/pnu-crawl-scope.json)"
        ),
    )
    parser.add_argument("--max-pages", type=positive_int, default=4000)
    parser.add_argument("--max-files", type=positive_int, default=2000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-pages-per-host", type=positive_int, default=500)
    parser.add_argument(
        "--max-response-mb",
        type=positive_int,
        default=25,
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
        help=(
            "crawl and record metadata without saving bodies; its output "
            "cannot later be resumed as a content-saving crawl"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="show the existing crawl database summary and exit",
    )
    return parser


def resolve_seeds(
    arguments: argparse.Namespace,
    scope_hosts: Sequence[str] = (),
) -> tuple[str, ...]:
    if arguments.replace_default_seeds:
        if not arguments.seed:
            raise ValueError("--replace-default-seeds requires at least one --seed")
        return tuple(arguments.seed)
    return tuple(
        dict.fromkeys(
            (
                *DEFAULT_SEEDS,
                *arguments.seed,
                *(f"https://{host}/" for host in scope_hosts),
            )
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    database_path = output / "state" / "crawl.sqlite3"
    if arguments.status:
        if not database_path.exists():
            print(json.dumps({"output": str(output), "status": "not_started"}, ensure_ascii=False))
            return 0
        store = CrawlStore(database_path, read_only=True)
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
        scope = load_crawl_scope(
            arguments.scope.expanduser().resolve(),
            extra_hosts=arguments.allow_domain,
        )
        seeds = resolve_seeds(arguments, scope.hosts)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    allowed_domains = scope.hosts
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
        scope=scope,
    )
    try:
        crawler = PnuCrawler(config)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
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
