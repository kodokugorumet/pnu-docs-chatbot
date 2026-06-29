#!/usr/bin/env python3
"""Download public document attachments from Korean financial institution sites.

The crawler is intentionally conservative:
- standard-library only
- seed URL based instead of whole-site crawling
- robots.txt aware by default
- rate limited
- resumable through URL de-duplication and existing-file checks

Examples:
  python scripts/crawl_finance_docs.py --target bok --dry-run --max-pages 20
  python scripts/crawl_finance_docs.py --target all --output src/data --max-files 300
"""

from __future__ import annotations

import argparse
import csv
import email.message
import hashlib
import html
import json
import os
import posixpath
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
DEFAULT_FILE_TYPES = (".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".xls", ".xlsx")
BLOCKED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mp3",
    ".mp4",
    ".png",
    ".svg",
    ".webp",
    ".woff",
    ".woff2",
    ".xml",
    ".zip",
}
DOWNLOAD_HINTS = (
    "download",
    "down",
    "filedown",
    "filedownload",
    "atch",
    "attach",
    "commonfile",
    "cmmn/file",
    "downloadfile",
)
KOREAN_FILE_HINTS = ("첨부", "다운로드", "파일", "서식", "전문", "뷰어")
DEFAULT_EXCLUDE_KEYWORDS = (
    "facebook",
    "instagram",
    "javascript:",
    "login",
    "logout",
    "mailto:",
    "sitemap",
    "twitter",
    "youtube",
)


@dataclass(frozen=True)
class TargetConfig:
    key: str
    name: str
    folder: str
    seeds: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    link_keywords: tuple[str, ...]
    page_url_keywords: tuple[str, ...]
    exclude_keywords: tuple[str, ...] = DEFAULT_EXCLUDE_KEYWORDS


TARGETS: dict[str, TargetConfig] = {
    "fss": TargetConfig(
        key="fss",
        name="금융감독원",
        folder="금융감독원",
        seeds=(
            "https://dart.fss.or.kr/info/main.do",
            "https://dart.fss.or.kr/dsaa003/selectGuideMain.ax?seqno=428",
            "https://www.fss.or.kr/fss/main/main.do",
        ),
        allowed_domains=("fss.or.kr", "dart.fss.or.kr"),
        link_keywords=(
            "공시",
            "규정",
            "기준",
            "서식",
            "작성기준",
            "감독",
            "업무",
            "보도자료",
            "자료",
            "첨부",
            "download",
        ),
        page_url_keywords=(
            "info",
            "guide",
            "dsaa",
            "bbs",
            "board",
            "notice",
            "main.do",
        ),
    ),
    "bok": TargetConfig(
        key="bok",
        name="한국은행",
        folder="한국은행",
        seeds=(
            "https://www.bok.or.kr/portal/singl/law/listSearch.do?menuNo=200200",
            "https://www.bok.or.kr/portal/bbs/P0002014/list.do?menuNo=200402",
            "https://www.bok.or.kr/portal/main/contents.do?menuNo=200026",
            "https://www.bok.or.kr/portal/main/contents.do?menuNo=200238",
        ),
        allowed_domains=("bok.or.kr",),
        link_keywords=(
            "법규",
            "규정",
            "세칙",
            "지침",
            "절차",
            "서식",
            "외환",
            "지급결제",
            "통화",
            "금융",
            "자료",
            "첨부",
            "download",
        ),
        page_url_keywords=("bbs", "contents", "law", "menuNo", "view.do", "list.do", "listSearch.do"),
    ),
    "krx": TargetConfig(
        key="krx",
        name="한국거래소",
        folder="한국거래소",
        seeds=(
            "https://rule.krx.co.kr/",
            "https://law.krx.co.kr/las",
            "https://kind.krx.co.kr/disclosureinfo/notice.do?method=searchNoticeMain",
            "https://kind.krx.co.kr/disclosureinfo/searchmaterials.do?method=searchMaterialsMain",
            "https://esg.krx.co.kr/contents/04/04010000/ESG04010000.jsp",
        ),
        allowed_domains=("krx.co.kr", "kind.krx.co.kr", "law.krx.co.kr", "rule.krx.co.kr"),
        link_keywords=(
            "규정",
            "시행세칙",
            "상장",
            "공시",
            "업무규정",
            "시장",
            "개정",
            "예고",
            "자료",
            "서식",
            "첨부",
            "download",
        ),
        page_url_keywords=(
            "las",
            "rule",
            "notice",
            "searchmaterials",
            "disclosureinfo",
            "contents",
            "board",
        ),
    ),
    "ksd": TargetConfig(
        key="ksd",
        name="한국예탁결제원",
        folder="한국예탁결제원",
        seeds=(
            "https://www.ksd.or.kr/",
            "https://www.ksd.or.kr/pension/service-information/serviceInfoApplicationGuide.home",
            "https://seibro.or.kr/websquare/control.jsp?menuNo=561&w2xPath=%2FIPORTAL%2Fuser%2Fetc%2FBIP_CMUC01034V.xml",
            "https://ta.ksd.or.kr/",
        ),
        allowed_domains=("ksd.or.kr", "seibro.or.kr", "ta.ksd.or.kr"),
        link_keywords=(
            "규정",
            "업무",
            "예탁",
            "결제",
            "전자등록",
            "증권대행",
            "서식",
            "자료",
            "공시",
            "첨부",
            "download",
        ),
        page_url_keywords=("websquare", "control.jsp", "board", "bbs", "notice", "data", "contents"),
    ),
}


@dataclass
class Link:
    url: str
    text: str = ""
    method: str = "GET"
    data: dict[str, str] | None = None


@dataclass
class FetchResult:
    url: str
    status: int
    headers: email.message.Message
    body: bytes


@dataclass
class ExistingIndex:
    urls: set[str]
    names: set[str]
    hashes: set[str]


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[Link] = []
        self._anchor_stack: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name.lower(): value or "" for name, value in attrs}
        if tag.lower() == "a" and attr_map.get("href"):
            self._anchor_stack.append({"url": attr_map["href"], "text": ""})

        for name in ("href", "src", "data-url", "data-href", "data-link", "action"):
            value = attr_map.get(name)
            if value:
                self.links.append(Link(value, attr_map.get("title", "") or attr_map.get("alt", "")))

        onclick = attr_map.get("onclick", "")
        for value in re.findall(r"""['"]([^'"]+(?:download|file|view|list|bbs|board)[^'"]*)['"]""", onclick, re.I):
            self.links.append(Link(value, attr_map.get("title", "")))

    def handle_data(self, data: str) -> None:
        if self._anchor_stack:
            self._anchor_stack[-1]["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._anchor_stack:
            anchor = self._anchor_stack.pop()
            self.links.append(Link(anchor["url"], clean_text(anchor["text"])))


class Crawler:
    def __init__(
        self,
        *,
        output_root: Path,
        file_types: tuple[str, ...],
        max_depth: int,
        max_pages: int,
        max_files: int,
        max_mb: int,
        delay: float,
        timeout: float,
        retries: int,
        dry_run: bool,
        ignore_robots: bool,
        extra_keywords: tuple[str, ...],
    ) -> None:
        self.output_root = output_root
        self.file_types = normalize_file_types(file_types)
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_files = max_files
        self.max_bytes = max_mb * 1024 * 1024
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.dry_run = dry_run
        self.ignore_robots = ignore_robots
        self.extra_keywords = tuple(keyword.lower() for keyword in extra_keywords)
        self.seen_pages: set[str] = set()
        self.seen_files: set[str] = set()
        self.robot_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())

    def crawl_target(self, target: TargetConfig) -> dict[str, int]:
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in target.seeds)
        pages = 0
        files = 0
        skipped = 0
        output_dir = self.output_root / target.folder

        if not self.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        existing = load_existing_index(output_dir)

        print(f"\n[{target.name}] start")
        while queue and pages < self.max_pages and files < self.max_files:
            page_url, depth = queue.popleft()
            page_url = normalize_url("", page_url)
            if not page_url or page_url in self.seen_pages:
                continue
            if not allowed_domain(page_url, target.allowed_domains):
                continue
            if should_exclude(page_url, target.exclude_keywords):
                continue
            if not self.can_fetch(page_url):
                skipped += 1
                print(f"  robots skip: {page_url}")
                continue

            result = self.fetch(page_url)
            if result is None:
                skipped += 1
                continue

            content_type = result.headers.get("content-type", "")
            if is_document_response(result, self.file_types):
                if files < self.max_files and self.save_document(target, result, output_dir, page_url, "", existing):
                    files += 1
                continue
            if "html" not in content_type.lower() and not looks_like_html(result.body):
                skipped += 1
                continue

            pages += 1
            self.seen_pages.add(page_url)
            links = extract_links(page_url, result.body)
            links.extend(self.dynamic_links_for_page(target, page_url))

            for link in links:
                url = normalize_url(page_url, link.url)
                file_key = request_identity(url, link.data)
                if not url or file_key in self.seen_files:
                    continue
                if not allowed_domain(url, target.allowed_domains):
                    continue
                if should_exclude(url, target.exclude_keywords):
                    continue

                if looks_like_document_link(url, link.text, self.file_types):
                    if is_probably_existing(url, link, existing, self.file_types):
                        self.seen_files.add(file_key)
                        continue
                    if files >= self.max_files:
                        break
                    if not self.can_fetch(url):
                        skipped += 1
                        print(f"  robots skip file: {url}")
                        continue
                    if self.dry_run:
                        data_label = f" data={link.data}" if link.data else ""
                        print(f"  would download: {target.name} | {link.text or '(no text)'} | {url}{data_label}")
                        self.seen_files.add(file_key)
                        files += 1
                    else:
                        fetched = self.fetch(url, referer=page_url, data=link.data)
                        if fetched and self.save_document(
                            target,
                            fetched,
                            output_dir,
                            page_url,
                            link.text,
                            existing,
                        ):
                            self.seen_files.add(file_key)
                            files += 1
                        else:
                            skipped += 1
                    continue

                if depth < self.max_depth and self.should_follow_page(target, url, link.text):
                    queue.append((url, depth + 1))

        print(f"[{target.name}] pages={pages} files={files} skipped={skipped}")
        return {"pages": pages, "files": files, "skipped": skipped}

    def dynamic_links_for_page(self, target: TargetConfig, page_url: str) -> list[Link]:
        if target.key != "krx" or "kind.krx.co.kr/disclosureinfo/" not in page_url:
            return []

        dynamic_requests: list[tuple[str, dict[str, str]]] = []
        if "/searchmaterials.do" in page_url:
            dynamic_requests.append(
                (
                    "https://kind.krx.co.kr/disclosureinfo/searchmaterials.do",
                    {
                        "method": "searchMaterialsSub",
                        "forward": "searchmaterials_sub",
                        "pageIndex": "1",
                        "currentPageSize": "50",
                    },
                )
            )
        if "/notice.do" in page_url:
            dynamic_requests.append(
                (
                    "https://kind.krx.co.kr/disclosureinfo/notice.do",
                    {
                        "method": "searchNoticeSub",
                        "forward": "searchnotice_sub",
                        "pageIndex": "1",
                        "currentPageSize": "50",
                    },
                )
            )

        links: list[Link] = []
        for url, data in dynamic_requests:
            result = self.fetch(url, referer=page_url, data=data)
            if result:
                links.extend(extract_links(url, result.body))
        return links

    def should_follow_page(self, target: TargetConfig, url: str, text: str) -> bool:
        path = urllib.parse.urlsplit(url).path.lower()
        extension = Path(path).suffix.lower()
        if extension in BLOCKED_EXTENSIONS or extension in self.file_types:
            return False
        haystack = f"{url} {text}".lower()
        keywords = tuple(k.lower() for k in target.link_keywords + target.page_url_keywords) + self.extra_keywords
        return any(keyword.lower() in haystack for keyword in keywords)

    def can_fetch(self, url: str) -> bool:
        if self.ignore_robots:
            return True
        parsed = urllib.parse.urlsplit(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if robots_url not in self.robot_cache:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
            except Exception:
                self.robot_cache[robots_url] = None
            else:
                self.robot_cache[robots_url] = parser

        parser = self.robot_cache.get(robots_url)
        return True if parser is None else parser.can_fetch(USER_AGENT, url)

    def fetch(
        self,
        url: str,
        referer: str | None = None,
        data: dict[str, str] | None = None,
    ) -> FetchResult | None:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,application/octet-stream,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
            "Connection": "close",
        }
        if referer:
            headers["Referer"] = referer
        encoded_data = None
        if data:
            encoded_data = urllib.parse.urlencode(data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"

        for attempt in range(1, self.retries + 2):
            if self.delay > 0:
                time.sleep(self.delay)
            request = urllib.request.Request(url, data=encoded_data, headers=headers)
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > self.max_bytes:
                        print(f"  skip large file: {url} ({content_length} bytes)")
                        return None
                    body = response.read(self.max_bytes + 1)
                    if len(body) > self.max_bytes:
                        print(f"  skip large response: {url}")
                        return None
                    return FetchResult(
                        url=response.geturl(),
                        status=getattr(response, "status", 200),
                        headers=response.headers,
                        body=body,
                    )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt > self.retries:
                    print(f"  fetch failed: {url} ({exc})")
                    return None
                time.sleep(min(2.0 * attempt, 5.0))
        return None

    def save_document(
        self,
        target: TargetConfig,
        result: FetchResult,
        output_dir: Path,
        source_page: str,
        link_text: str,
        existing: ExistingIndex,
    ) -> bool:
        if not is_document_response(result, self.file_types):
            return False

        filename = choose_filename(result, link_text, self.file_types)
        digest = hashlib.sha256(result.body).hexdigest()
        if result.url in existing.urls or filename in existing.names or digest in existing.hashes:
            print(f"  skip existing: {filename}")
            existing.urls.add(result.url)
            existing.names.add(filename)
            existing.hashes.add(digest)
            return False

        destination = output_dir / filename
        if destination.exists():
            print(f"  skip existing file: {destination.name}")
            existing.urls.add(result.url)
            existing.names.add(filename)
            existing.hashes.add(digest)
            return False

        destination.write_bytes(result.body)
        write_manifest(
            output_dir / "crawl_manifest.jsonl",
            {
                "institution": target.name,
                "filename": destination.name,
                "path": str(destination),
                "source_page": source_page,
                "download_url": result.url,
                "link_text": link_text,
                "content_type": result.headers.get("content-type", ""),
                "bytes": len(result.body),
                "sha256": digest,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        existing.urls.add(result.url)
        existing.names.add(destination.name)
        existing.hashes.add(digest)
        print(f"  saved: {destination.name}")
        return True


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def normalize_file_types(values: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        value = value.strip().lower()
        if not value:
            continue
        normalized.append(value if value.startswith(".") else f".{value}")
    return tuple(dict.fromkeys(normalized))


def normalize_url(base_url: str, raw_url: str) -> str:
    raw_url = html.unescape((raw_url or "").strip())
    if not raw_url:
        return ""
    lowered = raw_url.lower()
    if lowered.startswith(("javascript:", "mailto:", "tel:", "sms:", "#")):
        return ""
    joined = urllib.parse.urljoin(base_url, raw_url)
    parsed = urllib.parse.urlsplit(joined)
    if parsed.scheme not in {"http", "https"}:
        return ""
    path = urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%")
    query = urllib.parse.quote(urllib.parse.unquote(parsed.query), safe="=&;%/:,+")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, ""))


def allowed_domain(url: str, allowed_domains: tuple[str, ...]) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    host = host.lower().removeprefix("www.")
    for domain in allowed_domains:
        domain = domain.lower().removeprefix("www.")
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False


def should_exclude(url: str, keywords: tuple[str, ...]) -> bool:
    lowered = url.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def extract_links(base_url: str, body: bytes) -> list[Link]:
    text = decode_html(body)
    parser = LinkExtractor()
    try:
        parser.feed(text)
    except Exception:
        pass

    for match in re.finditer(r"""https?://[^\s"'<>]+""", text):
        parser.links.append(Link(match.group(0), ""))

    for filename in re.findall(r"""staticFileDown\(\s*['"]([^'"]+)['"]\s*\)""", text):
        parser.links.append(
            Link(
                "/pension/download/downloadStaticFile.home",
                urllib.parse.unquote(filename),
                "POST",
                {"fileNm": urllib.parse.unquote(filename)},
            )
        )

    for match in re.finditer(
        r"""<a\b[^>]*onclick=["'][^"']*getFSSFileUrl\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)[^"']*["'][^>]*>(.*?)</a>""",
        text,
        re.I | re.S,
    ):
        file_id = match.group(1).strip()
        file_sn = match.group(2).strip()
        link_text = clean_text(re.sub(r"<[^>]+>", " ", match.group(3)))
        parser.links.append(
            Link(
                "https://www.fss.or.kr/fss/cmmn/file/fileDown.do?"
                + urllib.parse.urlencode({"atchFileId": file_id, "fileSn": file_sn}),
                link_text or f"FSS attachment {file_sn}",
            )
        )

    seen: set[str] = set()
    links: list[Link] = []
    for link in parser.links:
        special_link = special_javascript_link(base_url, link)
        if special_link:
            link = special_link
        url = normalize_url(base_url, link.url)
        identity = request_identity(url, link.data)
        if url and identity not in seen:
            seen.add(identity)
            links.append(Link(url, clean_text(link.text), link.method, link.data))
    return links


def special_javascript_link(base_url: str, link: Link) -> Link | None:
    match = re.search(r"""staticFileDown\(\s*['"]([^'"]+)['"]\s*\)""", link.url, re.I)
    if not match:
        return None
    filename = urllib.parse.unquote(match.group(1))
    return Link(
        normalize_url(base_url, "/pension/download/downloadStaticFile.home"),
        link.text or filename,
        "POST",
        {"fileNm": filename},
    )


def request_identity(url: str, data: dict[str, str] | None = None) -> str:
    if not data:
        return url
    return f"{url} POST {urllib.parse.urlencode(sorted(data.items()))}"


def decode_html(body: bytes) -> str:
    sample = body[:4096].decode("ascii", errors="ignore")
    charset_match = re.search(r"charset=['\"]?([\w.-]+)", sample, re.I)
    encodings = []
    if charset_match:
        encodings.append(charset_match.group(1))
    encodings.extend(["utf-8", "cp949", "euc-kr"])
    for encoding in encodings:
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def looks_like_html(body: bytes) -> bool:
    sample = body[:512].lstrip().lower()
    return sample.startswith((b"<!doctype html", b"<html", b"<script")) or b"<html" in sample[:256]


def extension_from_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    path = urllib.parse.unquote(parsed.path)
    path_extension = posixpath.splitext(path)[1].lower()

    for value in query_values(parsed.query):
        query_extension = posixpath.splitext(urllib.parse.unquote(value))[1].lower()
        if query_extension in DEFAULT_FILE_TYPES:
            return query_extension
        if query_extension and not path_extension:
            return query_extension
    return path_extension


def query_values(query: str) -> list[str]:
    values: list[str] = []
    for _, raw_values in urllib.parse.parse_qs(query, keep_blank_values=True).items():
        values.extend(raw_values)
    return values


def filename_from_url_query(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    for key in ("fileNm", "fileName", "filename", "orgFileNm", "orignlFileNm", "atchFileNm"):
        for actual_key, values in params.items():
            if actual_key.lower() == key.lower() and values:
                return urllib.parse.unquote(values[0])
    for value in query_values(parsed.query):
        if posixpath.splitext(urllib.parse.unquote(value))[1]:
            return urllib.parse.unquote(value)
    return ""


def extension_from_text(text: str, file_types: tuple[str, ...]) -> str:
    lowered = text.lower()
    for ext in sorted(file_types, key=len, reverse=True):
        if ext in lowered:
            return ext
    return ""


def looks_like_document_link(url: str, link_text: str, file_types: tuple[str, ...]) -> bool:
    url_lower = urllib.parse.unquote(url).lower()
    text_lower = (link_text or "").lower()
    extension = extension_from_url(url)
    if extension in BLOCKED_EXTENSIONS:
        return False
    if extension in file_types or extension_from_text(text_lower, file_types):
        return True
    has_download_hint = any(hint in url_lower for hint in DOWNLOAD_HINTS)
    has_file_hint = any(hint in text_lower for hint in KOREAN_FILE_HINTS)
    return has_download_hint and has_file_hint


def is_document_response(result: FetchResult, file_types: tuple[str, ...]) -> bool:
    content_type = result.headers.get("content-type", "").lower()
    disposition = result.headers.get("content-disposition", "")
    filename = filename_from_content_disposition(disposition)
    if extension_from_url(result.url) in file_types:
        return True
    if extension_from_text(filename, file_types):
        return True
    if "text/html" in content_type or looks_like_html(result.body):
        return False
    if any(token in content_type for token in ("pdf", "hwp", "msword", "excel", "spreadsheet", "octet-stream")):
        return True
    return bool(disposition and "attachment" in disposition.lower())


def filename_from_content_disposition(value: str) -> str:
    if not value:
        return ""
    match = re.search(r"filename\*=([^;]+)", value, re.I)
    if match:
        encoded = match.group(1).strip().strip('"')
        parts = encoded.split("''", 1)
        if len(parts) == 2:
            try:
                return urllib.parse.unquote(parts[1], encoding=parts[0] or "utf-8")
            except LookupError:
                return urllib.parse.unquote(parts[1])
        return urllib.parse.unquote(encoded)

    match = re.search(r'filename="?([^";]+)"?', value, re.I)
    if match:
        filename = match.group(1).strip()
        if "%" in filename:
            return urllib.parse.unquote(filename)
        for encoding in ("utf-8", "cp949", "euc-kr"):
            try:
                return filename.encode("latin1").decode(encoding)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return filename
    return ""


def choose_filename(result: FetchResult, link_text: str, file_types: tuple[str, ...]) -> str:
    candidates = [
        filename_from_content_disposition(result.headers.get("content-disposition", "")),
        filename_from_url_query(result.url),
        link_text,
        posixpath.basename(urllib.parse.unquote(urllib.parse.urlsplit(result.url).path)),
    ]
    filename = next(
        (candidate for candidate in candidates if candidate and not looks_mojibake_name(candidate)),
        "",
    )
    if not filename:
        filename = next((candidate for candidate in candidates if candidate), "")
    filename = strip_download_endpoint_suffix(filename)
    filename_extension = Path(filename).suffix.lower()
    extension = (
        filename_extension
        if filename_extension and filename_extension != ".do"
        else extension_from_text(filename, file_types) or extension_from_url(result.url)
    )
    if not extension:
        content_type = result.headers.get("content-type", "").lower()
        if "pdf" in content_type:
            extension = ".pdf"
        elif "hwp" in content_type:
            extension = ".hwp"
        elif "excel" in content_type or "spreadsheet" in content_type:
            extension = ".xlsx"
        elif "msword" in content_type:
            extension = ".doc"
        else:
            extension = ".bin"

    filename = sanitize_filename(filename)
    if not filename or filename.lower() in {"download", "file", "view"}:
        filename = hashlib.sha1(result.url.encode("utf-8")).hexdigest()[:16]
    if not filename.lower().endswith(extension.lower()):
        filename = f"{filename}{extension}"
    return filename


def strip_download_endpoint_suffix(filename: str) -> str:
    match = re.search(r"(\.(?:pdf|hwp|hwpx|doc|docx|xls|xlsx|ppt|pptx|zip|jpg|jpeg|png|gif|mp4))\.do$", filename, re.I)
    if match:
        return filename[: -len(".do")]
    return filename


def sanitize_filename(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:180]


def looks_mojibake_name(value: str) -> bool:
    return bool(re.search(r"[ÃÂ]|ë\s|ì\s|í\s|[\x80-\x9f]", value))


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(2, 10000):
        candidate = parent / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique file name for {path}")


def write_manifest(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_existing_index(output_dir: Path) -> ExistingIndex:
    urls: set[str] = set()
    names: set[str] = set()
    hashes: set[str] = set()

    manifest = output_dir / "crawl_manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("download_url", "source_page"):
                value = row.get(key)
                if value:
                    urls.add(str(value))
            for key in ("filename", "link_text"):
                value = row.get(key)
                if value:
                    names.add(str(value))
            query_name = filename_from_url_query(str(row.get("download_url", "")))
            if query_name:
                names.add(query_name)
            digest = row.get("sha256")
            if digest:
                hashes.add(str(digest).lower())

    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_file() and path.name != "crawl_manifest.jsonl":
                names.add(path.name)

    return ExistingIndex(urls=urls, names=names, hashes=hashes)


def is_probably_existing(
    url: str,
    link: Link,
    existing: ExistingIndex,
    file_types: tuple[str, ...],
) -> bool:
    if url in existing.urls:
        return True
    query_name = filename_from_url_query(url)
    if query_name and query_name in existing.names:
        return True
    link_name = clean_text(link.text)
    if link_name and link_name in existing.names:
        return True
    if link_name and extension_from_text(link_name, file_types):
        return link_name in existing.names
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("all", *TARGETS.keys()),
        default="all",
        help="institution target to crawl",
    )
    parser.add_argument("--output", default="src/data", help="download root directory")
    parser.add_argument("--max-depth", type=int, default=2, help="maximum page-follow depth from seeds")
    parser.add_argument("--max-pages", type=int, default=200, help="maximum HTML pages per target")
    parser.add_argument("--max-files", type=int, default=100, help="maximum downloaded files per target")
    parser.add_argument("--max-mb", type=int, default=80, help="skip files larger than this size")
    parser.add_argument("--delay", type=float, default=0.6, help="seconds to wait before each request")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=1, help="HTTP retry count per URL")
    parser.add_argument("--dry-run", action="store_true", help="print candidate files without downloading")
    parser.add_argument("--ignore-robots", action="store_true", help="ignore robots.txt checks")
    parser.add_argument(
        "--file-type",
        action="append",
        dest="file_types",
        help="allowed file extension; can be repeated. Defaults to pdf/hwp/hwpx/doc/docx/xls/xlsx",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="extra keyword for deciding which pages to follow",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    target_keys = TARGETS.keys() if args.target == "all" else (args.target,)
    output_root = Path(args.output)

    crawler = Crawler(
        output_root=output_root,
        file_types=tuple(args.file_types or DEFAULT_FILE_TYPES),
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        max_files=args.max_files,
        max_mb=args.max_mb,
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
        dry_run=args.dry_run,
        ignore_robots=args.ignore_robots,
        extra_keywords=tuple(args.keyword),
    )

    started_at = datetime.now(timezone.utc).isoformat()
    summary_rows: list[dict[str, object]] = []
    for key in target_keys:
        target = TARGETS[key]
        stats = crawler.crawl_target(target)
        summary_rows.append(
            {
                "target": key,
                "institution": target.name,
                "pages": stats["pages"],
                "files": stats["files"],
                "skipped": stats["skipped"],
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "dry_run": args.dry_run,
            }
        )

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        write_summary(output_root / "crawl_summary.csv", summary_rows)

    total_files = sum(int(row["files"]) for row in summary_rows)
    print(f"\nDone. candidate/downloaded files: {total_files}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
