from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from collections import Counter
from contextlib import redirect_stdout
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_pnu_site import (  # noqa: E402
    CrawlConfig,
    FetchResult,
    PnuCrawler,
    ResponseRejectedError,
    canonicalize_url,
    classify_response_headers,
    filename_from_headers,
    host_matches,
    load_crawl_scope,
    parse_crawl_scope,
    resolve_seeds,
    should_skip_url,
    main as crawler_main,
)


def scope_payload(
    host: str,
    *,
    max_pages: int = 10,
    max_files: int = 10,
    max_bytes: int = 1024 * 1024,
) -> dict[str, object]:
    return {
        "version": 1,
        "tiers": {
            "core": {
                "hosts": [host],
                "max_items_per_host": 20,
            },
            "department": {
                "hosts": ["department.example.test"],
                "max_items_per_host": 5,
            },
        },
        "budgets": {
            "max_pages": max_pages,
            "max_files": max_files,
            "max_bytes": max_bytes,
        },
        "allowed_document_extensions": [
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
        ],
    }


class TrackingResponse:
    def __init__(self, url: str, content_type: str, filename: str, body: bytes) -> None:
        self.url = url
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        self.headers["Content-Length"] = str(len(body))
        self.body = body
        self.read_calls = 0

    def __enter__(self) -> "TrackingResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, limit: int) -> bytes:
        self.read_calls += 1
        return self.body[:limit]


class StaticOpener:
    def __init__(self, response: TrackingResponse) -> None:
        self.response = response

    def open(self, request: object, timeout: float) -> TrackingResponse:
        return self.response


class FixtureHandler(BaseHTTPRequestHandler):
    requests = Counter()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        type(self).requests[self.path] += 1
        if path == "/robots.txt":
            self._send(
                b"User-agent: *\nDisallow: /private/\n",
                "text/plain; charset=utf-8",
            )
            return
        if path == "/":
            self._send(
                """
                <!doctype html>
                <html lang="ko">
                  <head><title>부산대학교</title></head>
                  <body>
                    <a href="/department/index.do">학과 안내</a>
                    <a href="/bbs/dept/1/10/artclView.do?layout=unknown&utm_source=test">졸업 공지</a>
                    <a href="/bbs/dept/1/99/download.do">졸업서식 첨부파일</a>
                    <a href="/private/secret.do">비공개</a>
                    <a href="/images/logo.png">로고</a>
                    <a href="https://example.com/outside">외부 링크</a>
                  </body>
                </html>
                """.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/budget-root":
            self._send(
                """
                <html><head><title>학사 자료</title></head>
                <body>
                  <a href="/resource?id=graduation">졸업 서식</a>
                  <a href="/ambiguous">일반 안내</a>
                </body></html>
                """.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/resource":
            self._send(b"%PDF-1.4 extensionless", "application/pdf")
            return
        if path == "/ambiguous":
            self._send(
                b"<html><head><title>Other page</title></head></html>",
                "text/html; charset=utf-8",
            )
            return
        if path == "/redirect-private":
            port = self.server.server_address[1]
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://localhost:{port}/private/secret.do",
            )
            self.end_headers()
            return
        if path == "/department/index.do":
            self._send(
                b"<html><head><title>Department</title></head><body>curriculum</body></html>",
                "text/html; charset=utf-8",
            )
            return
        if path == "/bbs/dept/1/10/artclView.do":
            self._send(
                """
                <html><head><title>졸업 공지</title></head>
                <body>
                  졸업요건 안내
                  <a href="/bbs/dept/1/99/download.do">신청서</a>
                </body></html>
                """.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/bbs/dept/1/99/download.do":
            body = b"%PDF-1.4 fixture"
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition",
                'attachment; filename="graduation-form.pdf"',
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/private/secret.do":
            self._send(b"must not be fetched", "text/html")
            return
        self.send_error(404)

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class PnuCrawlerUnitTest(unittest.TestCase):
    def test_canonicalize_removes_tracking_and_normalizes(self) -> None:
        actual = canonicalize_url(
            "../notice/view.do?utm_source=test&page=2&layout=unknown&mCode=MN001#top",
            "https://Dept.PUSAN.ac.kr/a/b/",
        )
        self.assertEqual(
            actual,
            "https://dept.pusan.ac.kr/a/notice/view.do?mCode=MN001&page=2",
        )

    def test_domain_matching_does_not_accept_lookalike(self) -> None:
        self.assertTrue(host_matches("history.pusan.ac.kr", ("pusan.ac.kr",)))
        self.assertTrue(host_matches("pusan.ac.kr", ("pusan.ac.kr",)))
        self.assertFalse(host_matches("pusan.ac.kr.example.com", ("pusan.ac.kr",)))
        self.assertFalse(host_matches("evilpusan.ac.kr", ("pusan.ac.kr",)))

    def test_blocks_login_mutation_and_static_assets(self) -> None:
        self.assertEqual(
            should_skip_url("https://www.pusan.ac.kr/login.do"),
            (True, "blocked_path"),
        )
        self.assertEqual(
            should_skip_url("https://dept.pusan.ac.kr/board/delete.do"),
            (True, "mutating_route"),
        )
        self.assertEqual(
            should_skip_url("https://dept.pusan.ac.kr/images/logo.png"),
            (True, "static_asset"),
        )

    def test_decodes_url_encoded_attachment_filename(self) -> None:
        headers = Message()
        headers["Content-Disposition"] = (
            'attachment; filename="%EC%A1%B8%EC%97%85+%EC%8B%A0%EC%B2%AD%EC%84%9C.pdf"'
        )
        self.assertEqual(
            filename_from_headers(headers, "https://www.pusan.ac.kr/download.do", "application/pdf"),
            "졸업 신청서.pdf",
        )

    def test_infers_hwp_extensions_without_content_disposition(self) -> None:
        headers = Message()
        cases = (
            ("application/vnd.hancom.hwp", "download.hwp"),
            ("application/vnd.hancom.hwpx", "download.hwpx"),
        )
        for content_type, expected in cases:
            with self.subTest(content_type=content_type):
                self.assertEqual(
                    filename_from_headers(
                        headers,
                        "https://www.pusan.ac.kr/download",
                        content_type,
                    ),
                    expected,
                )

    def test_scope_uses_exact_hosts_and_tier_budgets(self) -> None:
        scope = parse_crawl_scope(scope_payload("www.pusan.ac.kr"))

        self.assertEqual(scope.tier_for_host("www.pusan.ac.kr").name, "core")
        self.assertEqual(
            scope.tier_for_host("department.example.test").max_items_per_host,
            5,
        )
        self.assertIsNone(scope.tier_for_host("child.www.pusan.ac.kr"))
        self.assertIsNone(scope.tier_for_host("pusan.ac.kr.example.com"))

    def test_default_scope_excludes_research_hosts(self) -> None:
        scope = load_crawl_scope(ROOT / "config" / "pnu-crawl-scope.json")

        self.assertIsNotNone(scope.tier_for_host("archaeology.pusan.ac.kr"))
        self.assertIsNotNone(scope.tier_for_host("physicaledu.pusan.ac.kr"))
        for host in (
            "educom.pusan.ac.kr",
            "his.pusan.ac.kr",
            "sciedu.pusan.ac.kr",
            "urbanpc.pusan.ac.kr",
        ):
            self.assertIsNone(scope.tier_for_host(host))

    def test_default_seed_set_includes_each_scoped_host(self) -> None:
        scope = load_crawl_scope(ROOT / "config" / "pnu-crawl-scope.json")
        arguments = argparse.Namespace(
            replace_default_seeds=False,
            seed=[],
        )

        seeds = resolve_seeds(arguments, scope.hosts)

        for host in scope.hosts:
            self.assertIn(f"https://{host}/", seeds)

    def test_rejects_hidden_binary_downloads_from_headers(self) -> None:
        cases = (
            ("application/x-msdownload", "campus-photo.jpg"),
            ("video/mp4", "orientation.mp4"),
            ("application/zip", "conference-data.zip"),
            ("application/json", "records.json"),
            ("application/octet-stream", "program.exe"),
        )
        for content_type, filename in cases:
            with self.subTest(content_type=content_type, filename=filename):
                headers = Message()
                headers["Content-Type"] = content_type
                headers["Content-Disposition"] = (
                    f'attachment; filename="{filename}"'
                )
                with self.assertRaises(ResponseRejectedError):
                    classify_response_headers(
                        "https://www.pusan.ac.kr/download.do",
                        headers,
                        content_type,
                    )

    def test_generic_binary_requires_allowed_filename(self) -> None:
        allowed = Message()
        allowed["Content-Disposition"] = 'attachment; filename="notice.pdf"'
        rejected = Message()
        rejected["Content-Disposition"] = 'attachment; filename="notice.bin"'

        self.assertEqual(
            classify_response_headers(
                "https://www.pusan.ac.kr/download.do",
                allowed,
                "application/octet-stream",
            ),
            "attachment",
        )
        with self.assertRaises(ResponseRejectedError):
            classify_response_headers(
                "https://www.pusan.ac.kr/download.do",
                rejected,
                "application/octet-stream",
            )

    def test_document_mime_cannot_bypass_scope_extension_allowlist(self) -> None:
        headers = Message()
        headers["Content-Disposition"] = 'attachment; filename="blocked.docx"'
        docx_type = (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        )

        with self.assertRaisesRegex(
            ResponseRejectedError,
            "outside scope",
        ):
            classify_response_headers(
                "https://www.pusan.ac.kr/download.do",
                headers,
                docx_type,
                allowed_extensions={".pdf"},
            )

        route_headers = Message()
        self.assertEqual(
            classify_response_headers(
                "https://www.pusan.ac.kr/download.do",
                route_headers,
                "application/pdf",
                allowed_extensions={".pdf"},
            ),
            "attachment",
        )
        with self.assertRaisesRegex(
            ResponseRejectedError,
            "configured extension scope",
        ):
            classify_response_headers(
                "https://www.pusan.ac.kr/download.do",
                route_headers,
                docx_type,
                allowed_extensions={".pdf"},
            )


class PnuCrawlerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FixtureHandler.requests.clear()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def make_config(self, output: Path) -> CrawlConfig:
        port = self.server.server_address[1]
        return CrawlConfig(
            output=output,
            seeds=(f"http://127.0.0.1:{port}/",),
            allowed_domains=("127.0.0.1",),
            max_pages=3,
            max_files=3,
            max_depth=3,
            max_pages_per_host=20,
            max_response_bytes=1024 * 1024,
            timeout=5,
            delay=0,
            retries=0,
            obey_robots=True,
            allow_private=True,
            refresh=False,
            dry_run=False,
        )

    def test_crawls_pages_and_attachment_then_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            config = self.make_config(output)

            crawler = PnuCrawler(config)
            try:
                first = crawler.run()
            finally:
                crawler.close()

            self.assertEqual(first["this_run"]["pages"], 3)
            self.assertEqual(first["this_run"]["files"], 1)
            self.assertEqual(first["database"]["pages"], 3)
            self.assertEqual(first["database"]["files"], 1)
            self.assertEqual(FixtureHandler.requests["/private/secret.do"], 0)
            self.assertEqual(FixtureHandler.requests["/images/logo.png"], 0)

            html_files = list(
                (output / "content" / "부산대학교" / "웹페이지").rglob("*.html")
            )
            pdf_files = list(
                (output / "content" / "부산대학교" / "첨부파일").rglob("*.pdf")
            )
            self.assertEqual(len(html_files), 3)
            self.assertEqual(len(pdf_files), 1)
            self.assertEqual(pdf_files[0].read_bytes(), b"%PDF-1.4 fixture")

            pages = [
                json.loads(line)
                for line in (output / "state" / "pages.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            attachments = [
                json.loads(line)
                for line in (output / "state" / "attachments.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(pages), 3)
            self.assertEqual(len(attachments), 1)
            self.assertIn("source_url", attachments[0])
            self.assertIn("storage_path", attachments[0])

            protected_request_counts = {
                path: FixtureHandler.requests[path]
                for path in (
                    "/",
                    "/bbs/dept/1/10/artclView.do?layout=unknown&utm_source=test",
                    "/bbs/dept/1/99/download.do",
                    "/department/index.do",
                )
            }
            resumed = PnuCrawler(config)
            try:
                second = resumed.run()
            finally:
                resumed.close()
            self.assertEqual(second["this_run"]["pages"], 0)
            self.assertEqual(second["this_run"]["files"], 0)
            for path, count in protected_request_counts.items():
                self.assertEqual(FixtureHandler.requests[path], count)
            self.assertEqual(FixtureHandler.requests["/private/secret.do"], 0)

    def test_output_lock_blocks_duplicate_crawler_and_recovers_after_close(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            config = self.make_config(output)
            url = config.seeds[0] + "claimed"
            first = PnuCrawler(config)
            try:
                first.store.enqueue(
                    url,
                    depth=0,
                    priority=0,
                    parent_url=None,
                    anchor_text="",
                    kind="page",
                )
                claimed = first.store.claim_next()
                self.assertEqual(claimed["url"], url)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "another crawler is already using output state",
                ):
                    PnuCrawler(config)

                status = first.store.connection.execute(
                    "SELECT status FROM frontier WHERE url = ?",
                    (url,),
                ).fetchone()["status"]
                self.assertEqual(status, "fetching")
            finally:
                first.close()

            resumed = PnuCrawler(config)
            try:
                recovered = resumed.store.connection.execute(
                    "SELECT status FROM frontier WHERE url = ?",
                    (url,),
                ).fetchone()["status"]
                self.assertEqual(recovered, "queued")
            finally:
                resumed.close()

    def test_status_is_read_only_and_does_not_requeue_active_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            config = self.make_config(output)
            url = config.seeds[0] + "claimed"
            crawler = PnuCrawler(config)
            try:
                crawler.store.enqueue(
                    url,
                    depth=0,
                    priority=0,
                    parent_url=None,
                    anchor_text="",
                    kind="page",
                )
                crawler.store.claim_next()

                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = crawler_main(
                        ["--output", str(output), "--status"]
                    )
                self.assertEqual(exit_code, 0)
                self.assertEqual(json.loads(stdout.getvalue())["fetching"], 1)

                status = crawler.store.connection.execute(
                    "SELECT status FROM frontier WHERE url = ?",
                    (url,),
                ).fetchone()["status"]
                self.assertEqual(status, "fetching")
            finally:
                crawler.close()

    def test_legacy_page_kind_migration_runs_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            config = self.make_config(output)
            url = config.seeds[0] + "ambiguous"
            crawler = PnuCrawler(config)
            try:
                crawler.store.enqueue(
                    url,
                    depth=0,
                    priority=0,
                    parent_url=None,
                    anchor_text="",
                    kind="page",
                )
                crawler.store.connection.execute(
                    """
                    DELETE FROM metadata
                    WHERE key = 'provisional_kind_version'
                    """
                )
                crawler.store.connection.execute(
                    "UPDATE frontier SET status = 'error' WHERE url = ?",
                    (url,),
                )
                crawler.store.connection.commit()
            finally:
                crawler.close()

            config.refresh = True
            migrated = PnuCrawler(config)
            try:
                row = migrated.store.connection.execute(
                    "SELECT status, kind FROM frontier WHERE url = ?",
                    (url,),
                ).fetchone()
                self.assertEqual((row["status"], row["kind"]), ("queued", "unknown"))
                migrated.store.connection.execute(
                    "UPDATE frontier SET kind = 'page' WHERE url = ?",
                    (url,),
                )
                migrated.store.connection.commit()
            finally:
                migrated.close()

            config.refresh = False
            reopened = PnuCrawler(config)
            try:
                kind = reopened.store.connection.execute(
                    "SELECT kind FROM frontier WHERE url = ?",
                    (url,),
                ).fetchone()["kind"]
                self.assertEqual(kind, "page")
            finally:
                reopened.close()

    def test_extensionless_attachment_uses_file_budget_after_page_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            config = self.make_config(output)
            config.seeds = (config.seeds[0] + "budget-root",)
            config.max_pages = 1
            config.max_files = 2
            ambiguous_before = FixtureHandler.requests["/ambiguous"]
            crawler = PnuCrawler(config)
            try:
                summary = crawler.run()
            finally:
                crawler.close()

            self.assertEqual(summary["this_run"]["pages"], 1)
            self.assertEqual(summary["this_run"]["files"], 1)
            stored = list((output / "content").rglob("*.pdf"))
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].read_bytes(), b"%PDF-1.4 extensionless")
            ambiguous_after_first = FixtureHandler.requests["/ambiguous"]
            self.assertEqual(ambiguous_after_first, ambiguous_before + 1)

            resumed = PnuCrawler(config)
            try:
                resumed.run()
            finally:
                resumed.close()
            self.assertEqual(
                FixtureHandler.requests["/ambiguous"],
                ambiguous_after_first,
            )

    def test_redirect_target_obeys_its_robots_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.make_config(Path(temporary) / "crawl")
            config.allowed_domains = ("127.0.0.1", "localhost")
            crawler = PnuCrawler(config)
            private_before = FixtureHandler.requests["/private/secret.do"]
            try:
                with self.assertRaisesRegex(
                    PermissionError,
                    "redirect target blocked by robots.txt",
                ):
                    crawler.fetch(config.seeds[0] + "redirect-private")
                self.assertIn("localhost", crawler.last_request_at)
            finally:
                crawler.close()

            self.assertEqual(
                FixtureHandler.requests["/private/secret.do"],
                private_before,
            )

    def test_hidden_download_is_rejected_before_body_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self.make_config(Path(temporary) / "crawl")
            crawler = PnuCrawler(config)
            response = TrackingResponse(
                config.seeds[0] + "download.do",
                "application/x-msdownload",
                "orientation-video.mp4",
                b"must not be read",
            )
            crawler.opener = StaticOpener(response)
            try:
                with self.assertRaises(ResponseRejectedError):
                    crawler.fetch(response.url, check_robots=False)
            finally:
                crawler.close()

            self.assertEqual(response.read_calls, 0)

    def test_declared_canonical_page_is_stored_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            config = self.make_config(output)
            config.max_depth = 0
            crawler = PnuCrawler(config)
            canonical = config.seeds[0] + "notice/42"
            aliases = (canonical + "?p=1", canonical + "?p=2")
            headers = Message()
            headers["Content-Type"] = "text/html; charset=utf-8"
            try:
                for index, alias in enumerate(aliases):
                    crawler.store.enqueue(
                        alias,
                        depth=0,
                        priority=10 - index,
                        parent_url=None,
                        anchor_text="공지",
                        kind="page",
                    )
                    row = crawler.store.claim_next()
                    self.assertIsNotNone(row)
                    body = (
                        '<html><head><link rel="canonical" href="{}">'
                        "<title>공지 {}</title></head><body>본문</body></html>"
                    ).format(canonical, index).encode("utf-8")
                    crawler.process_html(
                        row,
                        FetchResult(
                            requested_url=alias,
                            final_url=alias,
                            status=200,
                            headers=headers,
                            body=body,
                            content_type="text/html",
                            kind="page",
                        ),
                    )

                rows = list(
                    crawler.store.connection.execute(
                        """
                        SELECT status, canonical_url, storage_path, error
                        FROM frontier
                        WHERE url IN (?, ?)
                        ORDER BY url
                        """,
                        aliases,
                    )
                )
            finally:
                crawler.close()

            self.assertEqual(
                [row["status"] for row in rows],
                ["done", "skipped"],
            )
            self.assertTrue(
                all(row["canonical_url"] == canonical for row in rows)
            )
            self.assertIsNotNone(rows[0]["storage_path"])
            self.assertIsNone(rows[1]["storage_path"])
            self.assertIn("duplicate_canonical", rows[1]["error"])
            self.assertEqual(
                len(list((output / "content").rglob("*.html"))),
                1,
            )

    def test_scope_page_budget_is_cumulative_across_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            config = self.make_config(output)
            host = config.allowed_domains[0]
            config.scope = parse_crawl_scope(
                scope_payload(host, max_pages=1, max_files=3)
            )

            crawler = PnuCrawler(config)
            try:
                first = crawler.run()
            finally:
                crawler.close()
            self.assertEqual(first["this_run"]["pages"], 1)
            self.assertEqual(first["database"]["pages"], 1)
            self.assertGreater(first["database"].get("queued", 0), 0)

            resumed = PnuCrawler(config)
            try:
                second = resumed.run()
            finally:
                resumed.close()
            self.assertEqual(second["this_run"]["pages"], 0)
            self.assertEqual(second["database"]["pages"], 1)
            self.assertGreater(second["database"].get("queued", 0), 0)

    def test_rejects_scope_hash_mismatch_on_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            first_config = self.make_config(output)
            first_config.scope = parse_crawl_scope(
                scope_payload(first_config.allowed_domains[0], max_pages=1)
            )
            crawler = PnuCrawler(first_config)
            crawler.close()

            changed_config = self.make_config(output)
            changed_config.scope = parse_crawl_scope(
                scope_payload(changed_config.allowed_domains[0], max_pages=2)
            )
            with self.assertRaisesRegex(RuntimeError, "scope does not match"):
                PnuCrawler(changed_config)

    def test_legacy_database_is_rejected_before_any_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            database = output / "state" / "crawl.sqlite3"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE frontier (
                    url TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    kind TEXT NOT NULL
                );
                INSERT INTO frontier(url, status, kind)
                VALUES ('https://www.pusan.ac.kr/legacy', 'fetching', 'page');
                """
            )
            connection.commit()
            connection.close()
            before = database.read_bytes()

            config = self.make_config(output)
            config.scope = parse_crawl_scope(
                scope_payload(config.allowed_domains[0])
            )
            with self.assertRaisesRegex(RuntimeError, "no scope identity"):
                PnuCrawler(config)

            self.assertEqual(database.read_bytes(), before)
            connection = sqlite3.connect(database)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type = 'table'
                        """
                    )
                }
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(frontier)"
                    )
                }
                row = connection.execute(
                    "SELECT status, kind FROM frontier"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(tables, {"frontier"})
            self.assertEqual(columns, {"url", "status", "kind"})
            self.assertEqual(row, ("fetching", "page"))

    def test_rejects_switching_dry_run_database_to_content_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "crawl"
            dry_config = self.make_config(output)
            dry_config.scope = parse_crawl_scope(
                scope_payload(dry_config.allowed_domains[0])
            )
            dry_config.dry_run = True
            crawler = PnuCrawler(dry_config)
            crawler.close()

            content_config = self.make_config(output)
            content_config.scope = dry_config.scope
            content_config.dry_run = False
            with self.assertRaisesRegex(RuntimeError, "storage mode"):
                PnuCrawler(content_config)


if __name__ == "__main__":
    unittest.main()
