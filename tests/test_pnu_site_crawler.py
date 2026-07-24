from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from collections import Counter
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crawl_pnu_site import (  # noqa: E402
    CrawlConfig,
    PnuCrawler,
    canonicalize_url,
    filename_from_headers,
    host_matches,
    should_skip_url,
)


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


if __name__ == "__main__":
    unittest.main()
