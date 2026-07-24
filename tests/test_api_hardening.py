from __future__ import annotations

import json
import http.client
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bm25_search import build_index, rerank_results, search_index
from search_api import (
    ApiError,
    build_gemini_prompt,
    cors_origin_for,
    is_authorized,
    load_env_file,
    parse_top_k,
    public_results,
    SearchHandler,
    validate_question,
)


class QuietSearchHandler(SearchHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def patched_env(**values: str):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ApiHardeningTests(unittest.TestCase):
    def make_index(self, tmp: Path) -> Path:
        chunks_path = tmp / "chunks.jsonl"
        index_path = tmp / "bm25.sqlite"
        rows = []
        for index in range(2):
            text = f"ragtestterm 문서 {index} " + ("가나다 " * 40) + f"뒤쪽 핵심 조건 {index}"
            rows.append(
                {
                    "chunk_id": f"doc{index}#0000",
                    "doc_id": f"doc{index}",
                    "chunk_index": 0,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "한국거래소",
                        "source_path": f"src/data/한국거래소/test-{index}.pdf",
                        "relative_path": f"한국거래소/test-{index}.pdf",
                        "file_name": f"test-{index}.pdf",
                        "extension": ".pdf",
                        "parser": "test",
                    },
                }
            )
        chunks_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        build_index(chunks_path, index_path, batch_size=10)
        return index_path

    def test_env_file_does_not_override_existing_values_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patched_env(GEMINI_MODEL="from-env"):
            env_file = Path(tmp) / ".env"
            env_file.write_text("GEMINI_MODEL=from-file\n", encoding="utf-8")

            load_env_file(env_file)

            self.assertEqual(os.environ["GEMINI_MODEL"], "from-env")

            load_env_file(env_file, override=True)
            self.assertEqual(os.environ["GEMINI_MODEL"], "from-file")

    def test_top_k_is_clamped_to_safe_bounds(self) -> None:
        with patched_env(RAG_MAX_TOP_K="3"):
            self.assertEqual(parse_top_k("-1"), 1)
            self.assertEqual(parse_top_k("2"), 2)
            self.assertEqual(parse_top_k("999"), 3)
            self.assertEqual(parse_top_k("not-a-number", default=2), 2)

    def test_question_validation_rejects_empty_and_oversized_questions(self) -> None:
        with self.assertRaises(ApiError) as empty:
            validate_question("   ")
        self.assertEqual(empty.exception.error, "question_required")

        with patched_env(RAG_MAX_QUESTION_CHARS="100"):
            with self.assertRaises(ApiError) as oversized:
                validate_question("1" * 101)
        self.assertEqual(oversized.exception.error, "question_too_long")

    def test_cors_and_optional_api_token(self) -> None:
        with patched_env(RAG_ALLOWED_ORIGINS="http://localhost:5173"):
            self.assertEqual(cors_origin_for("http://localhost:5173"), "http://localhost:5173")
            self.assertIsNone(cors_origin_for("https://example.com"))

        with patched_env(RAG_API_TOKEN="secret"):
            self.assertTrue(is_authorized({"X-RAG-API-Key": "secret"}))
            self.assertTrue(is_authorized({"Authorization": "Bearer secret"}))
            self.assertFalse(is_authorized({"X-RAG-API-Key": "wrong"}))

    def test_search_can_return_full_text_for_internal_rag_only(self) -> None:
        long_text = "ragtestterm 상장폐지 제도 안내 " + ("가나다 " * 180) + "뒤쪽 핵심 조건은 개선심사 일정입니다."
        chunk = {
            "chunk_id": "doc1#0000",
            "doc_id": "doc1",
            "chunk_index": 0,
            "text": long_text,
            "char_count": len(long_text),
            "metadata": {
                "institution": "한국거래소",
                "source_path": "src/data/한국거래소/test.pdf",
                "relative_path": "한국거래소/test.pdf",
                "file_name": "test.pdf",
                "extension": ".pdf",
                "parser": "test",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            index_path = Path(tmp) / "bm25.sqlite"
            chunks_path.write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")
            build_index(chunks_path, index_path, batch_size=10)

            public = search_index(index_path, "ragtestterm", 5, None, preview_chars=50)
            internal = search_index(index_path, "ragtestterm", 5, None, preview_chars=50, include_text=True)

        self.assertNotIn("text", public[0])
        self.assertLessEqual(len(public[0]["preview"]), 50)
        self.assertIn("text", internal[0])
        self.assertIn("뒤쪽 핵심 조건", internal[0]["text"])

    def test_rerank_prefers_better_question_term_coverage_over_raw_rank(self) -> None:
        rows = [
            {
                "chunk_id": "weak#0000",
                "institution": "한국거래소",
                "file_name": "weak.pdf",
                "relative_path": "한국거래소/weak.pdf",
                "preview": "상장폐지 관련 일반 안내입니다.",
            },
            {
                "chunk_id": "strong#0000",
                "institution": "한국거래소",
                "file_name": "strong.pdf",
                "relative_path": "한국거래소/strong.pdf",
                "preview": "상장폐지 제도 개선 심사 일정과 이의신청 절차를 안내합니다.",
            },
        ]

        reranked = rerank_results("상장폐지 제도 개선 심사 일정", rows, top_k=1)

        self.assertEqual(reranked[0]["chunk_id"], "strong#0000")

    def test_prompt_uses_full_chunk_but_response_strips_internal_text(self) -> None:
        result = {
            "source_number": 1,
            "institution": "한국거래소",
            "file_name": "test.pdf",
            "chunk_index": 0,
            "chunk_id": "doc1#0000",
            "preview": "앞부분만 있는 preview",
            "text": "앞부분만 있는 preview. 뒤쪽 핵심 조건은 개선심사 일정입니다.",
        }

        prompt = build_gemini_prompt("상장폐지 개선심사 일정은?", [result])
        self.assertIn("뒤쪽 핵심 조건", prompt)
        self.assertNotIn("text", public_results([result])[0])

    def test_http_chat_enforces_origin_auth_and_public_response_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patched_env(
            RAG_ALLOWED_ORIGINS="http://localhost:5173",
            RAG_API_TOKEN="secret",
            RAG_GENERATION_MODE="extractive",
            RAG_MAX_TOP_K="1",
        ):
            QuietSearchHandler.index_path = self.make_index(Path(tmp))
            server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                body = json.dumps({"question": "ragtestterm", "top_k": 99}).encode("utf-8")

                unauthorized = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                unauthorized.request(
                    "POST",
                    "/chat",
                    body=body,
                    headers={"Content-Type": "application/json", "Origin": "http://localhost:5173"},
                )
                self.assertEqual(unauthorized.getresponse().status, 401)
                unauthorized.close()

                forbidden = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                forbidden.request(
                    "POST",
                    "/chat",
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "https://example.com",
                        "X-RAG-API-Key": "secret",
                    },
                )
                self.assertEqual(forbidden.getresponse().status, 403)
                forbidden.close()

                allowed = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                allowed.request(
                    "POST",
                    "/chat",
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "http://localhost:5173",
                        "X-RAG-API-Key": "secret",
                    },
                )
                response = allowed.getresponse()
                payload = json.loads(response.read().decode("utf-8"))
                allowed.close()

                self.assertEqual(response.status, 200)
                self.assertEqual(len(payload["results"]), 1)
                self.assertNotIn("text", payload["results"][0])
                self.assertIn("뒤쪽 핵심 조건", payload["answer"])
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
