from __future__ import annotations

import json
import http.client
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bm25_search import (
    build_dense_index,
    build_index,
    rerank_results,
    select_document_diverse_results,
    search_index,
)
from search_api import (
    ApiError,
    ParserIndexTarget,
    attribute_claim,
    build_rag_response,
    build_gemini_prompt,
    build_parser_index_registry,
    chat_candidate_limit,
    create_hybrid_retriever,
    cors_origin_for,
    extract_critical_values,
    generation_provider_status,
    is_authorized,
    load_env_file,
    local_models,
    normalize_retrieval_query,
    parse_top_k,
    public_results,
    replace_with_adjacent_temporal_contexts,
    resolve_parser_target,
    search_pipeline,
    select_answer_claims,
    select_distinct_contexts,
    SearchHandler,
    validate_question,
    validate_requested_model,
)
from rag.generators import build_prompt as build_generation_prompt
from rag.retrieval import lexical_fallback_rerank


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
    def post_chat(self, port: int, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            "/chat",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
        status = response.status
        connection.close()
        return status, body

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

    def test_parser_profile_target_defaults_and_rejects_unknown_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.sqlite"
            cascade_path = root / "cascade.sqlite"
            baseline_path.touch()
            cascade_path.touch()
            targets = {
                "baseline": ParserIndexTarget(
                    "baseline",
                    baseline_path,
                    None,
                    "dense_index_missing",
                ),
                "cascade": ParserIndexTarget(
                    "cascade",
                    cascade_path,
                    None,
                    "dense_index_missing",
                ),
            }

            default_target = resolve_parser_target(None, targets, "cascade")
            baseline_target = resolve_parser_target(
                "baseline",
                targets,
                "cascade",
            )

            self.assertEqual(default_target.profile, "cascade")
            self.assertEqual(baseline_target.index_path, baseline_path)
            with self.assertRaises(ApiError) as unknown:
                resolve_parser_target("unknown", targets, "cascade")
            with self.assertRaises(ApiError) as invalid_type:
                resolve_parser_target(123, targets, "cascade")

        self.assertEqual(unknown.exception.error, "invalid_parser_profile")
        self.assertEqual(invalid_type.exception.error, "invalid_parser_profile")

    def test_parser_index_registry_rejects_profile_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cascade_path = self.make_index(root)
            connection = sqlite3.connect(str(cascade_path))
            connection.execute(
                "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('profile', 'cascade')"
            )
            connection.commit()
            connection.close()

            registry, default_profile = build_parser_index_registry(
                cascade_path,
                [],
                "cascade",
            )

            self.assertEqual(default_profile, "cascade")
            self.assertEqual(registry, {"cascade": cascade_path})
            with self.assertRaises(ValueError):
                build_parser_index_registry(
                    cascade_path,
                    [f"baseline={cascade_path}"],
                    "cascade",
                )

    def test_http_chat_uses_requested_parser_profile_without_global_switching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patched_env(RAG_API_TOKEN=""):
            root = Path(tmp)
            baseline_path = root / "baseline.sqlite"
            cascade_path = root / "cascade.sqlite"
            baseline_path.touch()
            cascade_path.touch()
            targets = {
                "baseline": ParserIndexTarget(
                    "baseline",
                    baseline_path,
                    None,
                    "dense_index_missing",
                ),
                "cascade": ParserIndexTarget(
                    "cascade",
                    cascade_path,
                    None,
                    "dense_index_missing",
                ),
            }

            with (
                patch.object(
                    QuietSearchHandler,
                    "parser_targets",
                    targets,
                ),
                patch.object(
                    QuietSearchHandler,
                    "default_parser_profile",
                    "cascade",
                ),
            ):
                server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                port = server.server_address[1]

                try:
                    with patch(
                        "search_api.search_pipeline",
                        return_value=([], {"strategy": "test"}),
                    ) as search_mock:
                        status, payload = self.post_chat(
                            port,
                            {
                                "question": "ragtestterm",
                                "provider": "auto",
                                "parser_profile": "baseline",
                            },
                        )
                        invalid_status, invalid_payload = self.post_chat(
                            port,
                            {
                                "question": "ragtestterm",
                                "provider": "auto",
                                "parser_profile": "unknown",
                            },
                        )

                    self.assertEqual(status, 200)
                    self.assertEqual(payload["parser_profile"], "baseline")
                    self.assertEqual(
                        payload["retrieval"]["parser_profile"],
                        "baseline",
                    )
                    self.assertEqual(search_mock.call_args.args[0], baseline_path)
                    self.assertEqual(invalid_status, 400)
                    self.assertEqual(
                        invalid_payload["error"],
                        "invalid_parser_profile",
                    )
                    self.assertEqual(search_mock.call_count, 1)
                finally:
                    server.shutdown()
                    server.server_close()

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

    def test_local_model_allowlist_and_health_metadata(self) -> None:
        with patched_env(
            RAG_LOCAL_MODEL="local/default",
            RAG_LOCAL_MODELS="local/second, local/default,local/third",
            RAG_LOCAL_BASE_URL="http://127.0.0.1:8080/v1",
            RAG_LOCAL_API_KEY="test-only-value",
        ):
            models = local_models()
            status = generation_provider_status()["local"]

        self.assertEqual(models, ["local/default", "local/second", "local/third"])
        self.assertTrue(status["configured"])
        self.assertEqual(status["model"], "local/default")
        self.assertEqual(status["default_model"], "local/default")
        self.assertEqual(
            status["models"],
            [
                {"id": "local/default", "label": "default", "available": True},
                {"id": "local/second", "label": "second", "available": True},
                {"id": "local/third", "label": "third", "available": True},
            ],
        )
        self.assertNotIn("base_url", status)
        self.assertNotIn("api_key", status)

    def test_frontier_status_uses_gemini_configuration(self) -> None:
        with patched_env(
            RAG_GEMINI_API_KEY="gemini-secret",
            RAG_GEMINI_MODEL="gemini-test",
            GEMINI_API_KEY="",
            GOOGLE_API_KEY="",
            GEMINI_MODEL="",
            RAG_FRONTIER_MODEL="",
            RAG_FRONTIER_API_KEY="",
            OPENAI_API_KEY="",
        ):
            status = generation_provider_status()["frontier"]

        self.assertTrue(status["configured"])
        self.assertEqual(status["label"], "프론티어 AI (Gemini)")
        self.assertEqual(status["model"], "gemini-test")
        self.assertEqual(status["implementation"], "gemini")

        with patched_env(
            RAG_GEMINI_API_KEY="",
            GEMINI_API_KEY="",
            GOOGLE_API_KEY="",
            RAG_FRONTIER_MODEL="legacy-openai-model",
            RAG_FRONTIER_API_KEY="legacy-openai-key",
            OPENAI_API_KEY="",
        ):
            legacy_status = generation_provider_status()["frontier"]

        self.assertFalse(legacy_status["configured"])

    def test_http_frontier_alias_routes_to_gemini_generator(self) -> None:
        result = {
            "chunk_id": "doc#0000",
            "doc_id": "doc",
            "chunk_index": 0,
            "text": "ragtestterm 근거 문장입니다.",
            "preview": "ragtestterm 근거 문장입니다.",
            "institution": "테스트",
            "file_name": "test.pdf",
            "locations": [],
        }
        generated = SimpleNamespace(
            text="ragtestterm 근거 문장입니다.",
            requested="gemini",
            used="gemini",
            model="gemini-test",
            metadata=lambda: {
                "requested": "gemini",
                "used": "gemini",
                "model": "gemini-test",
                "fallback_reason": None,
                "attempts": [],
            },
        )
        with patched_env(
            RAG_API_TOKEN="",
            GEMINI_API_KEY="gemini-secret",
            GOOGLE_API_KEY="",
            GEMINI_MODEL="gemini-test",
        ):
            QuietSearchHandler.generation_semaphore = threading.BoundedSemaphore(2)
            server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                with (
                    patch(
                        "search_api.search_pipeline",
                        return_value=([result], {"strategy": "test"}),
                    ),
                    patch("search_api.generate", return_value=generated) as generate_mock,
                ):
                    status, payload = self.post_chat(
                        port,
                        {
                            "question": "ragtestterm",
                            "provider": "frontier",
                        },
                    )

                self.assertEqual(status, 200)
                self.assertEqual(
                    generate_mock.call_args.kwargs["requested"],
                    "gemini",
                )
                self.assertEqual(payload["generation"]["requested"], "frontier")
                self.assertEqual(payload["generation"]["used"], "frontier")
                self.assertEqual(payload["generation"]["implementation"], "gemini")
            finally:
                server.shutdown()
                server.server_close()

    def test_local_model_validation_uses_default_and_exact_allowlist(self) -> None:
        with patched_env(
            RAG_LOCAL_MODEL="local/default",
            RAG_LOCAL_MODELS="local/second,local/third",
        ):
            self.assertEqual(validate_requested_model("local"), "local/default")
            self.assertEqual(
                validate_requested_model("local", "local/second"),
                "local/second",
            )
            with self.assertRaises(ApiError) as unknown:
                validate_requested_model("local", "local/unknown")
            with self.assertRaises(ApiError) as padded:
                validate_requested_model("local", " local/second ")
            with self.assertRaises(ApiError) as wrong_provider:
                validate_requested_model("gemini", "local/second")

        self.assertEqual(unknown.exception.error, "invalid_local_model")
        self.assertEqual(padded.exception.error, "invalid_local_model")
        self.assertEqual(
            wrong_provider.exception.error,
            "model_not_allowed_for_provider",
        )

        with patched_env(
            RAG_LOCAL_MODEL="",
            RAG_LOCAL_MODELS="local/first,local/second",
        ):
            self.assertEqual(local_models(), ["local/first", "local/second"])
            self.assertEqual(validate_requested_model("local"), "local/first")

    def test_http_rejects_unknown_local_model_before_retrieval_or_generation(self) -> None:
        with patched_env(
            RAG_API_TOKEN="",
            RAG_LOCAL_MODEL="local/default",
            RAG_LOCAL_MODELS="local/default,local/second",
        ):
            server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                with (
                    patch("search_api.search_pipeline") as search_mock,
                    patch("search_api.generate") as generate_mock,
                ):
                    status, payload = self.post_chat(
                        port,
                        {
                            "question": "ragtestterm",
                            "provider": "local",
                            "model": "local/unknown",
                        },
                    )

                self.assertEqual(status, 400)
                self.assertEqual(payload["error"], "invalid_local_model")
                search_mock.assert_not_called()
                generate_mock.assert_not_called()
            finally:
                server.shutdown()
                server.server_close()

    def test_http_forwards_selected_local_model_to_generator(self) -> None:
        result = {
            "chunk_id": "doc#0000",
            "doc_id": "doc",
            "chunk_index": 0,
            "text": "ragtestterm 근거 문장입니다.",
            "preview": "ragtestterm 근거 문장입니다.",
            "institution": "테스트",
            "file_name": "test.pdf",
            "locations": [],
        }
        generated = SimpleNamespace(
            text="ragtestterm 근거 문장입니다.",
            requested="local",
            used="local",
            model="local/second",
            metadata=lambda: {
                "requested": "local",
                "used": "local",
                "model": "local/second",
                "fallback_reason": None,
                "attempts": [],
            },
        )
        with patched_env(
            RAG_API_TOKEN="",
            RAG_LOCAL_MODEL="local/default",
            RAG_LOCAL_MODELS="local/default,local/second",
        ):
            QuietSearchHandler.local_generation_semaphore = threading.BoundedSemaphore(1)
            server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                with (
                    patch(
                        "search_api.search_pipeline",
                        return_value=([result], {"strategy": "test"}),
                    ),
                    patch("search_api.generate", return_value=generated) as generate_mock,
                ):
                    status, _ = self.post_chat(
                        port,
                        {
                            "question": "ragtestterm",
                            "provider": "local",
                            "model": "local/second",
                        },
                    )

                self.assertEqual(status, 200)
                self.assertEqual(
                    generate_mock.call_args.kwargs["requested_model"],
                    "local/second",
                )
            finally:
                server.shutdown()
                server.server_close()

    def test_busy_direct_local_slot_does_not_block_gemini_slot(self) -> None:
        result = {
            "chunk_id": "doc#0000",
            "doc_id": "doc",
            "chunk_index": 0,
            "text": "ragtestterm 근거 문장입니다.",
            "preview": "ragtestterm 근거 문장입니다.",
            "institution": "테스트",
            "file_name": "test.pdf",
            "locations": [],
        }
        generated = SimpleNamespace(
            text="ragtestterm 근거 문장입니다.",
            requested="gemini",
            used="gemini",
            model="gemini-test",
            metadata=lambda: {
                "requested": "gemini",
                "used": "gemini",
                "model": "gemini-test",
                "fallback_reason": None,
                "attempts": [],
            },
        )
        with patched_env(
            RAG_API_TOKEN="",
            RAG_LOCAL_MODEL="local/default",
            RAG_LOCAL_MODELS="local/default",
        ):
            local_slot = threading.BoundedSemaphore(1)
            self.assertTrue(local_slot.acquire(blocking=False))
            QuietSearchHandler.local_generation_semaphore = local_slot
            QuietSearchHandler.generation_semaphore = threading.BoundedSemaphore(2)
            server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                with (
                    patch(
                        "search_api.search_pipeline",
                        return_value=([result], {"strategy": "test"}),
                    ),
                    patch("search_api.generate", return_value=generated) as generate_mock,
                ):
                    local_status, _ = self.post_chat(
                        port,
                        {
                            "question": "ragtestterm",
                            "provider": "local",
                        },
                    )
                    gemini_status, _ = self.post_chat(
                        port,
                        {
                            "question": "ragtestterm",
                            "provider": "gemini",
                        },
                    )

                self.assertEqual(local_status, 429)
                self.assertEqual(gemini_status, 200)
                generate_mock.assert_called_once()
                self.assertEqual(
                    generate_mock.call_args.kwargs["requested"],
                    "gemini",
                )
            finally:
                local_slot.release()
                server.shutdown()
                server.server_close()

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

    def test_document_diverse_selection_preserves_bm25_order(self) -> None:
        rows = [
            {
                "chunk_id": "first#0000",
                "document_id": "first",
                "bm25_score": -10.0,
                "retrieval": {"bm25": {"rank": 1, "score": -10.0}},
            },
            {
                "chunk_id": "first#0001",
                "document_id": "first",
                "bm25_score": -9.0,
                "retrieval": {"bm25": {"rank": 2, "score": -9.0}},
            },
            {
                "chunk_id": "first#0002",
                "document_id": "first",
                "bm25_score": -8.0,
                "retrieval": {"bm25": {"rank": 3, "score": -8.0}},
            },
            {
                "chunk_id": "second#0000",
                "document_id": "second",
                "bm25_score": -7.0,
                "retrieval": {"bm25": {"rank": 4, "score": -7.0}},
            },
        ]

        selected = select_document_diverse_results(rows, top_k=4)

        self.assertEqual(
            [row["chunk_id"] for row in selected],
            ["first#0000", "first#0001", "second#0000"],
        )
        self.assertEqual(
            [row["retrieval"]["final_rank"] for row in selected],
            [1, 2, 3],
        )
        self.assertTrue(
            all(row["retrieval"]["reranker"] is None for row in selected)
        )

    def test_context_dedupe_fills_requested_slots_and_preserves_fact_changes(
        self,
    ) -> None:
        shared = (
            "2026학년도 1학기 수료후연구생 신청 안내입니다. "
            "학생지원시스템에서 지도교수 추천서를 첨부해 신청합니다. "
        ) * 3
        near_duplicate = shared.replace("안내입니다", "안내 입니다", 1)
        changed_semester = shared.replace("1학기", "2학기", 1)
        rows = [
            {
                "chunk_id": "first#0000",
                "institution": "부산대학교",
                "text": shared,
                "retrieval": {"final_rank": 1},
            },
            {
                "chunk_id": "exact#0000",
                "institution": "부산대학교",
                "text": shared.replace(" ", "\u00a0"),
                "retrieval": {"final_rank": 2},
            },
            {
                "chunk_id": "near#0000",
                "institution": "부산대학교",
                "text": near_duplicate,
                "retrieval": {"final_rank": 3},
            },
            {
                "chunk_id": "changed#0000",
                "institution": "부산대학교",
                "text": changed_semester,
                "retrieval": {"final_rank": 4},
            },
            {
                "chunk_id": "other#0000",
                "institution": "부산대학교",
                "text": "완전히 다른 신청 취소 및 환불 안내 본문입니다.",
                "retrieval": {"final_rank": 5},
            },
        ]

        selected, diagnostics = select_distinct_contexts(rows, top_k=4)

        self.assertEqual(
            [row["chunk_id"] for row in selected],
            ["first#0000", "near#0000", "changed#0000", "other#0000"],
        )
        self.assertEqual(
            [row["retrieval"]["final_rank"] for row in selected],
            [1, 2, 3, 4],
        )
        self.assertEqual(diagnostics["candidate_count"], 5)
        self.assertEqual(diagnostics["kept_count"], 4)
        self.assertEqual(diagnostics["removed_count"], 1)
        self.assertEqual(diagnostics["exact_removed"], 1)
        self.assertEqual(diagnostics["near_removed"], 0)
        self.assertEqual(diagnostics["scanned_count"], 5)
        self.assertEqual(diagnostics["unscanned_count"], 0)
        self.assertEqual(chat_candidate_limit(8), 32)

    def test_context_dedupe_preserves_opposite_meaning(self) -> None:
        allowed = (
            "신청 대상자는 학생지원시스템에서 온라인으로 신청할 수 있습니다. "
            "자세한 절차와 제출 서류는 첨부 안내문을 확인하시기 바랍니다. "
        ) * 3
        denied = allowed.replace(
            "신청할 수 있습니다",
            "신청할 수 없습니다",
            1,
        )

        selected, diagnostics = select_distinct_contexts(
            [
                {
                    "chunk_id": "allowed#0000",
                    "institution": "부산대학교",
                    "text": allowed,
                },
                {
                    "chunk_id": "denied#0000",
                    "institution": "부산대학교",
                    "text": denied,
                },
            ],
            top_k=2,
        )

        self.assertEqual(
            [row["chunk_id"] for row in selected],
            ["allowed#0000", "denied#0000"],
        )
        self.assertEqual(diagnostics["removed_count"], 0)

    def test_temporal_neighbor_replaces_hit_with_relevant_table_chunk(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "2026학년도 2학기 수료후연구생 신청 안내",
                "대학원 석사 또는 박사과정 수료자로서 연구를 희망하는 자",
                " 신청기간",
                (
                    "구 분\t학생 신청기간\n"
                    "1차\t‘26. 8. 3.(월)∼8. 10.(월) 18:00\n"
                    "2차\t‘26. 9. 2.(수)∼9. 9.(수) 18:00"
                ),
                (
                    "구 분\t등록금 납부기간\n"
                    "1차 등록\t8. 24.(월)∼8. 27.(목)"
                ),
            ]
            rows = [
                {
                    "chunk_id": f"notice#000{index}",
                    "doc_id": "notice",
                    "document_id": "notice",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/notice.pdf",
                        "relative_path": "부산대학교/notice.pdf",
                        "file_name": "notice.pdf",
                        "extension": ".pdf",
                        "parser": "test",
                        "table_ids": (
                            ["notice:table"]
                            if index in {3, 4}
                            else []
                        ),
                    },
                }
                for index, text in enumerate(texts)
            ]
            chunks_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            build_index(chunks_path, index_path, batch_size=10)
            heading = {
                **rows[2],
                "preview": rows[2]["text"],
                "institution": "부산대학교",
                "metadata": {},
            }

            expanded, diagnostics = (
                replace_with_adjacent_temporal_contexts(
                    index_path,
                    [heading],
                    "수료후연구생 1차 신청 기간",
                )
            )
            registration_table = {
                **rows[4],
                "preview": rows[4]["text"],
                "institution": "부산대학교",
                "metadata": {},
            }
            replaced_registration, registration_diagnostics = (
                replace_with_adjacent_temporal_contexts(
                    index_path,
                    [registration_table],
                    "수료후연구생 1차 신청 기간",
                )
            )
            unchanged, disabled_diagnostics = (
                replace_with_adjacent_temporal_contexts(
                    index_path,
                    [heading],
                    "수료후연구생 신청 방법",
                )
            )

        self.assertEqual(
            [row["chunk_id"] for row in expanded],
            ["notice#0003"],
        )
        self.assertTrue(diagnostics["enabled"])
        self.assertEqual(diagnostics["replaced_count"], 1)
        self.assertEqual(
            expanded[0]["retrieval"]["context_expansion"][
                "anchor_chunk_id"
            ],
            "notice#0002",
        )
        self.assertEqual(
            [row["chunk_id"] for row in replaced_registration],
            ["notice#0003"],
        )
        self.assertEqual(
            registration_diagnostics["replaced_count"],
            1,
        )
        self.assertEqual(
            [row["chunk_id"] for row in unchanged],
            ["notice#0002"],
        )
        self.assertFalse(disabled_diagnostics["enabled"])

    def test_claim_attribution_accepts_korean_paraphrase(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "text": (
                    "온라인 신청이 원칙이며 학생지원시스템 로그인 후 학적, "
                    "학생신청, 수료후연구생 신청 순서로 진행한다. "
                    "부득이한 경우 소속 학과 문의 후 수기 신청할 수 있다."
                ),
            }
        ]

        claim = attribute_claim(
            (
                "온라인 시스템의 학적 관련 메뉴에서 신청할 수 있으며, "
                "상황에 따라 학과를 통한 수기 접수도 가능합니다."
            ),
            results,
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")
        self.assertEqual(claim["source_numbers"], [1])

    def test_claim_attribution_rejects_mismatched_critical_values(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "text": (
                    "2026학년도 2학기 신청 기간은 "
                    "2026년 8월 1일부터 2026년 8월 7일까지이다."
                ),
            }
        ]

        claim = attribute_claim(
            (
                "2026학년도 2학기 신청 마감일은 "
                "2026년 9월 30일입니다."
            ),
            results,
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"],
            "critical_value_mismatch",
        )
        self.assertIn(
            "date:2026-09-30",
            claim["missing_critical_values"],
        )

    def test_claim_attribution_parses_pnu_abbreviated_date_range(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "source_title": (
                    "2026학년도 2학기 수료후연구생 신청 안내"
                ),
                "text": (
                    "구 분\t학생 신청기간\t학과 승인기간\n"
                    "1차\t‘26. 8. 3.(월)∼8. 10.(월) 18:00\t"
                    "8. 13.(목)까지"
                ),
            }
        ]

        claim = attribute_claim(
            (
                "2026학년도 2학기 1차 신청 기간은 "
                "2026년 8월 3일부터 "
                "8월 10일 18시까지입니다."
            ),
            results,
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["missing_critical_values"], [])

    def test_claim_attribution_does_not_mix_values_across_table_rows(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "text": (
                    "1차\t‘26. 8. 3.(월)∼8. 10.(월) 18:00\n"
                    "2차\t‘26. 9. 2.(수)∼9. 9.(수) 18:00"
                ),
            }
        ]

        claim = attribute_claim(
            "1차 신청 시작일은 2026년 9월 2일입니다.",
            results,
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"],
            "critical_value_mismatch",
        )

    def test_claim_attribution_requires_meaningful_lexical_support(self) -> None:
        claim = attribute_claim(
            "학생지원시스템에서 등록금을 현금으로 납부합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "학생지원시스템에서 온라인으로 신청합니다.",
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"],
            "low_lexical_overlap",
        )

    def test_claim_attribution_normalizes_equivalent_currency_amounts(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "text": "수료후연구생 등록금은 10만원이며 온라인으로 납부한다.",
            }
        ]

        claim = attribute_claim(
            "수료후연구생 등록금은 100,000원이며 온라인으로 납부합니다.",
            results,
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_rejects_compound_currency_mismatches(self) -> None:
        cases = [
            ("등록금은 5천원입니다.", "등록금은 2만5천원입니다."),
            ("예산은 1조원입니다.", "예산은 2조원입니다."),
        ]
        for source, draft in cases:
            with self.subTest(source=source, draft=draft):
                claim = attribute_claim(
                    draft,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "critical_value_mismatch",
                )

    def test_claim_attribution_preserves_percent_direction(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "text": "전년 대비 증감률은 △0.3% 감소했습니다.",
            }
        ]

        decreased = attribute_claim(
            "전년 대비 증감률은 0.3% 감소했습니다.",
            results,
        )
        increased = attribute_claim(
            "전년 대비 증감률은 0.3% 증가했습니다.",
            results,
        )

        self.assertTrue(decreased["supported"])
        self.assertFalse(increased["supported"])
        self.assertEqual(
            increased["validation_reason"],
            "critical_value_mismatch",
        )

    def test_claim_attribution_normalizes_equivalent_clock_times(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "text": "온라인 신청은 18:00에 마감합니다.",
            }
        ]

        claim = attribute_claim(
            "온라인 신청은 오후 6시에 마감합니다.",
            results,
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_duration_is_not_extracted_as_clock_time(self) -> None:
        self.assertNotIn(
            "time_minutes:120",
            extract_critical_values("처리에는 2시간이 걸립니다."),
        )

    def test_time_expression_does_not_create_quantity_fact(self) -> None:
        self.assertNotIn(
            "quantity:0:부",
            extract_critical_values("고지서는 10:00부터 출력할 수 있습니다."),
        )

    def test_digit_dense_table_row_does_not_stall_currency_extraction(
        self,
    ) -> None:
        text = (
            "정보의생명공학대학 정보컴퓨터공학부(컴퓨터공학전공) "
            "1.22(목) (일반) 1927102 ~ 1927182 32 "
            "0.5208333333333334 IT관(102) 4 406"
        )

        started = time.perf_counter()
        facts = extract_critical_values(text)
        elapsed = time.perf_counter() - started

        self.assertFalse(
            any(value.startswith("amount_krw:") for value in facts)
        )
        self.assertLess(elapsed, 0.5)

    def test_substantive_negative_fact_is_not_model_abstention(self) -> None:
        text = "신청 상태는 시스템에서 확인할 수 없습니다."
        claim = attribute_claim(
            text,
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": text,
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_model_abstention_is_reported_without_attribution_error(self) -> None:
        response = build_rag_response(
            "신청 기간은?",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "수료후연구생 신청 방법을 안내한다.",
                }
            ],
            "제공된 문서에서 해당 내용을 확인할 수 없습니다.",
            "gemini:test",
        )

        self.assertEqual(
            response["answer"],
            "제공된 검색 근거에서 질문에 답할 내용을 확인할 수 없습니다.",
        )
        self.assertEqual(
            response["claims"][0]["validation_reason"],
            "model_abstention",
        )

    def test_short_model_abstention_is_not_replaced_by_extractive_answer(
        self,
    ) -> None:
        response = build_rag_response(
            "신청 기간은?",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "수료후연구생 신청 방법을 안내한다.",
                }
            ],
            "문서에서 확인되지 않습니다.",
            "gemini:test",
        )

        self.assertEqual(
            response["answer"],
            "제공된 검색 근거에서 질문에 답할 내용을 확인할 수 없습니다.",
        )
        self.assertEqual(
            response["claims"][0]["validation_reason"],
            "model_abstention",
        )

    def test_topic_prefixed_model_abstention_is_recognized(self) -> None:
        response = build_rag_response(
            "1차 신청 기간은?",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "수료후연구생 신청 방법을 안내한다.",
                }
            ],
            (
                "2026학년도 2학기 수료후연구생 1차 신청 기간은 "
                "제공된 문서에서 확인할 수 없습니다."
            ),
            "local:test",
        )

        self.assertEqual(
            response["answer"],
            "제공된 검색 근거에서 질문에 답할 내용을 확인할 수 없습니다.",
        )
        self.assertEqual(
            response["claims"][0]["validation_reason"],
            "model_abstention",
        )

    def test_supported_claim_does_not_report_rejected_source_values(self) -> None:
        claim = attribute_claim(
            "신청일은 2026년 9월 30일입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "wrong#0001",
                    "text": "신청일은 2026년 8월 1일입니다.",
                },
                {
                    "source_number": 2,
                    "chunk_id": "right#0001",
                    "text": "접수 신청일 2026-09-30일입니다.",
                },
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["missing_critical_values"], [])
        self.assertLessEqual(claim["best_score"], 1.0)

    def test_prompt_treats_duplicate_sources_as_one_fact_not_invalid_evidence(
        self,
    ) -> None:
        prompt = build_generation_prompt(
            "신청 방법은?",
            [
                {"chunk_id": "one", "text": "온라인으로 신청한다."},
                {"chunk_id": "two", "text": "온라인으로 신청한다."},
            ],
        )

        self.assertIn("반복되었다는 이유로 그 사실을 버리", prompt)

    def test_retrieval_query_removes_institution_and_request_boilerplate(self) -> None:
        self.assertEqual(
            normalize_retrieval_query(
                "부산대학교 휴학 관련 규정을 검색해줘",
                "부산대학교",
            ),
            "휴학",
        )
        self.assertEqual(normalize_retrieval_query("규정"), "규정")

    def test_hybrid_reranker_prefers_matching_article_heading_and_diversifies(self) -> None:
        rows = [
            {
                "chunk_id": f"election#000{index}",
                "doc_id": "election",
                "chunk_index": index,
                "institution": "부산대학교",
                "file_name": "부산대학교 총장임용후보자 선정규정.hwp",
                "text": "선거권자는 휴학 또는 정학 중이 아닌 학생이어야 한다.",
            }
            for index in range(4)
        ]
        rows.extend(
            [
                {
                    "chunk_id": "rules#0000",
                    "doc_id": "rules",
                    "chunk_index": 0,
                    "institution": "부산대학교",
                    "file_name": "부산대학교 학칙 전문.hwp",
                    "text": "제64조(휴학) 학생은 정당한 사유가 있을 때 휴학할 수 있다.",
                },
                {
                    "chunk_id": "guide#0000",
                    "doc_id": "guide",
                    "chunk_index": 0,
                    "institution": "부산대학교",
                    "file_name": "학생 안내.hwp",
                    "text": "휴학 신청은 학사과에 제출한다.",
                },
            ]
        )

        reranked = lexical_fallback_rerank("휴학", rows, top_k=4)

        self.assertEqual(reranked[0].chunk_id, "rules#0000")
        self.assertLessEqual(
            sum(hit.document_id == "election" for hit in reranked),
            2,
        )

    def test_hybrid_reranker_uses_crawl_source_metadata(self) -> None:
        rows = [
            {
                "chunk_id": "generic#0000",
                "doc_id": "generic",
                "institution": "부산대학교",
                "file_name": "notice.pdf",
                "relative_path": "부산대학교/notice.pdf",
                "preview": "일반 안내 본문",
            },
            {
                "chunk_id": "metadata#0000",
                "doc_id": "metadata",
                "institution": "부산대학교",
                "file_name": "attachment.pdf",
                "relative_path": "부산대학교/attachment.pdf",
                "preview": "첨부 문서 본문",
                "metadata": {
                    "source_title": "2026학년도 등록금 납부 안내",
                    "category": "registration",
                },
            },
        ]

        reranked = lexical_fallback_rerank(
            "2026학년도 등록금 납부",
            rows,
            top_k=2,
        )

        self.assertEqual(reranked[0].chunk_id, "metadata#0000")

    def test_extractive_claim_selection_matches_korean_word_endings(self) -> None:
        results = [
            {
                "text": "① 학생은 정당한 사유가 있을 때 휴학할 수 있다.",
            },
            {
                "text": "학생선거인은 개인정보 처리동의를 제출한다.",
            },
        ]

        claims = select_answer_claims("휴학", results)

        self.assertIn("휴학할 수 있다", claims[0])

    def test_hybrid_pipeline_runs_dense_rrf_and_returns_stage_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bm25_path = self.make_index(root)
            dense_path = root / "dense.sqlite"
            build_dense_index(bm25_path, dense_path, dimensions=32)
            retriever, warning = create_hybrid_retriever(
                bm25_path,
                dense_path,
            )

            results, trace = search_pipeline(
                bm25_path,
                retriever,
                "ragtestterm 핵심 조건",
                2,
                "한국거래소",
                include_text=False,
            )

        self.assertIsNone(warning)
        self.assertEqual(trace["strategy"], "BM25 + Dense + RRF")
        self.assertEqual(trace["lanes"]["bm25"]["status"], "ok")
        self.assertEqual(trace["lanes"]["dense"]["status"], "ok")
        self.assertEqual(trace["fusion"]["status"], "ok")
        self.assertEqual(len(results), 2)
        self.assertNotIn("text", results[0])
        self.assertIsNotNone(results[0]["scores"]["rrf"])
        self.assertIsNotNone(results[0]["scores"]["reranker"])

    def test_hybrid_retriever_disables_stale_dense_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bm25_path = self.make_index(root)
            dense_path = root / "dense.sqlite"
            build_dense_index(bm25_path, dense_path, dimensions=32)
            replacement_chunks = root / "replacement.jsonl"
            replacement_chunks.write_text(
                json.dumps(
                    {
                        "chunk_id": "replacement#0000",
                        "doc_id": "replacement",
                        "chunk_index": 0,
                        "text": "새 corpus revision",
                        "metadata": {"institution": "테스트"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            build_index(replacement_chunks, bm25_path, batch_size=10)

            retriever, warning = create_hybrid_retriever(
                bm25_path,
                dense_path,
            )
            _, trace = search_pipeline(
                bm25_path,
                retriever,
                "revision",
                1,
                None,
                include_text=False,
            )

        self.assertEqual(warning, "dense_corpus_revision_mismatch")
        self.assertIsNone(retriever)
        self.assertEqual(trace["lanes"]["dense"]["status"], "disabled")

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
        self.assertEqual(
            prompt,
            build_generation_prompt("상장폐지 개선심사 일정은?", [result]),
        )
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
