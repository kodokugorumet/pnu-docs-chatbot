from __future__ import annotations

import json
import hashlib
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
    demote_generic_candidates,
    rerank_results,
    select_document_diverse_results,
    search_index,
)
from search_api import (
    ApiError,
    MAX_CLAIMS,
    ParserIndexTarget,
    RetrievalModeState,
    attribute_claim,
    build_rag_response,
    build_gemini_prompt,
    build_parser_index_registry,
    chat_candidate_limit,
    create_hybrid_retriever,
    cors_origin_for,
    extract_critical_values,
    evaluation_context_trace,
    evaluation_trace_enabled,
    generation_input_trace,
    generation_provider_status,
    generation_may_use_local,
    is_loopback_bind_host,
    is_authorized,
    learned_dense_artifact_path,
    load_env_file,
    local_attempt_model,
    local_models,
    expand_service_retrieval_query,
    normalize_retrieval_query,
    parse_top_k,
    resolve_evaluation_trace_request,
    public_results,
    replace_with_adjacent_temporal_contexts,
    resolve_parser_target,
    resolve_retrieval_mode,
    search_pipeline,
    select_answer_claims,
    select_distinct_contexts,
    share_learned_dense_embedder,
    split_candidate_sentences,
    split_draft_claims,
    _supports_requested_deadline,
    _term_in_text,
    SearchHandler,
    validate_question,
    validate_requested_model,
    with_local_runtime_failure,
)
from local_model_runtime import (
    LocalModelBusyError,
    LocalModelUnmanagedError,
)
from rag.generators import build_prompt as build_generation_prompt
from rag.retrieval import lexical_fallback_rerank


class QuietSearchHandler(SearchHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class FakeLocalModelRuntime:
    def __init__(
        self,
        *,
        unload_error: Exception | None = None,
        generation_error: Exception | None = None,
    ) -> None:
        self.unload_error = unload_error
        self.generation_error = generation_error
        self.generation_models: list[str | None] = []
        self.loaded_models: list[str] = []
        self.active = 0
        self.unload_calls = 0

    def public_status(self) -> dict[str, object]:
        return {
            "runtime_state": "loaded" if self.loaded_models else "unloaded",
            "loaded_model": self.loaded_models[-1] if self.loaded_models else None,
            "unload_supported": True,
            "worker_running": bool(self.loaded_models),
        }

    @contextmanager
    def generation(self, model: str | None = None):
        self.generation_models.append(model)
        if self.generation_error:
            raise self.generation_error
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1

    def note_loaded(self, model: str | None) -> None:
        if model:
            self.loaded_models.append(model)

    def unload(self) -> dict[str, object]:
        self.unload_calls += 1
        if self.unload_error:
            raise self.unload_error
        released = bool(self.loaded_models)
        self.loaded_models.clear()
        return {
            "ok": True,
            "state": "unloaded",
            "loaded_model": None,
            "released": released,
        }

    def close(self) -> None:
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
    def test_format_download_titles_are_demoted_as_generic_shells(self) -> None:
        rows = [
            {
                "chunk_id": "pdf-shell",
                "source_title": "PDF 다운로드",
                "file_name": "",
                "text": "국가유공자 자녀 특별전형 지원 대상과 법률 조항 안내",
            },
            {
                "chunk_id": "scholarship",
                "source_title": (
                    "2026학년도 2학기 학부 우선선발장학금 신청 안내문.hwp"
                ),
                "file_name": "",
                "text": (
                    "국가유공자 자녀의 등록금 지원 신청 기간과 "
                    "학생지원시스템 신청 방법 안내"
                ),
            },
            {
                "chunk_id": "hwp-shell",
                "source_title": "HWP 다운로드",
                "file_name": "",
                "text": "국가유공자 지원 대상",
            },
        ]

        ordered = demote_generic_candidates(
            (
                "국가유공자 자녀인데 부산대에서 등록금을 지원받으려면 "
                "언제까지 어떻게 신청해야 하나요?"
            ),
            rows,
        )

        self.assertEqual(
            [row["chunk_id"] for row in ordered],
            ["scholarship", "pdf-shell", "hwp-shell"],
        )

    def test_general_foreign_admission_query_prefers_general_guide(self) -> None:
        rows = [
            {
                "chunk_id": "recommendation-track",
                "source_title": (
                    "2026학년도 후기 학부 외국인 특별전형 "
                    "재외한국교육원장 추천 트랙 신입학 모집요강.pdf"
                ),
                "file_name": "recommendation.pdf",
                "text": "외국인 특별전형 신입학 지원 자격 안내",
            },
            {
                "chunk_id": "human-rights",
                "source_title": "2020 대학 인권센터 역량강화 워크숍.pdf",
                "file_name": "workshop.pdf",
                "text": "피해자가 외국인 학생인 때 위원 자격 안내",
            },
            {
                "chunk_id": "graduate-guide",
                "source_title": (
                    "2026학년도 전기 일반대학원 외국인 특별전형 "
                    "모집요강.pdf"
                ),
                "file_name": "graduate.pdf",
                "text": "석사과정 외국인 지원 자격과 학력 안내",
            },
            {
                "chunk_id": "general-guide",
                "source_title": (
                    "2026학년도 후기 학부 외국인 특별전형 "
                    "신편입학 모집요강.pdf"
                ),
                "file_name": "general.pdf",
                "text": "지원 자격의 국적, 언어능력, 학력 기준 안내",
            },
        ]

        ordered = demote_generic_candidates(
            (
                "외국인 특별전형 모집요강 지원자격 국적 언어능력 학력 "
                "학부 신입학 자격"
            ),
            rows,
        )

        self.assertEqual(
            [row["chunk_id"] for row in ordered],
            [
                "general-guide",
                "human-rights",
                "recommendation-track",
                "graduate-guide",
            ],
        )

    def test_evaluation_trace_requires_server_opt_in(self) -> None:
        with patched_env(RAG_EVAL_TRACE="0"):
            self.assertFalse(evaluation_trace_enabled())
            with self.assertRaisesRegex(ApiError, "evaluation trace is disabled"):
                resolve_evaluation_trace_request(True)

        with patched_env(RAG_EVAL_TRACE="1"):
            self.assertTrue(evaluation_trace_enabled())
            self.assertTrue(resolve_evaluation_trace_request(True))
            self.assertFalse(resolve_evaluation_trace_request(False))

    def test_evaluation_context_trace_keeps_text_and_stable_hash(self) -> None:
        trace = evaluation_context_trace(
            [
                {
                    "chunk_id": "doc#0001",
                    "doc_id": "doc",
                    "chunk_index": 1,
                    "text": "등록금 납부기간은 8월 24일부터입니다.",
                    "source_title": "등록금 납부 안내",
                    "source_host": "www.pusan.ac.kr",
                    "retrieval": {"final_rank": 3},
                    "private_field": "must-not-leak",
                }
            ]
        )

        self.assertEqual(trace[0]["rank"], 1)
        self.assertEqual(trace[0]["document_id"], "doc")
        self.assertEqual(
            trace[0]["text"],
            "등록금 납부기간은 8월 24일부터입니다.",
        )
        self.assertEqual(len(trace[0]["text_sha256"]), 64)
        self.assertNotIn("private_field", trace[0])

    def test_generation_input_trace_hashes_exact_active_prompt(self) -> None:
        contexts = [
            {
                "source_number": 1,
                "chunk_id": "doc#0001",
                "text": "신청 마감은 8월 31일입니다.",
            }
        ]
        trace = generation_input_trace(
            "신청 마감은 언제인가요?",
            contexts,
            role_perspective="재학생에게 직접 적용되는 기준을 먼저 설명합니다.",
        )
        expected_prompt = build_generation_prompt(
            "신청 마감은 언제인가요?",
            contexts,
            role_perspective="재학생에게 직접 적용되는 기준을 먼저 설명합니다.",
        )

        self.assertEqual(trace["user_prompt"], expected_prompt)
        self.assertEqual(
            trace["prompt_sha256"],
            hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(len(trace["system_instruction_sha256"]), 64)

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

    def test_learned_dense_artifacts_are_scoped_by_parser_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "kure-v1"
            baseline = root / "baseline" / "kure-v1"
            legacy.mkdir(parents=True)

            self.assertEqual(
                learned_dense_artifact_path(root, "cascade", "kure-v1"),
                legacy,
            )
            self.assertEqual(
                learned_dense_artifact_path(root, "baseline", "kure-v1"),
                baseline,
            )

            baseline.mkdir(parents=True)
            self.assertEqual(
                learned_dense_artifact_path(root, "baseline", "kure-v1"),
                baseline,
            )

    def test_learned_dense_query_models_are_shared_across_profiles(self) -> None:
        first_embedder = SimpleNamespace(max_sequence_length=512)
        second_embedder = SimpleNamespace(max_sequence_length=512)
        first = SimpleNamespace(
            model_id="test/model",
            model_revision="revision",
            query_prefix="query: ",
            embedder=first_embedder,
        )
        second = SimpleNamespace(
            model_id="test/model",
            model_revision="revision",
            query_prefix="query: ",
            embedder=second_embedder,
        )
        cache: dict[tuple[object, ...], object] = {}

        share_learned_dense_embedder(first, cache)  # type: ignore[arg-type]
        share_learned_dense_embedder(second, cache)  # type: ignore[arg-type]

        self.assertIs(first.embedder, first_embedder)
        self.assertIs(second.embedder, first_embedder)

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

    def test_retrieval_mode_selection_is_explicit_and_fail_closed(self) -> None:
        fake_retriever = object()
        target = ParserIndexTarget(
            "cascade",
            Path("cascade.sqlite"),
            None,
            None,
            {
                "bm25": RetrievalModeState(
                    "bm25", "BM25", None
                ),
                "kure_dense": RetrievalModeState(
                    "kure_dense", "KURE Dense", fake_retriever  # type: ignore[arg-type]
                ),
                "snowflake_dense": RetrievalModeState(
                    "snowflake_dense",
                    "Snowflake Dense",
                    None,
                    "learned_dense_index_missing",
                ),
            },
        )

        self.assertEqual(resolve_retrieval_mode(target, "bm25").id, "bm25")
        self.assertIs(
            resolve_retrieval_mode(target, "kure_dense").retriever,
            fake_retriever,
        )
        with self.assertRaises(ApiError) as unknown:
            resolve_retrieval_mode(target, "unknown")
        with self.assertRaises(ApiError) as unavailable:
            resolve_retrieval_mode(target, "snowflake_dense")

        self.assertEqual(unknown.exception.error, "invalid_retrieval_mode")
        self.assertEqual(
            unavailable.exception.error,
            "retrieval_mode_unavailable",
        )

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

    def test_local_runtime_helpers_cover_auto_and_attempt_metadata(self) -> None:
        self.assertTrue(is_loopback_bind_host("127.0.0.1"))
        self.assertTrue(is_loopback_bind_host("::1"))
        self.assertFalse(is_loopback_bind_host("0.0.0.0"))
        with patched_env(RAG_AUTO_PROVIDER_ORDER="frontier,local,extractive"):
            self.assertTrue(generation_may_use_local("auto"))
        with patched_env(RAG_AUTO_PROVIDER_ORDER="frontier,extractive"):
            self.assertFalse(generation_may_use_local("auto"))
        with patched_env(RAG_AUTO_PROVIDER_ORDER=""):
            self.assertTrue(generation_may_use_local("auto"))
        with patched_env(RAG_AUTO_PROVIDER_ORDER="unknown,auto"):
            self.assertTrue(generation_may_use_local("auto"))
        self.assertTrue(generation_may_use_local("local"))
        self.assertFalse(generation_may_use_local("gemini"))
        self.assertEqual(
            local_attempt_model(
                [
                    {"provider": "frontier", "model": "remote"},
                    {
                        "provider": "local",
                        "model": "local/qwen",
                        "status": "success",
                    },
                ]
            ),
            "local/qwen",
        )
        self.assertIsNone(
            local_attempt_model(
                [
                    {
                        "provider": "local",
                        "model": "local/qwen",
                        "status": "error",
                    }
                ]
            )
        )
        runtime_failure = with_local_runtime_failure(
            {
                "requested": "local",
                "used": "extractive",
                "fallback_reason": None,
                "attempts": [],
            },
            code="local_model_runtime_unmanaged",
            model="local/qwen",
        )
        self.assertEqual(
            runtime_failure["fallback_reason"],
            "local:local_model_runtime_unmanaged",
        )
        self.assertEqual(
            runtime_failure["attempts"][0]["error"],
            "local_model_runtime_unmanaged",
        )

    def test_health_reports_runtime_separately_from_configured_model(self) -> None:
        runtime = FakeLocalModelRuntime()
        runtime.note_loaded("local/second")
        previous_runtime = QuietSearchHandler.local_model_runtime
        previous_tuning = QuietSearchHandler.retrieval_tuning
        previous_cap = QuietSearchHandler.context_chunks_per_document
        with tempfile.TemporaryDirectory() as tmp:
            QuietSearchHandler.index_path = self.make_index(Path(tmp))
            QuietSearchHandler.local_model_runtime = runtime
            QuietSearchHandler.retrieval_tuning = False
            QuietSearchHandler.context_chunks_per_document = 4
            server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                with patched_env(
                    RAG_API_TOKEN="",
                    RAG_EVAL_TRACE="0",
                    RAG_LOCAL_MODEL="local/default",
                    RAG_LOCAL_MODELS="local/default,local/second",
                ):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", port, timeout=5
                    )
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                    payload = json.loads(response.read().decode("utf-8"))
                    connection.close()

                local = payload["providers"]["local"]
                self.assertEqual(response.status, 200)
                self.assertEqual(local["default_model"], "local/default")
                self.assertEqual(local["loaded_model"], "local/second")
                self.assertEqual(local["runtime_state"], "loaded")
                self.assertTrue(local["unload_supported"])
                self.assertNotIn("base_url", local)
                self.assertNotIn("api_key", local)
                service_config = payload["service_config"]
                self.assertFalse(service_config["retrieval_tuning"])
                self.assertEqual(service_config["context_chunks_per_document"], 4)
                self.assertEqual(service_config["default_context_top_k"], 8)
                self.assertFalse(service_config["evaluation_trace_enabled"])
                generation = service_config["generation"]
                self.assertEqual(generation["provider"], "frontier")
                self.assertIn("gemini-3.5-flash-lite", generation["models"])
                self.assertEqual(
                    generation["models"]["gemini-3.5-flash-lite"],
                    {
                        "max_context_chars": 24000,
                        "max_output_tokens": 900,
                        "sampling_parameters": [],
                    },
                )
                self.assertIn("startup_git_commit", service_config["freeze"])
            finally:
                QuietSearchHandler.local_model_runtime = previous_runtime
                QuietSearchHandler.retrieval_tuning = previous_tuning
                QuietSearchHandler.context_chunks_per_document = previous_cap
                server.shutdown()
                server.server_close()

    def test_http_unloads_local_model_and_maps_busy_or_unmanaged_states(self) -> None:
        previous_runtime = QuietSearchHandler.local_model_runtime
        server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        def request_unload() -> tuple[int, dict[str, object]]:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request(
                "POST",
                "/local-model/unload",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            connection.close()
            return response.status, payload

        try:
            with patched_env(RAG_API_TOKEN=""):
                ready = FakeLocalModelRuntime()
                ready.note_loaded("local/qwen")
                QuietSearchHandler.local_model_runtime = ready
                status, payload = request_unload()
                self.assertEqual(status, 200)
                self.assertTrue(payload["released"])
                self.assertEqual(ready.unload_calls, 1)

                QuietSearchHandler.local_model_runtime = FakeLocalModelRuntime(
                    unload_error=LocalModelBusyError()
                )
                status, payload = request_unload()
                self.assertEqual(status, 409)
                self.assertEqual(payload["error"], "local_model_busy")

                QuietSearchHandler.local_model_runtime = FakeLocalModelRuntime(
                    unload_error=LocalModelUnmanagedError()
                )
                status, payload = request_unload()
                self.assertEqual(status, 501)
                self.assertEqual(
                    payload["error"],
                    "local_model_unload_not_supported",
                )
        finally:
            QuietSearchHandler.local_model_runtime = previous_runtime
            server.shutdown()
            server.server_close()

    def test_local_model_unload_uses_existing_api_authentication(self) -> None:
        previous_runtime = QuietSearchHandler.local_model_runtime
        runtime = FakeLocalModelRuntime()
        QuietSearchHandler.local_model_runtime = runtime
        server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        try:
            with patched_env(RAG_API_TOKEN="secret"):
                connection = http.client.HTTPConnection(
                    "127.0.0.1", port, timeout=5
                )
                connection.request(
                    "POST",
                    "/local-model/unload",
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
                connection.close()

            self.assertEqual(response.status, 401)
            self.assertEqual(runtime.unload_calls, 0)
        finally:
            QuietSearchHandler.local_model_runtime = previous_runtime
            server.shutdown()
            server.server_close()

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
        self.assertEqual(status["default_model"], "gemini-test")
        self.assertEqual(
            [model["id"] for model in status["models"]],
            ["gemini-test", "gemini-3.1-flash-lite"],
        )
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
                            "model": "gemini-test",
                        },
                    )

                self.assertEqual(status, 200)
                self.assertEqual(
                    generate_mock.call_args.kwargs["requested"],
                    "gemini",
                )
                self.assertEqual(
                    generate_mock.call_args.kwargs["requested_model"],
                    "gemini-test",
                )
                self.assertEqual(payload["generation"]["requested"], "frontier")
                self.assertEqual(payload["generation"]["used"], "frontier")
                self.assertEqual(payload["generation"]["implementation"], "gemini")
            finally:
                server.shutdown()
                server.server_close()

    def test_provider_model_validation_uses_exact_allowlists(self) -> None:
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
                validate_requested_model("auto", "local/second")

        self.assertEqual(unknown.exception.error, "invalid_local_model")
        self.assertEqual(padded.exception.error, "invalid_local_model")
        self.assertEqual(
            wrong_provider.exception.error,
            "model_not_allowed_for_provider",
        )

        with patched_env(
            GEMINI_MODEL="gemini-3.5-flash-lite",
            GEMINI_FALLBACK_MODELS="gemini-3.1-flash-lite",
        ):
            self.assertEqual(
                validate_requested_model(
                    "frontier", "gemini-3.1-flash-lite"
                ),
                "gemini-3.1-flash-lite",
            )
            self.assertEqual(
                validate_requested_model(
                    "gemini", "gemini-3.5-flash-lite"
                ),
                "gemini-3.5-flash-lite",
            )
            with self.assertRaises(ApiError) as unknown_gemini:
                validate_requested_model("frontier", "gemini-unknown")

        self.assertEqual(
            unknown_gemini.exception.error,
            "invalid_gemini_model",
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
            attempts=(
                SimpleNamespace(
                    provider="local",
                    model="local/second",
                    status="success",
                ),
            ),
            metadata=lambda: {
                "requested": "local",
                "used": "local",
                "model": "local/second",
                "fallback_reason": None,
                "attempts": [
                    {
                        "provider": "local",
                        "model": "local/second",
                        "status": "success",
                    }
                ],
            },
        )
        runtime = FakeLocalModelRuntime()
        previous_runtime = QuietSearchHandler.local_model_runtime
        with patched_env(
            RAG_API_TOKEN="",
            RAG_LOCAL_MODEL="local/default",
            RAG_LOCAL_MODELS="local/default,local/second",
        ):
            QuietSearchHandler.local_model_runtime = runtime
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
                self.assertEqual(runtime.generation_models, ["local/second"])
                self.assertEqual(runtime.loaded_models, ["local/second"])
                self.assertEqual(runtime.active, 0)
            finally:
                QuietSearchHandler.local_model_runtime = previous_runtime
                server.shutdown()
                server.server_close()

    def test_managed_runtime_conflict_excludes_untrusted_local_endpoint(self) -> None:
        result = {
            "chunk_id": "doc#0000",
            "doc_id": "doc",
            "chunk_index": 0,
            "text": "외부 프로세스에 보내면 안 되는 근거입니다.",
            "preview": "외부 프로세스에 보내면 안 되는 근거입니다.",
            "institution": "테스트",
            "file_name": "test.pdf",
            "locations": [],
        }
        generated = SimpleNamespace(
            text="안전한 추출 답변입니다.",
            requested="local",
            used="extractive",
            model=None,
            attempts=(),
            metadata=lambda: {
                "requested": "local",
                "used": "extractive",
                "model": None,
                "fallback_reason": None,
                "attempts": [],
            },
        )
        runtime = FakeLocalModelRuntime(
            generation_error=LocalModelUnmanagedError()
        )
        previous_runtime = QuietSearchHandler.local_model_runtime
        QuietSearchHandler.local_model_runtime = runtime
        server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]

        try:
            with (
                patched_env(
                    RAG_API_TOKEN="",
                    RAG_LOCAL_MODEL="local/default",
                    RAG_LOCAL_MODELS="local/default",
                ),
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
                        "provider": "local",
                        "model": "local/default",
                    },
                )

            self.assertEqual(status, 200)
            self.assertEqual(
                generate_mock.call_args.kwargs["excluded_providers"],
                ("local",),
            )
            self.assertEqual(runtime.loaded_models, [])
            self.assertEqual(runtime.active, 0)
            self.assertEqual(
                payload["generation"]["fallback_reason"],
                "local:local_model_unload_not_supported",
            )
        finally:
            QuietSearchHandler.local_model_runtime = previous_runtime
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

    def test_diverse_selection_preserved_chunk_replaces_same_document_lowest(
        self,
    ) -> None:
        # grant_031 (2026-08-11): CE 리랭커가 같은 문서의 표면 유사 청크 둘을
        # 상위로 올려 캡 2를 소진하면, BM25 1위(정답 조항) 청크가 컨텍스트에서
        # 탈락한다. preserve_chunk_id는 그 문서의 최하위 선택분과 교체돼
        # 원래 슬롯 위치를 물려받아야 한다.
        rows = [
            {"chunk_id": "byl#0031", "document_id": "byl"},
            {"chunk_id": "byl#0034", "document_id": "byl"},
            {"chunk_id": "other#0000", "document_id": "other"},
            {"chunk_id": "byl#0015", "document_id": "byl"},  # BM25 1위, CE가 강등
            {"chunk_id": "third#0000", "document_id": "third"},
        ]

        selected = select_document_diverse_results(
            rows, top_k=4, preserve_chunk_id="byl#0015"
        )

        self.assertEqual(
            [row["chunk_id"] for row in selected],
            ["byl#0031", "byl#0015", "other#0000", "third#0000"],
        )
        self.assertEqual(
            [row["retrieval"]["final_rank"] for row in selected],
            [1, 2, 3, 4],
        )

    def test_diverse_selection_preserve_is_noop_when_anchor_survives(self) -> None:
        rows = [
            {"chunk_id": "byl#0015", "document_id": "byl"},
            {"chunk_id": "byl#0031", "document_id": "byl"},
            {"chunk_id": "other#0000", "document_id": "other"},
        ]
        selected = select_document_diverse_results(
            rows, top_k=3, preserve_chunk_id="byl#0015"
        )
        self.assertEqual(
            [row["chunk_id"] for row in selected],
            ["byl#0015", "byl#0031", "other#0000"],
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

    def test_tuned_temporal_completion_keeps_anchor_and_adds_distance_two_date(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "2026학년도 2학기 재학생 등록금 납부 안내",
                "납부 대상 및 기간",
                (
                    "구 분\t납부대상자\t납부기간\n"
                    "본등록\t대학 재학생\t"
                    "2026. 8. 24.(월) ~ 8. 27.(목)"
                ),
                "학생지원시스템에서 고지서를 출력합니다.",
                (
                    "수납은행 본등록은 농협, 부산, 하나, 국민, "
                    "신한, 우리은행 전국지점입니다."
                ),
                "현금납부는 수납은행 방문 또는 가상계좌 이체로 진행합니다.",
                "신용카드 인터넷 납부는 은행 인터넷뱅킹에서 진행합니다.",
            ]
            rows = [
                {
                    "chunk_id": f"tuition#000{index}",
                    "doc_id": "tuition",
                    "document_id": "tuition",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/tuition.pdf",
                        "relative_path": "부산대학교/tuition.pdf",
                        "file_name": "tuition.pdf",
                        "source_title": (
                            "2026학년도 2학기 재학생 등록금 납부 계획"
                        ),
                        "extension": ".pdf",
                        "parser": "test",
                        "table_ids": ["tuition:period"] if index == 2 else [],
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
            selected = []
            for index in (4, 6):
                selected.append(
                    {
                        **rows[index],
                        "source_title": rows[index]["metadata"]["source_title"],
                        "file_name": "tuition.pdf",
                        "institution": "부산대학교",
                        "preview": rows[index]["text"],
                        "retrieval": {"final_rank": len(selected) + 1},
                    }
                )

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "이번 2학기 등록금 본등록은 언제까지 내고 어느 은행에 내나요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )
            control, control_diagnostics = (
                replace_with_adjacent_temporal_contexts(
                    index_path,
                    selected,
                    "이번 2학기 등록금 본등록은 언제까지 내고 어느 은행에 내나요?",
                    facet_completion=False,
                    max_chunks_per_document=2,
                )
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["tuition#0004", "tuition#0002"],
        )
        self.assertEqual(
            sum(row["document_id"] == "tuition" for row in completed),
            2,
        )
        self.assertEqual(diagnostics["mode"], "facet_sibling_completion")
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(
            diagnostics["completions"][0]["distance"],
            2,
        )
        self.assertEqual(
            completed[1]["retrieval"]["context_expansion"]["kind"],
            "temporal_sibling_completion",
        )
        self.assertEqual(
            [row["chunk_id"] for row in control],
            ["tuition#0004", "tuition#0006"],
        )
        self.assertEqual(
            control_diagnostics["mode"],
            "adjacent_table_replacement",
        )

    def test_tuned_procedure_completion_skips_nav_to_distance_two_sibling(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                (
                    "BIDV SMB 앱에 로그인하고 국제송금 메뉴에서 "
                    "한국학비결제를 클릭한 뒤 상세정보를 입력합니다."
                ),
                "모든 서비스 보기",
                (
                    "납부대상학교와 납부항목을 선택하고 관련 서류를 "
                    "업로드한 뒤 출금계좌를 선택하여 OTP 코드를 입력하면 "
                    "등록금 납부가 완료됩니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"bidv#000{index}",
                    "doc_id": "bidv",
                    "document_id": "bidv",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/bidv.pdf",
                        "relative_path": "부산대학교/bidv.pdf",
                        "file_name": "bidv.pdf",
                        "source_title": "BIDV 등록금 납부 방법",
                        "extension": ".pdf",
                        "parser": "test",
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
            anchor = {
                **rows[0],
                "source_title": "BIDV 등록금 납부 방법",
                "file_name": "bidv.pdf",
                "institution": "부산대학교",
                "preview": rows[0]["text"],
                "retrieval": {"final_rank": 1},
            }

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                [anchor],
                "베트남에서 BIDV 앱으로 등록금을 내는 절차가 어떻게 되나요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )
            control, control_diagnostics = (
                replace_with_adjacent_temporal_contexts(
                    index_path,
                    [anchor],
                    "베트남에서 BIDV 앱으로 등록금을 내는 절차가 어떻게 되나요?",
                    facet_completion=False,
                    max_chunks_per_document=2,
                )
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["bidv#0000", "bidv#0002"],
        )
        self.assertNotIn("bidv#0001", [row["chunk_id"] for row in completed])
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(diagnostics["skipped_noise_count"], 1)
        self.assertEqual(
            completed[1]["retrieval"]["context_expansion"]["kind"],
            "procedure_sibling_completion",
        )
        self.assertEqual([row["chunk_id"] for row in control], ["bidv#0000"])
        self.assertEqual(control_diagnostics["mode"], "adjacent_table_replacement")
        self.assertFalse(control_diagnostics["enabled"])

    def test_tuned_multipart_faq_completion_adds_adjacent_answer_facet(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "등록금 고지서는 학생지원시스템에서 출력합니다.",
                (
                    "등록금 납부 후 휴학하면 등록금은 이월됩니다. "
                    "복학 시 별도 절차 없이 납부 처리됩니다."
                ),
                (
                    "휴학 시 등록금 환불은 학과사무실에 반환 신청서를 "
                    "제출하며 1~2주 소요됩니다."
                ),
                "등록금 납부 확인은 학생지원시스템에서 합니다.",
            ]
            rows = [
                {
                    "chunk_id": f"faq#000{index}",
                    "doc_id": "faq",
                    "document_id": "faq",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/faq.pdf",
                        "relative_path": "부산대학교/faq.pdf",
                        "file_name": "faq.pdf",
                        "source_title": "재학생 등록금 자주하는 질문(FAQ)",
                        "extension": ".pdf",
                        "parser": "test",
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
            anchor = {
                **rows[2],
                "source_title": "재학생 등록금 자주하는 질문(FAQ)",
                "file_name": "faq.pdf",
                "institution": "부산대학교",
                "preview": rows[2]["text"],
                "retrieval": {"final_rank": 1},
            }

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                [anchor],
                (
                    "등록금을 이미 냈는데 휴학하려고 해요. "
                    "낸 등록금은 어떻게 되나요? 돌려받을 수도 있나요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )
            control, _ = replace_with_adjacent_temporal_contexts(
                index_path,
                [anchor],
                (
                    "등록금을 이미 냈는데 휴학하려고 해요. "
                    "낸 등록금은 어떻게 되나요? 돌려받을 수도 있나요?"
                ),
                facet_completion=False,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["faq#0002", "faq#0001"],
        )
        self.assertEqual(
            completed[1]["retrieval"]["context_expansion"]["kind"],
            "multipart_faq_sibling_completion",
        )
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual([row["chunk_id"] for row in control], ["faq#0002"])

    def test_tuned_multifacet_replaces_redundant_amount_with_scoped_date(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "장학사업 소개와 문의처를 안내합니다.",
                "장학사업 관련 일반 유의사항입니다.",
                (
                    "2026학년도 2학기 세종 장학사업 신청기간은 "
                    "2026. 7. 6.부터 8. 31.까지입니다."
                ),
                "장학사업 신청 서식 내려받기 안내입니다.",
                (
                    "세종 장학사업 지원 금액은 원금상환 분야 "
                    "1인당 최대 1,000,000원입니다."
                ),
                "세종 장학사업 담당 부서 연락처 안내입니다.",
                (
                    "이자지원액은 최대 500,000원이며, "
                    "2025. 7. 1.부터 2026. 6. 30.까지 발생한 "
                    "이자를 기준으로 합니다. 신청 자격을 장학금 "
                    "수혜 시기까지 지속 유지해야 합니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"aid#000{index}",
                    "doc_id": "aid",
                    "document_id": "aid",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/aid.pdf",
                        "relative_path": "부산대학교/aid.pdf",
                        "file_name": "aid.pdf",
                        "source_title": "2026학년도 세종 장학사업 신청 안내",
                        "extension": ".pdf",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "aid.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((4, 6), start=1)
            ]
            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "2026학년도 2학기 세종 장학사업의 신청 기간과 "
                    "지원 금액을 알려주세요."
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["aid#0004", "aid#0002"],
        )
        self.assertEqual(
            diagnostics["required_query_facets"],
            ["amount", "date"],
        )
        self.assertTrue(diagnostics["explicit_multifacet"])
        self.assertEqual(diagnostics["marginal_replacement_count"], 1)
        replacement = diagnostics["marginal_replacements"][0]
        self.assertEqual(replacement["reason"], "missing_query_facet")
        self.assertEqual(replacement["seed_chunk_id"], "aid#0004")
        self.assertFalse(replacement["seed_replaced"])
        self.assertEqual(replacement["removed_chunk_id"], "aid#0006")
        self.assertEqual(replacement["chunk_id"], "aid#0002")
        self.assertEqual(replacement["added_facets"], ["date"])
        self.assertEqual(replacement["missing_facets_before"], ["date"])

    def test_tuned_budget_breakdown_finds_adjacent_total_and_shares(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "2025학년도 교육·연구 및 학생지도비 기본계획입니다.",
                "교육·연구 및 학생지도비 사업별 지급 기준입니다.",
                "소요예산 표이며 단위는 천원입니다.",
                (
                    "영역\t2025학년도 예산액\t구성비(%)\n"
                    "교육영역\t6,502,600\t21.39\n"
                    "연구영역\t16,936,679\t55.70\n"
                    "학생지도영역\t6,965,921\t22.91\n"
                    "합계\t30,405,200\t100"
                ),
                (
                    "교육부의 승인 결과에 따라 영역별·사업별 "
                    "예산액은 조정될 수 있습니다."
                ),
                "지급대상자 선발 절차를 안내합니다.",
                (
                    "대상 인원 변동에 따라 영역별·사업별 "
                    "예산액은 조정될 수 있습니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"budget#000{index}",
                    "doc_id": "budget",
                    "document_id": "budget",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/교연비기본계획.pdf",
                        "relative_path": "부산대학교/교연비기본계획.pdf",
                        "file_name": "교연비기본계획.pdf",
                        "source_title": (
                            "2025학년도 교육·연구 및 학생지도비 "
                            "지급 기본계획"
                        ),
                        "extension": ".pdf",
                        "parser": "test",
                    },
                }
                for index, text in enumerate(texts)
            ]
            old_budget_text = (
                "영역\t2021학년도 예산액\t구성비(%)\n"
                "교육영역\t5,934,550\t21.00\n"
                "연구영역\t15,428,250\t54.40\n"
                "학생지도영역\t6,957,660\t24.60\n"
                "합계\t28,320,460\t100"
            )
            old_budget = {
                "chunk_id": "old-budget#0000",
                "doc_id": "old-budget",
                "document_id": "old-budget",
                "chunk_index": 0,
                "text": old_budget_text,
                "char_count": len(old_budget_text),
                "metadata": {
                    "institution": "부산대학교",
                    "source_path": "부산대학교/2021교연비기본계획.pdf",
                    "relative_path": "부산대학교/2021교연비기본계획.pdf",
                    "file_name": "2021교연비기본계획.pdf",
                    "source_title": (
                        "2021학년도 교육·연구 및 학생지도비 "
                        "지급 기본계획"
                    ),
                    "extension": ".pdf",
                    "parser": "test",
                },
            }
            rows.append(old_budget)
            chunks_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            build_index(chunks_path, index_path, batch_size=10)
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "교연비기본계획.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((4, 6), start=1)
            ]
            selected.append(
                {
                    **old_budget,
                    "source_title": old_budget["metadata"]["source_title"],
                    "file_name": "2021교연비기본계획.pdf",
                    "institution": "부산대학교",
                    "preview": old_budget["text"],
                    "retrieval": {"final_rank": 3},
                }
            )

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "2025학년도 교육·연구 및 학생지도비 전체 예산 "
                    "규모와 영역별 비중이 어떻게 되나요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["budget#0004", "budget#0003", "old-budget#0000"],
        )
        self.assertEqual(
            diagnostics["required_query_facets"],
            ["budget_amount", "budget_share"],
        )
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(
            diagnostics["completions"][0]["added_facets"],
            ["budget_amount", "budget_share"],
        )
        self.assertIn(
            "percent:21.39",
            extract_critical_values(rows[3]["text"]),
        )
        answer_claims = select_answer_claims(
            (
                "2025학년도 교육·연구 및 학생지도비 전체 예산 "
                "규모와 영역별 비중이 어떻게 되나요?"
            ),
            completed,
        )
        self.assertIn(
            "2025학년도 전체 예산액은 30,405,200이며, "
            "구성비 합계는 100%입니다.",
            answer_claims,
        )
        self.assertIn(
            (
                "영역별 예산액과 구성비는 교육영역 "
                "6,502,600(21.39%), 연구영역 "
                "16,936,679(55.70%), 학생지도영역 "
                "6,965,921(22.91%)입니다."
            ),
            answer_claims,
        )
        self.assertNotIn("28,320,460", " ".join(answer_claims))
        numbered = [
            {**row, "source_number": index}
            for index, row in enumerate(completed, start=1)
        ]
        response = build_rag_response(
            (
                "2025학년도 교육·연구 및 학생지도비 전체 예산 "
                "규모와 영역별 비중이 어떻게 되나요?"
            ),
            numbered,
            "\n".join(answer_claims),
            "extractive",
        )
        self.assertTrue(response["claims"])
        self.assertTrue(all(claim["supported"] for claim in response["claims"]))
        self.assertEqual(
            {citation["chunk_id"] for citation in response["citations"]},
            {"budget#0003"},
        )

    def test_claim_attribution_accepts_budget_share_only_table_summary(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "budget#0001",
            "source_title": "2025학년도 교육·연구 및 학생지도비 기본계획",
            "text": (
                "영역\t2024학년도 구성비(%)\t2025학년도 구성비(%)\n"
                "교육영역\t21.72\t21.39\n"
                "연구영역\t54.44\t55.70\n"
                "학생지도영역\t23.84\t22.91\n"
                "합계\t100\t100"
            ),
        }
        supported = attribute_claim(
            (
                "2025학년도 영역별 구성비는 교육영역 21.39퍼센트와 "
                "연구영역 55.70퍼센트 및 학생지도영역 22.91퍼센트로 "
                "합계 100퍼센트입니다."
            ),
            [source],
        )
        wrong_share = attribute_claim(
            (
                "2025학년도 영역별 구성비는 교육영역 21.40퍼센트와 "
                "연구영역 55.70퍼센트 및 학생지도영역 22.91퍼센트로 "
                "합계 100퍼센트입니다."
            ),
            [source],
        )
        wrong_year = attribute_claim(
            (
                "2026학년도 영역별 구성비는 교육영역 21.39퍼센트와 "
                "연구영역 55.70퍼센트 및 학생지도영역 22.91퍼센트로 "
                "합계 100퍼센트입니다."
            ),
            [source],
        )

        self.assertTrue(supported["supported"], supported)
        self.assertFalse(wrong_share["supported"], wrong_share)
        self.assertFalse(wrong_year["supported"], wrong_year)

    def test_tuned_multifacet_completes_primary_notice_despite_distractor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            correct_texts = [
                "우선선발장학금 안내입니다.",
                (
                    "국가유공자 자녀의 부산대학교 등록금 지원 신청 기간: "
                    "2026. 7. 13.부터 2026. 7. 17.까지입니다.\n"
                    "※ 수업료Ⅱ 금액 확인 방법: 학생지원시스템 – 등록 – "
                    "등록금책정표\n4. 신청방법 및 장학금 지급"
                ),
                (
                    "국가유공자 자녀는 온라인 신청: 부산대학교 "
                    "학생지원시스템 로그인 → 장학 → 장학금신청 → "
                    "우선선발장학금신청에서 증빙서류를 업로드합니다."
                ),
                "국가유공자 자녀 장학금 신청 시 제출할 증빙서류 목록입니다.",
            ]
            correct_rows = [
                {
                    "chunk_id": f"scholarship#000{index}",
                    "doc_id": "scholarship",
                    "document_id": "scholarship",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/우선선발장학금.hwp",
                        "relative_path": "부산대학교/우선선발장학금.hwp",
                        "file_name": "우선선발장학금.hwp",
                        "source_title": (
                            "2026학년도 2학기 학부 우선선발장학금 "
                            "신청 안내문"
                        ),
                        "extension": ".hwp",
                        "parser": "test",
                    },
                }
                for index, text in enumerate(correct_texts)
            ]
            distractor_text = (
                "국가유공자 자녀 전형 신청 기간: 2025. 9. 1.부터 "
                "2025. 9. 5.까지이며, 신청 방법은 입학 홈페이지에 "
                "로그인한 뒤 온라인 신청합니다."
            )
            distractor = {
                "chunk_id": "admission#0000",
                "doc_id": "admission",
                "document_id": "admission",
                "chunk_index": 0,
                "text": distractor_text,
                "char_count": len(distractor_text),
                "metadata": {
                    "institution": "부산대학교",
                    "source_path": "부산대학교/과거입학전형.pdf",
                    "relative_path": "부산대학교/과거입학전형.pdf",
                    "file_name": "과거입학전형.pdf",
                    "source_title": "과거 입학전형 안내",
                    "extension": ".pdf",
                    "parser": "test",
                },
            }
            all_rows = [*correct_rows, distractor]
            chunks_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in all_rows
                ),
                encoding="utf-8",
            )
            build_index(chunks_path, index_path, batch_size=10)

            def selected(row: dict[str, Any], rank: int) -> dict[str, Any]:
                return {
                    **row,
                    "source_title": row["metadata"]["source_title"],
                    "file_name": row["metadata"]["file_name"],
                    "institution": "부산대학교",
                    "preview": row["text"],
                    "retrieval": {"final_rank": rank},
                }

            initial = [
                selected(correct_rows[1], 1),
                selected(correct_rows[3], 2),
                selected(distractor, 3),
            ]
            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                initial,
                (
                    "국가유공자 자녀인데 부산대 등록금 지원을 언제까지 "
                    "어떻게 신청해야 하나요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["scholarship#0001", "scholarship#0002", "admission#0000"],
        )
        self.assertEqual(diagnostics["marginal_replacement_count"], 1)
        replacement = diagnostics["marginal_replacements"][0]
        self.assertEqual(replacement["removed_chunk_id"], "scholarship#0003")
        self.assertEqual(replacement["chunk_id"], "scholarship#0002")
        self.assertEqual(replacement["added_facets"], ["method"])
        self.assertEqual(replacement["missing_facets_before"], ["method"])

    def test_tuned_multifacet_can_seed_from_distant_answer_chunk(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "2026학년도 2학기 수료후연구생 신청 안내입니다.",
                (
                    "대학원 석사 또는 박사과정 수료자로서 "
                    "학위취득에 필요한 연구를 희망하는 자"
                ),
                "신청기간",
                "1차와 2차 학생 신청기간 안내",
                (
                    "신청방법: 학생지원시스템 로그인 → 학적 → "
                    "학생신청 → 수료후연구생 신청"
                ),
                "등록금 납부기간",
                "1차와 2차 등록금 납부기간 안내",
                (
                    "수료후연구생 등록금은 부산대학교 학생지원시스템 "
                    "등록금 책정표 참고. "
                    "납부금액은 수업료Ⅱ의 10%입니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"researcher#000{index}",
                    "doc_id": "researcher",
                    "document_id": "researcher",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/수료후연구생.pdf",
                        "relative_path": "부산대학교/수료후연구생.pdf",
                        "file_name": "수료후연구생.pdf",
                        "source_title": (
                            "2026학년도 2학기 수료후연구생 신청 안내문"
                        ),
                        "extension": ".pdf",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "수료후연구생.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((1, 0), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "박사과정 수료했는데 수료후연구생은 어떻게 "
                    "신청하고 등록금은 얼마나 내나요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["researcher#0001", "researcher#0007"],
        )
        self.assertEqual(diagnostics["completed_count"], 1)
        replacement = diagnostics["marginal_replacements"][0]
        self.assertEqual(replacement["added_facets"], ["amount"])
        self.assertEqual(replacement["missing_facets_before"], ["amount", "method"])

    def test_tuned_multifacet_preserves_rate_while_adding_distant_department(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                *[
                    f"AI학업장려 학자금대출 일반 안내 문단 {index}입니다."
                    for index in range(6)
                ],
                (
                    "대출대상: AI·SW중심대학 및 AI거점대학 사업 "
                    "대상학과 학부생"
                ),
                *[
                    f"AI학업장려 학자금대출 일반 안내 문단 {index}입니다."
                    for index in range(7, 13)
                ],
                (
                    "대출한도: 연간 200만 원 한도이며 개인 총 한도는 "
                    "1,000만 원입니다."
                ),
                (
                    "대출금리: 1.70%(변동금리). 대출한도: 연간 "
                    "200만 원 한도입니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"ai-loan#{index:04d}",
                    "doc_id": "ai-loan",
                    "document_id": "ai-loan",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/AI학업장려대출.hwp",
                        "relative_path": "부산대학교/AI학업장려대출.hwp",
                        "file_name": "AI학업장려대출.hwp",
                        "source_title": (
                            "2026학년도 2학기 AI학업장려 학자금대출 안내"
                        ),
                        "extension": ".hwp",
                        "parser": "test",
                    },
                }
                for index, text in enumerate(texts)
            ]
            distractor_texts = [
                (
                    "다른 사업 대출금리: 1.70%. 대출한도: 연간 "
                    "200만 원입니다."
                ),
                "다른 사업 대출대상: 별도 사업 대상학과 학부생입니다.",
            ]
            distractor_rows = [
                {
                    "chunk_id": f"other-loan#{index:04d}",
                    "doc_id": "other-loan",
                    "document_id": "other-loan",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/다른대출.hwp",
                        "relative_path": "부산대학교/다른대출.hwp",
                        "file_name": "다른대출.hwp",
                        "source_title": "다른 AI학업장려 학자금대출 안내",
                        "extension": ".hwp",
                        "parser": "test",
                    },
                }
                for index, text in enumerate(distractor_texts)
            ]
            chunks_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [*rows, *distractor_rows]
                ),
                encoding="utf-8",
            )
            build_index(chunks_path, index_path, batch_size=20)
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "AI학업장려대출.hwp",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((13, 14), start=1)
            ]
            selected.append(
                {
                    **distractor_rows[0],
                    "source_title": distractor_rows[0]["metadata"][
                        "source_title"
                    ],
                    "file_name": "다른대출.hwp",
                    "institution": "부산대학교",
                    "preview": distractor_rows[0]["text"],
                    "retrieval": {"final_rank": 3},
                }
            )

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "AI학업장려 학자금대출은 어떤 학과 학생이 받을 수 "
                    "있고, 금리랑 한도는 어떻게 되나요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["ai-loan#0006", "ai-loan#0014", "other-loan#0000"],
        )
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(
            diagnostics["required_query_facets"],
            ["amount", "eligibility", "rate"],
        )
        replacement = diagnostics["marginal_replacements"][0]
        self.assertEqual(replacement["removed_chunk_id"], "ai-loan#0013")
        self.assertEqual(replacement["added_facets"], ["eligibility"])

    def test_tuned_deadline_ignores_table_code_and_finds_submission_rule(
        self,
    ) -> None:
        query = (
            "토익 점수로 영어 졸업인증을 받으려면 성적표는 "
            "언제까지 내야 하나요?"
        )
        self.assertFalse(
            _supports_requested_deadline(
                (
                    "토익 졸업인증 성적표를 제출해야 합니다.\n"
                    "대체강좌 수강신청은 10일 전까지 종료됩니다."
                ),
                query,
            )
        )
        self.assertTrue(
            _supports_requested_deadline(
                "후기졸업은 7월 10일 이전까지 성적표를 제출합니다.",
                query,
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                (
                    "토익 점수로 영어 졸업인증을 받으려면 성적표를 "
                    "제출해야 합니다.\n"
                    "대체강좌 수강신청은 계절수업 수강신청 10일 "
                    "전까지 종료되어야 합니다."
                ),
                (
                    "시행절차: 졸업대상자는 전기졸업 1월 10일 이전, "
                    "후기졸업 7월 10일 이전까지 졸업인증 관련 "
                    "성적표를 제출하여야 한다."
                ),
                "영어능력 졸업인증제도 적용 범위 안내입니다.",
                (
                    "TOEIC 졸업인증 신청 구비서류는 공인영어시험 "
                    "성적표 사본입니다."
                ),
                "영어능력 졸업인증제도 처리 부서 안내입니다.",
                "영어능력 졸업인증제도 별지 서식 안내입니다.",
                (
                    "세칙 [별표 2-1]의 한국어 또는 영어 능력시험 "
                    "점수를 적용합니다."
                ),
                "영어능력 졸업인증제도 신청서 작성 안내입니다.",
                (
                    "TOEIC 성적표를 첨부하여 영어 졸업요건 인증을 "
                    "신청합니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"english#{index:04d}",
                    "doc_id": "english",
                    "document_id": "english",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/영어졸업인증.hwp",
                        "relative_path": "부산대학교/영어졸업인증.hwp",
                        "file_name": "영어졸업인증.hwp",
                        "source_title": (
                            "부산대학교 영어 능력 졸업인증제도 시행 지침"
                        ),
                        "extension": ".hwp",
                        "parser": "test",
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
            build_index(chunks_path, index_path, batch_size=20)
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "영어졸업인증.hwp",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((8, 3), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "토익 점수로 영어 졸업인증을 받으려면 성적표는 "
                    "언제까지 내야 하나요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["english#0008", "english#0001"],
        )
        self.assertNotIn("english#0006", [row["chunk_id"] for row in completed])
        self.assertEqual(diagnostics["completions"][0]["chunk_id"], "english#0001")

    def test_tuned_multifacet_finds_distant_thesis_deadline_and_fee(
        self,
    ) -> None:
        self.assertTrue(
            _term_in_text(
                "학위논문",
                "학위청구논문 심사계획일정 및 서류전체",
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "2025학년도 후기 학위청구논문 심사 일정 안내입니다.",
                (
                    "학위논문 심사요구서와 연구윤리서약서는 "
                    "2026. 04. 09.(목) 17시까지 제출합니다. "
                    "심사료는 석사 10만원, "
                    "박사 30만원입니다."
                ),
                *[
                    f"학위청구논문 심사 관련 별지 안내 {index}입니다."
                    for index in range(2, 18)
                ],
                "석사학위논문 계획서와 본 논문 제출 확인서입니다.",
                "학위청구논문 심사위원 추천서 서식입니다.",
                "학위청구논문 연구윤리 준수 서약서입니다.",
                "학위청구논문 표절방지 검사결과표 서식입니다.",
                "박사학위논문 계획서와 본 논문 제출 확인서입니다.",
            ]
            rows = [
                {
                    "chunk_id": f"thesis#{index:04d}",
                    "doc_id": "thesis",
                    "document_id": "thesis",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/학위논문심사.hwp",
                        "relative_path": "부산대학교/학위논문심사.hwp",
                        "file_name": "학위논문심사.hwp",
                        "source_title": (
                            "학위청구논문 심사계획일정 및 서류전체 "
                            "(2025 후기)"
                        ),
                        "extension": ".hwp",
                        "parser": "test",
                    },
                }
                for index, text in enumerate(texts)
            ]
            distractor_text = (
                "심사 서류 제출일은 2026. 04. 09.(목)입니다. "
                "교육대학원 석사 심사료는 10만원입니다."
            )
            distractor = {
                "chunk_id": "education-thesis#0000",
                "doc_id": "education-thesis",
                "document_id": "education-thesis",
                "chunk_index": 0,
                "text": distractor_text,
                "char_count": len(distractor_text),
                "metadata": {
                    "institution": "부산대학교",
                    "source_path": "부산대학교/교육대학원논문일정.hwp",
                    "relative_path": "부산대학교/교육대학원논문일정.hwp",
                    "file_name": "교육대학원논문일정.hwp",
                    "source_title": (
                        "2025학년도 후기 교육대학원 논문일정 및 제출서류"
                    ),
                    "extension": ".hwp",
                    "parser": "test",
                },
            }
            chunks_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [*rows, distractor]
                ),
                encoding="utf-8",
            )
            build_index(chunks_path, index_path, batch_size=30)
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "학위논문심사.hwp",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((18, 22), start=1)
            ]
            selected.append(
                {
                    **distractor,
                    "source_title": distractor["metadata"]["source_title"],
                    "file_name": "교육대학원논문일정.hwp",
                    "institution": "부산대학교",
                    "preview": distractor["text"],
                    "retrieval": {"final_rank": 3},
                }
            )

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "이번 2025학년도 후기 학위논문 심사 서류는 "
                    "언제까지 내야 하고 심사료는 얼마예요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertIn("thesis#0001", [row["chunk_id"] for row in completed])
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(
            diagnostics["marginal_replacements"][0]["added_facets"],
            ["amount", "date"],
        )

    def test_tuned_multifacet_completes_payment_destination_faq(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "학자금대출 이자지원 장학금 Q&A",
                "신청자격 관련 일반 안내입니다.",
                "질문 목차와 문서 이용 안내입니다.",
                "신청 서류 발급 방법 안내입니다.",
                (
                    "학자금 대출이자 지원의 소득분위 제한은 없나요? "
                    "대출이자 지원의 소득분위 제한은 없습니다."
                ),
                "지원 금액 관련 안내입니다.",
                (
                    "대출이자 지원액 입금은 어떻게 이루어지나요? "
                    "이자 지원액은 한국장학재단에 개설된 개인별 "
                    "원리금 상환계좌로 입금되어 상환됩니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"interest#000{index}",
                    "doc_id": "interest",
                    "document_id": "interest",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/이자지원Q&A.pdf",
                        "relative_path": "부산대학교/이자지원Q&A.pdf",
                        "file_name": "이자지원Q&A.pdf",
                        "source_title": "학자금대출 이자지원 Q&A",
                        "extension": ".pdf",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "이자지원Q&A.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((4, 2), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "학자금대출 이자지원 장학금은 소득분위 제한이 "
                    "있나요? 지원금은 제 통장으로 들어오는 건가요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["interest#0004", "interest#0006"],
        )
        self.assertEqual(
            diagnostics["required_query_facets"],
            ["destination", "eligibility"],
        )
        replacement = diagnostics["marginal_replacements"][0]
        self.assertEqual(replacement["added_facets"], ["destination"])

    def test_tuned_multifacet_rejects_conflicting_scope_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "장학사업 소개와 문의처를 안내합니다.",
                "장학사업 관련 일반 유의사항입니다.",
                (
                    "2025학년도 1학기 장학사업 신청기간은 "
                    "2025. 1. 6.부터 1. 31.까지입니다."
                ),
                "장학사업 신청 서식 내려받기 안내입니다.",
                "장학사업 지원 금액은 1인당 최대 1,000,000원입니다.",
                "장학사업 담당 부서 연락처 안내입니다.",
                "장학사업 지원액은 심사 결과에 따라 달라집니다.",
            ]
            rows = [
                {
                    "chunk_id": f"scope#000{index}",
                    "doc_id": "scope",
                    "document_id": "scope",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/scope.pdf",
                        "relative_path": "부산대학교/scope.pdf",
                        "file_name": "scope.pdf",
                        "source_title": "장학사업 신청 안내",
                        "extension": ".pdf",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "scope.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((4, 6), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "2026학년도 2학기 장학사업 신청 기간과 지원 금액을 "
                    "알려주세요."
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["scope#0004", "scope#0006"],
        )
        self.assertEqual(diagnostics["marginal_replacement_count"], 0)
        self.assertEqual(diagnostics["scope_conflict_skip_count"], 1)
        self.assertEqual(
            diagnostics["scope_conflict_skips"][0]["chunk_id"],
            "scope#0002",
        )

    def test_tuned_multifacet_replaces_redundant_method_with_eligibility(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "장학금 소개와 문의처를 안내합니다.",
                "장학금 관련 일반 유의사항입니다.",
                (
                    "신청 자격은 부산대학교 재학생이며, "
                    "재학생을 선발 대상으로 합니다."
                ),
                "장학금 신청 서식 내려받기 안내입니다.",
                (
                    "신청 방법은 학생지원시스템에 로그인하여 "
                    "온라인 신청하는 것입니다."
                ),
                "장학금 담당 부서 연락처 안내입니다.",
                "접수 방법은 학생지원시스템에서 신청서를 제출하는 것입니다.",
            ]
            rows = [
                {
                    "chunk_id": f"method#000{index}",
                    "doc_id": "method",
                    "document_id": "method",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/method.pdf",
                        "relative_path": "부산대학교/method.pdf",
                        "file_name": "method.pdf",
                        "source_title": "장학금 신청 자격 및 방법 안내",
                        "extension": ".pdf",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "method.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((4, 6), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "장학금 신청 자격과 신청 방법을 알려주세요.",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["method#0004", "method#0002"],
        )
        self.assertEqual(
            diagnostics["required_query_facets"],
            ["eligibility", "method"],
        )
        replacement = diagnostics["marginal_replacements"][0]
        self.assertEqual(replacement["removed_chunk_id"], "method#0006")
        self.assertEqual(replacement["added_facets"], ["eligibility"])

    def test_tuned_multifacet_records_when_search_seed_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "장학사업 소개와 문의처를 안내합니다.",
                "장학사업 관련 일반 유의사항입니다.",
                (
                    "2026학년도 2학기 장학사업 신청기간은 "
                    "2026. 7. 6.부터 8. 31.까지입니다."
                ),
                "장학사업 신청 서식 내려받기 안내입니다.",
                "2026학년도 2학기 장학사업 신청 및 지원 안내입니다.",
                "장학사업 담당 부서 연락처 안내입니다.",
                "장학사업 지원 금액은 1인당 최대 1,000,000원입니다.",
            ]
            rows = [
                {
                    "chunk_id": f"seed#000{index}",
                    "doc_id": "seed",
                    "document_id": "seed",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/seed.pdf",
                        "relative_path": "부산대학교/seed.pdf",
                        "file_name": "seed.pdf",
                        "source_title": (
                            "2026학년도 2학기 장학사업 신청 기간 및 "
                            "지원 금액 안내"
                        ),
                        "extension": ".pdf",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "seed.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((4, 6), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "2026학년도 2학기 장학사업 신청 기간과 지원 금액을 알려주세요.",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["seed#0002", "seed#0006"],
        )
        replacement = diagnostics["marginal_replacements"][0]
        self.assertEqual(replacement["seed_chunk_id"], "seed#0004")
        self.assertEqual(replacement["removed_chunk_id"], "seed#0004")
        self.assertTrue(replacement["seed_replaced"])
        expansion = completed[0]["retrieval"]["context_expansion"]
        self.assertEqual(expansion["seed_chunk_id"], "seed#0004")
        self.assertTrue(expansion["seed_replaced"])

    def test_tuned_single_facet_keeps_existing_completion_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "등록금 납부 관련 일반 안내입니다.",
                "등록금 납부 문의처 안내입니다.",
                "등록금 납부기간은 2026. 8. 24.부터 8. 27.까지입니다.",
                "등록금 고지서 출력 안내입니다.",
                "등록금 수납 은행은 농협과 부산은행입니다.",
            ]
            rows = [
                {
                    "chunk_id": f"single#000{index}",
                    "doc_id": "single",
                    "document_id": "single",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/single.pdf",
                        "relative_path": "부산대학교/single.pdf",
                        "file_name": "single.pdf",
                        "source_title": "2026학년도 등록금 납부 안내",
                        "extension": ".pdf",
                        "parser": "test",
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
            anchor = {
                **rows[4],
                "source_title": rows[4]["metadata"]["source_title"],
                "file_name": "single.pdf",
                "institution": "부산대학교",
                "preview": rows[4]["text"],
                "retrieval": {"final_rank": 1},
            }

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                [anchor],
                "등록금 납부 마감은 언제인가요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )
            _, duration_diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                [anchor],
                "성적 인정은 어떤 절차로 진행되고 시간이 얼마나 걸리나요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["single#0004", "single#0002"],
        )
        self.assertEqual(diagnostics["required_query_facets"], ["date"])
        self.assertFalse(diagnostics["explicit_multifacet"])
        self.assertEqual(diagnostics["marginal_replacement_count"], 0)
        self.assertEqual(duration_diagnostics["required_query_facets"], ["method"])
        self.assertFalse(duration_diagnostics["explicit_multifacet"])

    def test_tuned_single_amount_facet_completes_strong_notice_anchor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                (
                    "2026학년도 교연비 지급 대상과 기본 지급 원칙 안내이며 "
                    "실적급은 상·하반기에 각각 50%씩 지급합니다."
                ),
                "2026학년도 교연비 지급 제외 대상과 계획서 제출 안내입니다.",
                (
                    "교연비 직원과 조교의 개인별 연간 선급금 지급한도액은 "
                    "최대 460만원입니다."
                ),
                (
                    "교원 개인별 연간 지급한도액은 교육 470만원, 연구 "
                    "1,150만원, 학생지도 180만원으로 합계 1,800만원+α입니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"allowance#000{index}",
                    "doc_id": "allowance",
                    "document_id": "allowance",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/allowance.pdf",
                        "relative_path": "부산대학교/allowance.pdf",
                        "file_name": "allowance.pdf",
                        "source_title": (
                            "2026학년도 교육·연구 및 학생지도 비용 지급계획"
                        ),
                        "extension": ".pdf",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "allowance.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((0, 1), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "2026학년도 교연비는 교원 한 명이 연간 최대 얼마인가요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["allowance#0000", "allowance#0003"],
        )
        self.assertTrue(diagnostics["enabled"])
        self.assertEqual(diagnostics["required_query_facets"], ["amount"])
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(
            diagnostics["completions"][0]["kind"],
            "required_facet_sibling_completion",
        )

    def test_tuned_foreign_undergraduate_eligibility_completes_all_subfacets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "2026학년도 학부 외국인 특별전형 모집요강 목차입니다.",
                (
                    "언어능력 기준은 TOPIK 성적 또는 본교 언어교육원 "
                    "수료 기준을 적용하며 지원 자격을 심사합니다."
                ),
                "제출 서류의 번역과 공증에 관한 일반 안내입니다.",
                "전형료 납부와 환불 절차에 관한 일반 안내입니다.",
                (
                    "지원 자격: 지원자와 부모의 외국 국적 요건 및 "
                    "이중국적 제한을 확인합니다. 언어능력은 TOPIK 급수 "
                    "또는 영어시험 점수 기준을 적용합니다. 신입생은 "
                    "고등학교 졸업 이상의 학력 요건을 충족해야 합니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"foreign#000{index}",
                    "doc_id": "foreign",
                    "document_id": "foreign",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/foreign.pdf",
                        "relative_path": "부산대학교/foreign.pdf",
                        "file_name": "foreign.pdf",
                        "source_title": (
                            "2026학년도 전기 학부 외국인 특별전형 모집요강"
                        ),
                        "extension": ".pdf",
                        "parser": "test",
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
            anchor = {
                **rows[1],
                "source_title": rows[1]["metadata"]["source_title"],
                "file_name": "foreign.pdf",
                "institution": "부산대학교",
                "preview": rows[1]["text"],
                "retrieval": {"final_rank": 1},
            }

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                [anchor],
                (
                    "외국인 유학생인데 부산대 학부에 신입학하려면 "
                    "어떤 자격이 필요한가요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["foreign#0001", "foreign#0004"],
        )
        self.assertTrue(diagnostics["foreign_admission_eligibility"])
        self.assertEqual(
            diagnostics["required_query_facets"],
            ["education", "eligibility", "language", "nationality"],
        )
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(
            diagnostics["completions"][0]["added_facets"],
            ["education", "nationality"],
        )

    def test_tuned_exchange_selection_completes_scale_and_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "2027학년도 1학기 해외파견 프로그램 선발 요강",
                "선발 규모는 여러 언어, 국가, 대학의 인원 내외로 구성됩니다.",
                "파견 대학별 세부정보 안내입니다.",
                "전형일정",
                (
                    "온라인 접수 기간: 2026. 7. 17. ~ 7. 26.이며 "
                    "스마트학생정보시스템에서 지원하고 합격자 발표 일정도 "
                    "함께 공지합니다."
                ),
                "지원 자격과 항목별 배점기준 안내입니다.",
                "제출 서류 작성 요령 안내입니다.",
                "파견 유의사항 안내입니다.",
                "장학금 지급 관련 참고사항입니다.",
                "특정 국가 이공계 학생의 별도 지원 자격 안내입니다.",
            ]
            rows = [
                {
                    "chunk_id": f"exchange#000{index}",
                    "doc_id": "exchange",
                    "document_id": "exchange",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/exchange.hwp",
                        "relative_path": "부산대학교/exchange.hwp",
                        "file_name": "exchange.hwp",
                        "source_title": "교환 및 교비 프로그램 1차 선발 요강",
                        "extension": ".hwp",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "exchange.hwp",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((4, 9), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "교환학생 선발 규모와 지원 일정이 어떻게 되나요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["exchange#0004", "exchange#0001"],
        )
        self.assertTrue(diagnostics["exchange_selection"])
        self.assertEqual(
            diagnostics["required_query_facets"],
            ["date", "selection_scale", "selection_schedule"],
        )
        self.assertEqual(
            diagnostics["completions"][0]["added_facets"],
            ["selection_scale"],
        )

    def test_tuned_group_visa_completes_three_same_document_facets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                (
                    "단체접수는 출입국을 대신 방문하여 서류를 제출하는 "
                    "서비스입니다. 사전 예약 없이는 제출이 불가능하며 "
                    "반드시 예약 후 방문해야 합니다."
                ),
                "STEP 2. 체류기간 만료일을 확인한 뒤 신청 회차를 확인합니다.",
                "비자 신청 안내 목차입니다.",
                "접수처 위치 안내입니다.",
                "유학생 지원 안내입니다.",
                "신청 회차 확인 절차입니다.",
                (
                    "회차\t접수기간\t시간\t접수대상\n"
                    "1차\t2026. 8. 3.(월) ~ 5.(수)\t9:20 - 11:00\t"
                    "신입생, 재학생, 수료생"
                ),
                "접수 장소 안내입니다.",
                "신청서 작성 예시입니다.",
                "체류지 입증 서류 안내입니다.",
                "증명서 발급 안내입니다.",
                "정부초청장학생 안내입니다.",
                "여권 사본 안내입니다.",
                "준비 서류 표 머리말입니다.",
                (
                    "② D-2 체류기간 연장\n"
                    "준비 서류\t준비 방법 / 유의 사항\n"
                    "신청서\t사진 불필요\n"
                    "외국인등록증 및 여권 사본\t원본과 사본 준비\n"
                    "수수료(60,000원)\t현금만 가능"
                ),
            ]
            rows = [
                {
                    "chunk_id": f"visa#{index:04d}",
                    "doc_id": "visa",
                    "document_id": "visa",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/visa.pdf",
                        "relative_path": "부산대학교/visa.pdf",
                        "file_name": "visa.pdf",
                        "source_title": "2026학년도 비자 단체접수 안내",
                        "extension": ".pdf",
                        "parser": "test",
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
            build_index(chunks_path, index_path, batch_size=20)
            selected = [
                {
                    **rows[index],
                    "source_title": "2026학년도 비자 단체접수 안내",
                    "file_name": "visa.pdf",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((0, 1), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "D-2 비자 연장을 학교 단체접수로 어떻게 이용하나요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["visa#0000", "visa#0014", "visa#0006"],
        )
        self.assertTrue(diagnostics["group_visa_application"])
        self.assertTrue(diagnostics["per_document_cap_exception"])
        self.assertEqual(diagnostics["configured_per_document_cap"], 2)
        self.assertEqual(diagnostics["effective_per_document_cap"], 3)
        self.assertEqual(diagnostics["completed_count"], 2)
        self.assertEqual(
            {
                facet
                for action in diagnostics["completions"]
                for facet in action["added_facets"]
            },
            {"group_visa_schedule", "group_visa_fee"},
        )

    def test_tuned_assistive_copay_completes_three_document_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                f"관련 사업 안내의 일반 내용 {index}입니다. 충분한 길이의 설명입니다."
                for index in range(11)
            ]
            texts[2] = (
                "정보통신보조기기 보급 활성화 사업 최종 선정 장애학생을 "
                "대상으로 자부담금(개인부담금)을 전액 지원합니다."
            )
            texts[7] = (
                "제출서류는 보급결정 통지문, 재학증명서, 자부담금 납부 "
                "증빙자료, 통장사본입니다. 제출방법은 구글폼 접수 페이지에 "
                "첨부 후 제출하는 방식입니다."
            )
            texts[10] = (
                "정보통신보조기기 선정자가 자부담금을 먼저 납부한 뒤 "
                "납부 증빙자료 확인을 거쳐 지원하며 기기 수령확인서는 "
                "지원 후 요청합니다."
            )
            rows = [
                {
                    "chunk_id": f"assistive#{index:04d}",
                    "doc_id": "assistive",
                    "document_id": "assistive",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/자부담금지원.hwpx",
                        "relative_path": "부산대학교/자부담금지원.hwpx",
                        "file_name": "자부담금지원.hwpx",
                        "source_title": (
                            "정보통신보조기기 보급 자부담금 지원사업 신청 안내문"
                        ),
                        "extension": ".hwpx",
                        "parser": "test",
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
            build_index(chunks_path, index_path, batch_size=20)
            selected = [
                {
                    **rows[index],
                    "source_title": rows[index]["metadata"]["source_title"],
                    "file_name": "자부담금지원.hwpx",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((10, 2), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                (
                    "정보통신보조기기 보급사업에 선정됐는데 "
                    "자부담금을 지원받을 방법이 있나요?"
                ),
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["assistive#0010", "assistive#0007", "assistive#0002"],
        )
        self.assertTrue(diagnostics["assistive_device_copay_support"])
        self.assertEqual(
            diagnostics["required_query_facets"],
            [
                "assistive_copay_documents",
                "assistive_copay_procedure",
                "assistive_copay_summary",
            ],
        )
        self.assertEqual(diagnostics["completed_count"], 1)
        self.assertEqual(diagnostics["configured_per_document_cap"], 2)
        self.assertEqual(diagnostics["effective_per_document_cap"], 3)
        self.assertTrue(diagnostics["per_document_cap_exception"])

    def test_generic_device_support_keeps_configured_document_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            text = "일반 학습보조기기 지원 안내이며 신청 방법을 설명합니다."
            row = {
                "chunk_id": "generic#0000",
                "doc_id": "generic",
                "document_id": "generic",
                "chunk_index": 0,
                "text": text,
                "char_count": len(text),
                "metadata": {
                    "institution": "부산대학교",
                    "source_path": "부산대학교/일반지원.pdf",
                    "relative_path": "부산대학교/일반지원.pdf",
                    "file_name": "일반지원.pdf",
                    "source_title": "일반 학습보조기기 지원 안내",
                    "extension": ".pdf",
                    "parser": "test",
                },
            }
            chunks_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            build_index(chunks_path, index_path, batch_size=10)
            selected = [{
                **row,
                "source_title": row["metadata"]["source_title"],
                "file_name": "일반지원.pdf",
                "institution": "부산대학교",
                "preview": text,
                "retrieval": {"final_rank": 1},
            }]

            _, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "학습보조기기 지원을 신청하는 방법이 있나요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertFalse(diagnostics["assistive_device_copay_support"])
        self.assertEqual(diagnostics["effective_per_document_cap"], 2)
        self.assertFalse(diagnostics["per_document_cap_exception"])

    def test_tuned_third_party_report_completes_identity_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks_path = root / "chunks.jsonl"
            index_path = root / "bm25.sqlite"
            texts = [
                "센터이용 Q&A",
                "인권센터 이용안내와 사건신고 처리절차 메뉴입니다.",
                "상담 신청방법 안내입니다.",
                "센터이용 Q&A",
                (
                    "신고는 피해자만 할 수 있나요? 피해자가 아닌 제3자도 "
                    "신고할 수 있습니다. 피해자 본인의 의사를 확인하여 "
                    "상담·조사 여부를 결정합니다."
                ),
                "대학 구성원 여부에 따른 사건처리 제약 안내입니다.",
                "다른 대학 소속 가해자에 관한 신고 기관 안내입니다.",
                (
                    "익명 신고가 가능한가요? 사건처리를 위해서는 피해자 "
                    "본인의 인적사항을 인권센터에 알려주어야 합니다."
                ),
            ]
            rows = [
                {
                    "chunk_id": f"rights#000{index}",
                    "doc_id": "rights",
                    "document_id": "rights",
                    "chunk_index": index,
                    "text": text,
                    "char_count": len(text),
                    "metadata": {
                        "institution": "부산대학교",
                        "source_path": "부산대학교/rights.html",
                        "relative_path": "부산대학교/rights.html",
                        "file_name": "rights.html",
                        "source_title": "센터이용 Q&A",
                        "extension": ".html",
                        "parser": "test",
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
            selected = [
                {
                    **rows[index],
                    "source_title": "센터이용 Q&A",
                    "file_name": "rights.html",
                    "institution": "부산대학교",
                    "preview": rows[index]["text"],
                    "retrieval": {"final_rank": rank},
                }
                for rank, index in enumerate((6, 4), start=1)
            ]

            completed, diagnostics = replace_with_adjacent_temporal_contexts(
                index_path,
                selected,
                "피해자가 아닌 제가 대신 신고해도 되나요?",
                facet_completion=True,
                max_chunks_per_document=2,
            )

        self.assertEqual(
            [row["chunk_id"] for row in completed],
            ["rights#0007", "rights#0004"],
        )
        self.assertTrue(diagnostics["third_party_reporting"])
        self.assertEqual(
            diagnostics["completions"][0]["added_facets"],
            ["victim_identification"],
        )

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

    def test_claim_citation_excerpt_centers_the_supporting_passage(self) -> None:
        evidence = "신청 마감은 2026년 8월 31일 18시까지입니다."
        claim = attribute_claim(
            evidence,
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": ("무관한 안내 문구 " * 180) + evidence,
                    "locations": [{"block_id": "block-9", "page": 3}],
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertIn(evidence, claim["citations"][0]["excerpt"])
        self.assertGreater(claim["citations"][0]["excerpt_start"], 0)
        self.assertEqual(claim["citations"][0]["block_id"], "block-9")
        self.assertNotIn("locations", claim["citations"][0])

    def test_rag_response_top_level_citations_are_claim_specific(self) -> None:
        response = build_rag_response(
            "신청 마감은?",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "신청 마감은 2026년 8월 31일입니다.",
                    "locations": [{"block_id": "block-1", "page": 2}],
                }
            ],
            "신청 마감은 2026년 8월 31일입니다.",
            "gemini:test",
        )

        self.assertEqual(response["citations"][0]["claim_index"], 0)
        self.assertEqual(response["citations"][0]["block_id"], "block-1")
        self.assertIn(
            "2026년 8월 31일", response["citations"][0]["excerpt"]
        )

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

    def test_claim_attribution_rejects_reversed_permission(self) -> None:
        denied = attribute_claim(
            "등록금을 분할 납부할 수 없습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "등록금은 분할 납부할 수 있습니다.",
                }
            ],
        )

        self.assertFalse(denied["supported"])
        self.assertEqual(
            denied["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_requires_explicit_permission_evidence(self) -> None:
        claim = attribute_claim(
            "온라인 신청할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "온라인 신청 절차를 확인했습니다.",
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_binds_direction_value_after_relation(self) -> None:
        claim = attribute_claim(
            "학부 등록금 인상률은 3.95%입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "학부 등록금 인상률은 5%입니다.\n"
                        "대학원 등록금 인상률은 3.95%입니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_accepts_direction_value_after_relation(self) -> None:
        claim = attribute_claim(
            "학부 등록금은 3.95% 인상되었습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "학부 등록금은 인상되었습니다(3.95%).",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_preserves_tab_separated_relation_rows(self) -> None:
        claim = attribute_claim(
            "대학원 등록금은 3.95% 인상되었습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "학부 등록금은 3.95% 인상\t"
                        "대학원 등록금은 5% 인상"
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_splits_contrastive_permission_clause(self) -> None:
        claim = attribute_claim(
            "학부생은 신청이 불가합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "학부생은 신청이 가능하나 "
                        "대학원생은 신청이 불가합니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_extracts_each_permission_in_compound_clause(
        self,
    ) -> None:
        denied_application = attribute_claim(
            "학부생은 온라인으로 신청할 수 없습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "학부생은 온라인으로 신청할 수 있고 "
                        "수수료는 현금으로 낼 수 없습니다."
                    ),
                }
            ],
        )
        allowed_fee = attribute_claim(
            "수수료는 면제받을 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0002",
                    "text": (
                        "학부생은 신청할 수 없고 "
                        "수수료는 면제받을 수 있습니다."
                    ),
                }
            ],
        )

        self.assertFalse(denied_application["supported"])
        self.assertEqual(
            denied_application["validation_reason"],
            "semantic_relation_mismatch",
        )
        self.assertTrue(allowed_fee["supported"])

    def test_claim_attribution_rejects_exclusive_permission_subject(self) -> None:
        claim = attribute_claim(
            "학부 등록금은 분할 납부할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "대학원 등록금은 분할 납부할 수 있습니다.",
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_rejects_reversed_comparator(self) -> None:
        claim = attribute_claim(
            "직전 학기 성적은 80점 이하이어야 합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "직전 학기 성적은 80점 이상이어야 합니다.",
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_does_not_swap_subject_value_rows(self) -> None:
        claim = attribute_claim(
            "학부 등록금은 3.95% 인상되고 대학원 등록금은 동결됩니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "학부 등록금은 동결됩니다.\n"
                        "대학원 등록금은 3.95% 인상됩니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_accepts_matching_semantic_relations(self) -> None:
        cases = [
            (
                "등록금을 분할 납부할 수 있습니다.",
                "등록금은 분할 납부할 수 있습니다.",
            ),
            (
                "등록금을 분할 납부할 수 없습니다.",
                "등록금은 분할 납부할 수 없습니다.",
            ),
            (
                "직전 학기 성적은 80점 이상이어야 합니다.",
                "직전 학기 성적은 80점 이상이어야 합니다.",
            ),
            (
                "학부 등록금은 동결되고 대학원 등록금은 "
                "3.95% 인상됩니다.",
                "학부 등록금은 동결됩니다.\n"
                "대학원 등록금은 3.95% 인상됩니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(claim=claim_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertTrue(claim["supported"])
                self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_accepts_coordinated_frozen_subjects(self) -> None:
        claim = attribute_claim(
            "2026학년도 학부 및 대학원 등록금은 모두 동결되었습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "2026학년도 등록금 심의 결과\n"
                        "학부 동결\n"
                        "대학원 동결"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_requires_every_coordinated_direction_subject(
        self,
    ) -> None:
        claim_text = (
            "2026학년도 학부 및 대학원 등록금은 모두 동결되었습니다."
        )
        source_cases = [
            "2026학년도 등록금 심의 결과\n학부 동결",
            (
                "2026학년도 등록금 심의 결과\n"
                "학부 동결\n"
                "대학원 3.95% 인상"
            ),
        ]
        for source_text in source_cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

    def test_claim_attribution_splits_past_tense_direction_connective(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "2025학년도 부산대학교 학부 등록금은 동결로 책정되었고 "
                "대학원 등록금은 3.95% 인상으로 책정되었습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "source_title": "2025학년도 등록금 심의 결과",
                    "text": (
                        "학부 동결\n"
                        "대학원 3.95% 인상"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_ignores_generic_direction_object_when_scoped(
        self,
    ) -> None:
        claim = attribute_claim(
            "경영대학원 등록금은 5.49% 인상되었습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "경영대학원 5.49% 인상",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_recovers_subject_value_from_same_table_row(
        self,
    ) -> None:
        source_text = (
            "사업명\t분야명\t대상\t지원액\t비고\n"
            "디딤돌장학사업\t학자금대출 원금상환\t재학생\t"
            "1인당 최대 1,000,000원\t예산 범위 내\n"
            "디딤돌장학사업\t학자금대출 이자지원\t재학생\t"
            "1인당 최대 500,000원\t예산 범위 내"
        )
        supported = attribute_claim(
            (
                "학자금대출 원금상환 분야는 1인당 최대 1,000,000원을 "
                "예산 범위 내에서 지원합니다."
            ),
            [{"source_number": 1, "chunk_id": "notice#0001", "text": source_text}],
        )
        swapped = attribute_claim(
            "학자금대출 원금상환 지원액은 1인당 최대 500,000원입니다.",
            [{"source_number": 1, "chunk_id": "notice#0001", "text": source_text}],
        )

        self.assertTrue(supported["supported"])
        self.assertEqual(supported["validation_reason"], "supported")
        self.assertFalse(swapped["supported"])
        self.assertEqual(
            swapped["validation_reason"],
            "semantic_relation_mismatch",
        )

    def test_claim_attribution_combines_same_row_deadline_date_and_time(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "디딤돌장학사업 신청 기간은 2026년 7월 6일부터 "
                "2026년 8월 31일 18:00까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "source_title": "2026년도 디딤돌장학사업 신청 공고",
                    "text": (
                        "(신청기간) 2026. 7. 6.(월) ~ "
                        "8. 31.(월), 18:00까지"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_uses_location_section_scope_for_dates(self) -> None:
        """A generic web-page title must not hide the notice semester scope."""

        claim = attribute_claim(
            "이번 2학기 등록금 본등록은 2026. 8. 24.(월)부터 "
            "8. 27.(목)까지 납부해야 합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "source_title": "공지사항 내용 > 부산대학교",
                    "locations": [
                        {
                            "section_path": [
                                "공지사항",
                                "2026학년도 2학기 재학생 등록금 납부 안내",
                            ]
                        }
                    ],
                    "text": (
                        "이번 재학생 등록금 본등록은 2026. 8. 24.(월) ~ "
                        "8. 27.(목) 납부 일정입니다."
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_scopes_multiple_comparators_by_value(self) -> None:
        results = [
            {
                "source_number": 1,
                "chunk_id": "notice#0001",
                "text": "성적은 80점 이상 90점 이하이어야 합니다.",
            }
        ]

        self.assertTrue(attribute_claim("성적은 80점 이상입니다.", results)["supported"])
        self.assertTrue(attribute_claim("성적은 90점 이하입니다.", results)["supported"])

    def test_claim_attribution_binds_postfix_comparator_subject(self) -> None:
        claim = attribute_claim(
            "80점 이상인 학부생을 선발합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "80점 이상인 대학원생을 선발합니다. "
                        "학부생은 별도 심사로 선발합니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_inherits_omitted_comparator_subject(self) -> None:
        claim = attribute_claim(
            "성적은 90점 이하입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "성적은 80점 이상이며 90점 이하입니다.",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_accepts_complemented_comparator(self) -> None:
        claim = attribute_claim(
            "지원 성적은 80점 미만이어야 합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "지원 성적은 80점 이상이 아니어야 합니다.",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_inherits_permission_subject_across_connective(
        self,
    ) -> None:
        claim = attribute_claim(
            "수료후연구생은 신청이 가능합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "대상은 수료후연구생이며 신청이 가능합니다.",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_does_not_misread_relation_substrings(self) -> None:
        claim = attribute_claim(
            "점검 결과 이상 없음으로 처리되었습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "점검 결과 이상 없음으로 처리되었습니다.",
                }
            ],
        )
        unavoidable = attribute_claim(
            "불가피한 사유가 있으면 증빙을 제출합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0002",
                    "text": "불가피한 사유가 있으면 증빙을 제출합니다.",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertTrue(unavoidable["supported"])

    def test_claim_attribution_uses_later_compatible_source(self) -> None:
        claim = attribute_claim(
            "등록금을 분할 납부할 수 없습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "wrong#0001",
                    "text": "등록금은 분할 납부할 수 있습니다.",
                },
                {
                    "source_number": 2,
                    "chunk_id": "right#0001",
                    "text": "등록금은 분할 납부할 수 없습니다.",
                },
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["source_ids"], ["right#0001"])

    def test_claim_attribution_binds_permission_to_academic_year(self) -> None:
        claim = attribute_claim(
            "2026학년도에는 등록금을 분할 납부할 수 없습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "2025학년도에는 등록금을 분할 납부할 수 없습니다.\n"
                        "2026학년도에는 등록금을 분할 납부할 수 있습니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_uses_title_scope_for_permission(self) -> None:
        matching = attribute_claim(
            "2026학년도에는 등록금을 분할 납부할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "source_title": "2026학년도 등록금 안내",
                    "text": "등록금은 분할 납부할 수 있습니다.",
                }
            ],
        )
        reversed_claim = attribute_claim(
            "2026학년도에는 등록금을 분할 납부할 수 없습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "source_title": "2026학년도 등록금 안내",
                    "text": "등록금은 분할 납부할 수 있습니다.",
                }
            ],
        )

        self.assertTrue(matching["supported"])
        self.assertFalse(reversed_claim["supported"])
        self.assertEqual(
            reversed_claim["validation_reason"],
            "semantic_relation_mismatch",
        )

    def test_claim_attribution_uses_title_scope_for_other_relations(self) -> None:
        cases = [
            (
                "2026학년도 성적은 80점 이상이어야 합니다.",
                "2026학년도 장학금 안내",
                "성적은 80점 이상이어야 합니다.",
            ),
            (
                "2026학년도 학부 등록금은 3.95% 인상되었습니다.",
                "2026학년도 등록금 안내",
                "학부 등록금은 3.95% 인상되었습니다.",
            ),
        ]
        for claim_text, title, source_text in cases:
            with self.subTest(claim=claim_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "source_title": title,
                            "text": source_text,
                        }
                    ],
                )
                self.assertTrue(claim["supported"])
                self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_uses_title_round_scope(self) -> None:
        claim = attribute_claim(
            "2026학년도 1차에는 등록금을 분할 납부할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "source_title": "2026학년도 1차 등록금 안내",
                    "text": "등록금은 분할 납부할 수 있습니다.",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_binds_calendar_year(self) -> None:
        claim = attribute_claim(
            "2026년에는 등록금을 분할 납부할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "2025년에는 등록금을 분할 납부할 수 있습니다.",
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "critical_value_mismatch"
        )

    def test_claim_attribution_rejects_unscoped_ambiguous_permission(self) -> None:
        claim = attribute_claim(
            "등록금은 분할 납부할 수 없습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "2025학년도 등록금은 분할 납부할 수 없습니다.\n"
                        "2026학년도 등록금은 분할 납부할 수 있습니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_binds_comparator_to_academic_year(self) -> None:
        claim = attribute_claim(
            "2026학년도 성적은 80점 이상이어야 합니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "2025학년도 성적은 80점 이상이어야 합니다.\n"
                        "2026학년도 성적은 80점 이하이어야 합니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_binds_direction_value_to_subject(self) -> None:
        claim = attribute_claim(
            "대학원 등록금은 3.95% 인상되었습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "학부 등록금은 3.95% 인상되었습니다. "
                        "대학원 등록금은 5% 인상되었습니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_splits_parallel_direction_rows(self) -> None:
        for separator in (" / ", " 및 "):
            with self.subTest(separator=separator):
                claim = attribute_claim(
                    "대학원 등록금은 3.95% 인상됩니다.",
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": (
                                "학부 등록금은 3.95% 인상"
                                f"{separator}"
                                "대학원 등록금은 5% 인상"
                            ),
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

    def test_claim_attribution_normalizes_decrease_synonyms(self) -> None:
        claim = attribute_claim(
            "등록금은 3.95% 인하되었습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "등록금은 3.95% 감소했습니다.",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_understands_relation_negation(self) -> None:
        cases = [
            (
                "등록금은 인상되었습니다.",
                "등록금은 인상되지 않았습니다.",
                False,
            ),
            (
                "성적은 80점 이상이어야 합니다.",
                "성적은 80점 이상이 아니어야 합니다.",
                False,
            ),
            (
                "분할 납부는 허용됩니다.",
                "분할 납부는 금지되지 않습니다.",
                True,
            ),
        ]
        for claim_text, source_text, expected in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertEqual(claim["supported"], expected)

    def test_claim_attribution_accepts_explicit_assistive_copay_procedure(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "정보통신보조기기 보급사업 선정자는 자부담금을 먼저 납부한 "
                "뒤 납부 증빙자료 확인을 거쳐 지원받을 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "assistive#0010",
                    "source_title": "정보통신보조기기 자부담금 지원 안내",
                    "text": (
                        "본 사업은 선정자가 자부담금을 납부한 뒤, 납부 "
                        "증빙자료 확인을 거쳐 자부담금을 지원하는 방식으로 "
                        "진행됩니다."
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_rejects_negated_assistive_copay_procedure(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "정보통신보조기기 보급사업 선정자는 자부담금을 먼저 납부한 "
                "뒤 납부 증빙자료 확인을 거쳐 지원받을 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "assistive#0010",
                    "source_title": "정보통신보조기기 자부담금 지원 안내",
                    "text": (
                        "선정자가 자부담금을 납부한 뒤 납부 증빙자료를 "
                        "확인하지만 자부담금은 지원하지 않습니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_rejects_reversed_assistive_copay_procedure(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "정보통신보조기기 보급사업 선정자는 자부담금을 먼저 납부한 "
                "뒤 납부 증빙자료 확인을 거쳐 지원받을 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "assistive#0010",
                    "source_title": "정보통신보조기기 자부담금 지원 안내",
                    "text": (
                        "선정자에게 자부담금을 먼저 지원한 뒤 납부와 "
                        "증빙자료 확인을 진행합니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_accepts_group_visa_schedule_time_notation(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "1차 접수기간은 2026년 8월 3일부터 5일까지이며, 시간은 "
                "9시 20분부터 11시까지와 14시부터 16시까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "visa#0006",
                    "source_title": "2026학년도 2학기 D-2 비자 단체접수 안내",
                    "text": (
                        "회차\t접수기간\t시간\t접수대상\t"
                        "접수 제외 대상 (체류기간 만료일)\n"
                        "1차\t2026. 8. 3.(월) ~ 5.(수)\t"
                        "9:20 - 11:00, 14:00 - 16:00\t"
                        "제한없음(신입생, 재학생, 수료생) ※ 단, 2026년 "
                        "8월 21일 이후 졸업(수료)자 접수 불가\t"
                        "2026년 8월 7일 이전 인 자\n"
                        "2차\t2026. 8. 21.(금), 8. 25.(화)\t"
                        "9:20 - 11:00, 14:00 - 16:00\t"
                        "제한없음(신입생, 재학생, 수료생)\t"
                        "2026년 8월 27일 이전 인 자\n"
                        "3차\t2026. 9. 21.(월) ~ 22.(화)\t"
                        "9:20 - 11:00, 14:00 - 16:00\t"
                        "재학생, 수료생\t2026년 9월 28일 이전 인 자"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["validation_reason"], "supported")

    def test_claim_attribution_rejects_other_event_with_same_schedule(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "1차 접수기간은 2026년 8월 3일부터 5일까지이며, 시간은 "
                "9시 20분부터 11시까지와 14시부터 16시까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "interview#0006",
                    "source_title": "2026학년도 면접 안내",
                    "text": (
                        "회차\t면접기간\t면접시간\n"
                        "1차\t2026. 8. 3.(월) ~ 5.(수)\t"
                        "9:20 - 11:00, 14:00 - 16:00"
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])

    def test_claim_attribution_accepts_group_visa_round_audiences(
        self,
    ) -> None:
        claim_text = (
            "1차와 2차 접수대상은 신입생, 재학생, 수료생을 포함해 제한이 "
            "없으며, 3차 접수대상은 재학생과 수료생입니다."
        )
        source = {
            "source_number": 1,
            "chunk_id": "visa#0006",
            "source_title": "2026학년도 비자 단체접수 안내",
            "text": (
                "1차\t제한없음(신입생, 재학생, 수료생)\n"
                "2차\t제한없음(신입생, 재학생, 수료생)\n"
                "3차\t재학생, 수료생"
            ),
        }

        supported = attribute_claim(claim_text, [source])
        wrong_third_round = attribute_claim(
            claim_text,
            [
                {
                    **source,
                    "text": (
                        "1차\t제한없음(신입생, 재학생, 수료생)\n"
                        "2차\t제한없음(신입생, 재학생, 수료생)\n"
                        "3차\t신입생, 재학생, 수료생"
                    ),
                }
            ],
        )

        self.assertTrue(supported["supported"])
        self.assertFalse(wrong_third_round["supported"])

    def test_claim_attribution_accepts_group_visa_reservation_service(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "부산대학교는 출입국을 대신 방문하여 서류를 제출해 주는 "
                "D-2 비자 단체접수 서비스를 제공하며 사전 예약 후 이용할 "
                "수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "visa#0000",
                    "source_title": "2026학년도 D-2 비자 단체접수 안내",
                    "text": (
                        "단체접수는 출입국을 대신 방문하여 서류를 제출해 주는 "
                        "서비스이며, 개별접수와 제출서류는 동일합니다.\n"
                        "사전 예약 없이 서류 제출은 불가능 합니다. 반드시 "
                        "예약 후 방문해야 합니다."
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])

    def test_claim_attribution_rejects_ended_group_visa_service(self) -> None:
        claim = attribute_claim(
            (
                "부산대학교는 출입국을 대신 방문하여 서류를 제출해 주는 "
                "D-2 비자 단체접수 서비스를 제공하며 사전 예약 후 이용할 "
                "수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "visa#0000",
                    "source_title": "2026학년도 D-2 비자 단체접수 안내",
                    "text": (
                        "단체접수는 출입국을 대신 방문하여 서류를 제출해 주는 "
                        "서비스였으나 현재는 서비스를 중단했습니다.\n"
                        "사전 예약 없이 서류 제출은 불가능 합니다. 반드시 "
                        "예약 후 방문해야 합니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_accepts_group_visa_cash_fee_and_waiver(
        self,
    ) -> None:
        claim = attribute_claim(
            (
                "수수료는 60,000원이며 현금(만원권 이상)만 가능하고 "
                "정부초청장학생은 장학증서 제출 시 면제됩니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "visa#0014",
                    "source_title": "2026학년도 D-2 비자 단체접수 안내",
                    "text": (
                        "수수료(60,000원) - 현금만 가능 - 만원권 이상으로 "
                        "준비 - 정부초청장학생은 장학증서 제출시 면제"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])

    def test_claim_attribution_rejects_wrong_group_visa_cash_rule(self) -> None:
        claim = attribute_claim(
            (
                "수수료는 60,000원이며 현금(만원권 이상)만 가능하고 "
                "정부초청장학생은 장학증서 제출 시 면제됩니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "visa#0014",
                    "source_title": "2026학년도 D-2 비자 단체접수 안내",
                    "text": (
                        "수수료(60,000원) - 현금 사용 불가 - 만원권 미만으로 "
                        "준비 - 정부초청장학생은 장학증서 제출시 면제"
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_does_not_treat_negated_direction_as_entailment(
        self,
    ) -> None:
        cases = [
            (
                "2026학년도 학부 등록금은 동결됩니다.",
                "2026학년도 학부 등록금은 인상되지 않았습니다.",
            ),
            (
                "학부 등록금은 인상됩니다.",
                "학부 등록금은 인하되지 않았습니다.",
            ),
            (
                "학부 등록금은 인하됩니다.",
                "학부 등록금은 인상되지 않았습니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

    def test_claim_attribution_handles_negation_variants_and_modality(self) -> None:
        cases = [
            (
                "2026학년도 학부 등록금은 증가했습니다.",
                "2026학년도 학부 등록금은 증가하지는 않았습니다.",
            ),
            (
                "분할 납부는 금지됩니다.",
                "분할 납부는 금지된 것은 아닙니다.",
            ),
            (
                "지원 성적은 80점 미만이어야 합니다.",
                "지원 성적이 80점 이상이 아니면 탈락합니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

    def test_claim_attribution_rejects_unasserted_relation_evidence(self) -> None:
        cases = [
            (
                "분할 납부는 허용됩니다.",
                "분할 납부 허용 여부를 검토합니다.",
            ),
            (
                "분할 납부는 허용됩니다.",
                "분할 납부는 허용될 예정입니다.",
            ),
            (
                "등록금은 인상되었습니다.",
                "등록금 인상 여부를 검토하고 있습니다.",
            ),
            (
                "등록금은 인상되었습니다.",
                "등록금은 인상될 예정입니다.",
            ),
            (
                "등록금은 인상됩니다.",
                "등록금은 인상될 수도 있습니다.",
            ),
            (
                "지원 성적은 80점 이상이어야 합니다.",
                "지원 성적은 80점 이상일 필요가 없습니다.",
            ),
            (
                "분할 납부는 허용됩니다.",
                "분할 납부가 허용된 것은 아닙니다.",
            ),
            (
                "등록금은 인상될 예정입니다.",
                "등록금 인상 계획은 취소되었습니다.",
            ),
            (
                "신청 마감은 8월 31일까지입니다.",
                "신청 마감은 8월 31일까지가 아닙니다.",
            ),
            (
                "신청 마감은 8월 31일까지입니다.",
                "신청 마감은 8월 31일까지 연장될 예정입니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

        planned = attribute_claim(
            "등록금은 인상될 예정입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0002",
                    "text": "등록금은 인상될 예정입니다.",
                }
            ],
        )
        self.assertTrue(planned["supported"])

    def test_claim_attribution_rejects_conflicting_metadata_scope(self) -> None:
        claim = attribute_claim(
            "2026학년도에는 등록금을 분할 납부할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "source_title": "2026학년도 등록금 안내",
                    "file_name": "2025학년도 등록금 안내.pdf",
                    "text": "등록금은 분할 납부할 수 있습니다.",
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_normalizes_comparator_value_spelling(self) -> None:
        cases = [
            (
                "지원금은 100,000원 이하입니다.",
                "지원금은 10만원 이상입니다.",
            ),
            (
                "지원율은 80% 이상이어야 합니다.",
                "지원율은 80퍼센트 이하이어야 합니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

    def test_claim_attribution_handles_boundary_operator_aliases(self) -> None:
        cases = [
            ("지원금은 최소 100만원입니다.", "지원금은 최대 100만원입니다."),
            ("지원금은 100만원 이내입니다.", "지원금은 100만원 초과입니다."),
            ("지원금은 100만원 미달입니다.", "지원금은 100만원 이상입니다."),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

    def test_claim_attribution_rejects_temporal_boundary_reversal(self) -> None:
        cases = [
            (
                "신청은 8월 31일까지 가능합니다.",
                "신청은 8월 31일부터 가능합니다.",
            ),
            (
                "신청은 8월 31일 이후 가능합니다.",
                "신청은 8월 31일 이전 가능합니다.",
            ),
            (
                "신청 마감은 9월 1일까지입니다.",
                "신청 마감은 9월 1일부터입니다.",
            ),
            (
                "신청 기간은 8/10부터 8/3까지입니다.",
                "신청 기간은 8/3부터 8/10까지입니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

        positive = attribute_claim(
            "신청 기간은 8/3부터 8/10까지입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0002",
                    "text": "신청 기간은 8/3부터 8/10까지입니다.",
                }
            ],
        )
        self.assertTrue(positive["supported"])

    def test_claim_attribution_binds_temporal_boundary_to_subject(self) -> None:
        claim = attribute_claim(
            "학생 신청 기간은 8월 3일부터입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "학과 승인 기간은 8월 3일부터입니다. "
                        "학생 신청 기간은 8월 10일부터입니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(
            claim["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_handles_time_only_boundary(self) -> None:
        reversed_claim = attribute_claim(
            "고지서는 10:00부터 출력할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "고지서는 10:00까지 출력할 수 있습니다.",
                }
            ],
        )
        self.assertFalse(reversed_claim["supported"])
        self.assertEqual(
            reversed_claim["validation_reason"],
            "semantic_relation_mismatch",
        )

    def test_claim_attribution_accepts_categorical_available_bank_list(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "tuition#0004",
            "section_path": ["수납은행"],
            "text": (
                "본등록: 농협・부산・하나・국민・신한・우리은행"
                "(전국지점)"
            ),
        }

        supported = attribute_claim(
            (
                "본등록 시 납부 가능한 은행은 농협, 부산, 하나, 국민, "
                "신한, 우리은행 전국 지점입니다."
            ),
            [source],
        )
        unsupported_permission = attribute_claim(
            "본등록 시 신용카드 납부가 가능합니다.",
            [source],
        )
        location_paraphrase = attribute_claim(
            (
                "본등록 기간에는 농협, 부산, 하나, 국민, 신한, "
                "우리은행 전국 지점에서 납부할 수 있습니다."
            ),
            [source],
        )

        self.assertTrue(supported["supported"])
        self.assertTrue(location_paraphrase["supported"])
        self.assertFalse(unsupported_permission["supported"])

    def test_claim_attribution_inherits_temporal_audience_from_notice_title(self) -> None:
        claim = attribute_claim(
            (
                "2026학년도 2학기 재학생 본등록은 2026년 8월 24일부터 "
                "8월 27일까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0006",
                    "source_title": "2026학년도 2학기 재학생 등록금 납부 안내",
                    "text": (
                        "본등록: 2026. 8. 24.(월) ~ 8. 27.(목) "
                        "*고지서출력: 2026. 8. 24.(월), 10:00\n"
                        "추가등록: 2026. 9. 1.(화) ~ 9. 3.(목)\n"
                        "최종등록: 2026. 9. 21.(월) ~ 9. 23.(수)"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])

    def test_claim_attribution_accepts_named_official_procedure_availability(self) -> None:
        supported = attribute_claim(
            "BIDV SMB 앱을 통해 부산대학교 등록금을 납부할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "bidv#0000",
                    "source_title": "BIDV 등록금 납부 방법(BIDV-How to Use).pdf",
                    "text": (
                        "BIDV SMB APP을 통한 한국 대학 등록금 납부 방법. "
                        "BIDV SMB 앱에 로그인한 뒤 국제송금 메뉴에서 "
                        "한국학비결제를 선택해주세요."
                    ),
                }
            ],
        )
        denied = attribute_claim(
            "BIDV 앱으로 등록금을 납부할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "bidv#0001",
                    "source_title": "BIDV 등록금 납부 방법 안내",
                    "text": "BIDV 모바일 앱에서는 등록금 납부가 불가합니다.",
                }
            ],
        )

        self.assertTrue(supported["supported"])
        self.assertFalse(denied["supported"])

    def test_claim_attribution_accepts_official_ui_lookup_instruction(self) -> None:
        claim_text = (
            "등록금을 이월할 경우 복학 시 별도의 등록 절차 없이 납부 "
            "처리되며 학생지원시스템의 납부확인 메뉴에서 수기분 0원으로 "
            "확인할 수 있습니다."
        )
        source = {
            "source_number": 1,
            "chunk_id": "faq#0001",
            "source_title": "재학생 등록금 관련 자주하는 질문(FAQ) 안내",
            "text": (
                "등록금 납부 후 휴학을 하면 등록금이 이월되나요? "
                "이월됩니다. 복학 시 별도 등록절차 없이 납부처리 됩니다. "
                "학생지원시스템 → 등록 → 납부확인(영수증출력)에서 "
                "'수기분 0원' 확인. 전액 장학금 등록 후 혜택을 원하지 "
                "않을 경우 환불신청 가능합니다."
            ),
        }

        attributed = attribute_claim(claim_text, [source])
        response = build_rag_response(
            "납부한 등록금을 이월하면 복학할 때 어떻게 확인하나요?",
            [source],
            claim_text,
            "frontier:test",
        )

        self.assertTrue(attributed["supported"])
        self.assertEqual(attributed["source_ids"], ["faq#0001"])
        self.assertIn(claim_text, response["answer"])

    def test_claim_attribution_accepts_dated_ui_output_across_adjacent_lines(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "admission#payment",
            "source_title": "2026학년도 수시모집 합격자 유의사항",
            "text": (
                "고지서출력: 2026. 2. 3.(화) 10:00부터\n"
                "학생지원시스템 로그인 → 등록 → 고지서출력 → 고지서"
            ),
        }
        supported = attribute_claim(
            (
                "등록금 고지서는 2026년 2월 3일 10:00부터 "
                "학생지원시스템에 로그인하여 직접 출력할 수 있습니다."
            ),
            [source],
        )
        wrong_date = attribute_claim(
            (
                "등록금 고지서는 2026년 2월 4일 10:00부터 "
                "학생지원시스템에 로그인하여 직접 출력할 수 있습니다."
            ),
            [source],
        )
        wrong_object = attribute_claim(
            (
                "등록금 영수증은 2026년 2월 3일 10:00부터 "
                "학생지원시스템에 로그인하여 직접 출력할 수 있습니다."
            ),
            [source],
        )

        self.assertTrue(supported["supported"])
        self.assertFalse(wrong_date["supported"])
        self.assertFalse(wrong_object["supported"])

    def test_claim_attribution_rejects_unavailable_official_ui_lookup(self) -> None:
        attributed = attribute_claim(
            (
                "학생지원시스템의 납부확인 메뉴에서 수기분 0원으로 "
                "확인할 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "faq#unavailable",
                    "source_title": "재학생 등록금 관련 자주하는 질문(FAQ) 안내",
                    "text": (
                        "학생지원시스템 → 등록 → 납부확인 메뉴에서 "
                        "수기분 0원 확인 불가"
                    ),
                }
            ],
        )

        self.assertFalse(attributed["supported"])
        self.assertEqual(
            attributed["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_rejects_unrelated_ui_lookup_instruction(self) -> None:
        attributed = attribute_claim(
            (
                "학생지원시스템의 납부확인 메뉴에서 수기분 0원으로 "
                "확인할 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "scholarship#0001",
                    "source_title": "학생지원시스템 이용 안내",
                    "text": (
                        "학생지원시스템 → 장학 → 선발결과 메뉴에서 "
                        "수기분 0원 확인"
                    ),
                }
            ],
        )

        self.assertFalse(attributed["supported"])

    def test_claim_attribution_rejects_permission_from_unrelated_faq_item(self) -> None:
        claim = attribute_claim(
            (
                "분할납부자의 경우 휴학 시 잔여 등록금을 모두 완납해야 "
                "반환 등의 처리가 가능합니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "faq#0001",
                    "source_title": "재학생 등록금 FAQ",
                    "text": (
                        "등록금 납부 후 휴학하면 등록금이 이월됩니다. "
                        "학생의료공제회비 혜택을 원하지 않을 경우 "
                        "환불신청 가능합니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(claim["validation_reason"], "semantic_relation_mismatch")

    def test_claim_attribution_accepts_coordinated_installment_rounds(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "tuition#0012",
            "source_title": "2026학년도 2학기 재학생 등록금 납부 계획",
            "text": "분할 1 ・ 4차는 학자금대출 불가(학생과 1247)",
        }

        values = extract_critical_values(source["text"])
        attributed = attribute_claim(
            (
                "분할납부 신청자는 분할 1차와 4차 납부 때 "
                "학자금대출을 받을 수 없습니다."
            ),
            [source],
        )

        self.assertIn("round:1", values)
        self.assertIn("round:4", values)
        self.assertTrue(attributed["supported"], attributed)

    def test_claim_attribution_treats_supported_semesters_until_as_upper_bound(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "scholarship#0001",
            "source_title": "효명장학사업 모집 안내",
            "text": (
                "2026년 2학기 등록 예정자\n"
                "국내 학사 8학기, 국내외 석사 4학기, 국내외 박사 "
                "6학기까지 지원 가능\n"
                "성적이 우수하고 직전 학기 성적 증빙이 가능한 학생"
            ),
        }

        attributed = attribute_claim(
            "국내외 석사 4학기 이하인 자여야 합니다.",
            [source],
        )

        self.assertTrue(attributed["supported"], attributed)

        coordinated = attribute_claim(
            (
                "학부 8학기, 국내외 석사 4학기, 국내외 박사 "
                "6학기까지 지원 가능합니다."
            ),
            [source],
        )

        self.assertTrue(coordinated["supported"], coordinated)

    def test_claim_attribution_rejects_invented_result_of_required_payment(self) -> None:
        claim = attribute_claim(
            (
                "분할납부자의 경우 휴학 시 잔여 등록금을 완납해야 "
                "반환 등의 절차를 진행할 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "tuition#0012",
                    "source_title": "재학생 등록금 납부 계획",
                    "text": (
                        "분할납부자가 휴학 및 자퇴를 원할 경우 잔여 등록금을 "
                        "완납하여야 함. 국가장학금 감면에 따른 실제 등록금 "
                        "납부액이 1차 분할납입금보다 적을 경우 분할납부 신청이 "
                        "취소될 수 있음."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(claim["validation_reason"], "semantic_relation_mismatch")

    def test_claim_attribution_accepts_temporal_scope_variants(self) -> None:
        cases = [
            (
                "신청 기간은 2026년 8월 3일부터 "
                "2026년 8월 10일까지입니다.",
                "신청 기간은 2026년 8월 3일부터 8월 10일까지입니다.",
            ),
            (
                "신청 마감은 8월 31일까지입니다.",
                "신청 마감은 8월 31일 18:00까지입니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertTrue(claim["supported"])

    def test_claim_attribution_accepts_weekday_annotated_table_range(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "exchange#schedule",
            "source_title": "2027학년도 1학기 교환 프로그램 1차 선발 요강",
            "section_path": ["전형일정"],
            "text": (
                "1차\t온라인 지원\n"
                "(On-Line 지원)\t2026. 7. 17.(금) 9:00 ~ "
                "7. 26.(일) 18:00\t스마트학생정보시스템\n"
                "2차\t온라인 지원\n"
                "(On-Line 지원)\t2026. 8. 14.(금) 9:00 ~ "
                "8. 19.(수) 18:00\t예정"
            ),
        }

        supported = attribute_claim(
            (
                "1차 온라인 지원 기간은 2026. 7. 17.(금) 9:00부터 "
                "7. 26.(일) 18:00까지 스마트학생정보시스템을 통해 "
                "진행됩니다."
            ),
            [source],
        )
        wrong_round = attribute_claim(
            (
                "1차 온라인 지원 기간은 2026. 8. 14.(금) 9:00부터 "
                "8. 19.(수) 18:00까지입니다."
            ),
            [source],
        )

        self.assertTrue(supported["supported"], supported)
        self.assertFalse(wrong_round["supported"], wrong_round)

    def test_claim_attribution_rejects_relation_like_nouns(self) -> None:
        cases = [
            (
                "분할 납부가 가능합니다.",
                "분할 납부 가능성이 없습니다.",
            ),
            (
                "등록금은 인상되었습니다.",
                "등록금은 인상적인 운영 성과입니다.",
            ),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])

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

    def test_claim_attribution_expands_bare_day_range_endpoint(self) -> None:
        claim = attribute_claim(
            (
                "효명장학사업 접수기간은 2026년 7월 8일부터 "
                "2026년 7월 22일까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "hyomyeong#schedule",
                    "source_title": "2026년 하반기 효명장학사업 모집 공고",
                    "text": "접수기간\t26.7.8(수)~22(수) (2주간)",
                }
            ],
        )

        self.assertTrue(claim["supported"])
        self.assertEqual(claim["missing_critical_values"], [])
        self.assertIn(
            "date:2026-07-22",
            extract_critical_values("26.7.8(수)~22(수)"),
        )

    def test_critical_values_ignore_invalid_bare_day_range_endpoint(self) -> None:
        values = extract_critical_values("26.7.8(수)~32(수)")

        self.assertNotIn("date:2026-07-32", values)
        self.assertNotIn("month_day:07-32", values)

    def test_claim_attribution_combines_same_subject_schedule_rows(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "calendar#2026-2",
            "source_title": "학사일정 > 학사/학술 > 대학생활",
            "text": (
                "2026.07.24 - 2026.07.31\t"
                "2026년도 2학기 휴·복학 신청기간\n"
                "2026.08.10 - 2026.08.12\t"
                "2026년도 2학기 수강신청(학부)\n"
                "2026.08.18 - 2026.08.19\t"
                "2026년도 2학기 수강신청(대학원)\n"
                "2026.08.18 - 2026.08.19\t"
                "2026년도 2학기 수강신청(학부)\n"
                "2026.08.24 - 2026.08.27\t"
                "2026년도 2학기 휴·복학 신청기간"
            ),
        }

        leave = attribute_claim(
            (
                "2026학년도 2학기 휴·복학 신청 기간은 "
                "2026년 7월 24일부터 7월 31일까지이거나 "
                "2026년 8월 24일부터 8월 27일까지입니다."
            ),
            [source],
        )
        undergraduate = attribute_claim(
            (
                "2026학년도 2학기 수강신청(학부) 기간은 "
                "2026년 8월 10일부터 8월 12일까지 또는 "
                "2026년 8월 18일부터 8월 19일까지입니다."
            ),
            [source],
        )

        self.assertTrue(leave["supported"])
        self.assertTrue(undergraduate["supported"])

    def test_claim_attribution_keeps_scope_across_igeona_date_ranges(self) -> None:
        claim = attribute_claim(
            (
                "2026학년도 2학기 휴·복학 신청 기간은 "
                "2026년 7월 24일부터 7월 31일까지이거나 "
                "2026년 8월 24일부터 8월 27일까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "calendar#igeona",
                    "text": (
                        "2026.07.24 - 2026.07.31\t"
                        "2026년도 2학기 휴·복학 신청기간\n"
                        "2026.08.21 - 2026.08.24\t"
                        "2026년도 2학기 등록금 납부기간\n"
                        "2026.08.24 - 2026.08.27\t"
                        "2026년도 2학기 휴·복학 신청기간"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])

    def test_claim_attribution_combines_parallel_schedule_audiences(self) -> None:
        claim = attribute_claim(
            (
                "2026학년도 2학기 수강신청 기간은 학부, 대학원, "
                "타대생 모두 2026년 8월 10일부터 8월 12일까지 "
                "또는 2026년 8월 18일부터 8월 19일까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "calendar#audiences",
                    "text": (
                        "2026.08.10 - 2026.08.12\t"
                        "2026년도 2학기 수강신청(학부)\n"
                        "2026.08.10 - 2026.08.12\t"
                        "2026년도 2학기 수강신청(대학원)\n"
                        "2026.08.10 - 2026.08.12\t"
                        "2026년도 2학기 수강신청(타대생)\n"
                        "2026.08.18 - 2026.08.19\t"
                        "2026년도 2학기 수강신청(학부)\n"
                        "2026.08.18 - 2026.08.19\t"
                        "2026년도 2학기 수강신청(대학원)\n"
                        "2026.08.18 - 2026.08.19\t"
                        "2026년도 2학기 수강신청(타대생)"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])

    def test_claim_attribution_does_not_combine_other_schedule_subjects(self) -> None:
        claim = attribute_claim(
            (
                "2026학년도 2학기 휴·복학 신청 기간은 "
                "2026년 7월 24일부터 7월 31일까지이거나 "
                "2026년 8월 18일부터 8월 19일까지입니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "calendar#mixed",
                    "text": (
                        "2026.07.24 - 2026.07.31\t"
                        "2026년도 2학기 휴·복학 신청기간\n"
                        "2026.08.18 - 2026.08.19\t"
                        "2026년도 2학기 수강신청(학부)"
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"])
        self.assertEqual(claim["validation_reason"], "critical_value_mismatch")

    def test_claim_attribution_uses_table_row_scope_for_maximum_amount(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "scholarship#amounts",
            "text": (
                "사업명\t분야명\t지원액\n"
                "디딤돌 장학사업\t학자금대출 원금상환\t"
                "1인당 최대 1,000,000원\n"
                "디딤돌 장학사업\t학자금대출 이자지원\t"
                "1인당 최대 500,000원"
            ),
        }

        supported = attribute_claim(
            "학자금대출 원금상환 지원 금액은 1인당 최대 1,000,000원입니다.",
            [source],
        )
        swapped = attribute_claim(
            "학자금대출 원금상환 지원 금액은 1인당 최대 500,000원입니다.",
            [source],
        )

        self.assertTrue(supported["supported"])
        self.assertFalse(swapped["supported"])
        self.assertEqual(
            swapped["validation_reason"], "semantic_relation_mismatch"
        )

    def test_claim_attribution_keeps_comparator_asserted_before_capability(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "language-course#criteria",
            "source_title": "학위청구 외국어시험 대체강좌 수강생 모집",
            "text": (
                "바. 이수기준\n"
                "출석(20%), 과제(30%), 시험(50%)을 합하여 "
                "70점 이상인 자\n"
                "기준을 모두 충족해야 이수됨"
            ),
        }

        supported = attribute_claim(
            (
                "출석 20%, 과제 30%, 시험 50%를 합산하여 70점 이상을 "
                "취득해야 이수할 수 있습니다."
            ),
            [source],
        )
        wrong_threshold = attribute_claim(
            (
                "출석 20%, 과제 30%, 시험 50%를 합산하여 60점 이상을 "
                "취득해야 이수할 수 있습니다."
            ),
            [source],
        )

        self.assertTrue(supported["supported"], supported)
        self.assertFalse(wrong_threshold["supported"])

    def test_claim_attribution_accepts_documented_experience_activity(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "career-camp#activity",
            "source_title": "2026학년도 1차 진로부트캠프(온라인) 모집 안내",
            "text": (
                "프로그램 주요 내용\n"
                "90여종의 직무 VOD를 자유롭게 수강한 후 관심 진로분야"
                "(최대 3개)를 선택하여 실무과제 수행 및 현직자 피드백을 "
                "통한 직무체험"
            ),
        }

        supported = attribute_claim(
            (
                "90여종의 직무 VOD를 자유롭게 수강한 후 관심 진로분야를 "
                "최대 3개 선택하여 실무과제를 수행하고 현직자 피드백을 "
                "통해 직무를 체험할 수 있습니다."
            ),
            [source],
        )
        explicitly_unavailable = attribute_claim(
            "현직자 피드백을 통해 직무를 체험할 수 있습니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "career-camp#unavailable",
                    "source_title": "진로부트캠프 운영 변경 안내",
                    "text": "현직자 피드백을 통한 직무체험은 제공하지 않습니다.",
                }
            ],
        )

        self.assertTrue(supported["supported"], supported)
        self.assertFalse(explicitly_unavailable["supported"])

    def test_claim_attribution_accepts_documented_program_purpose(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "career-camp#purpose",
            "source_title": "2026학년도 1차 진로부트캠프(온라인) 모집 안내",
            "text": (
                "진로를 아직 정하지 못한 학생들이 여러 직무를 제한 없이 "
                "탐색하고 경험하여 진로를 구체화하도록 진로부트캠프 "
                "프로그램을 진행합니다."
            ),
        }

        claim = attribute_claim(
            (
                "진로를 정하지 못한 학생들이 여러 직무를 탐색하고 체험할 "
                "수 있는 진로부트캠프 프로그램을 진행합니다."
            ),
            [source],
        )

        self.assertTrue(claim["supported"], claim)

    def test_claim_attribution_accepts_documented_exam_preparation_program(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "skill-up#purpose",
            "source_title": "2026학년도 취업 Skill UP 프로그램(1차)",
            "text": (
                "데이터분석준전문가(ADsP) 및 AI프롬프트활용능력(AI-POT) "
                "교육 특강을 운영합니다. 특강을 통해 데이터 분석 및 AI "
                "활용 역량을 강화하여 자격증 취득을 지원하는 프로그램입니다."
            ),
        }
        claim = attribute_claim(
            (
                "데이터분석 자격증과 AI 활용 능력을 준비할 수 있는 교내 "
                "특강으로 2026학년도 1차 취업 Skill UP 프로그램이 운영됩니다."
            ),
            [source],
        )

        self.assertTrue(claim["supported"], claim)

    def test_claim_attribution_rejects_unavailable_exam_preparation_program(
        self,
    ) -> None:
        claim = attribute_claim(
            "데이터분석 자격증을 준비할 수 있는 특강이 운영됩니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "skill-up#closed",
                    "text": (
                        "데이터분석 자격증을 준비하는 교육 특강은 "
                        "운영하지 않습니다."
                    ),
                }
            ],
        )

        self.assertFalse(claim["supported"], claim)

    def test_claim_attribution_accepts_explicit_roster_eligibility(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "career-camp#eligibility",
            "source_title": "2026학년도 1차 진로부트캠프 모집 안내",
            "text": (
                "모집대상: 전체 학과(부) 1~3학년 재학생 "
                "(휴학생, 2025학년도 진로부트캠프 미수료자 제외)"
            ),
        }

        supported = attribute_claim(
            (
                "전체 학과(부) 1~3학년 재학생이 신청할 수 있으며 "
                "휴학생과 2025학년도 진로부트캠프 미수료자는 제외됩니다."
            ),
            [source],
        )
        excluded = attribute_claim(
            "휴학생이 진로부트캠프에 신청할 수 있습니다.",
            [source],
        )

        self.assertTrue(supported["supported"], supported)
        self.assertFalse(excluded["supported"])

    def test_claim_attribution_accepts_labeled_loan_eligibility(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "ai-loan#eligibility",
            "source_title": "AI학업장려 학자금대출 안내",
            "text": (
                "(대출대상) 교육부 또는 재단과 협약을 체결한 "
                "AI·SW중심대학 및 AI거점대학 사업 대상학과 학부생"
            ),
        }
        claim = attribute_claim(
            (
                "AI학업장려 학자금대출은 교육부 또는 재단과 협약을 "
                "체결한 AI·SW중심대학 및 AI거점대학 사업 대상학과 "
                "학부생이 받을 수 있습니다."
            ),
            [source],
        )

        self.assertTrue(claim["supported"], claim)

    def test_claim_attribution_accepts_labeled_application_eligibility(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "transfer#eligibility",
            "text": (
                "일반편입학 지원자격은 아래와 같습니다.\n"
                "- 국내외 대학에서 전문 학사학위 취득 예정자 또는 "
                "4년제 대학에서 2학년 이상 수료 예정자"
            ),
        }
        claim = attribute_claim(
            (
                "국내외 대학에서 전문 학사학위 취득 예정자 또는 "
                "4년제 대학에서 2학년 이상 수료 예정자가 "
                "지원할 수 있습니다."
            ),
            [source],
        )

        self.assertTrue(claim["supported"], claim)

    def test_claim_attribution_rejects_excluded_labeled_eligibility(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "loan#eligibility",
            "text": (
                "대출대상: 사업 대상학과 재학생\n"
                "지원 제한: 휴학생 및 수료생은 대출대상에서 제외"
            ),
        }
        excluded = attribute_claim(
            "휴학생 및 수료생이 학자금대출을 받을 수 있습니다.",
            [source],
        )
        unrelated = attribute_claim(
            "다른 대학 학부생이 학자금대출을 받을 수 있습니다.",
            [source],
        )

        self.assertFalse(excluded["supported"], excluded)
        self.assertFalse(unrelated["supported"], unrelated)

    def test_claim_attribution_accepts_documented_program_outcome(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "career-camp#outcome",
            "source_title": "2026학년도 1차 진로부트캠프 모집 안내",
            "text": (
                "수료 후 도출된 진로포트폴리오를 통해 체험 내용 정리 및 "
                "자신에게 적합한 직군(직무) 확인"
            ),
        }

        claim = attribute_claim(
            (
                "수료 후 도출된 진로포트폴리오를 통해 체험 내용을 정리하고 "
                "자신에게 적합한 직군이나 직무를 확인할 수 있습니다."
            ),
            [source],
        )

        self.assertTrue(claim["supported"], claim)

    def test_claim_attribution_accepts_official_ui_application_path(self) -> None:
        claim_text = (
            "부산대학교 학생지원시스템에 로그인하여 장학, 장학금신청, "
            "우선선발장학금신청 메뉴에서 온라인으로 신청할 수 있습니다."
        )
        supported = attribute_claim(
            claim_text,
            [
                {
                    "source_number": 1,
                    "chunk_id": "scholarship#apply",
                    "text": (
                        "온라인 신청: 부산대학교 학생지원시스템 로그인 → "
                        "장학 → 장학금신청 → 우선선발장학금신청에서 신청"
                    ),
                }
            ],
        )
        unrelated = attribute_claim(
            claim_text,
            [
                {
                    "source_number": 1,
                    "chunk_id": "scholarship#unrelated",
                    "text": (
                        "부산대학교 학생지원시스템 로그인 → 등록 → "
                        "납부확인 메뉴에서 영수증 출력"
                    ),
                }
            ],
        )
        unavailable = attribute_claim(
            claim_text,
            [
                {
                    "source_number": 1,
                    "chunk_id": "scholarship#closed",
                    "text": (
                        "부산대학교 학생지원시스템 로그인 → 장학 → "
                        "우선선발장학금신청 메뉴에서는 신청 불가"
                    ),
                }
            ],
        )

        self.assertTrue(supported["supported"])
        self.assertFalse(unrelated["supported"])
        self.assertFalse(unavailable["supported"])

    def test_claim_attribution_accepts_official_application_channels(self) -> None:
        supported = attribute_claim(
            (
                "학자금대출 신청은 한국장학재단 누리집 또는 모바일 "
                "애플리케이션에서 할 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "loan#method",
                    "text": (
                        "(신청방법) 한국장학재단 누리집(www.kosaf.go.kr), "
                        "모바일 애플리케이션에서 신청"
                    ),
                }
            ],
        )
        unrelated = attribute_claim(
            (
                "학자금대출 신청은 한국장학재단 누리집 또는 모바일 "
                "애플리케이션에서 할 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "loan#unrelated",
                    "text": "학생지원시스템 등록 메뉴에서 분할납부 신청",
                }
            ],
        )

        self.assertTrue(supported["supported"])
        self.assertFalse(unrelated["supported"])

    def test_claim_attribution_accepts_footnote_separated_time_ranges(self) -> None:
        claim = attribute_claim(
            (
                "학자금대출 신청 시간은 09:00~24:00이며 대출실행은 "
                "등록금 납부 기간 중 09:00~17:00에 가능합니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "loan#hours",
                    "text": (
                        "(신청시간) 09:00~24:00 ※ 대출실행은 "
                        "9:00~17:00 가능(등록금납부 기간에 가능)"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])

    def test_claim_attribution_accepts_enumerated_eligibility_list(self) -> None:
        claim = attribute_claim(
            (
                "효명장학금은 국내 4년제 정규 대학·대학원 석사과정, "
                "로스쿨, 또는 해외 유학 중인 석·박사과정에 재학 중인 "
                "시각장애인이 신청할 수 있습니다."
            ),
            [
                {
                    "source_number": 1,
                    "chunk_id": "hyomyeong#eligibility",
                    "text": (
                        "1. 선발대상 (아래사항 중 하나라도 해당할 경우, "
                        "신청 가능)\n"
                        "1) 국내 4년제 정규 대학·대학원(석사과정) 또는 "
                        "로스쿨에 재학 중인 시각장애인\n"
                        "2) 석·박사과정으로 해외 유학 중인 시각장애인"
                    ),
                }
            ],
        )

        self.assertTrue(claim["supported"])

    def test_claim_attribution_accepts_labeled_participation_benefits(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "buddy#benefits",
            "text": (
                "참여혜택 (총 16시간 이상 활동 및 활동보고서 제출자에 한함)\n"
                "1) 이수증 발급 및 비교과 마일리지 10점 부여\n"
                "2) 교환학생 지원 시 버디 활동 1회당 가산점 3점"
                "(최대 10점)"
            ),
        }
        claims = (
            "참여자는 이수증 발급 및 비교과 마일리지 10점을 부여받을 수 있습니다.",
            "교환학생 지원 시 버디 활동 1회당 가산점 3점을 최대 10점까지 받을 수 있습니다.",
        )
        for claim_text in claims:
            with self.subTest(claim=claim_text):
                claim = attribute_claim(claim_text, [source])
                self.assertTrue(claim["supported"], claim)

    def test_claim_attribution_rejects_wrong_or_unavailable_participation_benefit(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "buddy#benefits",
            "text": (
                "참여혜택\n"
                "1) 비교과 마일리지 10점 부여\n"
                "2) 교환학생 가산점 지급 없음"
            ),
        }
        wrong_value = attribute_claim(
            "참여자는 비교과 마일리지 20점을 부여받을 수 있습니다.",
            [source],
        )
        unavailable = attribute_claim(
            "참여자는 교환학생 가산점을 받을 수 있습니다.",
            [source],
        )

        self.assertFalse(wrong_value["supported"], wrong_value)
        self.assertFalse(unavailable["supported"], unavailable)

    def test_claim_attribution_accepts_exact_contact_table_row(self) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "orientation#contacts",
            "source_title": (
                "학과별 오리엔테이션 및 수강신청 일시·장소·연락처"
            ),
            "text": (
                "학과(부)\t행사명\t연락처\n"
                "국어국문학과\t오리엔테이션 및 수강신청 안내\t051-510-1507\n"
                "중어중문학과\t오리엔테이션 및 수강신청 안내\t051-510-1508"
            ),
        }
        claim = attribute_claim(
            (
                "문의 사항은 국어국문학과 연락처인 051-510-1507로 "
                "확인하실 수 있습니다."
            ),
            [source],
        )

        self.assertTrue(claim["supported"], claim)

    def test_claim_attribution_rejects_contact_from_different_owner_row(
        self,
    ) -> None:
        source = {
            "source_number": 1,
            "chunk_id": "orientation#contacts",
            "text": "중어중문학과\t연락처\t051-510-1507",
        }
        wrong_owner = attribute_claim(
            (
                "문의 사항은 국어국문학과 연락처인 051-510-1507로 "
                "확인하실 수 있습니다."
            ),
            [source],
        )
        unavailable = attribute_claim(
            "국어국문학과 연락처 051-510-1507로 연락할 수 있습니다.",
            [
                {
                    **source,
                    "text": "국어국문학과\t051-510-1507\t사용 중단",
                }
            ],
        )

        self.assertFalse(wrong_owner["supported"], wrong_owner)
        self.assertFalse(unavailable["supported"], unavailable)

    def test_semester_scope_does_not_treat_duration_as_term_number(self) -> None:
        values = extract_critical_values(
            "2026년 2학기 등록 예정자이며 학부는 8학기까지 지원합니다."
        )

        self.assertIn("semester:2", values)
        self.assertNotIn("semester:8", values)

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

    def test_critical_values_normalize_compound_man_currency(self) -> None:
        self.assertIn(
            "amount_krw:20000000",
            extract_critical_values("지원금은 2천만원입니다."),
        )
        self.assertIn(
            "amount_krw:120000000",
            extract_critical_values("지원금은 1억 2천만원입니다."),
        )

    def test_claim_attribution_compares_compound_man_currency(self) -> None:
        for amount in ("2천만원", "1억 2천만원"):
            with self.subTest(amount=amount):
                claim = attribute_claim(
                    f"지원금은 {amount} 이하입니다.",
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": f"지원금은 {amount} 이상입니다.",
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "semantic_relation_mismatch",
                )

        equivalent = attribute_claim(
            "지원금은 2천만원 이상입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0002",
                    "text": "지원금은 20,000,000원 이상입니다.",
                }
            ],
        )
        self.assertTrue(equivalent["supported"])

    def test_critical_values_preserve_quantity_units_and_decimals(self) -> None:
        self.assertIn(
            "quantity:12:개월",
            extract_critical_values("지원 기간은 12개월입니다."),
        )
        self.assertIn(
            "quantity:12:개",
            extract_critical_values("제출 서류는 12개입니다."),
        )
        self.assertIn(
            "quantity:3.5:학점",
            extract_critical_values("평점은 3.5학점입니다."),
        )

    def test_critical_values_inherit_shared_unit_for_quantity_range(self) -> None:
        for source_text in (
            "최종 성적 인정에는 통상 3~4개월이 소요됩니다.",
            "최종 성적 인정에는 통상 3-4개월이 소요됩니다.",
            "최종 성적 인정에는 통상 3에서 4개월이 소요됩니다.",
        ):
            with self.subTest(source=source_text):
                values = extract_critical_values(source_text)
                self.assertIn("quantity:3:개월", values)
                self.assertIn("quantity:4:개월", values)

    def test_claim_attribution_accepts_shared_unit_quantity_range(self) -> None:
        claim_text = (
            "최종 성적 인정은 성적표가 국제처에 도착한 후 통상적으로 "
            "3개월에서 4개월이 소요됩니다."
        )
        matching = attribute_claim(
            claim_text,
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": (
                        "최종 성적 인정은 성적표가 국제처에 도착한 후 "
                        "통상적으로 3~4개월이 소요됩니다."
                    ),
                }
            ],
        )
        wrong_endpoint = attribute_claim(
            claim_text,
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0002",
                    "text": (
                        "최종 성적 인정은 성적표가 국제처에 도착한 후 "
                        "통상적으로 2~4개월이 소요됩니다."
                    ),
                }
            ],
        )

        self.assertTrue(matching["supported"], matching)
        self.assertEqual(matching["validation_reason"], "supported")
        self.assertFalse(wrong_endpoint["supported"], wrong_endpoint)
        self.assertEqual(
            wrong_endpoint["validation_reason"],
            "critical_value_mismatch",
        )

    def test_claim_attribution_distinguishes_quantity_units_and_decimals(
        self,
    ) -> None:
        cases = [
            ("지원 기간은 12개월 이상입니다.", "지원 기간은 12개 이상입니다."),
            ("평점은 3.5학점 이상입니다.", "평점은 4.5학점 이상입니다."),
        ]
        for claim_text, source_text in cases:
            with self.subTest(source=source_text):
                claim = attribute_claim(
                    claim_text,
                    [
                        {
                            "source_number": 1,
                            "chunk_id": "notice#0001",
                            "text": source_text,
                        }
                    ],
                )
                self.assertFalse(claim["supported"])
                self.assertEqual(
                    claim["validation_reason"],
                    "critical_value_mismatch",
                )

        equivalent = attribute_claim(
            "평점은 3.50학점 이상입니다.",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0002",
                    "text": "평점은 3.5학점 이상입니다.",
                }
            ],
        )
        self.assertTrue(equivalent["supported"])

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

    def test_service_query_expansion_targets_document_type_not_answer_values(self) -> None:
        outsourcing = expand_service_retrieval_query(
            "학생증 발급과 증명서 발급을 외부 기관에 위탁하나요?"
        )
        admissions = expand_service_retrieval_query(
            "2027학년도 신입생은 총 몇 명이고 전년 대비 무엇이 달라지나요?"
        )
        group_visa = expand_service_retrieval_query(
            "D-2 비자 연장을 학교 단체접수로 어떻게 이용하나요?"
        )
        foreign_admission = expand_service_retrieval_query(
            "외국인 유학생인데 부산대 학부에 신입학하려면 어떤 자격이 필요한가요?"
        )
        exchange_selection = expand_service_retrieval_query(
            "교환학생으로 해외에 나가고 싶은데 선발 규모와 지원 일정이 어떻게 되나요?"
        )
        summer_career = expand_service_retrieval_query(
            "여름방학 동안 학교에서 자격증 대비 강의 같은 걸 들을 수 있나요?"
        )
        disability_support = expand_service_retrieval_query(
            "장애가 있는 학생인데 수업이나 시험에서 어떤 지원을 받을 수 있나요?"
        )
        third_party_report = expand_service_retrieval_query(
            "인권침해를 목격했는데 피해자가 아닌 제가 대신 신고해도 되나요?"
        )

        self.assertIn("개인정보처리", outsourcing)
        self.assertIn("위탁현황", outsourcing)
        self.assertIn("수탁기관", outsourcing)
        self.assertIn("대학입학전형", admissions)
        self.assertIn("기본계획", admissions)
        self.assertIn("주요", admissions)
        self.assertIn("변경사항", admissions)
        self.assertTrue(admissions.startswith("대학입학전형 기본계획"))
        self.assertIn("사전", group_visa)
        self.assertIn("예약", group_visa)
        self.assertIn("제출서류", group_visa)
        self.assertIn("수수료", group_visa)
        self.assertTrue(group_visa.startswith("비자 단체접수"))
        self.assertTrue(
            foreign_admission.startswith(
                "외국인 특별전형 모집요강 지원자격 국적 언어능력 학력"
            )
        )
        self.assertNotIn("TOPIK 3급", foreign_admission)
        self.assertNotIn("이중국적자는 지원 불가", foreign_admission)
        self.assertTrue(
            exchange_selection.startswith(
                "교환 교비 프로그램 1차 선발요강 선발규모 온라인지원 합격자발표"
            )
        )
        self.assertTrue(
            summer_career.startswith(
                "하계방학 취업역량 강화 비교과 프로그램 자격증 대비반"
            )
        )
        self.assertTrue(
            disability_support.startswith(
                "장애학생지원센터 학생지원기관 교수학습 강의지원 교재지원"
            )
        )
        self.assertTrue(
            third_party_report.startswith("센터이용 Q&A 제3자 신고")
        )
        self.assertEqual(
            expand_service_retrieval_query("휴학 신청 방법은?"),
            normalize_retrieval_query("휴학 신청 방법은?"),
        )

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

    def test_sentence_splitter_preserves_korean_dotted_dates(self) -> None:
        claims = split_candidate_sentences(
            (
                "본등록은 2026. 8. 24.(월) ~ 8. 27.(목)입니다. "
                "수납은행은 농협과 부산은행입니다."
            ),
            minimum_chars=8,
        )

        self.assertEqual(len(claims), 2)
        self.assertIn("2026. 8. 24.(월) ~ 8. 27.(목)", claims[0])
        self.assertEqual(claims[1], "수납은행은 농협과 부산은행입니다.")

    def test_sentence_splitter_preserves_spaced_weekday_after_short_year_date(
        self,
    ) -> None:
        claims = split_candidate_sentences(
            (
                "신청은 ‘26. 8. 3. (월)부터 시작합니다. "
                "다음 절차는 학생지원시스템에서 확인합니다."
            ),
            minimum_chars=8,
        )

        self.assertEqual(
            claims,
            [
                "신청은 ‘26. 8. 3. (월)부터 시작합니다.",
                "다음 절차는 학생지원시스템에서 확인합니다.",
            ],
        )

    def test_sentence_splitter_keeps_terminal_date_as_sentence_boundary(
        self,
    ) -> None:
        claims = split_candidate_sentences(
            (
                "마감일은 2026. 8. 24. "
                "다음 절차는 학생지원시스템에서 확인합니다."
            ),
            minimum_chars=8,
        )

        self.assertEqual(
            claims,
            [
                "마감일은 2026. 8. 24.",
                "다음 절차는 학생지원시스템에서 확인합니다.",
            ],
        )

    def test_sentence_splitter_keeps_conjoined_dotted_date_ranges(self) -> None:
        claims = split_draft_claims(
            (
                "휴·복학 신청 기간은 2026.07.24. - 2026.07.31. 및 "
                "2026.08.24. - 2026.08.27.입니다.\n"
                "수강신청 기간은 2026.08.10. - 2026.08.12. 및 "
                "2026.08.18. - 2026.08.19.입니다."
            )
        )

        self.assertEqual(len(claims), 2)
        self.assertIn("2026.08.24. - 2026.08.27.", claims[0])
        self.assertIn("2026.08.18. - 2026.08.19.", claims[1])

    def test_draft_claim_splitter_preserves_short_complete_fact(self) -> None:
        self.assertEqual(
            split_draft_claims("학부 등록금은 동결되었습니다."),
            ["학부 등록금은 동결되었습니다."],
        )

    def test_draft_claim_splitter_still_rejects_tiny_noise(self) -> None:
        self.assertEqual(split_draft_claims("메뉴"), [])

    def test_draft_claim_splitter_rejects_short_navigation_noise(self) -> None:
        for draft in (
            "공지사항 바로가기",
            "자세한 내용 보기",
            "첨부파일 다운로드",
        ):
            with self.subTest(draft=draft):
                self.assertEqual(split_draft_claims(draft), [])

    def test_draft_claim_splitter_does_not_silently_truncate_at_five(
        self,
    ) -> None:
        draft = "\n".join(
            f"항목 {index}의 신청 조건은 충족됩니다."
            for index in range(1, 7)
        )

        self.assertGreaterEqual(MAX_CLAIMS, 8)
        self.assertEqual(len(split_draft_claims(draft)), 6)

    def test_rag_response_reports_bounded_claim_truncation(self) -> None:
        claim_lines = [
            f"항목 {index}의 신청 조건은 충족됩니다."
            for index in range(1, MAX_CLAIMS + 2)
        ]
        response = build_rag_response(
            "신청 조건을 모두 알려줘",
            [
                {
                    "source_number": 1,
                    "chunk_id": "notice#0001",
                    "text": "\n".join(claim_lines),
                }
            ],
            "\n".join(claim_lines),
            "gemini:test",
        )

        self.assertEqual(len(response["claims"]), MAX_CLAIMS)
        self.assertEqual(
            response["postprocessing"],
            {
                "claim_limit": MAX_CLAIMS,
                "input_claim_count": MAX_CLAIMS + 1,
                "processed_claim_count": MAX_CLAIMS,
                "truncated_claim_count": 1,
                "claims_truncated": True,
            },
        )

    def test_bm25_pipeline_can_capture_raw_evaluation_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = self.make_index(Path(tmp))

            results, trace = search_pipeline(
                index_path,
                None,
                "ragtestterm 핵심 조건",
                1,
                "한국거래소",
                include_text=True,
                retrieval_mode="bm25",
                capture_candidates=True,
            )

        raw = trace["evaluation_candidates"]["raw_bm25"]
        self.assertEqual(len(results), 1)
        self.assertGreater(len(raw), len(results))
        self.assertIn("text", raw[0])

    def test_bm25_pipeline_forwards_context_document_cap(self) -> None:
        with patch("search_api.search_index", return_value=[]) as search_mock:
            search_pipeline(
                Path("unused.sqlite"),
                None,
                "등록금 납부",
                8,
                "부산대학교",
                include_text=True,
                retrieval_mode="bm25",
                max_chunks_per_document=4,
            )

        self.assertEqual(
            search_mock.call_args.kwargs["max_chunks_per_document"],
            4,
        )

    def test_bm25_pipeline_expands_only_the_tuned_service_lane(self) -> None:
        question = "2027학년도 신입생은 총 몇 명이고 전년 대비 달라지나요?"
        with patch("search_api.search_index", return_value=[]) as search_mock:
            _, control_trace = search_pipeline(
                Path("unused.sqlite"),
                None,
                question,
                8,
                "부산대학교",
                include_text=True,
                retrieval_mode="bm25",
                service_tuning=False,
            )
            _, tuned_trace = search_pipeline(
                Path("unused.sqlite"),
                None,
                question,
                8,
                "부산대학교",
                include_text=True,
                retrieval_mode="bm25",
                service_tuning=True,
            )

        normalized = normalize_retrieval_query(question, "부산대학교")
        self.assertEqual(search_mock.call_args_list[0].args[1], normalized)
        self.assertEqual(control_trace["retrieval_query"], normalized)
        self.assertEqual(control_trace["query_expansions"], [])
        self.assertIn("대학입학전형", search_mock.call_args_list[1].args[1])
        self.assertEqual(
            tuned_trace["query_expansions"],
            ["대학입학전형", "기본계획", "모집인원", "주요", "변경사항"],
        )

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

    def test_http_chat_preserves_json_null_institution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patched_env(
            RAG_API_TOKEN="",
            RAG_GENERATION_MODE="extractive",
            RAG_MAX_TOP_K="1",
        ):
            QuietSearchHandler.index_path = self.make_index(Path(tmp))
            server = ThreadingHTTPServer(("127.0.0.1", 0), QuietSearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]

            try:
                status, payload = self.post_chat(
                    port,
                    {
                        "question": "ragtestterm",
                        "institution": None,
                        "top_k": 1,
                    },
                )

                self.assertEqual(status, 200)
                self.assertIsNone(payload["institution"])
                self.assertEqual(len(payload["results"]), 1)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
