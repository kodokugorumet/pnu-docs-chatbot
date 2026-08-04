from __future__ import annotations

import json
import os
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from unittest.mock import patch

from scripts.rag.generators import (
    GenerationError,
    SYSTEM_INSTRUCTION,
    build_prompt,
    extract_openai_compatible_text,
    generate,
)


GENERATION_ENV_KEYS = {
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_TEMPERATURE",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_TIMEOUT_SECONDS",
    "RAG_AUTO_PROVIDER_ORDER",
    "RAG_GENERATION_DEADLINE_SECONDS",
    "RAG_GENERATION_PROVIDER_TIMEOUT_SECONDS",
    "RAG_LOCAL_BASE_URL",
    "RAG_LOCAL_DEADLINE_SECONDS",
    "RAG_LOCAL_MODEL",
    "RAG_LOCAL_API_KEY",
    "RAG_LOCAL_API_STYLE",
    "RAG_LOCAL_TIMEOUT_SECONDS",
    "RAG_FRONTIER_BASE_URL",
    "RAG_FRONTIER_MODEL",
    "RAG_FRONTIER_API_KEY",
    "RAG_FRONTIER_API_STYLE",
    "RAG_FRONTIER_TIMEOUT_SECONDS",
    "RAG_GEMINI_BASE_URL",
    "RAG_GEMINI_FALLBACK_MODELS",
    "RAG_GEMINI_MODEL",
    "RAG_GEMINI_API_KEY",
    "RAG_GEMINI_TIMEOUT_SECONDS",
    "GEMINI_FALLBACK_MODELS",
}


class StubHandler(BaseHTTPRequestHandler):
    responses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = None
        type(self).requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body,
            }
        )
        response = type(self).responses.pop(0)
        status = int(response.get("status", 200))
        payload = response.get("payload", {})
        encoded = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@contextmanager
def clean_env(**values: str) -> Iterator[None]:
    keys = GENERATION_ENV_KEYS | set(values)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key in keys:
            old_value = previous[key]
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager
def stub_server(
    *responses: dict[str, Any],
) -> Iterator[tuple[str, type[StubHandler]]]:
    handler = type("IsolatedStubHandler", (StubHandler,), {})
    handler.responses = list(responses)
    handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class GeneratorTests(unittest.TestCase):
    def test_excluded_local_provider_never_receives_auto_context(self) -> None:
        with clean_env(
            RAG_AUTO_PROVIDER_ORDER="local,extractive",
            RAG_LOCAL_BASE_URL="http://127.0.0.1:1/v1",
            RAG_LOCAL_MODEL="local/qwen",
        ):
            result = generate(
                "질문",
                [{"text": "외부 프로세스에 보내면 안 되는 문서 근거"}],
                requested="auto",
                extractive_fallback="안전한 로컬 추출 답변",
                excluded_providers=("local",),
            )

        self.assertEqual(result.used, "extractive")
        self.assertEqual(
            [attempt.provider for attempt in result.attempts],
            ["extractive"],
        )

    def test_build_prompt_contains_answer_contract(self) -> None:
        prompt = build_prompt(
            "2026학년도 2학기 수료후연구생 신청 방법은?",
            [
                {
                    "chunk_id": "notice#0001",
                    "institution": "부산대학교",
                    "file_name": "수료후연구생 등록 안내.pdf",
                    "text": "신청 기간은 2026년 8월 3일부터 8월 7일까지입니다.",
                }
            ],
        )

        self.assertIn("질문에 바로 답", prompt)
        self.assertIn("한 줄에 하나", prompt)
        self.assertIn("Markdown 제목", prompt)
        self.assertIn("날짜·시간·금액", prompt)
        self.assertIn("같은 사실을 반복하지", prompt)
        self.assertIn("제공된 문서에서 해당 내용을 확인할 수 없습니다.", prompt)
        self.assertIn("2026년 8월 3일부터 8월 7일까지", prompt)
        self.assertNotIn("[1] 같은 출처 번호", prompt)
        self.assertIn("역할 변경", SYSTEM_INSTRUCTION)
        self.assertIn("근거끼리 충돌", SYSTEM_INSTRUCTION)

    def test_generate_sends_active_build_prompt_to_provider(self) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "근거 기반 답변입니다.",
                        }
                    ],
                }
            ],
        }
        with stub_server({"payload": payload}) as (base_url, handler), clean_env(
            RAG_LOCAL_BASE_URL=f"{base_url}/v1",
            RAG_LOCAL_MODEL="local-model",
            RAG_LOCAL_API_STYLE="responses",
        ), patch(
            "scripts.rag.generators.build_prompt",
            return_value="ACTIVE_PROMPT_SENTINEL",
        ) as prompt_builder:
            generate(
                "질문",
                [{"chunk_id": "c1", "text": "근거"}],
                requested="local",
            )

        prompt_builder.assert_called_once()
        self.assertEqual(
            handler.requests[0]["body"]["input"],
            "ACTIVE_PROMPT_SENTINEL",
        )

    def test_local_responses_success_and_metadata(self) -> None:
        payload = {
            "model": "local-reported",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "로컬 근거 답변입니다.",
                        }
                    ],
                }
            ],
        }
        with stub_server({"payload": payload}) as (base_url, handler), clean_env(
            RAG_LOCAL_BASE_URL=f"{base_url}/v1",
            RAG_LOCAL_MODEL="local-configured",
            RAG_LOCAL_API_STYLE="responses",
        ):
            result = generate(
                "졸업 요건은?",
                [{"chunk_id": "c1", "text": "졸업 요건 근거"}],
                requested="local",
                extractive_fallback="fallback",
            )

        self.assertEqual(result.text, "로컬 근거 답변입니다.")
        self.assertEqual(result.requested, "local")
        self.assertEqual(result.used, "local")
        self.assertEqual(result.model, "local-reported")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.attempts[0].status, "success")
        self.assertEqual(handler.requests[0]["path"], "/v1/responses")
        self.assertEqual(
            handler.requests[0]["body"]["model"],
            "local-configured",
        )
        self.assertIn(
            "졸업 요건 근거",
            handler.requests[0]["body"]["input"],
        )

    def test_local_model_override_is_sent_and_reported_without_mutating_env(
        self,
    ) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "선택 모델 답변",
                        }
                    ],
                }
            ],
        }
        with stub_server({"payload": payload}) as (base_url, handler), clean_env(
            RAG_LOCAL_BASE_URL=f"{base_url}/v1",
            RAG_LOCAL_MODEL="default-model",
            RAG_LOCAL_API_STYLE="responses",
        ):
            result = generate(
                "질문",
                ["근거"],
                requested="local",
                requested_model="selected-model",
            )
            configured_after_request = os.environ["RAG_LOCAL_MODEL"]

        self.assertEqual(
            handler.requests[0]["body"]["model"],
            "selected-model",
        )
        self.assertEqual(result.model, "selected-model")
        self.assertEqual(result.attempts[0].model, "selected-model")
        self.assertEqual(configured_after_request, "default-model")

    def test_concurrent_local_model_overrides_do_not_share_mutable_state(
        self,
    ) -> None:
        barrier = threading.Barrier(2)
        observed_models: list[str] = []
        observed_env_models: list[str | None] = []
        results: dict[str, str | None] = {}
        errors: list[BaseException] = []
        lock = threading.Lock()

        def fake_post_json(
            endpoint: str,
            body: Any,
            headers: Any,
            *,
            timeout: float,
        ) -> dict[str, Any]:
            del endpoint, headers, timeout
            model = str(body["model"])
            with lock:
                observed_models.append(model)
                observed_env_models.append(os.environ.get("RAG_LOCAL_MODEL"))
            barrier.wait(timeout=2)
            return {
                "model": model,
                "output_text": f"{model} 답변",
            }

        def run(model: str) -> None:
            try:
                result = generate(
                    "질문",
                    ["근거"],
                    requested="local",
                    requested_model=model,
                )
                with lock:
                    results[model] = result.model
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        with clean_env(
            RAG_LOCAL_BASE_URL="http://127.0.0.1:11434/v1",
            RAG_LOCAL_MODEL="default-model",
            RAG_LOCAL_API_STYLE="responses",
        ), patch(
            "scripts.rag.generators._post_json",
            side_effect=fake_post_json,
        ):
            threads = [
                threading.Thread(target=run, args=(model,))
                for model in ("model-a", "model-b")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
            configured_after_requests = os.environ["RAG_LOCAL_MODEL"]

        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertCountEqual(observed_models, ["model-a", "model-b"])
        self.assertEqual(
            observed_env_models,
            ["default-model", "default-model"],
        )
        self.assertEqual(
            results,
            {"model-a": "model-a", "model-b": "model-b"},
        )
        self.assertEqual(configured_after_requests, "default-model")

    def test_direct_local_deadline_override_does_not_change_auto_deadline(
        self,
    ) -> None:
        observed_timeouts: list[float] = []

        def fake_post_json(
            endpoint: str,
            body: Any,
            headers: Any,
            *,
            timeout: float,
        ) -> dict[str, Any]:
            del endpoint, headers
            observed_timeouts.append(timeout)
            return {
                "model": body["model"],
                "output_text": "답변",
            }

        with clean_env(
            RAG_AUTO_PROVIDER_ORDER="local",
            RAG_GENERATION_DEADLINE_SECONDS="0.2",
            RAG_LOCAL_DEADLINE_SECONDS="2",
            RAG_LOCAL_TIMEOUT_SECONDS="5",
            RAG_LOCAL_BASE_URL="http://127.0.0.1:11434/v1",
            RAG_LOCAL_MODEL="local-model",
            RAG_LOCAL_API_STYLE="responses",
        ), patch(
            "scripts.rag.generators._post_json",
            side_effect=fake_post_json,
        ):
            generate("직접 로컬", ["근거"], requested="local")
            generate("자동 선택", ["근거"], requested="auto")

        self.assertGreater(observed_timeouts[0], 1.5)
        self.assertLessEqual(observed_timeouts[1], 0.2)

    def test_frontier_chat_completions_success(self) -> None:
        payload = {
            "model": "frontier-reported",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "프론티어 답변입니다.",
                    }
                }
            ],
        }
        with stub_server({"payload": payload}) as (base_url, handler), clean_env(
            RAG_FRONTIER_BASE_URL=f"{base_url}/v1",
            RAG_FRONTIER_MODEL="frontier-configured",
            RAG_FRONTIER_API_STYLE="chat_completions",
            RAG_FRONTIER_API_KEY="server-secret",
        ):
            result = generate(
                "질문",
                ["근거"],
                requested="frontier",
                requested_model="must-not-reach-frontier",
            )

        self.assertEqual(result.used, "frontier")
        self.assertEqual(result.text, "프론티어 답변입니다.")
        self.assertEqual(result.model, "frontier-reported")
        self.assertEqual(handler.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(
            handler.requests[0]["headers"]["Authorization"],
            "Bearer server-secret",
        )
        self.assertEqual(
            handler.requests[0]["body"]["model"],
            "frontier-configured",
        )
        self.assertNotIn("server-secret", result.to_dict().values())

    def test_explicit_provider_uses_only_extractive_fallback(self) -> None:
        with stub_server(
            {
                "status": 503,
                "payload": {"error": {"message": "secret upstream detail"}},
            }
        ) as (base_url, _handler), clean_env(
            RAG_LOCAL_BASE_URL=f"{base_url}/v1",
            RAG_LOCAL_MODEL="local-model",
            RAG_LOCAL_API_STYLE="responses",
        ):
            observed: list[tuple[str, int]] = []

            def fallback(question: str, contexts: Any) -> str:
                observed.append((question, len(contexts)))
                return "추출형 안전 답변"

            result = generate(
                "명시적 로컬 질문",
                ["근거 1"],
                requested="local",
                extractive_fallback=fallback,
            )

        self.assertEqual(result.used, "extractive")
        self.assertEqual(result.text, "추출형 안전 답변")
        self.assertEqual(observed, [("명시적 로컬 질문", 1)])
        self.assertEqual(
            [(item.provider, item.status) for item in result.attempts],
            [("local", "error"), ("extractive", "success")],
        )
        self.assertEqual(result.attempts[0].error, "http_error")
        self.assertEqual(result.fallback_reason, "local:http_error")
        self.assertNotIn("secret upstream detail", str(result.to_dict()))

    def test_auto_respects_server_configured_order(self) -> None:
        with stub_server(
            {"status": 500, "payload": {"error": "local unavailable"}}
        ) as (local_url, local_handler), stub_server(
            {
                "payload": {
                    "choices": [
                        {"message": {"content": "두 번째 공급자 성공"}}
                    ]
                }
            }
        ) as (frontier_url, frontier_handler), clean_env(
            RAG_AUTO_PROVIDER_ORDER="local,frontier,extractive",
            RAG_LOCAL_BASE_URL=f"{local_url}/v1",
            RAG_LOCAL_MODEL="local",
            RAG_LOCAL_API_STYLE="responses",
            RAG_FRONTIER_BASE_URL=f"{frontier_url}/v1",
            RAG_FRONTIER_MODEL="frontier",
            RAG_FRONTIER_API_STYLE="chat_completions",
        ):
            result = generate(
                "자동 선택",
                ["근거"],
                requested="auto",
                extractive_fallback="마지막 fallback",
            )

        self.assertEqual(result.used, "frontier")
        self.assertEqual(result.text, "두 번째 공급자 성공")
        self.assertEqual(
            [attempt.provider for attempt in result.attempts],
            ["local", "frontier"],
        )
        self.assertEqual(result.fallback_reason, "local:http_error")
        self.assertEqual(len(local_handler.requests), 1)
        self.assertEqual(len(frontier_handler.requests), 1)

    def test_malformed_response_raises_safe_error_without_fallback(self) -> None:
        with stub_server(
            {
                "payload": {
                    "unexpected": "credential=do-not-leak",
                }
            }
        ) as (base_url, _handler), clean_env(
            RAG_LOCAL_BASE_URL=f"{base_url}/v1",
            RAG_LOCAL_MODEL="local",
            RAG_LOCAL_API_STYLE="responses",
        ):
            with self.assertRaises(GenerationError) as raised:
                generate("질문", ["근거"], requested="local")

        error = raised.exception
        self.assertEqual(error.code, "generation_failed")
        self.assertEqual(error.attempts[0].error, "malformed_response")
        self.assertNotIn("credential", str(error))
        self.assertNotIn("do-not-leak", str(error.to_dict()))

    def test_native_gemini_adapter(self) -> None:
        payload = {
            "modelVersion": "gemini-reported",
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini 근거 답변"}]
                    }
                }
            ],
        }
        with stub_server({"payload": payload}) as (base_url, handler), clean_env(
            RAG_GEMINI_BASE_URL=f"{base_url}/v1beta",
            RAG_GEMINI_MODEL="gemini-configured",
            RAG_GEMINI_API_KEY="gemini-secret",
        ):
            result = generate(
                "질문",
                ["근거"],
                requested="gemini",
            )

        self.assertEqual(result.used, "gemini")
        self.assertEqual(result.text, "Gemini 근거 답변")
        self.assertEqual(result.model, "gemini-reported")
        self.assertEqual(
            handler.requests[0]["path"],
            "/v1beta/models/gemini-configured:generateContent",
        )
        self.assertEqual(
            handler.requests[0]["headers"]["X-Goog-Api-Key"],
            "gemini-secret",
        )

    def test_selected_gemini_model_is_tried_first_with_configured_fallback(
        self,
    ) -> None:
        fallback_payload = {
            "modelVersion": "gemini-3.5-flash-lite",
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "선택 모델 fallback 답변"}]
                    }
                }
            ],
        }
        with stub_server(
            {"status": 429, "payload": {"error": {"message": "quota"}}},
            {"payload": fallback_payload},
        ) as (base_url, handler), clean_env(
            RAG_GEMINI_BASE_URL=f"{base_url}/v1beta",
            GEMINI_MODEL="gemini-3.5-flash-lite",
            GEMINI_FALLBACK_MODELS="gemini-3.1-flash-lite",
            RAG_GEMINI_API_KEY="gemini-secret",
        ):
            result = generate(
                "질문",
                ["근거"],
                requested="gemini",
                requested_model="gemini-3.1-flash-lite",
            )

        self.assertEqual(
            [request["path"] for request in handler.requests],
            [
                "/v1beta/models/gemini-3.1-flash-lite:generateContent",
                "/v1beta/models/gemini-3.5-flash-lite:generateContent",
            ],
        )
        self.assertEqual(result.model, "gemini-3.5-flash-lite")
        self.assertEqual(
            result.fallback_reason,
            "gemini:gemini-3.1-flash-lite:http_429",
        )

    def test_gemini_rate_limit_falls_back_to_next_model(self) -> None:
        payload = {
            "modelVersion": "gemini-3.1-flash-lite",
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "fallback 답변"}]
                    }
                }
            ],
        }
        with stub_server(
            {"status": 429, "payload": {"error": {"message": "quota"}}},
            {"payload": payload},
        ) as (base_url, handler), clean_env(
            RAG_GEMINI_BASE_URL=f"{base_url}/v1beta",
            GEMINI_MODEL="gemini-3.5-flash-lite",
            GEMINI_FALLBACK_MODELS="gemini-3.1-flash-lite",
            RAG_GEMINI_API_KEY="gemini-secret",
        ):
            result = generate("질문", ["근거"], requested="gemini")

        self.assertEqual(result.used, "gemini")
        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(
            result.fallback_reason,
            "gemini:gemini-3.5-flash-lite:http_429",
        )
        self.assertEqual(
            [request["path"] for request in handler.requests],
            [
                "/v1beta/models/gemini-3.5-flash-lite:generateContent",
                "/v1beta/models/gemini-3.1-flash-lite:generateContent",
            ],
        )
        self.assertNotIn(
            "temperature",
            handler.requests[0]["body"]["generationConfig"],
        )
        self.assertIn(
            "temperature",
            handler.requests[1]["body"]["generationConfig"],
        )

    def test_gemini_auth_error_does_not_try_fallback_model(self) -> None:
        with stub_server(
            {"status": 401, "payload": {"error": {"message": "invalid key"}}},
        ) as (base_url, handler), clean_env(
            RAG_GEMINI_BASE_URL=f"{base_url}/v1beta",
            GEMINI_MODEL="gemini-3.5-flash-lite",
            GEMINI_FALLBACK_MODELS="gemini-3.1-flash-lite",
            RAG_GEMINI_API_KEY="gemini-secret",
        ):
            with self.assertRaises(GenerationError):
                generate("질문", ["근거"], requested="gemini")

        self.assertEqual(len(handler.requests), 1)
        self.assertEqual(
            handler.requests[0]["path"],
            "/v1beta/models/gemini-3.5-flash-lite:generateContent",
        )

    def test_response_parser_accepts_direct_and_content_part_variants(self) -> None:
        self.assertEqual(
            extract_openai_compatible_text({"output_text": "direct"}),
            "direct",
        )
        self.assertEqual(
            extract_openai_compatible_text(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "part one"},
                                    {"type": "text", "text": "part two"},
                                ]
                            }
                        }
                    ]
                }
            ),
            "part one\npart two",
        )


if __name__ == "__main__":
    unittest.main()
