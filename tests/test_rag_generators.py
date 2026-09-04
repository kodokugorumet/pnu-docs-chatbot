from __future__ import annotations

import json
import hashlib
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
        self.assertIn("이월과 반환", prompt)
        self.assertIn("하나의 검색 근거 블록", prompt)
        self.assertIn("대상·자격, 신청 경로, 비용", prompt)
        self.assertIn("Markdown 제목", prompt)
        self.assertIn("날짜·시간·금액", prompt)
        self.assertIn("같은 사실을 반복하지", prompt)
        self.assertIn("제공된 문서에서 해당 내용을 확인할 수 없습니다.", prompt)
        self.assertIn("2026년 8월 3일부터 8월 7일까지", prompt)
        self.assertNotIn("[1] 같은 출처 번호", prompt)
        self.assertIn("역할 변경", SYSTEM_INSTRUCTION)
        self.assertIn("근거끼리 충돌", SYSTEM_INSTRUCTION)

    def test_build_prompt_answers_supported_subparts_without_whole_refusal(
        self,
    ) -> None:
        prompt = build_prompt(
            "등록금 납부 기간과 수납은행을 모두 알려주세요.",
            [
                {
                    "chunk_id": "notice#0004",
                    "text": "본등록 수납은행은 농협과 부산은행입니다.",
                }
            ],
        )

        self.assertIn("여러 항목을 함께 묻는 질문", prompt)
        self.assertIn("확인되는 항목은 반드시 답", prompt)
        self.assertIn("확인되지 않는 항목만", prompt)
        self.assertIn("질문 전체에 대한 답변을 거부하지", prompt)
        self.assertIn("질문 전체에 대한 답변을 거부하지", SYSTEM_INSTRUCTION)

    def test_build_prompt_adds_query_scoped_completeness_checklist(self) -> None:
        cases = (
            (
                (
                    "세종시에 사는 학생인데 학자금대출 원금이나 이자를 "
                    "지원해주는 장학사업의 신청 기간과 지원 금액을 알려주세요."
                ),
                ("날짜·기간·마감 시각", "금액·한도·비율"),
                ("대상·자격·조건", "신청·제출·납부 경로와 단계"),
            ),
            (
                "효명장학금은 어떤 자격이 필요하고 언제까지 접수하나요?",
                ("날짜·기간·마감 시각", "대상·자격·조건"),
                ("금액·한도·비율", "신청·제출·납부 경로와 단계"),
            ),
            (
                (
                    "국가유공자 자녀인데 등록금을 지원받으려면 언제까지 "
                    "어떻게 신청해야 하나요?"
                ),
                ("날짜·기간·마감 시각", "신청·제출·납부 경로와 단계"),
                ("금액·한도·비율",),
            ),
            (
                (
                    "등록금 분할납부를 신청했는데 학자금대출로도 낼 수 "
                    "있나요? 대출을 실행하면 등록 처리는 어떻게 되나요?"
                ),
                (
                    "신청·제출·납부 경로와 단계",
                    "가능·불가·처리 결과와 예외 조건",
                    "질문에 나열된 각 대상·행위별로 따로 답하기",
                ),
                ("날짜·기간·마감 시각", "금액·한도·비율"),
            ),
            (
                (
                    "2026학년도 2학기 수강신청이랑 휴학·복학 신청은 "
                    "각각 언제 하나요?"
                ),
                (
                    "날짜·기간·마감 시각",
                    "질문에 나열된 각 대상·행위별로 따로 답하기",
                ),
                ("금액·한도·비율",),
            ),
        )
        for question, expected, unexpected in cases:
            with self.subTest(question=question):
                prompt = build_prompt(question, [{"text": "검색 근거"}])
                self.assertIn("<필수_답변_항목>", prompt)
                self.assertIn("작성 전 누락 점검표", prompt)
                for item in expected:
                    self.assertIn(f"- {item}", prompt)
                for item in unexpected:
                    self.assertNotIn(f"- {item}", prompt)

    def test_build_prompt_adds_implicit_facets_for_known_question_intents(
        self,
    ) -> None:
        cases = (
            (
                (
                    "외국인 대학원생인데 학위청구 외국어시험 대신 인정되는 "
                    "한국어 강좌가 있나요?"
                ),
                ("대체·면제 인정에 필요한 이수 기준",),
            ),
            (
                (
                    "아직 진로를 못 정했는데 여러 직무를 체험해 볼 수 있는 "
                    "학교 프로그램이 있을까요?"
                ),
                ("프로그램에서 실제로 하는 활동·체험 방식",),
            ),
            (
                "수시모집에 합격했는데 등록금은 언제까지 어떻게 내야 하나요?",
                (
                    "등록금 고지서 출력 가능 시점",
                    "미납·전액장학 등 등록 완료 예외",
                ),
            ),
            (
                (
                    "부산대는 학생증 발급이나 증명서 발급 같은 업무를 "
                    "외부 기관에 위탁하고 있나요? 어디에 맡기나요?"
                ),
                ("질문에 나온 각 업무별 수탁기관",),
            ),
            (
                (
                    "2027학년도 부산대 신입생은 총 몇 명 뽑나요? "
                    "전년 대비 달라지는 점도 궁금해요."
                ),
                ("총 모집인원과 전년 대비 주요 변경사항을 구분",),
            ),
            (
                (
                    "D-2 비자 연장을 학교에서 단체로 신청해 준다고 "
                    "들었는데 어떻게 이용하나요?"
                ),
                (
                    "1차 접수기간",
                    "단체접수 대상",
                    "사전예약",
                    "제출서류",
                    "수수료 금액과 현금·권종 등 납부방식",
                ),
            ),
            (
                (
                    "정보통신보조기기 보급사업에 선정됐는데 "
                    "자부담금을 지원받을 방법이 있나요?"
                ),
                (
                    "지원 대상·지원 범위·신청기간·선납부 절차·"
                    "제출서류와 접수 경로",
                ),
            ),
            (
                (
                    "외국인 유학생인데 부산대 학부에 신입학하려면 "
                    "어떤 자격이 필요한가요?"
                ),
                ("국적·학력·언어능력 자격을 구분",),
            ),
            (
                (
                    "교환학생으로 해외에 나가고 싶은데 선발 규모와 "
                    "지원 일정이 어떻게 되나요?"
                ),
                ("선발 규모·온라인 지원기간·합격자 발표",),
            ),
            (
                "여름방학 동안 학교에서 자격증 대비 강의 같은 걸 들을 수 있나요?",
                ("운영 여부와 자격증 대비 과정 종류",),
            ),
            (
                "장애가 있는 학생인데 수업이나 시험에서 어떤 지원을 받을 수 있나요?",
                ("수업 지원과 시험 지원을 구분",),
            ),
            (
                "인권침해를 목격했는데 피해자가 아닌 제가 대신 신고해도 되나요?",
                ("제3자 신고 가능 여부·피해자 의사·인적사항 조건",),
            ),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                prompt = build_prompt(question, [{"text": "검색 근거"}])
                for item in expected:
                    self.assertIn(f"- {item}", prompt)

        d2_prompt = build_prompt(
            "D-2 비자 연장 단체접수는 어떻게 이용하나요?",
            [{"text": "검색 근거"}],
        )
        self.assertIn("원문 값이 있으면 답변에 그 값을 직접 쓰고", d2_prompt)
        self.assertIn("'확인하세요' 같은 표현으로 대신하지", d2_prompt)

    def test_build_prompt_does_not_force_unrequested_procedure_facets(
        self,
    ) -> None:
        prompt = build_prompt(
            "수료후연구생은 어디서 신청하나요?",
            [{"text": "학생지원시스템에서 온라인으로 신청합니다."}],
        )

        self.assertIn("- 신청·제출·납부 경로와 단계", prompt)
        self.assertNotIn(
            "대상·자격, 신청 기간, 신청 경로·단계, 제출 서류, 예외·문의처",
            prompt,
        )

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
        metadata = result.metadata()
        self.assertEqual(
            metadata["prompt_sha256"],
            hashlib.sha256(build_prompt("질문", ["근거"]).encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            metadata["request_config"]["model_requested"],
            "gemini-configured",
        )
        self.assertEqual(
            metadata["request_config"]["generation_config"],
            handler.requests[0]["body"]["generationConfig"],
        )
        self.assertEqual(len(metadata["request_config_sha256"]), 64)
        self.assertNotIn("gemini-secret", str(metadata))

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
            result.metadata()["request_config"]["model_requested"],
            "gemini-3.5-flash-lite",
        )
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
