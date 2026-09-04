from __future__ import annotations

import importlib.util
import hashlib
import http.client
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_service_answers",
    ROOT / "scripts" / "evaluate_service_answers.py",
)
assert SPEC is not None and SPEC.loader is not None
evaluation_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation_module)


class EvaluateServiceAnswersTests(unittest.TestCase):
    def test_request_retries_only_retryable_http_statuses_with_exponential_backoff(
        self,
    ) -> None:
        calls = 0

        def operation() -> dict:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise HTTPError(
                    "https://example.invalid/chat",
                    503,
                    "unavailable",
                    {},
                    io.BytesIO(),
                )
            return {"answer": "ok"}

        sleeps: list[float] = []
        response, attempts = evaluation_module.call_with_technical_retries(
            operation,
            max_attempts=3,
            backoff_seconds=0.25,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(response, {"answer": "ok"})
        self.assertEqual(calls, 3)
        self.assertEqual(sleeps, [0.25, 0.5])
        self.assertEqual([row["status"] for row in attempts], ["error", "error", "ok"])
        self.assertTrue(attempts[0]["retryable"])

    def test_request_does_not_retry_terminal_http_statuses(self) -> None:
        for status in (400, 401, 403):
            with self.subTest(status=status):
                operation = mock.Mock(
                    side_effect=HTTPError(
                        "https://example.invalid/chat",
                        status,
                        "terminal client error",
                        {},
                        io.BytesIO(),
                    )
                )
                sleeps: list[float] = []

                with self.assertRaises(evaluation_module.RequestFailure) as caught:
                    evaluation_module.call_with_technical_retries(
                        operation,
                        max_attempts=5,
                        backoff_seconds=1.0,
                        sleep_fn=sleeps.append,
                    )

                self.assertEqual(operation.call_count, 1)
                self.assertEqual(sleeps, [])
                self.assertFalse(caught.exception.retryable)
                self.assertEqual(
                    caught.exception.attempts[0]["http_status"], status
                )

    def test_request_retries_network_timeout(self) -> None:
        operation = mock.Mock(side_effect=[TimeoutError("slow"), {"answer": "ok"}])
        sleeps: list[float] = []

        response, attempts = evaluation_module.call_with_technical_retries(
            operation,
            max_attempts=2,
            backoff_seconds=0.1,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(response, {"answer": "ok"})
        self.assertEqual(operation.call_count, 2)
        self.assertEqual(sleeps, [0.1])
        self.assertEqual([row["status"] for row in attempts], ["error", "ok"])

    def test_request_retries_bad_http_status_line(self) -> None:
        operation = mock.Mock(
            side_effect=[http.client.BadStatusLine("broken"), {"answer": "ok"}]
        )

        response, attempts = evaluation_module.call_with_technical_retries(
            operation,
            max_attempts=2,
            backoff_seconds=0.0,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(response, {"answer": "ok"})
        self.assertEqual(operation.call_count, 2)
        self.assertTrue(attempts[0]["retryable"])

    def test_request_does_not_retry_runtime_control_mismatch(self) -> None:
        operation = mock.Mock(side_effect=RuntimeError("provider fallback"))

        with self.assertRaisesRegex(RuntimeError, "provider fallback"):
            evaluation_module.call_with_technical_retries(
                operation,
                max_attempts=4,
                backoff_seconds=0.0,
                sleep_fn=lambda _: None,
            )

        self.assertEqual(operation.call_count, 1)

    def test_request_does_not_misclassify_local_os_error_as_network(self) -> None:
        operation = mock.Mock(side_effect=FileNotFoundError("missing local file"))

        with self.assertRaisesRegex(FileNotFoundError, "missing local file"):
            evaluation_module.call_with_technical_retries(
                operation,
                max_attempts=4,
                backoff_seconds=0.0,
                sleep_fn=lambda _: None,
            )

        self.assertEqual(operation.call_count, 1)

    def test_collection_error_attempts_have_unique_ids_without_answer_ids(self) -> None:
        base = evaluation_module.build_record_base(
            {"id": "case-1", "query": "질문"},
            experiment_id="exp",
            condition_id="c1",
            generation_run_id="run1",
            collector_config={"provider": "gemini"},
        )

        first = evaluation_module.build_collection_error_record(
            base,
            attempt_number=1,
            stage="chat_transport",
            error=TimeoutError("timed out"),
            request_attempts=[],
        )
        second = evaluation_module.build_collection_error_record(
            base,
            attempt_number=2,
            stage="response_control",
            error=RuntimeError("provider fallback"),
            request_attempts=[],
        )

        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertEqual(first["logical_answer_id"], base["answer_id"])
        self.assertNotIn("answer_id", first)
        self.assertEqual(first["id"], first["attempt_id"])
        self.assertEqual(first["record_type"], "answer_collection_error")

    def test_error_attempt_log_does_not_mark_case_completed_on_resume(self) -> None:
        config = {"provider": "gemini"}
        base = evaluation_module.build_record_base(
            {"id": "case-1", "query": "질문"},
            experiment_id="exp",
            condition_id="c1",
            generation_run_id="run1",
            collector_config=config,
        )
        error = evaluation_module.build_collection_error_record(
            base,
            attempt_number=1,
            stage="chat_transport",
            error=TimeoutError("timed out"),
            request_attempts=[],
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.errors.jsonl"
            path.write_text(json.dumps(error, ensure_ascii=False) + "\n", encoding="utf-8")
            counts = evaluation_module.load_error_attempt_counts(
                path,
                experiment_id="exp",
                condition_id="c1",
                generation_run_id="run1",
                collector_config_sha256=base["collector_config_sha256"],
            )

        self.assertEqual(counts, {"case-1": 1})

    def test_validate_answer_payload_rejects_empty_or_whitespace_answer(self) -> None:
        for value in ("", "  \n\t"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "non-empty"):
                    evaluation_module.validate_answer_payload({"answer": value})

    def test_control_mismatch_is_persisted_in_sidecar_then_resume_can_succeed(
        self,
    ) -> None:
        health = {
            "ready": True,
            "parser_profiles": [
                {
                    "id": "cascade",
                    "ready": True,
                    "corpus_revision": "corpus-a",
                    "retrieval_modes": [{"id": "bm25", "ready": True}],
                }
            ],
            "service_config": {
                "retrieval_tuning": True,
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": False,
            },
        }
        success = {
            "answer": "정상 답변",
            "institution": None,
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "generation": {
                "requested": "extractive",
                "used": "extractive",
                "model": None,
            },
            "results": [],
        }
        mismatch = {
            **success,
            "generation": {
                "requested": "extractive",
                "used": "frontier",
                "model": None,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            out = root / "answers.jsonl"
            cases.write_text(
                json.dumps({"id": "case-1", "query": "질문", "k": 5}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "evaluate_service_answers.py",
                "--cases",
                str(cases),
                "--out",
                str(out),
                "--experiment-id",
                "exp",
                "--condition-id",
                "c1",
                "--generation-run-id",
                "run1",
                "--provider",
                "extractive",
                "--parser-profile",
                "cascade",
                "--retrieval-mode",
                "bm25",
                "--allow-unpinned",
                "--max-attempts",
                "1",
                "--sleep",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluation_module, "call_health", return_value=health
            ), mock.patch.object(
                evaluation_module, "call_chat", return_value=mismatch
            ), self.assertRaises(SystemExit) as caught:
                evaluation_module.main()

            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(out.read_text(encoding="utf-8"), "")
            error_path = root / "answers.errors.jsonl"
            errors = [
                json.loads(line)
                for line in error_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["stage"], "response_control")
            self.assertNotIn("answer_id", errors[0])

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluation_module, "call_health", return_value=health
            ), mock.patch.object(
                evaluation_module, "call_chat", return_value=success
            ):
                evaluation_module.main()

            answers = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(answers), 1)
            self.assertEqual(answers[0]["case_id"], "case-1")
            self.assertEqual(answers[0]["collection_attempt_number"], 2)
            self.assertEqual(len(error_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_empty_answer_is_persisted_and_final_run_exits_incomplete(self) -> None:
        health = {
            "ready": True,
            "parser_profiles": [
                {
                    "id": "cascade",
                    "ready": True,
                    "corpus_revision": "corpus-a",
                    "retrieval_modes": [{"id": "bm25", "ready": True}],
                }
            ],
            "service_config": {
                "retrieval_tuning": True,
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": False,
            },
        }
        response = {
            "answer": "  \n",
            "institution": None,
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "generation": {
                "requested": "extractive",
                "used": "extractive",
                "model": None,
            },
            "results": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            out = root / "answers.jsonl"
            cases.write_text(
                json.dumps({"id": "case-1", "query": "질문"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "evaluate_service_answers.py",
                "--cases",
                str(cases),
                "--out",
                str(out),
                "--experiment-id",
                "exp",
                "--condition-id",
                "c1",
                "--generation-run-id",
                "run-empty",
                "--provider",
                "extractive",
                "--parser-profile",
                "cascade",
                "--retrieval-mode",
                "bm25",
                "--allow-unpinned",
                "--max-attempts",
                "1",
                "--sleep",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluation_module, "call_health", return_value=health
            ), mock.patch.object(
                evaluation_module, "call_chat", return_value=response
            ), self.assertRaises(SystemExit) as caught:
                evaluation_module.main()

            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(out.read_text(encoding="utf-8"), "")
            errors = [
                json.loads(line)
                for line in (root / "answers.errors.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["stage"], "response_validation")
            self.assertIn("non-empty", errors[0]["error"]["message"])

    def test_empty_selected_case_set_cannot_succeed(self) -> None:
        health = {
            "ready": True,
            "parser_profiles": [
                {
                    "id": "cascade",
                    "ready": True,
                    "corpus_revision": "corpus-a",
                    "retrieval_modes": [{"id": "bm25", "ready": True}],
                }
            ],
            "service_config": {
                "retrieval_tuning": True,
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "empty.jsonl"
            out = root / "answers.jsonl"
            cases.write_text("", encoding="utf-8")
            argv = [
                "evaluate_service_answers.py",
                "--cases",
                str(cases),
                "--out",
                str(out),
                "--experiment-id",
                "exp",
                "--condition-id",
                "c1",
                "--generation-run-id",
                "run-empty-cases",
                "--provider",
                "extractive",
                "--parser-profile",
                "cascade",
                "--retrieval-mode",
                "bm25",
                "--allow-unpinned",
            ]
            chat = mock.Mock()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                evaluation_module, "call_health", return_value=health
            ), mock.patch.object(
                evaluation_module, "call_chat", chat
            ), self.assertRaises(SystemExit) as caught:
                evaluation_module.main()

            self.assertEqual(caught.exception.code, 2)
            chat.assert_not_called()
            self.assertFalse(out.exists())

    def test_atomic_evidence_hit_accepts_any_evidence_option_per_required_claim(
        self,
    ) -> None:
        case = {
            "required_claims": [
                {
                    "claim_id": "c1",
                    "evidence_options": [
                        {"document_id": "doc-a", "quote": "첫 번째 근거"},
                        {"document_id": "doc-b", "quote": "등록 기간은 9월 3일까지"},
                    ],
                },
                {
                    "claim_id": "c2",
                    "evidence_options": [
                        {"document_id": "doc-c", "quote": "등록금은 10만원"},
                    ],
                },
            ]
        }
        results = [
            {
                "document_id": "doc-b",
                "preview": "안내: 등록 기간은   9월 3일까지 입니다.",
            },
            {"document_id": "doc-c", "preview": "다른 내용"},
        ]

        values = evaluation_module.atomic_evidence_hit(results, case, k=2)

        self.assertTrue(values["available"])
        self.assertEqual(values["found_claim_ids"], ["c1"])
        self.assertEqual(values["missing_claim_ids"], ["c2"])
        self.assertEqual(values["recall"], 0.5)
        self.assertEqual(values["claims"][0]["match"]["option_index"], 1)

    def test_atomic_evidence_uses_untruncated_final_generation_context(self) -> None:
        full_text = "앞부분 " + ("가" * 1000) + " 뒤쪽의 필수 근거"
        response = {
            "results": [
                {
                    "chunk_id": "doc-a#0001",
                    "document_id": "doc-a",
                    "preview": "앞부분만 공개됨",
                }
            ],
            "evaluation_trace": {
                "schema_version": 1,
                "retrieval_stages": {
                    "final_contexts": [
                        {
                            "rank": 1,
                            "stage": "final_contexts",
                            "chunk_id": "doc-a#0001",
                            "document_id": "doc-a",
                            "text": full_text,
                            "text_sha256": hashlib.sha256(
                                full_text.encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                }
            },
        }
        case = {
            "required_claims": [
                {
                    "claim_id": "c1",
                    "evidence_options": [
                        {"document_id": "doc-a", "quote": "뒤쪽의 필수 근거"}
                    ],
                }
            ]
        }

        contexts = evaluation_module.final_evaluation_contexts(response)

        self.assertIsNotNone(contexts)
        values = evaluation_module.atomic_evidence_hit(contexts, case, k=1)
        self.assertEqual(values["recall"], 1.0)

    def test_atomic_evidence_uses_frozen_fuzzy_matcher(self) -> None:
        case = {
            "required_claims": [
                {
                    "claim_id": "deadline",
                    "critical_values": ["9월 3일"],
                    "semantic_anchors": ["등록 신청", "9월 3일"],
                    "evidence_options": [
                        {
                            "document_id": "doc-a",
                            "quote": "등록 신청 마감은 9월 3일까지입니다",
                        }
                    ],
                }
            ]
        }
        results = [
            {
                "document_id": "doc-a",
                "text": "등록 신청 마감일은 9월 3일까지입니다.",
            }
        ]

        values = evaluation_module.atomic_evidence_hit(results, case, k=1)

        self.assertEqual(values["recall"], 1.0)
        self.assertEqual(
            values["claims"][0]["match"]["method"], "document_id+fuzzy"
        )

    def test_atomic_evidence_fails_closed_without_candidate_document_id(self) -> None:
        case = {
            "required_claims": [
                {
                    "claim_id": "c1",
                    "critical_values": ["10만원"],
                    "evidence_options": [
                        {"document_id": "doc-a", "quote": "등록금은 10만원"}
                    ],
                }
            ]
        }

        values = evaluation_module.atomic_evidence_hit(
            [{"text": "등록금은 10만원"}], case, k=1
        )

        self.assertEqual(values["recall"], 0.0)
        self.assertEqual(values["missing_claim_ids"], ["c1"])

    def test_atomic_evidence_preserves_table_relations(self) -> None:
        claim = {
            "claim_id": "fee",
            "critical_values": ["80,000원"],
            "evidence_type": "table",
            "evidence_options": [
                {
                    "document_id": "doc-table",
                    "quote": "학부 수수료 80,000원",
                    "table_evidence": {
                        "headers": ["구분", "수수료"],
                        "row_relations": [
                            {
                                "header": "수수료",
                                "row_anchor": "학부",
                                "value": "80,000원",
                            }
                        ],
                    },
                }
            ],
        }
        case = {"required_claims": [claim]}

        wrong_column = evaluation_module.atomic_evidence_hit(
            [
                {
                    "document_id": "doc-table",
                    "table_headers": ["구분", "수수료", "비고"],
                    "table_rows": [["학부", "90,000원", "80,000원"]],
                }
            ],
            case,
            k=1,
        )
        correct_column = evaluation_module.atomic_evidence_hit(
            [
                {
                    "document_id": "doc-table",
                    "table_headers": ["구분", "수수료"],
                    "table_rows": [["학부", "80,000원"]],
                }
            ],
            case,
            k=1,
        )

        self.assertEqual(wrong_column["recall"], 0.0)
        self.assertEqual(correct_column["recall"], 1.0)
        self.assertEqual(
            correct_column["claims"][0]["match"]["method"],
            "document_id+table",
        )

    def test_final_context_trace_fails_closed_when_text_is_missing(self) -> None:
        response = {
            "results": [{"chunk_id": "c1", "document_id": "d1"}],
            "evaluation_trace": {
                "schema_version": 1,
                "retrieval_stages": {
                    "final_contexts": [
                        {
                            "rank": 1,
                            "stage": "final_contexts",
                            "chunk_id": "c1",
                            "document_id": "d1",
                            "text_sha256": hashlib.sha256(b"").hexdigest(),
                        }
                    ]
                },
            },
        }

        self.assertIsNone(evaluation_module.final_evaluation_contexts(response))

    def test_evidence_hit_obeys_requested_cutoff(self) -> None:
        results = [
            {"chunk_id": f"doc#{index:04d}"}
            for index in range(1, 9)
        ]
        case = {"evidence": [{"chunk_id": "doc#0006"}]}

        at_five = evaluation_module.evidence_hit(results, case, k=5)
        at_eight = evaluation_module.evidence_hit(results, case, k=8)

        self.assertFalse(at_five["matched"])
        self.assertEqual(at_five["recall"], 0.0)
        self.assertTrue(at_eight["matched"])
        self.assertEqual(at_eight["recall"], 1.0)

    def test_evidence_hit_distinguishes_any_from_all_required_evidence(self) -> None:
        results = [{"chunk_id": "doc#0001"}]
        case = {
            "evidence": [
                {"chunk_id": "doc#0001"},
                {"chunk_id": "doc#0002"},
            ]
        }

        values = evaluation_module.evidence_hit(results, case, k=8)

        self.assertTrue(values["matched"])
        self.assertFalse(values["all_matched"])
        self.assertEqual(values["recall"], 0.5)
        self.assertEqual(values["missing"], ["doc#0002"])

    def test_evidence_hit_excludes_optional_answer_evidence(self) -> None:
        results = [{"chunk_id": "doc#0001"}]
        case = {
            "evidence": [
                {"chunk_id": "doc#0001"},
                {
                    "chunk_id": "doc#0002",
                    "required_for_answer": False,
                },
            ]
        }

        values = evaluation_module.evidence_hit(results, case, k=8)

        self.assertTrue(values["all_matched"])
        self.assertEqual(values["recall"], 1.0)
        self.assertEqual(values["wanted"], 1)
        self.assertEqual(values["missing"], [])

    def test_evidence_hit_rejects_non_boolean_answer_requirement(self) -> None:
        case = {
            "evidence": [
                {
                    "chunk_id": "doc#0001",
                    "required_for_answer": "false",
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "required_for_answer"):
            evaluation_module.evidence_hit([], case, k=8)

    def test_build_chat_body_records_all_experimental_controls(self) -> None:
        body = evaluation_module.build_chat_body(
            {"query": "등록금 납부일은?", "role": "재학생"},
            provider="gemini",
            model="gemini-test",
            top_k=8,
            parser_profile="cascade",
            retrieval_mode="bm25",
        )

        self.assertEqual(
            body,
            {
                "question": "등록금 납부일은?",
                "role": "재학생",
                "institution": None,
                "provider": "gemini",
                "model": "gemini-test",
                "top_k": 8,
                "parser_profile": "cascade",
                "retrieval_mode": "bm25",
            },
        )

    def test_build_chat_body_requests_eval_trace_explicitly(self) -> None:
        body = evaluation_module.build_chat_body(
            {"query": "등록금 납부일은?"},
            provider="extractive",
            model=None,
            top_k=8,
            parser_profile="cascade",
            retrieval_mode="bm25",
            eval_trace=True,
        )

        self.assertIs(body["eval_trace"], True)

    def test_validate_response_controls_rejects_provider_fallback(self) -> None:
        response = {
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "generation": {
                "requested": "frontier",
                "used": "extractive",
                "model": None,
            },
            "results": [{"corpus_revision": "corpus-a"}],
        }

        with self.assertRaisesRegex(RuntimeError, "generation provider"):
            evaluation_module.validate_response_controls(
                response,
                provider="gemini",
                model="gemini-test",
                parser_profile="cascade",
                retrieval_mode="bm25",
                expected_corpus_revision="corpus-a",
            )

    def test_validate_response_controls_rejects_mixed_corpus(self) -> None:
        response = {
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "generation": {
                "requested": "extractive",
                "used": "extractive",
                "model": None,
            },
            "results": [
                {"corpus_revision": "corpus-a"},
                {"corpus_revision": "corpus-b"},
            ],
        }

        with self.assertRaisesRegex(RuntimeError, "corpus revision"):
            evaluation_module.validate_response_controls(
                response,
                provider="extractive",
                model=None,
                parser_profile="cascade",
                retrieval_mode="bm25",
                expected_corpus_revision="corpus-a",
            )

    def test_validate_response_controls_rejects_prompt_hash_mismatch(self) -> None:
        response = {
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "generation": {
                "requested": "extractive",
                "used": "extractive",
                "model": None,
                "prompt_sha256": "a" * 64,
                "request_config_sha256": "c" * 64,
            },
            "results": [{"corpus_revision": "corpus-a"}],
            "evaluation_trace": {
                "generation_input": {"prompt_sha256": "b" * 64}
            },
        }

        with self.assertRaisesRegex(RuntimeError, "prompt hash"):
            evaluation_module.validate_response_controls(
                response,
                provider="extractive",
                model=None,
                parser_profile="cascade",
                retrieval_mode="bm25",
                expected_corpus_revision="corpus-a",
                require_eval_trace=True,
            )

    def test_validate_health_controls_pins_tuning_cap_and_revision(self) -> None:
        health = {
            "ready": True,
            "default_parser_profile": "cascade",
            "parser_profiles": [
                {
                    "id": "cascade",
                    "ready": True,
                    "corpus_revision": "corpus-a",
                    "retrieval_modes": [{"id": "bm25", "ready": True}],
                }
            ],
            "service_config": {
                "retrieval_tuning": True,
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": True,
            },
        }

        snapshot = evaluation_module.validate_health_controls(
            health,
            parser_profile="cascade",
            retrieval_mode="bm25",
            expected_corpus_revision="corpus-a",
            expected_retrieval_tuning=True,
            expected_context_chunks_per_document=2,
            require_eval_trace=True,
        )

        self.assertEqual(snapshot["corpus_revision"], "corpus-a")
        self.assertTrue(snapshot["retrieval_tuning"])

        with self.assertRaisesRegex(RuntimeError, "chunk cap"):
            evaluation_module.validate_health_controls(
                health,
                parser_profile="cascade",
                retrieval_mode="bm25",
                expected_corpus_revision="corpus-a",
                expected_retrieval_tuning=True,
                expected_context_chunks_per_document=4,
                require_eval_trace=True,
            )


    def test_validate_health_controls_reads_per_model_generation_caps(self) -> None:
        # /health의 생성 한도는 service_config.generation.models[<model>]에만
        # 있다. DEV 실행에서도 한도 검사를 요청하면 모델명을 넘겨야 하며,
        # 모델명 없이 상위 객체를 대조하면 반드시 mismatch가 나야 한다
        # (2026-09-04 DEV45 생성 preflight 결함의 회귀 테스트).
        health = {
            "ready": True,
            "default_parser_profile": "cascade",
            "parser_profiles": [
                {
                    "id": "cascade",
                    "ready": True,
                    "corpus_revision": "corpus-a",
                    "retrieval_modes": [{"id": "bm25", "ready": True}],
                }
            ],
            "service_config": {
                "retrieval_tuning": True,
                "context_chunks_per_document": 2,
                "evaluation_trace_enabled": True,
                "generation": {
                    "provider": "frontier",
                    "configured": True,
                    "allowed_models": ["gemini-3.5-flash-lite"],
                    "models": {
                        "gemini-3.5-flash-lite": {
                            "max_context_chars": 24000,
                            "max_output_tokens": 900,
                            "sampling_parameters": [],
                        }
                    },
                },
            },
        }

        snapshot = evaluation_module.validate_health_controls(
            health,
            parser_profile="cascade",
            retrieval_mode="bm25",
            expected_corpus_revision="corpus-a",
            expected_retrieval_tuning=True,
            expected_context_chunks_per_document=2,
            expected_generation_model="gemini-3.5-flash-lite",
            expected_generation_max_context_chars=24000,
            expected_generation_max_output_tokens=900,
            expected_generation_sampling_parameters=[],
            require_eval_trace=True,
        )
        self.assertEqual(snapshot["corpus_revision"], "corpus-a")

        with self.assertRaisesRegex(RuntimeError, "max_context_chars mismatch"):
            evaluation_module.validate_health_controls(
                health,
                parser_profile="cascade",
                retrieval_mode="bm25",
                expected_corpus_revision="corpus-a",
                expected_retrieval_tuning=True,
                expected_context_chunks_per_document=2,
                expected_generation_max_context_chars=24000,
                require_eval_trace=True,
            )


if __name__ == "__main__":
    unittest.main()
