from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "service_eval_artifacts", ROOT / "scripts" / "service_eval_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
artifacts_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts_module)


class ServiceEvalArtifactsTests(unittest.TestCase):
    def write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def answer_record(self, *, answer: str = "정답") -> dict:
        return artifacts_module.build_answer_identity(
            {
                "schema_version": artifacts_module.ANSWER_SCHEMA_VERSION,
                "record_type": "answer",
                "experiment_id": "exp-20260901",
                "condition_id": "c1",
                "generation_run_id": "run1",
                "case_id": "svc_reg_01",
                "id": "svc_reg_01",
                "answer": answer,
            }
        )

    def test_answer_identity_is_stable_and_bound_to_answer_text(self) -> None:
        first = self.answer_record(answer="동일 답변")
        second = self.answer_record(answer="동일 답변")

        self.assertEqual(first["answer_id"], second["answer_id"])
        self.assertEqual(first["answer_sha256"], second["answer_sha256"])
        self.assertEqual(len(first["answer_sha256"]), 64)

        changed = dict(first, answer="변경 답변")
        with self.assertRaisesRegex(ValueError, "answer_sha256"):
            artifacts_module.validate_answer_record(changed)

    def test_answer_id_changes_when_run_identity_changes(self) -> None:
        first = self.answer_record()
        second = self.answer_record()
        second.pop("answer_id")
        second.pop("answer_sha256")
        second["generation_run_id"] = "run2"
        second = artifacts_module.build_answer_identity(second)

        self.assertNotEqual(first["answer_id"], second["answer_id"])

    def test_answer_record_hash_detects_non_answer_payload_change(self) -> None:
        answer = self.answer_record()
        answer["citations"] = [{"source_number": 99}]

        with self.assertRaisesRegex(ValueError, "record_sha256"):
            artifacts_module.validate_answer_record(answer)

    def test_join_judgments_rejects_answer_hash_mismatch(self) -> None:
        answer = self.answer_record()
        judgment = artifacts_module.build_judgment_identity(
            {
                "schema_version": artifacts_module.JUDGMENT_SCHEMA_VERSION,
                "record_type": "judgment",
                "experiment_id": answer["experiment_id"],
                "judge_run_id": "judge1",
                "answer_id": answer["answer_id"],
                "answer_sha256": "0" * 64,
                "case_id": answer["case_id"],
                "judge": {"score": 2, "reason": "ok"},
            }
        )

        with self.assertRaisesRegex(ValueError, "answer_sha256 mismatch"):
            artifacts_module.join_answers_and_judgments(
                {answer["case_id"]: answer}, [judgment]
            )

    def test_join_judgments_requires_exactly_one_per_answer(self) -> None:
        answer = self.answer_record()

        with self.assertRaisesRegex(ValueError, "missing judgment"):
            artifacts_module.join_answers_and_judgments(
                {answer["case_id"]: answer}, []
            )

    def test_load_unique_jsonl_rejects_duplicate_stable_keys(self) -> None:
        answer = self.answer_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            self.write_jsonl(path, [answer, answer])

            with self.assertRaisesRegex(ValueError, "duplicate answer_id"):
                artifacts_module.load_unique_jsonl(path, key="answer_id")

    def test_final_retrieval_accepts_exact_extractive_generator_metadata(self) -> None:
        prompt = "질문"
        system = "지침"
        request_config = {
            "provider": "extractive",
            "model_requested": None,
            "prompt_used": False,
        }
        generation = {
            "requested": "extractive",
            "used": "extractive",
            "model": None,
            "request_config": request_config,
            "request_config_sha256": artifacts_module.sha256_json(request_config),
            "prompt_sha256": artifacts_module.sha256_text(prompt),
            "system_instruction_sha256": artifacts_module.sha256_text(system),
        }
        record = {
            "generation": generation,
            "evaluation_trace": {
                "schema_version": 1,
                "generation_input": {
                    "user_prompt": prompt,
                    "system_instruction": system,
                    "prompt_sha256": generation["prompt_sha256"],
                    "system_instruction_sha256": generation[
                        "system_instruction_sha256"
                    ],
                },
                "raw_draft": "초안",
                "sanitized_draft": "초안",
                "retrieval_stages": {"final_contexts": []},
            },
        }

        artifacts_module.validate_final_retrieval_provenance(record)
        record["generation"]["request_config"] = dict(
            request_config, prompt_used=True
        )
        with self.assertRaisesRegex(ValueError, "request_config mismatch"):
            artifacts_module.validate_final_retrieval_provenance(record)

    def test_strict_generation_requires_terminal_slot_fields(self) -> None:
        record = {
            "request_attempts": [
                {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
            ]
        }
        with self.assertRaisesRegex(ValueError, "slot_outcome"):
            artifacts_module.validate_final_generation_provenance(
                record,
                provider="frontier",
                model="gemini-3.5-flash-lite",
                max_output_tokens=900,
                require_collection_attempts=True,
            )

    def test_strict_generation_rejects_unapproved_retry_histories(self) -> None:
        base = {
            "slot_outcome": "answer",
            "answer_eligible_for_judge": True,
            "collection_attempt_number": 1,
        }
        retry = {
            "attempt_number": 1,
            "status": "error",
            "elapsed_ms": 1.0,
            "error_type": "TimeoutError",
            "error": "timeout",
            "retryable": True,
            "http_status": None,
        }
        ok = {"attempt_number": 2, "status": "ok", "elapsed_ms": 1.0}
        cases = (
            (
                {**base, "request_attempts": [retry, retry, retry, retry]},
                "1..3 rows",
            ),
            (
                {
                    **base,
                    "request_attempts": [
                        {**retry, "error_type": "ValueError"},
                        ok,
                    ],
                },
                "not a network error",
            ),
            (
                {
                    **base,
                    "request_attempts": [
                        {
                            **retry,
                            "error_type": "HTTPError",
                            "http_status": 400,
                        },
                        ok,
                    ],
                },
                "HTTP status/type",
            ),
        )
        for record, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                artifacts_module.validate_final_generation_provenance(
                    record,
                    provider="frontier",
                    model="gemini-3.5-flash-lite",
                    max_output_tokens=900,
                    require_collection_attempts=True,
                )

    def test_response_validation_terminal_allows_prior_retryable_attempt(self) -> None:
        attempts = [
            {
                "attempt_number": 1,
                "status": "error",
                "elapsed_ms": 1.0,
                "error_type": "TimeoutError",
                "error": "timeout",
                "retryable": True,
                "http_status": None,
            },
            {"attempt_number": 2, "status": "ok", "elapsed_ms": 2.0},
        ]
        record = {
            "answer": "[SERVICE_ERROR]",
            "slot_outcome": "service_error",
            "answer_eligible_for_judge": False,
            "collection_attempt_number": 1,
            "request_attempts": attempts,
            "service_error": {
                "stage": "response_validation",
                "type": "AnswerPayloadError",
                "message": "response answer must be a non-empty string",
                "retryable": False,
                "http_status": None,
                "request_attempts": attempts,
            },
        }
        artifacts_module.validate_final_generation_provenance(
            record,
            provider="frontier",
            model="gemini-3.5-flash-lite",
            max_output_tokens=900,
            require_collection_attempts=True,
        )


if __name__ == "__main__":
    unittest.main()
