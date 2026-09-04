from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.evaluate_grounded_claims_v2 import (
    AUTHORIZATION_PHRASE,
    build_c2_answer,
    collect_planned,
    load_dev_answers,
    main,
    plan_collection,
    reproject_planned,
)
from scripts.rag.grounded_claims_v2 import build_prompt, verify_response
from scripts.service_eval_artifacts import (
    ANSWER_SCHEMA_VERSION,
    build_answer_identity,
    validate_answer_record,
)


class EvaluateGroundedClaimsV2Tests(unittest.TestCase):
    def base_record(self, *, with_contexts: bool = True) -> dict:
        contexts = [
            {
                "rank": 1,
                "stage": "final_contexts",
                "source_number": 1,
                "chunk_id": "doc-tuition#1",
                "document_id": "doc-tuition",
                "source_title": "2026학년도 등록금 심의 결과",
                "source_url": "https://example.edu/tuition",
                "section_path": ["심의 결과"],
                "page": 1,
                "text": "2026학년도 학부 등록금은 동결되었습니다.",
            }
        ]
        record = {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "record_type": "answer",
            "experiment_id": "dev45-generation-current-v1",
            "condition_id": "c1",
            "generation_run_id": "run1",
            "case_id": "svc_test_01",
            "id": "svc_test_01",
            "case_sha256": "a" * 64,
            "query": "2026학년도 학부 등록금은 어떻게 됐나요?",
            "category": "registration",
            "role": None,
            "answer": "기존 C1 답변",
            "evaluation_trace": {
                "schema_version": 1,
                "retrieval_stages": {
                    "final_contexts": contexts if with_contexts else []
                },
            },
        }
        return build_answer_identity(record)

    def write_source(self, directory: Path, record: dict | None = None) -> Path:
        path = directory / "dev45-c1-run1.answers.jsonl"
        value = record or self.base_record()
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def structured_payload(self) -> dict:
        response = {
            "claims": [
                {
                    "text": "2026학년도 학부 등록금은 동결되었습니다.",
                    "evidence": [
                        {
                            "source_number": 1,
                            "quote": "2026학년도 학부 등록금은 동결되었습니다.",
                        }
                    ],
                }
            ],
            "unanswered": [],
        }
        return {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(response, ensure_ascii=False)}]}}
            ]
        }

    def test_dry_run_makes_no_call_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            source = self.write_source(directory)
            out = directory / "c2.answers.jsonl"
            argv = [
                "evaluate_grounded_claims_v2.py",
                "--answers",
                str(source),
                "--out",
                str(out),
                "--experiment-id",
                "dev45-c2-v1",
                "--generation-run-id",
                "run1",
                "--dry-run",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "scripts.evaluate_grounded_claims_v2._post_json"
            ) as post, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(), 0)

            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["external_calls"], 0)
            self.assertEqual(plan["planned_successful_calls"], 1)
            post.assert_not_called()
            self.assertFalse(out.exists())
            self.assertFalse(Path(str(out) + ".summary.json").exists())

    def test_valid_synthetic_collection_builds_judge_compatible_answer(self) -> None:
        base = self.base_record()
        planned, _ = plan_collection(
            [base], model="gemini-test", max_output_tokens=500
        )

        def fake_post(url: str, body: bytes, api_key: str, timeout: float):
            request = json.loads(body)
            self.assertIn("generateContent", url)
            self.assertEqual(api_key, "secret")
            self.assertEqual(timeout, 10.0)
            self.assertEqual(
                request["generationConfig"]["responseMimeType"],
                "application/json",
            )
            return self.structured_payload(), 200

        results = collect_planned(
            planned,
            api_key="secret",
            experiment_id="dev45-c2-v1",
            generation_run_id="run1",
            model="gemini-test",
            max_output_tokens=500,
            timeout=10.0,
            retries=1,
            sleep_seconds=0.0,
            source_artifact_sha256="b" * 64,
            post_json=fake_post,
        )

        self.assertEqual(len(results), 1)
        answer = results[0]
        validate_answer_record(answer)
        self.assertEqual(answer["condition_id"], "c2")
        self.assertNotIn("slot_outcome", answer)
        self.assertNotIn("answer_eligible_for_judge", answer)
        self.assertEqual(answer["postprocessing"]["accepted_claim_count"], 1)
        self.assertEqual(
            answer["evaluation_trace"]["retrieval_stages"],
            base["evaluation_trace"]["retrieval_stages"],
        )

    def test_symlink_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            source = self.write_source(directory)
            link = directory / "dev45-link.answers.jsonl"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "symlink"):
                load_dev_answers(link)

    def test_missing_final_contexts_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            source = self.write_source(
                directory, self.base_record(with_contexts=False)
            )
            with self.assertRaisesRegex(ValueError, "no final_contexts"):
                load_dev_answers(source)

    def test_malformed_model_response_fails_closed(self) -> None:
        base = self.base_record()
        planned, _ = plan_collection(
            [base], model="gemini-test", max_output_tokens=500
        )

        def fake_post(url: str, body: bytes, api_key: str, timeout: float):
            return {
                "candidates": [{"content": {"parts": [{"text": "not json"}]}}]
            }, 200

        with self.assertRaisesRegex(ValueError, "strict JSON"):
            collect_planned(
                planned,
                api_key="secret",
                experiment_id="dev45-c2-v1",
                generation_run_id="run1",
                model="gemini-test",
                max_output_tokens=500,
                timeout=10.0,
                retries=1,
                sleep_seconds=0.0,
                source_artifact_sha256="b" * 64,
                post_json=fake_post,
            )

    def test_offline_reprojection_reuses_raw_response_without_call(self) -> None:
        base = self.base_record()
        planned, _ = plan_collection(
            [base], model="gemini-test", max_output_tokens=500
        )
        first = collect_planned(
            planned,
            api_key="secret",
            experiment_id="dev45-c2-v1",
            generation_run_id="run1",
            model="gemini-test",
            max_output_tokens=500,
            timeout=10.0,
            retries=1,
            sleep_seconds=0.0,
            source_artifact_sha256="b" * 64,
            post_json=lambda *_: (self.structured_payload(), 200),
        )[0]

        projected = reproject_planned(
            planned,
            {first["case_id"]: first},
            experiment_id="dev45-c2-v2",
            generation_run_id="run1-reproject-v2",
            model="gemini-test",
            max_output_tokens=500,
            source_artifact_sha256="b" * 64,
            projection_artifact_sha256="c" * 64,
        )[0]

        validate_answer_record(projected)
        self.assertEqual(projected["answer"], first["answer"])
        self.assertEqual(
            projected["collector_config"]["reprojection"]["new_external_calls"],
            0,
        )
        self.assertEqual(
            projected["generation"]["response_reused_from"]["source_answer_id"],
            first["answer_id"],
        )

    def test_live_mode_requires_exact_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            source = self.write_source(directory)
            out = directory / "c2.answers.jsonl"
            argv = [
                "evaluate_grounded_claims_v2.py",
                "--answers",
                str(source),
                "--out",
                str(out),
                "--experiment-id",
                "dev45-c2-v1",
                "--generation-run-id",
                "run1",
                "--authorize-experimental-collection",
                "not-approved",
            ]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                main()
            self.assertFalse(out.exists())

    def test_build_c2_answer_rejects_tampered_identity_on_validation(self) -> None:
        base = self.base_record()
        contexts = base["evaluation_trace"]["retrieval_stages"]["final_contexts"]
        raw = json.dumps(
            {
                "claims": [
                    {
                        "text": "2026학년도 학부 등록금은 동결되었습니다.",
                        "evidence": [
                            {
                                "source_number": 1,
                                "quote": "2026학년도 학부 등록금은 동결되었습니다.",
                            }
                        ],
                    }
                ],
                "unanswered": [],
            },
            ensure_ascii=False,
        )
        prompt = build_prompt(base["query"], contexts)
        verified = verify_response(raw, contexts)
        answer = build_c2_answer(
            base=base,
            raw_response=raw,
            prompt=prompt,
            verified=verified,
            experiment_id="dev45-c2-v1",
            generation_run_id="run1",
            model="gemini-test",
            max_output_tokens=500,
            attempts=[],
            latency_ms=1.0,
            source_artifact_sha256="b" * 64,
        )
        answer["answer"] = "tampered"
        with self.assertRaisesRegex(ValueError, "answer_sha256 mismatch"):
            validate_answer_record(answer)

    def test_authorization_phrase_is_stable_and_explicit(self) -> None:
        self.assertEqual(AUTHORIZATION_PHRASE, "I_APPROVE_C2_DEV_GENERATION")


if __name__ == "__main__":
    unittest.main()
