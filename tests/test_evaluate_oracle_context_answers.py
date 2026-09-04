from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load_script(
    "evaluate_oracle_context_answers",
    ROOT / "scripts" / "evaluate_oracle_context_answers.py",
)
judge = load_script(
    "judge_service_answers_for_oracle_test",
    ROOT / "scripts" / "judge_service_answers.py",
)


class OracleContextAnswerTests(unittest.TestCase):
    @staticmethod
    def evidence_option(*, document_id: str, quote: str) -> dict:
        return {
            "document_id": document_id,
            "source_document_family_id": f"family-{document_id}",
            "source_sha256": "a" * 64,
            "source_path": f"raw/{document_id}.pdf",
            "source_url": f"https://example.edu/{document_id}",
            "source_title": f"공식 문서 {document_id}",
            "normalized_title_without_year": f"공식 문서 {document_id}",
            "quote": quote,
        }

    def case(
        self,
        *,
        case_id: str = "h2_core_academic_multi",
        category: str = "academic",
        quote: str = "신청 기간은 2026년 9월 14일부터 9월 18일까지입니다.",
    ) -> dict:
        return {
            "id": case_id,
            "split": oracle.CORE_SPLIT,
            "family_id": f"question-family-{case_id}",
            "category": category,
            "difficulty_type": oracle.DEFAULT_SELECTION_DIFFICULTY,
            "query": "신청 기간은 언제인가요?",
            "role": "pnu-student",
            "answerable": True,
            "required_claims": [
                {
                    "claim_id": "application_period",
                    "description": "신청 기간을 정확히 답한다.",
                    "critical_values": ["2026년 9월 14일", "9월 18일"],
                    "semantic_anchors": ["신청 기간"],
                    "evidence_options": [
                        self.evidence_option(
                            document_id=f"doc-{category}", quote=quote
                        )
                    ],
                }
            ],
            "optional_claims": [],
            "forbidden_claims": [],
            "expected_behavior": "answer",
        }

    def test_selection_is_fixed_to_one_case_per_core_category(self) -> None:
        cases: list[dict] = []
        for category in oracle.CORE_CATEGORIES:
            cases.append(
                self.case(
                    case_id=f"case-{category}-multi", category=category
                )
            )
            distractor = self.case(
                case_id=f"case-{category}-single", category=category
            )
            distractor["difficulty_type"] = "single_fact"
            cases.append(distractor)

        selected = oracle.select_oracle_cases(cases)

        self.assertEqual(len(selected), 9)
        self.assertEqual(
            [case["category"] for case in selected],
            list(oracle.CORE_CATEGORIES),
        )
        self.assertTrue(
            all(
                case["difficulty_type"] == oracle.DEFAULT_SELECTION_DIFFICULTY
                for case in selected
            )
        )

    def test_selection_fails_closed_on_missing_or_duplicate_category(self) -> None:
        cases = [
            self.case(case_id=f"case-{category}", category=category)
            for category in oracle.CORE_CATEGORIES
        ]
        with self.assertRaisesRegex(ValueError, "missing"):
            oracle.select_oracle_cases(cases[:-1])

        duplicate = self.case(
            case_id="duplicate-academic", category="academic"
        )
        with self.assertRaisesRegex(ValueError, "multiple"):
            oracle.select_oracle_cases([*cases, duplicate])

    def test_contexts_use_only_quotes_and_dedupe_shared_evidence(self) -> None:
        case = self.case()
        first_claim = case["required_claims"][0]
        shared = dict(first_claim["evidence_options"][0])
        case["required_claims"].append(
            {
                "claim_id": "same_source_second_claim",
                "description": "이 설명문은 generator context에 들어가면 안 된다.",
                "critical_values": ["9월 18일"],
                "semantic_anchors": ["신청 기간"],
                "evidence_options": [shared],
            }
        )

        contexts = oracle.build_oracle_contexts(case)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["text"], shared["quote"])
        self.assertNotIn("설명문", contexts[0]["text"])
        bindings = contexts[0]["gold_evidence_provenance"]
        self.assertEqual(
            [binding["claim_id"] for binding in bindings],
            ["application_period", "same_source_second_claim"],
        )
        self.assertTrue(all(len(binding["evidence_option_sha256"]) == 64 for binding in bindings))

    def test_extractive_smoke_builds_immutable_judge_compatible_answer(self) -> None:
        case = self.case()
        exact_answer = case["required_claims"][0]["evidence_options"][0]["quote"]

        record = oracle.collect_oracle_answer(
            case,
            experiment_id="oracle-fixture-exp",
            condition_id="c1-oracle-context",
            generation_run_id="fixture-run1",
            provider="extractive",
            model=None,
            collector_config={
                "fixture": True,
                "retrieval_bypassed": True,
                "service_performance_eligible": False,
            },
            extractive_fallback=lambda _question, _contexts: exact_answer,
            collected_at="2026-09-02T00:00:00+00:00",
        )

        oracle.validate_answer_record(record)
        judge.validate_answer_case(record, case)
        diagnostic = record["oracle_diagnostic"]
        self.assertTrue(diagnostic["diagnostic_only"])
        self.assertTrue(diagnostic["retrieval_bypassed"])
        self.assertFalse(diagnostic["service_performance_eligible"])
        self.assertEqual(diagnostic["provider_adapter_requested"], "extractive")
        self.assertIsNone(diagnostic["model_used"])
        self.assertEqual(
            diagnostic["request_config_sha256"],
            oracle.sha256_json(diagnostic["request_config"]),
        )
        self.assertEqual(
            diagnostic["sampling_parameters"]["seed"]["status"],
            "unsupported_not_sent",
        )
        self.assertEqual(
            diagnostic["gold_evidence_provenance_sha256"],
            oracle.sha256_json(diagnostic["gold_evidence_provenance"]),
        )
        self.assertEqual(
            diagnostic["prompt_sha256"],
            record["evaluation_trace"]["generation_input"]["prompt_sha256"],
        )
        self.assertEqual(record["retrieval"]["mode"], "oracle_gold_evidence")
        self.assertFalse(record["retrieval"]["service_performance_eligible"])
        self.assertTrue(record["answer"].strip())
        self.assertEqual(len(record["evaluation_trace"]["retrieval_stages"]["final_contexts"]), 1)
        self.assertEqual(
            diagnostic["gold_evidence_provenance"][0]["bindings"][0]["claim_id"],
            "application_period",
        )

    def test_generation_is_not_called_when_context_limit_drops_gold_quote(self) -> None:
        case = self.case(quote="가" * 500)
        generate_fn = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "truncated"):
            oracle.collect_oracle_answer(
                case,
                experiment_id="oracle-fixture-exp",
                condition_id="c1-oracle-context",
                generation_run_id="fixture-run1",
                provider="extractive",
                model=None,
                collector_config={"fixture": True},
                max_context_chars=100,
                generate_fn=generate_fn,
                extractive_fallback="unused",
            )

        generate_fn.assert_not_called()

    def test_mocked_gemini_preserves_public_and_adapter_provider_and_model(self) -> None:
        case = self.case()
        model = "gemini-3.5-flash-lite"
        exact_answer = case["required_claims"][0]["evidence_options"][0]["quote"]

        def fake_generate(
            question,
            contexts,
            *,
            requested,
            extractive_fallback,
            requested_model,
            role_perspective,
        ):
            del extractive_fallback
            trace = oracle.generation_input_trace(
                question, list(contexts), role_perspective=role_perspective
            )
            request_config = {
                "provider": "gemini",
                "model_requested": requested_model,
                "api_style": "generateContent",
                "generation_config": {"maxOutputTokens": 900},
                "prompt_used": True,
                "system_instruction_sha256": trace[
                    "system_instruction_sha256"
                ],
            }
            return oracle.GenerationResult(
                text=exact_answer,
                requested=requested,
                used="gemini",
                model=requested_model,
                fallback_reason=None,
                attempts=(),
                prompt_sha256=trace["prompt_sha256"],
                system_instruction_sha256=trace[
                    "system_instruction_sha256"
                ],
                request_config=request_config,
                request_config_sha256=oracle.sha256_json(request_config),
            )

        record = oracle.collect_oracle_answer(
            case,
            experiment_id="oracle-mock-exp",
            condition_id="c1-oracle-context",
            generation_run_id="mock-run1",
            provider="gemini",
            model=model,
            collector_config={"fixture": "mocked-gemini"},
            generate_fn=fake_generate,
            collected_at="2026-09-02T00:00:00+00:00",
        )

        self.assertEqual(record["generation"]["requested"], "frontier")
        self.assertEqual(record["generation"]["used"], "frontier")
        self.assertEqual(record["generation"]["implementation"], "gemini")
        diagnostic = record["oracle_diagnostic"]
        self.assertEqual(diagnostic["provider_input"], "gemini")
        self.assertEqual(diagnostic["provider_public_requested"], "frontier")
        self.assertEqual(diagnostic["provider_adapter_requested"], "gemini")
        self.assertEqual(diagnostic["model_requested"], model)
        self.assertEqual(diagnostic["model_used"], model)
        self.assertEqual(
            diagnostic["sampling_parameters"]["temperature"]["status"],
            "not_sent",
        )

    def test_draft_path_is_rejected_before_any_holdout_gate_or_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdout-v2.draft.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(oracle, "validate_paths") as validator:
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    oracle.require_frozen_holdout(
                        path,
                        [Path(directory) / "dev.json"],
                        corpus_index=Path(directory) / "corpus.sqlite",
                        signoff_path=Path(directory) / "signoff.json",
                    )
            validator.assert_not_called()

    def test_failed_signoff_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdout-v2.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                oracle,
                "validate_paths",
                return_value={
                    "ok": False,
                    "gates": {
                        "schema": {"ok": True},
                        "dev": {"ok": True},
                        "corpus": {"ok": True},
                        "signoff": {"ok": False},
                    },
                    "errors": ["signoff: two reviewers required"],
                },
            ):
                with self.assertRaisesRegex(ValueError, "two reviewers"):
                    oracle.require_frozen_holdout(
                        path,
                        [Path(directory) / "dev.json"],
                        corpus_index=Path(directory) / "corpus.sqlite",
                        signoff_path=Path(directory) / "signoff.json",
                    )

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oracle.answers.jsonl"
            output.write_text(json.dumps({"kept": True}) + "\n", encoding="utf-8")

            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    oracle.main(
                        [
                            "--out",
                            str(output),
                            "--experiment-id",
                            "exp",
                            "--generation-run-id",
                            "run1",
                            "--provider",
                            "gemini",
                            "--model",
                            "gemini-3.5-flash-lite",
                        ]
                    )

            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["kept"], True
            )


if __name__ == "__main__":
    unittest.main()
