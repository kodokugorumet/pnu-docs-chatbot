from __future__ import annotations

import json
import unittest

from scripts.rag.grounded_claims_v2 import (
    GroundedClaimsError,
    build_prompt,
    parse_response,
    verify_response,
)


class GroundedClaimsV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.contexts = [
            {
                "chunk_id": "doc-tuition#4",
                "document_id": "doc-tuition",
                "source_title": "2026학년도 등록금심의위원회 회의록",
                "source_url": "https://example.edu/tuition",
                "section_path": ["심의 결과"],
                "page": 1,
                "text": (
                    "2026학년도 학부 및 대학원 등록금\n학부 동결, 대학원 동결\n"
                    "신용카드는 등록금 전액 납부 시에만 사용 가능"
                ),
            },
            {
                "chunk_id": "doc-admission#2",
                "document_id": "doc-admission",
                "source_title": "입학 전형 일정",
                "source_url": "https://example.edu/admission",
                "section_path": ["전형 일정"],
                "page": 2,
                "text": "원서접수 마감일 18:00까지 전형료를 납부해야 합니다.",
            },
        ]

    def response(self, *, text: str, source: int, quote: str) -> str:
        return json.dumps(
            {
                "claims": [
                    {
                        "text": text,
                        "evidence": [
                            {"source_number": source, "quote": quote}
                        ],
                    }
                ],
                "unanswered": [],
            },
            ensure_ascii=False,
        )

    def test_valid_quote_bound_claim_is_kept_with_citation(self) -> None:
        response = self.response(
            text="2026학년도 학부와 대학원 등록금은 모두 동결되었습니다.",
            source=1,
            quote="2026학년도 학부 및 대학원 등록금 학부 동결, 대학원 동결",
        )
        result = verify_response(response, self.contexts)

        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.rejected_count, 0)
        self.assertIn("동결되었습니다", result.answer)
        self.assertIn("[1]", result.cited_answer)
        self.assertEqual(result.citations[0]["chunk_id"], "doc-tuition#4")

    def test_unknown_source_and_fabricated_quote_fail_closed(self) -> None:
        unknown = self.response(
            text="2026학년도 등록금은 동결되었습니다.",
            source=9,
            quote="2026학년도 등록금은 동결되었습니다.",
        )
        missing = self.response(
            text="2026학년도 등록금은 동결되었습니다.",
            source=1,
            quote="등록금은 전액 인하되었습니다.",
        )

        unknown_result = verify_response(unknown, self.contexts)
        missing_result = verify_response(missing, self.contexts)
        self.assertEqual(
            unknown_result.claims[0]["validation_reason"],
            "unknown_source_number",
        )
        self.assertEqual(
            missing_result.claims[0]["validation_reason"],
            "quote_not_in_source",
        )
        self.assertEqual(unknown_result.accepted_count, 0)
        self.assertEqual(missing_result.accepted_count, 0)

    def test_numeric_value_must_exist_in_quote(self) -> None:
        response = self.response(
            text="원서접수 마감은 16:00입니다.",
            source=2,
            quote="원서접수 마감일 18:00까지 전형료를 납부해야 합니다.",
        )
        result = verify_response(response, self.contexts)

        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(
            result.claims[0]["validation_reason"],
            "critical_value_not_in_quote",
        )
        self.assertEqual(result.claims[0]["missing_critical_values"], ["16:00"])

    def test_unrelated_time_does_not_support_credit_card_claim(self) -> None:
        response = self.response(
            text="신용카드 납부는 매일 18:00까지 가능합니다.",
            source=2,
            quote="원서접수 마감일 18:00까지 전형료를 납부해야 합니다.",
        )
        result = verify_response(response, self.contexts)

        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(
            result.claims[0]["validation_reason"],
            "insufficient_anchor_overlap",
        )

    def test_relation_direction_must_match_quote(self) -> None:
        contexts = [{"text": "2026학년도 학부 등록금은 인상되었습니다."}]
        response = self.response(
            text="2026학년도 학부 등록금은 동결되었습니다.",
            source=1,
            quote="2026학년도 학부 등록금은 인상되었습니다.",
        )
        result = verify_response(response, contexts)

        self.assertEqual(result.accepted_count, 0)
        self.assertEqual(
            result.claims[0]["validation_reason"],
            "relation_marker_mismatch",
        )

    def test_partial_answer_keeps_supported_claim_and_explicit_gap(self) -> None:
        payload = {
            "claims": [
                {
                    "text": "학부와 대학원 등록금은 모두 동결되었습니다.",
                    "evidence": [
                        {
                            "source_number": 1,
                            "quote": "학부 및 대학원 등록금 학부 동결, 대학원 동결",
                        }
                    ],
                }
            ],
            "unanswered": [
                "제공된 문서에서 등록금 인상률은 확인할 수 없습니다."
            ],
        }
        result = verify_response(payload, self.contexts)

        self.assertEqual(result.accepted_count, 1)
        self.assertIn("모두 동결", result.answer)
        self.assertIn("인상률은 확인할 수 없습니다", result.answer)

    def test_strict_schema_rejects_markdown_and_extra_keys(self) -> None:
        fenced = "```json\n{\"claims\": [], \"unanswered\": []}\n```"
        with self.assertRaises(GroundedClaimsError):
            parse_response(fenced)
        with self.assertRaises(GroundedClaimsError):
            parse_response({"claims": [], "unanswered": [], "score": 2})

    def test_prompt_is_bounded_and_requires_exact_quote(self) -> None:
        prompt = build_prompt(
            "등록금은 동결됐나요?",
            [{"chunk_id": "doc#1", "text": "가" * 500}],
            max_context_chars=120,
        )

        context = prompt.split("<검색_근거_시작>\n", 1)[1].split(
            "\n<검색_근거_끝>", 1
        )[0]
        self.assertLessEqual(len(context), 120)
        self.assertIn("원문에서 그대로 복사", prompt)
        self.assertIn("source_number", prompt)


if __name__ == "__main__":
    unittest.main()
