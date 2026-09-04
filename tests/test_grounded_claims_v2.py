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

    def test_document_title_can_supply_academic_year_scope(self) -> None:
        response = self.response(
            text="2026학년도 학부 등록금은 동결되었습니다.",
            source=1,
            quote="학부 동결, 대학원 동결",
        )
        result = verify_response(response, self.contexts)

        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(
            result.claims[0]["metadata_supported_critical_values"], ["2026"]
        )

    def test_whole_hour_matches_colon_zero_format(self) -> None:
        contexts = [{"text": "접수 시간은 10:00부터 17:00까지입니다."}]
        response = self.response(
            text="접수 시간은 10시부터 17시까지입니다.",
            source=1,
            quote="접수 시간은 10:00부터 17:00까지입니다.",
        )
        result = verify_response(response, contexts)

        self.assertEqual(result.accepted_count, 1)

    def test_split_date_numbers_match_dotted_source_date(self) -> None:
        contexts = [{"text": "접수기간 26.7.8(수)~22(수) (2주간)"}]
        response = self.response(
            text="접수기간은 26년 7월 8일부터 22일까지 2주간입니다.",
            source=1,
            quote="접수기간 26.7.8(수)~22(수) (2주간)",
        )
        result = verify_response(response, contexts)

        self.assertEqual(result.accepted_count, 1)

    def test_numeric_dense_table_claim_uses_source_title_for_entity_anchor(self) -> None:
        contexts = [
            {
                "source_title": "2025학년도 교육·연구 및 학생지도비 지급 기본계획",
                "text": "교육영역 6,271,600 21.72 6,502,600 21.39 231,000",
            }
        ]
        response = self.response(
            text="2025학년도 교육영역 예산액은 6,502,600이며 구성비는 21.39퍼센트입니다.",
            source=1,
            quote="교육영역 6,271,600 21.72 6,502,600 21.39 231,000",
        )
        result = verify_response(response, contexts)

        self.assertEqual(result.accepted_count, 1)

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
            "relation_marker_mismatch",
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

    def test_login_word_does_not_trigger_price_decrease_marker(self) -> None:
        contexts = [{"text": "온라인 신청은 시스템에 로그인하여 진행합니다."}]
        response = self.response(
            text="온라인 신청은 시스템에 로그인하여 진행합니다.",
            source=1,
            quote="온라인 신청은 시스템에 로그인하여 진행합니다.",
        )
        result = verify_response(response, contexts)

        self.assertEqual(result.accepted_count, 1)

    def test_compound_evidence_token_can_match_split_claim_anchor(self) -> None:
        contexts = [{"text": "선발시기 : 매년 2월, 8월"}]
        response = self.response(
            text="프로그램 선발 시기는 매년 2월과 8월입니다.",
            source=1,
            quote="선발시기 : 매년 2월, 8월",
        )
        result = verify_response(response, contexts)

        self.assertEqual(result.accepted_count, 1)

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

    def test_two_quotes_from_same_source_keep_one_citation_marker(self) -> None:
        payload = {
            "claims": [
                {
                    "text": "2026학년도 학부 등록금은 동결되었습니다.",
                    "evidence": [
                        {"source_number": 1, "quote": "2026학년도 학부"},
                        {"source_number": 1, "quote": "학부 동결, 대학원 동결"},
                    ],
                }
            ],
            "unanswered": [],
        }
        result = verify_response(payload, self.contexts)

        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.claims[0]["source_numbers"], [1])
        self.assertEqual(len(result.claims[0]["citations"]), 2)
        self.assertEqual(result.cited_answer.count("[1]"), 1)

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
