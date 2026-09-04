"""서비스 검색 조정(worklog S7)의 단위 테스트.

핵심 보증 두 가지:
1. 기본값(꺼짐)에서 기존 경로와 완전히 동일하게 동작한다 — 벤치마크
   스크립트는 새 인자를 쓰지 않으므로 바이트 불변이어야 한다.
2. 켰을 때 구어체 토큰 제거와 연도·상용구 강등이 설계대로 동작한다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bm25_search import (  # noqa: E402
    SERVICE_QUERY_STOP_TOKENS,
    demote_generic_candidates,
    fts_query,
)


def row(
    chunk_id: str,
    title: str,
    file_name: str = "",
    *,
    text: str = "",
    section_path: list[str] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_title": title,
        "file_name": file_name,
        "text": text,
        "section_path": section_path,
    }


class FtsQueryTests(unittest.TestCase):
    def test_default_behavior_unchanged(self) -> None:
        query = "연구수당은 언제 지급되나요"
        self.assertEqual(fts_query(query), fts_query(query, drop_tokens=None))
        self.assertIn("나요", fts_query(query).split(" OR "))

    def test_drop_tokens_removes_colloquial_terms(self) -> None:
        terms = fts_query(
            "등록금 인상됐나요", drop_tokens=SERVICE_QUERY_STOP_TOKENS
        ).split(" OR ")
        self.assertIn("등록금", terms)
        self.assertIn("인상", terms)
        for stop in ("됐나", "나요", "됐나요"):
            self.assertNotIn(stop, terms)

    def test_drop_tokens_keeps_content_bigrams(self) -> None:
        # 전체 어절과 내용 바이그램은 남아야 한다 (정보 손실 없음).
        terms = fts_query(
            "장학금 신청은 어떻게 하나요", drop_tokens=SERVICE_QUERY_STOP_TOKENS
        ).split(" OR ")
        self.assertIn("장학금", terms)
        self.assertIn("신청", terms)
        self.assertNotIn("어떻게", terms)
        self.assertNotIn("하나요", terms)

    def test_all_tokens_dropped_falls_back_to_original(self) -> None:
        terms = fts_query("어떻게", drop_tokens=SERVICE_QUERY_STOP_TOKENS)
        self.assertTrue(terms)


class DemoteGenericCandidatesTests(unittest.TestCase):
    def test_year_mismatch_demoted_and_order_preserved(self) -> None:
        rows = [
            row("c1", "2017학년도 비용 지급 계획.pdf"),
            row("c2", "2016학년도 비용 지급 계획.pdf"),
            row("c3", "2026학년도 비용 지급계획.pdf"),
            row("c4", "비용 지급 안내"),
        ]
        ordered = demote_generic_candidates("2026학년도 교연비 지급계획", rows)
        self.assertEqual([r["chunk_id"] for r in ordered], ["c3", "c4", "c1", "c2"])

    def test_no_query_year_is_noop(self) -> None:
        rows = [row("c1", "2017학년도 계획"), row("c2", "2026학년도 계획")]
        ordered = demote_generic_candidates("교연비 지급 한도", rows)
        self.assertEqual([r["chunk_id"] for r in ordered], ["c1", "c2"])

    def test_title_with_matching_year_not_demoted(self) -> None:
        rows = [row("c1", "2025~2026 통합 계획"), row("c2", "2024 계획")]
        ordered = demote_generic_candidates("2026 계획", rows)
        self.assertEqual([r["chunk_id"] for r in ordered], ["c1", "c2"])

    def test_generic_titles_demoted(self) -> None:
        rows = [
            row("c1", "FAQs"),
            row("c2", "부산대학교"),
            row("c3", "학위논문 심사 일정 안내.hwp"),
            row("c4", "국문(Korean)"),
        ]
        ordered = demote_generic_candidates("학위논문 심사 일정", rows)
        self.assertEqual(
            [r["chunk_id"] for r in ordered], ["c3", "c1", "c2", "c4"]
        )

    def test_file_name_year_counts(self) -> None:
        rows = [
            row("c1", "회의록 공개", "2016-minutes.pdf"),
            row("c2", "회의록 공개", "2026-minutes.pdf"),
        ]
        ordered = demote_generic_candidates("2026학년도 회의록", rows)
        self.assertEqual([r["chunk_id"] for r in ordered], ["c2", "c1"])

    def test_content_complete_generic_registration_page_beats_weak_title(
        self,
    ) -> None:
        rows = [
            row(
                "weak",
                "2026학년도 재학생 등록금 안내",
                text="등록금 납부 관련 일반 안내입니다.",
            ),
            row(
                "complete",
                "공지사항 내용 > 공지사항 > 공지/참여 | 부산대학교",
                text=(
                    "2026학년도 2학기 재학생 등록금 본등록 납부기간은 "
                    "8월 24일부터 27일까지이며 수납 은행은 농협, 부산은행, "
                    "하나은행, 국민은행, 신한은행, 우리은행입니다."
                ),
                section_path=["등록금", "2학기 본등록 납부 일정 및 은행"],
            ),
        ]

        ordered = demote_generic_candidates(
            "이번 2학기 등록금 본등록은 언제까지 내야 하고, 어느 은행에서 "
            "납부할 수 있나요?",
            rows,
        )

        self.assertEqual([r["chunk_id"] for r in ordered], ["complete", "weak"])

    def test_generic_page_missing_a_core_facet_stays_demoted(self) -> None:
        rows = [
            row(
                "specific",
                "2026학년도 2학기 재학생 등록금 납부 계획",
                text="본등록 일정 및 수납 은행 안내",
            ),
            row(
                "near_match",
                "[다운로드]",
                text=(
                    "2026학년도 2학기 합격자 등록금 납부 기간과 지정 은행 "
                    "안내"
                ),
            ),
        ]

        ordered = demote_generic_candidates(
            "이번 2학기 등록금 본등록은 언제까지 내야 하고 어느 은행에서 "
            "납부할 수 있나요?",
            rows,
        )

        self.assertEqual(
            [r["chunk_id"] for r in ordered], ["specific", "near_match"]
        )

    def test_rare_latin_title_promotes_bidv_procedure_over_minutes(self) -> None:
        rows = [
            row(
                "minutes",
                "2017학년도 제3차 등록금심의위원회 회의록.pdf",
                text="등록금 납부 및 수납 관련 위원회 심의 내용",
            ),
            row(
                "bidv",
                "BIDV 등록금 납부 방법(BIDV-How to Use).pdf",
                text=(
                    "BIDV SMB APP 로그인 후 국제송금에서 한국학비결제를 "
                    "선택하고 납부 대상 학교와 출금 계좌를 입력합니다."
                ),
            ),
        ]

        ordered = demote_generic_candidates(
            "베트남에서 BIDV 은행 앱으로 등록금을 납부하는 절차가 "
            "어떻게 되나요?",
            rows,
        )

        self.assertEqual([r["chunk_id"] for r in ordered], ["bidv", "minutes"])

    def test_procedure_query_demotes_tuition_minutes_without_rare_token(self) -> None:
        rows = [
            row(
                "minutes",
                "2017학년도 제3차 등록금심의위원회 회의록.pdf",
                text="등록금 납부 및 수납 관련 위원회 심의 내용",
            ),
            row(
                "guide",
                "모바일 등록금 납부 안내",
                text="은행 앱 로그인부터 출금 계좌 선택까지의 납부 절차",
            ),
        ]

        ordered = demote_generic_candidates(
            "은행 앱으로 등록금을 납부하는 절차와 방법",
            rows,
        )

        self.assertEqual([r["chunk_id"] for r in ordered], ["guide", "minutes"])

    def test_generic_latin_tokens_do_not_create_a_title_promotion(self) -> None:
        rows = [
            row("answer", "등록금 모바일 납부 안내", text="등록금 납부 방법"),
            row("generic", "PNU PDF APP 안내", text="일반 애플리케이션 안내"),
        ]

        ordered = demote_generic_candidates(
            "PNU PDF app으로 등록금을 납부하는 방법",
            rows,
        )

        self.assertEqual([r["chunk_id"] for r in ordered], ["answer", "generic"])

    def test_rare_title_does_not_override_year_mismatch(self) -> None:
        rows = [
            row("old", "2025학년도 TOPIK 지원 안내"),
            row("current", "2026학년도 외국인 지원 안내"),
        ]

        ordered = demote_generic_candidates(
            "2026학년도 TOPIK 지원 자격",
            rows,
        )

        self.assertEqual([r["chunk_id"] for r in ordered], ["current", "old"])

    def test_registration_minutes_remain_relevant_for_decision_query(self) -> None:
        rows = [
            row(
                "minutes",
                "2026학년도 제1차 등록금심의위원회 회의록.pdf",
                text="학부 등록금 동결, 대학원 등록금 동결",
            ),
            row("guide", "등록금 납부 방법", text="은행별 납부 절차"),
        ]

        ordered = demote_generic_candidates(
            "2026학년도 등록금 심의 결과가 인상인가요 동결인가요?",
            rows,
        )

        self.assertEqual([r["chunk_id"] for r in ordered], ["minutes", "guide"])


if __name__ == "__main__":
    unittest.main()
