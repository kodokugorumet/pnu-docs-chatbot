"""Parser-neutral extraction quality metrics and routing gates."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import Attempt, Block


@dataclass(frozen=True)
class QualityMetrics:
    total_chars: int
    non_whitespace_chars: int
    letter_chars: int
    hangul_chars: int
    hangul_letter_ratio: float
    control_chars: int
    control_ratio: float
    mojibake_chars: int
    mojibake_ratio: float
    block_count: int
    table_count: int
    table_cell_count: int
    pages_with_output: int
    expected_pages: Optional[int]
    page_coverage: Optional[float]
    reading_order_contiguous: bool
    ocr_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chars": self.total_chars,
            "non_whitespace_chars": self.non_whitespace_chars,
            "letter_chars": self.letter_chars,
            "hangul_chars": self.hangul_chars,
            "hangul_letter_ratio": self.hangul_letter_ratio,
            "control_chars": self.control_chars,
            "control_ratio": self.control_ratio,
            "mojibake_chars": self.mojibake_chars,
            "mojibake_ratio": self.mojibake_ratio,
            "block_count": self.block_count,
            "table_count": self.table_count,
            "table_cell_count": self.table_cell_count,
            "pages_with_output": self.pages_with_output,
            "expected_pages": self.expected_pages,
            "page_coverage": self.page_coverage,
            "reading_order_contiguous": self.reading_order_contiguous,
            "ocr_confidence": self.ocr_confidence,
        }


@dataclass(frozen=True)
class QualityAssessment:
    decision: str
    score: float
    hard_fail_reasons: Sequence[str]
    suspect_reasons: Sequence[str]
    metrics: QualityMetrics

    @property
    def hard_fail(self) -> bool:
        return bool(self.hard_fail_reasons)

    @property
    def suspect(self) -> bool:
        return not self.hard_fail and bool(self.suspect_reasons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "score": self.score,
            "hard_fail_reasons": list(self.hard_fail_reasons),
            "suspect_reasons": list(self.suspect_reasons),
            "metrics": self.metrics.to_dict(),
        }


def calculate_quality_metrics(
    text: Optional[str] = None,
    blocks: Iterable[Block] = (),
    expected_page_count: Optional[int] = None,
    ocr_confidence: Optional[float] = None,
) -> QualityMetrics:
    block_list = list(blocks)
    if text is None:
        # A table parent already contains all cell text. Avoid counting canonical
        # table_cell blocks twice when evaluating a full normalized result.
        text = "\n".join(
            block.text for block in block_list if block.block_type != "table_cell"
        )
    total_chars = len(text)
    non_whitespace_chars = sum(not character.isspace() for character in text)
    letter_chars = sum(character.isalpha() for character in text)
    hangul_chars = sum(_is_hangul(character) for character in text)
    control_chars = sum(
        unicodedata.category(character) == "Cc"
        and character not in {"\n", "\r", "\t"}
        for character in text
    )
    mojibake_chars = sum(_is_mojibake(character) for character in text)
    pages = {
        block.page
        for block in block_list
        if block.page is not None and block.text.strip()
    }
    orders = [block.reading_order for block in block_list]
    contiguous = (
        not orders
        or len(set(orders)) == len(orders)
        and sorted(orders) == list(range(len(orders)))
    )
    page_coverage = None
    if expected_page_count is not None:
        if expected_page_count < 0:
            raise ValueError("expected_page_count must be non-negative")
        page_coverage = (
            1.0
            if expected_page_count == 0
            else min(1.0, len(pages) / float(expected_page_count))
        )
    if ocr_confidence is not None and not 0.0 <= ocr_confidence <= 1.0:
        raise ValueError("ocr_confidence must be between zero and one")
    denominator = max(1, total_chars)
    return QualityMetrics(
        total_chars=total_chars,
        non_whitespace_chars=non_whitespace_chars,
        letter_chars=letter_chars,
        hangul_chars=hangul_chars,
        hangul_letter_ratio=round(
            hangul_chars / float(max(1, letter_chars)), 6
        ),
        control_chars=control_chars,
        control_ratio=round(control_chars / float(denominator), 6),
        mojibake_chars=mojibake_chars,
        mojibake_ratio=round(mojibake_chars / float(denominator), 6),
        block_count=len(block_list),
        table_count=sum(block.block_type == "table" for block in block_list),
        table_cell_count=sum(
            block.block_type == "table_cell" for block in block_list
        ),
        pages_with_output=len(pages),
        expected_pages=expected_page_count,
        page_coverage=(
            round(page_coverage, 6) if page_coverage is not None else None
        ),
        reading_order_contiguous=contiguous,
        ocr_confidence=ocr_confidence,
    )


def assess_quality(
    text: Optional[str] = None,
    blocks: Iterable[Block] = (),
    expected_page_count: Optional[int] = None,
    table_signal: bool = False,
    expect_korean: bool = False,
    ocr_confidence: Optional[float] = None,
    attempt: Optional[Attempt] = None,
    schema_errors: Iterable[str] = (),
    min_non_whitespace_chars: int = 20,
) -> QualityAssessment:
    block_list = list(blocks)
    metrics = calculate_quality_metrics(
        text=text,
        blocks=block_list,
        expected_page_count=expected_page_count,
        ocr_confidence=ocr_confidence,
    )
    hard_fail: List[str] = []
    suspect: List[str] = []

    if attempt is not None and (
        attempt.status in {"error", "timeout"} or attempt.timeout
    ):
        hard_fail.append("adapter_{}".format(
            "timeout" if attempt.status == "timeout" or attempt.timeout else "error"
        ))
    for error in schema_errors:
        hard_fail.append("schema_invalid:{}".format(error))
    if metrics.non_whitespace_chars < min_non_whitespace_chars:
        hard_fail.append(
            "below_min_chars:{}/{}".format(
                metrics.non_whitespace_chars, min_non_whitespace_chars
            )
        )
    if metrics.control_ratio > 0.01:
        hard_fail.append("control_ratio:{:.6f}".format(metrics.control_ratio))

    if metrics.mojibake_ratio > 0.005:
        suspect.append("mojibake_ratio:{:.6f}".format(metrics.mojibake_ratio))
    if (
        expected_page_count is not None
        and expected_page_count > 0
        and metrics.pages_with_output < expected_page_count
    ):
        suspect.append(
            "page_output_missing:{}/{}".format(
                metrics.pages_with_output, expected_page_count
            )
        )
    if table_signal and metrics.table_count == 0:
        suspect.append("table_signal_without_table_block")
    if not metrics.reading_order_contiguous:
        suspect.append("reading_order_noncontiguous")
    if (
        expect_korean
        and metrics.letter_chars > 0
        and metrics.hangul_letter_ratio < 0.2
    ):
        suspect.append(
            "low_hangul_ratio:{:.6f}".format(metrics.hangul_letter_ratio)
        )

    if hard_fail:
        decision = "hard_fail"
        score = 0.0
    else:
        decision = "suspect" if suspect else "pass"
        score = 100.0
        score -= min(35.0, metrics.mojibake_ratio * 1000.0)
        if metrics.page_coverage is not None:
            score -= (1.0 - metrics.page_coverage) * 30.0
        if table_signal and metrics.table_count == 0:
            score -= 20.0
        if not metrics.reading_order_contiguous:
            score -= 20.0
        if expect_korean and metrics.letter_chars:
            score -= max(0.0, 0.2 - metrics.hangul_letter_ratio) * 100.0
        score = round(max(0.0, score), 3)

    return QualityAssessment(
        decision=decision,
        score=score,
        hard_fail_reasons=tuple(hard_fail),
        suspect_reasons=tuple(suspect),
        metrics=metrics,
    )


def _is_hangul(character: str) -> bool:
    codepoint = ord(character)
    return (
        0xAC00 <= codepoint <= 0xD7A3
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    )


def _is_mojibake(character: str) -> bool:
    codepoint = ord(character)
    return (
        character == "\ufffd"
        or 0xE000 <= codepoint <= 0xF8FF
        or 0xF0000 <= codepoint <= 0xFFFFD
        or 0x100000 <= codepoint <= 0x10FFFD
    )
