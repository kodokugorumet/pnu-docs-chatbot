from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.document_parsing.core import (
    BLOCK_FIELDS,
    Attempt,
    Block,
    ParseResult,
    SourceDocument,
    assess_quality,
    blocks_to_legacy_chunks,
    clean_text,
    make_block_id,
    make_document_id,
    make_table_id,
    read_jsonl,
    write_jsonl,
)


class BlockSchemaTests(unittest.TestCase):
    def test_block_serializes_exact_eleven_fields_including_nulls(self) -> None:
        block = make_block(
            0,
            "heading",
            "졸업 과제",
            page=1,
            section_path=["졸업 과제"],
        )

        value = block.to_dict()

        self.assertEqual(tuple(value), BLOCK_FIELDS)
        self.assertEqual(len(value), 11)
        self.assertIsNone(value["table_id"])
        self.assertIsNone(value["row"])
        self.assertIsNone(value["column"])
        self.assertEqual(Block.from_dict(value), block)

    def test_table_parent_and_cells_validate_coordinates(self) -> None:
        table_id = make_table_id("doc_test", 0, "baseline")
        parent = make_block(
            0,
            "table",
            "항목\t값",
            table_id=table_id,
        )
        cell = make_block(
            1,
            "table_cell",
            "항목",
            table_id=table_id,
            row=0,
            column=0,
        )

        self.assertEqual(parent.table_id, cell.table_id)
        with self.assertRaisesRegex(ValueError, "require row and column"):
            make_block(2, "table_cell", "잘못된 셀", table_id=table_id)
        with self.assertRaisesRegex(ValueError, "only valid for table_cell"):
            make_block(2, "paragraph", "본문", row=0, column=0)
        with self.assertRaisesRegex(ValueError, "one-based"):
            make_block(2, "paragraph", "본문", page=0)

    def test_parse_result_rejects_noncontiguous_order(self) -> None:
        source = make_source()
        result = ParseResult(
            source,
            blocks=[
                make_block(0, "paragraph", "첫 번째 문단입니다."),
                make_block(2, "paragraph", "세 번째 문단입니다."),
            ],
            attempts=[Attempt("unit", "success")],
        )

        with self.assertRaisesRegex(ValueError, "contiguous"):
            result.validate()


class IdentifierAndTextTests(unittest.TestCase):
    def test_document_id_normalizes_unicode_and_separators(self) -> None:
        digest = hashlib.sha256(b"same bytes").hexdigest()
        composed = make_document_id("기관/가.pdf", digest)
        decomposed = make_document_id("기관\\가.pdf", digest)

        self.assertEqual(composed, decomposed)
        self.assertRegex(composed, r"^doc_[0-9a-f]{24}$")
        self.assertEqual(
            make_block_id(composed, 12, "baseline"),
            "{}:baseline:b000012".format(composed),
        )
        self.assertEqual(
            make_table_id(composed, 3, "baseline"),
            "{}:baseline:t0003".format(composed),
        )

    def test_clean_text_normalizes_controls_and_spacing(self) -> None:
        raw = "  A\u0301\t B\x00\r\n\r\n\r\n\r\n  문장  "

        self.assertEqual(clean_text(raw), "Á B\n\n\n문장")
        self.assertEqual(clean_text("A\tB", preserve_tabs=True), "A\tB")


class QualityGateTests(unittest.TestCase):
    def test_pass_metrics_ignore_table_cell_duplicate_text(self) -> None:
        table_id = make_table_id("doc_test", 0, "baseline")
        blocks = [
            make_block(
                0,
                "table",
                "구분\t내용\n1\t충분히 긴 한국어 본문 데이터입니다.",
                table_id=table_id,
                page=1,
            ),
            make_block(
                1,
                "table_cell",
                "구분",
                table_id=table_id,
                row=0,
                column=0,
                page=1,
            ),
        ]

        assessment = assess_quality(
            blocks=blocks,
            expected_page_count=1,
            table_signal=True,
            expect_korean=True,
        )

        self.assertEqual(assessment.decision, "pass")
        self.assertEqual(assessment.metrics.table_count, 1)
        self.assertEqual(assessment.metrics.table_cell_count, 1)
        self.assertEqual(assessment.metrics.pages_with_output, 1)

    def test_hard_fail_and_suspect_reasons_are_separate(self) -> None:
        hard_fail = assess_quality(
            text="짧음\x01\x02\x03",
            attempt=Attempt("unit", "timeout", timeout=True),
        )
        suspect_blocks = [
            make_block(
                0,
                "paragraph",
                "This is a sufficiently long English-only extraction result.",
                page=1,
            )
        ]
        suspect = assess_quality(
            blocks=suspect_blocks,
            expected_page_count=2,
            table_signal=True,
            expect_korean=True,
        )

        self.assertEqual(hard_fail.decision, "hard_fail")
        self.assertIn("adapter_timeout", hard_fail.hard_fail_reasons)
        self.assertTrue(
            any(reason.startswith("control_ratio:") for reason in hard_fail.hard_fail_reasons)
        )
        self.assertEqual(suspect.decision, "suspect")
        self.assertIn("page_output_missing:1/2", suspect.suspect_reasons)
        self.assertIn(
            "table_signal_without_table_block", suspect.suspect_reasons
        )
        self.assertTrue(
            any(reason.startswith("low_hangul_ratio:") for reason in suspect.suspect_reasons)
        )


class ChunkingTests(unittest.TestCase):
    def test_section_boundaries_and_table_cells_are_not_duplicated(self) -> None:
        table_id = make_table_id("doc_test", 0, "baseline")
        blocks = [
            make_block(0, "heading", "1장", section_path=["1장"], page=1),
            make_block(
                1,
                "paragraph",
                "첫 번째 절 본문",
                section_path=["1장"],
                page=1,
            ),
            make_block(2, "heading", "2장", section_path=["2장"], page=2),
            make_block(
                3,
                "table",
                "항목\t값\nA\t1",
                section_path=["2장"],
                table_id=table_id,
                page=2,
            ),
            make_block(
                4,
                "table_cell",
                "검색에 중복되면 안 됨",
                section_path=["2장"],
                table_id=table_id,
                row=0,
                column=0,
                page=2,
            ),
            make_block(
                5,
                "paragraph",
                "표 다음 본문",
                section_path=["2장"],
                page=2,
            ),
        ]

        chunks = blocks_to_legacy_chunks(
            blocks,
            metadata={
                "institution": "부산대학교",
                "relative_path": "부산대학교/문서.pdf",
            },
            max_chars=100,
            overlap=10,
        )

        self.assertEqual([chunk["chunk_index"] for chunk in chunks], [0, 1, 2, 3])
        self.assertEqual(chunks[0]["text"], "1장\n\n첫 번째 절 본문")
        self.assertEqual(chunks[1]["text"], "2장")
        self.assertEqual(chunks[2]["text"], "항목\t값\nA\t1")
        self.assertEqual(chunks[3]["text"], "표 다음 본문")
        self.assertNotIn(
            "검색에 중복되면 안 됨",
            "\n".join(chunk["text"] for chunk in chunks),
        )
        self.assertEqual(chunks[0]["metadata"]["page_start"], 1)
        self.assertEqual(chunks[0]["metadata"]["page_end"], 1)
        self.assertEqual(chunks[2]["metadata"]["table_ids"], [table_id])
        self.assertEqual(
            chunks[2]["metadata"]["block_ids"], [blocks[3].block_id]
        )
        self.assertEqual(chunks[0]["metadata"]["institution"], "부산대학교")

    def test_overlap_applies_only_inside_one_oversized_block(self) -> None:
        blocks = [
            make_block(
                0,
                "paragraph",
                "A" * 25,
                section_path=["A절"],
            ),
            make_block(
                1,
                "paragraph",
                "B" * 5,
                section_path=["A절"],
            ),
        ]

        chunks = blocks_to_legacy_chunks(blocks, max_chars=10, overlap=2)

        self.assertEqual(
            [chunk["text"] for chunk in chunks],
            ["A" * 10, "A" * 10, "A" * 9, "B" * 5],
        )

    def test_oversized_table_splits_by_rows_and_repeats_declared_header(self) -> None:
        table_id = make_table_id("doc_test", 0, "baseline")
        table = make_block(
            0,
            "table",
            "열1\t열2\nAAAA\t1111\nBBBB\t2222\nCCCC\t3333",
            table_id=table_id,
        )

        chunks = blocks_to_legacy_chunks(
            [table],
            max_chars=19,
            overlap=2,
            table_header_rows={table_id: 1},
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk["text"].startswith("열1\t열2\n") for chunk in chunks)
        )
        self.assertTrue(all(len(chunk["text"]) <= 19 for chunk in chunks))


class JsonlTests(unittest.TestCase):
    def test_jsonl_is_atomic_utf8_and_byte_deterministic(self) -> None:
        records_a = [{"b": 2, "a": "한글"}, {"z": None, "n": 1}]
        records_b = [{"a": "한글", "b": 2}, {"n": 1, "z": None}]
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.jsonl"
            second = Path(temp_dir) / "second.jsonl"

            self.assertEqual(write_jsonl(first, records_a), 2)
            self.assertEqual(write_jsonl(second, records_b), 2)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(first.read_bytes().endswith(b"\n"))
            self.assertEqual(read_jsonl(first), records_b)


def make_source() -> SourceDocument:
    return SourceDocument(
        path=Path("fixture.pdf"),
        document_id="doc_test",
        relative_path="기관/fixture.pdf",
    )


def make_block(
    reading_order: int,
    block_type: str,
    text: str,
    **kwargs: object
) -> Block:
    return Block(
        document_id="doc_test",
        block_id=make_block_id("doc_test", reading_order, "baseline"),
        block_type=block_type,
        text=text,
        reading_order=reading_order,
        parser="baseline/unit",
        **kwargs
    )


if __name__ == "__main__":
    unittest.main()
