from __future__ import annotations

import csv
import json
import hashlib
import sqlite3
import tempfile
import threading
import time
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.bm25_search import build_index, search_index
from scripts.document_parsing.core import (
    BLOCK_FIELDS,
    Attempt,
    Block,
    ParseResult,
    read_jsonl,
)
from scripts.document_parsing.output import (
    DATA_FILES,
    verify_profile_run,
    write_profile_run,
)
from scripts.document_parsing.pipeline import (
    PipelineConfig,
    PipelineOutcome,
    PipelineRunner,
    build_source_document,
)
from scripts.parse_pipeline import (
    _stage_sources,
    discover_input_files,
    load_corpus_manifest,
    run_sources,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_FIXTURE = """\
<!doctype html>
<html lang="ko">
  <body>
    <h1>졸업과제 안내</h1>
    <p>졸업과제 챗봇 파서 파이프라인 검증을 위한 충분히 긴 한국어 본문입니다.</p>
    <ul><li>파서 조합을 안정적으로 비교합니다.</li></ul>
    <table>
      <tr><th>프로필</th><th>용도</th></tr>
      <tr><td>baseline</td><td>기준선</td></tr>
    </table>
  </body>
</html>
"""
CHUNK_FIELDS = {
    "chunk_id",
    "doc_id",
    "chunk_index",
    "text",
    "char_count",
    "metadata",
}


def create_html_input(input_root: Path) -> Path:
    path = input_root / "부산대학교" / "guide.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HTML_FIXTURE, encoding="utf-8")
    return path


def manifest_record(path: Path, input_root: Path, **metadata) -> dict:
    value = {
        "input_relative_path": path.relative_to(input_root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    value.update(metadata)
    return value


def write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def run_baseline_html(
    input_root: Path,
    workspace: Path,
) -> tuple[PipelineOutcome, PipelineConfig]:
    path = create_html_input(input_root)
    config = PipelineConfig(
        profile="baseline",
        tools_dir=workspace / "tools",
        raw_output_dir=workspace / "raw",
        repo_root=REPO_ROOT,
        expect_korean=True,
    )
    source = build_source_document(
        path,
        input_root,
        profile="baseline",
        repo_root=REPO_ROOT,
    )
    return PipelineRunner(config).run(source), config


class ParsePipelineIntegrationTests(unittest.TestCase):
    def test_corpus_manifest_is_authoritative_and_preserves_source_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            included = create_html_input(input_root)
            excluded = input_root / "부산대학교" / "research.html"
            excluded.write_text(HTML_FIXTURE, encoding="utf-8")
            manifest_path = root / "corpus.jsonl"
            record = manifest_record(
                included,
                input_root,
                source_title="졸업과제 안내",
                source_url="https://www.pusan.ac.kr/notice/1",
                download_url="https://www.pusan.ac.kr/download/1",
                source_host="www.pusan.ac.kr",
                fetched_at="2026-07-25T00:00:00+00:00",
                published_at=None,
                category="academic",
                include_reason="student-facing academic notice",
                source_aliases=[
                    "https://www.pusan.ac.kr/notice/1?alias=1"
                ],
                crawl_storage_path="content/aa/guide.html",
            )
            write_manifest(manifest_path, [record])

            selection = load_corpus_manifest(
                manifest_path,
                input_root,
                "baseline",
                repo_root=REPO_ROOT,
            )

            self.assertEqual(
                [source.relative_path for source in selection.sources],
                ["부산대학교/guide.html"],
            )
            self.assertNotIn(
                excluded.resolve(),
                [source.path for source in selection.sources],
            )
            self.assertEqual(
                selection.selection_counts,
                {"manifest_entries": 1, "selected_files": 1},
            )
            source = selection.sources[0]
            self.assertEqual(source.source_title, "졸업과제 안내")
            self.assertEqual(
                source.source_url,
                "https://www.pusan.ac.kr/notice/1",
            )
            self.assertEqual(
                source.source_aliases,
                ("https://www.pusan.ac.kr/notice/1?alias=1",),
            )

            staged = _stage_sources(
                list(selection.sources),
                root / "staged",
                input_root,
            )
            self.assertNotEqual(staged[0].path, source.path)
            for field in (
                "source_title",
                "source_url",
                "download_url",
                "source_host",
                "fetched_at",
                "published_at",
                "category",
                "include_reason",
                "source_aliases",
                "crawl_storage_path",
            ):
                self.assertEqual(
                    getattr(staged[0], field),
                    getattr(source, field),
                )

            config = PipelineConfig(
                profile="baseline",
                tools_dir=root / "tools",
                raw_output_dir=root / "work" / "raw",
                repo_root=REPO_ROOT,
                expect_korean=True,
            )
            outcome = PipelineRunner(config).run(staged[0])
            output_dir = root / "run"
            write_profile_run(
                output_dir,
                [outcome],
                config,
                input_root,
                run_id="manifest-metadata",
                source_manifest_sha256=(
                    selection.source_manifest_sha256
                ),
                selection_counts=selection.selection_counts,
            )

            document = read_jsonl(output_dir / "documents.jsonl")[0]
            chunk = read_jsonl(output_dir / "chunks.jsonl")[0]
            for field in (
                "source_title",
                "source_url",
                "download_url",
                "source_host",
                "fetched_at",
                "published_at",
                "category",
                "include_reason",
                "source_aliases",
                "crawl_storage_path",
            ):
                expected = document[field]
                self.assertEqual(chunk["metadata"][field], expected)
            run_manifest = json.loads(
                (output_dir / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                run_manifest["source_manifest_sha256"],
                selection.source_manifest_sha256,
            )
            self.assertEqual(
                run_manifest["selection_counts"],
                {"manifest_entries": 1, "selected_files": 1},
            )
            self.assertTrue(
                verify_profile_run(output_dir)["valid"],
                verify_profile_run(output_dir)["errors"],
            )

    def test_corpus_manifest_rejects_invalid_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            included = create_html_input(input_root)
            valid = manifest_record(included, input_root)
            cases = (
                ("bad-sha", "SHA-256 mismatch", [
                    dict(valid, sha256="0" * 64),
                ]),
                ("bad-size", "size mismatch", [
                    dict(valid, size_bytes=valid["size_bytes"] + 1),
                ]),
                (
                    "duplicate-path",
                    "duplicate input_relative_path",
                    [valid, dict(valid)],
                ),
                ("escaping-path", "invalid input_relative_path", [
                    dict(valid, input_relative_path="../guide.html"),
                ]),
                ("bad-url", "source_url must be an HTTP", [
                    dict(valid, source_url="file:///tmp/guide.html"),
                ]),
                ("credential-url", "without credentials", [
                    dict(
                        valid,
                        source_url="https://user:secret@www.pusan.ac.kr/guide",
                    ),
                ]),
            )
            for label, message, records in cases:
                with self.subTest(label=label):
                    manifest_path = root / (label + ".jsonl")
                    write_manifest(manifest_path, records)
                    with self.assertRaisesRegex(ValueError, message):
                        load_corpus_manifest(
                            manifest_path,
                            input_root,
                            "baseline",
                            repo_root=REPO_ROOT,
                        )

    def test_legacy_discovery_without_manifest_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory) / "input"
            first = create_html_input(input_root)
            second = input_root / "부산대학교" / "second.html"
            second.write_text(HTML_FIXTURE, encoding="utf-8")

            discovered = discover_input_files(input_root)
            self.assertEqual(discovered, [first.resolve(), second.resolve()])
            source = build_source_document(
                discovered[0],
                input_root,
                "baseline",
                repo_root=REPO_ROOT,
            )
            self.assertIsNone(source.source_url)
            self.assertIsNone(source.source_title)
            self.assertEqual(source.source_aliases, ())

    def test_closing_stream_waits_for_active_parser_workers(self) -> None:
        active = 0
        finished = []
        lock = threading.Lock()

        def delayed_parse(_runner, source, _max_file_mb):
            nonlocal active
            with lock:
                active += 1
            try:
                time.sleep(0.01 if source == 0 else 0.08)
                return source
            finally:
                with lock:
                    active -= 1
                    finished.append(source)

        runner = SimpleNamespace(
            config=SimpleNamespace(profile="test")
        )
        with mock.patch(
            "scripts.parse_pipeline._parse_one",
            side_effect=delayed_parse,
        ):
            outcomes = run_sources(
                runner,
                list(range(6)),
                workers=2,
                max_file_mb=None,
            )
            self.assertEqual(next(outcomes), 0)
            outcomes.close()

        self.assertEqual(active, 0)
        self.assertIn(0, finished)

    def test_managed_runner_does_not_import_rejected_native_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            path = input_root / "fixture.docx"
            path.parent.mkdir(parents=True)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "word/document.xml",
                    (
                        "<?xml version='1.0' encoding='UTF-8'?>"
                        "<w:document "
                        "xmlns:w='http://schemas.openxmlformats.org/"
                        "wordprocessingml/2006/main'><w:body/>"
                        "</w:document>"
                    ),
                )
            source = build_source_document(
                path,
                input_root,
                profile="baseline",
                repo_root=REPO_ROOT,
            )
            runtime_report = {
                "capabilities": {
                    "core_python": {
                        "available": True,
                        "location": str(Path(__file__)),
                        "version": None,
                        "detail": "approved",
                    },
                    "python_docx": {
                        "available": False,
                        "location": None,
                        "version": None,
                        "detail": "pinned version is unavailable",
                    }
                }
            }
            config = PipelineConfig(
                profile="baseline",
                tools_dir=root / "tools",
                raw_output_dir=root / "raw",
                repo_root=REPO_ROOT,
                runtime_report=runtime_report,
            )

            outcome = PipelineRunner(config).run(source)

            self.assertEqual(outcome.result.attempts[0].status, "unavailable")
            self.assertEqual(
                outcome.result.attempts[0].parser,
                "baseline/python-docx",
            )
            self.assertIn(
                "not approved by doctor",
                outcome.result.attempts[0].reason,
            )

    def test_synthetic_html_runs_end_to_end_with_canonical_table_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, _ = run_baseline_html(root / "input", root)

            self.assertEqual(outcome.status, "parsed")
            self.assertEqual(outcome.sniffed_format, "html")
            self.assertEqual(outcome.mime_type, "text/html")
            self.assertEqual(outcome.quality.decision, "pass")
            self.assertEqual(outcome.selected_parsers, ("baseline/html-dom",))
            self.assertEqual(
                [block.reading_order for block in outcome.result.blocks],
                list(range(len(outcome.result.blocks))),
            )

            block_types = [block.block_type for block in outcome.result.blocks]
            self.assertEqual(block_types[:3], ["heading", "paragraph", "list_item"])
            self.assertEqual(block_types.count("table"), 1)
            self.assertEqual(block_types.count("table_cell"), 4)
            outcome.result.validate()

            table = next(
                block
                for block in outcome.result.blocks
                if block.block_type == "table"
            )
            cells = [
                block
                for block in outcome.result.blocks
                if block.block_type == "table_cell"
            ]
            self.assertEqual(table.text, "프로필\t용도\nbaseline\t기준선")
            self.assertTrue(all(cell.table_id == table.table_id for cell in cells))
            self.assertEqual(
                [(cell.row, cell.column) for cell in cells],
                [(0, 0), (0, 1), (1, 0), (1, 1)],
            )

            for index, block in enumerate(outcome.result.blocks):
                serialized = block.to_dict()
                self.assertEqual(tuple(serialized), BLOCK_FIELDS)
                self.assertEqual(len(serialized), 11)
                self.assertEqual(
                    block.block_id,
                    "{}:baseline:b{:06d}".format(
                        outcome.source.document_id, index
                    ),
                )
                self.assertEqual(Block.from_dict(serialized), block)

    def test_router_marks_drm_and_unknown_inputs_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            input_root.mkdir()
            fixtures = (
                (
                    "protected.hwp",
                    b"\x9b DRMONE  This Document is encrypted by Fasoo DRM",
                    "drm",
                    "fasoo_drm_signature",
                ),
                ("unknown.bin", b"\x00\x01\x02\x03", "unknown", "unknown"),
            )
            config = PipelineConfig(
                profile="baseline",
                tools_dir=root / "tools",
                raw_output_dir=root / "raw",
                repo_root=REPO_ROOT,
            )
            runner = PipelineRunner(config)

            for name, content, expected_format, sniff_reason in fixtures:
                with self.subTest(name=name):
                    path = input_root / name
                    path.write_bytes(content)
                    source = build_source_document(
                        path,
                        input_root,
                        profile="baseline",
                        repo_root=REPO_ROOT,
                    )

                    outcome = runner.run(source)

                    self.assertEqual(outcome.status, "unsupported")
                    self.assertEqual(outcome.sniffed_format, expected_format)
                    self.assertEqual(outcome.result.blocks, [])
                    self.assertEqual(outcome.selected_parsers, ())
                    self.assertEqual(len(outcome.result.attempts), 1)
                    attempt = outcome.result.attempts[0]
                    self.assertEqual(attempt.status, "unsupported")
                    self.assertEqual(attempt.parser, "baseline/router")
                    self.assertEqual(
                        attempt.metadata["sniff_reason"], sniff_reason
                    )
                    self.assertIn(
                        "unsupported_format:{}".format(expected_format),
                        outcome.reason,
                    )

    def test_ids_blocks_and_chunks_are_deterministic_across_input_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_config = run_baseline_html(
                root / "first-input", root / "first-work"
            )
            second, second_config = run_baseline_html(
                root / "second-input", root / "second-work"
            )

            self.assertEqual(first.source.document_id, second.source.document_id)
            self.assertEqual(
                [block.to_dict() for block in first.result.blocks],
                [block.to_dict() for block in second.result.blocks],
            )

            first_output = root / "run-one"
            second_output = root / "run-two"
            write_profile_run(
                first_output,
                [first],
                first_config,
                root / "first-input",
                run_id="deterministic",
                chunk_chars=90,
                chunk_overlap=10,
            )
            write_profile_run(
                second_output,
                [second],
                second_config,
                root / "second-input",
                run_id="deterministic",
                chunk_chars=90,
                chunk_overlap=10,
            )

            self.assertEqual(
                (first_output / "blocks.jsonl").read_bytes(),
                (second_output / "blocks.jsonl").read_bytes(),
            )
            self.assertEqual(
                (first_output / "chunks.jsonl").read_bytes(),
                (second_output / "chunks.jsonl").read_bytes(),
            )
            chunks = read_jsonl(first_output / "chunks.jsonl")
            self.assertTrue(chunks)
            self.assertEqual(
                [chunk["chunk_index"] for chunk in chunks],
                list(range(len(chunks))),
            )
            for index, chunk in enumerate(chunks):
                self.assertEqual(
                    chunk["chunk_id"],
                    "{}:baseline#{:04d}".format(
                        first.source.document_id, index
                    ),
                )

    def test_profile_output_verifies_and_builds_current_bm25_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            outcome, config = run_baseline_html(input_root, root / "work")
            output_dir = root / "immutable-run" / "baseline"

            result = write_profile_run(
                output_dir,
                [outcome],
                config,
                input_root,
                run_id="integration-test",
                runtime_report={"profile": "baseline", "ready": True},
                chunk_chars=180,
                chunk_overlap=20,
            )
            verification = verify_profile_run(output_dir)

            self.assertEqual(result["summary"]["counts"]["parsed"], 1)
            self.assertTrue(verification["valid"], verification["errors"])
            self.assertEqual(verification["documents"], 1)
            self.assertEqual(
                set(path.name for path in output_dir.iterdir()),
                set(DATA_FILES) | {"run_manifest.json"},
            )

            block_records = read_jsonl(output_dir / "blocks.jsonl")
            self.assertTrue(block_records)
            for value in block_records:
                self.assertEqual(set(value), set(BLOCK_FIELDS))
                self.assertEqual(len(value), 11)
                Block.from_dict(value)

            chunks_path = output_dir / "chunks.jsonl"
            chunks = read_jsonl(chunks_path)
            self.assertTrue(chunks)
            for chunk in chunks:
                self.assertEqual(set(chunk), CHUNK_FIELDS)
                self.assertEqual(chunk["doc_id"], outcome.source.document_id)
                self.assertEqual(chunk["char_count"], len(chunk["text"]))
                self.assertEqual(
                    chunk["metadata"]["institution"], "부산대학교"
                )
                self.assertEqual(
                    chunk["metadata"]["parser"], "baseline/html-dom"
                )
                self.assertTrue(
                    set(chunk["metadata"]["block_ids"]).issubset(
                        {block.block_id for block in outcome.result.blocks}
                    )
                )

            index_path = root / "index" / "bm25.sqlite"
            index_summary = build_index(chunks_path, index_path, batch_size=10)
            self.assertEqual(index_summary["chunk_count"], len(chunks))
            self.assertEqual(
                index_summary["institutions"], {"부산대학교": len(chunks)}
            )

            matches = search_index(
                index_path,
                "졸업과제 챗봇",
                top_k=5,
                institution="부산대학교",
            )
            self.assertTrue(matches)
            self.assertEqual(matches[0]["doc_id"], outcome.source.document_id)

            with sqlite3.connect(str(index_path)) as connection:
                indexed = connection.execute(
                    "SELECT chunk_id, parser, char_count FROM chunks "
                    "ORDER BY chunk_index"
                ).fetchall()
            self.assertEqual(len(indexed), len(chunks))
            self.assertTrue(
                all(row[1] == "baseline/html-dom" for row in indexed)
            )
            self.assertEqual(
                [row[2] for row in indexed],
                [chunk["char_count"] for chunk in chunks],
            )

            manifest = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(manifest["files"]), set(DATA_FILES)
            )

            del manifest["files"]["parse_summary.json"]
            (output_dir / "run_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            invalid = verify_profile_run(output_dir)
            self.assertFalse(invalid["valid"])
            self.assertTrue(
                any(
                    "manifest missing checksums" in error
                    for error in invalid["errors"]
                )
            )

    def test_unselected_enrichment_failure_does_not_invalidate_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            outcome, config = run_baseline_html(input_root, root / "work")
            result = ParseResult(
                outcome.source,
                blocks=list(outcome.result.blocks),
                attempts=list(outcome.result.attempts)
                + [
                    Attempt(
                        parser="baseline/pdfplumber",
                        status="error",
                        reason="synthetic table enrichment failure",
                    )
                ],
            )

            finished = PipelineRunner(config)._finish(
                outcome.source,
                "html",
                "text/html",
                result,
            )

            self.assertEqual(finished.status, "parsed")
            self.assertFalse(finished.quality.hard_fail)
            selected_attempt = next(
                attempt
                for attempt in finished.result.attempts
                if attempt.parser == "baseline/html-dom"
            )
            enrichment_attempt = next(
                attempt
                for attempt in finished.result.attempts
                if attempt.parser == "baseline/pdfplumber"
            )
            self.assertIn("quality", selected_attempt.metadata)
            self.assertNotIn("quality", enrichment_attempt.metadata)

    def test_failed_ocr_page_does_not_poison_a_selected_ocr_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, config = run_baseline_html(
                root / "input", root / "work"
            )
            source = outcome.source
            result = ParseResult(
                source,
                blocks=[
                    Block(
                        source.document_id,
                        "ocr-page-one",
                        "paragraph",
                        "첫 페이지에서 OCR로 선택된 충분히 긴 한국어 본문입니다.",
                        0,
                        "baseline/tesseract",
                        page=1,
                    ),
                    Block(
                        source.document_id,
                        "text-page-two",
                        "paragraph",
                        "둘째 페이지에서 텍스트 파서로 선택된 충분히 긴 본문입니다.",
                        1,
                        "baseline/pymupdf",
                        page=2,
                    ),
                ],
                attempts=[
                    Attempt(
                        "baseline/pymupdf",
                        "success",
                        metadata={"page_count": 2},
                    ),
                    Attempt(
                        "baseline/tesseract",
                        "success",
                        metadata={"page": 1},
                    ),
                    Attempt(
                        "baseline/tesseract",
                        "error",
                        reason="synthetic page failure",
                        metadata={"page": 2},
                    ),
                ],
            )

            finished = PipelineRunner(config)._finish(
                source, "pdf", "application/pdf", result
            )

            self.assertEqual(finished.status, "parsed")
            self.assertFalse(finished.quality.hard_fail)

    def test_normalization_preserves_composed_page_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, config = run_baseline_html(
                root / "input", root / "work"
            )
            source = outcome.source
            pages = (1, 1, 2, 3)
            blocks = [
                Block(
                    source.document_id,
                    "local-{}".format(index),
                    "paragraph",
                    "페이지 {}의 충분히 긴 한국어 본문 {}".format(
                        page, index
                    ),
                    0 if page > 1 else index,
                    "baseline/unit",
                    page=page,
                )
                for index, page in enumerate(pages)
            ]
            result = ParseResult(
                source,
                blocks=blocks,
                attempts=[
                    Attempt(
                        "baseline/unit",
                        "success",
                        metadata={"page_count": 3},
                    )
                ],
            )

            finished = PipelineRunner(config)._finish(
                source, "pdf", "application/pdf", result
            )

            self.assertEqual(
                [block.page for block in finished.result.blocks],
                list(pages),
            )

    def test_staging_rejects_an_input_changed_after_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            path = create_html_input(input_root)
            source = build_source_document(
                path,
                input_root,
                profile="baseline",
                repo_root=REPO_ROOT,
            )
            path.write_text("changed after hashing", encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "input changed after discovery"
            ):
                _stage_sources(
                    [source],
                    root / "staged",
                    input_root,
                )

    def test_parse_report_neutralizes_spreadsheet_formulas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            outcome, config = run_baseline_html(input_root, root / "work")
            source = replace(
                outcome.source,
                institution="=HYPERLINK(\"https://example.invalid\")",
                relative_path="+formula/guide.html",
            )
            output_dir = root / "run"
            write_profile_run(
                output_dir,
                [replace(outcome, source=source)],
                config,
                input_root,
                run_id="csv-safety",
            )

            with (output_dir / "parse_report.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))

            self.assertTrue(row["institution"].startswith("'="))
            self.assertTrue(row["relative_path"].startswith("'+"))

    def test_page_composition_never_drops_unlabeled_document_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, config = run_baseline_html(root / "input", root / "work")
            source = outcome.source
            default = ParseResult(
                source,
                blocks=[
                    Block(
                        document_id=source.document_id,
                        block_id="docling-unlabeled",
                        block_type="paragraph",
                        text="Docling이 추출한 페이지 표식 없는 전체 문서 본문입니다.",
                        reading_order=0,
                        parser="challenger/docling",
                    )
                ],
                attempts=[
                    Attempt(parser="challenger/docling", status="success")
                ],
            )
            page_override = ParseResult(
                source,
                blocks=[
                    Block(
                        document_id=source.document_id,
                        block_id="paddle-page-one",
                        block_type="paragraph",
                        text="Paddle이 추출한 첫 번째 페이지만의 OCR 본문입니다.",
                        reading_order=0,
                        parser="challenger/pp-structurev3",
                        page=1,
                    )
                ],
                attempts=[
                    Attempt(
                        parser="challenger/pp-structurev3",
                        status="success",
                    )
                ],
            )
            runner = PipelineRunner(config)

            composed = runner._compose_pages(
                source,
                page_count=2,
                default=default,
                overrides=[({1}, page_override)],
                results=[default, page_override],
            )

            self.assertEqual(
                [block.text for block in composed.blocks],
                ["Docling이 추출한 페이지 표식 없는 전체 문서 본문입니다."],
            )

    def test_verifier_reports_malformed_chunk_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            outcome, config = run_baseline_html(input_root, root / "work")
            output_dir = root / "run"
            write_profile_run(
                output_dir,
                [outcome],
                config,
                input_root,
                run_id="malformed",
            )
            chunks = read_jsonl(output_dir / "chunks.jsonl")
            chunks[0]["char_count"] = "not-an-integer"
            (output_dir / "chunks.jsonl").write_text(
                "\n".join(
                    json.dumps(item, ensure_ascii=False, sort_keys=True)
                    for item in chunks
                )
                + "\n",
                encoding="utf-8",
            )

            verification = verify_profile_run(output_dir)

            self.assertFalse(verification["valid"])
            self.assertTrue(
                any(
                    "char_count mismatch" in error
                    for error in verification["errors"]
                )
            )

    def test_raw_artifacts_are_checksum_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            outcome, config = run_baseline_html(input_root, root / "work")
            output_dir = root / "run"
            raw_path = output_dir / "raw" / "worker.stdout"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text("original raw parser output", encoding="utf-8")
            outcome.result.raw_artifacts.append(str(raw_path))
            write_profile_run(
                output_dir,
                [outcome],
                config,
                input_root,
                run_id="raw-integrity",
            )
            self.assertTrue(verify_profile_run(output_dir)["valid"])

            raw_path.write_text("tampered raw parser output", encoding="utf-8")
            verification = verify_profile_run(output_dir)

            self.assertFalse(verification["valid"])
            self.assertTrue(
                any(
                    "raw artifact checksum mismatch" in error
                    for error in verification["errors"]
                )
            )

    def test_verifier_rejects_duplicate_and_non_contiguous_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            outcome, config = run_baseline_html(input_root, root / "work")
            output_dir = root / "run"
            write_profile_run(
                output_dir,
                [outcome],
                config,
                input_root,
                run_id="duplicate-chunks",
                chunk_chars=90,
                chunk_overlap=10,
            )
            chunks = read_jsonl(output_dir / "chunks.jsonl")
            self.assertGreaterEqual(len(chunks), 2)
            chunks[1]["chunk_id"] = chunks[0]["chunk_id"]
            chunks[1]["chunk_index"] = chunks[0]["chunk_index"]
            chunks_path = output_dir / "chunks.jsonl"
            chunks_path.write_text(
                "\n".join(
                    json.dumps(item, ensure_ascii=False, sort_keys=True)
                    for item in chunks
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = output_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["chunks.jsonl"] = hashlib.sha256(
                chunks_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )

            verification = verify_profile_run(output_dir)

            self.assertFalse(verification["valid"])
            self.assertTrue(
                any(
                    "duplicate chunk_id" in error
                    for error in verification["errors"]
                )
            )
            self.assertTrue(
                any(
                    "non-contiguous or duplicate chunk_index" in error
                    for error in verification["errors"]
                )
            )

    def test_verifier_rejects_internal_parser_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            outcome, config = run_baseline_html(input_root, root / "work")
            outcome.result.blocks[0] = Block(
                document_id=outcome.result.blocks[0].document_id,
                block_id=outcome.result.blocks[0].block_id,
                block_type=outcome.result.blocks[0].block_type,
                text="본문 \ue000TABLE_START\ue001 유출",
                reading_order=outcome.result.blocks[0].reading_order,
                parser=outcome.result.blocks[0].parser,
            )
            output_dir = root / "run"
            write_profile_run(
                output_dir,
                [outcome],
                config,
                input_root,
                run_id="sentinel",
            )

            verification = verify_profile_run(output_dir)

            self.assertFalse(verification["valid"])
            self.assertTrue(
                any(
                    "internal parser sentinel" in error
                    for error in verification["errors"]
                )
            )


if __name__ == "__main__":
    unittest.main()
