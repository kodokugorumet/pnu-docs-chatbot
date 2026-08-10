from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.curate_pnu_corpus import (
    CurationError,
    DEFAULT_CORE_HOSTS,
    MANIFEST_FIELDS,
    _category_match,
    _load_host_file,
    _load_scope_file,
    curate_corpus,
)
from scripts.derive_curated_run import DerivationError, derive_curated_run
from scripts.document_parsing.core import read_jsonl
from scripts.document_parsing.output import (
    DATA_FILES,
    sha256_path,
    verify_profile_run,
    write_profile_run,
)
from scripts.document_parsing.pipeline import (
    PipelineConfig,
    PipelineRunner,
    build_source_document,
)
from scripts.parse_pipeline import load_corpus_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
HTML_TEMPLATE = """\
<!doctype html>
<html lang="ko"><body>
<h1>{title}</h1>
<p>부산대학교 학생을 위한 학사 정보와 졸업 요건을 충분히 설명하는 검증 본문입니다.</p>
<p>수강 신청, 장학 제도, 학적 처리 절차를 정확하게 안내합니다.</p>
</body></html>
"""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _create_crawl_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE frontier (
            url TEXT PRIMARY KEY,
            host TEXT NOT NULL,
            parent_url TEXT,
            anchor_text TEXT,
            status TEXT NOT NULL,
            kind TEXT NOT NULL,
            fetched_at TEXT,
            content_type TEXT,
            final_url TEXT,
            title TEXT,
            storage_path TEXT,
            sha256 TEXT,
            size_bytes INTEGER
        )
        """
    )
    return connection


def _insert_row(
    connection: sqlite3.Connection,
    *,
    url: str,
    host: str,
    storage_path: str | None,
    body: bytes | None = None,
    parent_url: str | None = None,
    anchor_text: str = "",
    kind: str = "page",
    content_type: str = "text/html",
    title: str = "",
    final_url: str | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO frontier(
            url, host, parent_url, anchor_text, status, kind, fetched_at,
            content_type, final_url, title, storage_path, sha256, size_bytes
        ) VALUES (?, ?, ?, ?, 'done', ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            url,
            host,
            parent_url,
            anchor_text,
            kind,
            "2026-07-25T00:00:00+00:00",
            content_type,
            final_url or url,
            title,
            storage_path,
            sha256 if sha256 is not None else (_sha(body) if body is not None else None),
            size_bytes if size_bytes is not None else (
                len(body) if body is not None else None
            ),
        ),
    )


class PnuCorpusCurationTests(unittest.TestCase):
    def test_international_hostname_is_not_an_internship_match(self) -> None:
        category, term = _category_match(
            {
                "url": "https://international.pusan.ac.kr/notices/guide",
                "title": "외국인 학생 안내",
            }
        )

        self.assertEqual(category, "international")
        self.assertEqual(term, "외국인")

    def _fixture(self, root: Path) -> tuple[Path, Path, list[Path]]:
        crawl_root = root / "crawl"
        input_root = crawl_root / "content"
        database = crawl_root / "state" / "crawl.sqlite3"
        connection = _create_crawl_database(database)
        created: list[Path] = []

        def raw(relative: str, body: bytes) -> str:
            path = crawl_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            created.append(path)
            return relative

        duplicate_body = "<html><body>핵심 학생 안내 문서입니다.</body></html>".encode(
            "utf-8"
        )
        core_path = raw(
            "content/부산대학교/웹페이지/www.pusan.ac.kr/core.html",
            duplicate_body,
        )
        _insert_row(
            connection,
            url="https://www.pusan.ac.kr/core",
            host="www.pusan.ac.kr",
            storage_path=core_path,
            body=duplicate_body,
            title="대학 행사 소개",
        )

        duplicate_path = raw(
            "content/부산대학교/웹페이지/dept.pusan.ac.kr/duplicate.html",
            duplicate_body,
        )
        _insert_row(
            connection,
            url="https://dept.pusan.ac.kr/graduation/duplicate",
            host="dept.pusan.ac.kr",
            storage_path=duplicate_path,
            body=duplicate_body,
            anchor_text="졸업 안내",
            title="학과 졸업 공지",
        )

        parent_url = "https://dept.pusan.ac.kr/notice/graduate"
        _insert_row(
            connection,
            url=parent_url,
            host="dept.pusan.ac.kr",
            storage_path=None,
            title="2026학년도 졸업 신청 안내",
        )
        pdf_body = b"%PDF-1.4 curated graduation attachment"
        pdf_path = raw(
            "content/부산대학교/첨부파일/dept.pusan.ac.kr/graduation.pdf",
            pdf_body,
        )
        _insert_row(
            connection,
            url="https://dept.pusan.ac.kr/download.do?id=1",
            final_url="https://dept.pusan.ac.kr/files/graduation.pdf",
            host="dept.pusan.ac.kr",
            storage_path=pdf_path,
            body=pdf_body,
            parent_url=parent_url,
            anchor_text="붙임 졸업 신청서.pdf",
            kind="attachment",
            content_type="application/pdf",
            title="hashed-graduation.pdf",
        )

        irrelevant_body = b"%PDF-1.4 research symposium"
        irrelevant_path = raw(
            "content/부산대학교/첨부파일/dept.pusan.ac.kr/symposium.pdf",
            irrelevant_body,
        )
        _insert_row(
            connection,
            url="https://dept.pusan.ac.kr/files/symposium.pdf",
            host="dept.pusan.ac.kr",
            storage_path=irrelevant_path,
            body=irrelevant_body,
            kind="attachment",
            content_type="application/pdf",
            title="연구 심포지엄 자료",
        )

        image_body = b"\xff\xd8image"
        image_path = raw(
            "content/부산대학교/첨부파일/www.pusan.ac.kr/poster.jpg",
            image_body,
        )
        _insert_row(
            connection,
            url="https://www.pusan.ac.kr/download.do?id=poster",
            host="www.pusan.ac.kr",
            storage_path=image_path,
            body=image_body,
            kind="attachment",
            content_type="application/octet-stream",
            title="학생 안내 포스터.jpg",
        )

        corrupt_body = b"%PDF actual"
        corrupt_path = raw(
            "content/부산대학교/첨부파일/www.pusan.ac.kr/corrupt.pdf",
            corrupt_body,
        )
        _insert_row(
            connection,
            url="https://www.pusan.ac.kr/files/corrupt.pdf",
            host="www.pusan.ac.kr",
            storage_path=corrupt_path,
            body=corrupt_body,
            kind="attachment",
            content_type="application/pdf",
            sha256="0" * 64,
        )

        other_body = b"%PDF other host"
        other_path = raw(
            "content/부산대학교/첨부파일/research.pusan.ac.kr/paper.pdf",
            other_body,
        )
        _insert_row(
            connection,
            url="https://research.pusan.ac.kr/paper.pdf",
            host="research.pusan.ac.kr",
            storage_path=other_path,
            body=other_body,
            kind="attachment",
            content_type="application/pdf",
            title="졸업 논문 연구",
        )

        _insert_row(
            connection,
            url="https://www.pusan.ac.kr/files/escape.pdf",
            host="www.pusan.ac.kr",
            storage_path="../escape.pdf",
            body=b"not on disk",
            kind="attachment",
            content_type="application/pdf",
        )
        connection.commit()
        connection.close()
        return crawl_root, database, created

    def test_curator_filters_validates_deduplicates_and_never_copies_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crawl_root, database, raw_paths = self._fixture(root)
            raw_hashes = {path: sha256_path(path) for path in raw_paths}
            output = root / "curated"

            summary = curate_corpus(
                crawl_root=crawl_root,
                database=database,
                input_root=crawl_root / "content",
                output_dir=output,
                academic_hosts=("dept.pusan.ac.kr",),
                max_documents=20,
                max_total_bytes=1024 * 1024,
            )

            manifest = read_jsonl(output / "curated-manifest.jsonl")
            rejected = read_jsonl(output / "rejected.jsonl")
            self.assertEqual(len(manifest), 2)
            self.assertEqual(summary["selected_documents"], 2)
            self.assertEqual(
                [item["input_relative_path"] for item in manifest],
                sorted(item["input_relative_path"] for item in manifest),
            )
            self.assertTrue(
                all(tuple(sorted(item)) == tuple(sorted(MANIFEST_FIELDS)) for item in manifest)
            )

            core = next(
                item
                for item in manifest
                if item["source_host"] == "www.pusan.ac.kr"
            )
            self.assertEqual(core["source_url"], "https://www.pusan.ac.kr/core")
            self.assertIn(
                "https://dept.pusan.ac.kr/graduation/duplicate",
                core["source_aliases"],
            )
            attachment = next(
                item
                for item in manifest
                if item["input_relative_path"].endswith("graduation.pdf")
            )
            self.assertEqual(
                attachment["source_url"],
                "https://dept.pusan.ac.kr/notice/graduate",
            )
            self.assertEqual(
                attachment["download_url"],
                "https://dept.pusan.ac.kr/files/graduation.pdf",
            )
            self.assertEqual(attachment["published_at"], None)
            self.assertEqual(attachment["category"], "graduation")

            reasons = {item["reason"] for item in rejected}
            self.assertTrue(
                {
                    "academic_irrelevant",
                    "duplicate_sha256",
                    "hard_rejected_extension",
                    "host_out_of_scope",
                    "missing_storage_path",
                    "sha256_mismatch",
                    "unsafe_storage_path",
                }.issubset(reasons)
            )
            self.assertEqual(
                {path: sha256_path(path) for path in raw_paths},
                raw_hashes,
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"curated-manifest.jsonl", "rejected.jsonl", "summary.json"},
            )
            parser_selection = load_corpus_manifest(
                output / "curated-manifest.jsonl",
                crawl_root / "content",
                "baseline",
                repo_root=REPO_ROOT,
            )
            self.assertEqual(len(parser_selection.sources), 2)
            self.assertEqual(
                [source.relative_path for source in parser_selection.sources],
                [item["input_relative_path"] for item in manifest],
            )

    def test_curator_trusts_actual_url_host_instead_of_database_host(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crawl_root = root / "crawl"
            input_root = crawl_root / "content"
            database = crawl_root / "state" / "crawl.sqlite3"
            connection = _create_crawl_database(database)

            included_body = b"<html>trusted PNU URL</html>"
            included_relative = (
                "content/부산대학교/웹페이지/"
                "www.pusan.ac.kr/trusted.html"
            )
            included_path = crawl_root / included_relative
            included_path.parent.mkdir(parents=True, exist_ok=True)
            included_path.write_bytes(included_body)
            _insert_row(
                connection,
                url="https://www.pusan.ac.kr/trusted",
                host="attacker.example",
                storage_path=included_relative,
                body=included_body,
                title="학생 학사 안내",
            )

            external_body = b"<html>external redirect</html>"
            external_relative = (
                "content/부산대학교/웹페이지/"
                "www.pusan.ac.kr/external.html"
            )
            external_path = crawl_root / external_relative
            external_path.write_bytes(external_body)
            _insert_row(
                connection,
                url="https://www.pusan.ac.kr/redirect",
                final_url="https://attacker.example/external",
                host="www.pusan.ac.kr",
                storage_path=external_relative,
                body=external_body,
                title="학생 학사 안내",
            )
            connection.commit()
            connection.close()

            output = root / "curated"
            curate_corpus(
                crawl_root=crawl_root,
                database=database,
                input_root=input_root,
                output_dir=output,
                core_hosts=("www.pusan.ac.kr",),
                max_documents=10,
                max_total_bytes=1024 * 1024,
            )

            manifest = read_jsonl(output / "curated-manifest.jsonl")
            rejected = read_jsonl(output / "rejected.jsonl")
            self.assertEqual(len(manifest), 1)
            self.assertEqual(
                manifest[0]["source_url"],
                "https://www.pusan.ac.kr/trusted",
            )
            self.assertEqual(
                manifest[0]["source_host"],
                "www.pusan.ac.kr",
            )
            self.assertFalse(
                any(
                    "attacker.example" in str(value)
                    for item in manifest
                    for value in item.values()
                )
            )
            external = next(
                item
                for item in rejected
                if item["source_host"] == "attacker.example"
            )
            self.assertEqual(external["reason"], "host_out_of_scope")

    def test_curator_enforces_budget_and_does_not_publish_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crawl_root, database, _ = self._fixture(root)
            output = root / "too-large"

            with self.assertRaisesRegex(CurationError, "document budget"):
                curate_corpus(
                    crawl_root=crawl_root,
                    database=database,
                    input_root=crawl_root / "content",
                    output_dir=output,
                    academic_hosts=("dept.pusan.ac.kr",),
                    max_documents=1,
                    max_total_bytes=1024 * 1024,
                )

            self.assertFalse(output.exists())

    def test_curator_deduplicates_legacy_pages_by_declared_canonical_url(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crawl_root = root / "crawl"
            database = crawl_root / "state" / "crawl.sqlite3"
            connection = _create_crawl_database(database)
            canonical_url = "https://www.pusan.ac.kr/notice/42"
            aliases = (
                canonical_url + "?p=10",
                canonical_url + "?p=11",
            )
            for index, alias in enumerate(aliases):
                body = (
                    '<html><head><link rel="canonical" href="{}">'
                    "<title>2026 학사 공지</title></head>"
                    "<body>서로 다른 페이지 탐색 UI {}</body></html>"
                ).format(canonical_url, index).encode("utf-8")
                relative = (
                    "content/부산대학교/웹페이지/www.pusan.ac.kr/"
                    "notice-{}.html".format(index)
                )
                path = crawl_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                _insert_row(
                    connection,
                    url=alias,
                    host="www.pusan.ac.kr",
                    storage_path=relative,
                    body=body,
                    content_type="text/html",
                    title="2026 학사 공지",
                )
            connection.commit()
            connection.close()
            output = root / "curated"

            summary = curate_corpus(
                crawl_root=crawl_root,
                database=database,
                input_root=crawl_root / "content",
                output_dir=output,
                core_hosts=("www.pusan.ac.kr",),
                max_documents=10,
                max_total_bytes=1024 * 1024,
            )

            manifest = read_jsonl(output / "curated-manifest.jsonl")
            rejected = read_jsonl(output / "rejected.jsonl")
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["source_url"], canonical_url)
            self.assertEqual(
                manifest[0]["source_aliases"],
                list(aliases),
            )
            self.assertEqual(
                Counter(item["reason"] for item in rejected),
                Counter({"duplicate_canonical_url": 1}),
            )
            self.assertEqual(
                summary["rejection_reasons"]["duplicate_canonical_url"],
                1,
            )

    def test_nested_scope_file_supplies_exact_core_department_and_budgets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.json"
            scope = {
                "tiers": {
                    "core": {
                        "max_items_per_host": 500,
                        "hosts": ["www.pusan.ac.kr", "go.pusan.ac.kr"],
                    },
                    "department": {
                        "max_items_per_host": 100,
                        "hosts": [
                            "archaeology.pusan.ac.kr",
                            "medicine.pusan.ac.kr",
                        ],
                    },
                },
                "budgets": {
                    "max_pages": 4000,
                    "max_files": 2000,
                    "max_bytes": 1610612736,
                },
                "allowed_document_extensions": [".pdf", ".hwp"],
                "version": 1,
            }
            path.write_text(json.dumps(scope), encoding="utf-8")

            self.assertEqual(
                _load_host_file(path),
                [
                    "archaeology.pusan.ac.kr",
                    "medicine.pusan.ac.kr",
                ],
            )
            parsed_scope = _load_scope_file(path)
            tiers = {tier.name: tier for tier in parsed_scope.tiers}
            self.assertEqual(
                list(tiers["core"].hosts),
                ["go.pusan.ac.kr", "www.pusan.ac.kr"],
            )
            self.assertEqual(
                list(tiers["department"].hosts),
                [
                    "archaeology.pusan.ac.kr",
                    "medicine.pusan.ac.kr",
                ],
            )
            self.assertEqual(tiers["core"].max_items_per_host, 500)
            self.assertEqual(tiers["department"].max_items_per_host, 100)
            self.assertEqual(
                parsed_scope.allowed_document_extensions,
                (".hwp", ".pdf"),
            )
            self.assertEqual(
                parsed_scope.max_pages + parsed_scope.max_files,
                6000,
            )
            self.assertEqual(parsed_scope.max_bytes, 1610612736)

    def test_curator_enforces_per_host_limit_and_scope_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crawl_root = root / "crawl"
            content_root = crawl_root / "content"
            database = crawl_root / "state" / "crawl.sqlite3"
            connection = _create_crawl_database(database)
            fixtures = (
                (".html", "page", "text/html"),
                (".pdf", "attachment", "application/pdf"),
                (".pdf", "attachment", "application/pdf"),
                (".txt", "attachment", "text/plain"),
            )
            for index, (extension, kind, content_type) in enumerate(fixtures):
                body = "졸업 안내 {}".format(index).encode("utf-8")
                relative = (
                    "content/부산대학교/첨부파일/"
                    "dept.pusan.ac.kr/file-{}{}".format(index, extension)
                )
                path = crawl_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                _insert_row(
                    connection,
                    url="https://dept.pusan.ac.kr/graduation/{}{}".format(
                        index,
                        extension,
                    ),
                    host="dept.pusan.ac.kr",
                    storage_path=relative,
                    body=body,
                    kind=kind,
                    content_type=content_type,
                    title="졸업 안내 {}".format(index),
                )
            connection.commit()
            connection.close()
            output = root / "curated"

            summary = curate_corpus(
                crawl_root=crawl_root,
                database=database,
                input_root=content_root,
                output_dir=output,
                core_hosts=(),
                academic_hosts=("dept.pusan.ac.kr",),
                max_documents=10,
                max_total_bytes=1024,
                max_items_per_host={"dept.pusan.ac.kr": 1},
                allowed_extensions={".html", ".pdf"},
            )

            manifest = read_jsonl(output / "curated-manifest.jsonl")
            rejected = read_jsonl(output / "rejected.jsonl")
            self.assertEqual(len(manifest), 1)
            self.assertTrue(manifest[0]["input_relative_path"].endswith(".pdf"))
            self.assertEqual(
                summary["max_items_per_host"],
                {"dept.pusan.ac.kr": 1},
            )
            self.assertEqual(summary["allowed_extensions"], [".html", ".pdf"])
            self.assertEqual(
                Counter(item["reason"] for item in rejected),
                Counter(
                    {
                        "max_items_per_host": 2,
                        "unsupported_extension": 1,
                    }
                ),
            )

            too_many_files = root / "too-many-files"
            with self.assertRaisesRegex(CurationError, "file budget"):
                curate_corpus(
                    crawl_root=crawl_root,
                    database=database,
                    input_root=content_root,
                    output_dir=too_many_files,
                    core_hosts=(),
                    academic_hosts=("dept.pusan.ac.kr",),
                    max_documents=10,
                    max_total_bytes=1024,
                    max_pages=10,
                    max_files=1,
                    max_items_per_host={"dept.pusan.ac.kr": 10},
                    allowed_extensions={".html", ".pdf"},
                )
            self.assertFalse(too_many_files.exists())

        self.assertEqual(
            set(DEFAULT_CORE_HOSTS),
            {
                "pusan.ac.kr",
                "www.pusan.ac.kr",
                "dorm.pusan.ac.kr",
                "graduate.pusan.ac.kr",
                "go.pusan.ac.kr",
                "international.pusan.ac.kr",
                "job.pusan.ac.kr",
                "lib.pusan.ac.kr",
                "equality.pusan.ac.kr",
                "sedu-support.pusan.ac.kr",
            },
        )


class DerivedCuratedRunTests(unittest.TestCase):
    def _source_run(
        self,
        root: Path,
    ) -> tuple[Path, list, Path]:
        input_root = root / "input"
        paths = []
        for name in ("a.html", "b.html"):
            path = input_root / "부산대학교" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                HTML_TEMPLATE.format(title="{} 졸업 안내".format(name)),
                encoding="utf-8",
            )
            paths.append(path)

        source_run = root / "source-run" / "baseline"
        raw_dir = source_run / "raw"
        raw_dir.mkdir(parents=True)
        config = PipelineConfig(
            profile="baseline",
            tools_dir=root / "tools",
            raw_output_dir=raw_dir,
            repo_root=REPO_ROOT,
            expect_korean=True,
        )
        outcomes = []
        for index, path in enumerate(paths):
            source = build_source_document(
                path,
                input_root,
                profile="baseline",
                repo_root=REPO_ROOT,
            )
            outcome = PipelineRunner(config).run(source)
            artifact = raw_dir / "doc-{}.txt".format(index)
            artifact.write_text("raw artifact {}".format(index), encoding="utf-8")
            outcome.result.raw_artifacts.append(str(artifact))
            outcomes.append(outcome)
        write_profile_run(
            source_run,
            outcomes,
            config,
            input_root,
            run_id="source-fixture",
            chunk_chars=120,
            chunk_overlap=20,
        )
        self.assertTrue(verify_profile_run(source_run)["valid"])
        return source_run, outcomes, input_root

    def test_derived_run_filters_every_artifact_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run, outcomes, _ = self._source_run(root)
            selected = outcomes[0].source
            manifest_path = root / "curated-manifest.jsonl"
            _write_jsonl(
                manifest_path,
                [
                    {
                        "input_relative_path": selected.relative_path,
                        "sha256": selected.source_sha256,
                        "size_bytes": selected.size_bytes,
                        "source_title": "원문 졸업 안내",
                        "source_url": "https://www.pusan.ac.kr/notice/1",
                        "download_url": "https://www.pusan.ac.kr/download/1",
                        "source_host": "www.pusan.ac.kr",
                        "fetched_at": "2026-07-25T00:00:00+00:00",
                        "published_at": None,
                        "crawl_storage_path": (
                            "content/부산대학교/웹페이지/"
                            "www.pusan.ac.kr/a.html"
                        ),
                        "source_aliases": [
                            "https://www.pusan.ac.kr/notice/1?alias=1"
                        ],
                        "category": "graduation",
                        "include_reason": "core_host:www.pusan.ac.kr",
                    }
                ],
            )
            source_hashes = {
                name: sha256_path(source_run / name)
                for name in tuple(DATA_FILES) + ("run_manifest.json",)
            }
            output = root / "derived-run" / "baseline"

            result = derive_curated_run(
                source_run=source_run,
                curated_manifest=manifest_path,
                output_dir=output,
                run_id="curated-fixture",
            )

            verification = verify_profile_run(output)
            self.assertTrue(verification["valid"], verification["errors"])
            self.assertTrue(result["verification"]["valid"])
            documents = read_jsonl(output / "documents.jsonl")
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["document_id"], selected.document_id)
            self.assertEqual(documents[0]["source_title"], "원문 졸업 안내")
            self.assertEqual(
                documents[0]["source_url"],
                "https://www.pusan.ac.kr/notice/1",
            )
            self.assertEqual(
                documents[0]["source_aliases"],
                ["https://www.pusan.ac.kr/notice/1?alias=1"],
            )
            metadata_fields = {
                "source_title",
                "source_url",
                "download_url",
                "source_host",
                "fetched_at",
                "published_at",
                "crawl_storage_path",
                "source_aliases",
                "category",
                "include_reason",
            }
            self.assertTrue(metadata_fields.issubset(documents[0]))
            self.assertIsNone(documents[0]["published_at"])
            self.assertTrue(
                all(
                    block["document_id"] == selected.document_id
                    for block in read_jsonl(output / "blocks.jsonl")
                )
            )
            chunks = read_jsonl(output / "chunks.jsonl")
            self.assertTrue(
                all(chunk["doc_id"] == selected.document_id for chunk in chunks)
            )
            self.assertTrue(chunks)
            self.assertTrue(
                all(metadata_fields.issubset(chunk["metadata"]) for chunk in chunks)
            )
            self.assertTrue(
                all(
                    chunk["metadata"]["source_url"]
                    == "https://www.pusan.ac.kr/notice/1"
                    and chunk["metadata"]["source_title"] == "원문 졸업 안내"
                    and chunk["metadata"]["category"] == "graduation"
                    for chunk in chunks
                )
            )
            self.assertTrue(
                all(
                    attempt["document_id"] == selected.document_id
                    for attempt in read_jsonl(output / "attempts.jsonl")
                )
            )
            self.assertTrue((output / "raw" / "doc-0.txt").is_file())
            self.assertFalse((output / "raw" / "doc-1.txt").exists())
            self.assertEqual(result["summary"]["file_count"], 1)
            self.assertEqual(result["summary"]["run_id"], "curated-fixture")
            derived_manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                derived_manifest["derived_from"]["curated_manifest_sha256"],
                sha256_path(manifest_path),
            )
            self.assertEqual(
                derived_manifest["source_manifest_sha256"],
                sha256_path(manifest_path),
            )
            self.assertEqual(
                derived_manifest["selection_counts"],
                {"manifest_entries": 1, "selected_files": 1},
            )
            self.assertEqual(
                {
                    name: sha256_path(source_run / name)
                    for name in tuple(DATA_FILES) + ("run_manifest.json",)
                },
                source_hashes,
            )

    def test_derived_run_fails_closed_on_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_run, outcomes, _ = self._source_run(root)
            selected = outcomes[0].source
            manifest_path = root / "bad-manifest.jsonl"
            _write_jsonl(
                manifest_path,
                [
                    {
                        "input_relative_path": selected.relative_path,
                        "sha256": "0" * 64,
                        "size_bytes": selected.size_bytes,
                    }
                ],
            )
            output = root / "must-not-exist"

            with self.assertRaisesRegex(DerivationError, "sha256 mismatch"):
                derive_curated_run(
                    source_run=source_run,
                    curated_manifest=manifest_path,
                    output_dir=output,
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
