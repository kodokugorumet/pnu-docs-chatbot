from __future__ import annotations

import unittest

from scripts.rag.models import SearchHit, citation_from_hit
from scripts.search_api import citation_for_result


class SourceMetadataContractTests(unittest.TestCase):
    def test_search_hit_keeps_public_http_source_metadata(self) -> None:
        hit = SearchHit(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_title="학사 일정",
            source_url="https://www.pusan.ac.kr/notice/1",
            download_url="https://www.pusan.ac.kr/download/1",
            fetched_at="2026-07-25T01:00:00+00:00",
            published_at=None,
            source_aliases=(
                "https://dept.pusan.ac.kr/notice/1",
                "javascript:alert(1)",
            ),
        )

        serialized = hit.to_dict()

        self.assertEqual(
            serialized["source_url"],
            "https://www.pusan.ac.kr/notice/1",
        )
        self.assertEqual(
            serialized["source_aliases"],
            ["https://dept.pusan.ac.kr/notice/1"],
        )

    def test_non_http_source_links_are_not_exposed(self) -> None:
        hit = SearchHit(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_url="javascript:alert(1)",
            download_url="file:///tmp/source.pdf",
        )

        self.assertIsNone(hit.source_url)
        self.assertIsNone(hit.download_url)

    def test_credential_bearing_source_links_are_not_exposed(self) -> None:
        hit = SearchHit(
            chunk_id="chunk-1",
            document_id="doc-1",
            source_url="https://user:secret@www.pusan.ac.kr/notice",
            download_url="https://user:secret@www.pusan.ac.kr/download",
            source_aliases=(
                "https://user:secret@www.pusan.ac.kr/alias",
            ),
        )

        self.assertIsNone(hit.source_url)
        self.assertIsNone(hit.download_url)
        self.assertEqual(hit.source_aliases, ())

    def test_citation_contract_preserves_source_links(self) -> None:
        hit = SearchHit(
            chunk_id="chunk-1",
            document_id="doc-1",
            preview="졸업 요건 안내",
            source_title="졸업 요건",
            source_url="https://www.pusan.ac.kr/notice/2",
            download_url="https://www.pusan.ac.kr/download/2",
            fetched_at="2026-07-25T01:00:00+00:00",
        )

        canonical = citation_from_hit(hit, [hit.chunk_id]).to_dict()
        public = citation_for_result(hit.to_dict())

        for value in (canonical, public):
            self.assertEqual(value["source_title"], "졸업 요건")
            self.assertEqual(
                value["source_url"],
                "https://www.pusan.ac.kr/notice/2",
            )
            self.assertEqual(
                value["download_url"],
                "https://www.pusan.ac.kr/download/2",
            )


if __name__ == "__main__":
    unittest.main()
