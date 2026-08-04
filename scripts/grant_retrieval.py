"""연구비 규정 평가용 검색 모드 어댑터.

BM25·learned dense·하이브리드를 같은 호출 규약으로 감싸, 검색 평가와 생성
평가가 동일한 방식으로 검색기를 바꿔 끼울 수 있게 한다. 반환값은 기존 평가
코드가 쓰는 dict 리스트로 통일한다(`SearchHit.to_dict()`).

배선 방식은 `search_api`의 검색 모드 구성과 같지만, 평가 스크립트에 HTTP 서버
모듈을 끌어들이지 않도록 아티팩트 경로를 인자로 직접 받는다. `rag` 패키지는
BM25 구현에 의존하지 않는 재사용 계층이므로, 두 레인을 엮는 이 글루 코드는
`search_api`와 같은 스크립트 계층에 둔다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from bm25_search import (
    load_chunks_by_ids,
    search_bm25_candidates,
    search_index,
)
from rag.learned_dense import LearnedDenseIndex
from rag.retrieval import HybridRetriever


MODES = ("bm25", "dense", "hybrid")

Searcher = Callable[[str, int], list[dict[str, Any]]]


def build_searcher(
    index_path: Path,
    mode: str = "bm25",
    dense_artifact: Path | None = None,
    *,
    preview_chars: int = 700,
) -> Searcher:
    """검색 모드 하나를 `(query, top_k) -> list[dict]` 호출로 만들어 준다."""

    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode: {mode} (expected {MODES})")

    index_path = Path(index_path)
    if mode == "bm25":
        def bm25_searcher(query: str, top_k: int) -> list[dict[str, Any]]:
            return search_index(
                index_path,
                query,
                top_k,
                None,
                preview_chars=preview_chars,
                include_text=True,
            )

        return bm25_searcher

    if dense_artifact is None:
        raise ValueError(f"mode={mode} requires a dense artifact directory")

    learned_index = LearnedDenseIndex(
        Path(dense_artifact),
        source_index=index_path,
        row_loader=lambda chunk_ids, source=index_path: (
            load_chunks_by_ids(source, chunk_ids)
        ),
    )

    def bm25_lane(
        *,
        query: str,
        top_k: int,
        institution: str | None,
    ) -> list[dict[str, Any]]:
        return search_bm25_candidates(
            index_path,
            query,
            top_k,
            institution,
            preview_chars=preview_chars,
            include_text=True,
        )

    if mode == "dense":
        retriever = HybridRetriever(
            bm25_search=None,
            dense_index=learned_index,
            candidate_multiplier=1,
            reranker=None,
        )
    else:
        retriever = HybridRetriever(
            bm25_search=bm25_lane,
            dense_index=learned_index,
        )

    def learned_searcher(query: str, top_k: int) -> list[dict[str, Any]]:
        return _as_dicts(retriever.search(query, top_k=top_k).hits)

    return learned_searcher


def _as_dicts(hits: Sequence[Any]) -> list[dict[str, Any]]:
    """평가 코드가 기대하는 dict 형태로 변환한다.

    `text`만 있고 `preview`가 비면 생성 평가가 근거 본문을 잃으므로 되메운다.
    """

    rows: list[dict[str, Any]] = []
    for hit in hits:
        row = hit.to_dict() if hasattr(hit, "to_dict") else dict(hit)
        if not row.get("preview"):
            row["preview"] = row.get("text") or ""
        rows.append(row)
    return rows
