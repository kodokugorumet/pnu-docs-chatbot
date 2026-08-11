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
    DEFAULT_RERANK_CANDIDATE_MULTIPLIER,
    load_chunks_by_ids,
    search_bm25_candidates,
    select_document_diverse_results,
)
from rag.learned_dense import LearnedDenseIndex
from rag.retrieval import HybridRetriever, lexical_fallback_rerank


MODES = ("bm25", "dense", "hybrid")

Searcher = Callable[[str, int], list[dict[str, Any]]]


def _candidate_limit(top_k: int) -> int:
    """`search_index`가 쓰는 후보 폭과 같은 값."""

    return max(20, top_k, top_k * DEFAULT_RERANK_CANDIDATE_MULTIPLIER)


def build_searcher(
    index_path: Path,
    mode: str = "bm25",
    dense_artifact: Path | None = None,
    *,
    preview_chars: int = 700,
    diversify: bool = True,
    max_chunks_per_document: int = 2,
    min_chars: int = 0,
    reranker: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    anchor_bm25_top1: bool = False,
) -> Searcher:
    """검색 모드 하나를 `(query, top_k) -> list[dict]` 호출로 만들어 준다.

    `diversify`와 `min_chars`는 두 레인에 **같은 후처리**를 적용하기 위한 것이다.
    BM25는 `search_index` 안에서 이미 문서 다양화를 하는데 dense 경로에는 그 단계가
    없어서, 문서 단위 지표가 BM25에만 유리하게 기울어 있었다(2026-08-04 원인 분석).
    `diversify=True`인 bm25 모드는 `search_index`와 동일한 동작이다.

    `reranker`는 다양화 **이전**의 넓은 후보 풀에 적용한다. 문서당 캡이
    "그 문서의 어느 청크를 남길지"를 BM25 순위로 정해 버리기 전에, 리랭커가
    그 선택을 하게 하기 위해서다(2026-08-05, 실패 16문항이 전부 "정답 문서의
    엉뚱한 조각" 문제라는 8/4 밤 분석에 대응).

    `anchor_bm25_top1`은 그 위임의 상한이다: BM25 전체 1위 청크만은 리랭커가
    같은 문서의 다른 청크로 대체할 수 없다(다양화 캡에서 탈락 시 그 문서의
    최하위 선택분과 교체). 질문 표면형이 정답 조항과 어긋날 때 CE가 어휘
    정합 청크를 밀어내는 실패(grant_031, 2026-08-11)에 대응한다.
    """

    if mode not in MODES:
        raise ValueError(f"unknown retrieval mode: {mode} (expected {MODES})")

    index_path = Path(index_path)

    def _bm25_top1_chunk_id(rows: list[dict[str, Any]]) -> str | None:
        """후보 풀에서 BM25 레인 1위 청크를 찾는다 (하이브리드 융합 이후에도
        `retrieval.bm25.rank`가 레인 순위를 보존한다)."""
        best: tuple[int, str] | None = None
        for row in rows:
            bm25 = (row.get("retrieval") or {}).get("bm25") or {}
            rank = bm25.get("rank")
            if rank is None:
                continue
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id and (best is None or rank < best[0]):
                best = (int(rank), chunk_id)
        return best[1] if best else None

    def postprocess(
        rows: list[dict[str, Any]],
        top_k: int,
        preserve_chunk_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if min_chars > 0:
            rows = [
                row for row in rows
                if (row.get("char_count") or len(row.get("text") or "")) >= min_chars
            ]
        if diversify:
            return select_document_diverse_results(
                rows, top_k,
                max_chunks_per_document=max_chunks_per_document,
                preserve_chunk_id=preserve_chunk_id,
            )
        return rows[:top_k]

    if mode == "bm25":
        def bm25_searcher(query: str, top_k: int) -> list[dict[str, Any]]:
            rows = search_bm25_candidates(
                index_path,
                query,
                _candidate_limit(top_k),
                None,
                preview_chars=preview_chars,
                include_text=True,
            )
            # bm25 모드 후보는 리랭커 이전 순서가 곧 BM25 순위다.
            anchor = None
            if anchor_bm25_top1 and rows:
                anchor = str(rows[0].get("chunk_id") or "") or None
            if reranker is not None:
                rows = reranker(query, rows)
            return postprocess(rows, top_k, preserve_chunk_id=anchor)

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
        # CE 리랭커를 밖에서 붙일 때는 내부 기본 lexical 리랭커를 꺼서
        # 이중 리랭킹을 피한다. 그러면 CE가 받는 입력 순서 = 순수 RRF 순위라
        # fusion="rrf"가 (레인 융합 순위 × CE 순위)의 깨끗한 2단 융합이 된다.
        retriever = HybridRetriever(
            bm25_search=bm25_lane,
            dense_index=learned_index,
            reranker=None if reranker is not None else lexical_fallback_rerank,
        )

    def learned_searcher(query: str, top_k: int) -> list[dict[str, Any]]:
        # 후처리로 걸러낼 몫을 감안해 BM25와 같은 폭으로 후보를 넉넉히 받는다.
        width = _candidate_limit(top_k) if (diversify or min_chars > 0) else top_k
        rows = _as_dicts(retriever.search(query, top_k=width).hits)
        anchor = _bm25_top1_chunk_id(rows) if anchor_bm25_top1 else None
        if reranker is not None:
            rows = reranker(query, rows)
        return postprocess(rows, top_k, preserve_chunk_id=anchor)

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
