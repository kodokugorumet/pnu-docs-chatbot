#!/usr/bin/env python3
"""평가셋 53문항에 대해 검색 컨텍스트를 뽑아 RAGAS 입력 형태로 저장한다.

검색과 채점을 분리하는 이유가 둘이다. 첫째, ragas는 langchain 계열 의존성 때문에
별도 venv에 격리돼 있어 임베딩 계열(sentence-transformers)과 같은 환경에 두기 어렵다.
둘째, 컨텍스트를 파일로 남겨두면 채점을 다시 돌릴 때 검색을 반복하지 않아도 되고
사람이 직접 열어볼 수도 있다.

출력 한 줄이 RAGAS 한 샘플이다:
  {"id", "user_input", "retrieved_contexts": [...], "reference", "meta": {...}}

사용 예 (임베딩 venv에서 실행):
  .parser-tools/venvs/embedding/bin/python scripts/export_grant_contexts.py \
      --index processed/index/grant-rules-20260804.sqlite \
      --out processed/eval/20260804-ctx-challenger.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grant_retrieval import MODES, build_searcher  # noqa: E402
from rag.grant_router import filter_hits, route  # noqa: E402
from rag.parent_expand import expand_hits  # noqa: E402


def _content_terms(query: str, minimum_length: int = 4) -> list[str]:
    """질의에서 신규성 판정에 쓸 내용어 후보 (한글 4자 이상 토큰)."""
    terms = []
    for token in query.split():
        stripped = "".join(ch for ch in token if "가" <= ch <= "힣")
        if len(stripped) >= minimum_length:
            terms.append(stripped)
    return terms


def _longest_known_prefix(term: str, texts: list[str], minimum_length: int = 4) -> str | None:
    """`texts` 어딘가에 등장하는 term의 최장 접두(조사 제거 근사). 없으면 None."""
    for size in range(len(term), minimum_length - 1, -1):
        prefix = term[:size]
        if any(prefix in text for text in texts):
            return prefix
    return None


def collect_fallback_rows(
    query: str,
    primary_hits: list[dict],
    fallback_search,
    *,
    slots: int,
    novel_slots: int,
) -> list[dict]:
    """폴백층(D50) 진입 행 수집 — 두 트리거 모두 1군을 밀어내지 않는다.

    ① 잔여 슬롯: 1군이 top_k 미달일 때 부족분을 채운다 (실측상 희귀).
    ② 신규 용어: 질의 내용어가 폴백층에는 있는데 1군 컨텍스트에 전혀 없으면
       해당 용어를 담은 폴백 행을 최대 novel_slots개 **추가**한다. 폴백층이
       유일한 근거 공급원인 질문(예: `학술활동수당`)을 위한 경로다.
    """
    taken: list[dict] = []
    seen = {h.get("chunk_id") for h in primary_hits}
    candidates = fallback_search(query, max(slots, novel_slots, 0) + 8)

    def take(row: dict) -> None:
        row["retrieval_tier"] = "fallback"
        taken.append(row)
        seen.add(row.get("chunk_id"))

    for row in candidates:
        if len(taken) >= max(0, slots):
            break
        if row.get("chunk_id") not in seen:
            take(row)

    if novel_slots > 0:
        primary_texts = [(h.get("text") or h.get("preview") or "") for h in primary_hits]
        fallback_texts = [(r.get("text") or r.get("preview") or "") for r in candidates]
        novel_prefixes = []
        for term in _content_terms(query):
            prefix = _longest_known_prefix(term, fallback_texts)
            if prefix and not any(prefix in text for text in primary_texts):
                novel_prefixes.append(prefix)
        if novel_prefixes:
            added = 0
            for row in candidates:
                if added >= novel_slots:
                    break
                if row.get("chunk_id") in seen:
                    continue
                text = row.get("text") or row.get("preview") or ""
                if any(prefix in text for prefix in novel_prefixes):
                    take(row)
                    added += 1

    return taken


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path,
                    default=Path("processed/index/grant-rules-20260804.sqlite"))
    ap.add_argument("--eval-file", type=Path,
                    default=Path("config/grant-rules-answer-eval.jsonl"))
    ap.add_argument("--retrieval-mode", choices=MODES, default="bm25")
    ap.add_argument("--dense-artifact", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=8,
                    help="LLM에 넘길 컨텍스트 청크 수 (생성 평가와 맞출 것)")
    ap.add_argument("--candidates", type=int, default=30,
                    help="라우터 필터 이전 후보 폭")
    ap.add_argument("--no-routing", action="store_true")
    ap.add_argument("--no-diversify", action="store_true")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--rerank-model", default=None,
                    help="cross-encoder 리랭커 모델명(HF). 다양화 이전 후보 풀에 적용")
    ap.add_argument("--rerank-fusion", choices=["rrf"], default=None,
                    help="CE 단독 재정렬 대신 BM25 순위와 RRF 융합")
    ap.add_argument("--max-chunks-per-doc", type=int, default=2,
                    help="다양화 시 문서당 청크 상한 (기준선=2)")
    ap.add_argument("--anchor-bm25-top1", action="store_true",
                    help="BM25 전체 1위 청크는 다양화 캡에서 탈락 불가 "
                         "(같은 문서 최하위 선택분과 교체, grant_031 대응)")
    ap.add_argument("--fallback-docs", type=Path, default=None,
                    help="[실험용] 같은 인덱스 안에서 문서를 후보·라우터 "
                         "단계 강등. 레인 순위 교란이 남아 정본 보존 불가 "
                         "(D50 1·2차 시도) — 정본 경로는 --fallback-index")
    ap.add_argument("--fallback-index", type=Path, default=None,
                    help="폴백층 전용 BM25 인덱스 (D50 채택안). 1군 검색이 "
                         "top_k를 못 채운 문항에만 잔여 슬롯을 채운다 — "
                         "1군 인덱스가 분리돼 있어 벤치마크 컨텍스트 불변")
    ap.add_argument("--fallback-novel-slots", type=int, default=0,
                    help="[실험용] 신규 용어 트리거로 '추가'되는 폴백 행 상한. "
                         "D50 A/B에서 순 회귀(-1.00/11문항)로 기각 — 기본 0. "
                         "추가 진입조차 내용 충돌로 생성을 오염시킨다")
    ap.add_argument("--parent-expand", action="store_true",
                    help="최종 히트를 section_path 부모로 확장 (parent-child)")
    ap.add_argument("--parent-chars", type=int, default=3000,
                    help="부모 하나의 글자 예산")
    ap.add_argument("--parent-total-chars", type=int, default=13000,
                    help="문항 전체 글자 예산 (기준선 중앙 10,061자와 동급 유지)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    items = [
        json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reranker = None
    if args.rerank_model:
        # sentence-transformers 의존이라 요청 시에만 import (ragas venv 오염 방지)
        from rag.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(
            args.rerank_model, fusion=args.rerank_fusion
        ).rerank

    fallback_files: frozenset[str] | None = None
    if args.fallback_docs:
        payload = json.loads(args.fallback_docs.read_text(encoding="utf-8"))
        fallback_files = frozenset(payload["fallback_file_names"])

    search = build_searcher(
        args.index, args.retrieval_mode, args.dense_artifact,
        diversify=not args.no_diversify, min_chars=args.min_chars,
        max_chunks_per_document=args.max_chunks_per_doc, reranker=reranker,
        anchor_bm25_top1=args.anchor_bm25_top1,
        demote_files=fallback_files,
    )

    fallback_search = None
    if args.fallback_index:
        # 폴백층은 별도 인덱스라 1군 레인 순위·IDF에 영향이 없다 (D50).
        # bm25 단독 조회면 충분 — 잔여 슬롯 보충용이다.
        fallback_search = build_searcher(
            args.fallback_index, "bm25",
            diversify=not args.no_diversify,
            max_chunks_per_document=args.max_chunks_per_doc,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as sink:
        for item in items:
            query = item["query"]
            raw = search(query, args.candidates)
            if args.no_routing:
                scope = None
                hits = raw[: args.top_k]
            else:
                scope = route(query)
                hits = filter_hits(raw, scope, args.top_k, query,
                                   fallback_files=fallback_files)
            fallback_rows: list[dict] = []
            if fallback_search is not None:
                fallback_rows = collect_fallback_rows(
                    query, hits, fallback_search,
                    slots=args.top_k - len(hits),
                    novel_slots=args.fallback_novel_slots,
                )
            def hit_source(hit: dict) -> dict:
                return {
                    "institution": hit.get("institution"),
                    "file_name": hit.get("file_name"),
                    "chunk_id": hit.get("chunk_id"),
                    "document_id": hit.get("document_id"),
                    "char_count": hit.get("char_count"),
                    "ce_score": hit.get("cross_encoder_score"),
                }

            if args.parent_expand:
                contexts, expand_meta = expand_hits(
                    args.index, hits,
                    per_parent_chars=args.parent_chars,
                    total_chars=args.parent_total_chars,
                )
                # sources를 컨텍스트와 1:1 정렬 — 병합으로 컨텍스트가 히트보다
                # 적어질 수 있어, anchor chunk_id로 히트를 역참조해 재구성한다.
                # (생성 평가가 컨텍스트별 메타를 zip으로 쓰기 위한 전제)
                by_id = {h.get("chunk_id"): h for h in hits}
                sources = []
                for m in expand_meta:
                    src = hit_source(by_id.get(m["chunk_id"], {}))
                    src.update({"n_chunks": m["n_chunks"], "chars": m["chars"],
                                "mode": m["mode"]})
                    sources.append(src)
            else:
                contexts = [
                    (hit.get("text") or hit.get("preview") or "") for hit in hits
                ]
                expand_meta = None
                sources = [hit_source(h) for h in hits]
            # 폴백 행은 1군 컨텍스트 뒤에 "추가"한다 — 1군을 밀어내는 진입
            # 경로가 없으므로 D46·D49의 밀어내기 회귀가 구조적으로 불가능.
            for row in fallback_rows:
                contexts.append(row.get("text") or row.get("preview") or "")
                src = hit_source(row)
                src["tier"] = "fallback"
                sources.append(src)
            record = {
                "id": item["id"],
                "user_input": query,
                "retrieved_contexts": contexts,
                "reference": item["reference_answer"],
                "meta": {
                    "section": item["section"],
                    "scope": scope,
                    "parent_expand": expand_meta,
                    "sources": sources,
                },
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"mode={args.retrieval_mode} routing={not args.no_routing} "
          f"top_k={args.top_k} index={args.index.name} "
          f"rerank={args.rerank_model or 'none'} cap={args.max_chunks_per_doc} "
          f"parent={args.parent_expand}")
    print(f"wrote {written} samples → {args.out}")


if __name__ == "__main__":
    main()
