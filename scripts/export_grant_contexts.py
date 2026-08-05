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
    ap.add_argument("--max-chunks-per-doc", type=int, default=2,
                    help="다양화 시 문서당 청크 상한 (기준선=2)")
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
        reranker = CrossEncoderReranker(args.rerank_model).rerank

    search = build_searcher(
        args.index, args.retrieval_mode, args.dense_artifact,
        diversify=not args.no_diversify, min_chars=args.min_chars,
        max_chunks_per_document=args.max_chunks_per_doc, reranker=reranker,
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
                hits = filter_hits(raw, scope, args.top_k, query)
            if args.parent_expand:
                contexts, expand_meta = expand_hits(
                    args.index, hits,
                    per_parent_chars=args.parent_chars,
                    total_chars=args.parent_total_chars,
                )
            else:
                contexts = [
                    (hit.get("text") or hit.get("preview") or "") for hit in hits
                ]
                expand_meta = None
            record = {
                "id": item["id"],
                "user_input": query,
                "retrieved_contexts": contexts,
                "reference": item["reference_answer"],
                "meta": {
                    "section": item["section"],
                    "scope": scope,
                    "parent_expand": expand_meta,
                    "sources": [
                        {
                            "institution": hit.get("institution"),
                            "file_name": hit.get("file_name"),
                            "chunk_id": hit.get("chunk_id"),
                            "char_count": hit.get("char_count"),
                            "ce_score": hit.get("cross_encoder_score"),
                        }
                        for hit in hits
                    ],
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
