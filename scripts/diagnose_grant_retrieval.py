#!/usr/bin/env python3
"""검색 모드별 실패 원인 진단.

Hit@k 같은 문서 단위 지표는 "정답 문서가 상위에 왔는가"만 본다. 생성 품질을
좌우하는 건 그게 아니라 **근거 숫자가 든 청크가 컨텍스트에 들어왔는가**이고,
문서 단위 지표는 검색 경로마다 다른 후처리(문서 다양화 등)에 크게 흔들린다.
그래서 평가 점수와 별개로 다음 네 가지를 함께 측정한다.

1. 근거 커버리지 — 기준답안의 수치가 실제 검색 컨텍스트에 등장하는가
2. 청크 길이 분포 — dense는 길이 사전확률이 없어 짧은 청크를 과다 검색한다
3. 문서/기관 다양성 — `search_index`의 문서 다양화가 BM25에만 붙어 있어
   문서 단위 비교가 불공정해진다. 그 격차를 수치로 드러낸다
4. 라우터 폴백 발동률 — `filter_hits`는 스코프 내 결과가 부족하면 원본으로
   보충한다. 어휘가 희소한 BM25는 이 구제를 자주 받지만 dense는 거의 못 받는다

사용 예:
  python scripts/diagnose_grant_retrieval.py \
      --index processed/index/grant-rules-20260803.sqlite \
      --learned-root processed/index/learned-dense/grant-rules-20260803 \
      --generation-eval processed/eval/20260803-grant-generation-v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grant_retrieval import build_searcher  # noqa: E402
from rag.grant_router import route, filter_hits  # noqa: E402

# 기준답안에서 뽑는 근거 토큰: 금액·비율·기간처럼 답을 확정하는 수치.
NUMERIC = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:원|만원|천원|%|퍼센트|일|개월|년|시간|회|배)")

SHORT_CHUNK_CHARS = 100


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _modes(learned_root: Path) -> dict[str, tuple[str, Path | None]]:
    kure = learned_root / "kure-v1"
    snow = learned_root / "snowflake-arctic-l-v2-ko"
    modes: dict[str, tuple[str, Path | None]] = {"bm25": ("bm25", None)}
    if kure.is_dir():
        modes["kure-dense"] = ("dense", kure)
        modes["kure-hybrid"] = ("hybrid", kure)
    if snow.is_dir():
        modes["snow-dense"] = ("dense", snow)
        modes["snow-hybrid"] = ("hybrid", snow)
    return modes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path,
                    default=Path("processed/index/grant-rules-20260803.sqlite"))
    ap.add_argument("--learned-root", type=Path,
                    default=Path("processed/index/learned-dense/grant-rules-20260803"))
    ap.add_argument("--eval-file", type=Path,
                    default=Path("config/grant-rules-answer-eval.jsonl"))
    ap.add_argument("--generation-eval", type=Path,
                    default=Path("processed/eval/20260803-grant-generation-v2.jsonl"),
                    help="근거 커버리지 대상(생성 0점 문항)을 뽑을 채점 결과")
    ap.add_argument("--top-k", type=int, default=8,
                    help="근거 커버리지 측정에 쓸 컨텍스트 크기")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    evalset = {
        json.loads(line)["id"]: json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    modes = _modes(args.learned_root)
    report: dict[str, dict] = {name: {} for name in modes}

    # --- 인덱스 자체의 청크 길이 분포 (비교 기준선) ---
    connection = sqlite3.connect(str(args.index))
    try:
        corpus_lengths = [row[0] or 0 for row in connection.execute(
            "SELECT char_count FROM chunks"
        )]
    finally:
        connection.close()
    corpus_short = sum(1 for value in corpus_lengths if value < SHORT_CHUNK_CHARS)
    print(f"인덱스 {len(corpus_lengths):,} 청크: 중앙값 {statistics.median(corpus_lengths):.0f}자, "
          f"{SHORT_CHUNK_CHARS}자 미만 {corpus_short}개 "
          f"({corpus_short / max(len(corpus_lengths), 1) * 100:.1f}%)")

    # --- 근거 커버리지 대상: 생성 0점이면서 기준답안에 수치가 있는 문항 ---
    targets: dict[str, set[str]] = {}
    if args.generation_eval.is_file():
        scored = [
            json.loads(line)
            for line in args.generation_eval.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for record in scored:
            if (record.get("judge") or {}).get("score") != 0:
                continue
            item = evalset.get(record["id"])
            if not item:
                continue
            numbers = {_squash(t) for t in NUMERIC.findall(item["reference_answer"])}
            if numbers:
                targets[record["id"]] = numbers
        print(f"근거 커버리지 대상: 생성 0점 중 기준답안에 수치가 있는 {len(targets)}문항\n")
    else:
        print(f"(생성 채점 결과 없음: {args.generation_eval} — 근거 커버리지는 건너뜀)\n")

    for name, (mode, artifact) in modes.items():
        search = build_searcher(args.index, mode, artifact)
        lengths: list[int] = []
        doc_counts: list[int] = []
        inst_counts: list[int] = []
        fallbacks = 0
        covered: list[str] = []
        per_question: dict[str, str] = {}

        for qid, item in evalset.items():
            query = item["query"]
            scope = route(query)
            raw = search(query, 30)

            # 라우터 폴백: 스코프 내 후보가 top_k에 못 미치면 원본으로 보충된다.
            if scope is not None:
                scoped = [h for h in raw if (h.get("institution") or "") in scope]
                if len(scoped) < 5:
                    fallbacks += 1

            ranked = filter_hits(raw, scope, 5, query)
            lengths += [h.get("char_count") or 0 for h in ranked]
            doc_counts.append(len({h.get("document_id") for h in ranked}))
            inst_counts.append(len({h.get("institution") for h in ranked}))

            if qid in targets:
                context = filter_hits(raw, scope, args.top_k, query)
                blob = _squash(" ".join(
                    (h.get("text") or h.get("preview") or "") for h in context
                ))
                found = sum(1 for number in targets[qid] if number in blob)
                per_question[qid] = f"{found}/{len(targets[qid])}"
                if found:
                    covered.append(qid)

        short = sum(1 for value in lengths if value < SHORT_CHUNK_CHARS)
        report[name] = {
            "evidence_covered": len(covered),
            "evidence_total": len(targets),
            "evidence_per_question": per_question,
            "median_chars": statistics.median(lengths) if lengths else 0,
            "short_chunk_ratio": short / max(len(lengths), 1),
            "distinct_documents": statistics.mean(doc_counts) if doc_counts else 0,
            "distinct_institutions": statistics.mean(inst_counts) if inst_counts else 0,
            "router_fallbacks": fallbacks,
        }

    header = (f"{'mode':14s}{'근거':>10s}{'중앙길이':>10s}{'짧은청크':>10s}"
              f"{'문서수':>9s}{'기관수':>9s}{'폴백':>7s}")
    print(header)
    print("-" * len(header))
    for name, values in report.items():
        print(f"{name:14s}"
              f"{values['evidence_covered']:>7d}/{values['evidence_total']:<2d}"
              f"{values['median_chars']:>9.0f}자"
              f"{values['short_chunk_ratio'] * 100:>9.1f}%"
              f"{values['distinct_documents']:>9.2f}"
              f"{values['distinct_institutions']:>9.2f}"
              f"{values['router_fallbacks']:>7d}")

    print("\n읽는 법:")
    print("  근거    = 기준답안 수치가 컨텍스트에 최소 1개 등장한 문항 수 (높을수록 좋음)")
    print("  짧은청크 = 검색 결과 중 100자 미만 비율 (dense의 길이 편향 지표)")
    print("  문서수  = top-5 안의 서로 다른 문서 수 (BM25만 문서 다양화가 적용된다)")
    print("  폴백    = 스코프 내 후보 부족으로 원본 보충이 발동한 문항 수")

    if args.json_out:
        args.json_out.write_text(
            json.dumps({"index": str(args.index), "modes": report},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
