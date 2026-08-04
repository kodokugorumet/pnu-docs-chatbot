#!/usr/bin/env python3
"""53문항 연구비 규정 평가셋에 대한 BM25 검색 평가.

각 문항의 출처 섹션(section)을 기대 문서 조건(기관/제목 부분일치)으로
매핑해 Hit@k와 MRR을 계산한다. 기대 문서가 아직 corpus에 없는 섹션은
`uncovered`로 따로 집계한다.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm25_search import search_index  # noqa: E402
from rag.grant_router import route, filter_hits  # noqa: E402

# section 부분 문자열 → (기관 목록, 제목 부분 문자열 목록). 둘 중 하나라도
# 일치하면 관련 문서로 본다. None 기관은 기관 무관.
SECTION_RULES: list[tuple[str, list[str] | None, list[str]]] = [
    ("공무원 여비규정", ["국가법령"], ["공무원 여비"]),
    ("부산대학교 연구비 관리", ["부산대학교 산학협력단"], ["연구비 관리 세부 지침", "연구비 관리에 관한 규정"]),
    ("국립수산과학원", ["국립수산과학원"], ["연구용역비"]),
    ("인문사회분야", ["한국연구재단"], ["인문사회"]),
    ("식품의약품안전처", ["식품의약품안전처"], ["연구개발비관리지침"]),
    ("중소기업기술정보진흥원", ["중소벤처기업부"], ["관리지침"]),
    ("산업기술혁신사업", ["산업통상자원부"], ["공통 운영요령"]),
    ("성보호에 관한 법률", ["국가법령"], ["성보호"]),
    # 국연법 계열: 법령 본문·시행령·시행규칙 또는 과기부 사용기준 고시
    ("국가연구개발혁신법", ["국가법령", "과학기술정보통신부"], ["국가연구개발혁신법", "연구개발비 사용 기준"]),
    ("한국연구재단 - 국가연구개발혁신법", ["국가법령", "과학기술정보통신부"], ["국가연구개발혁신법", "연구개발비 사용 기준"]),
]

UNCOVERED_MARKERS = ["극지연구소", "가스공사", "전력연구원", "한국전력", "창의재단", "환경부", "선박해양플랜트"]

# 문항별 추가 인정 조건. 기준답안이 복수 출처를 인용하거나(002, 051)
# "자체규정을 따르라"가 정답인 경우(014) 해당 문서도 관련 문서로 본다.
EXTRA_ACCEPT: dict[str, list[tuple[list[str], list[str]]]] = {
    "grant_002": [(["부산대학교 산학협력단"], ["연구비 관리 세부 지침"])],
    "grant_051": [(["과학기술정보통신부"], ["연구개발비 사용 기준"])],
    "grant_014": [(["부산대학교 산학협력단"], ["취업규칙"])],
}


def expected_for(section: str):
    for marker in UNCOVERED_MARKERS:
        if marker in section:
            return None  # corpus 미수집
    for marker, insts, titles in SECTION_RULES:
        if marker in section:
            return insts, titles
    # 기본: 국연법 계열로 취급
    return (["국가법령", "과학기술정보통신부"], ["국가연구개발혁신법", "연구개발비 사용 기준"])


def hit_rank(hits, accept_rules):
    for rank, h in enumerate(hits, start=1):
        title = (h.get("source_title") or "") + " " + (h.get("file_name") or "")
        inst = h.get("institution") or ""
        for insts, titles in accept_rules:
            inst_ok = insts is None or inst in insts
            title_ok = any(t in title for t in titles)
            if inst_ok and title_ok:
                return rank
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=Path("processed/index/grant-rules-20260803.sqlite"))
    ap.add_argument("--eval-file", type=Path, default=Path("config/grant-rules-answer-eval.jsonl"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--routing", action="store_true", help="기관 스코프 라우팅 적용")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in args.eval_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    covered = uncovered = hits_at_k = 0
    rr_sum = 0.0
    per_section: dict[str, list[int]] = {}
    failures = []
    results = []

    for it in items:
        exp = expected_for(it["section"])
        if exp is None:
            uncovered += 1
            results.append({"id": it["id"], "status": "uncovered", "section": it["section"]})
            continue
        covered += 1
        accept = [exp] + EXTRA_ACCEPT.get(it["id"], [])
        if args.routing:
            scope = route(it["query"])
            raw_hits = search_index(args.index, it["query"], max(30, args.top_k * 6), None)
            hits = filter_hits(raw_hits, scope, args.top_k, it["query"])
        else:
            hits = search_index(args.index, it["query"], args.top_k, None)
        rank = hit_rank(hits, accept)
        key = it["section"].split("-")[0].split("–")[0].strip()[:24]
        per_section.setdefault(key, []).append(1 if rank else 0)
        if rank:
            hits_at_k += 1
            rr_sum += 1.0 / rank
        else:
            failures.append((it["id"], it["section"][:40], it["query"][:60],
                             [(h.get("institution"), (h.get("source_title") or h.get("file_name") or "")[:40]) for h in hits[:3]]))
        results.append({"id": it["id"], "status": "hit" if rank else "miss", "rank": rank,
                        "section": it["section"], "query": it["query"]})

    print(f"covered {covered} / uncovered {uncovered} (total {len(items)})")
    print(f"Hit@{args.top_k}: {hits_at_k}/{covered} = {hits_at_k/max(covered,1):.3f}")
    print(f"MRR: {rr_sum/max(covered,1):.4f}")
    print("\nper-section:")
    for k, v in sorted(per_section.items(), key=lambda x: -len(x[1])):
        print(f"  {sum(v)}/{len(v)}  {k}")
    if failures:
        print("\nFAILURES:")
        for fid, sec, q, top in failures:
            print(f"- {fid} [{sec}] {q}")
            for inst, t in top:
                print(f"    got: [{inst}] {t}")
    if args.json_out:
        args.json_out.write_text(json.dumps({
            "index": str(args.index), "top_k": args.top_k,
            "covered": covered, "uncovered": uncovered,
            "hit_at_k": hits_at_k, "mrr": rr_sum / max(covered, 1),
            "results": results,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
