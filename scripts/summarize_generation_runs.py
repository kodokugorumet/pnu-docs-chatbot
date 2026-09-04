#!/usr/bin/env python3
"""생성 평가 런(judge.score JSONL)의 집계: n런 평균, 제외 문항, A/B 승패.

지금까지 확보50·n=3 평균·문항별 승패는 worklog에 수기로만 기록됐다.
이 스크립트가 그 계산을 재현 가능하게 만든다. 연구비 트랙과 서비스
트랙의 산출물(evaluate_grant_generation / evaluate_service_answers)
모두에서 동작한다 — 레코드에 id와 judge.score만 있으면 된다.

사용 예 (확보50 재현):
  python3 scripts/summarize_generation_runs.py \
      --runs processed/eval/20260811-gen-anchor14-run1.jsonl \
             processed/eval/20260811-gen-anchor14-run2.jsonl \
             processed/eval/20260811-gen-anchor14-run3.jsonl \
      --exclude grant_036,grant_037,grant_038

A/B 비교 (B가 도전자):
  python3 scripts/summarize_generation_runs.py \
      --runs <기준 run들> --runs-b <도전 run들> --per-question
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_scores(paths: list[Path], exclude: set[str]) -> dict[str, list[int]]:
    """id → 런별 score 목록. judge 실패(None)와 오류 레코드는 건너뛴다."""
    scores: dict[str, list[int]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("id")
            if qid is None or qid in exclude:
                continue
            score = (rec.get("judge") or {}).get("score")
            if score is None:
                continue
            scores.setdefault(qid, []).append(int(score))
    return scores


def per_run_means(paths: list[Path], exclude: set[str]) -> list[tuple[str, float, int]]:
    out = []
    for path in paths:
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") in exclude:
                continue
            score = (rec.get("judge") or {}).get("score")
            if score is not None:
                values.append(int(score))
        mean = sum(values) / len(values) if values else float("nan")
        out.append((path.name, mean, len(values)))
    return out


def question_means(scores: dict[str, list[int]]) -> dict[str, float]:
    return {qid: sum(vals) / len(vals) for qid, vals in scores.items() if vals}


def load_retrieval(paths: list[Path], exclude: set[str]) -> dict[str, dict]:
    """id → 검색 진단 (서비스 트랙 레코드에만 존재. 같은 구성의 런은 검색이
    동일하므로 첫 등장 레코드를 쓴다)."""
    out: dict[str, dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("id")
            if qid is None or qid in exclude or qid in out:
                continue
            hit = rec.get("retrieval_hit")
            if hit is None:
                continue
            ev = rec.get("evidence_hit") or {}
            out[qid] = {"hit": bool(hit.get("matched")), "rank": hit.get("rank"),
                        "ev": ev.get("matched")}
    return out


def print_retrieval(tag: str, info: dict[str, dict]) -> None:
    if not info:
        return
    hits = sum(1 for v in info.values() if v["hit"])
    mrr = sum(1.0 / v["rank"] if v["rank"] else 0.0 for v in info.values()) / len(info)
    line = f"  {tag} retrieval: hit {hits}/{len(info)} = {hits / len(info):.3f}, MRR {mrr:.3f}"
    ev_known = [v for v in info.values() if v["ev"] is not None]
    if ev_known:
        ev_hits = sum(1 for v in ev_known if v["ev"])
        line += f", evidence chunk {ev_hits}/{len(ev_known)} = {ev_hits / len(ev_known):.3f}"
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--runs-b", type=Path, nargs="+", help="A/B 비교 시 도전자 run들")
    ap.add_argument("--exclude", default="", help="쉼표로 구분한 제외 문항 id")
    ap.add_argument("--per-question", action="store_true")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    exclude = {token.strip() for token in args.exclude.split(",") if token.strip()}

    scores_a = load_scores(args.runs, exclude)
    means_a = question_means(scores_a)
    overall_a = sum(means_a.values()) / len(means_a) if means_a else float("nan")

    print(f"[A] runs={len(args.runs)} questions={len(means_a)} excluded={len(exclude)}")
    for name, mean, count in per_run_means(args.runs, exclude):
        print(f"  {name}: mean={mean:.3f} (n={count})")
    print(f"  A overall (문항 평균의 평균) = {overall_a:.4f}")
    retrieval_a = load_retrieval(args.runs, exclude)
    print_retrieval("A", retrieval_a)

    payload: dict = {
        "excluded": sorted(exclude),
        "a": {"runs": [str(p) for p in args.runs], "overall": overall_a,
              "per_question": dict(sorted(means_a.items()))},
    }

    if args.runs_b:
        scores_b = load_scores(args.runs_b, exclude)
        means_b = question_means(scores_b)
        overall_b = sum(means_b.values()) / len(means_b) if means_b else float("nan")
        common = sorted(set(means_a) & set(means_b))
        wins = [q for q in common if means_b[q] > means_a[q]]
        losses = [q for q in common if means_b[q] < means_a[q]]
        ties = [q for q in common if means_b[q] == means_a[q]]
        delta_common = (
            sum(means_b[q] - means_a[q] for q in common) / len(common) if common else float("nan")
        )
        print(f"[B] runs={len(args.runs_b)} questions={len(means_b)}")
        for name, mean, count in per_run_means(args.runs_b, exclude):
            print(f"  {name}: mean={mean:.3f} (n={count})")
        print(f"  B overall = {overall_b:.4f}")
        retrieval_b = load_retrieval(args.runs_b, exclude)
        print_retrieval("B", retrieval_b)
        if retrieval_a and retrieval_b:
            both = set(retrieval_a) & set(retrieval_b)
            gained = sorted(q for q in both if retrieval_b[q]["hit"] and not retrieval_a[q]["hit"])
            lost = sorted(q for q in both if retrieval_a[q]["hit"] and not retrieval_b[q]["hit"])
            if gained:
                print(f"  B에서 새로 적중: {', '.join(gained)}")
            if lost:
                print(f"  B에서 적중 상실: {', '.join(lost)}")
        print(f"[A/B] 공통 {len(common)}문항: B승 {len(wins)} / 무 {len(ties)} / B패 {len(losses)}"
              f", 공통 문항 Δ(B-A)={delta_common:+.4f}")
        only_a = sorted(set(means_a) - set(means_b))
        only_b = sorted(set(means_b) - set(means_a))
        if only_a:
            print(f"  A에만 있는 문항 {len(only_a)}개: {', '.join(only_a[:10])}{' …' if len(only_a) > 10 else ''}")
        if only_b:
            print(f"  B에만 있는 문항 {len(only_b)}개: {', '.join(only_b[:10])}{' …' if len(only_b) > 10 else ''}")
        if args.per_question:
            for qid in common:
                if means_a[qid] != means_b[qid]:
                    print(f"    {qid}: {means_a[qid]:.2f} -> {means_b[qid]:.2f} ({means_b[qid] - means_a[qid]:+.2f})")
        payload["b"] = {"runs": [str(p) for p in args.runs_b], "overall": overall_b,
                        "per_question": dict(sorted(means_b.items()))}
        payload["ab"] = {"common": len(common), "wins": wins, "ties": len(ties),
                         "losses": losses, "delta_common": delta_common}
    elif args.per_question:
        for qid, mean in sorted(means_a.items()):
            print(f"    {qid}: {mean:.2f}")

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
