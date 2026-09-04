#!/usr/bin/env python3
"""부산대 서비스 평가셋(pnu-service-answer-eval.jsonl)의 무결성 검증.

평가셋은 LLM이 초안을 만들므로, 기준답안이 실제 문서에 근거하는지를
사람 눈이 아니라 기계 검사로 보증한다. 검사 항목:
  1. 스키마: 필수 필드, id 형식·중복, k 범위
  2. evidence: chunk_id가 인덱스에 존재하고 quote(20자 이상)가 해당
     청크 text의 연속 부분 문자열인지
  3. expected: 하나 이상의 근거 청크가 source_title_contains와
     source_host를 만족하는지 (교차 문서 답변은 다른 출처도 허용)
  4. 앵커 문서 중복 (경고 — 역할 변형 문항은 base와 앵커를 공유)

사용 예:
  python3 scripts/validate_service_eval.py \
      --cases config/pnu-service-answer-eval.jsonl \
      --index processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
REQUIRED_FIELDS = ("id", "query", "category", "expected", "reference", "evidence", "k")
MIN_QUOTE_CHARS = 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=REPO_ROOT / "config" / "pnu-service-answer-eval.jsonl")
    ap.add_argument(
        "--index", type=Path,
        default=REPO_ROOT / "processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite",
    )
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    cases = []
    for lineno, line in enumerate(args.cases.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: JSON 파싱 실패: {exc}")
    if errors:
        return report(len(cases), errors, warnings)

    seen_ids: set[str] = set()
    con = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    anchor_docs: dict[str, list[str]] = {}

    for case in cases:
        cid = str(case.get("id", "?"))
        for field in REQUIRED_FIELDS:
            if field not in case:
                errors.append(f"{cid}: 필수 필드 누락: {field}")
        if not ID_RE.match(cid):
            errors.append(f"{cid}: id 형식 위반")
        if cid in seen_ids:
            errors.append(f"{cid}: id 중복")
        seen_ids.add(cid)
        if not isinstance(case.get("k"), int) or not 1 <= case.get("k", 0) <= 20:
            errors.append(f"{cid}: k는 1~20 정수여야 함")
        expected = case.get("expected") or {}
        if not str(expected.get("source_title_contains", "")).strip():
            errors.append(f"{cid}: expected.source_title_contains 비어 있음")
        evidence = case.get("evidence") or []
        if not evidence:
            errors.append(f"{cid}: evidence 비어 있음")
        required_evidence_count = 0
        expected_source_matched = False
        for pos, item in enumerate(evidence, 1):
            if not isinstance(item, dict):
                errors.append(f"{cid} evidence#{pos}: 객체가 아님")
                continue
            required = item.get("required_for_answer", True)
            if not isinstance(required, bool):
                errors.append(
                    f"{cid} evidence#{pos}: required_for_answer는 boolean이어야 함"
                )
            elif required:
                required_evidence_count += 1
            chunk_id = str(item.get("chunk_id", ""))
            quote = str(item.get("quote", ""))
            if len(quote) < MIN_QUOTE_CHARS:
                errors.append(f"{cid} evidence#{pos}: quote가 {MIN_QUOTE_CHARS}자 미만")
            row = con.execute(
                "SELECT text, source_title, source_host, doc_id FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                errors.append(f"{cid} evidence#{pos}: chunk_id가 인덱스에 없음: {chunk_id}")
                continue
            if quote not in row["text"]:
                errors.append(f"{cid} evidence#{pos}: quote가 청크 본문의 부분 문자열이 아님")
            want_title = str(expected.get("source_title_contains", ""))
            want_host = str(expected.get("source_host", "") or "")
            title_matches = (
                not want_title
                or want_title in (row["source_title"] or "")
            )
            host_matches = (
                not want_host
                or want_host == (row["source_host"] or "")
            )
            expected_source_matched = (
                expected_source_matched
                or (title_matches and host_matches)
            )
            anchor_docs.setdefault(row["doc_id"], []).append(cid)
        if not required_evidence_count:
            errors.append(f"{cid}: 필수 evidence가 없음")
        if evidence and not expected_source_matched:
            errors.append(
                f"{cid}: expected.source_title_contains/source_host를 "
                "동시에 만족하는 근거 청크가 없음"
            )

    for doc_id, ids in sorted(anchor_docs.items()):
        distinct = sorted(set(ids))
        if len(distinct) > 1:
            warnings.append(f"앵커 문서 공유: {doc_id} ← {', '.join(distinct)}")

    con.close()
    return report(len(cases), errors, warnings)


def report(case_count: int, errors: list[str], warnings: list[str]) -> int:
    for message in errors:
        print(f"ERROR {message}")
    for message in warnings:
        print(f"WARN  {message}")
    print(f"cases={case_count} errors={len(errors)} warnings={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
