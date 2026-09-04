#!/usr/bin/env python3
"""Offline regression check for draft-answer sentence post-processing.

This test does not call a model or send document content anywhere. It treats each
evaluation case's official reference answer as a known-good draft, runs the legacy
and current sentence splitters, and compares retention of dates, amounts, times, and
other critical values. It is a safety check for the post-processor, not an end-to-end
generation score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_api import extract_critical_values, normalize_text, split_draft_claims  # noqa: E402


DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-eval.jsonl"
MAX_CLAIMS = 5


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def legacy_low_quality(text: str) -> bool:
    # Exact pre-fix behavior from search_api.py.
    from bm25_search import tokenize

    terms = tokenize(text, include_ngrams=False)
    if len(terms) >= 18 and len(set(terms)) / len(terms) < 0.42:
        return True
    if len(re.findall(r"[-_=]{4,}", text)) >= 1:
        return True
    return len(text) < 20


def legacy_split_candidate_sentences(text: str) -> list[str]:
    text = re.sub(r"([\[\(]page \d+[\]\)])", "", text, flags=re.IGNORECASE)
    rough_parts = re.split(
        r"\n+|(?<=[.!?。！？])\s+|(?<=다\.)\s+|(?<=임\.)\s+|(?<=음\.)\s+",
        text,
    )
    candidates: list[str] = []
    for part in rough_parts:
        subparts = re.split(r"\s+(?=[①②③④⑤⑥⑦⑧⑨⑩□ㅇ◦])", part)
        for subpart in subparts:
            subpart = normalize_text(subpart)
            if len(subpart) > 280:
                subpart = subpart[:280].rsplit(" ", 1)[0].strip()
            if subpart and not legacy_low_quality(subpart):
                candidates.append(subpart)
    return candidates


def legacy_split_draft_claims(text: str) -> list[str]:
    claims: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = normalize_text(raw_line)
        if line:
            claims.extend(legacy_split_candidate_sentences(line))
    return claims[:MAX_CLAIMS]


def critical_retention(reference: str, claims: list[str]) -> dict[str, Any]:
    expected = set(extract_critical_values(reference))
    retained = set(extract_critical_values("\n".join(claims)))
    found = expected & retained
    return {
        "expected": sorted(expected),
        "retained": sorted(found),
        "missing": sorted(expected - found),
        "rate": len(found) / len(expected) if expected else 1.0,
    }


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    improved = unchanged = regressed = 0
    for case in cases:
        reference = str(case.get("reference") or "")
        legacy_claims = legacy_split_draft_claims(reference)
        current_claims = split_draft_claims(reference)
        legacy_values = critical_retention(reference, legacy_claims)
        current_values = critical_retention(reference, current_claims)
        delta = current_values["rate"] - legacy_values["rate"]
        if delta > 0:
            improved += 1
        elif delta < 0:
            regressed += 1
        else:
            unchanged += 1
        rows.append(
            {
                "id": case.get("id"),
                "category": case.get("category"),
                "legacy_claims": legacy_claims,
                "current_claims": current_claims,
                "legacy": legacy_values,
                "current": current_values,
                "critical_value_retention_delta": delta,
                "claims_changed": legacy_claims != current_claims,
            }
        )

    return {
        "schema_version": 1,
        "method": "official reference as known-good draft; no model/API calls",
        "scope": "sentence splitting and critical-value preservation only",
        "cases": len(rows),
        "summary": {
            "improved": improved,
            "unchanged": unchanged,
            "regressed": regressed,
            "legacy_mean_critical_value_retention": mean(
                row["legacy"]["rate"] for row in rows
            ),
            "current_mean_critical_value_retention": mean(
                row["current"]["rate"] for row in rows
            ),
            "changed_claim_lists": sum(row["claims_changed"] for row in rows),
        },
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--only", help="comma-separated case IDs")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        cases = [case for case in cases if case.get("id") in wanted]
    result = evaluate(cases)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    summary = result["summary"]
    print(
        f"cases={result['cases']} improved/unchanged/regressed="
        f"{summary['improved']}/{summary['unchanged']}/{summary['regressed']}"
    )
    print(
        "critical-value retention "
        f"{summary['legacy_mean_critical_value_retention']:.4f} -> "
        f"{summary['current_mean_critical_value_retention']:.4f}"
    )
    if args.fail_on_regression and summary["regressed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
