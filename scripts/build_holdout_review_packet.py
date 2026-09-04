#!/usr/bin/env python3
"""Build a deterministic Markdown packet for independent holdout review.

This command deliberately produces no approval or sign-off manifest. It only
lays out the authored labels and source evidence beside empty Reviewer A/B
checklists so two humans can review the exact, SHA-bound cases file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .immutable_outputs import publish_immutable_texts
except ImportError:  # Direct execution and importlib-based tests.
    script_directory = str(Path(__file__).resolve().parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    from immutable_outputs import publish_immutable_texts  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "config" / "pnu-service-answer-holdout-v2.draft.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "holdout-v2-human-review.md"
REVIEW_CHECKS = (
    "source_verified",
    "gold_verified",
    "answerability_verified",
    "label_verified",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_jsonl_bytes(payload: bytes, *, source: str) -> list[dict[str, Any]]:
    """Decode object-per-line JSON and reject ambiguous duplicate case IDs."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: must be UTF-8: {exc}") from exc

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: case must be a JSON object")
        case_id = str(value.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"{source}:{line_number}: id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"{source}:{line_number}: duplicate id {case_id!r}")
        seen_ids.add(case_id)
        cases.append(value)
    if not cases:
        raise ValueError(f"{source}: contains no cases")
    return cases


def _text(value: Any, *, fallback: str = "—") -> str:
    rendered = str(value).strip() if value is not None else ""
    return rendered or fallback


def _table_text(value: Any) -> str:
    return _text(value).replace("|", "\\|").replace("\n", "<br>")


def _quote_block(value: Any) -> list[str]:
    text = _text(value, fallback="(비어 있음)")
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _list_items(values: Any, *, empty: str = "_(없음)_") -> list[str]:
    if not isinstance(values, list) or not values:
        return [empty]
    return [f"- {_text(value)}" for value in values]


def _local_source_link(
    source_path: Any,
    *,
    repo_root: Path,
    output_path: Path,
) -> str:
    raw = _text(source_path, fallback="")
    if not raw:
        return "_(source_path 없음)_"
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    exists = resolved.is_file()
    relative = Path(os.path.relpath(resolved, output_path.parent.resolve())).as_posix()
    # Angle-bracket destinations preserve spaces and parentheses in local paths.
    target = relative.replace("<", "%3C").replace(">", "%3E")
    label = raw.replace("[", "\\[").replace("]", "\\]")
    link = f"[{label}](<{target}>)"
    return link if exists else f"{link} **(로컬 파일 없음)**"


def _render_evidence(
    option: Mapping[str, Any],
    *,
    number: int,
    repo_root: Path,
    output_path: Path,
) -> list[str]:
    lines = [f"##### 근거 옵션 {number}", ""]
    lines.extend(
        [
            f"- 원문 파일: {_local_source_link(option.get('source_path'), repo_root=repo_root, output_path=output_path)}",
            f"- 문서 제목: {_text(option.get('source_title'))}",
            f"- document_id: `{_text(option.get('document_id'))}`",
            f"- source_sha256: `{_text(option.get('source_sha256'))}`",
        ]
    )
    source_url = _text(option.get("source_url"), fallback="")
    if source_url:
        safe_url = source_url.replace("<", "%3C").replace(">", "%3E")
        lines.append(f"- 게시 원본 URL: [공식 페이지](<{safe_url}>)")

    table_evidence = option.get("table_evidence")
    if isinstance(table_evidence, Mapping):
        headers = table_evidence.get("headers")
        if isinstance(headers, list) and headers:
            lines.append("- 표 헤더: " + " / ".join(_text(item) for item in headers))
        relations = table_evidence.get("row_relations")
        if isinstance(relations, list) and relations:
            lines.append("- 표 행 관계:")
            for relation in relations:
                if isinstance(relation, Mapping):
                    lines.append(
                        "  - "
                        f"{_text(relation.get('row_anchor'))} → "
                        f"{_text(relation.get('header'))} = "
                        f"{_text(relation.get('value'))}"
                    )

    lines.extend(["", "인용문:", ""])
    lines.extend(_quote_block(option.get("quote")))
    lines.append("")
    return lines


def _render_claims(
    claims: Any,
    *,
    kind: str,
    repo_root: Path,
    output_path: Path,
) -> list[str]:
    title = {
        "required": "필수 주장 (required claims)",
        "optional": "선택 주장 (optional claims)",
        "forbidden": "금지 주장 (forbidden claims)",
    }[kind]
    lines = [f"### {title}", ""]
    if not isinstance(claims, list) or not claims:
        suffix = " — 비답변/범위확인 oracle을 검토" if kind == "required" else ""
        lines.extend([f"_(없음{suffix})_", ""])
        return lines

    for number, claim in enumerate(claims, start=1):
        if not isinstance(claim, Mapping):
            lines.extend([f"#### {number}. 잘못된 claim 형식", "", _text(claim), ""])
            continue
        claim_id = _text(claim.get("claim_id"))
        lines.extend(
            [
                f"#### {number}. `{claim_id}`",
                "",
                f"- 설명: {_text(claim.get('description'))}",
            ]
        )
        critical_values = claim.get("critical_values")
        if kind != "forbidden" or critical_values:
            lines.append("- 핵심 값 (critical values):")
            if isinstance(critical_values, list) and critical_values:
                lines.extend(f"  - `{_text(value)}`" for value in critical_values)
            else:
                lines.append("  - _(없음)_")
        semantic_anchors = claim.get("semantic_anchors")
        if isinstance(semantic_anchors, list) and semantic_anchors:
            lines.append(
                "- 의미 앵커: " + " / ".join(f"`{_text(value)}`" for value in semantic_anchors)
            )
        if claim.get("evidence_type") is not None:
            lines.append(f"- evidence_type: `{_text(claim.get('evidence_type'))}`")
        lines.append("")

        evidence_options = claim.get("evidence_options")
        if isinstance(evidence_options, list) and evidence_options:
            for option_number, option in enumerate(evidence_options, start=1):
                if isinstance(option, Mapping):
                    lines.extend(
                        _render_evidence(
                            option,
                            number=option_number,
                            repo_root=repo_root,
                            output_path=output_path,
                        )
                    )
                else:
                    lines.extend([f"##### 근거 옵션 {option_number}", "", _text(option), ""])
        elif kind in {"required", "optional"}:
            lines.extend(["_근거 옵션 없음_", ""])
    return lines


def _render_challenge_oracle(case: Mapping[str, Any]) -> list[str]:
    lines = ["### 챌린지 oracle", ""]
    oracle = case.get("challenge_oracle")
    if not isinstance(oracle, Mapping):
        return lines + ["_(해당 없음)_", ""]

    lines.extend(["#### 반드시 해야 함 (must_do)", ""])
    lines.extend(_list_items(oracle.get("must_do")))
    lines.extend(["", "#### 하면 안 됨 (must_not_do)", ""])
    lines.extend(_list_items(oracle.get("must_not_do")))
    surface = _text(oracle.get("injection_surface"), fallback="")
    payload = _text(oracle.get("injection_payload"), fallback="")
    if surface or payload:
        lines.extend(["", "#### 주입 정보", ""])
        lines.append(f"- injection_surface: `{surface or '—'}`")
        if payload:
            lines.extend(["- injection_payload:", ""])
            lines.extend(_quote_block(payload))
    lines.append("")
    return lines


def _render_review_checklist(reviewer: str) -> list[str]:
    lines = [f"#### Reviewer {reviewer} (독립 검토)", "", "- reviewer_id: ____________________"]
    lines.extend(f"- [ ] `{check}`" for check in REVIEW_CHECKS)
    lines.extend(
        [
            "- [ ] 최종 판정: PASS (위 4개가 모두 PASS일 때만)",
            "- [ ] 최종 판정: REVISE",
            "- [ ] 최종 판정: BLOCK",
            "- 메모: ________________________________________________",
            "",
        ]
    )
    return lines


def render_packet(
    cases: Sequence[Mapping[str, Any]],
    *,
    cases_sha256: str,
    cases_path: Path,
    repo_root: Path,
    output_path: Path,
) -> str:
    """Render the complete packet without timestamps or environment-specific state."""

    split_counts = Counter(_text(case.get("split"), fallback="") for case in cases)
    answerable_counts = Counter(bool(case.get("answerable")) for case in cases)
    try:
        cases_display = cases_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        cases_display = cases_path.resolve().as_posix()

    lines = [
        "# PNU 서비스 답변 Holdout v2 — 사람 2인 독립 검토 패킷",
        "",
        "> 이 문서는 **읽기 전용 검토 보조 자료**이며 승인 또는 sign-off manifest가 아니다. "
        "Reviewer A와 B는 서로의 판정을 보지 않고 독립 검토하며, 실제 판정은 "
        "각자에게 배정된 SHA-bound JSON 응답 파일에만 기록한다.",
        "",
        "## 검토 대상 고정 정보",
        "",
        f"- 입력 파일: `{cases_display}`",
        f"- cases_sha256: `{cases_sha256}`",
        f"- 전체 문항: {len(cases)}",
        f"- holdout-core: {split_counts['holdout-core']}",
        f"- holdout-challenge: {split_counts['holdout-challenge']}",
        f"- answerable=true / false: {answerable_counts[True]} / {answerable_counts[False]}",
        "- 생성기: `scripts/build_holdout_review_packet.py` (시각·환경값 없는 결정적 출력)",
        "",
        "cases 파일이 한 바이트라도 바뀌면 이 SHA와 패킷을 다시 생성해야 한다. "
        "최종 sign-off는 반드시 같은 `cases_sha256`에 별도로 결합한다.",
        "",
        "## PASS 기준",
        "",
        "각 reviewer는 아래 4개 항목을 **모두 독립적으로 PASS**한 경우에만 해당 case를 PASS로 표시한다.",
        "",
        "1. `source_verified`: 답변 가능 문항은 링크한 로컬 원문에서 각 인용문과 표 행 관계를 직접 찾을 수 있고, "
        "문서 제목·경로·source SHA가 의도한 출처와 일치한다. 답변 불가 문항은 긍정 근거가 비어 있는 것이 의도와 맞다.",
        "2. `gold_verified`: 모든 required claim과 critical value가 원문에 의해 지지되고 충분하며, "
        "forbidden claim과 challenge oracle은 허위·누락·과잉 일반화를 정확히 차단한다.",
        "3. `answerability_verified`: 질문과 동결 코퍼스만 기준으로 `answerable` 값이 타당하다. "
        "실시간·개인정보·불명확한 버전/범위를 추측해서 답할 수 있다고 판정하지 않는다.",
        "4. `label_verified`: role, split, category, difficulty/challenge type, expected behavior가 질문과 gold의 실제 요구를 정확히 나타낸다.",
        "",
        "하나라도 실패하거나 판단 근거가 불충분하면 PASS가 아니라 REVISE 또는 BLOCK으로 표시하고 메모를 남긴다. "
        "Reviewer ID·체크·판정은 이 파일에 채우지 않고 각자의 별도 JSON 응답에 기록한다.",
        "",
        "## 문항별 검토",
        "",
    ]

    for index, case in enumerate(cases, start=1):
        case_id = _text(case.get("id"))
        lines.extend(
            [
                f"## {index:02d}. `{case_id}`",
                "",
                "| 필드 | 값 |",
                "|---|---|",
                f"| split | `{_table_text(case.get('split'))}` |",
                f"| category | `{_table_text(case.get('category'))}` |",
                f"| difficulty_type | `{_table_text(case.get('difficulty_type'))}` |",
                f"| challenge_type | `{_table_text(case.get('challenge_type'))}` |",
                f"| role | `{_table_text(case.get('role'))}` |",
                f"| answerable | `{str(bool(case.get('answerable'))).lower()}` |",
                f"| expected_behavior | `{_table_text(case.get('expected_behavior'))}` |",
                f"| family_id | `{_table_text(case.get('family_id'))}` |",
                f"| authoring_status | `{_table_text(case.get('authoring_status'))}` |",
                "",
                "### 질문",
                "",
            ]
        )
        lines.extend(_quote_block(case.get("query")))
        lines.extend(["", "### 기대 동작", ""])
        lines.append(f"`{_text(case.get('expected_behavior'))}`")
        lines.append("")
        lines.extend(
            _render_claims(
                case.get("required_claims"),
                kind="required",
                repo_root=repo_root,
                output_path=output_path,
            )
        )
        optional_claims = case.get("optional_claims")
        if isinstance(optional_claims, list) and optional_claims:
            lines.extend(
                _render_claims(
                    optional_claims,
                    kind="optional",
                    repo_root=repo_root,
                    output_path=output_path,
                )
            )
        lines.extend(
            _render_claims(
                case.get("forbidden_claims"),
                kind="forbidden",
                repo_root=repo_root,
                output_path=output_path,
            )
        )
        lines.extend(_render_challenge_oracle(case))
        lines.extend(["### 독립 검토 체크", ""])
        lines.extend(_render_review_checklist("A"))
        lines.extend(_render_review_checklist("B"))
        lines.extend(["---", ""])

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail when --output is missing or stale",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = args.cases.read_bytes()
        cases = load_jsonl_bytes(payload, source=str(args.cases))
        rendered = render_packet(
            cases,
            cases_sha256=sha256_bytes(payload),
            cases_path=args.cases,
            repo_root=REPO_ROOT,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"stale: cannot read {args.output}: {exc}", file=sys.stderr)
            return 1
        if current != rendered:
            print(
                f"stale: {args.output} does not match {args.cases}",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {args.output} matches {args.cases}")
        return 0

    try:
        publish_immutable_texts(
            {args.output: rendered},
            authoritative_path=args.output,
        )
    except (OSError, ValueError) as exc:
        print(f"error: cannot write {args.output}: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output} ({len(cases)} cases, "
        f"sha256={sha256_bytes(payload)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
