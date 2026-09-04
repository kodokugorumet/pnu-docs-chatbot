#!/usr/bin/env python3
"""Audit frozen service-rule activation from answer artifacts.

The analyzer never imports the serving modules and never calls the service.
It consumes only saved answer JSONL traces.  A rule can be:

* observed_active / observed_inactive: the saved trace exposes the decision;
* trace_absent: the frozen service did not record a rule identifier;
* trigger_possible: a separate query/artifact-side inference, never presented
  as proof that the rule actually changed retrieval or postprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


RULE_ID_RE = re.compile(
    r"^\| `(?P<id>(?:QRY-NORM|RANK|EXP|FACET|GEN|POST)-\d{3})` \|"
)
CLASS_RE = re.compile(r"\| (?P<classification>일반|표적) \|")
TARGET_RE = re.compile(r"`(svc_[a-z0-9_]+)`")


class ActivationError(ValueError):
    """Raised when an input or output violates the audit contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: Path) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = RULE_ID_RE.match(line)
        if match is None:
            continue
        rule_id = match.group("id")
        if rule_id in seen:
            raise ActivationError(f"duplicate inventory rule id: {rule_id}")
        class_match = CLASS_RE.search(line)
        if class_match is None:
            raise ActivationError(
                f"inventory row {line_number} has no 일반/표적 classification"
            )
        seen.add(rule_id)
        rules.append(
            {
                "rule_id": rule_id,
                "classification": (
                    "general"
                    if class_match.group("classification") == "일반"
                    else "targeted"
                ),
                "target_case_ids": sorted(set(TARGET_RE.findall(line))),
                "inventory_line": line_number,
            }
        )
    if not rules:
        raise ActivationError(f"no rule rows found in inventory: {path}")
    return rules


def load_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[tuple[str, str, str]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ActivationError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ActivationError(f"{path}:{line_number}: expected object")
                case_id = str(record.get("case_id") or record.get("id") or "")
                condition_id = str(record.get("condition_id") or "unknown")
                generation_run_id = str(
                    record.get("generation_run_id") or "unknown"
                )
                if not case_id:
                    raise ActivationError(f"{path}:{line_number}: missing case_id")
                key = (case_id, condition_id, generation_run_id)
                if key in seen_ids:
                    raise ActivationError(
                        "duplicate case/condition/run across inputs: " + "/".join(key)
                    )
                seen_ids.add(key)
                record = dict(record)
                record["_input_path"] = str(path)
                records.append(record)
    if not records:
        raise ActivationError("answer inputs contain no records")
    return records


def _question(record: dict[str, Any]) -> str:
    request = record.get("request")
    request = request if isinstance(request, dict) else {}
    return str(record.get("query") or request.get("question") or "")


def _retrieval(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("retrieval")
    return value if isinstance(value, dict) else {}


def _trace(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("evaluation_trace")
    return value if isinstance(value, dict) else {}


def _neighbor(record: dict[str, Any]) -> dict[str, Any]:
    value = _retrieval(record).get("neighbor_expansion")
    return value if isinstance(value, dict) else {}


def _prompt(record: dict[str, Any]) -> str:
    generation_input = _trace(record).get("generation_input")
    generation_input = generation_input if isinstance(generation_input, dict) else {}
    return str(generation_input.get("user_prompt") or "")


def _claims(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("claims")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _stage(record: dict[str, Any], name: str) -> list[dict[str, Any]]:
    stages = _trace(record).get("retrieval_stages")
    stages = stages if isinstance(stages, dict) else {}
    value = stages.get(name)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _query_matches(rule_id: str, question: str) -> bool | None:
    q = question
    checks: dict[str, Callable[[str], bool]] = {
        "EXP-001": lambda value: bool(
            re.search(r"학생증|증명서", value)
            and re.search(r"외부\s*기관|위탁", value)
        ),
        "EXP-002": lambda value: bool(
            re.search(r"(?:19|20)\d{2}\s*학년도", value)
            and re.search(r"신입생|신입학", value)
            and re.search(r"총\s*몇|몇\s*명|모집\s*인원", value)
            and re.search(r"전년\s*대비|달라|변경", value)
        ),
        "EXP-003": _is_group_visa,
        "EXP-004": _is_foreign_admission,
        "EXP-005": _is_exchange,
        "EXP-006": _is_summer_certificate,
        "EXP-007": _is_disability_support,
        "EXP-008": _is_third_party,
        "GEN-007": lambda value: bool(
            re.search(r"대신|대체|면제|인정", value)
            and re.search(r"시험|강좌|과목|이수", value)
        ),
        "GEN-008": lambda value: bool(
            re.search(r"체험|경험", value)
            and re.search(r"프로그램|직무|진로|업무|현장", value)
        ),
        "GEN-009": lambda value: bool(
            "등록금" in value and re.search(r"언제|어떻게|납부|내야|내나요|내면", value)
        ),
        "GEN-010": lambda value: bool(
            re.search(r"학생증|증명서", value)
            and re.search(r"외부\s*기관|위탁", value)
        ),
        "GEN-011": lambda value: _query_matches("EXP-002", value) is True,
        "GEN-012": _is_group_visa,
        "GEN-013": _is_copay,
        "GEN-014": _is_foreign_admission,
        "GEN-015": _is_exchange,
        "GEN-016": _is_summer_certificate,
        "GEN-017": _is_disability_support,
        "GEN-018": _is_third_party,
    }
    checker = checks.get(rule_id)
    return checker(q) if checker is not None else None


def _is_group_visa(value: str) -> bool:
    return bool(
        re.search(r"(?<![A-Za-z0-9])D\s*[-‐‑‒–—]?\s*2(?!\d)", value, re.I)
        and re.search(r"비자|체류", value)
        and "연장" in value
        and "단체" in value
    )


def _is_foreign_admission(value: str) -> bool:
    return all(
        re.search(pattern, value)
        for pattern in (r"외국인", r"학부", r"신입학|신입생", r"자격|조건|요건")
    )


def _is_exchange(value: str) -> bool:
    return bool(
        re.search(r"교환학생|해외\s*파견", value)
        and re.search(r"선발\s*규모", value)
        and re.search(r"지원\s*일정|접수\s*일정", value)
    )


def _is_summer_certificate(value: str) -> bool:
    return bool(
        "여름방학" in value
        and "자격증" in value
        and re.search(r"강의|특강|프로그램|대비", value)
    )


def _is_disability_support(value: str) -> bool:
    return bool("장애" in value and "지원" in value and re.search(r"수업|시험", value))


def _is_third_party(value: str) -> bool:
    return bool(
        "신고" in value
        and re.search(r"피해자가?\s*아닌|대신\s*신고|제\s*3\s*자|목격", value)
    )


def _is_copay(value: str) -> bool:
    return bool(
        re.search(r"정보통신\s*보조기기", value)
        and re.search(r"자부담금|개인부담금", value)
        and "지원" in value
    )


EXPANSION_TERMS: dict[str, tuple[str, ...]] = {
    "EXP-001": ("개인정보처리", "위탁현황", "수탁기관"),
    "EXP-002": ("대학입학전형", "기본계획", "모집인원", "변경사항"),
    "EXP-003": ("단체접수", "체류기간", "사전", "예약", "제출서류", "수수료"),
    "EXP-004": ("특별전형", "모집요강", "지원자격", "국적", "언어능력", "학력"),
    "EXP-005": ("교비", "프로그램", "선발요강", "선발규모", "온라인지원"),
    "EXP-006": ("하계방학", "취업역량", "비교과", "대비반"),
    "EXP-007": ("장애학생지원센터", "교수학습", "강의지원", "교재지원"),
    "EXP-008": ("센터이용", "Q&A", "제3자"),
}


PROMPT_MARKERS: dict[str, str] = {
    "GEN-001": "아래 검색 근거만 사용해 질문에 바로 답하세요.",
    "GEN-002": "질문 전체에 대한 답변을 거부하지",
    "GEN-003": "이월과 반환",
    "GEN-004": "하나의 검색 근거 블록",
    "GEN-005": "<필수_답변_항목>",
    "GEN-006": "질문에 나열된 각 대상·행위별로 따로 답하기",
    "GEN-007": "대체·면제 인정에 필요한 이수 기준",
    "GEN-008": "프로그램에서 실제로 하는 활동·체험 방식",
    "GEN-009": "등록금 고지서 출력 가능 시점",
    "GEN-010": "질문에 나온 각 업무별 수탁기관",
    "GEN-011": "총 모집인원과 전년 대비 주요 변경사항을 구분",
    "GEN-012": "수수료 금액과 현금·권종 등 납부방식",
    "GEN-013": "지원 대상·지원 범위·신청기간·선납부 절차·제출서류와 접수 경로",
    "GEN-014": "국적·학력·언어능력 자격을 구분",
    "GEN-015": "선발 규모·온라인 지원기간·합격자 발표",
    "GEN-016": "운영 여부와 자격증 대비 과정 종류",
    "GEN-017": "수업 지원과 시험 지원을 구분",
    "GEN-018": "제3자 신고 가능 여부·피해자 의사·인적사항 조건",
}


def _observed(rule_id: str, record: dict[str, Any]) -> tuple[bool | None, list[str]]:
    retrieval = _retrieval(record)
    neighbor = _neighbor(record)
    prompt = _prompt(record)
    claims = _claims(record)
    evidence: list[str] = []

    if rule_id in EXPANSION_TERMS:
        possible = _query_matches(rule_id, _question(record))
        if not possible:
            return False, ["query trigger false"]
        retrieval_query = str(retrieval.get("retrieval_query") or "")
        terms = EXPANSION_TERMS[rule_id]
        matched = [term for term in terms if term.lower() in retrieval_query.lower()]
        evidence.append(f"retrieval_query expansion terms={','.join(matched)}")
        return bool(matched), evidence

    if rule_id == "RANK-006":
        strategy = str(retrieval.get("strategy") or "")
        if not strategy:
            return None, []
        return "document-diverse" in strategy.lower(), [f"strategy={strategy}"]
    if rule_id == "RANK-007":
        diag = retrieval.get("context_deduplication")
        if not isinstance(diag, dict):
            return None, []
        removed = int(diag.get("removed_count") or 0)
        return removed > 0, [f"removed_count={removed}"]
    if rule_id == "FACET-001":
        if "required_query_facets" not in neighbor:
            return None, []
        facets = neighbor.get("required_query_facets") or []
        return bool(facets), ["required_query_facets=" + ",".join(map(str, facets))]
    if rule_id == "FACET-002":
        if "completed_count" not in neighbor:
            return None, []
        value = int(neighbor.get("completed_count") or 0)
        return value > 0, [f"completed_count={value}"]
    if rule_id == "FACET-003":
        if "scope_conflict_skip_count" not in neighbor:
            return None, []
        value = int(neighbor.get("scope_conflict_skip_count") or 0)
        return value > 0, [f"scope_conflict_skip_count={value}"]
    if rule_id == "FACET-004":
        if "marginal_replacement_count" not in neighbor:
            return None, []
        value = int(neighbor.get("marginal_replacement_count") or 0)
        return value > 0, [f"marginal_replacement_count={value}"]
    intent_field = {
        "FACET-009": "foreign_admission_eligibility",
        "FACET-010": "exchange_selection",
        "FACET-011": "group_visa_application",
        "FACET-012": "assistive_device_copay_support",
        "FACET-013": "third_party_reporting",
    }.get(rule_id)
    if intent_field is not None:
        if intent_field not in neighbor:
            return None, []
        return bool(neighbor.get(intent_field)), [f"neighbor.{intent_field}"]
    if rule_id == "FACET-014":
        if "considered_count" not in neighbor:
            return None, []
        value = int(neighbor.get("considered_count") or 0)
        return value > 0, [f"considered_count={value}"]
    if rule_id in PROMPT_MARKERS:
        if not prompt:
            return None, []
        marker = PROMPT_MARKERS[rule_id]
        return marker in prompt, [f"prompt marker={marker}"]
    if rule_id == "POST-001":
        if "raw_draft" not in _trace(record):
            return None, []
        return True, ["evaluation_trace.raw_draft present"]
    reason_rule = {
        "POST-002": "model_abstention",
        "POST-005": "low_lexical_overlap",
        "POST-006": "semantic_relation_mismatch",
    }.get(rule_id)
    if reason_rule is not None:
        if not claims:
            return None, []
        count = sum(item.get("validation_reason") == reason_rule for item in claims)
        return count > 0, [f"{reason_rule}={count}"]
    if rule_id == "POST-003":
        if not claims:
            return None, []
        count = sum(
            bool(item.get("missing_critical_values"))
            or item.get("validation_reason") == "critical_value_mismatch"
            for item in claims
        )
        numeric_claims = sum(
            bool(re.search(r"\d", str(item.get("text") or ""))) for item in claims
        )
        return bool(count or numeric_claims), [
            f"critical_mismatch_or_missing={count}",
            f"numeric_claims={numeric_claims}",
        ]
    if rule_id == "POST-019":
        if not claims:
            return None, []
        cited = sum(bool(item.get("citations")) for item in claims if item.get("supported"))
        return cited > 0, [f"supported_claims_with_citations={cited}"]
    return None, []


def _possible(
    rule: dict[str, Any], record: dict[str, Any]
) -> tuple[bool | None, str]:
    rule_id = rule["rule_id"]
    question = _question(record)
    query_result = _query_matches(rule_id, question)
    if query_result is not None:
        return query_result, "query trigger"
    if rule_id == "GEN-006":
        return bool(
            question.count("?") >= 2
            or re.search(r"각각|모두|둘\s*다|\b및\b|[·/]", question)
        ), "query multipart trigger"
    if rule_id == "FACET-001":
        return bool(
            re.search(
                r"언제|기간|일정|날짜|마감|금액|한도|얼마|비용|수수료|금리|이율|"
                r"자격|대상|방법|절차|어떻게|어디서|계좌|통장|예산|비중|구성비",
                question,
            )
        ), "generic facet query trigger"
    if rule_id == "FACET-002":
        return bool(_neighbor(record).get("enabled")), "neighbor enabled trace"
    if rule_id == "FACET-003":
        return bool(re.search(r"(?:19|20)\d{2}|[12]\s*학기", question)), "year/semester query"
    if rule_id == "POST-002":
        raw = str(_trace(record).get("raw_draft") or "")
        return "확인할 수 없" in raw or "근거" in raw and "없" in raw, "raw draft text"
    if rule_id in {"POST-003", "POST-004"}:
        raw = str(_trace(record).get("raw_draft") or "")
        if rule_id == "POST-004":
            return bool(re.search(r"\d+\s*(?:~|[-–—]|에서|부터)\s*\d+\s*[가-힣]+", raw)), "raw draft range"
        return bool(re.search(r"\d", raw)), "raw draft numeric value"
    if rule["classification"] == "targeted":
        return str(record.get("case_id") or "") in set(rule["target_case_ids"]), "inventory target membership"
    return True, "general rule evaluated on service path"


def analyze_records(
    records: Sequence[dict[str, Any]],
    rules: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    per_record: list[dict[str, Any]] = []
    summary_counters: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    targeted_observed_cases: set[str] = set()
    targeted_possible_cases: set[str] = set()
    inventory_target_cases = {
        case_id
        for rule in rules
        if rule["classification"] == "targeted"
        for case_id in rule["target_case_ids"]
    }

    for record in records:
        case_id = str(record.get("case_id") or record.get("id") or "")
        condition_id = str(record.get("condition_id") or "unknown")
        activations: list[dict[str, Any]] = []
        for rule in rules:
            observed, evidence = _observed(rule["rule_id"], record)
            possible, possible_basis = _possible(rule, record)
            trace_status = (
                "observed_active"
                if observed is True
                else "observed_inactive"
                if observed is False
                else "trace_absent"
            )
            item = {
                "rule_id": rule["rule_id"],
                "classification": rule["classification"],
                "trace_status": trace_status,
                "trigger_possible": possible,
                "possible_basis": possible_basis,
                "evidence": evidence,
            }
            activations.append(item)
            counter = summary_counters[condition_id][rule["rule_id"]]
            counter[trace_status] += 1
            counter[
                "possible_true"
                if possible is True
                else "possible_false"
                if possible is False
                else "possible_unknown"
            ] += 1
            if rule["classification"] == "targeted":
                if observed is True:
                    targeted_observed_cases.add(case_id)
                if possible is True:
                    targeted_possible_cases.add(case_id)
        per_record.append(
            {
                "case_id": case_id,
                "condition_id": condition_id,
                "generation_run_id": str(
                    record.get("generation_run_id") or "unknown"
                ),
                "input_path": record.get("_input_path"),
                "rule_activations": activations,
            }
        )

    condition_summaries: dict[str, Any] = {}
    for condition_id, by_rule in sorted(summary_counters.items()):
        condition_n = sum(
            item["condition_id"] == condition_id for item in per_record
        )
        condition_summaries[condition_id] = {
            "n": condition_n,
            "rules": {
                rule_id: {
                    **dict(counts),
                    "observed_activation_rate": (
                        counts["observed_active"] / condition_n if condition_n else 0.0
                    ),
                    "possible_activation_rate": (
                        counts["possible_true"] / condition_n if condition_n else 0.0
                    ),
                }
                for rule_id, counts in sorted(by_rule.items())
            },
        }

    missing_observed = sorted(inventory_target_cases - targeted_observed_cases)
    missing_possible = sorted(inventory_target_cases - targeted_possible_cases)
    observed_spillover = sorted(targeted_observed_cases - inventory_target_cases)
    possible_spillover = sorted(targeted_possible_cases - inventory_target_cases)
    return {
        "schema_version": "pnu.rule-activation-analysis.v1",
        "methodology": {
            "observed": "saved trace directly exposes the rule outcome",
            "possible": "query/artifact inference only; not causal proof",
            "trace_absent": "service trace has no stable rule identifier",
            "version_warning": (
                "An answer artifact can predate frozen rules. Prompt absence or "
                "missing neighbor fields is preserved rather than reconstructed."
            ),
        },
        "inventory": {
            "rule_count": len(rules),
            "general_rule_count": sum(
                rule["classification"] == "general" for rule in rules
            ),
            "targeted_rule_count": sum(
                rule["classification"] == "targeted" for rule in rules
            ),
            "target_case_count": len(inventory_target_cases),
            "target_case_ids": sorted(inventory_target_cases),
        },
        "coverage": {
            "record_count": len(per_record),
            "observed_targeted_active_case_count": len(targeted_observed_cases),
            "observed_targeted_active_case_ids": sorted(targeted_observed_cases),
            "observed_targeted_intended_case_count": len(
                targeted_observed_cases & inventory_target_cases
            ),
            "observed_targeted_spillover_case_ids": observed_spillover,
            "possible_targeted_case_count": len(targeted_possible_cases),
            "possible_targeted_case_ids": sorted(targeted_possible_cases),
            "possible_targeted_intended_case_count": len(
                targeted_possible_cases & inventory_target_cases
            ),
            "possible_targeted_spillover_case_ids": possible_spillover,
            "inventory_target_cases_without_observed_activation": missing_observed,
            "inventory_target_cases_without_possible_activation": missing_possible,
            "difference_explanation": (
                "Observed coverage is expected to be lower than inventory target "
                "coverage because candidate demotion and postprocessor bridge IDs "
                "were not recorded, and the DEV45 v1 artifacts predate several "
                "2026-09-04 rules. Possible coverage uses query triggers or explicit "
                "inventory target membership and is not an activation claim."
            ),
        },
        "conditions": condition_summaries,
        "records": per_record,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ActivationError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ActivationError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "case_id",
        "condition_id",
        "generation_run_id",
        "observed_active_rules",
        "observed_inactive_rules",
        "trace_absent_rules",
        "possible_rules",
        "targeted_observed_active_count",
        "targeted_possible_count",
    )
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for record in payload["records"]:
            activations = record["rule_activations"]
            writer.writerow(
                {
                    "case_id": record["case_id"],
                    "condition_id": record["condition_id"],
                    "generation_run_id": record["generation_run_id"],
                    "observed_active_rules": ";".join(
                        item["rule_id"]
                        for item in activations
                        if item["trace_status"] == "observed_active"
                    ),
                    "observed_inactive_rules": ";".join(
                        item["rule_id"]
                        for item in activations
                        if item["trace_status"] == "observed_inactive"
                    ),
                    "trace_absent_rules": ";".join(
                        item["rule_id"]
                        for item in activations
                        if item["trace_status"] == "trace_absent"
                    ),
                    "possible_rules": ";".join(
                        item["rule_id"]
                        for item in activations
                        if item["trigger_possible"] is True
                    ),
                    "targeted_observed_active_count": sum(
                        item["classification"] == "targeted"
                        and item["trace_status"] == "observed_active"
                        for item in activations
                    ),
                    "targeted_possible_count": sum(
                        item["classification"] == "targeted"
                        and item["trigger_possible"] is True
                        for item in activations
                    ),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", action="append", required=True, type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/rule-inventory-20260904.md"),
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.inventory, *args.answers):
        if not path.is_file() or path.is_symlink():
            raise ActivationError(f"input must be a regular non-symlink file: {path}")
    if args.out_json.resolve() == args.out_csv.resolve():
        raise ActivationError("JSON and CSV outputs must differ")
    rules = load_inventory(args.inventory)
    records = load_jsonl(args.answers)
    payload = analyze_records(records, rules)
    payload["inputs"] = {
        "inventory": {
            "path": str(args.inventory),
            "sha256": sha256_file(args.inventory),
        },
        "answers": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in args.answers
        ],
    }
    write_json(args.out_json, payload)
    write_csv(args.out_csv, payload)
    print(
        f"records={len(records)} rules={len(rules)} "
        f"target_cases={payload['inventory']['target_case_count']}"
    )
    print(f"json={args.out_json}")
    print(f"csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
