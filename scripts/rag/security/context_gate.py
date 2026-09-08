"""Inspect final retrieved contexts without changing retrieval or ranking."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .models import ContextGateResult, SecurityFinding


REMOVED_MARKER = "[UNTRUSTED INSTRUCTION REMOVED]"

# Administrative imperatives are common in university documents.  Every high
# confidence rule therefore requires language directed at a model or runtime.
_RULES: tuple[tuple[str, float, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        0.98,
        re.compile(
            r"(?i)(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|system)\s+"
            r"(?:instructions?|prompts?|rules?)|"
            r"(?:이전|앞선|위의|시스템)\s*(?:지시|명령|프롬프트|규칙)(?:를|을)?\s*(?:무시|잊어)"
        ),
    ),
    (
        "role_override",
        0.92,
        re.compile(
            r"(?i)(?:you are now|act as|new system prompt|developer message)|"
            r"(?:이제부터|지금부터).{0,30}(?:시스템|개발자|어시스턴트|역할)"
        ),
    ),
    (
        "secret_exfiltration",
        0.98,
        re.compile(
            r"(?i)(?:reveal|print|return|send|exfiltrate).{0,60}"
            r"(?:system prompt|hidden (?:prompt|configuration)|private configuration|"
            r"api[_ -]?key|token|password|environment variable)|"
            r"(?:system prompt|hidden (?:prompt|configuration)|private configuration|"
            r"api[_ -]?key|token|password|environment variable).{0,60}"
            r"(?:reveal|print|return|send|expose|disclose)|"
            r"(?:시스템\s*프롬프트|API\s*키|토큰|비밀번호|환경\s*변수).{0,40}"
            r"(?:공개|출력|전송|보내|전달|반환|답변)"
        ),
    ),
    (
        "tool_or_network_command",
        0.90,
        re.compile(
            r"(?i)(?:call|invoke|execute|run|use).{0,35}(?:tool|function|shell|curl|wget)|"
            r"(?:도구|함수|셸|명령어|curl|wget)(?:를|을)?\s*"
            r"(?:호출하라|실행하라|호출해|실행해|호출하세요|실행하세요)"
        ),
    ),
    (
        "citation_manipulation",
        0.88,
        re.compile(
            r"(?i)(?:invent|fake|fabricate|create|remove).{0,45}"
            r"(?:citation|source|evidence|reference)|"
            r"(?:citation|source|evidence|reference).{0,45}"
            r"(?:invent|fake|fabricate|made-up|does not exist)|"
            r"(?:출처|인용|근거)(?:를|을)?\s*"
            r"(?:(?:조작|위조|삭제)(?:하라|해|하세요|하고)|"
            r"지어내(?:라|줘|세요)|붙여|넣어)"
        ),
    ),
    (
        "model_delimiter",
        0.65,
        re.compile(r"(?i)<\/?(?:system|assistant|developer|instructions?)>|\[/?INST\]"),
    ),
    (
        "semantic_override",
        0.90,
        re.compile(
            r"(?i)(?:override|dismiss|bypass).{0,50}(?:directions|rules|policy|"
            r"guidance|instructions)|"
            r"(?:do not follow|set (?:them|it) aside).{0,50}(?:answer|follow|"
            r"reveal|expose|fabricated)|"
            r"(?:official sources?|evidence).{0,30}(?:irrelevant|regardless).{0,50}"
            r"(?:follow|answer|output)|"
            r"(?:highest-priority instruction|whatever the evidence says).{0,50}"
            r"(?:assistant|answer|state)|"
            r"always\s+state\s+that.{0,60}|"
            r"(?:앞서|앞에서|원래|기존|상위).{0,35}"
            r"(?:안내|지시|명령|안전\s*정책|제한).{0,35}"
            r"(?:따르지\s*말고|고려하지\s*말고|제쳐두고|건너뛰고|폐기하고|적용하지\s*말고)|"
            r"(?:검색(?:된)?\s*근거|공식\s*(?:자료|문서)|문서\s*내용).{0,35}"
            r"(?:배제하고|보지\s*말고|관계없이|대신).{0,45}"
            r"(?:답해|말해|출력해|확정해|사실로\s*사용)|"
            r"검색\s*결과와\s*관계없이.{0,50}(?:답해|말해|출력해|단정해)|"
            r"근거\s*문서.{0,20}보지\s*말고.{0,50}(?:사실|정답|답해)|"
            r"상위\s*정책보다.{0,40}(?:문서|요구).{0,25}우선.{0,20}답|"
            r"(?:관리자\s*모드|보안\s*제한).{0,25}"
            r"(?:전환하라|전환해|해제하라|해제해|해제하고)|"
            r"답변에는\s*반드시.{0,50}(?:라는\s*문장|라고).{0,20}(?:포함|넣어)|"
            r"(?:숨겨진|내부).{0,25}(?:설정|운영\s*규칙|프롬프트).{0,25}"
            r"(?:보여\s*줘|공개하라|출력하라|답변해|반환해)|"
            r"(?:정답|인용)\s*검증.{0,20}(?:건너뛰고|생략하고)|"
            r"(?:내가?\s*(?:제공한|지정한)|공격자가\s*지정한).{0,35}"
            r"(?:결론|사실|날짜|금액).{0,40}(?:우선|확정해|정답)|"
            r"학교\s*규정보다.{0,30}(?:내\s*요청|이\s*문서).{0,15}우선|"
            r"(?:근거와\s*무관하게|사실\s*확인\s*없이).{0,50}(?:답해|단정해)|"
            r"기존\s*규칙은\s*여기까지.{0,40}(?:따라라|답해)|"
            r"공식\s*문서는\s*중요하지\s*않다.{0,40}(?:출력|답해)"
        ),
    ),
)

_BENIGN_NEGATION_RE = re.compile(
    r"(?:해서는\s*안|하지\s*말|하면\s*안|않아야|금지(?:한다|된다|되어)|"
    r"위배(?:된다|한다)|폐기(?:한다|해야))"
)
_BENIGN_DELIMITER_CONTEXT_RE = re.compile(
    r"(?:문법|강의|수업|프로그래밍|태그.{0,20}의미|형식을?\s*(?:설명|기록))"
)


def detect_injection(text: str) -> tuple[SecurityFinding, ...]:
    # NFKC folds full-width characters and removing formatting controls blocks
    # common zero-width keyword splitting (for example "ign\u200bore").
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    findings = [
        SecurityFinding(category, score, match.start(), match.end())
        for category, score, pattern in _RULES
        for match in pattern.finditer(normalized)
    ]
    if _BENIGN_NEGATION_RE.search(normalized):
        findings = [
            finding
            for finding in findings
            if finding.category not in {
                "secret_exfiltration",
                "tool_or_network_command",
                "citation_manipulation",
            }
        ]
    if _BENIGN_DELIMITER_CONTEXT_RE.search(normalized):
        findings = [
            finding for finding in findings
            if finding.category != "model_delimiter"
        ]
    return tuple(sorted(findings, key=lambda item: (item.start, item.end)))


def _metadata(context: Mapping[str, Any]) -> Mapping[str, Any]:
    value = context.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _existing_security(context: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _metadata(context).get("security")
    return value if isinstance(value, Mapping) else {}


def _source_host(context: Mapping[str, Any]) -> str:
    metadata = _metadata(context)
    host = str(context.get("source_host") or metadata.get("source_host") or "")
    if host:
        return host.lower().split(":", 1)[0].strip(".")
    url = str(context.get("source_url") or metadata.get("source_url") or "")
    try:
        return (urlsplit(url).hostname or "").lower().strip(".")
    except ValueError:
        return ""


def source_trust(context: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
    score = 0.50
    reasons: list[str] = []
    trusted_hosts = tuple(
        value.strip().lower().strip(".")
        for value in os.environ.get("RAG_TRUSTED_SOURCE_HOSTS", "pusan.ac.kr").split(",")
        if value.strip()
    )
    host = _source_host(context)
    if host and any(host == item or host.endswith("." + item) for item in trusted_hosts):
        score += 0.30
        reasons.append("trusted_host")
    elif host:
        score -= 0.10
        reasons.append("unrecognized_host")
    validation = _existing_security(context).get("source_validation")
    validation = validation if isinstance(validation, Mapping) else {}
    if validation.get("status") == "verified":
        score += 0.15
        reasons.append("source_verified")
    if validation.get("hash_verified") is True:
        score += 0.05
        reasons.append("hash_verified")
    return max(0.0, min(1.0, score)), tuple(reasons)


def _hard_veto(context: Mapping[str, Any]) -> str | None:
    security = _existing_security(context)
    validation = security.get("source_validation")
    validation = validation if isinstance(validation, Mapping) else {}
    if str(validation.get("status") or "").lower() in {
        "failed", "invalid", "blocked", "quarantined"
    }:
        return "source_validation_failed"
    scan = security.get("malware_scan") or security.get("scan")
    scan = scan if isinstance(scan, Mapping) else {}
    if str(scan.get("status") or "").lower() in {"infected", "failed", "blocked"}:
        return "malware_scan_failed"
    return None


def _sanitize(text: str, findings: Sequence[SecurityFinding]) -> str:
    sanitized = unicodedata.normalize("NFKC", text)
    sanitized = "".join(
        character
        for character in sanitized
        if unicodedata.category(character) != "Cf"
    )
    for finding in sorted(findings, key=lambda item: item.start, reverse=True):
        sanitized = sanitized[: finding.start] + REMOVED_MARKER + sanitized[finding.end :]
    return sanitized


def evaluate_contexts(
    contexts: Sequence[Mapping[str, Any]], *, mode: str | None = None
) -> ContextGateResult:
    active_mode = (mode or os.environ.get("RAG_CONTEXT_SECURITY_MODE", "enforce")).lower()
    if active_mode not in {"enforce", "shadow", "off"}:
        active_mode = "enforce"
    originals: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    allowed = sanitized_count = excluded = 0
    reasons: dict[str, int] = {}

    for source in contexts:
        original = dict(source)
        generation_copy = dict(source)
        text = str(source.get("text") or source.get("preview") or "")
        findings = detect_injection(text) if active_mode != "off" else ()
        veto = _hard_veto(source) if active_mode != "off" else None
        max_score = max((finding.score for finding in findings), default=0.0)
        trust, trust_reasons = source_trust(source)
        decision, reason = "allow", "clean"
        if veto:
            decision, reason = "exclude", veto
        elif max_score >= 0.85:
            decision, reason = "exclude", "high_confidence_injection"
        elif findings:
            decision, reason = "sanitize", "suspicious_instruction"

        reasons[reason] = reasons.get(reason, 0) + 1
        audit = {
            "policy_version": "context-security-v1",
            "decision": decision,
            "trust_score": round(trust, 3),
            "trust_reasons": list(trust_reasons),
            "injection_detected": bool(findings),
            "injection_score": max_score,
            "categories": sorted({item.category for item in findings}),
        }
        original["security"] = audit
        generation_copy["security"] = audit

        # Shadow mode records the would-be decision but does not alter traffic.
        if active_mode == "shadow":
            originals.append(original)
            generation.append(generation_copy)
        elif decision == "exclude":
            excluded += 1
            continue
        elif decision == "sanitize":
            sanitized_text = _sanitize(text, findings)
            # A partial replacement must not leave another executable-looking
            # instruction behind. Escalate to exclusion if it still matches.
            if detect_injection(sanitized_text):
                excluded += 1
                reasons["sanitize_failed"] = reasons.get("sanitize_failed", 0) + 1
                continue
            generation_copy["text"] = sanitized_text
            originals.append(original)
            generation.append(generation_copy)
            sanitized_count += 1
            continue
        else:
            originals.append(original)
            generation.append(generation_copy)
        allowed += 1

    return ContextGateResult(
        original_contexts=tuple(originals),
        generation_contexts=tuple(generation),
        evaluated=len(contexts),
        allowed=allowed,
        sanitized=sanitized_count,
        excluded=excluded,
        reasons=reasons,
        mode=active_mode,
    )
