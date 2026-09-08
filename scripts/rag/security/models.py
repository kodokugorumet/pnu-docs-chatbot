"""Stable contracts shared by the two RAG security gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SecurityFinding:
    category: str
    score: float
    start: int
    end: int


@dataclass(frozen=True)
class ContextGateResult:
    original_contexts: tuple[dict[str, Any], ...]
    generation_contexts: tuple[dict[str, Any], ...]
    evaluated: int
    allowed: int
    sanitized: int
    excluded: int
    reasons: Mapping[str, int]
    mode: str

    def summary(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.original_contexts else "blocked",
            "policy_version": "context-security-v1",
            "mode": self.mode,
            "evaluated": self.evaluated,
            "allowed": self.allowed,
            "sanitized": self.sanitized,
            "excluded": self.excluded,
            "reasons": dict(self.reasons),
        }


@dataclass(frozen=True)
class OutputGateResult:
    response: dict[str, Any]
    decision: str
    claims_checked: int
    claims_supported: int
    invalid_citations: int

    def summary(self) -> dict[str, Any]:
        return {
            "status": "blocked" if self.decision == "abstain" else "passed",
            "policy_version": "output-security-v1",
            "decision": self.decision,
            "claims_checked": self.claims_checked,
            "claims_supported": self.claims_supported,
            "invalid_citations": self.invalid_citations,
        }
