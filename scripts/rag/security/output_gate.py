"""Final invariant checks around the existing claim/citation verifier."""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Mapping, Sequence

from .models import OutputGateResult


_ABSTAIN_MESSAGE = "검증된 안전 컨텍스트에서 답변의 인용 근거를 확인하지 못했습니다."


def _normalized_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _citation_valid(
    citation: Mapping[str, Any],
    contexts_by_id: Mapping[str, Mapping[str, Any]],
    contexts_by_number: Mapping[int, Mapping[str, Any]],
) -> bool:
    chunk_id = str(citation.get("chunk_id") or "")
    context = contexts_by_id.get(chunk_id)
    if context is None:
        return False
    source_number = citation.get("source_number")
    if isinstance(source_number, bool) or not isinstance(source_number, int):
        return False
    if contexts_by_number.get(source_number) is not context:
        return False
    excerpt = citation.get("excerpt")
    digest = citation.get("excerpt_sha256")
    if not isinstance(excerpt, str) or not excerpt or not isinstance(digest, str):
        return False
    if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != digest:
        return False
    source_text = context.get("text") or context.get("preview") or ""
    if _normalized_text(excerpt) not in _normalized_text(source_text):
        return False
    return True


def enforce_output(
    response: Mapping[str, Any], safe_contexts: Sequence[Mapping[str, Any]]
) -> OutputGateResult:
    secured = dict(response)
    contexts_by_id = {
        str(context.get("chunk_id")): context
        for context in safe_contexts if context.get("chunk_id")
    }
    contexts_by_number = {
        int(context.get("source_number")): context
        for context in safe_contexts
        if isinstance(context.get("source_number"), int)
        and not isinstance(context.get("source_number"), bool)
    }
    claims_value = response.get("claims")
    top_citations = response.get("citations")
    schema_valid = isinstance(claims_value, list) and isinstance(top_citations, list)
    claims = claims_value if isinstance(claims_value, list) else []
    supported = [
        claim for claim in claims
        if isinstance(claim, Mapping) and claim.get("supported") is True
    ]
    invalid = 0
    for claim in supported:
        source_ids = claim.get("source_ids")
        ids = source_ids if isinstance(source_ids, list) else []
        citations = claim.get("citations")
        citations = citations if isinstance(citations, list) else []
        if not ids or any(str(value) not in contexts_by_id for value in ids):
            invalid += 1
            continue
        if not citations or any(
            not isinstance(citation, Mapping)
            or not _citation_valid(citation, contexts_by_id, contexts_by_number)
            for citation in citations
        ):
            invalid += 1

    if isinstance(top_citations, list):
        invalid += sum(
            not isinstance(citation, Mapping)
            or not _citation_valid(citation, contexts_by_id, contexts_by_number)
            for citation in top_citations
        )

    if not schema_valid or not safe_contexts or invalid or not claims:
        decision = "abstain"
        secured.update(
            answer=_ABSTAIN_MESSAGE,
            cited_answer=_ABSTAIN_MESSAGE,
            claims=[],
            citations=[],
        )
    elif claims and not supported:
        decision = "abstain"
    elif len(supported) < len(claims):
        decision = "partial_answer"
    else:
        decision = "answer"

    return OutputGateResult(
        response=secured,
        decision=decision,
        claims_checked=len(claims),
        claims_supported=len(supported),
        invalid_citations=invalid,
    )
