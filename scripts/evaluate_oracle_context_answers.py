#!/usr/bin/env python3
"""Collect a retrieval-bypassed oracle-context generation diagnostic.

This collector is deliberately separate from ``evaluate_service_answers.py``.
It feeds verified holdout evidence quotes directly to the same generator and
postprocessor used by ``/chat`` so a retrieved-context failure can be separated
from a generator/postprocessor failure.  Oracle rows are diagnostic only and
must never be included in service performance aggregates.

The CLI is fail-closed:

* a path whose name contains ``draft`` is rejected;
* all holdout schema, DEV-separation, corpus-provenance, and two-person sign-off
  gates must pass before the output file is created;
* exactly one pre-specified ``multi_evidence`` Core case per category is used;
* provider, model, prompt, request config, case, sign-off, selection, and gold
  evidence provenance hashes are stored in every immutable answer row;
* an existing output is never overwritten or appended.  A failed batch must be
  rerun with a new generation_run_id and path.

No network call is made by importing this module.  Unit tests exercise the
collector through the generator's in-process extractive adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from holdout_gold import (  # noqa: E402
    CORE_CATEGORIES,
    CORE_DIFFICULTY_TYPES,
    CORE_SPLIT,
    load_jsonl,
    sha256_file,
)
from rag.generators import (  # noqa: E402
    GenerationResult,
    build_prompt,
    generate,
)
from rag import role_router  # noqa: E402
from search_api import (  # noqa: E402
    build_rag_response,
    evaluation_context_trace,
    generation_input_trace,
    number_sources,
    public_generation_metadata,
    public_provider_name,
    public_results,
    resolve_generation_provider,
    strip_untrusted_citation_markers,
)
from service_eval_artifacts import (  # noqa: E402
    ANSWER_SCHEMA_VERSION,
    build_answer_identity,
    sha256_json,
    sha256_text,
    validate_answer_record,
)
from validate_service_holdout import (  # noqa: E402
    DEFAULT_CORPUS_INDEX,
    validate_paths,
)


DEFAULT_CASES = (
    REPO_ROOT / "config" / "pnu-service-answer-holdout-v2.jsonl"
)
DEFAULT_DEV_MANIFEST = (
    REPO_ROOT / "config" / "pnu-service-dev-source-manifest.json"
)
DEFAULT_SIGNOFF = REPO_ROOT / "evidence" / "holdout-v2-signoff.json"
ORACLE_DIAGNOSTIC_VERSION = "pnu.oracle-context-generator-diagnostic.v1"
ORACLE_CONTEXT_BUILDER_VERSION = "gold-evidence-options-v1"
DEFAULT_SELECTION_DIFFICULTY = "multi_evidence"
DEFAULT_MAX_CONTEXT_CHARS = 24_000
DEFAULT_MAX_OUTPUT_TOKENS = 900
DEFAULT_DEADLINE_SECONDS = 45.0
GenerateFn = Callable[..., GenerationResult]


def _is_draft_path(path: Path) -> bool:
    return "draft" in path.name.casefold()


def require_frozen_holdout(
    cases_path: Path,
    dev_paths: Sequence[Path],
    *,
    corpus_index: Path,
    signoff_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return a passing four-gate summary or reject before generation.

    A valid sign-off is bound to the byte hash of the cases file, but the
    explicit filename guard also makes it hard to accidentally execute the
    authoring draft mentioned in the protocol.
    """

    if _is_draft_path(cases_path):
        raise ValueError(
            "oracle generation is forbidden for a draft holdout path; "
            "freeze and sign off the final holdout first"
        )
    if not dev_paths:
        raise ValueError("at least one DEV source manifest is required")
    summary = validate_paths(
        cases_path,
        list(dev_paths),
        corpus_index=corpus_index,
        signoff_path=signoff_path,
        repo_root=repo_root,
    )
    if summary.get("ok") is not True:
        errors = summary.get("errors") or ["unknown holdout gate failure"]
        raise ValueError(
            "frozen holdout preflight failed: " + "; ".join(map(str, errors))
        )
    gates = summary.get("gates")
    required_gates = {"schema", "dev", "corpus", "signoff"}
    if not isinstance(gates, Mapping) or set(gates) != required_gates:
        raise ValueError("holdout preflight omitted one or more required gates")
    if any(
        not isinstance(gates[name], Mapping)
        or gates[name].get("ok") is not True
        for name in required_gates
    ):
        raise ValueError("holdout preflight contains a failed required gate")
    return summary


def select_oracle_cases(
    cases: Sequence[dict[str, Any]],
    *,
    difficulty_type: str = DEFAULT_SELECTION_DIFFICULTY,
) -> list[dict[str, Any]]:
    """Select exactly one Core case per category by a fixed difficulty rule."""

    if difficulty_type not in CORE_DIFFICULTY_TYPES:
        raise ValueError(f"unsupported Core difficulty_type {difficulty_type!r}")
    selected_by_category: dict[str, dict[str, Any]] = {}
    for case in cases:
        if (
            case.get("split") != CORE_SPLIT
            or case.get("difficulty_type") != difficulty_type
        ):
            continue
        category = str(case.get("category") or "").strip()
        if category not in CORE_CATEGORIES:
            raise ValueError(f"selected case has unknown Core category {category!r}")
        if category in selected_by_category:
            raise ValueError(
                f"oracle selection has multiple {difficulty_type!r} cases "
                f"for category {category!r}"
            )
        selected_by_category[category] = case

    missing = [
        category for category in CORE_CATEGORIES if category not in selected_by_category
    ]
    if missing:
        raise ValueError(
            f"oracle selection is missing {difficulty_type!r} Core categories: "
            f"{missing}"
        )
    return [selected_by_category[category] for category in CORE_CATEGORIES]


def _evidence_provenance(
    *,
    claim: Mapping[str, Any],
    claim_index: int,
    option: Mapping[str, Any],
    option_index: int,
) -> dict[str, Any]:
    quote = str(option.get("quote") or "").strip()
    return {
        "claim_id": str(claim.get("claim_id") or "").strip(),
        "claim_index": claim_index,
        "evidence_option_index": option_index,
        "evidence_option_sha256": sha256_json(option),
        "quote_sha256": sha256_text(quote),
        "document_id": option.get("document_id"),
        "source_document_family_id": option.get("source_document_family_id"),
        "source_sha256": option.get("source_sha256"),
        "source_path": option.get("source_path"),
        "source_url": option.get("source_url"),
        "source_title": option.get("source_title"),
        "normalized_title_without_year": option.get(
            "normalized_title_without_year"
        ),
        "evidence_type": claim.get("evidence_type"),
        "table_evidence": option.get("table_evidence"),
    }


def build_oracle_contexts(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build de-duplicated generator contexts from gold evidence quotes only.

    Claim descriptions and reference answers are intentionally excluded from
    ``text``.  Alternative evidence options are all retained.  Identical source
    quote/options shared by claims become one context with multiple provenance
    bindings instead of overweighting repeated gold text.
    """

    case_id = str(case.get("id") or "").strip()
    if not case_id:
        raise ValueError("oracle case is missing id")
    required_claims = case.get("required_claims")
    if not isinstance(required_claims, list) or not required_claims:
        raise ValueError(f"oracle case {case_id!r} has no required_claims")

    contexts: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for claim_index, claim in enumerate(required_claims):
        if not isinstance(claim, Mapping):
            raise ValueError(
                f"oracle case {case_id!r} required_claims[{claim_index}] "
                "must be an object"
            )
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            raise ValueError(
                f"oracle case {case_id!r} required_claims[{claim_index}] "
                "is missing claim_id"
            )
        options = claim.get("evidence_options")
        if not isinstance(options, list) or not options:
            raise ValueError(
                f"oracle case {case_id!r} claim {claim_id!r} has no "
                "evidence_options"
            )
        for option_index, option in enumerate(options):
            if not isinstance(option, Mapping):
                raise ValueError(
                    f"oracle case {case_id!r} claim {claim_id!r} evidence "
                    f"option {option_index} must be an object"
                )
            quote = str(option.get("quote") or "").strip()
            document_id = str(option.get("document_id") or "").strip()
            if not quote or not document_id:
                raise ValueError(
                    f"oracle case {case_id!r} claim {claim_id!r} evidence "
                    f"option {option_index} needs document_id and quote"
                )
            identity_payload = {
                "document_id": document_id,
                "source_sha256": option.get("source_sha256"),
                "source_path": option.get("source_path"),
                "source_url": option.get("source_url"),
                "quote": quote,
                "table_evidence": option.get("table_evidence"),
            }
            context_identity = sha256_json(identity_payload)
            provenance = _evidence_provenance(
                claim=claim,
                claim_index=claim_index,
                option=option,
                option_index=option_index,
            )
            if context_identity in positions:
                contexts[positions[context_identity]]["gold_evidence_provenance"].append(
                    provenance
                )
                continue

            source_path = str(option.get("source_path") or "").strip()
            context = {
                "chunk_id": f"oracle:{case_id}:{context_identity[:20]}",
                "document_id": document_id,
                "chunk_index": len(contexts),
                "institution": "부산대학교",
                "file_name": Path(source_path).name if source_path else None,
                "source_path": source_path or None,
                "source_title": option.get("source_title"),
                "source_url": option.get("source_url"),
                "text": quote,
                "retrieval": {
                    "mode": "oracle_gold_evidence",
                    "retrieval_bypassed": True,
                },
                "gold_evidence_provenance": [provenance],
            }
            positions[context_identity] = len(contexts)
            contexts.append(context)
    return contexts


@contextmanager
def pinned_generation_environment(
    *,
    max_context_chars: int,
    max_output_tokens: int,
    deadline_seconds: float,
    answer_style: str,
) -> Iterator[None]:
    """Pin settings read from environment and restore the caller afterwards."""

    if max_context_chars < 100:
        raise ValueError("max_context_chars must be at least 100")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    if answer_style not in {"standard", "structured"}:
        raise ValueError("answer_style must be 'standard' or 'structured'")

    updates = {
        "RAG_GENERATION_MAX_CONTEXT_CHARS": str(max_context_chars),
        "RAG_GENERATION_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "RAG_LOCAL_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "RAG_FRONTIER_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "RAG_GEMINI_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "GEMINI_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "RAG_GENERATION_DEADLINE_SECONDS": str(deadline_seconds),
    }
    keys = set(updates) | {"RAG_ANSWER_STYLE"}
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(updates)
        if answer_style == "structured":
            os.environ["RAG_ANSWER_STYLE"] = "structured"
        else:
            os.environ.pop("RAG_ANSWER_STYLE", None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_generation_contract(
    result: GenerationResult,
    *,
    adapter_provider: str,
    model: str | None,
    expected_prompt_sha256: str,
    expected_system_instruction_sha256: str,
    max_output_tokens: int,
) -> None:
    if result.requested != adapter_provider or result.used != adapter_provider:
        raise RuntimeError(
            "oracle generator provider mismatch/fallback: "
            f"requested={adapter_provider!r}, result.requested={result.requested!r}, "
            f"result.used={result.used!r}"
        )
    if adapter_provider != "extractive" and result.model != model:
        raise RuntimeError(
            f"oracle generator model mismatch: expected {model!r}, "
            f"got {result.model!r}"
        )
    if result.prompt_sha256 != expected_prompt_sha256:
        raise RuntimeError("oracle generator prompt hash mismatch")
    if result.system_instruction_sha256 != expected_system_instruction_sha256:
        raise RuntimeError("oracle generator system instruction hash mismatch")
    if result.request_config_sha256 != sha256_json(result.request_config):
        raise RuntimeError("oracle generator request config hash mismatch")
    config_provider = str(result.request_config.get("provider") or "")
    if config_provider != adapter_provider:
        raise RuntimeError(
            f"oracle request config provider mismatch: {config_provider!r}"
        )
    embedded_system_sha = result.request_config.get(
        "system_instruction_sha256"
    )
    if embedded_system_sha not in (None, result.system_instruction_sha256):
        raise RuntimeError(
            "oracle request config system instruction hash mismatch"
        )
    if adapter_provider == "extractive":
        if result.model is not None:
            raise RuntimeError("extractive oracle result unexpectedly reports a model")
        return
    configured_tokens = result.request_config.get("max_output_tokens")
    generation_config = result.request_config.get("generation_config")
    if isinstance(generation_config, Mapping):
        configured_tokens = generation_config.get("maxOutputTokens")
    if configured_tokens != max_output_tokens:
        raise RuntimeError(
            "oracle generator max-output-token mismatch: "
            f"expected {max_output_tokens}, got {configured_tokens!r}"
        )


def _missing_context_blocks(
    prompt: str, contexts: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return context IDs whose own Source block lost its full gold quote."""

    missing: list[str] = []
    for index, context in enumerate(contexts, 1):
        chunk_id = str(context.get("chunk_id") or "")
        start_marker = f"Source {index}\n"
        start = prompt.find(start_marker)
        if start < 0:
            missing.append(chunk_id)
            continue
        next_marker = f"\n\nSource {index + 1}\n"
        end = prompt.find(next_marker, start + len(start_marker))
        block = prompt[start:] if end < 0 else prompt[start:end]
        if f"ID: {chunk_id}\n" not in block or str(context["text"]) not in block:
            missing.append(chunk_id)
    return missing


def collect_oracle_answer(
    case: dict[str, Any],
    *,
    experiment_id: str,
    condition_id: str,
    generation_run_id: str,
    provider: str,
    model: str | None,
    collector_config: Mapping[str, Any],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    answer_style: str = "standard",
    generate_fn: GenerateFn = generate,
    extractive_fallback: Any = None,
    collected_at: str | None = None,
) -> dict[str, Any]:
    """Collect one hash-bound, judge-compatible oracle answer record."""

    public_requested_provider = public_provider_name(provider)
    adapter_provider = resolve_generation_provider(public_requested_provider)
    if adapter_provider != "extractive" and not str(model or "").strip():
        raise ValueError("a model must be pinned for a network oracle provider")
    role_profile = role_router.resolve(case.get("role"))
    role_perspective = (
        role_profile.perspective if role_profile.id != "general" else None
    )
    raw_contexts = build_oracle_contexts(case)
    numbered_contexts = number_sources(raw_contexts)

    with pinned_generation_environment(
        max_context_chars=max_context_chars,
        max_output_tokens=max_output_tokens,
        deadline_seconds=deadline_seconds,
        answer_style=answer_style,
    ):
        input_trace = generation_input_trace(
            str(case["query"]),
            numbered_contexts,
            role_perspective=role_perspective,
        )
        prompt = build_prompt(
            str(case["query"]),
            numbered_contexts,
            role_perspective=role_perspective,
        )
        if input_trace["user_prompt"] != prompt:
            raise RuntimeError("oracle generation trace differs from active prompt")
        missing_quotes = _missing_context_blocks(prompt, numbered_contexts)
        if missing_quotes:
            raise RuntimeError(
                "oracle gold evidence was truncated from the prompt: "
                f"{missing_quotes}"
            )

        started = time.perf_counter()
        generated = generate_fn(
            str(case["query"]),
            numbered_contexts,
            requested=adapter_provider,
            extractive_fallback=extractive_fallback,
            requested_model=model,
            role_perspective=role_perspective,
        )
        generation_elapsed_ms = (time.perf_counter() - started) * 1000

    _assert_generation_contract(
        generated,
        adapter_provider=adapter_provider,
        model=model,
        expected_prompt_sha256=input_trace["prompt_sha256"],
        expected_system_instruction_sha256=input_trace[
            "system_instruction_sha256"
        ],
        max_output_tokens=max_output_tokens,
    )
    raw_draft = generated.text
    sanitized_draft = strip_untrusted_citation_markers(raw_draft)
    generator_label = (
        f"{generated.used}:{generated.model}"
        if generated.model
        else generated.used
    )
    rag = build_rag_response(
        str(case["query"]),
        numbered_contexts,
        sanitized_draft,
        generator_label,
    )
    generation_metadata = public_generation_metadata(
        generated.metadata(), requested_provider=public_requested_provider
    )
    context_trace = evaluation_context_trace(
        numbered_contexts, stage="final_contexts"
    )
    provenance = [
        {
            "source_number": context["source_number"],
            "chunk_id": context["chunk_id"],
            "document_id": context["document_id"],
            "text_sha256": sha256_text(context["text"]),
            "bindings": context["gold_evidence_provenance"],
        }
        for context in numbered_contexts
    ]
    oracle_contexts_sha256 = sha256_json(numbered_contexts)
    provenance_sha256 = sha256_json(provenance)
    generation_config = generated.request_config.get("generation_config")
    if not isinstance(generation_config, Mapping):
        generation_config = generated.request_config
    temperature = generation_config.get("temperature")
    collector_config_value = dict(collector_config)
    record: dict[str, Any] = {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "record_type": "answer",
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "generation_run_id": generation_run_id,
        "case_id": case["id"],
        "id": case["id"],
        "case_sha256": sha256_json(case),
        "collector_config": collector_config_value,
        "collector_config_sha256": sha256_json(collector_config_value),
        "collected_at": collected_at or datetime.now(timezone.utc).isoformat(),
        "category": case.get("category"),
        "difficulty_type": case.get("difficulty_type"),
        "family_id": case.get("family_id"),
        "role": case.get("role"),
        "query": case["query"],
        "latency_ms": round(generation_elapsed_ms, 3),
        "answer": rag["answer"],
        "cited_answer": rag["cited_answer"],
        "claims": rag["claims"],
        "citations": rag.get("citations", []),
        "postprocessing": rag["postprocessing"],
        "generator": rag["generator"],
        "generation": generation_metadata,
        "role_mapping": role_router.public_role(role_profile, case.get("role")),
        "request": {
            "question": case["query"],
            "role": case.get("role"),
            "provider": public_requested_provider,
            "provider_adapter": adapter_provider,
            "model": model,
            "context_source": "gold_evidence_options",
            "context_count": len(numbered_contexts),
            "answer_style": answer_style,
            "max_context_chars": max_context_chars,
            "max_output_tokens": max_output_tokens,
        },
        "response_config": {
            "retrieval_bypassed": True,
            "generation_requested": generation_metadata.get("requested"),
            "generation_used": generation_metadata.get("used"),
            "generation_model": generation_metadata.get("model"),
        },
        "retrieval": {
            "mode": "oracle_gold_evidence",
            "retrieval_bypassed": True,
            "service_performance_eligible": False,
            "result_count": len(numbered_contexts),
        },
        "sources": public_results(numbered_contexts),
        "evaluation_trace": {
            "schema_version": 1,
            "retrieval_stages": {
                "raw_bm25": [],
                "post_retrieval_pool": [],
                "post_neighbor_expansion": [],
                "final_contexts": context_trace,
            },
            "raw_draft": raw_draft,
            "sanitized_draft": sanitized_draft,
            "generation_input": input_trace,
            "timing_ms": {"generation": round(generation_elapsed_ms, 3)},
        },
        "oracle_diagnostic": {
            "schema_version": ORACLE_DIAGNOSTIC_VERSION,
            "context_builder_version": ORACLE_CONTEXT_BUILDER_VERSION,
            "diagnostic_only": True,
            "retrieval_bypassed": True,
            "service_performance_eligible": False,
            "aggregation_instruction": "exclude_from_service_performance",
            "provider_input": provider,
            "provider_public_requested": public_requested_provider,
            "provider_adapter_requested": adapter_provider,
            "model_requested": model,
            "model_used": generated.model,
            "prompt_sha256": generated.prompt_sha256,
            "system_instruction_sha256": generated.system_instruction_sha256,
            "request_config": dict(generated.request_config),
            "request_config_sha256": generated.request_config_sha256,
            "sampling_parameters": {
                "temperature": {
                    "status": "sent" if temperature is not None else "not_sent",
                    "value": temperature,
                },
                "seed": {"status": "unsupported_not_sent", "value": None},
            },
            "oracle_contexts_sha256": oracle_contexts_sha256,
            "gold_evidence_provenance": provenance,
            "gold_evidence_provenance_sha256": provenance_sha256,
        },
    }
    record = build_answer_identity(record)
    validate_answer_record(record)
    return record


def build_collector_config(
    *,
    cases_path: Path,
    signoff_path: Path,
    dev_paths: Sequence[Path],
    corpus_index: Path,
    holdout_summary: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
    provider: str,
    model: str,
    difficulty_type: str,
    answer_style: str,
    max_context_chars: int,
    max_output_tokens: int,
    deadline_seconds: float,
) -> dict[str, Any]:
    selected_case_ids = [str(case["id"]) for case in selected_cases]
    return {
        "collector": "evaluate_oracle_context_answers.py",
        "oracle_diagnostic_version": ORACLE_DIAGNOSTIC_VERSION,
        "context_builder_version": ORACLE_CONTEXT_BUILDER_VERSION,
        "cases_path": str(cases_path),
        "cases_file_sha256": sha256_file(cases_path),
        "signoff_path": str(signoff_path),
        "signoff_file_sha256": sha256_file(signoff_path),
        "dev_manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in dev_paths
        ],
        "corpus_index_path": str(corpus_index),
        "corpus_index_sha256": sha256_file(corpus_index),
        "holdout_preflight_sha256": sha256_json(holdout_summary),
        "selection_rule": {
            "split": CORE_SPLIT,
            "difficulty_type": difficulty_type,
            "one_per_category": list(CORE_CATEGORIES),
        },
        "selected_case_ids": selected_case_ids,
        "selected_case_ids_sha256": sha256_json(selected_case_ids),
        "provider_input": provider,
        "provider_public_requested": public_provider_name(provider),
        "provider_adapter_requested": resolve_generation_provider(
            public_provider_name(provider)
        ),
        "model_requested": model,
        "answer_style": answer_style,
        "max_context_chars": max_context_chars,
        "max_output_tokens": max_output_tokens,
        "deadline_seconds": deadline_seconds,
        "retrieval_bypassed": True,
        "service_performance_eligible": False,
    }


def append_new_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--dev-manifest",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--corpus-index", type=Path, default=DEFAULT_CORPUS_INDEX)
    parser.add_argument("--signoff", type=Path, default=DEFAULT_SIGNOFF)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--condition-id", default="c1-oracle-context")
    parser.add_argument("--generation-run-id", required=True)
    parser.add_argument(
        "--provider", required=True, choices=("local", "frontier", "gemini")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--selection-difficulty-type",
        choices=CORE_DIFFICULTY_TYPES,
        default=DEFAULT_SELECTION_DIFFICULTY,
    )
    parser.add_argument(
        "--answer-style", choices=("standard", "structured"), default="standard"
    )
    parser.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument(
        "--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS
    )
    parser.add_argument("--sleep", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(
            "--out already exists; oracle artifacts are immutable, so use a "
            "new generation_run_id and path"
        )
    if args.sleep < 0:
        parser.error("--sleep must be non-negative")
    if args.max_context_chars < 100:
        parser.error("--max-context-chars must be at least 100")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.deadline_seconds <= 0:
        parser.error("--deadline-seconds must be positive")
    dev_paths = args.dev_manifest or [DEFAULT_DEV_MANIFEST]

    try:
        holdout_summary = require_frozen_holdout(
            args.cases,
            dev_paths,
            corpus_index=args.corpus_index,
            signoff_path=args.signoff,
        )
        cases = load_jsonl(args.cases)
        selected = select_oracle_cases(
            cases, difficulty_type=args.selection_difficulty_type
        )
        collector_config = build_collector_config(
            cases_path=args.cases,
            signoff_path=args.signoff,
            dev_paths=dev_paths,
            corpus_index=args.corpus_index,
            holdout_summary=holdout_summary,
            selected_cases=selected,
            provider=args.provider,
            model=args.model,
            difficulty_type=args.selection_difficulty_type,
            answer_style=args.answer_style,
            max_context_chars=args.max_context_chars,
            max_output_tokens=args.max_output_tokens,
            deadline_seconds=args.deadline_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    for position, case in enumerate(selected, 1):
        try:
            record = collect_oracle_answer(
                case,
                experiment_id=args.experiment_id,
                condition_id=args.condition_id,
                generation_run_id=args.generation_run_id,
                provider=args.provider,
                model=args.model,
                collector_config=collector_config,
                max_context_chars=args.max_context_chars,
                max_output_tokens=args.max_output_tokens,
                deadline_seconds=args.deadline_seconds,
                answer_style=args.answer_style,
            )
        except Exception as exc:
            print(
                f"oracle collection stopped at {case['id']}: {exc}; "
                "keep the partial file and rerun as a new immutable batch",
                file=sys.stderr,
            )
            return 1
        append_new_jsonl(args.out, record)
        print(
            f"[{position}/{len(selected)}] {case['id']} "
            f"answer_id={record['answer_id']} diagnostic_only=true",
            flush=True,
        )
        if position < len(selected):
            time.sleep(args.sleep)
    print(
        f"oracle answers={len(selected)} service_performance_eligible=false "
        f"out={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
