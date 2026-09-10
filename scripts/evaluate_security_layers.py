#!/usr/bin/env python3
"""Deterministic security-layer benchmark for offline graduation-project runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag.security import detect_injection, enforce_output, evaluate_contexts


DEFAULT_CASES = Path("config/security-layer-eval.json")


def _context(chunk_id: str, text: str, source_number: int = 1) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": "security-eval-document",
        "source_number": source_number,
        "text": text,
        "preview": text,
        "locations": [],
    }


def _citation(chunk_id: str, excerpt: str, source_number: int = 1) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_number": source_number,
        "excerpt": excerpt,
        "excerpt_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    }


def _response(claim: str, citation: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": claim,
        "cited_answer": claim + " [1]",
        "claims": [{
            "text": claim,
            "supported": True,
            "source_ids": [citation["chunk_id"]],
            "citations": [citation],
        }],
        "citations": [citation],
    }


def evaluate_context_gate(spec: dict[str, Any]) -> dict[str, Any]:
    clean = list(spec["clean_contexts"])
    attacks = list(spec["attack_contexts"])
    false_positives = 0
    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    exact_actions = 0
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, str]] = []

    for index, text in enumerate(clean):
        result = evaluate_contexts([_context(f"clean-{index}", text)], mode="enforce")
        actual = "exclude" if result.excluded else "sanitize" if result.sanitized else "allow"
        if actual == "allow":
            true_negatives += 1
        else:
            false_positives += 1
            failures.append({"id": f"clean-{index}", "expected": "allow", "actual": actual})

    for index, case in enumerate(attacks):
        result = evaluate_contexts(
            [_context(f"attack-{index}", str(case["text"]))], mode="enforce"
        )
        actual = "exclude" if result.excluded else "sanitize" if result.sanitized else "allow"
        expected = str(case["expected"])
        detected = actual in {"sanitize", "exclude"}
        if detected:
            true_positives += 1
        else:
            false_negatives += 1
        exact_actions += actual == expected
        categories[str(case["category"])]["total"] += 1
        categories[str(case["category"])]["detected"] += detected
        categories[str(case["category"])]["exact"] += actual == expected
        if actual != expected:
            failures.append({"id": f"attack-{index}", "expected": expected, "actual": actual})

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    return {
        "clean_count": len(clean),
        "attack_count": len(attacks),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
        "false_positive_rate": round(false_positives / max(1, len(clean)), 4),
        "exact_action_accuracy": round(exact_actions / max(1, len(attacks)), 4),
        "confusion": {"tp": true_positives, "tn": true_negatives, "fp": false_positives, "fn": false_negatives},
        "categories": {
            category: {
                "count": values["total"],
                "detection_rate": round(values["detected"] / values["total"], 4),
                "exact_action_rate": round(values["exact"] / values["total"], 4),
            }
            for category, values in sorted(categories.items())
        },
        "failures": failures,
    }


def evaluate_output_gate() -> dict[str, Any]:
    evidences = [
        "신청 기간은 2026년 8월 20일까지이다.",
        "등록금 납부 금액은 1,200,000원이다.",
        "접수 장소는 학생지원시스템이다.",
        "교육 시간은 오전 10시부터 오후 5시까지이다.",
        "제출 서류는 신청서와 재학증명서이다.",
        "지원 대상은 부산대학교 재학생이다.",
    ]
    valid_accepted = 0
    invalid_total = 0
    invalid_escaped = 0
    by_attack: Counter[str] = Counter()
    escaped_by_attack: Counter[str] = Counter()

    for index, evidence in enumerate(evidences, start=1):
        chunk_id = f"output-{index}"
        safe = [_context(chunk_id, evidence)]
        valid = _citation(chunk_id, evidence)
        if enforce_output(_response(evidence, valid), safe).decision == "answer":
            valid_accepted += 1
        attacks: dict[str, dict[str, Any]] = {}
        unknown = _citation("unknown", evidence)
        attacks["unknown_chunk"] = _response(evidence, unknown)
        wrong_number = _citation(chunk_id, evidence, source_number=2)
        attacks["wrong_source_number"] = _response(evidence, wrong_number)
        bad_hash = {**valid, "excerpt_sha256": "0" * 64}
        attacks["bad_hash"] = _response(evidence, bad_hash)
        forged = _citation(chunk_id, "원문에 없는 조작된 근거")
        attacks["forged_excerpt"] = _response("조작된 주장", forged)
        missing = _response(evidence, valid)
        missing["claims"][0]["citations"] = []
        attacks["missing_claim_citation"] = missing
        attacks["malformed_claims"] = {"answer": "조작", "claims": "invalid", "citations": []}
        attacks["malformed_citations"] = {"answer": "조작", "claims": [], "citations": "invalid"}
        attacks["empty_claims"] = {"answer": "조작", "claims": [], "citations": []}

        for attack_type, response in attacks.items():
            invalid_total += 1
            by_attack[attack_type] += 1
            decision = enforce_output(response, safe).decision
            if decision != "abstain":
                invalid_escaped += 1
                escaped_by_attack[attack_type] += 1

    return {
        "valid_count": len(evidences),
        "valid_output_acceptance_rate": round(valid_accepted / len(evidences), 4),
        "invalid_count": invalid_total,
        "invalid_output_escape_rate": round(invalid_escaped / invalid_total, 4),
        "attacks": {
            name: {"count": count, "escaped": escaped_by_attack[name]}
            for name, count in sorted(by_attack.items())
        },
    }


def evaluate_latency(spec: dict[str, Any], iterations: int) -> dict[str, Any]:
    contexts = [
        _context(f"latency-{index}", text)
        for index, text in enumerate(
            list(spec["clean_contexts"])[:4] + [case["text"] for case in spec["attack_contexts"][:4]]
        )
    ]
    safe = [_context("latency-output", "신청 기간은 8월 20일까지이다.")]
    citation = _citation("latency-output", safe[0]["text"])
    response = _response(safe[0]["text"], citation)
    context_times: list[float] = []
    output_times: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        evaluate_contexts(contexts, mode="enforce")
        context_times.append((time.perf_counter_ns() - started) / 1_000_000)
        started = time.perf_counter_ns()
        enforce_output(response, safe)
        output_times.append((time.perf_counter_ns() - started) / 1_000_000)

    def summarize(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return {
            "mean_ms": round(statistics.fmean(values), 4),
            "median_ms": round(statistics.median(values), 4),
            "p95_ms": round(ordered[p95_index], 4),
            "max_ms": round(max(values), 4),
        }

    return {
        "iterations": iterations,
        "contexts_per_iteration": len(contexts),
        "context_gate": summarize(context_times),
        "output_gate": summarize(output_times),
    }


def evaluate_corpus_alerts(root: Path, limit: int) -> dict[str, Any]:
    """Measure unlabeled alert volume on real converted text for manual review."""

    evaluated = allowed = sanitized = excluded = 0
    alerts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.txt")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        paragraphs = [value.strip() for value in raw.split("\n\n") if value.strip()]
        for paragraph_index, paragraph in enumerate(paragraphs):
            for segment_index, start in enumerate(range(0, len(paragraph), 4000)):
                if evaluated >= limit:
                    break
                sample = paragraph[start : start + 4000]
                result = evaluate_contexts(
                    [_context(f"{path.name}:{paragraph_index}:{segment_index}", sample)],
                    mode="enforce",
                )
                evaluated += 1
                if result.excluded:
                    action = "exclude"
                    excluded += 1
                elif result.sanitized:
                    action = "sanitize"
                    sanitized += 1
                else:
                    allowed += 1
                    continue
                if len(alerts) < 30:
                    alerts.append({
                        "path": str(path.relative_to(root)),
                        "paragraph": paragraph_index,
                        "segment": segment_index,
                        "action": action,
                        "categories": sorted({
                            finding.category for finding in detect_injection(sample)
                        }),
                        "text_preview": " ".join(sample.split())[:240],
                    })
            if evaluated >= limit:
                break
        if evaluated >= limit:
            break
    return {
        "root": str(root),
        "evaluated": evaluated,
        "allowed": allowed,
        "sanitized": sanitized,
        "excluded": excluded,
        "alert_candidate_rate": round((sanitized + excluded) / max(1, evaluated), 6),
        "note": "unlabeled corpus alerts require manual review and are not an FPR",
        "alert_samples": alerts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--corpus-limit", type=int, default=5000)
    args = parser.parse_args()
    spec = json.loads(args.cases.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "scope": "deterministic_offline_security_gates",
        "context_gate": evaluate_context_gate(spec),
        "output_gate": evaluate_output_gate(),
        "latency": evaluate_latency(spec, max(1, args.iterations)),
        "not_measured": [
            "live_llm_attack_success_rate",
            "secure_answer_rate",
            "clean_answer_preservation_rate"
        ]
    }
    if args.corpus_root is not None:
        report["corpus_alerts"] = evaluate_corpus_alerts(
            args.corpus_root, max(1, args.corpus_limit)
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
