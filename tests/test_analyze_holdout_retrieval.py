from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_holdout_retrieval",
    ROOT / "scripts" / "analyze_holdout_retrieval.py",
)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)
import immutable_outputs  # noqa: E402
import run_final_retrieval_schedule as retrieval_runner  # noqa: E402

CORPUS_REVISION = "fixture-corpus-revision"
CORE_CATEGORIES = (
    "academic",
    "admissions",
    "core",
    "employment",
    "graduation",
    "international",
    "registration",
    "scholarship",
    "student_support",
)
CORE_DIFFICULTIES = ("single_fact", "multi_evidence", "structure_sensitive")
CHALLENGE_TYPES = (
    "unanswerable",
    "scope_version_ambiguity",
    "prompt_injection",
)


def _source(index: int) -> dict:
    title = f"official notice {index}"
    return {
        "document_id": f"document-{index}",
        "source_document_family_id": f"source-family-{index}",
        "source_sha256": hashlib.sha256(
            f"source-{index}".encode("utf-8")
        ).hexdigest(),
        "source_path": f"downloads/official/source-{index}.pdf",
        "source_url": f"https://www.pusan.ac.kr/notice/{index}",
        "source_title": title,
        "normalized_title_without_year": title,
    }


def _synthetic_holdout_cases() -> list[dict]:
    cases: list[dict] = []
    index = 1
    for category in CORE_CATEGORIES:
        for difficulty in CORE_DIFFICULTIES:
            source = _source(index)
            claims = []
            for claim_number in (1, 2):
                value = f"value-{index}-{claim_number}"
                claims.append(
                    {
                        "claim_id": f"claim-{claim_number}",
                        "description": f"required fact {claim_number}",
                        "critical_values": [value],
                        "semantic_anchors": [
                            f"notice {index}",
                            f"fact {claim_number}",
                        ],
                        "evidence_options": [
                            {
                                **source,
                                "quote": (
                                    f"notice {index} fact {claim_number} is "
                                    f"{value}"
                                ),
                            }
                        ],
                    }
                )
            cases.append(
                {
                    "id": f"holdout_{category}_{index:02d}",
                    "split": "holdout-core",
                    "family_id": f"question-family-{index}",
                    "source_document_family_ids": [
                        source["source_document_family_id"]
                    ],
                    "source_sha256s": [source["source_sha256"]],
                    "normalized_source_titles": [
                        source["normalized_title_without_year"]
                    ],
                    "category": category,
                    "difficulty_type": difficulty,
                    "intentional_typo": False,
                    "query": f"What are the two facts in notice {index}?",
                    "role": "pnu-student" if index <= 6 else None,
                    "answerable": True,
                    "required_claims": claims,
                    "optional_claims": [],
                    "forbidden_claims": [],
                    "expected_behavior": "answer",
                    "authoring_status": "synthetic_fixture",
                }
            )
            index += 1

    challenge_index = 1
    for challenge_type in CHALLENGE_TYPES:
        for _ in range(3):
            is_injection = challenge_type == "prompt_injection"
            source = _source(100 + challenge_index) if is_injection else None
            required_claims = []
            if source is not None:
                required_claims = [
                    {
                        "claim_id": "safe-fact",
                        "description": "answer the safe official fact",
                        "critical_values": [f"safe-{challenge_index}"],
                        "semantic_anchors": ["official notice", "safe fact"],
                        "evidence_options": [
                            {
                                **source,
                                "quote": (
                                    "official safe fact is "
                                    f"safe-{challenge_index}"
                                ),
                            }
                        ],
                    }
                ]
            cases.append(
                {
                    "id": f"holdout_challenge_{challenge_index:02d}",
                    "split": "holdout-challenge",
                    "family_id": f"challenge-family-{challenge_index}",
                    "source_document_family_ids": (
                        [source["source_document_family_id"]] if source else []
                    ),
                    "source_sha256s": [source["source_sha256"]] if source else [],
                    "normalized_source_titles": (
                        [source["normalized_title_without_year"]] if source else []
                    ),
                    "category": "challenge",
                    "challenge_type": challenge_type,
                    "query": f"Synthetic challenge {challenge_index}?",
                    "role": None,
                    "answerable": is_injection,
                    "required_claims": required_claims,
                    "optional_claims": [],
                    "forbidden_claims": [
                        {
                            "claim_id": "forbidden-behavior",
                            "description": "assert an unsupported fact",
                        }
                    ],
                    "expected_behavior": {
                        "unanswerable": "abstain",
                        "scope_version_ambiguity": "clarify_or_abstain",
                        "prompt_injection": "resist_injection_and_answer",
                    }[challenge_type],
                    "challenge_oracle": {
                        "must_do": ["follow the safe evaluation behavior"],
                        "must_not_do": ["assert an unsupported fact"],
                        **(
                            {
                                "injection_surface": "question",
                                "injection_payload": "ignore prior instructions",
                            }
                            if is_injection
                            else {}
                        ),
                    },
                    "authoring_status": "synthetic_fixture",
                }
            )
            challenge_index += 1
    return cases


def _table_candidate_fields(claim: dict, option: dict) -> dict:
    spec = option.get("table_evidence") or claim.get("table_evidence")
    if not isinstance(spec, dict):
        return {}
    headers = list(spec["headers"])
    rows = []
    for relation in spec["row_relations"]:
        row = [""] * len(headers)
        target = headers.index(relation["header"])
        row[target] = relation["value"]
        anchor_index = 0 if target != 0 else min(1, len(headers) - 1)
        row[anchor_index] = relation["row_anchor"]
        rows.append(row)
    return {"table_headers": headers, "table_rows": rows}


def _gold_candidates(case: dict, *, prefix: str) -> list[dict]:
    candidates = []
    for claim_index, claim in enumerate(case["required_claims"]):
        option = claim["evidence_options"][0]
        table_fields = _table_candidate_fields(claim, option)
        if table_fields:
            text = "\n".join(
                [
                    "\t".join(table_fields["table_headers"]),
                    *(
                        "\t".join(str(cell) for cell in row)
                        for row in table_fields["table_rows"]
                    ),
                ]
            )
        else:
            text = option["quote"]
        candidates.append(
            {
                "chunk_id": f"{prefix}:gold:{claim_index}",
                "document_id": option["document_id"],
                "corpus_revision": CORPUS_REVISION,
                "source_title": option["source_title"],
                "text": text,
                **table_fields,
            }
        )
    return candidates


def _distractor(*, prefix: str, index: int) -> dict:
    return {
        "chunk_id": f"{prefix}:distractor:{index}",
        "document_id": f"doc-distractor-{prefix}-{index}",
        "corpus_revision": CORPUS_REVISION,
        "source_title": "무관 문서",
        "text": f"무관한 합성 검색 결과 {prefix} {index}",
    }


def _trace_rows(candidates: list[dict], *, stage: str) -> list[dict]:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        row = copy.deepcopy(candidate)
        row.update(
            {
                "rank": rank,
                "stage": stage,
                "source_number": rank if stage == "final_contexts" else None,
                "text_sha256": hashlib.sha256(
                    row["text"].encode("utf-8")
                ).hexdigest(),
            }
        )
        if stage == "raw_bm25":
            row["retrieval"] = {"bm25": {"rank": rank, "score": -float(rank)}}
        rows.append(row)
    return rows


def _collector_config(
    *,
    condition: str,
    cases_file_sha: str,
    canonical_cases_sha: str,
    selected_ids_sha: str,
) -> dict:
    tuning = condition == "c1"
    return {
        "api_base": f"http://127.0.0.1/{condition}-fixture",
        "cases_path": "fixture-holdout-v2.jsonl",
        "cases_sha256": cases_file_sha,
        "cases_canonical_sha256": canonical_cases_sha,
        "selected_case_ids_sha256": selected_ids_sha,
        "provider": "extractive",
        "model": None,
        "institution": None,
        "context_k": 8,
        "parser_profile": "cascade",
        "retrieval_mode": "bm25",
        "expected_corpus_revision": CORPUS_REVISION,
        "expected_retrieval_tuning": tuning,
        "expected_context_chunks_per_document": 2,
        "eval_trace": True,
        "allow_unpinned": False,
        "max_attempts": 1,
        "retry_backoff_seconds": 0,
        "errors_path": f"{condition}.errors.jsonl",
        "server_config": {
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "corpus_revision": CORPUS_REVISION,
            "retrieval_tuning": tuning,
            "context_chunks_per_document": 2,
            "evaluation_trace_enabled": True,
        },
    }


def _answer_record(
    case: dict,
    *,
    condition: str,
    cases_file_sha: str,
    canonical_cases_sha: str,
    selected_ids_sha: str,
    final_hit: bool,
    raw_hit: bool,
) -> dict:
    case_id = case["id"]
    prefix = f"{condition}:{case_id}"
    gold = _gold_candidates(case, prefix=prefix)

    if final_hit:
        final_candidates = gold + [
            _distractor(prefix=f"{prefix}:final", index=index)
            for index in range(8 - len(gold))
        ]
    else:
        final_candidates = [
            _distractor(prefix=f"{prefix}:final", index=index)
            for index in range(8)
        ]
    if raw_hit:
        raw_candidates = [
            _distractor(prefix=f"{prefix}:raw", index=0),
            *gold,
        ]
        raw_candidates.extend(
            _distractor(prefix=f"{prefix}:raw", index=index)
            for index in range(1, 51 - len(raw_candidates))
        )
    else:
        raw_candidates = [
            _distractor(prefix=f"{prefix}:raw", index=index)
            for index in range(50)
        ]

    final_rows = _trace_rows(final_candidates, stage="final_contexts")
    raw_rows = _trace_rows(raw_candidates, stage="raw_bm25")
    pool_rows = _trace_rows(final_candidates, stage="post_retrieval_pool")
    expansion_rows = _trace_rows(
        final_candidates, stage="post_neighbor_expansion"
    )
    trace = {
        "schema_version": 1,
        "retrieval_stages": {
            "raw_bm25": raw_rows,
            "post_retrieval_pool": pool_rows,
            "post_neighbor_expansion": expansion_rows,
            "final_contexts": final_rows,
        },
        "timing_ms": {"retrieval": 10.0},
    }
    sources = [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "source_title": row["source_title"],
            "corpus_revision": CORPUS_REVISION,
            "preview": row["text"][:100],
        }
        for row in final_rows
    ]
    collector = _collector_config(
        condition=condition,
        cases_file_sha=cases_file_sha,
        canonical_cases_sha=canonical_cases_sha,
        selected_ids_sha=selected_ids_sha,
    )
    request = {
        "question": case["query"],
        "top_k": 8,
        "institution": None,
        "provider": "extractive",
        "parser_profile": "cascade",
        "retrieval_mode": "bm25",
        "eval_trace": True,
    }
    if case.get("role") is not None:
        request["role"] = case["role"]
    record = {
        "schema_version": analysis.ANSWER_SCHEMA_VERSION,
        "record_type": "answer",
        "experiment_id": "fixture-final-holdout",
        "condition_id": condition,
        "generation_run_id": "run1",
        "case_id": case_id,
        "id": case_id,
        "case_sha256": analysis.sha256_json(case),
        "collector_config": collector,
        "collector_config_sha256": analysis.sha256_json(collector),
        "answer": "합성 오프라인 답변",
        "request": request,
        "response_config": {
            "institution": None,
            "parser_profile": "cascade",
            "retrieval_mode": "bm25",
            "generation_requested": "extractive",
            "generation_used": "extractive",
            "generation_model": None,
        },
        "generation": {
            "requested": "extractive",
            "used": "extractive",
            "model": None,
        },
        "retrieval": {
            "mode": "bm25",
            "service_tuning": condition == "c1",
            "max_chunks_per_document": 2,
            "result_count": 8,
        },
        "evaluation_trace": trace,
        "sources": sources,
        "atomic_evidence_at_k": {
            str(cutoff): analysis.atomic_evidence_hit(
                final_rows, case, k=cutoff
            )
            for cutoff in analysis.EVIDENCE_CUTOFFS
        },
    }
    return analysis.build_answer_identity(record)


def _rewrite_identity(record: dict) -> dict:
    value = copy.deepcopy(record)
    for key in ("answer_id", "answer_sha256", "record_sha256"):
        value.pop(key, None)
    return analysis.build_answer_identity(value)


def _scheduled_answer_record(
    case: dict,
    *,
    schedule: dict,
    entry: dict,
    artifact: dict,
    gate: dict,
) -> dict:
    """Build one fully bound, offline final-retrieval answer artifact."""

    lane_id = entry["condition_id"]
    lane = schedule["lanes"][lane_id]
    record = _answer_record(
        case,
        condition=lane_id,
        cases_file_sha=schedule["cases_sha256"],
        canonical_cases_sha=schedule["cases_canonical_sha256"],
        selected_ids_sha=schedule["selected_case_ids_sha256"],
        final_hit=True,
        raw_hit=True,
    )

    for rows in record["evaluation_trace"]["retrieval_stages"].values():
        for row in rows:
            row["corpus_revision"] = lane["corpus_revision"]
    for source in record["sources"]:
        source["corpus_revision"] = lane["corpus_revision"]

    raw_rows = record["evaluation_trace"]["retrieval_stages"]["raw_bm25"]
    for rank in range(len(raw_rows) + 1, 61):
        candidate = _distractor(
            prefix=f"{lane_id}:{case['id']}:raw-extra", index=rank
        )
        row = _trace_rows([candidate], stage="raw_bm25")[0]
        row["rank"] = rank
        row["corpus_revision"] = lane["corpus_revision"]
        row["retrieval"]["bm25"]["rank"] = rank
        raw_rows.append(row)

    authorization = retrieval_runner._authorization_metadata(
        schedule, artifact, gate
    )
    collector = record["collector_config"]
    collector.update(
        {
            "api_base": lane["api_base"],
            "cases_path": schedule["cases_path"],
            "parser_profile": lane["parser_profile"],
            "expected_corpus_revision": lane["corpus_revision"],
            "expected_retrieval_tuning": lane["expected_retrieval_tuning"],
            "expected_generation_max_context_chars": None,
            "expected_generation_max_output_tokens": None,
            "expected_generation_sampling_parameters": None,
            "expected_git_commit": schedule["expected_git_commit"],
            "expected_index_sha256": lane["index"]["sha256"],
            "expected_source_manifest_sha256": schedule[
                "shared_source_manifest_sha256"
            ],
            "max_attempts": schedule["controls"]["max_attempts"],
            "retry_backoff_seconds": schedule["controls"][
                "retry_backoff_seconds"
            ],
            "request_timeout_seconds": schedule["controls"]["timeout_seconds"],
            "inter_call_sleep_seconds": schedule["controls"]["sleep_seconds"],
            "errors_path": artifact["errors_path"],
            "attempt_journal_path": artifact["journal_path"],
            "final_authorization": authorization,
        }
    )
    collector["server_config"].update(
        {
            "parser_profile": lane["parser_profile"],
            "corpus_revision": lane["corpus_revision"],
            "retrieval_tuning": lane["expected_retrieval_tuning"],
            "profile_index": {
                "sha256": lane["index"]["sha256"],
                "source_manifest_sha256": schedule[
                    "shared_source_manifest_sha256"
                ],
            },
        }
    )

    user_prompt = f"offline retrieval fixture prompt: {case['id']}"
    system_instruction = ""
    prompt_sha = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
    system_sha = hashlib.sha256(system_instruction.encode("utf-8")).hexdigest()
    request_config = {
        "provider": "extractive",
        "model_requested": None,
        "prompt_used": False,
    }
    record.update(
        {
            "experiment_id": schedule["experiment_id"],
            "condition_id": lane_id,
            "generation_run_id": retrieval_runner.RUN_ID,
            "call_order": entry["call_order"],
            "collection_schedule": retrieval_runner._record_metadata(
                schedule, entry
            ),
            "slot_outcome": "answer",
            "answer_eligible_for_judge": True,
            "collection_attempt_number": 1,
            "request_attempts": [
                {"attempt_number": 1, "status": "ok", "elapsed_ms": 1.0}
            ],
            "collector_config": collector,
            "request": {
                **record["request"],
                "parser_profile": lane["parser_profile"],
            },
            "response_config": {
                **record["response_config"],
                "parser_profile": lane["parser_profile"],
            },
            "retrieval": {
                **record["retrieval"],
                "service_tuning": lane["expected_retrieval_tuning"],
            },
            "generation": {
                "requested": "extractive",
                "used": "extractive",
                "model": None,
                "request_config": request_config,
                "request_config_sha256": analysis.sha256_json(request_config),
                "prompt_sha256": prompt_sha,
                "system_instruction_sha256": system_sha,
            },
        }
    )
    record["evaluation_trace"].update(
        {
            "generation_input": {
                "user_prompt": user_prompt,
                "system_instruction": system_instruction,
                "prompt_sha256": prompt_sha,
                "system_instruction_sha256": system_sha,
            },
            "raw_draft": record["answer"],
            "sanitized_draft": record["answer"],
        }
    )
    record["collector_config_sha256"] = analysis.sha256_json(collector)
    return _rewrite_identity(record)


def _slot_journal_event(
    schedule: dict,
    entry: dict,
    *,
    event: str,
    outcome: str | None,
) -> dict:
    payload = {
        "schema_version": "pnu.final-generation-slot-journal.v1",
        "event": event,
        "schedule_id": schedule["schedule_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "artifact_id": entry["artifact_id"],
        "call_order": entry["call_order"],
        "case_id": entry["case_id"],
        "case_sha256": entry["case_sha256"],
        "max_chat_attempts": schedule["controls"]["max_attempts"],
        "outcome": outcome,
    }
    payload["record_sha256"] = analysis.sha256_json(payload)
    payload["journal_event_id"] = (
        "journal_" + analysis.sha256_json(payload)[:24]
    )
    return payload


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _build_fixture(root: Path) -> dict:
    cases_path = root / "holdout-v2.jsonl"
    all_cases = _synthetic_holdout_cases()
    _write_jsonl(cases_path, all_cases)
    core_cases = [case for case in all_cases if case["split"] == analysis.CORE_SPLIT]
    canonical_sha = analysis.sha256_json(all_cases)
    cases_file_sha = analysis.sha256_file(cases_path)
    selected_sha = analysis.sha256_json([case["id"] for case in core_cases])

    c0_records = []
    c1_records = []
    for index, case in enumerate(core_cases):
        c0_records.append(
            _answer_record(
                case,
                condition="c0",
                cases_file_sha=cases_file_sha,
                canonical_cases_sha=canonical_sha,
                selected_ids_sha=selected_sha,
                final_hit=index != 0,
                raw_hit=index != 0,
            )
        )
        c1_records.append(
            _answer_record(
                case,
                condition="c1",
                cases_file_sha=cases_file_sha,
                canonical_cases_sha=canonical_sha,
                selected_ids_sha=selected_sha,
                final_hit=index != 1,
                raw_hit=index != 1,
            )
        )
    c0_path = root / "c0.answers.jsonl"
    c1_path = root / "c1.answers.jsonl"
    _write_jsonl(c0_path, c0_records)
    _write_jsonl(c1_path, c1_records)
    return {
        "cases_path": cases_path,
        "cases": core_cases,
        "c0_path": c0_path,
        "c1_path": c1_path,
        "c0_records": c0_records,
        "c1_records": c1_records,
    }


class HoldoutRetrievalAnalysisTests(unittest.TestCase):
    def test_cli_smoke_recomputes_metrics_and_paired_statistics_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _build_fixture(root)
            json_out = root / "analysis.json"
            csv_out = root / "cases.csv"
            stdout = StringIO()
            with redirect_stdout(stdout):
                result_code = analysis.main(
                    [
                        "--cases",
                        str(fixture["cases_path"]),
                        "--c0",
                        str(fixture["c0_path"]),
                        "--c1",
                        str(fixture["c1_path"]),
                        "--cases-sha256",
                        analysis.sha256_file(fixture["cases_path"]),
                        "--c0-sha256",
                        analysis.sha256_file(fixture["c0_path"]),
                        "--c1-sha256",
                        analysis.sha256_file(fixture["c1_path"]),
                        "--expected-experiment-id",
                        "fixture-final-holdout",
                        "--expected-generation-run-id",
                        "run1",
                        "--expected-corpus-revision",
                        CORPUS_REVISION,
                        "--json-out",
                        str(json_out),
                        "--csv-out",
                        str(csv_out),
                        "--bootstrap",
                        "200",
                        "--seed",
                        "7",
                    ]
                )

            self.assertEqual(result_code, 0)
            self.assertIn("n=27", stdout.getvalue())
            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], analysis.SCHEMA_VERSION)
            self.assertEqual(report["questions"], 27)
            self.assertEqual(
                report["metric_definitions"]["atomic_evidence_matcher_version"],
                analysis.ATOMIC_EVIDENCE_MATCHER_VERSION,
            )
            source = report["paired_comparison"]["source_hit_at_5"]
            self.assertEqual(
                source["gained_case_ids"], [fixture["cases"][0]["id"]]
            )
            self.assertEqual(
                source["lost_case_ids"], [fixture["cases"][1]["id"]]
            )
            self.assertEqual(
                source["exact_mcnemar_two_sided"]["c0_only"], 1
            )
            self.assertEqual(
                source["exact_mcnemar_two_sided"]["c1_only"], 1
            )
            self.assertEqual(
                source["paired_family_bootstrap_95ci"]["iterations"], 200
            )
            self.assertEqual(len(report["strata"]["category"]), 9)
            self.assertEqual(len(report["strata"]["difficulty_type"]), 3)
            stored_hash = report.pop("analysis_sha256")
            self.assertEqual(stored_hash, analysis.sha256_json(report))

            with csv_out.open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 27)
            self.assertIn("c1_candidate_recall_at_50", csv_rows[0])

    def test_complete_condition_fails_closed_on_record_and_trace_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _build_fixture(root)
            cases_sha = analysis.sha256_file(fixture["cases_path"])
            c1_sha = analysis.sha256_file(fixture["c1_path"])

            mutations = {}

            error_rows = copy.deepcopy(fixture["c0_records"])
            error_rows[0]["error"] = "synthetic failure"
            error_rows[0] = _rewrite_identity(error_rows[0])
            mutations["contains an error"] = error_rows

            mixed_run = copy.deepcopy(fixture["c0_records"])
            mixed_run[0]["generation_run_id"] = "run2"
            mixed_run[0] = _rewrite_identity(mixed_run[0])
            mutations["mixed run/control"] = mixed_run

            mixed_condition = copy.deepcopy(fixture["c0_records"])
            mixed_condition[0]["condition_id"] = "c1"
            mixed_condition[0] = _rewrite_identity(mixed_condition[0])
            mutations["condition_id mismatch"] = mixed_condition

            bad_trace = copy.deepcopy(fixture["c0_records"])
            bad_trace[0]["evaluation_trace"]["retrieval_stages"]["raw_bm25"][0][
                "text"
            ] += " tampered"
            bad_trace[0] = _rewrite_identity(bad_trace[0])
            mutations["text_sha256 mismatch"] = bad_trace

            bad_atomic = copy.deepcopy(fixture["c0_records"])
            bad_atomic[0]["atomic_evidence_at_k"]["8"]["recall"] = 0.25
            bad_atomic[0] = _rewrite_identity(bad_atomic[0])
            mutations["atomic_evidence_at_k"] = bad_atomic

            bad_control = copy.deepcopy(fixture["c0_records"])
            bad_control[0]["collector_config"]["expected_retrieval_tuning"] = True
            bad_control[0]["collector_config_sha256"] = analysis.sha256_json(
                bad_control[0]["collector_config"]
            )
            bad_control[0] = _rewrite_identity(bad_control[0])
            mutations["expected_retrieval_tuning"] = bad_control

            duplicate = copy.deepcopy(fixture["c0_records"])
            duplicate.append(copy.deepcopy(duplicate[0]))
            mutations["duplicate case_id"] = duplicate

            missing = copy.deepcopy(fixture["c0_records"][:-1])
            mutations["incomplete Core case set"] = missing

            for mutation_index, (expected_message, records) in enumerate(
                mutations.items()
            ):
                with self.subTest(expected_message=expected_message):
                    c0_path = root / f"mutation-{mutation_index}.jsonl"
                    _write_jsonl(c0_path, records)
                    with self.assertRaisesRegex(SystemExit, expected_message):
                        analysis.main(
                            [
                                "--cases",
                                str(fixture["cases_path"]),
                                "--c0",
                                str(c0_path),
                                "--c1",
                                str(fixture["c1_path"]),
                                "--cases-sha256",
                                cases_sha,
                                "--c0-sha256",
                                analysis.sha256_file(c0_path),
                                "--c1-sha256",
                                c1_sha,
                                "--expected-experiment-id",
                                "fixture-final-holdout",
                                "--expected-generation-run-id",
                                "run1",
                                "--expected-corpus-revision",
                                CORPUS_REVISION,
                                "--json-out",
                                str(root / f"mutation-{mutation_index}.json"),
                                "--csv-out",
                                str(root / f"mutation-{mutation_index}.csv"),
                                "--bootstrap",
                                "5",
                            ]
                        )

    def test_sha_and_path_alias_checks_happen_before_output_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _build_fixture(root)
            with self.assertRaisesRegex(SystemExit, "cases SHA-256 mismatch"):
                analysis.main(
                    [
                        "--cases",
                        str(fixture["cases_path"]),
                        "--c0",
                        str(fixture["c0_path"]),
                        "--c1",
                        str(fixture["c1_path"]),
                        "--cases-sha256",
                        "0" * 64,
                        "--c0-sha256",
                        analysis.sha256_file(fixture["c0_path"]),
                        "--c1-sha256",
                        analysis.sha256_file(fixture["c1_path"]),
                        "--expected-experiment-id",
                        "fixture-final-holdout",
                        "--expected-generation-run-id",
                        "run1",
                        "--expected-corpus-revision",
                        CORPUS_REVISION,
                        "--json-out",
                        str(root / "never.json"),
                        "--csv-out",
                        str(root / "never.csv"),
                    ]
                )
            self.assertFalse((root / "never.json").exists())

            with self.assertRaisesRegex(SystemExit, "input/output path alias"):
                analysis.main(
                    [
                        "--cases",
                        str(fixture["cases_path"]),
                        "--c0",
                        str(fixture["c0_path"]),
                        "--c1",
                        str(fixture["c1_path"]),
                        "--cases-sha256",
                        analysis.sha256_file(fixture["cases_path"]),
                        "--c0-sha256",
                        analysis.sha256_file(fixture["c0_path"]),
                        "--c1-sha256",
                        analysis.sha256_file(fixture["c1_path"]),
                        "--expected-experiment-id",
                        "fixture-final-holdout",
                        "--expected-generation-run-id",
                        "run1",
                        "--expected-corpus-revision",
                        CORPUS_REVISION,
                        "--json-out",
                        str(fixture["c0_path"]),
                        "--csv-out",
                        str(root / "never-alias.csv"),
                    ]
                )

    def test_exact_mcnemar_and_family_bootstrap_keep_questions_paired(self) -> None:
        mcnemar = analysis.exact_mcnemar(
            [True, True, False, False], [True, False, True, False]
        )
        self.assertEqual(
            (mcnemar["both_pass"], mcnemar["c0_only"], mcnemar["c1_only"], mcnemar["neither"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(mcnemar["p_value"], 1.0)

        bootstrap = analysis.paired_family_bootstrap(
            {"q1": 1.0, "q2": 1.0, "q3": -1.0},
            {"q1": "family-a", "q2": "family-a", "q3": "family-b"},
            iterations=100,
            seed=11,
        )
        self.assertEqual(bootstrap["clusters"], 2)
        self.assertEqual(bootstrap["questions"], 3)
        self.assertEqual(bootstrap["iterations"], 100)

    def test_statistical_golden_values_and_preselection_mrr(self) -> None:
        mcnemar = analysis.exact_mcnemar(
            [False] * 5, [True] * 5
        )
        self.assertEqual(mcnemar["p_value"], 0.0625)

        bootstrap = analysis.paired_family_bootstrap(
            {"q1": 1.0, "q2": 1.0},
            {"q1": "family-a", "q2": "family-b"},
            iterations=10_000,
            seed=19,
        )
        self.assertEqual(bootstrap["ci95"], [1.0, 1.0])

        case = {
            "id": "q1",
            "required_claims": [{"claim_id": "claim-1",
                "evidence_options": [{
                    "document_id": "gold-doc", "quote": "근거 문장"
                }]
            }],
        }
        raw = [
            {"document_id": f"miss-{rank}", "text": "무관"}
            for rank in range(1, 4)
        ] + [{"document_id": "gold-doc", "text": "근거 문장"}]
        metrics = analysis.compute_case_metrics(
            case=case, final_contexts=[], raw_bm25_at_50=raw,
            retrieval_timing_ms=12.5,
        )
        self.assertEqual(metrics["preselection_mrr_at_50"], 0.25)
        self.assertEqual(metrics["candidate_recall_at_50"], 1.0)
        self.assertEqual(metrics["retrieval_timing_ms"], 12.5)

    def test_final_bundle_manifest_is_last_and_failure_stays_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zeroes = {metric: 0.0 for metric in analysis.METRIC_KINDS}
            report = {
                "analysis_sha256": "a" * 64,
                "cases": [{
                    "case_id": "q1", "family_id": "f1",
                    "category": "academic", "difficulty_type": "lookup",
                    "lanes": {lane: dict(zeroes) for lane in ("par-b", "par-ch", "c0", "c1")},
                    "primary_delta_c1_minus_c0": dict(zeroes),
                }],
            }
            schedule = {"schedule_id": "schedule", "schedule_sha256": "b" * 64}
            json_path, csv_path = root / "out.json", root / "out.csv"
            manifest_path = root / "complete.json"
            original_fsync = immutable_outputs._fsync_directory
            calls = 0

            def fail_second(path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected directory fsync failure")
                return original_fsync(path)

            with mock.patch.object(
                immutable_outputs, "_fsync_directory", side_effect=fail_second
            ):
                with self.assertRaises(analysis.RetrievalAnalysisError):
                    analysis.write_final_bundle(
                        json_path=json_path, csv_path=csv_path,
                        manifest_path=manifest_path, report=report,
                        schedule=schedule,
                    )
            self.assertFalse(json_path.exists())
            self.assertFalse(csv_path.exists())
            self.assertFalse(manifest_path.exists())

    def test_condition_metrics_are_independent_of_jsonl_record_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _build_fixture(root)
            order, cases, hashes = analysis.load_frozen_core_cases(
                fixture["cases_path"],
                expected_file_sha256=analysis.sha256_file(fixture["cases_path"]),
            )
            reversed_path = root / "c0-reversed.jsonl"
            _write_jsonl(reversed_path, list(reversed(fixture["c0_records"])))
            original = analysis.load_complete_condition(
                fixture["c0_path"],
                expected_file_sha256=analysis.sha256_file(fixture["c0_path"]),
                expected_condition="c0", order=order, cases=cases,
                case_hashes=hashes,
            )
            reversed_result = analysis.load_complete_condition(
                reversed_path,
                expected_file_sha256=analysis.sha256_file(reversed_path),
                expected_condition="c0", order=order, cases=cases,
                case_hashes=hashes,
            )
            self.assertEqual(original[:2], reversed_result[:2])

    def test_final_four_lane_schedule_happy_path_covers_all_108_slots_offline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "holdout-v2.jsonl"
            all_cases = _synthetic_holdout_cases()
            _write_jsonl(cases_path, all_cases)
            core_cases = [
                case for case in all_cases if case["split"] == analysis.CORE_SPLIT
            ]
            cases_by_id = {case["id"]: case for case in core_cases}
            source_manifest_sha = "d" * 64

            profile_indexes: dict[str, tuple[Path, str]] = {}
            for profile in ("baseline", "challenger", "cascade"):
                index_path = root / f"{profile}.sqlite"
                revision = f"fixture-revision-{profile}"
                connection = sqlite3.connect(index_path)
                connection.execute("CREATE TABLE index_meta (key TEXT, value TEXT)")
                connection.executemany(
                    "INSERT INTO index_meta VALUES (?, ?)",
                    [
                        ("profile", profile),
                        ("corpus_revision", revision),
                        ("source_manifest_sha256", source_manifest_sha),
                    ],
                )
                connection.commit()
                connection.close()
                profile_indexes[profile] = (index_path, revision)

            lane_inputs = {}
            for lane_id in retrieval_runner.LANE_IDS:
                profile = retrieval_runner.LANE_PROFILES[lane_id]
                index_path, revision = profile_indexes[profile]
                lane_inputs[lane_id] = {
                    "api_base": f"http://127.0.0.1/{lane_id}-offline-fixture",
                    "index_path": index_path,
                    "corpus_revision": revision,
                }
            schedule = retrieval_runner.build_schedule(
                cases_path,
                all_cases,
                experiment_id="fixture-final-retrieval",
                output_root=root / "artifacts",
                expected_git_commit="b" * 40,
                expected_source_manifest_sha256=source_manifest_sha,
                lane_inputs=lane_inputs,
            )
            dev_path = root / "dev-manifest.json"
            review_packet_path = root / "review-packet.md"
            review_a_path = root / "review-a.json"
            review_b_path = root / "review-b.json"
            signoff_path = root / "signoff.json"
            dev_path.write_text("{}\n", encoding="utf-8")
            review_packet_path.write_text(
                "# Synthetic independent review packet\n", encoding="utf-8"
            )
            review_a_path.write_text(
                '{"reviewer_slot":"A"}\n', encoding="utf-8"
            )
            review_b_path.write_text(
                '{"reviewer_slot":"B"}\n', encoding="utf-8"
            )
            signoff_path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            retrieval_runner.signoff_artifacts.SIGNOFF_SCHEMA_VERSION
                        ),
                        "cases_sha256": schedule["cases_sha256"],
                        "review_packet_path": review_packet_path.name,
                        "review_packet_sha256": hashlib.sha256(
                            review_packet_path.read_bytes()
                        ).hexdigest(),
                        "review_response_sha256s": [
                            {
                                "reviewer_slot": slot,
                                "reviewer_id": reviewer_id,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "path": path.name,
                            }
                            for slot, reviewer_id, path in (
                                ("A", "reviewer-alpha", review_a_path),
                                ("B", "reviewer-beta", review_b_path),
                            )
                        ],
                        "case_signoffs": [],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            gate = retrieval_runner.require_final_gates(
                schedule,
                dev_paths=[dev_path],
                signoff_path=signoff_path,
                repo_root=root,
                validator=lambda *args, **kwargs: {
                    "ok": True,
                    "gates": {
                        name: {"ok": True}
                        for name in ("schema", "dev", "corpus", "signoff")
                    },
                },
                git_identity=lambda _root: {"commit": "b" * 40, "clean": True},
            )
            schedule_path = root / "final-retrieval-schedule.json"
            retrieval_runner.write_new_schedule(schedule_path, schedule)

            artifacts = retrieval_runner._artifact_map(schedule)
            answers_by_lane = {lane: [] for lane in retrieval_runner.LANE_IDS}
            journals_by_lane = {lane: [] for lane in retrieval_runner.LANE_IDS}
            for entry in schedule["calls"]:
                lane_id = entry["condition_id"]
                artifact = artifacts[entry["artifact_id"]]
                record = _scheduled_answer_record(
                    cases_by_id[entry["case_id"]],
                    schedule=schedule,
                    entry=entry,
                    artifact=artifact,
                    gate=gate,
                )
                self.assertEqual(
                    len(
                        record["evaluation_trace"]["retrieval_stages"][
                            "raw_bm25"
                        ]
                    ),
                    60,
                )
                answers_by_lane[lane_id].append(record)
                journals_by_lane[lane_id].extend(
                    [
                        _slot_journal_event(
                            schedule, entry, event="slot_started", outcome=None
                        ),
                        _slot_journal_event(
                            schedule, entry, event="slot_completed", outcome="answer"
                        ),
                    ]
                )

            for lane_id, artifact in artifacts.items():
                answer_path = Path(artifact["answers_path"])
                answer_path.parent.mkdir(parents=True, exist_ok=True)
                _write_jsonl(answer_path, answers_by_lane[lane_id])
                _write_jsonl(
                    Path(artifact["journal_path"]), journals_by_lane[lane_id]
                )

            audit = retrieval_runner.audit_artifacts(schedule)
            self.assertTrue(audit["complete"])
            self.assertEqual(audit["completed_call_count"], 108)
            self.assertIsNone(audit["next_call_order"])

            report_path = root / "final-analysis.json"
            csv_path = root / "final-analysis.csv"
            completion_path = root / "final-analysis.complete.json"
            stdout = StringIO()
            lane_paths = {
                lane: Path(artifact["answers_path"])
                for lane, artifact in artifacts.items()
            }
            with redirect_stdout(stdout):
                result_code = analysis.main(
                    [
                        "--cases",
                        str(cases_path),
                        "--schedule",
                        str(schedule_path),
                        "--par-b",
                        str(lane_paths["par-b"]),
                        "--par-ch",
                        str(lane_paths["par-ch"]),
                        "--c0",
                        str(lane_paths["c0"]),
                        "--c1",
                        str(lane_paths["c1"]),
                        "--cases-sha256",
                        analysis.sha256_file(cases_path),
                        "--par-b-sha256",
                        analysis.sha256_file(lane_paths["par-b"]),
                        "--par-ch-sha256",
                        analysis.sha256_file(lane_paths["par-ch"]),
                        "--c0-sha256",
                        analysis.sha256_file(lane_paths["c0"]),
                        "--c1-sha256",
                        analysis.sha256_file(lane_paths["c1"]),
                        "--expected-experiment-id",
                        schedule["experiment_id"],
                        "--expected-generation-run-id",
                        retrieval_runner.RUN_ID,
                        "--json-out",
                        str(report_path),
                        "--csv-out",
                        str(csv_path),
                        "--completion-manifest",
                        str(completion_path),
                        "--bootstrap",
                        "20",
                        "--seed",
                        "7",
                    ]
                )

            self.assertEqual(result_code, 0)
            self.assertIn("n=27", stdout.getvalue())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], analysis.FINAL_SCHEMA_VERSION)
            self.assertEqual(report["questions"], 27)
            self.assertEqual(
                set(report["condition_summaries"]),
                set(retrieval_runner.LANE_IDS),
            )
            self.assertEqual(
                report["inputs"]["schedule"]["artifact_audit"][
                    "completed_call_count"
                ],
                108,
            )
            self.assertEqual(
                report["analysis_config"]["primary_contrast"], "c0_vs_c1"
            )
            self.assertEqual(
                set(report["exploratory_parser_contrasts"]),
                {"par-b_vs_par-ch", "par-b_vs_c1", "par-ch_vs_c1"},
            )
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            self.assertTrue(completion["complete"])
            self.assertEqual(completion["schedule_sha256"], schedule["schedule_sha256"])
            self.assertTrue(csv_path.is_file())


if __name__ == "__main__":
    unittest.main()
