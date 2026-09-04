from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "analyze_shadow_retrieval.py"
    spec = importlib.util.spec_from_file_location("analyze_shadow_retrieval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = _load_module()


def _case(case_id: str, bucket: str, *, challenge_type: str = "") -> dict:
    case = {
        "id": case_id,
        "family_id": f"family-{case_id}",
        "shadow_bucket": bucket,
        "category": "synthetic",
        "answerable": True,
        "required_claims": [
            {
                "claim_id": "c1",
                "evidence_options": [{"document_id": f"gold-{case_id}"}],
            }
        ],
    }
    if challenge_type:
        case["challenge_type"] = challenge_type
    return case


def _record(case_id: str, *, hit: bool, recall: float) -> dict:
    contexts = [
        {
            "document_id": f"gold-{case_id}" if hit else "wrong-document",
            "chunk_id": "chunk-1",
        }
    ]
    evidence = {
        str(k): {
            "available": True,
            "all_matched": hit,
            "recall": recall,
        }
        for k in analyzer.K_VALUES
    }
    return {
        "id": case_id,
        "latency_ms": 10,
        "evaluation_trace": {"retrieval_stages": {"final_contexts": contexts}},
        "atomic_evidence_at_k": evidence,
    }


class AnalyzeShadowRetrievalTests(unittest.TestCase):
    def test_exact_mcnemar_uses_paired_discordance(self) -> None:
        result = analyzer._exact_mcnemar(
            [True, True, False, False],
            [True, False, True, False],
        )

        self.assertEqual(result["c0_only"], 1)
        self.assertEqual(result["c1_only"], 1)
        self.assertEqual(result["discordant"], 2)
        self.assertEqual(result["p_value_two_sided_exact"], 1.0)

    def test_analyze_reports_question_level_gain(self) -> None:
        cases = [
            _case("core", "simple"),
            _case("role", "role_variant"),
            _case("inject", "challenge", challenge_type="prompt_injection"),
        ]
        c0 = {
            "core": _record("core", hit=False, recall=0.0),
            "role": _record("role", hit=True, recall=1.0),
            "inject": _record("inject", hit=True, recall=1.0),
        }
        c1 = {
            "core": _record("core", hit=True, recall=1.0),
            "role": _record("role", hit=True, recall=1.0),
            "inject": _record("inject", hit=True, recall=1.0),
        }

        report, rows = analyzer.analyze(
            cases,
            c0,
            c1,
            inputs={"fixture": True},
            iterations=100,
            seed=7,
        )

        core = report["groups"]["core"]
        self.assertEqual(core["c0"]["all_evidence_at_8"]["hits"], 0)
        self.assertEqual(core["c1"]["all_evidence_at_8"]["hits"], 1)
        self.assertEqual(
            core["paired"]["all_evidence_at_8"]["gained_case_ids"],
            ["core"],
        )
        self.assertEqual(report["case_change_counts_at_8"]["gained"], 1)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
