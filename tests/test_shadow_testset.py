from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module("build_shadow_testset", "scripts/build_shadow_testset.py")
validator = _load_module("validate_shadow_testset", "scripts/validate_shadow_testset.py")


class ShadowTestsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases_path = ROOT / "config" / "pnu-service-shadow60-v1.jsonl"
        cls.blueprint_path = (
            ROOT / "config" / "pnu-service-shadow60-v1.blueprint.json"
        )
        cls.index_path = (
            ROOT
            / "processed"
            / "index"
            / "pnu-20260725-curated-cascade-v5-allow-suspect.sqlite"
        )
        cls.source_manifest_path = (
            ROOT
            / "processed"
            / "curation"
            / "20260725-pnu-curated-v5"
            / "curated-manifest.jsonl"
        )
        cls.dev_manifest_path = (
            ROOT / "config" / "pnu-service-dev-source-manifest.json"
        )
        cls.cases = validator._read_jsonl(cls.cases_path)
        cls.documents = validator._load_documents(cls.index_path)
        cls.source_shas = validator._load_source_shas(cls.source_manifest_path)
        cls.dev = validator._dev_fingerprints(cls.dev_manifest_path)

    def errors_for(self, cases: list[dict]) -> list[str]:
        return validator.collect_errors(
            cases,
            documents=self.documents,
            source_shas=self.source_shas,
            dev=self.dev,
        )

    def test_frozen_shadow60_passes_preflight(self) -> None:
        report = validator.validate_paths(
            self.cases_path,
            index_path=self.index_path,
            source_manifest_path=self.source_manifest_path,
            dev_manifest_path=self.dev_manifest_path,
        )

        self.assertEqual(report["case_count"], 60)
        self.assertEqual(report["bucket_counts"], validator.EXPECTED_BUCKETS)
        self.assertEqual(report["unique_core_source_documents"], 42)
        self.assertEqual(report["dev_source_overlap_count"], 0)

    def test_blueprint_materializes_byte_equivalent_records(self) -> None:
        blueprint = json.loads(self.blueprint_path.read_text(encoding="utf-8"))
        rebuilt = builder.build_cases(
            blueprint,
            chunks=builder._load_chunks(self.index_path),
            manifest=builder._load_manifest(self.source_manifest_path),
        )

        self.assertEqual(rebuilt, self.cases)

    def test_rejects_missing_critical_value(self) -> None:
        cases = copy.deepcopy(self.cases)
        cases[0]["required_claims"][0]["critical_values"].append(
            "근거에 존재하지 않는 합성 값"
        )

        errors = self.errors_for(cases)

        self.assertTrue(any("absent from evidence quote" in error for error in errors))

    def test_rejects_composition_drift(self) -> None:
        cases = copy.deepcopy(self.cases)
        cases[0]["shadow_bucket"] = "simple"

        errors = self.errors_for(cases)

        self.assertTrue(any("bucket counts" in error for error in errors))
        self.assertTrue(any("simple case must contain 1" in error for error in errors))

    def test_rejects_role_variant_without_parent_family(self) -> None:
        cases = copy.deepcopy(self.cases)
        role_case = next(
            case for case in cases if case["shadow_bucket"] == "role_variant"
        )
        role_case["family_id"] = "unrelated-family"

        errors = self.errors_for(cases)

        self.assertTrue(any("must share parent family_id" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
