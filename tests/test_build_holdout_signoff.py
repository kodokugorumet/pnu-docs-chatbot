from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


signoff_module = _load_module(
    "build_holdout_signoff", ROOT / "scripts" / "build_holdout_signoff.py"
)
validator_module = _load_module(
    "validate_service_holdout_for_signoff_test",
    ROOT / "scripts" / "validate_service_holdout.py",
)


class BuildHoldoutSignoffTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Any]:
        cases = [
            {"id": "core-1", "query": "첫 질문", "split": "holdout-core"},
            {
                "id": "challenge-1",
                "query": "둘째 질문",
                "split": "holdout-challenge",
            },
            {"id": "core-2", "query": "셋째 질문", "split": "holdout-core"},
        ]
        cases_path = root / "config" / "cases.jsonl"
        packet_path = root / "docs" / "review.md"
        reviewer_a_path = root / "reviews" / "review-a.json"
        reviewer_b_path = root / "reviews" / "review-b.json"
        output_path = root / "evidence" / "signoff.json"
        cases_path.parent.mkdir(parents=True)
        packet_path.parent.mkdir(parents=True)
        cases_path.write_bytes(
            b"".join(
                (
                    json.dumps(case, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                for case in cases
            )
        )
        packet_path.write_bytes("# 정확한 검토 패킷\n".encode("utf-8"))
        return {
            "root": root,
            "cases": cases,
            "cases_path": cases_path,
            "packet_path": packet_path,
            "reviewer_a_path": reviewer_a_path,
            "reviewer_b_path": reviewer_b_path,
            "output_path": output_path,
        }

    def _create(self, fixture: dict[str, Any]) -> int:
        return signoff_module.main(
            [
                "--create-templates",
                "--cases",
                str(fixture["cases_path"]),
                "--review-packet",
                str(fixture["packet_path"]),
                "--review-a",
                str(fixture["reviewer_a_path"]),
                "--review-b",
                str(fixture["reviewer_b_path"]),
            ]
        )

    def _complete(
        self,
        path: Path,
        reviewer_id: str,
        *,
        notes_prefix: str = "checked",
    ) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["reviewer_id"] = reviewer_id
        value["independent_review_confirmed"] = True
        for index, record in enumerate(value["case_reviews"]):
            for check in signoff_module.REVIEW_CHECKS:
                record[check] = True
            record["decision"] = "PASS"
            record["notes"] = f"{notes_prefix}-{index + 1}"
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return value

    def _merge(
        self,
        fixture: dict[str, Any],
        *,
        output_path: Optional[Path] = None,
    ) -> int:
        return signoff_module.main(
            [
                "--merge",
                "--cases",
                str(fixture["cases_path"]),
                "--review-packet",
                str(fixture["packet_path"]),
                "--review-a",
                str(fixture["reviewer_a_path"]),
                "--review-b",
                str(fixture["reviewer_b_path"]),
                "--output",
                str(output_path or fixture["output_path"]),
            ]
        )

    def _completed_fixture(self, root: Path) -> dict[str, Any]:
        fixture = self._fixture(root)
        self.assertEqual(self._create(fixture), 0)
        self._complete(fixture["reviewer_a_path"], "reviewer-alpha", notes_prefix="A")
        self._complete(fixture["reviewer_b_path"], "reviewer-beta", notes_prefix="B")
        return fixture

    def test_create_binds_exact_bytes_and_emits_separate_safe_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            self.assertEqual(self._create(fixture), 0)
            raw_a = fixture["reviewer_a_path"].read_bytes()
            raw_b = fixture["reviewer_b_path"].read_bytes()
            reviewer_a = json.loads(raw_a)
            reviewer_b = json.loads(raw_b)
            cases_sha = hashlib.sha256(
                fixture["cases_path"].read_bytes()
            ).hexdigest()
            packet_sha = hashlib.sha256(
                fixture["packet_path"].read_bytes()
            ).hexdigest()
            self.assertNotEqual(raw_a, raw_b)
            self.assertEqual(reviewer_a["reviewer_slot"], "A")
            self.assertEqual(reviewer_b["reviewer_slot"], "B")
            self.assertEqual(reviewer_a["reviewer_id"], "__REVIEWER_A_ID__")
            self.assertEqual(reviewer_b["reviewer_id"], "__REVIEWER_B_ID__")
            for response in (reviewer_a, reviewer_b):
                self.assertEqual(response["cases_sha256"], cases_sha)
                self.assertEqual(response["review_packet_sha256"], packet_sha)
                self.assertIs(response["independent_review_confirmed"], False)
                self.assertEqual(
                    [record["case_id"] for record in response["case_reviews"]],
                    [case["id"] for case in fixture["cases"]],
                )
                self.assertEqual(
                    len(
                        {record["case_id"] for record in response["case_reviews"]}
                    ),
                    len(fixture["cases"]),
                )
                for record in response["case_reviews"]:
                    self.assertEqual(record["decision"], "PENDING")
                    self.assertEqual(record["notes"], "")
                    for check in signoff_module.REVIEW_CHECKS:
                        self.assertIsNone(record[check])

    def test_create_is_deterministic_across_fresh_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._fixture(root / "first")
            second = self._fixture(root / "second")
            self.assertEqual(self._create(first), 0)
            self.assertEqual(self._create(second), 0)
            self.assertEqual(
                first["reviewer_a_path"].read_bytes(),
                second["reviewer_a_path"].read_bytes(),
            )
            self.assertEqual(
                first["reviewer_b_path"].read_bytes(),
                second["reviewer_b_path"].read_bytes(),
            )

    def test_pair_publish_rejects_preexisting_companion_without_partial_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture["reviewer_b_path"].parent.mkdir(parents=True)
            fixture["reviewer_b_path"].write_bytes(b"preserve me")
            self.assertEqual(self._create(fixture), 2)
            self.assertFalse(fixture["reviewer_a_path"].exists())
            self.assertEqual(fixture["reviewer_b_path"].read_bytes(), b"preserve me")

    def test_check_templates_accepts_current_and_rejects_stale_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            self.assertEqual(self._create(fixture), 0)
            arguments = [
                "--check-templates",
                "--cases",
                str(fixture["cases_path"]),
                "--review-packet",
                str(fixture["packet_path"]),
                "--review-a",
                str(fixture["reviewer_a_path"]),
                "--review-b",
                str(fixture["reviewer_b_path"]),
            ]
            self.assertEqual(signoff_module.main(arguments), 0)
            fixture["reviewer_a_path"].write_bytes(b"{}\n")
            self.assertEqual(signoff_module.main(arguments), 1)

    def test_check_templates_rejects_stale_cases_and_packet(self) -> None:
        for stale_input in ("cases_path", "packet_path"):
            with self.subTest(stale_input=stale_input):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._fixture(Path(directory))
                    self.assertEqual(self._create(fixture), 0)
                    fixture[stale_input].write_bytes(
                        fixture[stale_input].read_bytes() + b" \n"
                    )
                    self.assertEqual(
                        signoff_module.main(
                            [
                                "--check-templates",
                                "--cases",
                                str(fixture["cases_path"]),
                                "--review-packet",
                                str(fixture["packet_path"]),
                                "--review-a",
                                str(fixture["reviewer_a_path"]),
                                "--review-b",
                                str(fixture["reviewer_b_path"]),
                            ]
                        ),
                        1,
                    )

    def test_valid_merge_preserves_provenance_and_matches_existing_validator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._completed_fixture(root)
            raw_a = fixture["reviewer_a_path"].read_bytes()
            raw_b = fixture["reviewer_b_path"].read_bytes()
            old_root = signoff_module.REPO_ROOT
            try:
                signoff_module.REPO_ROOT = root
                self.assertEqual(self._merge(fixture), 0)
            finally:
                signoff_module.REPO_ROOT = old_root
            result = json.loads(fixture["output_path"].read_text(encoding="utf-8"))

            self.assertEqual(
                result["cases_sha256"],
                hashlib.sha256(fixture["cases_path"].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["review_packet_sha256"],
                hashlib.sha256(fixture["packet_path"].read_bytes()).hexdigest(),
            )
            self.assertEqual(result["review_packet_path"], "docs/review.md")
            provenance = result["review_response_sha256s"]
            self.assertEqual(
                [entry["sha256"] for entry in provenance],
                [hashlib.sha256(raw_a).hexdigest(), hashlib.sha256(raw_b).hexdigest()],
            )
            self.assertEqual(
                [entry["reviewer_id"] for entry in provenance],
                ["reviewer-alpha", "reviewer-beta"],
            )
            self.assertEqual(
                [entry["path"] for entry in provenance],
                ["reviews/review-a.json", "reviews/review-b.json"],
            )
            self.assertEqual(
                [record["case_id"] for record in result["case_signoffs"]],
                [case["id"] for case in fixture["cases"]],
            )
            self.assertEqual(
                result["case_signoffs"][0]["reviewers"][0]["notes"], "A-1"
            )
            gate = validator_module._validate_signoff_gate(
                fixture["cases"],
                fixture["output_path"],
                expected_cases_sha256=result["cases_sha256"],
                cases_path=fixture["cases_path"],
                repo_root=root,
            )
            self.assertTrue(gate["ok"], gate["errors"])
            self.assertEqual(gate["approved_case_count"], len(fixture["cases"]))

    def test_merge_rejects_stale_cases_or_review_packet(self) -> None:
        for stale_input in ("cases_path", "packet_path"):
            with self.subTest(stale_input=stale_input):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._completed_fixture(Path(directory))
                    fixture[stale_input].write_bytes(
                        fixture[stale_input].read_bytes() + b" \n"
                    )
                    self.assertEqual(self._merge(fixture), 2)
                    self.assertFalse(fixture["output_path"].exists())

    def test_merge_rejects_duplicate_missing_extra_and_reordered_cases(self) -> None:
        mutations = {
            "duplicate": lambda records: records.__setitem__(
                1, {**records[1], "case_id": records[0]["case_id"]}
            ),
            "missing": lambda records: records.pop(),
            "extra": lambda records: records.append(
                {**records[-1], "case_id": "unknown-extra"}
            ),
            "reordered": lambda records: records.reverse(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._completed_fixture(Path(directory))
                    response = json.loads(
                        fixture["reviewer_b_path"].read_text(encoding="utf-8")
                    )
                    mutate(response["case_reviews"])
                    fixture["reviewer_b_path"].write_text(
                        json.dumps(response, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    self.assertEqual(self._merge(fixture), 2)
                    self.assertFalse(fixture["output_path"].exists())

    def test_merge_requires_json_true_not_integer_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._completed_fixture(Path(directory))
            response = json.loads(
                fixture["reviewer_a_path"].read_text(encoding="utf-8")
            )
            response["case_reviews"][0]["source_verified"] = 1
            fixture["reviewer_a_path"].write_text(
                json.dumps(response, sort_keys=True) + "\n", encoding="utf-8"
            )
            self.assertEqual(self._merge(fixture), 2)
            self.assertFalse(fixture["output_path"].exists())

    def test_merge_requires_confirmation_checks_and_exact_pass(self) -> None:
        mutations = {
            "confirmation false": lambda response: response.__setitem__(
                "independent_review_confirmed", False
            ),
            "confirmation integer": lambda response: response.__setitem__(
                "independent_review_confirmed", 1
            ),
            "check false": lambda response: response["case_reviews"][0].__setitem__(
                "gold_verified", False
            ),
            "lowercase pass": lambda response: response["case_reviews"][0].__setitem__(
                "decision", "pass"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._completed_fixture(Path(directory))
                    response = json.loads(
                        fixture["reviewer_a_path"].read_text(encoding="utf-8")
                    )
                    mutate(response)
                    fixture["reviewer_a_path"].write_text(
                        json.dumps(response, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    self.assertEqual(self._merge(fixture), 2)

    def test_merge_rejects_template_placeholder_and_same_normalized_identity(
        self,
    ) -> None:
        identities = [
            ("__REVIEWER_A_ID__", "reviewer-beta"),
            ("Alice", "Ａｌｉｃｅ"),
        ]
        for reviewer_a, reviewer_b in identities:
            with self.subTest(reviewer_a=reviewer_a, reviewer_b=reviewer_b):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._fixture(Path(directory))
                    self.assertEqual(self._create(fixture), 0)
                    self._complete(fixture["reviewer_a_path"], reviewer_a)
                    self._complete(fixture["reviewer_b_path"], reviewer_b)
                    self.assertEqual(self._merge(fixture), 2)

    def test_merge_rejects_duplicate_json_keys_and_shape_extensions(self) -> None:
        for mutation in ("duplicate-key", "extra-key", "missing-key"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = self._completed_fixture(Path(directory))
                    path = fixture["reviewer_a_path"]
                    if mutation == "duplicate-key":
                        text = path.read_text(encoding="utf-8")
                        text = text.replace(
                            '"reviewer_id": "reviewer-alpha"',
                            '"reviewer_id": "attacker",\n  '
                            '"reviewer_id": "reviewer-alpha"',
                            1,
                        )
                        path.write_text(text, encoding="utf-8")
                    else:
                        value = json.loads(path.read_text(encoding="utf-8"))
                        if mutation == "extra-key":
                            value["approval_override"] = True
                        else:
                            del value["review_packet_sha256"]
                        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                    self.assertEqual(self._merge(fixture), 2)

    def test_create_and_merge_reject_symlink_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            real_cases = fixture["cases_path"]
            linked_cases = root / "config" / "cases-link.jsonl"
            linked_cases.symlink_to(real_cases)
            fixture["cases_path"] = linked_cases
            self.assertEqual(self._create(fixture), 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._completed_fixture(root)
            real_response = fixture["reviewer_a_path"]
            linked_response = root / "reviews" / "linked-a.json"
            linked_response.symlink_to(real_response)
            fixture["reviewer_a_path"] = linked_response
            self.assertEqual(self._merge(fixture), 2)

    def test_merge_rejects_hardlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._completed_fixture(Path(directory))
            fixture["reviewer_b_path"].unlink()
            os.link(fixture["reviewer_a_path"], fixture["reviewer_b_path"])
            with self.assertRaisesRegex(ValueError, "non-alias"):
                signoff_module.merge_responses(
                    cases_path=fixture["cases_path"],
                    review_packet_path=fixture["packet_path"],
                    reviewer_a_path=fixture["reviewer_a_path"],
                    reviewer_b_path=fixture["reviewer_b_path"],
                    output_path=fixture["output_path"],
                    repo_root=fixture["root"],
                )

    def test_merge_rejects_provenance_paths_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._completed_fixture(root)
            declared_repo = root / "declared-repository"
            declared_repo.mkdir()
            with self.assertRaisesRegex(ValueError, "inside the repository"):
                signoff_module.merge_responses(
                    cases_path=fixture["cases_path"],
                    review_packet_path=fixture["packet_path"],
                    reviewer_a_path=fixture["reviewer_a_path"],
                    reviewer_b_path=fixture["reviewer_b_path"],
                    output_path=fixture["output_path"],
                    repo_root=declared_repo,
                )
            self.assertFalse(fixture["output_path"].exists())

    def test_merge_is_no_clobber_and_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._completed_fixture(Path(directory))
            fixture["output_path"].parent.mkdir(parents=True)
            fixture["output_path"].write_bytes(b"preserve signoff")
            self.assertEqual(self._merge(fixture), 2)
            self.assertEqual(
                fixture["output_path"].read_bytes(), b"preserve signoff"
            )

    def test_cases_loader_rejects_duplicate_and_non_string_ids(self) -> None:
        invalid_payloads = (
            b'{"id":"same"}\n{"id":"same"}\n',
            b'{"id":1}\n',
            b'{"id":"x","id":"y"}\n',
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    signoff_module.load_case_ids(payload, source="fixture.jsonl")

    def test_main_parser_requires_exactly_one_mode_and_merge_output(self) -> None:
        with self.assertRaises(SystemExit):
            signoff_module.main(["--review-a", "a", "--review-b", "b"])
        with self.assertRaises(SystemExit):
            signoff_module.main(
                [
                    "--create-templates",
                    "--merge",
                    "--review-a",
                    "a",
                    "--review-b",
                    "b",
                ]
            )
        with self.assertRaises(SystemExit):
            signoff_module.main(
                ["--merge", "--review-a", "a", "--review-b", "b"]
            )

    def test_script_entrypoint_runs_create_and_check_without_external_calls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            common = [
                sys.executable,
                str(ROOT / "scripts" / "build_holdout_signoff.py"),
                "--cases",
                str(fixture["cases_path"]),
                "--review-packet",
                str(fixture["packet_path"]),
                "--review-a",
                str(fixture["reviewer_a_path"]),
                "--review-b",
                str(fixture["reviewer_b_path"]),
            ]
            create = subprocess.run(
                [*common, "--create-templates"],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            check = subprocess.run(
                [*common, "--check-templates"],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(create.returncode, 0, create.stderr)
            self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
