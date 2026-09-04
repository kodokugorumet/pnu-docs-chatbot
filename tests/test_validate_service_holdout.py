from __future__ import annotations

import copy
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from tests import test_holdout_gold as holdout_fixture


validator_cli = holdout_fixture.validator_cli


class ServiceHoldoutValidatorTests(unittest.TestCase):
    def valid_cases(self) -> list[dict]:
        return copy.deepcopy(holdout_fixture.HoldoutGoldTests().valid_cases())

    @staticmethod
    def reviewer(reviewer_id: str) -> dict:
        return {
            "reviewer_id": reviewer_id,
            "source_verified": True,
            "gold_verified": True,
            "answerability_verified": True,
            "label_verified": True,
        }

    def write_cases(self, path: Path, cases: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(case, ensure_ascii=False) + "\n" for case in cases
            ),
            encoding="utf-8",
        )

    def write_signoff(
        self,
        path: Path,
        cases: list[dict],
        *,
        reviewer_ids: tuple[str, ...] = ("reviewer-a", "reviewer-b"),
    ) -> list[dict]:
        cases_bytes = "".join(
            json.dumps(case, ensure_ascii=False) + "\n" for case in cases
        ).encode("utf-8")
        cases_sha256 = hashlib.sha256(cases_bytes).hexdigest()
        packet_path = path.parent / "review-packet.md"
        packet_path.write_text("# Frozen review packet\n", encoding="utf-8")
        packet_sha256 = hashlib.sha256(packet_path.read_bytes()).hexdigest()

        reviewers_by_slot: list[tuple[str, str, list[dict]]] = []
        response_provenance: list[dict] = []
        for reviewer_index, reviewer_id in enumerate(reviewer_ids):
            slot = chr(ord("A") + reviewer_index)
            reviews = [
                {
                    "case_id": case["id"],
                    "source_verified": True,
                    "gold_verified": True,
                    "answerability_verified": True,
                    "label_verified": True,
                    "decision": "PASS",
                    "notes": "",
                }
                for case in cases
            ]
            response = {
                "schema_version": (
                    validator_cli.signoff_artifacts.RESPONSE_SCHEMA_VERSION
                ),
                "reviewer_slot": slot,
                "cases_sha256": cases_sha256,
                "review_packet_sha256": packet_sha256,
                "reviewer_id": reviewer_id,
                "independent_review_confirmed": True,
                "case_reviews": reviews,
            }
            response_path = path.parent / f"review-{slot.lower()}.json"
            response_path.write_text(
                json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            response_provenance.append(
                {
                    "reviewer_slot": slot,
                    "reviewer_id": reviewer_id,
                    "sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                    "path": response_path.name,
                }
            )
            reviewers_by_slot.append((slot, reviewer_id, reviews))

        records = [
            {
                "case_id": case["id"],
                "reviewers": [
                    {
                        "reviewer_slot": slot,
                        "reviewer_id": reviewer_id,
                        "source_verified": reviews[index]["source_verified"],
                        "gold_verified": reviews[index]["gold_verified"],
                        "answerability_verified": reviews[index][
                            "answerability_verified"
                        ],
                        "label_verified": reviews[index]["label_verified"],
                        "decision": reviews[index]["decision"],
                        "notes": reviews[index]["notes"],
                    }
                    for slot, reviewer_id, reviews in reviewers_by_slot
                ],
            }
            for index, case in enumerate(cases)
        ]
        path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        validator_cli.signoff_artifacts.SIGNOFF_SCHEMA_VERSION
                    ),
                    "cases_sha256": cases_sha256,
                    "review_packet_path": packet_path.name,
                    "review_packet_sha256": packet_sha256,
                    "review_response_sha256s": response_provenance,
                    "case_signoffs": records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return records

    def build_fixture(self, directory: Path) -> dict:
        cases = self.valid_cases()
        raw_directory = directory / "raw"
        raw_directory.mkdir()

        documents: dict[str, dict] = {}
        for case in cases:
            claims = case.get("required_claims", [])
            for claim in claims:
                for option in claim.get("evidence_options", []):
                    document = documents.setdefault(
                        option["document_id"],
                        {
                            "source_title": option["source_title"],
                            "source_url": option["source_url"],
                            "quotes": [],
                            "options": [],
                        },
                    )
                    document["quotes"].append(option["quote"])
                    document["options"].append(option)

        index_path = directory / "corpus.sqlite"
        with sqlite3.connect(index_path) as connection:
            connection.execute(
                """
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_index INTEGER NOT NULL,
                    document_id TEXT NOT NULL,
                    source_path TEXT,
                    source_title TEXT,
                    source_url TEXT,
                    text TEXT NOT NULL
                )
                """
            )
            for sequence, (document_id, document) in enumerate(
                sorted(documents.items())
            ):
                source_path = f"raw/{document_id}.txt"
                source_bytes = (
                    "\n".join(document["quotes"]) + "\n"
                ).encode("utf-8")
                (directory / source_path).write_bytes(source_bytes)
                source_sha = hashlib.sha256(source_bytes).hexdigest()
                for option in document["options"]:
                    option["source_path"] = source_path
                    option["source_sha256"] = source_sha
                connection.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, chunk_index, document_id, source_path,
                        source_title, source_url, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"chunk-{sequence}",
                        0,
                        document_id,
                        source_path,
                        document["source_title"],
                        document["source_url"],
                        "\n".join(document["quotes"]),
                    ),
                )

        for case in cases:
            options = [
                option
                for claim in case.get("required_claims", [])
                for option in claim.get("evidence_options", [])
            ]
            case["source_sha256s"] = sorted(
                {option["source_sha256"] for option in options}
            )

        cases_path = directory / "holdout.jsonl"
        self.write_cases(cases_path, cases)
        dev_path = directory / "dev-manifest.json"
        dev_path.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "dev-document",
                            "source_document_family_id": "dev-family",
                            "source_sha256": hashlib.sha256(b"dev").hexdigest(),
                            "source_title": "2025 DEV 전용 공지",
                            "normalized_title_without_year": "dev 전용 공지",
                            "source_url": "https://example.edu/dev-only",
                            "source_path": "raw/dev-only.txt",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        signoff_path = directory / "signoff.json"
        self.write_signoff(signoff_path, cases)
        return {
            "cases": cases,
            "cases_path": cases_path,
            "dev_path": dev_path,
            "index_path": index_path,
            "signoff_path": signoff_path,
            "repo_root": directory,
        }

    def validate(self, fixture: dict) -> dict:
        return validator_cli.validate_paths(
            fixture["cases_path"],
            [fixture["dev_path"]],
            corpus_index=fixture["index_path"],
            signoff_path=fixture["signoff_path"],
            repo_root=fixture["repo_root"],
        )

    def test_cli_requires_dev_manifest_and_one_review_mode(self) -> None:
        parser = validator_cli.build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--signoff", "signoff.json"])
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--dev-manifest", "dev.json"])
        args = parser.parse_args(
            ["--dev-manifest", "dev.json", "--pre-review"]
        )
        self.assertTrue(args.pre_review)

    def test_pre_review_passes_only_the_three_machine_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            summary = validator_cli.validate_paths(
                fixture["cases_path"],
                [fixture["dev_path"]],
                corpus_index=fixture["index_path"],
                repo_root=fixture["repo_root"],
                pre_review=True,
            )

        self.assertTrue(summary["ok"])
        self.assertEqual(set(summary["gates"]), {"schema", "dev", "corpus"})
        self.assertEqual(
            summary["gates"]["corpus"]["verified_evidence_option_count"], 57
        )

    def test_all_four_gates_pass_with_verified_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            summary = self.validate(fixture)

        self.assertTrue(summary["ok"])
        self.assertEqual(
            set(summary["gates"]), {"schema", "dev", "corpus", "signoff"}
        )
        self.assertTrue(all(gate["ok"] for gate in summary["gates"].values()))
        self.assertEqual(
            summary["gates"]["corpus"]["verified_evidence_option_count"], 57
        )
        self.assertEqual(summary["gates"]["signoff"]["approved_case_count"], 36)

    def test_dev_gate_fails_closed_when_sha_title_and_url_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            fixture["dev_path"].write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "dev-document",
                                "source_document_family_id": "dev-family",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["ok"])
        self.assertTrue(summary["gates"]["schema"]["ok"])
        self.assertFalse(summary["gates"]["dev"]["ok"])
        self.assertTrue(summary["gates"]["corpus"]["ok"])
        self.assertTrue(summary["gates"]["signoff"]["ok"])
        errors = " ".join(summary["gates"]["dev"]["errors"])
        self.assertIn("source SHA-256", errors)
        self.assertIn("normalized source titles", errors)
        self.assertIn("source URLs", errors)

    def test_dev_sha_leak_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            case = fixture["cases"][0]
            options = [
                option
                for claim in case["required_claims"]
                for option in claim["evidence_options"]
            ]
            original_sha = options[0]["source_sha256"]
            for option in options:
                option["source_sha256"] = original_sha.upper()
            case["source_sha256s"] = [original_sha.upper()]
            self.write_cases(fixture["cases_path"], fixture["cases"])
            fixture["dev_path"].write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                **options[0],
                                "source_document_family_id": "mislabeled-dev-family",
                                "source_sha256": original_sha,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["dev"]["ok"])
        self.assertTrue(
            any(
                "DEV source SHA-256 leak" in error
                for error in summary["gates"]["dev"]["errors"]
            )
        )

    def test_corpus_gate_rejects_fake_raw_source_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            case = fixture["cases"][0]
            option = case["required_claims"][0]["evidence_options"][0]
            option["source_sha256"] = "0" * 64
            case["source_sha256s"] = sorted(
                {
                    item["source_sha256"]
                    for claim in case["required_claims"]
                    for item in claim["evidence_options"]
                }
            )
            self.write_cases(fixture["cases_path"], fixture["cases"])
            summary = self.validate(fixture)

        self.assertTrue(summary["gates"]["schema"]["ok"])
        self.assertFalse(summary["gates"]["corpus"]["ok"])
        self.assertTrue(
            any(
                "source_sha256 mismatch" in error
                for error in summary["gates"]["corpus"]["errors"]
            )
        )

    def test_corpus_gate_rejects_quote_not_present_in_indexed_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            option = fixture["cases"][0]["required_claims"][0][
                "evidence_options"
            ][0]
            option["quote"] = "값1-1과 관련된 인덱스에 없는 완전히 다른 문장입니다."
            self.write_cases(fixture["cases_path"], fixture["cases"])
            summary = self.validate(fixture)

        self.assertTrue(summary["gates"]["schema"]["ok"])
        self.assertFalse(summary["gates"]["corpus"]["ok"])
        self.assertTrue(
            any(
                "quote/claim does not match" in error
                for error in summary["gates"]["corpus"]["errors"]
            )
        )

    def test_corpus_gate_rejects_index_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            option = fixture["cases"][0]["required_claims"][0][
                "evidence_options"
            ][0]
            option["source_url"] = "https://example.edu/not-the-index-url"
            self.write_cases(fixture["cases_path"], fixture["cases"])
            summary = self.validate(fixture)

        self.assertTrue(summary["gates"]["schema"]["ok"])
        self.assertFalse(summary["gates"]["corpus"]["ok"])
        self.assertTrue(
            any(
                "source_url mismatch" in error
                for error in summary["gates"]["corpus"]["errors"]
            )
        )

    def test_signoff_requires_exactly_two_distinct_complete_reviewers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            self.write_signoff(
                fixture["signoff_path"],
                fixture["cases"],
                reviewer_ids=("reviewer-a",),
            )
            one_reviewer = self.validate(fixture)
            self.write_signoff(
                fixture["signoff_path"],
                fixture["cases"],
                reviewer_ids=("reviewer-a", "reviewer-b", "reviewer-c"),
            )
            three_reviewers = self.validate(fixture)
            self.write_signoff(fixture["signoff_path"], fixture["cases"])
            valid = self.validate(fixture)

        self.assertFalse(one_reviewer["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "exactly 2 distinct reviewers" in error
                for error in one_reviewer["gates"]["signoff"]["errors"]
            )
        )
        self.assertFalse(three_reviewers["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "exactly 2 reviewer records" in error
                for error in three_reviewers["gates"]["signoff"]["errors"]
            )
        )
        self.assertTrue(valid["gates"]["signoff"]["ok"])
        self.assertTrue(valid["ok"])
        self.assertEqual(
            valid["gates"]["signoff"]["reviewer_ids"],
            ["reviewer-a", "reviewer-b"],
        )

    def test_signoff_requires_same_two_reviewers_for_every_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            payload = json.loads(
                fixture["signoff_path"].read_text(encoding="utf-8")
            )
            payload["case_signoffs"][-1]["reviewers"] = [
                self.reviewer("reviewer-a"),
                self.reviewer("reviewer-c"),
            ]
            fixture["signoff_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["signoff"]["ok"])
        self.assertEqual(summary["gates"]["signoff"]["approved_case_count"], 35)
        self.assertTrue(
            any(
                "same 2-person reviewer roster" in error
                for error in summary["gates"]["signoff"]["errors"]
            )
        )

    def test_signoff_revalidates_bound_packet_and_response_bytes(self) -> None:
        for artifact_name, expected_error in (
            ("review-packet.md", "review_packet_sha256"),
            ("review-a.json", "does not match response bytes"),
        ):
            with self.subTest(artifact_name=artifact_name):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = self.build_fixture(Path(temporary))
                    artifact = fixture["signoff_path"].parent / artifact_name
                    artifact.write_bytes(artifact.read_bytes() + b" \n")
                    summary = self.validate(fixture)

                self.assertFalse(summary["gates"]["signoff"]["ok"])
                errors = " ".join(summary["gates"]["signoff"]["errors"])
                self.assertIn(expected_error, errors)

    def test_signoff_rejects_review_packet_aliasing_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            payload = json.loads(
                fixture["signoff_path"].read_text(encoding="utf-8")
            )
            cases_sha256 = hashlib.sha256(
                fixture["cases_path"].read_bytes()
            ).hexdigest()
            payload["review_packet_path"] = fixture["cases_path"].name
            payload["review_packet_sha256"] = cases_sha256
            for provenance in payload["review_response_sha256s"]:
                response_path = fixture["repo_root"] / provenance["path"]
                response = json.loads(response_path.read_text(encoding="utf-8"))
                response["review_packet_sha256"] = cases_sha256
                response_path.write_text(
                    json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                provenance["sha256"] = hashlib.sha256(
                    response_path.read_bytes()
                ).hexdigest()
            fixture["signoff_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "review_packet_path aliases cases" in error
                for error in summary["gates"]["signoff"]["errors"]
            )
        )

    def test_signoff_rejects_empty_review_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            payload = json.loads(
                fixture["signoff_path"].read_text(encoding="utf-8")
            )
            packet_path = fixture["repo_root"] / payload["review_packet_path"]
            packet_path.write_bytes(b"")
            packet_sha256 = hashlib.sha256(b"").hexdigest()
            payload["review_packet_sha256"] = packet_sha256
            for provenance in payload["review_response_sha256s"]:
                response_path = fixture["repo_root"] / provenance["path"]
                response = json.loads(response_path.read_text(encoding="utf-8"))
                response["review_packet_sha256"] = packet_sha256
                response_path.write_text(
                    json.dumps(response, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                provenance["sha256"] = hashlib.sha256(
                    response_path.read_bytes()
                ).hexdigest()
            fixture["signoff_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["signoff"]["ok"])
        self.assertIn(
            "review packet must not be empty",
            " ".join(summary["gates"]["signoff"]["errors"]),
        )

    def test_repo_artifact_reader_rejects_parent_symlink_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            safe_directory = root / "reviews"
            safe_directory.mkdir(parents=True)
            (safe_directory / "response.json").write_text(
                '{"safe":true}\n', encoding="utf-8"
            )
            outside = Path(temporary) / "outside"
            outside.mkdir()
            (outside / "response.json").write_text(
                '{"external":true}\n', encoding="utf-8"
            )
            original_resolver = validator_cli._resolve_review_artifact_path
            swapped = False

            def swap_parent_after_resolution(*args, **kwargs):
                nonlocal swapped
                resolved = original_resolver(*args, **kwargs)
                if not swapped:
                    swapped = True
                    safe_directory.rename(root / "reviews-original")
                    safe_directory.symlink_to(outside, target_is_directory=True)
                return resolved

            with mock.patch.object(
                validator_cli,
                "_resolve_review_artifact_path",
                side_effect=swap_parent_after_resolution,
            ):
                with self.assertRaisesRegex(
                    ValueError, "cannot securely open|symlink"
                ):
                    validator_cli._read_repo_review_artifact(
                        root,
                        "reviews/response.json",
                        label="review response",
                    )

    def test_signoff_case_rows_must_match_bound_independent_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            payload = json.loads(
                fixture["signoff_path"].read_text(encoding="utf-8")
            )
            payload["case_signoffs"][0]["reviewers"][0]["notes"] = "tampered"
            fixture["signoff_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "does not exactly match the bound Reviewer A response" in error
                for error in summary["gates"]["signoff"]["errors"]
            )
        )

    def test_signoff_rejects_non_exact_case_id_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            payload = json.loads(
                fixture["signoff_path"].read_text(encoding="utf-8")
            )
            payload["case_signoffs"][0]["case_id"] += " "
            fixture["signoff_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "must be a non-empty, trimmed string" in error
                for error in summary["gates"]["signoff"]["errors"]
            )
        )

    def test_signoff_strict_json_rejects_duplicate_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            current = fixture["signoff_path"].read_text(encoding="utf-8")
            duplicate = current.replace(
                '"cases_sha256":',
                f'"cases_sha256":"{"0" * 64}","cases_sha256":',
                1,
            )
            fixture["signoff_path"].write_text(duplicate, encoding="utf-8")
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "duplicate JSON object key" in error
                for error in summary["gates"]["signoff"]["errors"]
            )
        )

    def test_cases_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            lines = fixture["cases_path"].read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace("{", '{"id":"shadow-id",', 1)
            fixture["cases_path"].write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                self.validate(fixture)

    def test_cases_change_during_validation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            original_sha256 = hashlib.sha256(
                fixture["cases_path"].read_bytes()
            ).hexdigest()
            original_dev_gate = validator_cli._validate_dev_gate

            def mutate_after_snapshot(*args, **kwargs):
                result = original_dev_gate(*args, **kwargs)
                fixture["cases_path"].write_bytes(
                    fixture["cases_path"].read_bytes() + b"\n"
                )
                return result

            with mock.patch.object(
                validator_cli,
                "_validate_dev_gate",
                side_effect=mutate_after_snapshot,
            ):
                summary = self.validate(fixture)

        self.assertFalse(summary["ok"])
        self.assertEqual(summary["cases_sha256"], original_sha256)
        self.assertIn(
            "cases bytes changed during validation",
            summary["gates"]["schema"]["errors"],
        )

    def test_signoff_requires_each_case_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            records = self.write_signoff(
                fixture["signoff_path"], fixture["cases"]
            )
            missing_id = records[-1]["case_id"]
            records = records[:-1] + [copy.deepcopy(records[0])]
            fixture["signoff_path"].write_text(
                json.dumps(
                    {
                        "cases_sha256": hashlib.sha256(
                            fixture["cases_path"].read_bytes()
                        ).hexdigest(),
                        "case_signoffs": records,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            summary = self.validate(fixture)

        errors = " ".join(summary["gates"]["signoff"]["errors"])
        self.assertFalse(summary["gates"]["signoff"]["ok"])
        self.assertIn("must occur exactly once", errors)
        self.assertIn(f"missing case {missing_id!r}", errors)

    def test_signoff_rejects_bare_json_list_without_sha_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            payload = json.loads(
                fixture["signoff_path"].read_text(encoding="utf-8")
            )
            fixture["signoff_path"].write_text(
                json.dumps(payload["case_signoffs"], ensure_ascii=False),
                encoding="utf-8",
            )
            summary = self.validate(fixture)

        self.assertFalse(summary["gates"]["signoff"]["ok"])
        errors = " ".join(summary["gates"]["signoff"]["errors"])
        self.assertIn("JSON object", errors)

    def test_signoff_must_bind_exact_cases_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            payload = json.loads(fixture["signoff_path"].read_text(encoding="utf-8"))
            payload.pop("cases_sha256")
            fixture["signoff_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            missing = self.validate(fixture)

            payload["cases_sha256"] = "0" * 64
            fixture["signoff_path"].write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            mismatch = self.validate(fixture)

        self.assertFalse(missing["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "must bind approvals" in error
                for error in missing["gates"]["signoff"]["errors"]
            )
        )
        self.assertFalse(mismatch["gates"]["signoff"]["ok"])
        self.assertTrue(
            any(
                "cases_sha256 mismatch" in error
                for error in mismatch["gates"]["signoff"]["errors"]
            )
        )


if __name__ == "__main__":
    unittest.main()
