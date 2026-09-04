from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_holdout_review_packet",
    ROOT / "scripts" / "build_holdout_review_packet.py",
)
assert SPEC is not None and SPEC.loader is not None
packet_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packet_module)


class BuildHoldoutReviewPacketTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
        source = root / "downloads" / "원문 자료.pdf"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"frozen source")
        cases = [
            {
                "id": "core-1",
                "split": "holdout-core",
                "family_id": "family-1",
                "category": "registration",
                "difficulty_type": "single_fact",
                "role": "pnu-student",
                "answerable": True,
                "expected_behavior": "answer",
                "authoring_status": "draft_unreviewed",
                "query": "신청 기간은 언제인가요?",
                "required_claims": [
                    {
                        "claim_id": "period",
                        "description": "신청 기간을 답한다.",
                        "critical_values": ["2026.09.01", "2026.09.02"],
                        "semantic_anchors": ["신청 기간"],
                        "evidence_options": [
                            {
                                "document_id": "doc-1",
                                "source_path": "downloads/원문 자료.pdf",
                                "source_title": "원문 자료",
                                "source_sha256": "a" * 64,
                                "source_url": "https://example.edu/notice/1",
                                "quote": "신청 기간은 2026.09.01~2026.09.02입니다.",
                            }
                        ],
                    }
                ],
                "optional_claims": [],
                "forbidden_claims": [
                    {"claim_id": "wrong_date", "description": "다른 날짜를 단정한다."}
                ],
            },
            {
                "id": "challenge-1",
                "split": "holdout-challenge",
                "family_id": "challenge-family-1",
                "category": "challenge",
                "challenge_type": "unanswerable",
                "role": "pnu-student",
                "answerable": False,
                "expected_behavior": "abstain",
                "authoring_status": "draft_unreviewed",
                "query": "제 개인 결과를 알려줘.",
                "required_claims": [],
                "optional_claims": [],
                "forbidden_claims": [
                    {"claim_id": "invent_result", "description": "개인 결과를 만든다."}
                ],
                "challenge_oracle": {
                    "must_do": ["확인할 수 없다고 답한다."],
                    "must_not_do": ["결과를 추측하지 않는다."],
                },
            },
        ]
        cases_path = root / "config" / "cases.jsonl"
        cases_path.parent.mkdir(parents=True)
        cases_path.write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
            encoding="utf-8",
        )
        return cases_path, source, cases

    def test_packet_contains_sha_evidence_oracle_and_only_empty_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path, _, cases = self._fixture(root)
            output = root / "docs" / "review.md"
            payload = cases_path.read_bytes()
            rendered = packet_module.render_packet(
                cases,
                cases_sha256=packet_module.sha256_bytes(payload),
                cases_path=cases_path,
                repo_root=root,
                output_path=output,
            )

        self.assertIn(packet_module.sha256_bytes(payload), rendered)
        self.assertIn("source_verified", rendered)
        self.assertIn("gold_verified", rendered)
        self.assertIn("answerability_verified", rendered)
        self.assertIn("label_verified", rendered)
        self.assertIn("[downloads/원문 자료.pdf](<../downloads/원문 자료.pdf>)", rendered)
        self.assertIn("신청 기간은 2026.09.01~2026.09.02입니다.", rendered)
        self.assertIn("`2026.09.01`", rendered)
        self.assertIn("반드시 해야 함 (must_do)", rendered)
        self.assertIn("확인할 수 없다고 답한다.", rendered)
        self.assertEqual(rendered.count("#### Reviewer A"), 2)
        self.assertEqual(rendered.count("#### Reviewer B"), 2)
        self.assertNotIn("[x]", rendered.casefold())
        self.assertNotIn('"case_signoffs"', rendered)

    def test_render_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path, _, cases = self._fixture(root)
            output = root / "docs" / "review.md"
            payload = cases_path.read_bytes()
            kwargs = {
                "cases_sha256": packet_module.sha256_bytes(payload),
                "cases_path": cases_path,
                "repo_root": root,
                "output_path": output,
            }
            first = packet_module.render_packet(cases, **kwargs)
            second = packet_module.render_packet(cases, **kwargs)

        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

    def test_check_mode_detects_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path, _, _ = self._fixture(root)
            output = root / "docs" / "review.md"
            old_root = packet_module.REPO_ROOT
            try:
                packet_module.REPO_ROOT = root
                self.assertEqual(
                    packet_module.main(
                        ["--cases", str(cases_path), "--output", str(output)]
                    ),
                    0,
                )
                self.assertEqual(
                    packet_module.main(
                        [
                            "--cases",
                            str(cases_path),
                            "--output",
                            str(output),
                            "--check",
                        ]
                    ),
                    0,
                )
                output.write_text("stale\n", encoding="utf-8")
                self.assertEqual(
                    packet_module.main(
                        [
                            "--cases",
                            str(cases_path),
                            "--output",
                            str(output),
                            "--check",
                        ]
                    ),
                    1,
                )
            finally:
                packet_module.REPO_ROOT = old_root

    def test_create_is_no_clobber_and_preserves_existing_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path, _, _ = self._fixture(root)
            output = root / "docs" / "review.md"
            old_root = packet_module.REPO_ROOT
            try:
                packet_module.REPO_ROOT = root
                self.assertEqual(
                    packet_module.main(
                        ["--cases", str(cases_path), "--output", str(output)]
                    ),
                    0,
                )
                original = output.read_bytes()
                self.assertEqual(
                    packet_module.main(
                        ["--cases", str(cases_path), "--output", str(output)]
                    ),
                    2,
                )
                self.assertEqual(output.read_bytes(), original)
            finally:
                packet_module.REPO_ROOT = old_root

    def test_loader_rejects_duplicate_case_ids(self) -> None:
        payload = b'{"id":"same"}\n{"id":"same"}\n'
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            packet_module.load_jsonl_bytes(payload, source="fixture.jsonl")


if __name__ == "__main__":
    unittest.main()
