from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "immutable_outputs_for_test", ROOT / "scripts" / "immutable_outputs.py"
)
assert SPEC is not None and SPEC.loader is not None
immutable = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(immutable)


class ImmutableOutputsTests(unittest.TestCase):
    def test_exclusive_run_lock_rejects_concurrent_or_stale_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "run" / ".final-run.lock"
            owner = {"kind": "test", "schedule_id": "schedule-1"}

            with immutable.exclusive_run_lock(lock_path, owner):
                self.assertTrue(lock_path.is_file())
                stored = lock_path.read_text(encoding="utf-8")
                self.assertIn('"schedule_id":"schedule-1"', stored)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    with immutable.exclusive_run_lock(lock_path, owner):
                        self.fail("a second owner must never acquire the lock")

            self.assertFalse(os.path.lexists(lock_path))

    def test_publish_is_no_clobber_and_preserves_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("original\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already exists"):
                immutable.publish_immutable_texts(
                    {output: "replacement\n"}, authoritative_path=output
                )

            self.assertEqual(output.read_text(encoding="utf-8"), "original\n")

    def test_existing_broken_symlink_is_not_treated_as_a_free_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            output.symlink_to(root / "missing-target.json")

            with self.assertRaisesRegex(ValueError, "already exists"):
                immutable.require_new_outputs([output])

            self.assertTrue(output.is_symlink())

    def test_authority_is_published_last_and_failure_removes_companion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            companion = root / "report.csv"
            authority = root / "report.json"
            real_fsync_directory = immutable._fsync_directory
            calls = 0

            def fail_after_authority_link(path: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected directory fsync failure")
                real_fsync_directory(path)

            with mock.patch.object(
                immutable, "_fsync_directory", side_effect=fail_after_authority_link
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    immutable.publish_immutable_texts(
                        {authority: "json\n", companion: "csv\n"},
                        authoritative_path=authority,
                    )

            self.assertFalse(authority.exists())
            self.assertFalse(companion.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_link_race_cannot_replace_competing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            companion = root / "report.csv"
            authority = root / "report.json"
            real_link = os.link
            calls = 0

            def race_on_authority(source: Path, target: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    Path(target).write_text("competitor\n", encoding="utf-8")
                real_link(source, target)

            with mock.patch.object(immutable.os, "link", side_effect=race_on_authority):
                with self.assertRaises(FileExistsError):
                    immutable.publish_immutable_texts(
                        {companion: "csv\n", authority: "json\n"},
                        authoritative_path=authority,
                    )

            self.assertEqual(authority.read_text(encoding="utf-8"), "competitor\n")
            self.assertFalse(companion.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_success_publishes_all_files_with_authority_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            companion = root / "report.csv"
            authority = root / "report.json"

            immutable.publish_immutable_texts(
                {authority: "json\n", companion: "csv\n"},
                authoritative_path=authority,
            )

            self.assertEqual(companion.read_text(encoding="utf-8"), "csv\n")
            self.assertEqual(authority.read_text(encoding="utf-8"), "json\n")
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
