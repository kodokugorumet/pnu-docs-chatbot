import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import venv
from pathlib import Path
from unittest import mock

from scripts.document_parsing.runtime import (
    ArtifactManifestError,
    doctor,
    load_artifact_manifest,
    prepare,
    verify_checksum,
    verify_sha256,
)
from scripts.document_parsing.runtime import manager
from scripts.document_parsing.runtime.capabilities import platform_key
from scripts.document_parsing.pipeline import RuntimeCommands


class ChecksumTests(unittest.TestCase):
    def test_sha256_and_sha512_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = Path(temporary) / "payload.bin"
            payload.write_bytes(b"parser-runtime-test")
            sha256 = hashlib.sha256(payload.read_bytes()).hexdigest()
            sha512 = hashlib.sha512(payload.read_bytes()).hexdigest()

            self.assertTrue(verify_sha256(payload, sha256))
            self.assertTrue(verify_checksum(payload, sha512, "sha512"))
            self.assertFalse(verify_sha256(payload, "0" * 64))

    def test_rejects_malformed_checksum(self):
        with tempfile.NamedTemporaryFile() as payload:
            with self.assertRaises(ValueError):
                verify_sha256(payload.name, "not-a-checksum")


class ManifestTests(unittest.TestCase):
    def test_checked_in_manifest_has_expected_pins_and_profiles(self):
        manifest = load_artifact_manifest()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["pins"]["hwplib"], "1.1.10")
        self.assertEqual(manifest["pins"]["hwpxlib"], "1.0.8")
        self.assertEqual(manifest["pins"]["unhwp"], "0.3.0")
        self.assertEqual(manifest["pins"]["docling"], "2.114.0")
        self.assertEqual(
            manifest["pins"]["docling_models"], "2.114.0-default"
        )
        self.assertEqual(manifest["pins"]["paddleocr"], "3.7.0")
        self.assertEqual(manifest["pins"]["paddlepaddle"], "3.2.0")
        self.assertEqual(
            manifest["pins"]["pp_ocr_detection"],
            "PP-OCRv5_mobile_det",
        )
        self.assertEqual(manifest["pins"]["tika"], "3.3.2")
        self.assertEqual(
            manifest["capabilities"]["docling"]["exact_version"],
            "2.114.0",
        )
        self.assertEqual(
            manifest["capabilities"]["paddleocr"]["exact_version"],
            "3.7.0",
        )
        self.assertEqual(
            manifest["capabilities"]["unhwp"]["exact_version"],
            "0.3.0",
        )
        self.assertEqual(
            set(manifest["profiles"]),
            {"baseline", "challenger", "cascade"},
        )

    def test_invalid_manifest_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profiles": {
                            "baseline": {
                                "required": ["missing"],
                                "optional": [],
                                "artifacts": [],
                            }
                        },
                        "capabilities": {},
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ArtifactManifestError):
                load_artifact_manifest(path)


def _write_test_manifest(
    directory,
    payload,
    archive=False,
    capability_kind="file",
):
    checksum = hashlib.sha256(payload).hexdigest()
    destination = "bin/tool" if archive else "lib/tool.bin"
    variant = {
        "url": "https://example.invalid/tool",
        "checksum": checksum,
        "checksum_algorithm": "sha256",
        "destination": destination,
        "archive": "tar.gz" if archive else "file",
    }
    if archive:
        variant["archive_member"] = "tool"

    manifest = {
        "schema_version": 1,
        "pins": {"tool": "1.0.0"},
        "profiles": {
            "baseline": {
                "required": ["tool"],
                "optional": [],
                "artifacts": ["tool"],
            }
        },
        "capabilities": {
            "tool": {
                "kind": capability_kind,
                "relative_paths": [destination],
                "commands": [],
                "version_args": ["--version"],
            }
        },
        "artifacts": {
            "tool": {
                "version": "1.0.0",
                "license": "Apache-2.0",
                "variants": {
                    "any": variant,
                },
            }
        },
    }
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class RuntimeManagerTests(unittest.TestCase):
    def test_docling_2114_official_model_layout_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = root / "docling-project--docling-layout-heron"
            tableformer = (
                root
                / "docling-project--docling-models"
                / "model_artifacts"
                / "tableformer"
                / "accurate"
            )
            layout.mkdir(parents=True)
            tableformer.mkdir(parents=True)
            (layout / "model.safetensors").write_bytes(b"layout")
            (layout / "config.json").write_text("{}", encoding="utf-8")
            (tableformer / "tableformer_accurate.safetensors").write_bytes(
                b"table"
            )
            (tableformer / "tm_config.json").write_text(
                "{}",
                encoding="utf-8",
            )

            self.assertTrue(RuntimeCommands._complete_docling_models(root))

    def test_docling_model_layout_rejects_missing_official_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tableformer = (
                root
                / "docling-project--docling-models"
                / "model_artifacts"
                / "tableformer"
                / "accurate"
            )
            tableformer.mkdir(parents=True)
            (tableformer / "tableformer_accurate.safetensors").write_bytes(
                b"table"
            )
            (tableformer / "tm_config.json").write_text(
                "{}",
                encoding="utf-8",
            )

            self.assertFalse(RuntimeCommands._complete_docling_models(root))

    def test_doctor_is_json_serializable_and_does_not_create_tools_dir(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "does-not-exist"
            result = doctor("baseline", tools_dir)

            json.dumps(result)
            self.assertFalse(tools_dir.exists())
            self.assertEqual(result["profile"], "baseline")
            self.assertIn("ready", result)
            self.assertIn("missing_required", result)
            self.assertEqual(result["tools_dir"], str(tools_dir.resolve()))

    def test_prepare_defaults_to_inspection_without_writes_or_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            with mock.patch.object(
                manager,
                "_download",
                side_effect=AssertionError("dry prepare must not download"),
            ):
                result = prepare("baseline", tools_dir)

            self.assertFalse(result["execute"])
            self.assertFalse(result["changed"])
            self.assertFalse(tools_dir.exists())
            self.assertTrue(
                all(
                    action["status"] in {"installed", "would_download"}
                    for action in result["actions"]
                )
            )
            json.dumps(result)

    def test_explicit_prepare_downloads_verified_file_inside_tools_dir(self):
        payload = b"verified parser artifact"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_test_manifest(root, payload)
            tools_dir = root / "tools"

            def fake_download(url, destination):
                self.assertEqual(url, "https://example.invalid/tool")
                destination.write_bytes(payload)

            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ), mock.patch.object(manager, "_download", side_effect=fake_download):
                result = prepare("baseline", tools_dir, execute=True)

            installed = tools_dir / "lib" / "tool.bin"
            self.assertEqual(installed.read_bytes(), payload)
            self.assertTrue(result["changed"])
            self.assertTrue(result["ready"])
            self.assertEqual(result["actions"][0]["status"], "installed")
            self.assertTrue(
                (tools_dir / ".receipts" / "tool.json").is_file()
            )

    def test_checksum_failure_does_not_install_download(self):
        expected = b"expected"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_test_manifest(root, expected)
            tools_dir = root / "tools"

            def fake_download(url, destination):
                destination.write_bytes(b"tampered")

            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ), mock.patch.object(manager, "_download", side_effect=fake_download):
                result = prepare("baseline", tools_dir, execute=True)

            self.assertFalse((tools_dir / "lib" / "tool.bin").exists())
            self.assertEqual(result["actions"][0]["status"], "error")
            self.assertIn("checksum mismatch", result["actions"][0]["error"])

    def test_archive_install_extracts_only_declared_member(self):
        executable = b"#!/bin/sh\nprintf 'tool 1.0.0\\n'\n"
        archive_buffer = io.BytesIO()
        with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
            member = tarfile.TarInfo("tool")
            member.size = len(executable)
            archive.addfile(member, io.BytesIO(executable))
            ignored = b"must not escape"
            extra = tarfile.TarInfo("ignored")
            extra.size = len(ignored)
            archive.addfile(extra, io.BytesIO(ignored))
        archive_payload = archive_buffer.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _write_test_manifest(
                root,
                archive_payload,
                archive=True,
                capability_kind="command",
            )
            tools_dir = root / "tools"

            def fake_download(url, destination):
                destination.write_bytes(archive_payload)

            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ), mock.patch.object(manager, "_download", side_effect=fake_download):
                result = prepare("baseline", tools_dir, execute=True)

            installed = tools_dir / "bin" / "tool"
            self.assertTrue(installed.is_file())
            self.assertEqual(installed.read_bytes(), executable)
            self.assertFalse((tools_dir / "bin" / "ignored").exists())
            self.assertEqual(result["actions"][0]["status"], "installed")

            installed.write_bytes(b"tampered executable")
            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ):
                inspection = prepare("baseline", tools_dir)
                diagnosis = doctor("baseline", tools_dir)

            self.assertEqual(
                inspection["actions"][0]["previous_status"],
                "present_unverified",
            )
            self.assertFalse(diagnosis["ready"])
            self.assertIn(
                "integrity check failed",
                diagnosis["capabilities"]["tool"]["detail"],
            )

    def test_hwp_worker_receipt_binds_the_installed_jar_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            tools_dir = Path(temporary) / "tools"
            destination = tools_dir / "java" / "hwp-parser-worker.jar"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"verified worker")
            manager._write_hwp_worker_receipt(
                tools_dir,
                destination,
                "source-hash",
                "1.0.0",
            )

            self.assertEqual(
                manager._hwp_worker_status(
                    tools_dir,
                    destination,
                    "source-hash",
                ),
                "installed",
            )
            destination.write_bytes(b"replaced worker")
            self.assertEqual(
                manager._hwp_worker_status(
                    tools_dir,
                    destination,
                    "source-hash",
                ),
                "checksum_mismatch",
            )

    def test_paddle_command_requires_complete_explicit_local_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            tools_dir = Path(temporary) / "tools"
            repo_root = Path(__file__).resolve().parents[1]
            model_paths = (
                tools_dir
                / "models"
                / "paddle"
                / "PP-StructureV3"
                / "layout_detection",
                tools_dir
                / "models"
                / "paddle"
                / "PP-StructureV3"
                / "text_detection",
                tools_dir
                / "models"
                / "paddle"
                / "korean_PP-OCRv5_mobile_rec",
            )
            for path in model_paths:
                path.mkdir(parents=True)
                (path / "placeholder").write_text(
                    "not a model",
                    encoding="utf-8",
                )
            commands = RuntimeCommands(repo_root, tools_dir)
            with mock.patch.object(
                commands,
                "_python_for",
                return_value=Path(sys.executable),
            ):
                self.assertIsNone(commands.paddle("challenger"))
                for path in model_paths:
                    (path / "inference.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )
                    (path / "inference.pdiparams").write_bytes(b"weights")
                    (path / "inference.yml").write_text(
                        "Global: {}",
                        encoding="utf-8",
                    )
                command = commands.paddle("challenger")

            self.assertIsNotNone(command)
            self.assertIn("--layout-model-dir", command)
            self.assertIn("--text-detection-model-dir", command)
            self.assertIn("PP-OCRv5_mobile_det", command)
            self.assertIn("--recognition-model-dir", command)

            runtime_report = {
                "pins": {
                    "pp_ocr_detection": "stale-detection-pin",
                    "pp_ocr_korean": "stale-recognition-pin",
                },
                "capabilities": {
                    "paddleocr": {
                        "available": True,
                        "location": os.path.abspath(sys.executable),
                        "version": "9.1.0",
                    },
                    "paddlepaddle": {
                        "available": True,
                        "location": os.path.abspath(sys.executable),
                        "version": "9.2.0",
                    },
                    "pp_structure_v3_model": {
                        "available": True,
                        "location": str(model_paths[0]),
                        "version": "layout-from-report",
                    },
                    "pp_ocrv5_detection_model": {
                        "available": True,
                        "location": str(model_paths[1]),
                        "version": "detection-from-report",
                    },
                    "pp_ocrv5_korean_model": {
                        "available": True,
                        "location": str(model_paths[2]),
                        "version": "recognition-from-report",
                    },
                },
            }
            managed = RuntimeCommands(
                repo_root,
                tools_dir,
                runtime_report,
            )
            managed_command = managed.paddle("challenger")

            self.assertIsNotNone(managed_command)
            self.assertIn("detection-from-report", managed_command)
            self.assertIn("recognition-from-report", managed_command)
            self.assertNotIn("stale-detection-pin", managed_command)

    def test_doctor_can_probe_python_import_with_current_interpreter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema_version": 1,
                "pins": {},
                "profiles": {
                    "baseline": {
                        "required": ["stdlib_json"],
                        "optional": [],
                        "artifacts": [],
                    }
                },
                "capabilities": {
                    "stdlib_json": {
                        "kind": "python_import",
                        "module": "json",
                        "distribution": "",
                        "interpreter_relative_paths": [],
                        "interpreter_commands": [],
                        "allow_current_interpreter": True,
                    }
                },
                "artifacts": {},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ):
                result = doctor("baseline", root / "tools")

            self.assertTrue(result["ready"])
            self.assertEqual(
                result["capabilities"]["stdlib_json"]["location"],
                os.path.abspath(sys.executable),
            )
            self.assertEqual(
                result["capabilities"]["stdlib_json"]["environment_id"],
                os.path.abspath(sys.prefix),
            )

    def test_doctor_preserves_a_real_venv_launcher_and_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            environment = tools_dir / "venvs" / "core"
            venv.EnvBuilder(with_pip=False).create(str(environment))
            interpreter = environment / "bin" / "python"
            python_family = "{}.{}".format(
                sys.version_info.major,
                sys.version_info.minor,
            )
            manifest = {
                "schema_version": 1,
                "pins": {"python": python_family},
                "profiles": {
                    "baseline": {
                        "required": ["core_python"],
                        "optional": [],
                        "artifacts": [],
                    }
                },
                "capabilities": {
                    "core_python": {
                        "kind": "python_import",
                        "module": "json",
                        "distribution": "",
                        "interpreter_relative_paths": [
                            "venvs/core/bin/python"
                        ],
                        "interpreter_commands": [],
                        "runtime_group": "core",
                    }
                },
                "artifacts": {},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ):
                result = doctor("baseline", tools_dir)

            self.assertTrue(result["ready"])
            expected_environment = (
                tools_dir.resolve() / "venvs" / "core"
            )
            self.assertEqual(
                result["capabilities"]["core_python"]["location"],
                str(expected_environment / "bin" / "python"),
            )
            self.assertEqual(
                result["capabilities"]["core_python"]["environment_id"],
                str(expected_environment),
            )

    def test_doctor_skips_wrong_command_version_and_uses_exact_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            bin_dir = tools_dir / "bin"
            bin_dir.mkdir(parents=True)
            prerelease = bin_dir / "prerelease-tool"
            old = bin_dir / "old-tool"
            exact = bin_dir / "exact-tool"
            prerelease.write_text(
                "#!/bin/sh\necho 'tool 1.2.0-rc1'\n",
                encoding="utf-8",
            )
            old.write_text(
                "#!/bin/sh\necho 'tool 1.1.0'\n",
                encoding="utf-8",
            )
            exact.write_text(
                "#!/bin/sh\necho 'tool 1.2.0'\n",
                encoding="utf-8",
            )
            prerelease.chmod(0o755)
            old.chmod(0o755)
            exact.chmod(0o755)
            manifest = {
                "schema_version": 1,
                "pins": {},
                "profiles": {
                    "baseline": {
                        "required": ["tool"],
                        "optional": [],
                        "artifacts": [],
                    }
                },
                "capabilities": {
                    "tool": {
                        "kind": "command",
                        "relative_paths": [
                            "bin/prerelease-tool",
                            "bin/old-tool",
                            "bin/exact-tool",
                        ],
                        "commands": [],
                        "version_args": ["--version"],
                        "exact_version": "1.2.0",
                    }
                },
                "artifacts": {},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ):
                result = doctor("baseline", tools_dir)

            self.assertTrue(result["ready"])
            self.assertEqual(
                result["capabilities"]["tool"]["location"],
                str(tools_dir.resolve() / "bin" / "exact-tool"),
            )
            self.assertEqual(
                result["capabilities"]["tool"]["version"],
                "1.2.0",
            )

    def test_doctor_rejects_python_runtime_group_split_across_venvs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            environment = tools_dir / "venvs" / "group"
            venv.EnvBuilder(with_pip=False).create(str(environment))
            python_family = "{}.{}".format(
                sys.version_info.major,
                sys.version_info.minor,
            )
            manifest = {
                "schema_version": 1,
                "pins": {"python": python_family},
                "profiles": {
                    "baseline": {
                        "required": ["first", "second"],
                        "optional": [],
                        "artifacts": [],
                    }
                },
                "capabilities": {
                    "first": {
                        "kind": "python_import",
                        "module": "json",
                        "distribution": "",
                        "interpreter_relative_paths": [
                            "venvs/group/bin/python"
                        ],
                        "interpreter_commands": [],
                        "runtime_group": "test",
                    },
                    "second": {
                        "kind": "python_import",
                        "module": "json",
                        "distribution": "",
                        "interpreter_relative_paths": [],
                        "interpreter_commands": [],
                        "allow_current_interpreter": True,
                        "runtime_group": "test",
                    },
                },
                "artifacts": {},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ):
                result = doctor("baseline", tools_dir)

            self.assertFalse(result["ready"])
            self.assertIn("second", result["missing_required"])
            self.assertIn(
                "different interpreters",
                result["capabilities"]["second"]["detail"],
            )

    def test_doctor_requires_tessdata_languages_in_one_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools_dir = root / "tools"
            korean = tools_dir / "first" / "kor.traineddata"
            english = tools_dir / "second" / "eng.traineddata"
            korean.parent.mkdir(parents=True)
            english.parent.mkdir(parents=True)
            korean.write_bytes(b"korean")
            english.write_bytes(b"english")
            manifest = {
                "schema_version": 1,
                "pins": {},
                "profiles": {
                    "baseline": {
                        "required": ["kor", "eng"],
                        "optional": [],
                        "artifacts": [],
                    }
                },
                "capabilities": {
                    "kor": {
                        "kind": "tessdata",
                        "filename": "kor.traineddata",
                        "relative_paths": ["first/kor.traineddata"],
                        "shared_parent_group": "languages",
                    },
                    "eng": {
                        "kind": "tessdata",
                        "filename": "eng.traineddata",
                        "relative_paths": ["second/eng.traineddata"],
                        "shared_parent_group": "languages",
                    },
                },
                "artifacts": {},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"PARSER_ARTIFACT_MANIFEST": str(manifest_path)},
            ):
                result = doctor("baseline", tools_dir)

            self.assertFalse(result["ready"])
            self.assertIn("eng", result["missing_required"])
            self.assertIn(
                "must share one directory",
                result["capabilities"]["eng"]["detail"],
            )

    def test_unknown_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                doctor("unknown", temporary)

    def test_platform_key_matches_supported_shape(self):
        self.assertIn("-", platform_key())


if __name__ == "__main__":
    unittest.main()
