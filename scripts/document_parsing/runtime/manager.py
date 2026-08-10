"""Safe parser runtime preparation and diagnostics."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

from .artifacts import (
    load_artifact_manifest,
    verify_checksum,
)
from .capabilities import platform_key, probe_capability


PathLike = Union[str, os.PathLike]
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 1024 * 1024 * 1024


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _selected_profiles(
    profile: str,
    manifest: Dict[str, Any],
) -> List[str]:
    known = sorted(manifest["profiles"])
    if profile == "all":
        return known
    if profile not in manifest["profiles"]:
        raise ValueError(
            "unknown parser profile {!r}; expected one of {}".format(
                profile, ", ".join(known + ["all"])
            )
        )
    return [profile]


def _ordered_union(values: Iterable[Iterable[str]]) -> List[str]:
    result: List[str] = []
    seen = set()
    for group in values:
        for value in group:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _profile_config(
    profile: str,
    manifest: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    selected = _selected_profiles(profile, manifest)
    required = _ordered_union(
        manifest["profiles"][name].get("required", []) for name in selected
    )
    optional = _ordered_union(
        manifest["profiles"][name].get("optional", []) for name in selected
    )
    optional = [name for name in optional if name not in set(required)]
    artifacts = _ordered_union(
        manifest["profiles"][name].get("artifacts", []) for name in selected
    )
    return selected, required, optional, artifacts


def doctor(profile: str, tools_dir: PathLike) -> Dict[str, Any]:
    """Return a JSON-serializable, read-only capability report."""

    manifest = load_artifact_manifest()
    tools_path = Path(tools_dir).expanduser().resolve()
    selected, required, optional, _ = _profile_config(profile, manifest)
    names = required + [name for name in optional if name not in set(required)]
    results: Dict[str, Dict[str, Any]] = {}
    for name in names:
        specification = dict(manifest["capabilities"][name])
        if (
            specification.get("kind") == "python_import"
            and manifest.get("pins", {}).get("python")
        ):
            specification.setdefault(
                "python_version_prefix",
                manifest["pins"]["python"],
            )
        integrity_error = _local_integrity_error(
            name,
            specification,
            manifest,
            tools_path,
        )
        if integrity_error is not None:
            result = {
                "name": name,
                "kind": specification["kind"],
                "available": False,
                "location": integrity_error["location"],
                "version": specification.get("version"),
                "detail": integrity_error["detail"],
            }
        else:
            result = probe_capability(
                name,
                specification,
                tools_path,
            )
        result["required"] = name in set(required)
        result["setup_hint"] = specification.get("setup_hint")
        results[name] = result

    runtime_groups: Dict[str, List[str]] = {}
    shared_parent_groups: Dict[str, List[str]] = {}
    for name in names:
        specification = manifest["capabilities"][name]
        runtime_group = specification.get("runtime_group")
        if runtime_group:
            runtime_groups.setdefault(str(runtime_group), []).append(name)
        parent_group = specification.get("shared_parent_group")
        if parent_group:
            shared_parent_groups.setdefault(str(parent_group), []).append(name)

    for group, members in runtime_groups.items():
        available = [
            name
            for name in members
            if results[name].get("available") and results[name].get("location")
        ]
        if len(available) < 2:
            continue
        anchor_name = (
            "core_python"
            if "core_python" in available
            else available[0]
        )
        anchor = str(
            results[anchor_name].get("environment_id")
            or os.path.abspath(str(results[anchor_name]["location"]))
        )
        for name in available:
            environment_id = str(
                results[name].get("environment_id")
                or os.path.abspath(str(results[name]["location"]))
            )
            if environment_id == anchor:
                continue
            results[name]["available"] = False
            results[name]["detail"] = (
                "runtime group {!r} resolved across different interpreters; "
                "expected {}".format(group, anchor)
            )
            results[name]["runtime_group"] = group
            results[name]["expected_location"] = anchor

    for group, members in shared_parent_groups.items():
        available = [
            name
            for name in members
            if results[name].get("available") and results[name].get("location")
        ]
        if len(available) < 2:
            continue
        anchor_parent = str(
            Path(results[available[0]]["location"]).resolve().parent
        )
        for name in available:
            parent = str(Path(results[name]["location"]).resolve().parent)
            if parent == anchor_parent:
                continue
            results[name]["available"] = False
            results[name]["detail"] = (
                "resource group {!r} must share one directory; expected "
                "{}".format(group, anchor_parent)
            )
            results[name]["shared_parent_group"] = group
            results[name]["expected_parent"] = anchor_parent

    missing_required = [
        name for name in required if not results[name]["available"]
    ]
    missing_optional = [
        name for name in optional if not results[name]["available"]
    ]
    return {
        "schema_version": 1,
        "profile": profile,
        "selected_profiles": selected,
        "platform": platform_key(),
        "tools_dir": str(tools_path),
        "manifest": manifest["_manifest_path"],
        "ready": not missing_required,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "capabilities": results,
        "pins": manifest.get("pins", {}),
    }


def _safe_destination(tools_dir: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("artifact destination must be a non-empty relative path")
    destination = (tools_dir / relative).resolve()
    try:
        destination.relative_to(tools_dir)
    except ValueError as exc:
        raise ValueError(
            "artifact destination escapes tools_dir: {!r}".format(relative)
        ) from exc
    return destination


def _artifact_variant(
    name: str,
    artifact: Dict[str, Any],
    current_platform: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    variants = artifact["variants"]
    if current_platform in variants:
        return current_platform, variants[current_platform]
    if "any" in variants:
        return "any", variants["any"]
    return None, None


def _receipt_path(tools_dir: Path, artifact_name: str) -> Path:
    return tools_dir / ".receipts" / "{}.json".format(artifact_name)


def _matching_receipt(
    tools_dir: Path,
    artifact_name: str,
    variant: Dict[str, Any],
) -> bool:
    receipt_path = _receipt_path(tools_dir, artifact_name)
    try:
        with receipt_path.open("r", encoding="utf-8") as source:
            receipt = json.load(source)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    destination = _safe_destination(tools_dir, variant["destination"])
    installed_sha256 = receipt.get("installed_sha256")
    try:
        destination_sha256 = _sha256_path(destination)
    except OSError:
        return False
    return (
        receipt.get("url") == variant["url"]
        and receipt.get("checksum") == variant["checksum"]
        and receipt.get("checksum_algorithm")
        == variant.get("checksum_algorithm", "sha256")
        and receipt.get("destination") == variant["destination"]
        and isinstance(installed_sha256, str)
        and installed_sha256 == destination_sha256
    )


def _installed_status(
    tools_dir: Path,
    artifact_name: str,
    variant: Dict[str, Any],
) -> str:
    destination = _safe_destination(tools_dir, variant["destination"])
    if not destination.is_file():
        return "missing"
    if variant.get("archive", "file") == "file":
        algorithm = variant.get("checksum_algorithm", "sha256")
        try:
            return (
                "installed"
                if verify_checksum(destination, variant["checksum"], algorithm)
                else "checksum_mismatch"
            )
        except OSError:
            return "unreadable"
    return (
        "installed"
        if _matching_receipt(tools_dir, artifact_name, variant)
        else "present_unverified"
    )


def _local_integrity_error(
    capability_name: str,
    specification: Dict[str, Any],
    manifest: Dict[str, Any],
    tools_dir: Path,
) -> Optional[Dict[str, str]]:
    relative_paths = set(specification.get("relative_paths", []))
    if capability_name == "hwp_worker_jar":
        destination = tools_dir / "java" / "hwp-parser-worker.jar"
        if not destination.exists():
            return None
        try:
            source_hash = _hwp_worker_source_hash(
                _repo_root() / "parser-workers" / "java"
            )
            status = _hwp_worker_status(
                tools_dir,
                destination,
                source_hash,
            )
        except (OSError, ValueError) as exc:
            status = "{}:{}".format(type(exc).__name__, exc)
        if status != "installed":
            return {
                "location": str(destination),
                "detail": "local HWP worker integrity check failed: {}".format(
                    status
                ),
            }
        return None

    current_platform = platform_key()
    for artifact_name, artifact in manifest.get("artifacts", {}).items():
        _, variant = _artifact_variant(
            artifact_name,
            artifact,
            current_platform,
        )
        if variant is None or variant["destination"] not in relative_paths:
            continue
        destination = _safe_destination(
            tools_dir,
            variant["destination"],
        )
        if not destination.exists():
            continue
        status = _installed_status(
            tools_dir,
            artifact_name,
            variant,
        )
        if status != "installed":
            return {
                "location": str(destination),
                "detail": "local artifact integrity check failed: {}".format(
                    status
                ),
            }
    return None


def verified_artifact_path(
    artifact_name: str,
    tools_dir: PathLike,
) -> Optional[Path]:
    """Return a prepared artifact only when its pinned integrity still holds."""

    manifest = load_artifact_manifest()
    artifact = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, dict):
        return None
    _, variant = _artifact_variant(
        artifact_name,
        artifact,
        platform_key(),
    )
    if variant is None:
        return None
    tools_path = Path(tools_dir).expanduser().resolve()
    if _installed_status(
        tools_path,
        artifact_name,
        variant,
    ) != "installed":
        return None
    return _safe_destination(tools_path, variant["destination"])


def verified_hwp_worker_path(tools_dir: PathLike) -> Optional[Path]:
    """Return the installed Java worker after source and JAR verification."""

    tools_path = Path(tools_dir).expanduser().resolve()
    destination = tools_path / "java" / "hwp-parser-worker.jar"
    try:
        source_hash = _hwp_worker_source_hash(
            _repo_root() / "parser-workers" / "java"
        )
        if (
            _hwp_worker_status(
                tools_path,
                destination,
                source_hash,
            )
            != "installed"
        ):
            return None
    except (OSError, ValueError):
        return None
    return destination.resolve()


def _download(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("artifact URL must use HTTPS: {}".format(url))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "pnu-docs-chatbot-parser-prepare/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            _copy_limited(
                response,
                output,
                MAX_DOWNLOAD_BYTES,
                "artifact download",
            )


def _copy_limited(
    source: Any,
    output: Any,
    maximum_bytes: int,
    label: str,
) -> int:
    copied = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        copied += len(chunk)
        if copied > maximum_bytes:
            raise ValueError(
                "{} exceeds {} bytes".format(label, maximum_bytes)
            )
        output.write(chunk)
    return copied


def _read_archive_member(
    archive_path: Path,
    archive_kind: str,
    member_name: str,
    output_path: Path,
) -> None:
    if Path(member_name).is_absolute() or ".." in Path(member_name).parts:
        raise ValueError("unsafe archive member path: {!r}".format(member_name))

    if archive_kind == "tar.gz":
        with tarfile.open(str(archive_path), mode="r:gz") as archive:
            try:
                member = archive.getmember(member_name)
            except KeyError as exc:
                raise ValueError(
                    "archive does not contain {!r}".format(member_name)
                ) from exc
            if not member.isfile():
                raise ValueError(
                    "archive member {!r} is not a regular file".format(member_name)
                )
            if member.size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(
                    "archive member {!r} exceeds {} bytes".format(
                        member_name, MAX_ARCHIVE_MEMBER_BYTES
                    )
                )
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(
                    "cannot read archive member {!r}".format(member_name)
                )
            with source, output_path.open("wb") as output:
                _copy_limited(
                    source,
                    output,
                    MAX_ARCHIVE_MEMBER_BYTES,
                    "archive member",
                )
        return

    if archive_kind == "zip":
        with zipfile.ZipFile(str(archive_path), mode="r") as archive:
            try:
                info = archive.getinfo(member_name)
            except KeyError as exc:
                raise ValueError(
                    "archive does not contain {!r}".format(member_name)
                ) from exc
            if info.is_dir():
                raise ValueError(
                    "archive member {!r} is not a regular file".format(member_name)
                )
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(
                    "archive member {!r} exceeds {} bytes".format(
                        member_name, MAX_ARCHIVE_MEMBER_BYTES
                    )
                )
            with archive.open(info, mode="r") as source, output_path.open(
                "wb"
            ) as output:
                _copy_limited(
                    source,
                    output,
                    MAX_ARCHIVE_MEMBER_BYTES,
                    "archive member",
                )
        return

    raise ValueError("unsupported archive type {!r}".format(archive_kind))


def _write_receipt(
    tools_dir: Path,
    artifact_name: str,
    artifact: Dict[str, Any],
    variant_name: str,
    variant: Dict[str, Any],
) -> None:
    receipt_path = _receipt_path(tools_dir, artifact_name)
    if receipt_path.parent.is_symlink():
        raise ValueError("artifact receipt directory must not be a symlink")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    destination = _safe_destination(tools_dir, variant["destination"])
    receipt = {
        "artifact": artifact_name,
        "version": artifact.get("version"),
        "variant": variant_name,
        "url": variant["url"],
        "checksum": variant["checksum"],
        "checksum_algorithm": variant.get("checksum_algorithm", "sha256"),
        "destination": variant["destination"],
        "installed_sha256": _sha256_path(destination),
        "license": artifact.get("license"),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(artifact_name),
        suffix=".receipt.tmp",
        dir=str(receipt_path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                receipt,
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, str(receipt_path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _install_artifact(
    tools_dir: Path,
    artifact_name: str,
    artifact: Dict[str, Any],
    variant_name: str,
    variant: Dict[str, Any],
) -> None:
    destination = _safe_destination(tools_dir, variant["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_kind = variant.get("archive", "file")
    temporary_download = None
    temporary_install = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".{}-".format(artifact_name),
            suffix=".download",
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temporary_download = Path(handle.name)
        _download(variant["url"], temporary_download)
        algorithm = variant.get("checksum_algorithm", "sha256")
        if not verify_checksum(
            temporary_download,
            variant["checksum"],
            algorithm,
        ):
            raise ValueError(
                "{} download checksum mismatch".format(artifact_name)
            )

        if archive_kind == "file":
            os.replace(str(temporary_download), str(destination))
            temporary_download = None
        else:
            with tempfile.NamedTemporaryFile(
                prefix=".{}-".format(artifact_name),
                suffix=".install",
                dir=str(destination.parent),
                delete=False,
            ) as handle:
                temporary_install = Path(handle.name)
            _read_archive_member(
                temporary_download,
                archive_kind,
                variant["archive_member"],
                temporary_install,
            )
            os.chmod(
                str(temporary_install),
                os.stat(str(temporary_install)).st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH,
            )
            os.replace(str(temporary_install), str(destination))
            temporary_install = None

        _write_receipt(
            tools_dir,
            artifact_name,
            artifact,
            variant_name,
            variant,
        )
    finally:
        for temporary in (temporary_download, temporary_install):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _hwp_worker_source_hash(worker_root: Path) -> str:
    digest = hashlib.sha256()
    sources = [worker_root / "pom.xml"]
    sources.extend(sorted((worker_root / "src").rglob("*.java")))
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(str(source))
        relative = source.relative_to(worker_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with source.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _hwp_worker_receipt(tools_dir: Path) -> Path:
    return tools_dir / ".receipts" / "hwp-worker.json"


def _hwp_worker_status(
    tools_dir: Path,
    destination: Path,
    source_hash: str,
) -> str:
    if not destination.is_file():
        return "missing"
    try:
        receipt = json.loads(
            _hwp_worker_receipt(tools_dir).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return "present_unverified"
    expected_jar_hash = receipt.get("installed_sha256")
    try:
        actual_jar_hash = _sha256_path(destination)
    except OSError:
        return "unreadable"
    if (
        receipt.get("source_sha256") == source_hash
        and receipt.get("destination") == "java/hwp-parser-worker.jar"
        and isinstance(expected_jar_hash, str)
        and expected_jar_hash == actual_jar_hash
    ):
        return "installed"
    if receipt.get("source_sha256") != source_hash:
        return "source_changed"
    return "checksum_mismatch"


def _find_executable(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_file() and os.access(str(resolved), os.X_OK):
            return resolved
    return None


def _find_maven() -> Optional[Path]:
    candidates: List[Path] = []
    configured = os.environ.get("PARSER_MAVEN")
    if configured:
        candidates.append(Path(configured))
    discovered = shutil.which("mvn")
    if discovered:
        candidates.append(Path(discovered))
    local_share = Path.home() / ".local" / "share"
    if local_share.is_dir():
        candidates.extend(
            sorted(local_share.glob("maven*/**/bin/mvn"), key=lambda path: str(path))
        )
    return _find_executable(candidates)


def _find_java_home(tools_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    configured = os.environ.get("JAVA_HOME")
    if configured:
        candidates.append(Path(configured))
    candidates.append(tools_dir / "jdk")
    local_share = Path.home() / ".local" / "share"
    if local_share.is_dir():
        candidates.extend(
            sorted(local_share.glob("jdk*/**/Contents/Home"), key=lambda path: str(path))
        )
        candidates.extend(
            sorted(local_share.glob("jdk*"), key=lambda path: str(path))
        )
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        java = resolved / "bin" / "java"
        if java.is_file() and os.access(str(java), os.X_OK):
            return resolved
    return None


def _write_hwp_worker_receipt(
    tools_dir: Path,
    destination: Path,
    source_hash: str,
    version: str,
) -> None:
    receipt_path = _hwp_worker_receipt(tools_dir)
    if receipt_path.parent.is_symlink():
        raise ValueError("HWP worker receipt directory must not be a symlink")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "hwp_worker",
        "version": version,
        "source_sha256": source_hash,
        "destination": "java/hwp-parser-worker.jar",
        "installed_sha256": _sha256_path(destination),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".hwp-worker.",
        suffix=".tmp",
        dir=str(receipt_path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, str(receipt_path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _prepare_hwp_worker(
    tools_dir: Path,
    version: str,
    execute: bool,
) -> Dict[str, Any]:
    worker_root = _repo_root() / "parser-workers" / "java"
    destination = tools_dir / "java" / "hwp-parser-worker.jar"
    action: Dict[str, Any] = {
        "build": "hwp_worker",
        "version": version,
        "source": str(worker_root),
        "destination": str(destination),
        "changed": False,
    }
    try:
        source_hash = _hwp_worker_source_hash(worker_root)
    except (OSError, ValueError) as exc:
        action.update(
            {
                "status": "error",
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
        )
        return action
    action["source_sha256"] = source_hash
    status = _hwp_worker_status(tools_dir, destination, source_hash)
    if status == "installed":
        action["status"] = "installed"
        return action
    if not execute:
        action["status"] = "would_build"
        action["previous_status"] = status
        return action

    maven = _find_maven()
    java_home = _find_java_home(tools_dir)
    if maven is None or java_home is None:
        missing = []
        if maven is None:
            missing.append("Maven")
        if java_home is None:
            missing.append("JDK")
        action.update(
            {
                "status": "unavailable",
                "error": "{} not found".format(" and ".join(missing)),
            }
        )
        return action

    environment = dict(os.environ)
    environment["JAVA_HOME"] = str(java_home)
    try:
        completed = subprocess.run(
            [str(maven), "-q", "-DskipTests", "package"],
            cwd=str(worker_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=900,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        action.update(
            {
                "status": "error",
                "error": "{}: {}".format(type(exc).__name__, exc),
            }
        )
        return action
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        action.update(
            {
                "status": "error",
                "error": "Maven exited with {}: {}".format(
                    completed.returncode, detail[-4000:]
                ),
            }
        )
        return action

    built = worker_root / "target" / "hwp-parser-worker.jar"
    if not built.is_file():
        action.update(
            {
                "status": "error",
                "error": "Maven completed without target/hwp-parser-worker.jar",
            }
        )
        return action
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".hwp-parser-worker.",
        suffix=".jar",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    try:
        shutil.copyfile(str(built), temporary_name)
        os.replace(temporary_name, str(destination))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    _write_hwp_worker_receipt(
        tools_dir,
        destination,
        source_hash,
        version,
    )
    action["status"] = "built"
    action["changed"] = True
    action["maven"] = str(maven)
    action["java_home"] = str(java_home)
    return action


def prepare(
    profile: str,
    tools_dir: PathLike,
    execute: bool = False,
) -> Dict[str, Any]:
    """Inspect or explicitly download pinned artifacts into *tools_dir*.

    ``execute=False`` is the safe default: it does not create directories,
    access the network, or mutate files.  ``execute=True`` downloads only the
    manifest artifacts for the selected profile.  For profiles using HWP/HWPX,
    it also builds the checked-in, version-pinned Java worker and copies the
    resulting JAR into ``tools_dir``.
    """

    manifest = load_artifact_manifest()
    tools_path = Path(tools_dir).expanduser().resolve()
    selected, required, _, artifact_names = _profile_config(profile, manifest)
    before = doctor(profile, tools_path)
    current_platform = platform_key()
    actions: List[Dict[str, Any]] = []

    for artifact_name in artifact_names:
        artifact = manifest["artifacts"][artifact_name]
        variant_name, variant = _artifact_variant(
            artifact_name,
            artifact,
            current_platform,
        )
        if variant is None or variant_name is None:
            actions.append(
                {
                    "artifact": artifact_name,
                    "version": artifact.get("version"),
                    "status": "unsupported_platform",
                    "platform": current_platform,
                    "changed": False,
                    "error": "no artifact variant for this platform",
                }
            )
            continue

        status = _installed_status(tools_path, artifact_name, variant)
        action = {
            "artifact": artifact_name,
            "version": artifact.get("version"),
            "license": artifact.get("license"),
            "platform": variant_name,
            "destination": str(
                _safe_destination(tools_path, variant["destination"])
            ),
            "url": variant["url"],
            "checksum_algorithm": variant.get(
                "checksum_algorithm", "sha256"
            ),
            "checksum": variant["checksum"],
            "status": status,
            "changed": False,
        }
        if status == "installed":
            actions.append(action)
            continue
        if not execute:
            action["status"] = "would_download"
            action["previous_status"] = status
            actions.append(action)
            continue

        try:
            _install_artifact(
                tools_path,
                artifact_name,
                artifact,
                variant_name,
                variant,
            )
            action["status"] = "installed"
            action["changed"] = True
        except (
            OSError,
            ValueError,
            urllib.error.URLError,
            tarfile.TarError,
            zipfile.BadZipFile,
        ) as exc:
            action["status"] = "error"
            action["error"] = "{}: {}".format(type(exc).__name__, exc)
        actions.append(action)

    builds: List[Dict[str, Any]] = []
    if "hwp_worker_jar" in required:
        builds.append(
            _prepare_hwp_worker(
                tools_path,
                str(manifest.get("pins", {}).get("hwp_worker", "1.0.0")),
                execute,
            )
        )

    after = doctor(profile, tools_path) if execute else before
    manual_actions = []
    for name in after["missing_required"] + after["missing_optional"]:
        hint = after["capabilities"][name].get("setup_hint")
        if hint:
            manual_actions.append(
                {
                    "capability": name,
                    "required": after["capabilities"][name]["required"],
                    "hint": hint,
                }
            )

    errors = [
        action["artifact"]
        for action in actions
        if action["status"] in {"error", "unsupported_platform"}
    ]
    errors.extend(
        action["build"]
        for action in builds
        if action["status"] in {"error", "unavailable"}
    )
    return {
        "schema_version": 1,
        "profile": profile,
        "selected_profiles": selected,
        "platform": current_platform,
        "tools_dir": str(tools_path),
        "execute": bool(execute),
        "changed": any(action["changed"] for action in actions + builds),
        "ready": after["ready"],
        "errors": errors,
        "actions": actions,
        "builds": builds,
        "manual_actions": manual_actions,
        "before": before,
        "after": after,
    }
