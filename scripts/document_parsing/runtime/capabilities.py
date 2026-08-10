"""Read-only runtime capability probes for parser profiles."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .artifacts import verify_checksum


_VERSION_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+){0,3})(?!\d)")


def platform_key() -> str:
    """Return the manifest platform key for the current interpreter."""

    os_name = {
        "darwin": "macos",
        "linux": "linux",
        "win32": "windows",
    }.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    arch = {
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "amd64": "x86_64",
        "x86_64": "x86_64",
    }.get(machine, machine)
    return "{}-{}".format(os_name, arch)


def _parse_version(value: str) -> Optional[Tuple[int, ...]]:
    match = _VERSION_RE.search(value)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _meets_minimum(actual: Optional[Tuple[int, ...]], minimum: str) -> bool:
    if actual is None:
        return False
    required = tuple(int(part) for part in minimum.split("."))
    width = max(len(actual), len(required))
    return actual + (0,) * (width - len(actual)) >= required + (0,) * (
        width - len(required)
    )


def _matches_exact(
    actual: Optional[Tuple[int, ...]],
    expected: str,
) -> bool:
    if actual is None:
        return False
    required = tuple(int(part) for part in expected.split("."))
    width = max(len(actual), len(required))
    return actual + (0,) * (width - len(actual)) == required + (0,) * (
        width - len(required)
    )


def _matches_prefix(
    actual: Optional[Tuple[int, ...]],
    expected: str,
) -> bool:
    if actual is None:
        return False
    required = tuple(int(part) for part in expected.split("."))
    return actual[: len(required)] == required


def _contains_stable_version_token(output: str, expected: str) -> bool:
    return bool(
        re.search(
            r"(?<![\d.]){}(?![\d.]|[-+~A-Za-z])".format(
                re.escape(expected)
            ),
            output,
        )
    )


def _safe_relative(
    tools_dir: Path,
    value: str,
    *,
    allow_final_symlink: bool = False,
) -> Optional[Path]:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    base = tools_dir.resolve()
    candidate = base / relative
    try:
        candidate.parent.resolve().relative_to(base)
    except ValueError:
        return None
    if allow_final_symlink:
        return candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(base)
    except (OSError, ValueError):
        return None
    return resolved


def _candidate_commands(
    spec: Dict[str, Any],
    tools_dir: Path,
) -> Iterable[str]:
    seen = set()
    for relative in spec.get("relative_paths", []):
        candidate = _safe_relative(
            tools_dir,
            relative,
            allow_final_symlink=True,
        )
        if candidate is not None and candidate.is_file():
            value = str(candidate)
            if value not in seen:
                seen.add(value)
                yield value

    if spec.get("java_home", False):
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            candidate = Path(java_home).expanduser() / "bin" / "java"
            if candidate.is_file():
                value = str(candidate.resolve())
                if value not in seen:
                    seen.add(value)
                    yield value
        local_share = Path.home() / ".local" / "share"
        if local_share.is_dir():
            for candidate in sorted(
                local_share.glob("jdk*/**/bin/java"),
                key=lambda path: str(path),
            ):
                if candidate.is_file():
                    value = str(candidate.resolve())
                    if value not in seen:
                        seen.add(value)
                        yield value

    for command in spec.get("commands", []):
        found = shutil.which(command)
        if found:
            value = os.path.abspath(found)
            if value not in seen:
                seen.add(value)
                yield value


def _run(
    argv: Sequence[str],
    timeout: float = 5.0,
) -> Tuple[int, str]:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip()


def _probe_command(
    name: str,
    spec: Dict[str, Any],
    tools_dir: Path,
) -> Dict[str, Any]:
    attempted: List[str] = []
    rejected_versions: List[Dict[str, Any]] = []
    version_args = spec.get("version_args", ["--version"])
    for command in _candidate_commands(spec, tools_dir):
        attempted.append(command)
        code, output = _run([command] + list(version_args))
        if code != 0:
            continue
        parsed = _parse_version(output)
        minimum = spec.get("min_version")
        if minimum and not _meets_minimum(parsed, minimum):
            rejected_versions.append(
                {
                    "location": command,
                    "version": ".".join(str(part) for part in parsed)
                    if parsed
                    else None,
                    "reason": "below minimum {}".format(minimum),
                }
            )
            continue
        exact = spec.get("exact_version")
        if exact and (
            not _matches_exact(parsed, exact)
            or not _contains_stable_version_token(output, exact)
        ):
            rejected_versions.append(
                {
                    "location": command,
                    "version": ".".join(str(part) for part in parsed)
                    if parsed
                    else None,
                    "reason": "does not equal pinned version {}".format(exact),
                }
            )
            continue
        prefix = spec.get("version_prefix")
        if prefix and not _matches_prefix(parsed, prefix):
            rejected_versions.append(
                {
                    "location": command,
                    "version": ".".join(str(part) for part in parsed)
                    if parsed
                    else None,
                    "reason": "does not match required version family {}".format(
                        prefix
                    ),
                }
            )
            continue
        return {
            "name": name,
            "kind": spec["kind"],
            "available": True,
            "location": command,
            "version": ".".join(str(part) for part in parsed)
            if parsed
            else None,
            "detail": output.splitlines()[0] if output else "command is executable",
        }

    result = {
        "name": name,
        "kind": spec["kind"],
        "available": False,
        "location": None,
        "version": None,
        "detail": "command not found",
        "attempted": attempted,
    }
    if rejected_versions:
        result["detail"] = "no command matched the required version"
        result["rejected_versions"] = rejected_versions
    return result


def _path_candidates(spec: Dict[str, Any], tools_dir: Path) -> Iterable[Path]:
    seen = set()
    for relative in spec.get("relative_paths", []):
        candidate = _safe_relative(tools_dir, relative)
        if candidate is not None:
            value = str(candidate)
            if value not in seen:
                seen.add(value)
                yield candidate
    for absolute in spec.get("absolute_paths", []):
        candidate = Path(absolute).expanduser()
        value = str(candidate)
        if value not in seen:
            seen.add(value)
            yield candidate


def _regular_nonempty_file(path: Path) -> bool:
    try:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
        )
    except OSError:
        return False


def _probe_file_or_model(
    name: str,
    spec: Dict[str, Any],
    tools_dir: Path,
) -> Dict[str, Any]:
    attempted = []
    for candidate in _path_candidates(spec, tools_dir):
        attempted.append(str(candidate))
        is_present = candidate.is_dir() if spec["kind"] == "model" else candidate.is_file()
        if not is_present:
            continue
        if spec["kind"] == "model":
            required_files = spec.get("required_files", [])
            missing = [
                relative
                for relative in required_files
                if not _regular_nonempty_file(candidate / relative)
            ]
            if missing:
                continue
            required_globs = spec.get("required_globs", [])
            missing_globs = []
            for pattern in required_globs:
                try:
                    match = next(
                        (
                            path
                            for path in candidate.glob(pattern)
                            if _regular_nonempty_file(path)
                        ),
                        None,
                    )
                except OSError:
                    match = None
                if match is None:
                    missing_globs.append(pattern)
            if missing_globs:
                continue
            if spec.get("require_nonempty", False):
                try:
                    next(candidate.iterdir())
                except (StopIteration, OSError):
                    continue
        checksum = spec.get("checksum")
        if checksum:
            try:
                matches = verify_checksum(
                    candidate,
                    checksum,
                    spec.get("checksum_algorithm", "sha256"),
                )
            except OSError:
                matches = False
            if not matches:
                return {
                    "name": name,
                    "kind": spec["kind"],
                    "available": False,
                    "location": str(candidate.resolve()),
                    "version": spec.get("version"),
                    "detail": "local file checksum does not match the pinned artifact",
                }
        return {
            "name": name,
            "kind": spec["kind"],
            "available": True,
            "location": str(candidate.resolve()),
            "version": spec.get("version"),
            "detail": "local {} is present".format(spec["kind"]),
        }
    return {
        "name": name,
        "kind": spec["kind"],
        "available": False,
        "location": None,
        "version": None,
        "detail": "{} not found".format(spec["kind"]),
        "attempted": attempted,
    }


def _python_candidates(spec: Dict[str, Any], tools_dir: Path) -> Iterable[str]:
    command_spec = {
        "relative_paths": spec.get("interpreter_relative_paths", []),
        "commands": spec.get("interpreter_commands", []),
    }
    yielded = False
    for command in _candidate_commands(command_spec, tools_dir):
        yielded = True
        yield command
    if spec.get("allow_current_interpreter", False):
        current = os.path.abspath(sys.executable)
        if not yielded or current not in set(
            _candidate_commands(command_spec, tools_dir)
        ):
            yield current


_IMPORT_PROBE = r"""
import importlib
import json
import os
import sys
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata
module = {module!r}
distribution = {distribution!r}
required_imports = {required_imports!r}
required_symbols = {required_symbols!r}
errors = []
try:
    importlib.import_module(module)
except Exception as exc:
    errors.append("{{}}: {{}}".format(type(exc).__name__, exc))
for required_module in required_imports:
    try:
        importlib.import_module(required_module)
    except Exception as exc:
        errors.append(
            "{{}}: {{}}: {{}}".format(
                required_module,
                type(exc).__name__,
                exc,
            )
        )
for required_module, names in required_symbols.items():
    try:
        imported = importlib.import_module(required_module)
        missing = [name for name in names if not hasattr(imported, name)]
        if missing:
            errors.append(
                "{{}} missing {{}}".format(
                    required_module,
                    ",".join(missing),
                )
            )
    except Exception as exc:
        errors.append(
            "{{}}: {{}}: {{}}".format(
                required_module,
                type(exc).__name__,
                exc,
            )
        )
found = not errors
version = None
if found and distribution:
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps({{
    "found": found,
    "version": version,
    "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    "environment_id": os.path.abspath(sys.prefix),
    "errors": errors,
}}))
"""


def _probe_python_import(
    name: str,
    spec: Dict[str, Any],
    tools_dir: Path,
) -> Dict[str, Any]:
    attempted = []
    rejected_versions: List[Dict[str, Any]] = []
    import_errors: List[Dict[str, Any]] = []
    module = spec["module"]
    distribution = spec.get("distribution", module)
    for interpreter in _python_candidates(spec, tools_dir):
        attempted.append(interpreter)
        script = _IMPORT_PROBE.format(
            module=module,
            distribution=distribution,
            required_imports=tuple(spec.get("required_imports", ())),
            required_symbols=dict(spec.get("required_symbols", {})),
        )
        code, output = _run(
            [interpreter, "-c", script],
            timeout=float(spec.get("probe_timeout_seconds", 15)),
        )
        if code != 0:
            continue
        try:
            result = json.loads(output.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            continue
        if not result.get("found"):
            import_errors.append(
                {
                    "location": interpreter,
                    "errors": result.get("errors") or [
                        "module import failed"
                    ],
                }
            )
            continue
        version = result.get("version")
        python_version = result.get("python_version")
        environment_id = result.get("environment_id")
        python_prefix = spec.get("python_version_prefix")
        if python_prefix and not _matches_prefix(
            _parse_version(python_version or ""),
            python_prefix,
        ):
            rejected_versions.append(
                {
                    "location": interpreter,
                    "version": version,
                    "python_version": python_version,
                    "reason": (
                        "interpreter does not match required Python family {}"
                    ).format(python_prefix),
                }
            )
            continue
        minimum = spec.get("min_version")
        if minimum and not _meets_minimum(_parse_version(version or ""), minimum):
            rejected_versions.append(
                {
                    "location": interpreter,
                    "version": version,
                    "reason": "below minimum {}".format(minimum),
                }
            )
            continue
        exact = spec.get("exact_version")
        if exact and version != exact:
            rejected_versions.append(
                {
                    "location": interpreter,
                    "version": version,
                    "reason": "does not equal pinned version {}".format(exact),
                }
            )
            continue
        return {
            "name": name,
            "kind": spec["kind"],
            "available": True,
            "location": interpreter,
            "version": version,
            "python_version": python_version,
            "environment_id": environment_id,
            "detail": "Python module {!r} is importable".format(module),
        }
    result = {
        "name": name,
        "kind": spec["kind"],
        "available": False,
        "location": None,
        "version": None,
        "detail": "Python module {!r} is not importable".format(module),
        "attempted": attempted,
    }
    if rejected_versions:
        result["detail"] = (
            "Python module {!r} was found, but no interpreter matched the "
            "required version".format(module)
        )
        result["rejected_versions"] = rejected_versions
    elif import_errors:
        result["detail"] = (
            "Python module {!r} or its required API could not be imported"
        ).format(module)
        result["import_errors"] = import_errors
    return result


def _probe_tessdata(
    name: str,
    spec: Dict[str, Any],
    tools_dir: Path,
) -> Dict[str, Any]:
    filename = spec["filename"]
    candidates = list(_path_candidates(spec, tools_dir))
    tessdata_prefix = os.environ.get("TESSDATA_PREFIX")
    if tessdata_prefix:
        prefix = Path(tessdata_prefix).expanduser()
        candidates.extend([prefix / filename, prefix / "tessdata" / filename])
    attempted = []
    for candidate in candidates:
        attempted.append(str(candidate))
        if _regular_nonempty_file(candidate):
            return {
                "name": name,
                "kind": spec["kind"],
                "available": True,
                "location": str(candidate.resolve()),
                "version": spec.get("version"),
                "detail": "{} is present".format(filename),
            }
    return {
        "name": name,
        "kind": spec["kind"],
        "available": False,
        "location": None,
        "version": None,
        "detail": "{} not found".format(filename),
        "attempted": attempted,
    }


def probe_capability(
    name: str,
    spec: Dict[str, Any],
    tools_dir: Path,
) -> Dict[str, Any]:
    """Probe one capability without changing local or global state."""

    kind = spec["kind"]
    if kind == "command":
        return _probe_command(name, spec, tools_dir)
    if kind in {"file", "model"}:
        return _probe_file_or_model(name, spec, tools_dir)
    if kind == "python_import":
        return _probe_python_import(name, spec, tools_dir)
    if kind == "tessdata":
        return _probe_tessdata(name, spec, tools_dir)
    raise ValueError("unsupported capability kind {!r}".format(kind))
