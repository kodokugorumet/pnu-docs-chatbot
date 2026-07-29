#!/usr/bin/env python3
"""Run and validate the block-first document parser profiles.

Examples:
  python scripts/parse_pipeline.py prepare --profile all
  python scripts/parse_pipeline.py prepare --profile baseline --execute
  python scripts/parse_pipeline.py doctor --profile cascade
  python scripts/parse_pipeline.py run --profile all --input src/data
  python scripts/parse_pipeline.py verify-run --run processed/runs/20260723T120000Z
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import site
import stat
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit


# Running ``python scripts/parse_pipeline.py`` puts scripts/, rather than the
# repository, on sys.path. Add the repository before importing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.document_parsing.adapters import sniff_source
from scripts.document_parsing.core import (
    Attempt,
    ParseResult,
    SourceDocument,
    assess_quality,
    normalize_relative_path,
)
from scripts.document_parsing.output import (
    verify_profile_run,
    write_profile_run,
)
from scripts.document_parsing.pipeline import (
    PROFILES,
    PipelineConfig,
    PipelineOutcome,
    PipelineRunner,
    build_source_document,
)
from scripts.document_parsing.runtime import doctor, prepare


PROFILE_CHOICES = tuple(PROFILES) + ("all",)
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
MAX_WORKERS = 32
CORPUS_MANIFEST_REQUIRED_FIELDS = frozenset(
    {"input_relative_path", "sha256", "size_bytes"}
)
CORPUS_MANIFEST_SOURCE_FIELDS = frozenset(
    {
        "source_title",
        "source_url",
        "download_url",
        "source_host",
        "fetched_at",
        "published_at",
        "category",
        "include_reason",
        "source_aliases",
        "crawl_storage_path",
    }
)
CORPUS_MANIFEST_FIELDS = (
    CORPUS_MANIFEST_REQUIRED_FIELDS | CORPUS_MANIFEST_SOURCE_FIELDS
)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _isolate_python_import_environment() -> None:
    """Match doctor by excluding PYTHONPATH and per-user site packages."""

    excluded = set()
    for value in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if value:
            excluded.add(os.path.realpath(os.path.abspath(value)))
    try:
        user_sites = site.getusersitepackages()
    except (AttributeError, OSError):
        user_sites = ()
    if isinstance(user_sites, str):
        user_sites = (user_sites,)
    excluded.update(
        os.path.realpath(os.path.abspath(value))
        for value in user_sites
        if value
    )
    sys.path[:] = [
        value
        for value in sys.path
        if os.path.realpath(os.path.abspath(value or os.curdir))
        not in excluded
    ]
    os.environ.pop("PYTHONPATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"


def _default_path(relative: str) -> str:
    return str(REPO_ROOT / relative)


def _json_print(value: Mapping[str, Any], stream: Any = sys.stdout) -> None:
    json.dump(
        value,
        stream,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    stream.write("\n")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _profile_names(profile: str) -> Tuple[str, ...]:
    return tuple(PROFILES) if profile == "all" else (profile,)


def _normalize_extensions(values: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if not values:
        return None
    extensions: Set[str] = set()
    for value in values:
        for item in value.split(","):
            normalized = item.strip().lower()
            if not normalized:
                continue
            if normalized == "*":
                return None
            if not normalized.startswith("."):
                normalized = "." + normalized
            extensions.add(normalized)
    return extensions or None


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class CorpusManifestSelection:
    """Verified, deterministic parser inputs selected by a JSONL manifest."""

    sources: Tuple[SourceDocument, ...]
    source_manifest_sha256: str
    selection_counts: Mapping[str, int]


def _validate_http_url(value: Any, field: str, line_number: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            "corpus manifest line {} {} must be a non-empty HTTP(S) URL".format(
                line_number, field
            )
        )
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "corpus manifest line {} {} must be an HTTP(S) URL".format(
                line_number, field
            )
        )
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            "corpus manifest line {} {} is invalid: {}".format(
                line_number, field, exc
            )
        ) from exc
    if (
        not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        and not (1 <= port <= 65535)
    ):
        raise ValueError(
            "corpus manifest line {} {} must contain a valid host "
            "without credentials".format(line_number, field)
        )
    return value


def _manifest_source_metadata(
    record: Mapping[str, Any],
    line_number: int,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for field in sorted(CORPUS_MANIFEST_SOURCE_FIELDS - {"source_aliases"}):
        if field not in record:
            continue
        value = record[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(
                "corpus manifest line {} {} must be null or a string".format(
                    line_number, field
                )
            )
        metadata[field] = value

    for field in ("source_url", "download_url"):
        value = metadata.get(field)
        if value is not None:
            metadata[field] = _validate_http_url(
                value, field, line_number
            )

    aliases = record.get("source_aliases", [])
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list) or any(
        not isinstance(value, str) for value in aliases
    ):
        raise ValueError(
            "corpus manifest line {} source_aliases must be a list of "
            "HTTP(S) URLs".format(line_number)
        )
    metadata["source_aliases"] = tuple(
        _validate_http_url(value, "source_aliases", line_number)
        for value in aliases
    )
    return metadata


def load_corpus_manifest(
    manifest_path: Path,
    input_root: Path,
    profile: str,
    repo_root: Optional[Path] = None,
) -> CorpusManifestSelection:
    """Load and verify an authoritative JSONL inclusion manifest.

    Each entry names one regular file below ``input_root`` and pins its byte
    size and SHA-256. No directory discovery or suffix filtering is performed.
    """

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("input directory does not exist: {}".format(root))
    requested_manifest = Path(manifest_path).expanduser()
    if requested_manifest.is_symlink():
        raise ValueError("corpus manifest must not be a symlink")
    manifest = requested_manifest.resolve()
    if not manifest.is_file():
        raise ValueError("corpus manifest does not exist: {}".format(manifest))
    data = manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("corpus manifest must be UTF-8 JSONL") from exc

    sources: List[SourceDocument] = []
    seen_relative_paths: Set[str] = set()
    entry_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        entry_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "corpus manifest line {} is invalid JSON: {}".format(
                    line_number, exc
                )
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                "corpus manifest line {} must contain an object".format(
                    line_number
                )
            )

        has_canonical_path = "input_relative_path" in record
        has_legacy_path = "relative_path" in record
        if has_canonical_path and has_legacy_path:
            raise ValueError(
                "corpus manifest line {} must not contain both "
                "input_relative_path and relative_path".format(line_number)
            )
        allowed_fields = set(CORPUS_MANIFEST_FIELDS)
        if has_legacy_path and not has_canonical_path:
            allowed_fields.add("relative_path")
        unexpected = sorted(set(record) - allowed_fields)
        if unexpected:
            raise ValueError(
                "corpus manifest line {} has unsupported fields: {}".format(
                    line_number, ", ".join(unexpected)
                )
            )

        path_value = record.get(
            "input_relative_path",
            record.get("relative_path"),
        )
        if not isinstance(path_value, str):
            raise ValueError(
                "corpus manifest line {} input_relative_path must be a "
                "string".format(line_number)
            )
        try:
            relative_path = normalize_relative_path(path_value)
        except ValueError as exc:
            raise ValueError(
                "corpus manifest line {} has invalid input_relative_path: "
                "{}".format(line_number, exc)
            ) from exc
        if relative_path in seen_relative_paths:
            raise ValueError(
                "corpus manifest contains duplicate input_relative_path: "
                "{}".format(relative_path)
            )
        seen_relative_paths.add(relative_path)

        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or not SHA256_PATTERN.fullmatch(expected_sha256)
        ):
            raise ValueError(
                "corpus manifest line {} sha256 must be a 64-character "
                "hex digest".format(line_number)
            )
        expected_sha256 = expected_sha256.lower()
        expected_size = record.get("size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError(
                "corpus manifest line {} size_bytes must be a non-negative "
                "integer".format(line_number)
            )

        candidate = root / Path(relative_path)
        if candidate.is_symlink():
            raise ValueError(
                "corpus manifest input must not be a symlink: {}".format(
                    relative_path
                )
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "corpus manifest input does not exist: {}".format(
                    relative_path
                )
            ) from exc
        if not _path_is_within(resolved, root):
            raise ValueError(
                "corpus manifest input escapes --input: {}".format(
                    relative_path
                )
            )
        if not resolved.is_file():
            raise ValueError(
                "corpus manifest input is not a regular file: {}".format(
                    relative_path
                )
            )

        actual_size = resolved.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "corpus manifest size mismatch for {}: expected {}, got "
                "{}".format(relative_path, expected_size, actual_size)
            )
        source = build_source_document(
            resolved,
            root,
            profile,
            repo_root=repo_root,
            source_metadata=_manifest_source_metadata(
                record, line_number
            ),
        )
        if source.relative_path != relative_path:
            raise ValueError(
                "corpus manifest path resolves to a different input path: "
                "{}".format(relative_path)
            )
        if source.source_sha256 != expected_sha256:
            raise ValueError(
                "corpus manifest SHA-256 mismatch for {}".format(
                    relative_path
                )
            )
        sources.append(source)

    if not sources:
        raise ValueError("corpus manifest contains no input entries")
    sources.sort(key=lambda source: source.relative_path)
    return CorpusManifestSelection(
        sources=tuple(sources),
        source_manifest_sha256=manifest_sha256,
        selection_counts={
            "manifest_entries": entry_count,
            "selected_files": len(sources),
        },
    )


def discover_input_files(
    input_root: Path,
    institutions: Optional[Set[str]] = None,
    extensions: Optional[Set[str]] = None,
    excluded_roots: Iterable[Path] = (),
    limit: Optional[int] = None,
) -> List[Path]:
    """Return a stable, filtered list without following symlinked files."""

    root = Path(input_root).resolve()
    if not root.is_dir():
        raise ValueError("input directory does not exist: {}".format(root))
    excluded = tuple(Path(item).resolve() for item in excluded_roots)
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = path.resolve()
        if any(_path_is_within(resolved, excluded_root) for excluded_root in excluded):
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        institution = relative.parts[0] if len(relative.parts) > 1 else ""
        if institutions and institution not in institutions:
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        files.append(resolved)
    files.sort(key=lambda item: item.relative_to(root).as_posix())
    if limit is not None:
        files = files[:limit]
    return files


def _sniff_details(source: SourceDocument) -> Tuple[str, str]:
    try:
        sniffed = sniff_source(source.path)
        return sniffed.format, sniffed.mime_type
    except (OSError, ValueError):
        return source.extension.lstrip(".") or "unknown", (
            source.mime_type or "application/octet-stream"
        )


def _nonparsed_outcome(
    source: SourceDocument,
    status: str,
    reason: str,
    attempt_status: str,
    metadata: Optional[Mapping[str, Any]] = None,
    error_type: Optional[str] = None,
    min_chars: int = 20,
    expect_korean: bool = False,
) -> PipelineOutcome:
    format_name, mime_type = _sniff_details(source)
    attempt = Attempt(
        parser="{}/router".format(source.profile),
        status=attempt_status,
        reason=reason,
        metadata=dict(metadata or {}),
        error_type=error_type,
    )
    result = ParseResult(source, attempts=[attempt])
    quality = assess_quality(
        blocks=(),
        attempt=attempt,
        min_non_whitespace_chars=min_chars,
        expect_korean=expect_korean,
    )
    return PipelineOutcome(
        source=source,
        sniffed_format=format_name,
        mime_type=mime_type,
        result=result,
        quality=quality,
        status=status,
        reason=reason,
        selected_parsers=(),
    )


def _parse_one(
    runner: PipelineRunner,
    source: SourceDocument,
    max_file_mb: Optional[float],
) -> PipelineOutcome:
    config = runner.config
    if max_file_mb is not None:
        maximum_bytes = int(max_file_mb * 1024 * 1024)
        if source.size_bytes > maximum_bytes:
            return _nonparsed_outcome(
                source,
                status="deferred",
                reason="over_max_file_mb:{}".format(max_file_mb),
                attempt_status="skipped",
                metadata={
                    "size_bytes": source.size_bytes,
                    "max_file_mb": max_file_mb,
                    "max_bytes": maximum_bytes,
                },
                min_chars=config.min_chars,
                expect_korean=config.expect_korean,
            )
    try:
        return runner.run(source)
    except Exception as exc:
        reason = "{}: {}".format(type(exc).__name__, exc)
        return _nonparsed_outcome(
            source,
            status="error",
            reason=reason,
            attempt_status="error",
            metadata={"stage": "pipeline"},
            error_type=type(exc).__name__,
            min_chars=config.min_chars,
            expect_korean=config.expect_korean,
        )


def run_sources(
    runner: PipelineRunner,
    sources: Sequence[SourceDocument],
    workers: int,
    max_file_mb: Optional[float],
) -> Iterator[PipelineOutcome]:
    """Parse with bounded concurrency and stream source-order results."""

    if not sources:
        return
    worker_count = max(1, min(int(workers), MAX_WORKERS, len(sources)))
    if worker_count == 1:
        for index, source in enumerate(sources, start=1):
            yield _parse_one(runner, source, max_file_mb)
            if index % 25 == 0 or index == len(sources):
                print(
                    "[{}] completed {}/{}".format(
                        runner.config.profile, index, len(sources)
                    ),
                    file=sys.stderr,
                )
        return

    pending: Dict[Future, int] = {}
    ready: Dict[int, PipelineOutcome] = {}
    next_index = 0
    next_yield = 0
    completed = 0
    # Keep both active threads and the executor queue bounded. This matters for
    # corpora with many thousands of paths and large SourceDocument objects.
    window = max(worker_count, worker_count * 2)
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="parser-{}".format(runner.config.profile),
    )
    try:
        while (
            next_index < len(sources)
            and len(pending) + len(ready) < window
        ):
            future = executor.submit(
                _parse_one, runner, sources[next_index], max_file_mb
            )
            pending[future] = next_index
            next_index += 1

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                ready[index] = future.result()
                completed += 1
            while next_yield in ready:
                yield ready.pop(next_yield)
                next_yield += 1
            while (
                next_index < len(sources)
                and len(pending) + len(ready) < window
            ):
                future = executor.submit(
                    _parse_one, runner, sources[next_index], max_file_mb
                )
                pending[future] = next_index
                next_index += 1
            if completed % 25 == 0 or completed == len(sources):
                print(
                    "[{}] completed {}/{}".format(
                        runner.config.profile, completed, len(sources)
                    ),
                    file=sys.stderr,
                )
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _stage_sources(
    sources: Sequence[SourceDocument],
    staging_root: Path,
    input_root: Path,
) -> List[SourceDocument]:
    """Copy verified inputs once so every profile parses identical bytes."""

    root = Path(input_root).resolve()
    destination_root = Path(staging_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=False)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError(
            "this platform cannot safely stage parser inputs without O_NOFOLLOW"
        )

    staged: List[SourceDocument] = []
    for source in sources:
        source_path = Path(source.path)
        try:
            source_path.resolve().relative_to(root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "input escaped its root before staging: {}".format(
                    source.relative_path
                )
            ) from exc

        flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        input_descriptor = os.open(str(source_path), flags)
        temporary_name: Optional[str] = None
        try:
            before = os.fstat(input_descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(
                    "input is not a regular file: {}".format(
                        source.relative_path
                    )
                )
            output_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".input.",
                suffix=".tmp",
                dir=str(destination_root),
            )
            digest = hashlib.sha256()
            copied = 0
            with os.fdopen(input_descriptor, "rb") as input_handle:
                input_descriptor = -1
                with os.fdopen(output_descriptor, "wb") as output_handle:
                    while True:
                        chunk = input_handle.read(1024 * 1024)
                        if not chunk:
                            break
                        copied += len(chunk)
                        digest.update(chunk)
                        output_handle.write(chunk)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                after = os.fstat(input_handle.fileno())

            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or copied != source.size_bytes
                or digest.hexdigest() != source.source_sha256
            ):
                raise RuntimeError(
                    "input changed after discovery: {}".format(
                        source.relative_path
                    )
                )

            suffix = source.extension
            if (
                not suffix
                or "/" in suffix
                or "\\" in suffix
                or len(suffix) > 32
            ):
                suffix = ".bin"
            destination = destination_root / (
                "{}{}".format(source.document_id, suffix)
            )
            os.replace(temporary_name, str(destination))
            temporary_name = None
            staged.append(replace(source, path=destination))
        finally:
            if input_descriptor >= 0:
                os.close(input_descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
    return staged


@contextmanager
def _run_claim(output_root: Path, run_id: str) -> Iterator[Path]:
    """Serialize publishers for one run ID without reserving the final path."""

    final_path = output_root / run_id
    if final_path.exists():
        raise FileExistsError("run already exists; refusing to overwrite: {}".format(final_path))
    lock_path = output_root / ".{}.lock".format(run_id)
    try:
        descriptor = os.open(
            str(lock_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        raise FileExistsError(
            "run is already being created (or has a stale lock): {}".format(
                lock_path
            )
        )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
            lock.write("{}\n".format(os.getpid()))
        if final_path.exists():
            raise FileExistsError(
                "run already exists; refusing to overwrite: {}".format(final_path)
            )
        yield final_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _validate_run_id(value: str) -> str:
    if not value or len(value) > 128:
        raise ValueError("run-id must contain between 1 and 128 characters")
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError("run-id must be one path-safe name")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(character not in allowed for character in value):
        raise ValueError("run-id may contain only ASCII letters, digits, '.', '_' and '-'")
    if value.startswith("."):
        raise ValueError("run-id must not begin with '.'")
    return value


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def command_prepare(args: argparse.Namespace) -> int:
    report = prepare(args.profile, Path(args.tools_dir), execute=args.execute)
    _json_print(report)
    return 1 if report.get("errors") else 0


def command_doctor(args: argparse.Namespace) -> int:
    report = doctor(args.profile, Path(args.tools_dir))
    _json_print(report)
    return 0 if report.get("ready") else 1


def command_run(args: argparse.Namespace) -> int:
    if not sys.flags.no_user_site:
        raise RuntimeError(
            "parser runs require Python user-site isolation; invoke the CLI "
            "normally so it can restart with -s"
        )
    _isolate_python_import_environment()
    if args.chunk_overlap >= args.chunk_chars:
        raise ValueError("--chunk-overlap must be smaller than --chunk-chars")
    if args.workers > MAX_WORKERS:
        raise ValueError(
            "--workers must be at most {} to bound parser concurrency".format(
                MAX_WORKERS
            )
        )

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    tools_dir = Path(args.tools_dir).expanduser().resolve()
    run_id = _validate_run_id(args.run_id or _default_run_id())
    profiles = _profile_names(args.profile)
    institutions = set(args.institution or ()) or None
    extensions = _normalize_extensions(args.extensions)
    source_manifest_sha256: Optional[str] = None
    selection_counts: Dict[str, int]

    runtime_report = doctor(args.profile, tools_dir)
    profile_runtime_reports = {
        profile: (
            runtime_report
            if len(profiles) == 1
            else doctor(profile, tools_dir)
        )
        for profile in profiles
    }
    if not runtime_report.get("ready") and not args.allow_missing:
        missing = ", ".join(runtime_report.get("missing_required") or ())
        raise RuntimeError(
            "required parser runtime is not ready ({}); run 'doctor' or pass "
            "--allow-missing to record unavailable attempts".format(
                missing or "unknown capabilities"
            )
        )
    if not runtime_report.get("ready"):
        print(
            "warning: continuing with missing runtime capabilities: {}".format(
                ", ".join(runtime_report.get("missing_required") or ())
            ),
            file=sys.stderr,
        )
    core_capability = (
        runtime_report.get("capabilities", {}).get("core_python", {})
    )
    if core_capability.get("available"):
        approved_environment = core_capability.get("environment_id")
        approved_python_version = str(
            core_capability.get("python_version") or ""
        )
        current_environment = os.path.abspath(sys.prefix)
        if (
            approved_environment
            and os.path.abspath(str(approved_environment))
            != current_environment
        ):
            raise RuntimeError(
                "doctor validated the core Python environment at {}, but "
                "this CLI is running in {}; invoke this script with the "
                "validated environment so in-process adapters match "
                "doctor".format(
                    approved_environment,
                    current_environment,
                )
            )
        approved_family = ".".join(
            approved_python_version.split(".")[:2]
        )
        current_family = "{}.{}".format(
            sys.version_info.major,
            sys.version_info.minor,
        )
        if approved_family and approved_family != current_family:
            raise RuntimeError(
                "doctor validated Python {}, but this CLI is running "
                "Python {}; invoke the approved interpreter".format(
                    approved_python_version,
                    current_family,
                )
            )

    if args.corpus_manifest:
        if institutions or extensions or args.limit is not None:
            raise ValueError(
                "--corpus-manifest is authoritative and cannot be combined "
                "with --institution, --extensions, or --limit"
            )
        selection = load_corpus_manifest(
            Path(args.corpus_manifest),
            input_root,
            profiles[0],
            repo_root=REPO_ROOT,
        )
        base_sources = list(selection.sources)
        files = [source.path for source in base_sources]
        source_manifest_sha256 = selection.source_manifest_sha256
        selection_counts = dict(selection.selection_counts)
    else:
        discovered_files = discover_input_files(
            input_root,
            institutions=institutions,
            extensions=extensions,
            excluded_roots=(output_root, tools_dir),
            limit=None,
        )
        files = (
            discovered_files
            if args.limit is None
            else discovered_files[: args.limit]
        )
        if not files:
            raise ValueError("no input files matched the selected filters")
        # Hash and identify each source once; profile is not part of document_id.
        base_sources = [
            build_source_document(
                path,
                input_root,
                profiles[0],
                repo_root=REPO_ROOT,
            )
            for path in files
        ]
        selection_counts = {
            "discovered_files": len(discovered_files),
            "selected_files": len(files),
        }

    print(
        "selected {} files; profiles={}; workers={}".format(
            len(files), ",".join(profiles), args.workers
        ),
        file=sys.stderr,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with _run_claim(output_root, run_id) as final_run_dir:
        temporary_run_dir = Path(
            tempfile.mkdtemp(
                prefix=".{}.".format(run_id),
                suffix=".tmp",
                dir=str(output_root),
            )
        )
        try:
            staging_root = temporary_run_dir / ".inputs"
            staged_sources = _stage_sources(
                base_sources,
                staging_root,
                input_root,
            )
            prepublish_verification: Dict[str, Dict[str, Any]] = {}
            for profile in profiles:
                profile_runtime = profile_runtime_reports[profile]
                profile_dir = temporary_run_dir / profile
                raw_dir = profile_dir / "raw"
                config = PipelineConfig(
                    profile=profile,
                    tools_dir=tools_dir,
                    raw_output_dir=raw_dir,
                    repo_root=REPO_ROOT,
                    runtime_report=profile_runtime,
                    expect_korean=args.expect_korean,
                    min_chars=args.min_chars,
                    enable_vl_review=args.enable_vl_review,
                    fast_timeout_seconds=args.fast_timeout_seconds,
                    tesseract_timeout_seconds=args.tesseract_timeout_seconds,
                    heavy_timeout_seconds=args.heavy_timeout_seconds,
                    max_blocks_per_document=args.max_blocks,
                )
                runner = PipelineRunner(config)
                sources = [
                    source
                    if source.profile == profile
                    else replace(source, profile=profile)
                    for source in staged_sources
                ]
                outcomes = run_sources(
                    runner,
                    sources,
                    workers=args.workers,
                    max_file_mb=args.max_file_mb,
                )
                try:
                    write_profile_run(
                        profile_dir,
                        outcomes,
                        config,
                        input_root,
                        run_id,
                        runtime_report=profile_runtime,
                        chunk_chars=args.chunk_chars,
                        chunk_overlap=args.chunk_overlap,
                        published_output_dir=final_run_dir / profile,
                        source_manifest_sha256=source_manifest_sha256,
                        selection_counts=selection_counts,
                    )
                finally:
                    close_outcomes = getattr(outcomes, "close", None)
                    if close_outcomes is not None:
                        close_outcomes()
                verification = verify_profile_run(profile_dir)
                prepublish_verification[profile] = verification
                if not verification.get("valid"):
                    raise RuntimeError(
                        "{} profile output failed verification: {}".format(
                            profile,
                            "; ".join(verification.get("errors") or ("unknown error",)),
                        )
                    )

            shutil.rmtree(str(staging_root))
            if final_run_dir.exists():
                raise FileExistsError(
                    "run appeared while parsing; refusing to overwrite: {}".format(
                        final_run_dir
                    )
                )
            os.rename(str(temporary_run_dir), str(final_run_dir))
        except BaseException:
            if temporary_run_dir.exists():
                shutil.rmtree(str(temporary_run_dir))
            raise

    verification: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Mapping[str, Any]] = {}
    valid = True
    for profile in profiles:
        profile_dir = final_run_dir / profile
        result = verify_profile_run(profile_dir)
        verification[profile] = result
        valid = valid and bool(result.get("valid"))
        summaries[profile] = json.loads(
            (profile_dir / "parse_summary.json").read_text(encoding="utf-8")
        )

    report = {
        "schema_version": 1,
        "run_id": run_id,
        "input": str(input_root),
        "output": str(final_run_dir),
        "file_count": len(files),
        "source_manifest_sha256": source_manifest_sha256,
        "selection_counts": selection_counts,
        "profiles": list(profiles),
        "summaries": summaries,
        "verification": verification,
        "valid": valid,
    }
    _json_print(report)
    return 0 if valid else 1


def _verification_targets(run_root: Path) -> List[Tuple[str, Path]]:
    if (run_root / "run_manifest.json").is_file():
        return [(run_root.name, run_root)]
    targets = [
        (child.name, child)
        for child in sorted(run_root.iterdir(), key=lambda item: item.name)
        if child.is_dir() and (child / "run_manifest.json").is_file()
    ]
    return targets


def command_verify_run(args: argparse.Namespace) -> int:
    run_root = Path(args.run).expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError("run directory does not exist: {}".format(run_root))
    targets = _verification_targets(run_root)
    if not targets:
        raise ValueError(
            "no profile run_manifest.json found under {}".format(run_root)
        )
    results: Dict[str, Dict[str, Any]] = {}
    for name, path in targets:
        results[name] = verify_profile_run(path)
    valid = all(result.get("valid") for result in results.values())
    report = {
        "schema_version": 1,
        "run": str(run_root),
        "valid": valid,
        "profiles": results,
    }
    _json_print(report)
    return 0 if valid else 1


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Inspect or download checksum-pinned local parser artifacts.",
    )
    prepare_parser.add_argument(
        "--profile", choices=PROFILE_CHOICES, default="all"
    )
    prepare_parser.add_argument(
        "--tools-dir", default=_default_path(".parser-tools")
    )
    prepare_parser.add_argument(
        "--execute",
        action="store_true",
        help="Download manifest artifacts. The default is a read-only preview.",
    )
    prepare_parser.set_defaults(handler=command_prepare)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Report parser runtime capabilities without changing files."
    )
    doctor_parser.add_argument(
        "--profile", choices=PROFILE_CHOICES, default="all"
    )
    doctor_parser.add_argument(
        "--tools-dir", default=_default_path(".parser-tools")
    )
    doctor_parser.set_defaults(handler=command_doctor)

    run_parser = subparsers.add_parser(
        "run", help="Parse inputs into one immutable, verified run directory."
    )
    run_parser.add_argument(
        "--profile", choices=PROFILE_CHOICES, default="cascade"
    )
    run_parser.add_argument(
        "--input",
        default=_default_path("src/data"),
        help="Root containing institution document folders.",
    )
    run_parser.add_argument(
        "--corpus-manifest",
        help=(
            "Authoritative UTF-8 JSONL inclusion list. Each "
            "input_relative_path is resolved below --input and pinned by "
            "sha256 and size_bytes."
        ),
    )
    run_parser.add_argument(
        "--output",
        default=_default_path("processed/runs"),
        help="Parent for immutable run directories.",
    )
    run_parser.add_argument(
        "--tools-dir", default=_default_path(".parser-tools")
    )
    run_parser.add_argument(
        "--run-id",
        help="Immutable run directory name; defaults to a UTC timestamp.",
    )
    run_parser.add_argument(
        "--institution",
        action="append",
        help="Top-level institution directory to include. May be repeated.",
    )
    run_parser.add_argument(
        "--extensions",
        nargs="+",
        help="Optional suffix allowlist, e.g. --extensions .pdf .hwp",
    )
    run_parser.add_argument(
        "--limit",
        type=_non_negative_int,
        help="Take the first N files after deterministic sorting.",
    )
    run_parser.add_argument(
        "--workers", type=_positive_int, default=DEFAULT_WORKERS
    )
    run_parser.add_argument(
        "--chunk-chars", type=_positive_int, default=1800
    )
    run_parser.add_argument(
        "--chunk-overlap", type=_non_negative_int, default=250
    )
    run_parser.add_argument("--min-chars", type=_non_negative_int, default=20)
    run_parser.add_argument(
        "--max-blocks",
        type=_positive_int,
        default=250000,
        help="Reject a document that expands beyond this many blocks.",
    )
    run_parser.add_argument(
        "--max-file-mb",
        type=_positive_float,
        default=250.0,
        help="Record larger files as deferred instead of parsing them.",
    )
    run_parser.add_argument(
        "--expect-korean",
        action="store_true",
        help="Enable the Korean-text quality gate.",
    )
    run_parser.add_argument(
        "--enable-vl-review",
        action="store_true",
        help="Run the optional cascade VL review without selecting its output.",
    )
    run_parser.add_argument(
        "--fast-timeout-seconds", type=_positive_int, default=120
    )
    run_parser.add_argument(
        "--tesseract-timeout-seconds", type=_positive_int, default=120
    )
    run_parser.add_argument(
        "--heavy-timeout-seconds", type=_positive_int, default=1800
    )
    run_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue and record unavailable attempts when doctor is not ready.",
    )
    run_parser.set_defaults(handler=command_run)

    verify_parser = subparsers.add_parser(
        "verify-run",
        help="Validate one profile directory or all profiles in a run.",
    )
    verify_parser.add_argument("--run", required=True)
    verify_parser.set_defaults(handler=command_verify_run)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if (
            argv is None
            and args.handler is command_run
            and (
                not sys.flags.no_user_site
                or bool(os.environ.get("PYTHONPATH"))
            )
        ):
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment["PYTHONNOUSERSITE"] = "1"
            os.execve(
                sys.executable,
                [
                    sys.executable,
                    "-s",
                    str(Path(__file__).resolve()),
                ]
                + sys.argv[1:],
                environment,
            )
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
