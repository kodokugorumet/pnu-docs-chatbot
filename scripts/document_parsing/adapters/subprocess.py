"""Safe adapters for isolated Java, Rust, OCR, and model workers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.document_parsing.core import Block, ParseResult, SourceDocument

from .base import (
    AdapterContext,
    clean_text,
    executable_available,
    make_attempt,
    make_block,
    make_result,
    parser_name,
    reindex_blocks,
    source_document_id,
    source_path,
    unavailable_result,
)
from .sniff import sniff_source


COMMAND_ENV = {
    "java_hwp": "PARSER_JAVA_HWP_CMD",
    "unhwp": "PARSER_UNHWP_CMD",
    "tesseract": "PARSER_TESSERACT_CMD",
    "docling": "PARSER_DOCLING_CMD",
    "paddle": "PARSER_PADDLE_CMD",
    "tika": "PARSER_TIKA_CMD",
}
MAX_WORKER_STDOUT_BYTES = 64 * 1024 * 1024
MAX_WORKER_STDERR_BYTES = 8 * 1024 * 1024
MAX_WORKER_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_WORKER_JSON_FILES = 1_000


def parse_subprocess(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
    default_parser: str = "subprocess",
    command: Optional[Sequence[str]] = None,
    command_env: Optional[str] = None,
) -> ParseResult:
    """Execute one worker invocation and normalize its JSON/JSONL/text output.

    Worker JSON contract::

        {"blocks": [{"block_type": "paragraph", "text": "...", "page": 1,
                     "section_path": null, "table_id": null,
                     "row": null, "column": null}],
         "metadata": {}, "raw_artifacts": []}

    A JSON list, JSONL block objects, or tagged/plain text are accepted as
    compatibility fallbacks.
    """

    context = context or AdapterContext()
    parser = parser_name(context, default_parser)
    try:
        resolved = _resolve_command(context, command, command_env)
    except ValueError as exc:
        return unavailable_result(
            source,
            parser,
            "unsafe worker command: {}".format(exc),
        )
    if not resolved:
        name = command_env or parser
        return unavailable_result(
            source,
            parser,
            "no command configured for {}; set AdapterContext.command or {}".format(
                parser, name
            ),
        )

    path = source_path(source)
    raw_root = context.raw_output_dir
    temporary: Optional[tempfile.TemporaryDirectory] = None
    if raw_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="parser-worker-")
        output_dir = Path(temporary.name)
    else:
        attempt_key = hashlib.sha256(
            source.relative_path.encode("utf-8")
        ).hexdigest()[:12]
        output_dir = Path(raw_root) / "{}-{}-{}".format(
            _safe_id(source_document_id(source)),
            _safe_id(parser),
            attempt_key,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    replacements = {
        "input": str(path),
        "output": str(output_path),
        "output_dir": str(output_dir),
        "profile": str(context.options.get("profile", "")),
    }
    argv = [_replace_tokens(str(part), replacements) for part in resolved]
    if not any("{input}" in str(part) for part in resolved):
        argv.append(str(path))
    if not executable_available(argv):
        if temporary is not None:
            temporary.cleanup()
        return unavailable_result(
            source,
            parser,
            "worker executable is not available: {}".format(argv[0]),
        )

    environment = os.environ.copy()
    if context.options.get("managed_runtime"):
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
    if context.env:
        environment.update({str(key): str(value) for key, value in context.env.items()})
    started = time.monotonic()
    stdout_capture = tempfile.TemporaryFile()
    stderr_capture = tempfile.TemporaryFile()

    def _kill_process_group(process: "subprocess.Popen[Any]") -> None:
        """Kill the worker's whole process group, then reap the worker."""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    try:
        # start_new_session puts the worker (and any grandchildren it spawns,
        # e.g. Paddle's multiprocessing helpers) into its own process group so
        # a timeout can kill the whole tree.  With plain subprocess.run only
        # the direct child is signalled and orphaned compute processes keep
        # burning CPU for hours after their result has been discarded
        # (observed twice on the 516p manual, 2026-08-10).
        with subprocess.Popen(
            argv,
            stdout=stdout_capture,
            stderr=stderr_capture,
            env=environment,
            start_new_session=True,
        ) as process:
            try:
                process.wait(timeout=max(1, int(context.timeout_seconds)))
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                raise
            completed = subprocess.CompletedProcess(
                argv, process.returncode or 0
            )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        stderr = _read_capture(
            stderr_capture,
            MAX_WORKER_STDERR_BYTES,
            truncate=True,
        )
        stdout_capture.close()
        stderr_capture.close()
        if temporary is not None:
            temporary.cleanup()
        return make_result(
            source,
            attempts=[
                make_attempt(
                    parser,
                    "timeout",
                    reason="worker exceeded {} seconds".format(context.timeout_seconds),
                    elapsed_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                    stderr=stderr[-8000:] or None,
                    timeout=True,
                )
            ],
        )
    except OSError as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        stdout_capture.close()
        stderr_capture.close()
        if temporary is not None:
            temporary.cleanup()
        return make_result(
            source,
            attempts=[
                make_attempt(
                    parser,
                    "error",
                    reason="{}: {}".format(type(exc).__name__, exc),
                    elapsed_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                )
            ],
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)
    try:
        stdout = _read_capture(
            stdout_capture,
            MAX_WORKER_STDOUT_BYTES,
        )
        stderr = _read_capture(
            stderr_capture,
            MAX_WORKER_STDERR_BYTES,
        )
    except ValueError as exc:
        stdout_capture.close()
        stderr_capture.close()
        if temporary is not None:
            temporary.cleanup()
        return make_result(
            source,
            attempts=[
                make_attempt(
                    parser,
                    "error",
                    reason=str(exc),
                    elapsed_ms=elapsed_ms,
                    error_type="WorkerOutputLimitError",
                    exit_code=completed.returncode,
                )
            ],
        )
    finally:
        if not stdout_capture.closed:
            stdout_capture.close()
        if not stderr_capture.closed:
            stderr_capture.close()
    raw_artifacts: List[Any] = []
    if raw_root is not None:
        stdout_path = output_dir / "worker.stdout"
        stderr_path = output_dir / "worker.stderr"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        raw_artifacts.extend([str(stdout_path), str(stderr_path)])

    if completed.returncode != 0:
        if temporary is not None:
            temporary.cleanup()
        return make_result(
            source,
            attempts=[
                make_attempt(
                    parser,
                    "error",
                    reason="worker exited with status {}".format(completed.returncode),
                    elapsed_ms=elapsed_ms,
                    error_type="SubprocessError",
                    exit_code=completed.returncode,
                    stderr=stderr[-8000:],
                )
            ],
            raw_artifacts=raw_artifacts,
        )

    payload_text = stdout
    payload_artifact: Optional[Path] = None
    try:
        if (
            output_path.is_file()
            and not output_path.is_symlink()
            and output_path.stat().st_size
        ):
            payload_text = _read_text_file_limited(
                output_path,
                MAX_WORKER_PAYLOAD_BYTES,
            )
            payload_artifact = output_path
        else:
            candidates = list(
                islice(
                    (
                        item
                        for item in output_dir.rglob("*.json")
                        if not item.is_symlink()
                    ),
                    MAX_WORKER_JSON_FILES + 1,
                )
            )
            if len(candidates) > MAX_WORKER_JSON_FILES:
                raise ValueError(
                    "worker created more than {} JSON files".format(
                        MAX_WORKER_JSON_FILES
                    )
                )
            candidates = sorted(
                candidates,
                key=lambda item: (
                    0 if item.name == "content.json" else 1,
                    str(item),
                ),
            )
            if candidates:
                payload_artifact = candidates[0]
                payload_text = _read_text_file_limited(
                    payload_artifact,
                    MAX_WORKER_PAYLOAD_BYTES,
                )
    except (OSError, UnicodeError, ValueError) as exc:
        if temporary is not None:
            temporary.cleanup()
        return make_result(
            source,
            attempts=[
                make_attempt(
                    parser,
                    "error",
                    reason="invalid worker payload: {}: {}".format(
                        type(exc).__name__,
                        exc,
                    ),
                    elapsed_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                    exit_code=completed.returncode,
                    stderr=stderr[-8000:],
                )
            ],
            raw_artifacts=raw_artifacts,
        )
    if raw_root is not None and payload_artifact is not None:
        raw_artifacts.append(str(payload_artifact))

    try:
        blocks, worker_metadata, worker_artifacts = _normalize_worker_output(
            source, parser, payload_text
        )
        maximum_blocks = int(
            context.options.get("max_blocks", 250000)
        )
        if len(blocks) > maximum_blocks:
            raise ValueError(
                "worker produced {} blocks; limit is {}".format(
                    len(blocks),
                    maximum_blocks,
                )
            )
        if temporary is not None:
            worker_artifacts = [
                artifact
                for artifact in worker_artifacts
                if not _is_path_inside(artifact, Path(temporary.name))
            ]
        raw_artifacts.extend(worker_artifacts)
    except Exception as exc:
        if temporary is not None:
            temporary.cleanup()
        return make_result(
            source,
            attempts=[
                make_attempt(
                    parser,
                    "error",
                    reason="invalid worker output: {}: {}".format(type(exc).__name__, exc),
                    elapsed_ms=elapsed_ms,
                    error_type=type(exc).__name__,
                    exit_code=completed.returncode,
                    stderr=stderr[-8000:],
                )
            ],
            raw_artifacts=raw_artifacts,
        )

    metadata = {
        "command": [argv[0]],
        "block_count": len(blocks),
    }
    metadata.update(
        {
            key: context.options[key]
            for key in ("version", "model", "device", "runtime_override")
            if key in context.options
        }
    )
    metadata.update(worker_metadata)
    if temporary is not None:
        temporary.cleanup()
    return make_result(
        source,
        blocks=blocks,
        attempts=[
            make_attempt(
                parser,
                "success",
                elapsed_ms=elapsed_ms,
                metadata=metadata,
                exit_code=completed.returncode,
                stderr=stderr[-8000:] or None,
                parser_version=_optional_string(context.options.get("version")),
                model=_optional_string(context.options.get("model")),
                device=_optional_string(context.options.get("device")),
            )
        ],
        raw_artifacts=raw_artifacts,
    )


def _read_capture(
    handle: Any,
    maximum_bytes: int,
    truncate: bool = False,
) -> str:
    handle.flush()
    size = os.fstat(handle.fileno()).st_size
    if size > maximum_bytes and not truncate:
        raise ValueError(
            "worker output exceeds {} bytes".format(maximum_bytes)
        )
    handle.seek(max(0, size - maximum_bytes) if truncate else 0)
    return _decode_output(handle.read(maximum_bytes))


def _read_text_file_limited(path: Path, maximum_bytes: int) -> str:
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError(
            "worker payload exceeds {} bytes".format(maximum_bytes)
        )
    return path.read_text(encoding="utf-8", errors="replace")


def parse_java_hwp(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    format_name = sniff_source(source_path(source)).format
    default = "java-hwpxlib" if format_name == "hwpx" else "java-hwplib"
    return parse_subprocess(
        source,
        context,
        default_parser=default,
        command_env=COMMAND_ENV["java_hwp"],
    )


def parse_unhwp(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    default = (
        "unhwp",
        "convert",
        "{input}",
        "-o",
        "{output_dir}",
        "--all",
        "--cleanup",
        "none",
    )
    return parse_subprocess(
        source,
        context,
        default_parser="unhwp",
        command=default,
        command_env=COMMAND_ENV["unhwp"],
    )


def parse_tesseract(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    if sniff_source(source_path(source)).format == "pdf":
        return _parse_tesseract_pdf(source, context)
    return _parse_tesseract_input(source, context)


def _parse_tesseract_input(
    source: SourceDocument,
    context: Optional[AdapterContext],
) -> ParseResult:
    default = ("tesseract", "{input}", "stdout", "-l", "kor+eng")
    return parse_subprocess(
        source,
        context,
        default_parser="tesseract-kor-eng",
        command=default,
        command_env=COMMAND_ENV["tesseract"],
    )


def _parse_tesseract_pdf(
    source: SourceDocument,
    context: Optional[AdapterContext],
) -> ParseResult:
    parser = parser_name(context, "tesseract-kor-eng")
    try:
        import fitz
    except ImportError:
        return unavailable_result(
            source,
            parser,
            "optional dependency 'PyMuPDF/fitz' is required to render PDF pages for Tesseract",
        )
    context = context or AdapterContext(parser=parser)
    requested_page = _optional_int(context.options.get("page"))
    blocks: List[Block] = []
    attempts = []
    artifacts: List[Any] = []
    try:
        with tempfile.TemporaryDirectory(prefix="tesseract-pages-") as directory:
            with fitz.open(str(source_path(source))) as document:
                page_numbers = (
                    [requested_page]
                    if requested_page is not None
                    else list(range(1, document.page_count + 1))
                )
                for page_number in page_numbers:
                    if page_number < 1 or page_number > document.page_count:
                        raise ValueError(
                            "requested OCR page {} is outside 1..{}".format(
                                page_number, document.page_count
                            )
                        )
                    page = document[page_number - 1]
                    image_path = Path(directory) / "page-{:06d}.png".format(page_number)
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(300.0 / 72.0, 300.0 / 72.0))
                    pixmap.save(str(image_path))
                    # MuPDF caches decoded page images.  On image-only PDFs this
                    # can retain hundreds of MiB per page until process exit,
                    # even after the PDF is closed.  Release the page objects and
                    # purge the native store before invoking Tesseract so a
                    # multi-page scan stays bounded to roughly one rendered page.
                    del pixmap
                    del page
                    tools = getattr(fitz, "TOOLS", None)
                    if tools is not None:
                        shrink = getattr(tools, "store_shrink", None)
                        if callable(shrink):
                            shrink(100)
                    image_source = replace(
                        source,
                        path=image_path,
                        relative_path="{}#page={}".format(source.relative_path, page_number),
                    )
                    result = _parse_tesseract_input(image_source, context)
                    for attempt in result.attempts:
                        metadata = dict(attempt.metadata)
                        metadata["page"] = page_number
                        attempts.append(replace(attempt, metadata=metadata))
                    artifacts.extend(result.raw_artifacts)
                    for block in result.blocks:
                        blocks.append(
                            make_block(
                                source,
                                parser,
                                len(blocks),
                                str(block.block_type),
                                str(block.text),
                                page=page_number,
                                section_path=block.section_path,
                                table_id=block.table_id,
                                row=block.row,
                                column=block.column,
                            )
                        )
            tools = getattr(fitz, "TOOLS", None)
            if tools is not None:
                empty = getattr(tools, "store_empty", None)
                if callable(empty):
                    empty()
        return make_result(
            source,
            blocks=reindex_blocks(source, parser, blocks),
            attempts=attempts,
            raw_artifacts=artifacts,
        )
    except Exception as exc:
        return make_result(
            source,
            blocks=reindex_blocks(source, parser, blocks),
            attempts=attempts
            + [
                make_attempt(
                    parser,
                    "error",
                    reason="{}: {}".format(type(exc).__name__, exc),
                    error_type=type(exc).__name__,
                )
            ],
            raw_artifacts=artifacts,
        )


def parse_docling(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    return parse_subprocess(
        source,
        context,
        default_parser="docling-standard",
        command_env=COMMAND_ENV["docling"],
    )


def parse_paddle(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    return parse_subprocess(
        source,
        context,
        default_parser="pp-structurev3-pp-ocrv5-korean",
        command_env=COMMAND_ENV["paddle"],
    )


def parse_tika(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    tika_jar = os.environ.get("PARSER_TIKA_JAR")
    default: Optional[Sequence[str]] = None
    if tika_jar:
        default = ("java", "-jar", tika_jar, "--text", "{input}")
    return parse_subprocess(
        source,
        context,
        default_parser="tika",
        command=default,
        command_env=COMMAND_ENV["tika"],
    )


def _resolve_command(
    context: AdapterContext,
    default: Optional[Sequence[str]],
    command_env: Optional[str],
) -> Optional[List[str]]:
    managed = bool(context.options.get("managed_runtime"))
    allow_custom = bool(context.options.get("allow_custom_command"))
    value: Any = context.command
    if not value and (not managed or allow_custom):
        value = context.options.get("command")
    if not value and command_env and (not managed or allow_custom):
        if context.env and command_env in context.env:
            value = context.env[command_env]
        else:
            value = os.environ.get(command_env)
    if not value and not managed:
        value = default
    if isinstance(value, str):
        parts = shlex.split(value)
        value = parts
    if value:
        parts = [str(part) for part in value]
        if parts and Path(parts[0]).name.lower() in {
            "sh",
            "bash",
            "zsh",
            "dash",
            "ksh",
            "cmd",
            "cmd.exe",
            "powershell",
            "powershell.exe",
            "pwsh",
            "pwsh.exe",
        }:
            raise ValueError("shell launchers are not accepted")
        placeholders = (
            "{input}",
            "{output}",
            "{output_dir}",
            "{profile}",
        )
        for part in parts:
            for placeholder in placeholders:
                if placeholder in part and part != placeholder:
                    raise ValueError(
                        "{} must be a standalone argv item".format(
                            placeholder
                        )
                    )
        return parts
    return None


def _replace_tokens(value: str, replacements: Mapping[str, str]) -> str:
    for key, replacement in replacements.items():
        value = value.replace("{" + key + "}", replacement)
    return value


def _normalize_worker_output(
    source: SourceDocument,
    parser: str,
    payload_text: str,
) -> Tuple[List[Block], Dict[str, Any], List[Any]]:
    value = payload_text.strip()
    if not value:
        return [], {}, []
    parsed: Any
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = _parse_jsonl(value)
        if parsed is None:
            return _tagged_text_blocks(source, parser, value), {"output_format": "text"}, []

    metadata: Dict[str, Any] = {}
    raw_artifacts: List[Any] = []
    if isinstance(parsed, Mapping):
        metadata = dict(parsed.get("metadata") or {})
        raw_artifacts = list(parsed.get("raw_artifacts") or [])
        raw_blocks = parsed.get("blocks")
        if raw_blocks is None and isinstance(parsed.get("sections"), list):
            raw_blocks = _unhwp_raw_blocks(parsed)
            metadata["output_format"] = "unhwp_raw_content"
        if raw_blocks is None and "text" in parsed:
            return (
                _tagged_text_blocks(source, parser, str(parsed["text"])),
                metadata,
                raw_artifacts,
            )
        if raw_blocks is None:
            raw_blocks = [parsed]
    elif isinstance(parsed, list):
        raw_blocks = parsed
    else:
        return _tagged_text_blocks(source, parser, str(parsed)), metadata, raw_artifacts
    return (
        _canonicalize_worker_blocks(source, parser, raw_blocks or []),
        metadata,
        raw_artifacts,
    )


def _unhwp_raw_blocks(document: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Translate unhwp RawContent's externally-tagged Rust model."""

    result: List[Mapping[str, Any]] = []
    headings: List[str] = []
    table_number = 0
    for section in document.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        for raw_block in section.get("content") or []:
            if not isinstance(raw_block, Mapping):
                continue
            paragraph = raw_block.get("Paragraph")
            if paragraph is None:
                paragraph = raw_block.get("paragraph")
            table = raw_block.get("Table")
            if table is None:
                table = raw_block.get("table")
            if isinstance(paragraph, Mapping):
                text = _unhwp_paragraph_text(paragraph)
                if not text:
                    continue
                style = paragraph.get("style")
                style = style if isinstance(style, Mapping) else {}
                heading_level = _first_int(
                    style,
                    ("heading_level", "outline_level", "headingLevel", "outlineLevel"),
                )
                list_value = (
                    style.get("list_type")
                    or style.get("listType")
                    or style.get("list_style")
                    or style.get("numbering")
                    or style.get("bullet")
                )
                is_list = list_value not in (
                    None,
                    False,
                    "",
                    "None",
                    "none",
                ) or bool(
                    re.match(
                        r"^\s*(?:[-*•◦▪▫‣⁃]|\d+[.)]|[가-힣][.)])\s+",
                        text,
                    )
                )
                if is_list:
                    block_type = "list_item"
                elif heading_level is not None and heading_level > 0:
                    level = max(1, min(6, heading_level))
                    headings = headings[: level - 1]
                    headings.append(text)
                    block_type = "heading"
                else:
                    block_type = "paragraph"
                result.append(
                    {
                        "block_type": block_type,
                        "text": text,
                        "section_path": list(headings) or None,
                    }
                )
            elif isinstance(table, Mapping):
                rows = _unhwp_table_rows(table)
                if not rows:
                    continue
                table_key = "unhwp:t{:04d}".format(table_number)
                table_number += 1
                result.append(
                    {
                        "block_type": "table",
                        "text": "\n".join("\t".join(row) for row in rows),
                        "section_path": list(headings) or None,
                        "table_id": table_key,
                    }
                )
                for row_number, row in enumerate(rows):
                    for column_number, cell in enumerate(row):
                        result.append(
                            {
                                "block_type": "table_cell",
                                "text": cell,
                                "section_path": list(headings) or None,
                                "table_id": table_key,
                                "row": row_number,
                                "column": column_number,
                            }
                        )
    return result


def _unhwp_paragraph_text(paragraph: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for item in paragraph.get("content") or []:
        if isinstance(item, str):
            if item.lower() == "linebreak":
                parts.append("\n")
            continue
        if not isinstance(item, Mapping):
            continue
        text_run = item.get("Text")
        if text_run is None:
            text_run = item.get("text")
        if isinstance(text_run, Mapping):
            parts.append(str(text_run.get("text") or ""))
        elif isinstance(text_run, str):
            parts.append(text_run)
        link = item.get("Link")
        if isinstance(link, Mapping):
            parts.append(str(link.get("text") or ""))
        footnote = item.get("Footnote")
        if isinstance(footnote, str):
            parts.append(footnote)
        if "LineBreak" in item:
            parts.append("\n")
    return clean_text("".join(parts))


def _unhwp_table_rows(table: Mapping[str, Any]) -> List[List[str]]:
    grid: Dict[Tuple[int, int], str] = {}
    maximum_column = -1
    rows = table.get("rows") or []
    for row_number, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            continue
        column_number = 0
        for cell in raw_row.get("cells") or []:
            while (row_number, column_number) in grid:
                column_number += 1
            if not isinstance(cell, Mapping):
                cell = {}
            paragraphs = cell.get("content") or []
            text = "\n".join(
                _unhwp_paragraph_text(paragraph)
                for paragraph in paragraphs
                if isinstance(paragraph, Mapping)
            )
            row_span = max(1, _coerce_positive_int(cell.get("rowspan"), 1))
            column_span = max(1, _coerce_positive_int(cell.get("colspan"), 1))
            grid[(row_number, column_number)] = clean_text(text)
            for span_row in range(row_number, row_number + row_span):
                for span_column in range(column_number, column_number + column_span):
                    grid.setdefault((span_row, span_column), "")
            maximum_column = max(maximum_column, column_number + column_span - 1)
            column_number += column_span
    maximum_row = max((coordinate[0] for coordinate in grid), default=-1)
    if maximum_row < 0 or maximum_column < 0:
        return []
    return [
        [grid.get((row, column), "") for column in range(maximum_column + 1)]
        for row in range(maximum_row + 1)
    ]


def _first_int(value: Mapping[str, Any], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        if key in value and value[key] is not None:
            try:
                return int(value[key])
            except (TypeError, ValueError):
                continue
    return None


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _parse_jsonl(value: str) -> Optional[List[Any]]:
    result: List[Any] = []
    for line in value.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(item, Mapping) and item.get("type") == "block":
            item = item.get("block", item)
        result.append(item)
    return result or None


def _canonicalize_worker_blocks(
    source: SourceDocument,
    parser: str,
    raw_blocks: Iterable[Any],
) -> List[Block]:
    allowed_types = {"heading", "paragraph", "list_item", "table", "table_cell", "caption"}
    result: List[Block] = []
    table_number = 0
    table_ids: Dict[str, str] = {}
    active_anonymous_table: Optional[str] = None
    for raw in raw_blocks:
        if isinstance(raw, str):
            block_type = "paragraph"
            text = raw
            values: Mapping[str, Any] = {}
        elif isinstance(raw, Mapping):
            values = raw
            block_type = str(values.get("block_type") or values.get("type") or "paragraph")
            text = values.get("text", "")
        else:
            continue
        if block_type not in allowed_types:
            block_type = "paragraph"
        text = clean_text(
            str(text or ""),
            preserve_tabs=block_type == "table",
        )
        # Empty layout-only tables still need their canonical parent.  Their
        # cells may all be blank (for example, a form spacer or image grid),
        # but dropping the parent leaves otherwise valid table_cell blocks
        # with a dangling table_id.
        if not text and block_type not in {"table", "table_cell"}:
            continue
        incoming_table_id = values.get("table_id")
        table_id: Optional[str] = None
        if block_type in {"table", "table_cell"}:
            if incoming_table_id:
                key = str(incoming_table_id)
            elif block_type == "table":
                key = "anonymous-table-{}".format(table_number)
                active_anonymous_table = key
            elif active_anonymous_table is not None:
                key = active_anonymous_table
            else:
                key = "anonymous-table-{}".format(table_number)
                active_anonymous_table = key
            if key not in table_ids:
                table_ids[key] = "{}:{}:t{:04d}".format(
                    source_document_id(source), _safe_id(parser), table_number
                )
                table_number += 1
            table_id = table_ids[key]
        section_path = values.get("section_path")
        if isinstance(section_path, str):
            section_path = [section_path]
        result.append(
            make_block(
                source,
                parser,
                len(result),
                block_type,
                text,
                page=_optional_int(values.get("page")),
                section_path=section_path,
                table_id=table_id,
                row=_optional_int(values.get("row")),
                column=_optional_int(values.get("column")),
            )
        )
    return result


def _tagged_text_blocks(
    source: SourceDocument,
    parser: str,
    value: str,
) -> List[Block]:
    result: List[Block] = []
    current_page: Optional[int] = None
    current_type = "paragraph"
    buffer: List[str] = []
    tag_pattern = re.compile(
        r"^\[(heading|paragraph|list_item|caption|page)(?:\s+(\d+))?\]\s*(.*)$",
        re.IGNORECASE,
    )

    def flush() -> None:
        if not buffer:
            return
        text = clean_text("\n".join(buffer))
        del buffer[:]
        if text:
            result.append(
                make_block(
                    source,
                    parser,
                    len(result),
                    current_type,
                    text,
                    page=current_page,
                )
            )

    for line in value.splitlines():
        match = tag_pattern.match(line.strip())
        if match:
            flush()
            tag = match.group(1).lower()
            if tag == "page":
                current_page = int(match.group(2) or match.group(3) or 0) or None
                current_type = "paragraph"
            else:
                current_type = tag
                if match.group(3):
                    buffer.append(match.group(3))
            continue
        if not line.strip():
            flush()
        else:
            buffer.append(line)
    flush()
    return result


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _decode_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-") or "parser"


def _is_path_inside(value: Any, parent: Path) -> bool:
    if not isinstance(value, (str, os.PathLike)):
        return False
    try:
        Path(value).resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True
