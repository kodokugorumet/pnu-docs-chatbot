"""Shared adapter types and result-construction helpers.

Adapters deliberately have a very small contract: they accept a
``SourceDocument`` plus an ``AdapterContext`` and always return a ``ParseResult``.
Missing optional dependencies and external-process failures are represented by an
``Attempt`` instead of leaking an exception into the pipeline.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from scripts.document_parsing.core import Attempt, Block, ParseResult, SourceDocument


@dataclass(frozen=True)
class AdapterContext:
    """Runtime controls shared by in-process and subprocess adapters."""

    parser: str = ""
    timeout_seconds: int = 120
    env: Optional[Mapping[str, str]] = None
    raw_output_dir: Optional[Path] = None
    command: Optional[Sequence[str]] = None
    options: Mapping[str, Any] = field(default_factory=dict)


def source_path(source: SourceDocument) -> Path:
    """Return the concrete input path while tolerating compatible source models."""

    value = getattr(source, "path", None)
    if value is None:
        value = getattr(source, "source_path", None)
    if value is None:
        raise ValueError("SourceDocument does not contain path/source_path")
    return Path(value)


def source_document_id(source: SourceDocument) -> str:
    value = getattr(source, "document_id", None)
    if value is None:
        value = getattr(source, "doc_id", None)
    if not value:
        raise ValueError("SourceDocument does not contain document_id/doc_id")
    return str(value)


def parser_name(context: Optional[AdapterContext], default: str) -> str:
    if context is not None and context.parser:
        return context.parser
    return default


def parser_version(context: Optional[AdapterContext]) -> Optional[str]:
    if context is None:
        return None
    value = context.options.get("version")
    return str(value) if value else None


def executable_available(command: Optional[Sequence[str]]) -> bool:
    if not command:
        return False
    executable = str(command[0])
    if os.path.sep in executable:
        path = Path(executable).expanduser()
        return path.is_file() and os.access(str(path), os.X_OK)
    return shutil.which(executable) is not None


def adapter_available(
    name: str,
    context: Optional[AdapterContext] = None,
) -> bool:
    """Check an adapter capability without importing heavy packages eagerly."""

    import importlib.util

    normalized = name.lower().replace("-", "_")
    module_names = {
        "pymupdf": ("fitz",),
        "pdfplumber": ("pdfplumber",),
        "pypdf": ("pypdf",),
        "native_hwp": ("olefile",),
        "hwp": ("olefile",),
        "docx": ("docx",),
        "xlsx": ("openpyxl",),
        "xls": ("xlrd",),
    }
    if normalized in module_names:
        return all(importlib.util.find_spec(module) is not None for module in module_names[normalized])
    if normalized in {"native_hwpx", "hwpx", "native_html", "html", "pptx"}:
        return True
    if context is not None and context.command:
        return executable_available(context.command)
    return False


def _constructor_kwargs(cls: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    """Filter values for dataclass evolution without hiding constructor errors."""

    if dataclasses.is_dataclass(cls):
        accepted = {item.name for item in dataclasses.fields(cls) if item.init}
        return {key: value for key, value in values.items() if key in accepted}
    return dict(values)


def make_attempt(
    parser: str,
    status: str,
    reason: str = "",
    elapsed_ms: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
    error_type: Optional[str] = None,
    exit_code: Optional[int] = None,
    stderr: Optional[str] = None,
    timeout: bool = False,
    parser_version: Optional[str] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    fallback_reason: Optional[str] = None,
    peak_memory_bytes: Optional[int] = None,
) -> Attempt:
    values = {
        "parser": parser,
        "status": status,
        "reason": reason,
        "elapsed_ms": elapsed_ms,
        "metadata": dict(metadata or {}),
        "error_type": error_type,
        "exit_code": exit_code,
        "stderr": stderr,
        "timeout": timeout,
        "parser_version": parser_version,
        "model": model,
        "device": device,
        "fallback_reason": fallback_reason,
        "peak_memory_bytes": peak_memory_bytes,
    }
    return Attempt(**_constructor_kwargs(Attempt, values))


def make_block(
    source: SourceDocument,
    parser: str,
    reading_order: int,
    block_type: str,
    text: str,
    page: Optional[int] = None,
    section_path: Optional[Sequence[str]] = None,
    table_id: Optional[str] = None,
    row: Optional[int] = None,
    column: Optional[int] = None,
) -> Block:
    document_id = source_document_id(source)
    if block_type == "table":
        normalized_text = "\n".join(
            "\t".join(clean_text(cell) for cell in line.split("\t"))
            for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        ).strip()
    else:
        normalized_text = clean_text(text)
    values = {
        "document_id": document_id,
        "block_id": "{}:{}:b{:06d}".format(document_id, _id_part(parser), reading_order),
        "block_type": block_type,
        "text": normalized_text,
        "reading_order": reading_order,
        "parser": parser,
        "page": page,
        "section_path": list(section_path) if section_path else None,
        "table_id": table_id,
        "row": row,
        "column": column,
    }
    return Block(**_constructor_kwargs(Block, values))


def make_result(
    source: SourceDocument,
    blocks: Optional[Sequence[Block]] = None,
    attempts: Optional[Sequence[Attempt]] = None,
    raw_artifacts: Optional[Sequence[Any]] = None,
) -> ParseResult:
    values = {
        "document": source,
        "source": source,
        "blocks": list(blocks or []),
        "attempts": list(attempts or []),
        "raw_artifacts": list(raw_artifacts or []),
    }
    return ParseResult(**_constructor_kwargs(ParseResult, values))


def unavailable_result(
    source: SourceDocument,
    parser: str,
    reason: str,
) -> ParseResult:
    return make_result(
        source,
        attempts=[make_attempt(parser=parser, status="unavailable", reason=reason)],
    )


def error_result(
    source: SourceDocument,
    parser: str,
    exc: BaseException,
    elapsed_ms: int = 0,
) -> ParseResult:
    return make_result(
        source,
        attempts=[
            make_attempt(
                parser=parser,
                status="error",
                reason="{}: {}".format(type(exc).__name__, exc),
                elapsed_ms=elapsed_ms,
                error_type=type(exc).__name__,
            )
        ],
    )


def reindex_blocks(
    source: SourceDocument,
    parser: str,
    blocks: Sequence[Block],
) -> List[Block]:
    """Return canonical blocks with contiguous order and deterministic IDs."""

    result: List[Block] = []
    table_ids: Dict[str, str] = {}
    for index, block in enumerate(blocks):
        table_id = getattr(block, "table_id", None)
        if table_id is not None and table_id not in table_ids:
            table_ids[table_id] = "{}:{}:t{:04d}".format(
                source_document_id(source),
                _id_part(parser),
                len(table_ids),
            )
        result.append(
            make_block(
                source=source,
                parser=parser,
                reading_order=index,
                block_type=str(getattr(block, "block_type")),
                text=str(getattr(block, "text")),
                page=getattr(block, "page", None),
                section_path=getattr(block, "section_path", None),
                table_id=table_ids.get(table_id),
                row=getattr(block, "row", None),
                column=getattr(block, "column", None),
            )
        )
    return result


def clean_text(value: str, preserve_tabs: bool = False) -> str:
    import re

    value = value.replace("\x00", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    if preserve_tabs:
        value = re.sub(r" +", " ", value)
        value = re.sub(r" *\t *", "\t", value)
    else:
        value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def _id_part(value: str) -> str:
    import re

    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")
    return cleaned or "parser"
