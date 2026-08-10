"""PDF adapters, page preflight, and deterministic table overlay."""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.document_parsing.core import Block, ParseResult, SourceDocument

from .base import (
    AdapterContext,
    clean_text,
    error_result,
    make_attempt,
    make_block,
    make_result,
    parser_name,
    parser_version,
    reindex_blocks,
    source_document_id,
    source_path,
    unavailable_result,
)


LIST_PREFIX = re.compile(r"^\s*(?:[-*•▪◦‣]|\d+[.)]|[가-힣A-Za-z][.)])\s+")


@dataclass(frozen=True)
class PagePreflight:
    page: int
    text_char_count: int
    word_count: int
    image_area_ratio: float
    has_tables: bool
    column_count: int
    reading_order_anomaly: bool

    @property
    def classification(self) -> str:
        return classify_pdf_page(self)


def classify_pdf_page(preflight: Any) -> str:
    """Classify a page with the agreed scan/table/complex/clean precedence."""

    value = _preflight_values(preflight)
    if value["text_char_count"] < 40 and (
        value["word_count"] < 10 or value["image_area_ratio"] >= 0.5
    ):
        return "scan"
    if value["has_tables"]:
        return "table"
    if value["column_count"] >= 2 or value["reading_order_anomaly"]:
        return "complex"
    return "clean"


def preflight_pdf_page(page: Any, page_number: Optional[int] = None) -> PagePreflight:
    text = clean_text(page.get_text("text") or "")
    try:
        words = page.get_text("words") or []
    except Exception:
        words = []
    blocks = _text_blocks(page)
    page_rect = getattr(page, "rect", None)
    page_width = float(getattr(page_rect, "width", 0.0) or 0.0)
    page_height = float(getattr(page_rect, "height", 0.0) or 0.0)
    page_area = page_width * page_height
    image_area = _image_area(page, page_rect)
    image_area_ratio = min(1.0, image_area / page_area) if page_area > 0 else 0.0
    has_tables = _page_has_tables(page)
    column_count = _column_count(blocks, page_width)
    anomaly = _reading_order_anomaly(blocks, page_height)
    number = page_number
    if number is None:
        number = int(getattr(page, "number", 0)) + 1
    return PagePreflight(
        page=number,
        text_char_count=len("".join(text.split())),
        word_count=len(words),
        image_area_ratio=image_area_ratio,
        has_tables=has_tables,
        column_count=column_count,
        reading_order_anomaly=anomaly,
    )


def parse_pymupdf(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    parser = parser_name(context, "pymupdf")
    started = time.monotonic()
    try:
        import fitz
    except ImportError:
        return unavailable_result(source, parser, "optional dependency 'PyMuPDF/fitz' is not installed")

    try:
        path = source_path(source)
        page_rows: List[Tuple[int, Dict[str, Any]]] = []
        sizes: List[float] = []
        preflight: List[Dict[str, Any]] = []
        with fitz.open(str(path)) as document:
            for page_number, page in enumerate(document, start=1):
                page_info = preflight_pdf_page(page, page_number)
                preflight.append(
                    {
                        "page": page_info.page,
                        "classification": page_info.classification,
                        "text_char_count": page_info.text_char_count,
                        "word_count": page_info.word_count,
                        "image_area_ratio": round(page_info.image_area_ratio, 6),
                        "has_tables": page_info.has_tables,
                        "column_count": page_info.column_count,
                        "reading_order_anomaly": page_info.reading_order_anomaly,
                    }
                )
                page_dict = page.get_text("dict") or {}
                for raw_block in page_dict.get("blocks", []):
                    if raw_block.get("type", 0) != 0:
                        continue
                    lines: List[str] = []
                    block_sizes: List[float] = []
                    for line in raw_block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(str(span.get("text", "")) for span in spans)
                        if line_text.strip():
                            lines.append(line_text)
                        for span in spans:
                            if str(span.get("text", "")).strip() and span.get("size"):
                                size = float(span["size"])
                                block_sizes.append(size)
                                sizes.append(size)
                    text = clean_text("\n".join(lines))
                    if text:
                        page_rows.append(
                            (
                                page_number,
                                {
                                    "text": text,
                                    "font_size": max(block_sizes) if block_sizes else 0.0,
                                    "bbox": raw_block.get("bbox"),
                                },
                            )
                        )

        body_size = statistics.median(sizes) if sizes else 0.0
        heading_sizes = sorted(
            {
                round(row["font_size"], 2)
                for _, row in page_rows
                if _is_heading(row["text"], row["font_size"], body_size)
            },
            reverse=True,
        )
        blocks: List[Block] = []
        headings: List[str] = []
        for page_number, row in page_rows:
            text = row["text"]
            block_type = "paragraph"
            section_path: Optional[Sequence[str]] = list(headings) or None
            if _is_heading(text, row["font_size"], body_size):
                block_type = "heading"
                level = heading_sizes.index(round(row["font_size"], 2)) + 1
                headings = headings[: level - 1]
                headings.append(text)
                section_path = list(headings)
            elif LIST_PREFIX.match(text):
                block_type = "list_item"
            blocks.append(
                make_block(
                    source,
                    parser,
                    len(blocks),
                    block_type,
                    text,
                    page=page_number,
                    section_path=section_path,
                )
            )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return make_result(
            source,
            blocks=blocks,
            attempts=[
                make_attempt(
                    parser,
                    "success",
                    elapsed_ms=elapsed_ms,
                    metadata={
                        "page_count": len(preflight),
                        "page_preflight": preflight,
                    },
                    parser_version=parser_version(context),
                )
            ],
        )
    except Exception as exc:
        return error_result(source, parser, exc, int((time.monotonic() - started) * 1000))


def parse_pypdf(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    parser = parser_name(context, "pypdf")
    started = time.monotonic()
    try:
        import pypdf
    except ImportError:
        return unavailable_result(source, parser, "optional dependency 'pypdf' is not installed")
    try:
        reader = pypdf.PdfReader(str(source_path(source)))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                pass
        blocks: List[Block] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            for paragraph in _paragraphs(text):
                blocks.append(
                    make_block(
                        source,
                        parser,
                        len(blocks),
                        "paragraph",
                        paragraph,
                        page=page_number,
                    )
                )
        return make_result(
            source,
            blocks=blocks,
            attempts=[
                make_attempt(
                    parser,
                    "success",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    metadata={"page_count": len(reader.pages)},
                    parser_version=parser_version(context),
                )
            ],
        )
    except Exception as exc:
        return error_result(source, parser, exc, int((time.monotonic() - started) * 1000))


def parse_pdfplumber_tables(
    source: SourceDocument,
    context: Optional[AdapterContext] = None,
) -> ParseResult:
    parser = parser_name(context, "pdfplumber")
    started = time.monotonic()
    try:
        import pdfplumber
    except ImportError:
        return unavailable_result(source, parser, "optional dependency 'pdfplumber' is not installed")
    try:
        blocks: List[Block] = []
        table_number = 0
        with pdfplumber.open(str(source_path(source))) as document:
            for page_number, page in enumerate(document.pages, start=1):
                tables: Iterable[Any]
                try:
                    tables = page.find_tables() or []
                except Exception:
                    tables = []
                for table in tables:
                    rows = table.extract() if hasattr(table, "extract") else table
                    normalized_rows = _normalize_table_rows(rows or [])
                    if not normalized_rows:
                        continue
                    table_id = "{}:{}:t{:04d}".format(
                        source_document_id(source),
                        _safe_parser_id(parser),
                        table_number,
                    )
                    table_number += 1
                    parent_text = "\n".join("\t".join(row) for row in normalized_rows)
                    blocks.append(
                        make_block(
                            source,
                            parser,
                            len(blocks),
                            "table",
                            parent_text,
                            page=page_number,
                            table_id=table_id,
                        )
                    )
                    for row_number, row in enumerate(normalized_rows):
                        for column_number, cell in enumerate(row):
                            blocks.append(
                                make_block(
                                    source,
                                    parser,
                                    len(blocks),
                                    "table_cell",
                                    cell,
                                    page=page_number,
                                    table_id=table_id,
                                    row=row_number,
                                    column=column_number,
                                )
                            )
        return make_result(
            source,
            blocks=blocks,
            attempts=[
                make_attempt(
                    parser,
                    "success",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    metadata={"table_count": table_number},
                    parser_version=parser_version(context),
                )
            ],
        )
    except Exception as exc:
        return error_result(source, parser, exc, int((time.monotonic() - started) * 1000))


def overlay_pdf_tables(
    source: SourceDocument,
    base_result: ParseResult,
    table_result: ParseResult,
    parser: str,
) -> ParseResult:
    """Merge table blocks and remove exact cell/row duplicates from body text."""

    base_blocks = list(getattr(base_result, "blocks", []) or [])
    table_blocks = list(getattr(table_result, "blocks", []) or [])
    base_table_pages = {
        getattr(block, "page", None)
        for block in base_blocks
        if getattr(block, "block_type", "") == "table"
    }
    if None in base_table_pages:
        table_blocks = []
    else:
        table_blocks = [
            block
            for block in table_blocks
            if getattr(block, "page", None) not in base_table_pages
        ]
    duplicates_by_page: Dict[Optional[int], set] = {}
    for block in table_blocks:
        if getattr(block, "block_type", "") not in {"table", "table_cell"}:
            continue
        page = getattr(block, "page", None)
        duplicate_set = duplicates_by_page.setdefault(page, set())
        text = _comparison_text(getattr(block, "text", ""))
        if text:
            duplicate_set.add(text)
        if getattr(block, "block_type", "") == "table":
            for row in str(getattr(block, "text", "")).splitlines():
                normalized = _comparison_text(row.replace("\t", " "))
                if normalized:
                    duplicate_set.add(normalized)

    kept: List[Block] = []
    for block in base_blocks:
        block_type = getattr(block, "block_type", "")
        page = getattr(block, "page", None)
        text = _comparison_text(getattr(block, "text", ""))
        if block_type in {"paragraph", "list_item"} and text in duplicates_by_page.get(page, set()):
            continue
        kept.append(block)

    merged: List[Block] = []
    pages = sorted(
        {getattr(block, "page", None) for block in kept + table_blocks},
        key=lambda value: (-1 if value is None else value),
    )
    for page in pages:
        merged.extend(block for block in kept if getattr(block, "page", None) == page)
        merged.extend(block for block in table_blocks if getattr(block, "page", None) == page)
    attempts = list(getattr(base_result, "attempts", []) or [])
    attempts.extend(list(getattr(table_result, "attempts", []) or []))
    reindexed: List[Block] = []
    for index, block in enumerate(merged):
        # Keep the real adapter on every canonical block.  A composite PDF can
        # legitimately contain PyMuPDF/Docling body blocks and pdfplumber table
        # blocks on the same page.
        reindexed.append(
            make_block(
                source,
                getattr(block, "parser", None) or parser,
                index,
                getattr(block, "block_type"),
                getattr(block, "text"),
                page=getattr(block, "page", None),
                section_path=getattr(block, "section_path", None),
                table_id=getattr(block, "table_id", None),
                row=getattr(block, "row", None),
                column=getattr(block, "column", None),
            )
        )
    return make_result(
        source,
        blocks=reindexed,
        attempts=attempts,
        raw_artifacts=list(getattr(base_result, "raw_artifacts", []) or [])
        + list(getattr(table_result, "raw_artifacts", []) or []),
    )


def _preflight_values(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        getter = value.get
    else:
        getter = lambda key, default=None: getattr(value, key, default)
    return {
        "text_char_count": int(getter("text_char_count", 0)),
        "word_count": int(getter("word_count", 0)),
        "image_area_ratio": float(getter("image_area_ratio", 0.0)),
        "has_tables": bool(getter("has_tables", False)),
        "column_count": int(getter("column_count", 1)),
        "reading_order_anomaly": bool(getter("reading_order_anomaly", False)),
    }


def _text_blocks(page: Any) -> List[Tuple[float, float, float, float, str]]:
    try:
        raw = page.get_text("blocks") or []
    except Exception:
        return []
    result: List[Tuple[float, float, float, float, str]] = []
    for item in raw:
        if len(item) < 5:
            continue
        text = str(item[4] or "").strip()
        if text:
            result.append((float(item[0]), float(item[1]), float(item[2]), float(item[3]), text))
    return result


def _image_area(page: Any, page_rect: Any) -> float:
    total = 0.0
    try:
        images = page.get_images(full=True) or []
    except Exception:
        return total
    seen = set()
    for image in images:
        if not image:
            continue
        xref = image[0]
        if xref in seen:
            continue
        seen.add(xref)
        try:
            rects = page.get_image_rects(xref) or []
        except Exception:
            rects = []
        for rect in rects:
            try:
                clipped = rect & page_rect if page_rect is not None else rect
                total += max(0.0, float(clipped.width)) * max(0.0, float(clipped.height))
            except Exception:
                continue
    return total


def _page_has_tables(page: Any) -> bool:
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return False
    try:
        result = finder()
        tables = getattr(result, "tables", result)
        return bool(tables)
    except Exception:
        return False


def _column_count(
    blocks: Sequence[Tuple[float, float, float, float, str]],
    page_width: float,
) -> int:
    if len(blocks) < 4 or page_width <= 0:
        return 1
    centers = sorted((left + right) / 2.0 for left, _, right, _, _ in blocks)
    threshold = max(40.0, page_width * 0.12)
    groups = 1
    for left, right in zip(centers, centers[1:]):
        if right - left > threshold:
            groups += 1
    return min(groups, 3)


def _reading_order_anomaly(
    blocks: Sequence[Tuple[float, float, float, float, str]],
    page_height: float,
) -> bool:
    if len(blocks) < 3:
        return False
    tolerance = max(30.0, page_height * 0.05)
    previous_y = blocks[0][1]
    for block in blocks[1:]:
        current_y = block[1]
        if current_y + tolerance < previous_y:
            return True
        previous_y = current_y
    return False


def _is_heading(text: str, font_size: float, body_size: float) -> bool:
    compact = " ".join(text.split())
    return bool(
        compact
        and len(compact) <= 160
        and body_size > 0
        and font_size >= body_size * 1.2
        and compact[-1:] not in {".", "。"}
    )


def _paragraphs(text: str) -> List[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"\n\s*\n+", cleaned)
    if len(parts) == 1:
        parts = cleaned.splitlines()
    return [clean_text(part) for part in parts if clean_text(part)]


def _normalize_table_rows(rows: Sequence[Sequence[Any]]) -> List[List[str]]:
    width = max((len(row) for row in rows), default=0)
    result: List[List[str]] = []
    for raw_row in rows:
        row = [clean_text("" if cell is None else str(cell)) for cell in raw_row]
        row.extend([""] * (width - len(row)))
        if any(row):
            result.append(row)
    return result


def _comparison_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_parser_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-") or "parser"
