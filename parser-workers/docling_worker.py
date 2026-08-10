#!/usr/bin/env python3
"""One-shot Docling worker that emits the parser adapter JSON contract.

The worker intentionally lives outside the core environment.  It can be
invoked through ``PARSER_DOCLING_CMD`` or an adapter command template:

    .parser-tools/venvs/docling/bin/python parser-workers/docling_worker.py \
      --input input.pdf --output result.json --profile challenger
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


TEXT_LABELS = {
    "text": "paragraph",
    "paragraph": "paragraph",
    "section_header": "heading",
    "title": "heading",
    "list_item": "list_item",
    "caption": "caption",
}


def _value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    raw = getattr(value, "value", value)
    return raw if raw is not None else default


def _page(item: Any) -> int | None:
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    page_no = getattr(provenance[0], "page_no", None)
    try:
        return int(page_no) if page_no is not None else None
    except (TypeError, ValueError):
        return None


def _section_path(stack: list[str]) -> list[str] | None:
    return list(stack) if stack else None


def _table_blocks(item: Any, document: Any, table_index: int) -> list[dict[str, Any]]:
    table_id = f"worker:t{table_index:04d}"
    rows: list[list[str]] = []
    try:
        dataframe = item.export_to_dataframe(doc=document)
        rows = [
            ["" if value is None else str(value).strip() for value in row]
            for row in dataframe.itertuples(index=False, name=None)
        ]
        columns = [str(value).strip() for value in dataframe.columns]
        if columns and not all(value.isdigit() for value in columns):
            rows.insert(0, columns)
    except Exception:
        data = getattr(item, "data", None)
        grid = getattr(data, "grid", None) if data is not None else None
        if grid:
            for row in grid:
                rows.append(
                    [
                        str(getattr(cell, "text", "") or "").strip()
                        for cell in row
                    ]
                )

    if not rows:
        fallback = str(getattr(item, "text", "") or "").strip()
        rows = [[fallback]] if fallback else []

    parent_text = "\n".join("\t".join(cell.replace("\t", " ") for cell in row) for row in rows)
    blocks: list[dict[str, Any]] = [
        {
            "block_type": "table",
            "text": parent_text,
            "page": _page(item),
            "section_path": None,
            "table_id": table_id,
            "row": None,
            "column": None,
        }
    ]
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            blocks.append(
                {
                    "block_type": "table_cell",
                    "text": cell,
                    "page": _page(item),
                    "section_path": None,
                    "table_id": table_id,
                    "row": row_index,
                    "column": column_index,
                }
            )
    return blocks


def convert_document(document: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    sections: list[str] = []
    table_index = 0

    iterator: Iterable[Any]
    try:
        iterator = document.iterate_items()
    except Exception:
        text = document.export_to_text()
        return [
            {
                "block_type": "paragraph",
                "text": part.strip(),
                "page": None,
                "section_path": None,
                "table_id": None,
                "row": None,
                "column": None,
            }
            for part in str(text).split("\n\n")
            if part.strip()
        ]

    for entry in iterator:
        item = entry[0] if isinstance(entry, tuple) else entry
        class_name = type(item).__name__.lower()
        label = str(_value(getattr(item, "label", ""), "")).lower()
        if "table" in class_name or label == "table":
            table_index += 1
            table_blocks = _table_blocks(item, document, table_index)
            for block in table_blocks:
                block["section_path"] = _section_path(sections)
            blocks.extend(table_blocks)
            continue

        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        block_type = TEXT_LABELS.get(label, "paragraph")
        if "sectionheader" in class_name or "title" in class_name:
            block_type = "heading"
        elif "listitem" in class_name:
            block_type = "list_item"
        elif "caption" in class_name:
            block_type = "caption"

        if block_type == "heading":
            level = getattr(item, "level", None)
            try:
                level_int = max(1, int(level or 1))
            except (TypeError, ValueError):
                level_int = 1
            sections[:] = sections[: level_int - 1]
            sections.append(text)

        blocks.append(
            {
                "block_type": block_type,
                "text": text,
                "page": _page(item),
                "section_path": _section_path(sections),
                "table_id": None,
                "row": None,
                "column": None,
            }
        )
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="challenger")
    parser.add_argument("--page", type=int)
    parser.add_argument("--ocr", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    artifacts_path = os.environ.get("DOCLING_ARTIFACTS_PATH")
    if not artifacts_path:
        raise RuntimeError(
            "DOCLING_ARTIFACTS_PATH is required; runtime model downloads are disabled"
        )
    artifact_root = Path(artifacts_path).resolve()
    if not artifact_root.is_dir():
        raise RuntimeError(
            "Docling artifact directory does not exist: {}".format(
                artifact_root
            )
        )

    pipeline_options = PdfPipelineOptions(
        artifacts_path=artifact_root
    )
    pipeline_options.do_ocr = bool(args.ocr)
    pipeline_options.do_table_structure = True
    pipeline_options.enable_remote_services = False
    pipeline_options.allow_external_plugins = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    page_range = (args.page, args.page) if args.page else None
    kwargs = {"page_range": page_range} if page_range else {}
    result = converter.convert(args.input, **kwargs)
    payload = {
        "blocks": convert_document(result.document),
        "metadata": {
            "worker": "docling",
            "profile": args.profile,
            "page_range": list(page_range) if page_range else None,
            "artifacts_path": str(artifact_root),
            "network_model_downloads": False,
        },
        "raw_artifacts": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
