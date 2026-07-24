#!/usr/bin/env python3
"""One-shot PP-StructureV3 worker for scanned documents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")


def _markdown_blocks(markdown: str, page: int | None) -> list[dict[str, Any]]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []
    table_index = 0

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = "\n".join(paragraph).strip()
        paragraph.clear()
        if not text:
            return
        block_type = "heading" if text.startswith("#") else "paragraph"
        if block_type == "heading":
            text = text.lstrip("#").strip()
        blocks.append(
            {
                "block_type": block_type,
                "text": text,
                "page": page,
                "section_path": None,
                "table_id": None,
                "row": None,
                "column": None,
            }
        )

    index = 0
    while index < len(lines):
        line = lines[index]
        if "|" in line and index + 1 < len(lines) and TABLE_SEPARATOR_RE.match(lines[index + 1]):
            flush_paragraph()
            table_index += 1
            table_id = f"worker:t{table_index:04d}"
            table_lines = [line]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            rows = [
                [cell.strip() for cell in row.strip().strip("|").split("|")]
                for row in table_lines
            ]
            blocks.append(
                {
                    "block_type": "table",
                    "text": "\n".join("\t".join(row) for row in rows),
                    "page": page,
                    "section_path": None,
                    "table_id": table_id,
                    "row": None,
                    "column": None,
                }
            )
            for row_index, row in enumerate(rows):
                for column_index, cell in enumerate(row):
                    blocks.append(
                        {
                            "block_type": "table_cell",
                            "text": cell,
                            "page": page,
                            "section_path": None,
                            "table_id": table_id,
                            "row": row_index,
                            "column": column_index,
                        }
                    )
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
        index += 1
    flush_paragraph()
    return blocks


def _result_markdown(result: Any) -> str:
    markdown = getattr(result, "markdown", None)
    if isinstance(markdown, dict):
        return str(
            markdown.get("markdown_texts")
            or markdown.get("markdown")
            or markdown.get("text")
            or ""
        )
    if markdown is not None:
        return str(markdown)
    payload = getattr(result, "json", None) or getattr(result, "res", None)
    if isinstance(payload, dict):
        return str(payload.get("markdown") or payload.get("text") or "")
    return ""


def _result_ocr_text(result: Any) -> str:
    payload = getattr(result, "json", None) or getattr(result, "res", None)
    if not isinstance(payload, Mapping):
        return ""
    nested = payload.get("res")
    if isinstance(nested, Mapping):
        payload = nested
    overall = payload.get("overall_ocr_res")
    if not isinstance(overall, Mapping):
        return ""
    values = overall.get("rec_texts")
    if not isinstance(values, list):
        return ""
    lines = [str(value).strip() for value in values if str(value).strip()]
    return "\n".join(lines)


def _result_text(result: Any) -> str:
    markdown = _result_markdown(result)
    ocr_text = _result_ocr_text(result)
    if not ocr_text:
        return markdown
    if not markdown.strip():
        return ocr_text
    # PP-Structure can concatenate adjacent OCR lines inside one layout block.
    # When both representations contain the same characters, retain the raw
    # OCR line boundaries instead of the lossy concatenated markdown.
    compact_markdown = re.sub(r"\s+", "", markdown)
    compact_ocr = re.sub(r"\s+", "", ocr_text)
    return ocr_text if compact_markdown == compact_ocr else markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", default="challenger")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layout-model-dir", required=True)
    parser.add_argument("--text-detection-model-dir", required=True)
    parser.add_argument("--recognition-model-dir", required=True)
    parser.add_argument(
        "--detection-model",
        default="PP-OCRv5_mobile_det",
    )
    parser.add_argument(
        "--recognition-model",
        default="korean_PP-OCRv5_mobile_rec",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from paddleocr import PPStructureV3

    model_directories = {
        "layout": Path(args.layout_model_dir).resolve(),
        "text_detection": Path(args.text_detection_model_dir).resolve(),
        "text_recognition": Path(args.recognition_model_dir).resolve(),
    }
    required_files = (
        "inference.json",
        "inference.pdiparams",
        "inference.yml",
    )
    for label, directory in model_directories.items():
        missing = [
            name
            for name in required_files
            if not (directory / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                "{} model is incomplete at {}: missing {}".format(
                    label,
                    directory,
                    ", ".join(missing),
                )
            )

    pipeline = PPStructureV3(
        layout_detection_model_dir=str(model_directories["layout"]),
        text_detection_model_dir=str(
            model_directories["text_detection"]
        ),
        text_detection_model_name=args.detection_model,
        text_recognition_model_name=args.recognition_model,
        text_recognition_model_dir=str(
            model_directories["text_recognition"]
        ),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
        device=args.device,
    )
    blocks: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    for page_index, result in enumerate(pipeline.predict(input=args.input), start=1):
        markdown = _result_text(result)
        blocks.extend(_markdown_blocks(markdown, page_index))
        raw = getattr(result, "json", None) or getattr(result, "res", None)
        if isinstance(raw, dict):
            raw_pages.append(raw)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw.json")
    raw_path.write_text(
        json.dumps(raw_pages, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )
    payload = {
        "blocks": blocks,
        "metadata": {
            "worker": "pp-structure-v3",
            "profile": args.profile,
            "recognition_model": args.recognition_model,
            "detection_model": args.detection_model,
            "local_model_directories": {
                key: str(value)
                for key, value in model_directories.items()
            },
            "network_model_downloads": False,
            "device": args.device,
        },
        "raw_artifacts": [str(raw_path)],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
