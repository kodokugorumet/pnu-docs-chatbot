"""Heading-aware conversion from canonical blocks to legacy BM25 chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import Block


DEFAULT_CHUNK_CHARS = 1800
DEFAULT_CHUNK_OVERLAP = 250
INDEXED_BLOCK_TYPES = frozenset(
    {"heading", "paragraph", "list_item", "table", "caption"}
)


@dataclass(frozen=True)
class _ChunkPart:
    text: str
    blocks: Sequence[Block]
    section_path: Optional[Tuple[str, ...]]
    table_ids: Sequence[str]


def blocks_to_legacy_chunks(
    blocks: Iterable[Block],
    metadata: Optional[Mapping[str, Any]] = None,
    doc_id: Optional[str] = None,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    table_header_rows: Optional[Mapping[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Convert canonical blocks into the current BM25 ``chunks.jsonl`` shape.

    Normal blocks are combined only inside one ``section_path``. Tables are
    independent chunks and canonical ``table_cell`` blocks are never indexed,
    avoiding duplicate table text. Overlap applies only when one source block
    or one table row is itself too large.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")
    ordered = sorted(
        list(blocks), key=lambda block: (block.reading_order, block.block_id)
    )
    if not ordered:
        return []
    document_ids = {block.document_id for block in ordered}
    if len(document_ids) != 1:
        raise ValueError("all blocks must belong to one document")
    resolved_doc_id = doc_id or ordered[0].document_id
    header_rows = dict(table_header_rows or {})

    parts: List[_ChunkPart] = []
    current: List[Block] = []
    current_length = 0
    current_section: Optional[Tuple[str, ...]] = None

    def flush() -> None:
        nonlocal current, current_length, current_section
        if current:
            text = "\n\n".join(block.text for block in current).strip()
            if text:
                parts.append(
                    _ChunkPart(
                        text=text,
                        blocks=tuple(current),
                        section_path=current_section,
                        table_ids=(),
                    )
                )
        current = []
        current_length = 0
        current_section = None

    for block in ordered:
        if block.block_type not in INDEXED_BLOCK_TYPES:
            continue
        section = (
            tuple(block.section_path) if block.section_path is not None else None
        )
        if block.block_type == "table":
            flush()
            parts.extend(
                _table_parts(
                    block,
                    max_chars=max_chars,
                    overlap=overlap,
                    header_rows=header_rows.get(block.table_id or "", 0),
                )
            )
            continue

        if current and section != current_section:
            flush()
        if len(block.text) > max_chars:
            flush()
            for piece in _split_oversized_text(block.text, max_chars, overlap):
                parts.append(
                    _ChunkPart(
                        text=piece,
                        blocks=(block,),
                        section_path=section,
                        table_ids=(),
                    )
                )
            continue

        next_length = current_length + len(block.text) + (2 if current else 0)
        if current and next_length > max_chars:
            flush()
        if not current:
            current_section = section
        current.append(block)
        current_length += len(block.text) + (2 if current_length else 0)
    flush()

    base_metadata = dict(metadata or {})
    chunks: List[Dict[str, Any]] = []
    for index, part in enumerate(parts):
        chunk_metadata = dict(base_metadata)
        chunk_metadata["block_ids"] = [block.block_id for block in part.blocks]
        pages = sorted(
            {
                block.page
                for block in part.blocks
                if block.page is not None
            }
        )
        chunk_metadata["page_start"] = pages[0] if pages else None
        chunk_metadata["page_end"] = pages[-1] if pages else None
        chunk_metadata["section_path"] = (
            list(part.section_path)
            if part.section_path is not None
            else None
        )
        chunk_metadata["table_ids"] = list(part.table_ids)
        if "parser" not in chunk_metadata:
            parsers = list(dict.fromkeys(block.parser for block in part.blocks))
            chunk_metadata["parser"] = (
                parsers[0] if len(parsers) == 1 else "+".join(parsers)
            )
        chunks.append(
            {
                "chunk_id": "{}#{:04d}".format(resolved_doc_id, index),
                "doc_id": resolved_doc_id,
                "chunk_index": index,
                "text": part.text,
                "char_count": len(part.text),
                "metadata": chunk_metadata,
            }
        )
    return chunks


def _table_parts(
    block: Block,
    max_chars: int,
    overlap: int,
    header_rows: int,
) -> List[_ChunkPart]:
    table_id = block.table_id or ""
    section = (
        tuple(block.section_path) if block.section_path is not None else None
    )
    if len(block.text) <= max_chars:
        return [
            _ChunkPart(
                text=block.text,
                blocks=(block,),
                section_path=section,
                table_ids=(table_id,),
            )
        ]
    rows = block.text.splitlines()
    if not rows:
        rows = [block.text]
    header_count = min(max(0, header_rows), len(rows))
    headers = rows[:header_count]
    data_rows = rows[header_count:]
    if not data_rows:
        data_rows = headers
        headers = []

    texts: List[str] = []
    current = list(headers)
    for row in data_rows:
        candidate = "\n".join(current + [row])
        if len(candidate) <= max_chars:
            current.append(row)
            continue
        if len(current) > len(headers):
            texts.append("\n".join(current))
            current = list(headers)
        candidate = "\n".join(current + [row])
        if len(candidate) <= max_chars:
            current.append(row)
            continue
        prefix = "\n".join(headers)
        available = max_chars - len(prefix) - (1 if prefix else 0)
        if available <= 0:
            # A declared header that consumes a whole chunk cannot be repeated
            # safely. Preserve it once and split the data row independently.
            if prefix:
                texts.extend(_split_oversized_text(prefix, max_chars, overlap))
            prefix = ""
            available = max_chars
        for piece in _split_oversized_text(row, available, min(overlap, available - 1)):
            texts.append("{}\n{}".format(prefix, piece) if prefix else piece)
        current = list(headers)
    if len(current) > len(headers):
        texts.append("\n".join(current))
    return [
        _ChunkPart(
            text=text,
            blocks=(block,),
            section_path=section,
            table_ids=(table_id,),
        )
        for text in texts
        if text
    ]


def _split_oversized_text(
    text: str,
    max_chars: int,
    overlap: int,
) -> List[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    step = max_chars - overlap
    pieces = []
    start = 0
    while start < len(text):
        piece = text[start : start + max_chars].strip()
        if piece:
            pieces.append(piece)
        if start + max_chars >= len(text):
            break
        start += step
    return pieces
