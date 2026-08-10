"""Deterministic UTF-8 JSONL serialization helpers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


def json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(
        "Object of type {} is not JSON serializable".format(
            type(value).__name__
        )
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
        allow_nan=False,
    )


def write_jsonl(
    path: Path,
    records: Iterable[Any],
    sort_key: Optional[Callable[[Any], Any]] = None,
) -> int:
    """Atomically write JSONL with stable object-key ordering and LF endings."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values: Iterable[Any]
    if sort_key is None:
        values = records
    else:
        sorted_values = list(records)
        sorted_values.sort(key=sort_key)
        values = sorted_values
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(output_path.name),
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(canonical_json(value))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(output_path))
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return count


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON at {}:{}: {}".format(
                        input_path, line_number, exc
                    )
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    "Expected JSON object at {}:{}".format(
                        input_path, line_number
                    )
                )
            yield value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))
