"""Unicode-safe deterministic text normalization."""

from __future__ import annotations

import re
import unicodedata


HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")
EXCESS_BLANK_LINES_RE = re.compile(r"\n{4,}")


def clean_text(value: str, preserve_tabs: bool = False) -> str:
    """Normalize parser text while preserving paragraph boundaries.

    Control characters other than newlines (and optionally tabs) become spaces.
    The function does not perform semantic cleanup, language correction, or
    whitespace-sensitive table reconstruction.
    """

    if not isinstance(value, str):
        raise TypeError("clean_text expects a string")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = []
    for character in value:
        if character == "\n" or (preserve_tabs and character == "\t"):
            cleaned.append(character)
        elif unicodedata.category(character) == "Cc":
            cleaned.append(" ")
        else:
            cleaned.append(character)
    value = "".join(cleaned)
    if preserve_tabs:
        lines = [
            re.sub(r" +", " ", line).strip(" ")
            for line in value.split("\n")
        ]
    else:
        lines = [
            HORIZONTAL_SPACE_RE.sub(" ", line).strip()
            for line in value.split("\n")
        ]
    value = "\n".join(lines)
    value = EXCESS_BLANK_LINES_RE.sub("\n\n\n", value)
    return value.strip()
