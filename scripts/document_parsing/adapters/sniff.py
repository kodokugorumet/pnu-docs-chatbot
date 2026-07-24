"""Content-first document type detection."""

from __future__ import annotations

import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
MAX_ZIP_ENTRIES = 100_000
MAX_ZIP_ENTRY_BYTES = 512 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 1_000


@dataclass(frozen=True)
class SniffedSource:
    format: str
    mime_type: str
    container: Optional[str]
    extension: str
    reason: str


def sniff_source(path: Path) -> SniffedSource:
    path = Path(path)
    extension = path.suffix.lower()
    with path.open("rb") as handle:
        prefix = handle.read(8192)

    if prefix.startswith(b"\x9b DRMONE") or b"Fasoo DRM" in prefix[:256]:
        return SniffedSource(
            "drm",
            "application/x-fasoo-drm",
            None,
            extension,
            "fasoo_drm_signature",
        )
    if prefix.startswith(b"%PDF-"):
        return SniffedSource("pdf", "application/pdf", None, extension, "pdf_magic")
    if prefix.startswith(OLE_MAGIC):
        if extension == ".xls":
            guessed = "xls"
            mime = "application/vnd.ms-excel"
        elif extension in {".hwp", ".hwpx"}:
            guessed = "hwp"
            mime = "application/x-hwp"
        else:
            guessed = "ole"
            mime = "application/x-ole-storage"
        return SniffedSource(guessed, mime, "ole", extension, "ole_magic")
    if prefix.startswith(ZIP_MAGICS) and zipfile.is_zipfile(str(path)):
        return _sniff_zip(path, extension)
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return SniffedSource("image", "image/png", None, extension, "png_magic")
    if prefix.startswith(b"\xff\xd8\xff"):
        return SniffedSource("image", "image/jpeg", None, extension, "jpeg_magic")
    if prefix.startswith((b"II*\x00", b"MM\x00*")):
        return SniffedSource("image", "image/tiff", None, extension, "tiff_magic")

    lowered = prefix.lstrip().lower()
    if lowered.startswith((b"<!doctype html", b"<html")) or b"<html" in lowered[:1024]:
        return SniffedSource("html", "text/html", None, extension, "html_signature")

    extension_formats = {
        ".hwp": ("hwp", "application/x-hwp"),
        ".hwpx": ("hwpx", "application/hwp+zip"),
        ".pdf": ("pdf", "application/pdf"),
        ".html": ("html", "text/html"),
        ".htm": ("html", "text/html"),
        ".docx": ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ".xlsx": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ".xls": ("xls", "application/vnd.ms-excel"),
        ".pptx": ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    }
    if extension in extension_formats:
        format_name, mime_type = extension_formats[extension]
        return SniffedSource(format_name, mime_type, None, extension, "extension_fallback")
    guessed_mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return SniffedSource("unknown", guessed_mime, None, extension, "unknown")


def _sniff_zip(path: Path, extension: str) -> SniffedSource:
    with zipfile.ZipFile(str(path)) as archive:
        members = archive.infolist()
        _validate_zip_members(members)
        names = {member.filename for member in members}
        lowered_names = {name.lower() for name in names}
        mimetype = ""
        if "mimetype" in names:
            try:
                mimetype = archive.read("mimetype").decode("ascii", errors="ignore").strip().lower()
            except (KeyError, OSError):
                mimetype = ""

    if (
        "application/hwp+zip" in mimetype
        or "version.xml" in lowered_names
        and any(name.startswith("contents/") for name in lowered_names)
        or any(name.startswith("contents/section") and name.endswith(".xml") for name in lowered_names)
    ):
        return SniffedSource("hwpx", "application/hwp+zip", "zip", extension, "hwpx_zip_members")
    if "word/document.xml" in lowered_names:
        return SniffedSource(
            "docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "zip",
            extension,
            "docx_zip_members",
        )
    if "xl/workbook.xml" in lowered_names:
        return SniffedSource(
            "xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "zip",
            extension,
            "xlsx_zip_members",
        )
    if "ppt/presentation.xml" in lowered_names:
        return SniffedSource(
            "pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "zip",
            extension,
            "pptx_zip_members",
        )
    return SniffedSource("zip", "application/zip", "zip", extension, "zip_magic")


def _validate_zip_members(
    members: list[zipfile.ZipInfo],
) -> None:
    if len(members) > MAX_ZIP_ENTRIES:
        raise ValueError(
            "ZIP contains too many entries: {}".format(len(members))
        )
    total = 0
    for member in members:
        if member.file_size > MAX_ZIP_ENTRY_BYTES:
            raise ValueError(
                "ZIP entry is too large: {}".format(member.filename)
            )
        total += member.file_size
        if total > MAX_ZIP_TOTAL_BYTES:
            raise ValueError("ZIP expands beyond the configured limit")
        if (
            member.file_size > 0
            and member.compress_size == 0
        ):
            raise ValueError(
                "ZIP entry has an invalid compression size: {}".format(
                    member.filename
                )
            )
        if (
            member.compress_size > 0
            and member.file_size
            > member.compress_size * MAX_ZIP_COMPRESSION_RATIO
        ):
            raise ValueError(
                "ZIP entry compression ratio is unsafe: {}".format(
                    member.filename
                )
            )
