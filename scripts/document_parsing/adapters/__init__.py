"""Public adapter API for all parser profiles."""

from .base import AdapterContext, adapter_available
from .native import (
    parse_native_html,
    parse_native_hwp,
    parse_native_hwpx,
    parse_native_office,
)
from .pdf import (
    PagePreflight,
    classify_pdf_page,
    overlay_pdf_tables,
    parse_pdfplumber_tables,
    parse_pymupdf,
    parse_pypdf,
    preflight_pdf_page,
)
from .sniff import SniffedSource, sniff_source
from .subprocess import (
    parse_docling,
    parse_java_hwp,
    parse_paddle,
    parse_subprocess,
    parse_tesseract,
    parse_tika,
    parse_unhwp,
)

__all__ = [
    "AdapterContext",
    "PagePreflight",
    "SniffedSource",
    "adapter_available",
    "classify_pdf_page",
    "overlay_pdf_tables",
    "parse_docling",
    "parse_java_hwp",
    "parse_native_html",
    "parse_native_hwp",
    "parse_native_hwpx",
    "parse_native_office",
    "parse_paddle",
    "parse_pdfplumber_tables",
    "parse_pymupdf",
    "parse_pypdf",
    "parse_subprocess",
    "parse_tesseract",
    "parse_tika",
    "parse_unhwp",
    "preflight_pdf_page",
    "sniff_source",
]
