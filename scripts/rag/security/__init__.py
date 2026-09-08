"""Public entry points for the two-stage RAG security layer."""

from .context_gate import REMOVED_MARKER, detect_injection, evaluate_contexts
from .output_gate import enforce_output

__all__ = [
    "REMOVED_MARKER",
    "detect_injection",
    "enforce_output",
    "evaluate_contexts",
]
