"""Reusable RAG service components."""

from .generators import (
    Context,
    ExtractiveFallback,
    GenerationAttempt,
    GenerationError,
    GenerationResult,
    SUPPORTED_PROVIDERS,
    build_prompt,
    extract_gemini_text,
    extract_openai_compatible_text,
    generate,
)

__all__ = [
    "Context",
    "ExtractiveFallback",
    "GenerationAttempt",
    "GenerationError",
    "GenerationResult",
    "SUPPORTED_PROVIDERS",
    "build_prompt",
    "extract_gemini_text",
    "extract_openai_compatible_text",
    "generate",
]
