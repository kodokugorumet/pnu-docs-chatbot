#!/usr/bin/env python3
"""Benchmark a learned dense embedder on a deterministic corpus sample.

This script measures document-embedding throughput only. Model download/load,
tokenization accounting, warmup, and embedding are timed separately so a pilot
run can be extrapolated to the full corpus without mixing one-time setup cost
into the estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence


DEFAULT_INDEX = Path(
    "processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--source-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--warmup-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def load_sample(
    source_index: Path, sample_size: int, seed: int
) -> tuple[list[str], list[str], dict[str, int]]:
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    connection = sqlite3.connect(str(source_index))
    try:
        rows = connection.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        full_char_count = int(
            connection.execute(
                "SELECT COALESCE(SUM(length(text)), 0) FROM chunks"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    if sample_size > len(rows):
        raise ValueError(
            f"sample size {sample_size} exceeds corpus size {len(rows)}"
        )
    selected_positions = sorted(
        random.Random(seed).sample(range(len(rows)), sample_size)
    )
    selected = [rows[position] for position in selected_positions]
    chunk_ids = [str(row[0]) for row in selected]
    texts = [str(row[1] or "") for row in selected]
    return chunk_ids, texts, {
        "full_chunk_count": len(rows),
        "full_char_count": full_char_count,
        "sample_char_count": sum(len(text) for text in texts),
    }


def sample_sha256(chunk_ids: Sequence[str], texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for chunk_id, text in zip(chunk_ids, texts):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def apply_dtype(model: Any, dtype: str) -> None:
    if dtype == "float16":
        model.half()
    elif dtype == "bfloat16":
        model.bfloat16()
    elif dtype != "float32":
        raise ValueError(f"unsupported dtype: {dtype}")


def synchronize(torch: Any, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def token_lengths(model: Any, texts: Sequence[str]) -> list[int]:
    encoded = model.tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=model.max_seq_length,
        return_length=True,
    )
    lengths = encoded.get("length")
    if lengths is None:
        lengths = [len(value) for value in encoded["input_ids"]]
    return [int(value) for value in lengths]


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    source_index = args.source_index.resolve()
    chunk_ids, texts, corpus = load_sample(
        source_index, args.sample_size, args.seed
    )

    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    device = choose_device(torch, args.device)
    load_started = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    apply_dtype(model, args.dtype)
    synchronize(torch, device)
    load_seconds = time.perf_counter() - load_started
    rss_after_load = peak_rss_mib()

    tokenize_started = time.perf_counter()
    lengths = token_lengths(model, texts)
    tokenization_seconds = time.perf_counter() - tokenize_started

    warmup_count = min(max(1, args.warmup_size), len(texts))
    with torch.inference_mode():
        model.encode(
            texts[:warmup_count],
            batch_size=min(args.batch_size, warmup_count),
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    synchronize(torch, device)

    encode_runs = []
    embeddings = None
    for _ in range(args.repeats):
        encode_started = time.perf_counter()
        with torch.inference_mode():
            embeddings = model.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=args.repeats == 1,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        synchronize(torch, device)
        encode_runs.append(time.perf_counter() - encode_started)

    assert embeddings is not None
    encode_seconds = statistics.median(encode_runs)

    embeddings_float = embeddings.float()
    if not bool(torch.isfinite(embeddings_float).all().item()):
        raise RuntimeError("embedding output contains non-finite values")
    norms = torch.linalg.vector_norm(embeddings_float, dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-3)):
        raise RuntimeError("normalized embeddings do not have unit norm")

    sample_tokens = sum(lengths)
    full_chunks = corpus["full_chunk_count"]
    full_chars = corpus["full_char_count"]
    sample_chars = corpus["sample_char_count"]
    count_scale = full_chunks / len(texts)
    char_scale = full_chars / sample_chars if sample_chars else count_scale
    report = {
        "schema_version": 1,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "model": args.model,
        "source_index": str(source_index),
        "device": device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sample_size": len(texts),
        "sample_sha256": sample_sha256(chunk_ids, texts),
        "sample_char_count": sample_chars,
        "sample_token_count": sample_tokens,
        "sample_token_length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sample_tokens / len(lengths),
        },
        "full_chunk_count": full_chunks,
        "full_char_count": full_chars,
        "model_max_sequence_length": int(model.max_seq_length),
        "embedding_dimensions": int(embeddings.shape[1]),
        "output_tensor_dtype": str(embeddings.dtype),
        "model_load_seconds": load_seconds,
        "tokenization_seconds": tokenization_seconds,
        "embedding_seconds": encode_seconds,
        "embedding_seconds_runs": encode_runs,
        "embedding_seconds_mean": statistics.fmean(encode_runs),
        "embedding_seconds_median": encode_seconds,
        "throughput": {
            "chunks_per_second": len(texts) / encode_seconds,
            "tokens_per_second": sample_tokens / encode_seconds,
            "chars_per_second": sample_chars / encode_seconds,
        },
        "estimated_full_embedding_seconds_by_chunk_count": (
            encode_seconds * count_scale
        ),
        "estimated_full_embedding_seconds_by_char_count": (
            encode_seconds * char_scale
        ),
        "peak_rss_mib": peak_rss_mib(),
        "rss_after_model_load_mib": rss_after_load,
        "validation": {
            "all_finite": True,
            "mean_vector_norm": float(norms.mean().item()),
            "first_vector_checksum": hashlib.sha256(
                embeddings_float[0].cpu().numpy().tobytes()
            ).hexdigest(),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
