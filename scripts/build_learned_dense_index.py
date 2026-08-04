#!/usr/bin/env python3
"""Build a reproducible learned-dense matrix for the active BM25 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_SOURCE_INDEX = Path(
    "processed/index/pnu-20260725-curated-cascade-v5-allow-suspect.sqlite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu"), default="auto"
    )
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--storage-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--tokenizer-batch-size", type=int, default=512)
    return parser.parse_args()


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_sha256(chunk_ids: Sequence[str], texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for chunk_id, text in zip(chunk_ids, texts):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def load_corpus(source_index: Path) -> tuple[list[str], list[str], dict[str, str]]:
    connection = sqlite3.connect(str(source_index))
    try:
        rows = connection.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        meta = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM index_meta ORDER BY key"
            )
        }
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("source index contains no chunks")
    chunk_ids = [str(row[0]) for row in rows]
    texts = [str(row[1] or "") for row in rows]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise RuntimeError("source index contains duplicate chunk IDs")
    expected = int(meta.get("chunk_count", len(rows)))
    if expected != len(rows):
        raise RuntimeError(
            f"source index metadata says {expected} chunks but contains {len(rows)}"
        )
    return chunk_ids, texts, meta


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
        raise ValueError(f"unsupported model dtype: {dtype}")


def synchronize(torch: Any, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def count_tokens(model: Any, texts: Sequence[str], batch_size: int) -> dict[str, Any]:
    total = 0
    minimum: int | None = None
    maximum = 0
    truncated = 0
    for group in batches(texts, batch_size):
        encoded = model.tokenizer(
            list(group),
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=model.max_seq_length,
            return_length=True,
        )
        lengths = encoded.get("length")
        if lengths is None:
            lengths = [len(value) for value in encoded["input_ids"]]
        for raw_length in lengths:
            length = int(raw_length)
            total += length
            minimum = length if minimum is None else min(minimum, length)
            maximum = max(maximum, length)
            if length >= int(model.max_seq_length):
                truncated += 1
    return {
        "total": total,
        "min": minimum or 0,
        "max": maximum,
        "mean": total / len(texts),
        "at_max_sequence_length": truncated,
    }


def write_chunk_ids(path: Path, chunk_ids: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk_id in chunk_ids:
            handle.write(json.dumps(chunk_id, ensure_ascii=False))
            handle.write("\n")


def normalized_prompts(model: Any) -> dict[str, str]:
    prompts = getattr(model, "prompts", {}) or {}
    return {str(key): str(value) for key, value in prompts.items()}


def model_revision(model: Any) -> str | None:
    try:
        value = model[0].auto_model.config._commit_hash
    except (AttributeError, IndexError, TypeError):
        return None
    return str(value) if value else None


def verify_artifact(
    vectors_path: Path,
    chunk_ids_path: Path,
    *,
    expected_count: int,
    expected_dimensions: int,
    expected_dtype: str,
) -> dict[str, Any]:
    import numpy as np

    vectors = np.load(vectors_path, mmap_mode="r")
    if vectors.shape != (expected_count, expected_dimensions):
        raise RuntimeError(f"unexpected saved vector shape: {vectors.shape}")
    if str(vectors.dtype) != expected_dtype:
        raise RuntimeError(f"unexpected saved vector dtype: {vectors.dtype}")
    if not bool(np.isfinite(vectors).all()):
        raise RuntimeError("saved vectors contain non-finite values")
    probe_positions = sorted({0, expected_count // 2, expected_count - 1})
    probe = np.asarray(vectors[probe_positions], dtype=np.float32)
    norms = np.linalg.norm(probe, axis=1)
    if not bool(np.allclose(norms, np.ones_like(norms), atol=2e-3)):
        raise RuntimeError("saved probe vectors are not unit-normalized")
    with chunk_ids_path.open("r", encoding="utf-8") as handle:
        id_count = sum(1 for line in handle if line.strip())
    if id_count != expected_count:
        raise RuntimeError(
            f"saved chunk ID count {id_count} does not match {expected_count}"
        )
    return {
        "verified": True,
        "probe_positions": probe_positions,
        "probe_norms": [float(value) for value in norms],
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    source_index = args.source_index.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    chunk_ids, texts, source_meta = load_corpus(source_index)
    source_digest = corpus_sha256(chunk_ids, texts)

    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    device = choose_device(torch, args.device)
    build_started = time.perf_counter()
    load_started = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    apply_dtype(model, args.model_dtype)
    synchronize(torch, device)
    load_seconds = time.perf_counter() - load_started

    token_started = time.perf_counter()
    tokens = count_tokens(model, texts, args.tokenizer_batch_size)
    tokenization_seconds = time.perf_counter() - token_started

    warmup_size = min(args.batch_size, len(texts))
    with torch.inference_mode():
        model.encode(
            texts[:warmup_size],
            batch_size=warmup_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    synchronize(torch, device)

    embedding_started = time.perf_counter()
    with torch.inference_mode():
        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    synchronize(torch, device)
    embedding_seconds = time.perf_counter() - embedding_started
    vectors = np.asarray(vectors, dtype=args.storage_dtype, order="C")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunk_ids):
        raise RuntimeError(f"unexpected embedding shape: {vectors.shape}")
    if not bool(np.isfinite(vectors).all()):
        raise RuntimeError("embedding output contains non-finite values")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        vectors_path = temporary / "vectors.npy"
        chunk_ids_path = temporary / "chunk_ids.jsonl"
        manifest_path = temporary / "manifest.json"

        save_started = time.perf_counter()
        np.save(vectors_path, vectors, allow_pickle=False)
        write_chunk_ids(chunk_ids_path, chunk_ids)
        save_seconds = time.perf_counter() - save_started

        artifact = verify_artifact(
            vectors_path,
            chunk_ids_path,
            expected_count=len(chunk_ids),
            expected_dimensions=int(vectors.shape[1]),
            expected_dtype=str(vectors.dtype),
        )
        manifest = {
            "schema_version": 1,
            "created_at_unix": time.time(),
            "model": {
                "id": args.model,
                "revision": model_revision(model),
                "max_sequence_length": int(model.max_seq_length),
                "prompts": normalized_prompts(model),
                "default_prompt_name": getattr(model, "default_prompt_name", None),
                "inference_dtype": args.model_dtype,
            },
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__,
                "device": device,
                "mps_available": bool(torch.backends.mps.is_available()),
                "batch_size": args.batch_size,
            },
            "source": {
                "index": str(source_index),
                "corpus_revision": source_meta.get("corpus_revision"),
                "manifest_sha256": source_meta.get("manifest_sha256"),
                "chunk_count": len(chunk_ids),
                "char_count": sum(len(text) for text in texts),
                "corpus_sha256": source_digest,
            },
            "vectors": {
                "file": vectors_path.name,
                "sha256": sha256_file(vectors_path),
                "count": len(chunk_ids),
                "dimensions": int(vectors.shape[1]),
                "dtype": str(vectors.dtype),
                "normalized": True,
                "bytes": vectors_path.stat().st_size,
            },
            "chunk_ids": {
                "file": chunk_ids_path.name,
                "sha256": sha256_file(chunk_ids_path),
                "count": len(chunk_ids),
                "bytes": chunk_ids_path.stat().st_size,
            },
            "tokens": tokens,
            "timings": {
                "model_load_seconds": load_seconds,
                "tokenization_seconds": tokenization_seconds,
                "embedding_seconds": embedding_seconds,
                "save_seconds": save_seconds,
                "total_seconds": time.perf_counter() - build_started,
            },
            "throughput": {
                "chunks_per_second": len(texts) / embedding_seconds,
                "tokens_per_second": tokens["total"] / embedding_seconds,
                "chars_per_second": sum(len(text) for text in texts)
                / embedding_seconds,
            },
            "peak_rss_mib": peak_rss_mib(),
            "validation": artifact,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "model": args.model,
                "chunk_count": len(chunk_ids),
                "dimensions": int(vectors.shape[1]),
                "embedding_seconds": embedding_seconds,
                "save_seconds": save_seconds,
                "peak_rss_mib": peak_rss_mib(),
                "verified": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
