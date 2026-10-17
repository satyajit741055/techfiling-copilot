"""Token-budget validation and embedding of chunks with a sentence-transformers model.

Logic ported from exploration/01_chunking.ipynb (BGE, normalized float32 vectors).
Configuration is loaded from centralized settings (config.settings).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ingestion.chunking.chunker import (
    MAX_CHARS, MIN_CHARS, ROOT, TABLE_MAX_CHARS,
)


def _get_settings():
    """Lazy import to avoid circular imports."""
    from config.settings import settings
    return settings


# Legacy constants for backward compatibility - prefer using settings
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDINGS_FILENAME = "embeddings_gpu_base.npy"  # name expected by retriver/retriever.py
MANIFEST_FILENAME = "run_manifest.json"
CHUNKS_DIR = ROOT / "data" / "chunks"
MODEL_CACHE_DIR = ROOT / "models" / "sentence_transformers"
TOKEN_UTILIZATION = 0.94


def get_embedding_model_name() -> str:
    """Get embedding model name from settings."""
    return _get_settings().EMBEDDING_MODEL


def get_model_cache_dir() -> Path:
    """Get model cache directory from settings."""
    return _get_settings().MODEL_CACHE_DIR


def get_device() -> str:
    """Get compute device from settings."""
    return _get_settings().effective_device


def load_model(
    name: str | None = None,
    device: str | None = None,
    cache_dir: Path | None = None,
) -> SentenceTransformer:
    """Load a sentence-transformers model on the requested or best device.

    Args:
        name: Hugging Face model name or local model path. If None, uses
            EMBEDDING_MODEL from settings.
        device: Explicit device such as ``cuda`` or ``cpu``. If None, uses
            DEVICE from settings (with auto-detection).
        cache_dir: Directory used to cache downloaded model files. If None,
            uses MODEL_CACHE_DIR from settings.

    Returns:
        The initialized sentence-transformers model.
    """
    # Use settings defaults if not provided
    if name is None:
        name = get_embedding_model_name()
    if device is None:
        device = get_device()
    if cache_dir is None:
        cache_dir = get_model_cache_dir()
    
    return SentenceTransformer(name, device=device, cache_folder=str(cache_dir))


def token_limits(model: SentenceTransformer) -> dict:
    """Return model, tokenizer, effective, and recommended token limits."""
    effective = int(min(model.max_seq_length, model.tokenizer.model_max_length))
    return {
        "model_max_tokens": int(model.max_seq_length),
        "tokenizer_max_tokens": int(model.tokenizer.model_max_length),
        "effective_max_tokens": effective,
        "recommended_token_limit": int(effective * TOKEN_UTILIZATION),
    }

def count_tokens(model: SentenceTransformer, chunks: list[dict]) -> list[int]:
    """Count untruncated tokens in each chunk's embedding text."""
    return [
        len(model.tokenizer.encode(c["embed_text"], add_special_tokens=True, truncation=False))
        for c in chunks
    ]


def validate_token_budget(chunks: list[dict], token_counts: list[int], limit: int) -> None:
    """Ensure no chunk's embedding text exceeds the model token limit.

    Args:
        chunks: Ordered chunk dictionaries being validated.
        token_counts: Token count corresponding to each chunk.
        limit: Maximum permitted token count per chunk.

    Raises:
        ValueError: If one or more chunks exceed ``limit``.
    """
    oversized = [
        {"chunk_id": c["chunk_id"], "tokens": n} for c, n in zip(chunks, token_counts) if n > limit
    ]
    if oversized:
        raise ValueError(f"{len(oversized)} chunks exceed the {limit}-token model limit: {oversized[:10]}")

def embed_chunks(model: SentenceTransformer, chunks: list[dict], batch_size: int = 64) -> np.ndarray:
    """Encode chunk text as normalized float32 embedding vectors.

    Args:
        model: Initialized sentence-transformers model.
        chunks: Chunk dictionaries containing ``embed_text``.
        batch_size: Number of texts encoded per inference batch.

    Returns:
        A two-dimensional array with one embedding row per chunk.
    """
    return model.encode(
        [c["embed_text"] for c in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def token_summary(values: list[int]) -> dict:
    """Calculate descriptive statistics for a sequence of token counts."""
    s = pd.Series(values)
    return {
        "count": int(s.count()), "mean": round(float(s.mean()), 2), "std": round(float(s.std()), 2),
        "min": int(s.min()), "p25": float(s.quantile(0.25)), "p50": float(s.quantile(0.5)),
        "p75": float(s.quantile(0.75)), "max": int(s.max()),
    }

def chunk_summary(chunks: list[dict], token_counts: list[int], limits: dict, min_chars: int) -> dict:
    """Summarize chunk types, sizes, and token-limit utilization.

    Args:
        chunks: Chunk dictionaries included in the summary.
        token_counts: Token count corresponding to each chunk.
        limits: Token-limit metadata returned by :func:`token_limits`.
        min_chars: Character threshold used to classify tiny chunks.

    Returns:
        Aggregate chunk counts and token statistics.
    """
    types = pd.Series([c["type"] for c in chunks]).value_counts().to_dict()
    return {
        "chunks": len(chunks),
        "chunks_by_type": {k: int(v) for k, v in types.items()},
        "tiny_chunks": sum(len(c["text"].strip()) < min_chars for c in chunks),
        "over_recommended_limit": sum(n > limits["recommended_token_limit"] for n in token_counts),
        "over_model_limit": sum(n > limits["effective_max_tokens"] for n in token_counts),
        "token_statistics": token_summary(token_counts),
    }

    
def build_manifest(
    chunks: list[dict],
    token_counts: list[int],
    model: SentenceTransformer,
    model_name: str,
    run_timestamp: str,
    output_directory: str,
    chunking_config: dict,
) -> dict:
    """Build run-level and per-document embedding metadata.

    Args:
        chunks: All chunks included in the embedding run.
        token_counts: Token count corresponding to each chunk.
        model: Model used to create the embeddings.
        model_name: Model identifier recorded in the manifest.
        run_timestamp: Timestamp or run identifier for this output.
        output_directory: Output location recorded in the manifest.
        chunking_config: Character limits used to generate the chunks.

    Returns:
        A serializable manifest containing model and chunk statistics.
    """
    limits = token_limits(model)
    min_chars = chunking_config["min_chars"]
    doc_ids = list(dict.fromkeys(c["doc_id"] for c in chunks))
    return {
        "run_timestamp": run_timestamp,
        "output_directory": output_directory,
        "chunking_config": chunking_config,
        "embedding_model": {
            "name": model_name,
            "dimension": int(model.get_embedding_dimension()),
            **limits,
        },
        "overall": chunk_summary(chunks, token_counts, limits, min_chars),
        "documents": {
            d: chunk_summary(
                [c for c in chunks if c["doc_id"] == d],
                [n for c, n in zip(chunks, token_counts) if c["doc_id"] == d],
                limits, min_chars,
            )
            for d in doc_ids
        },
    }


def embed_and_save(
    chunks: list[dict],
    out_dir: Path,
    model_name: str = MODEL_NAME,
    device: str | None = None,
    batch_size: int = 64,
    chunking_config: dict | None = None,
    model: SentenceTransformer | None = None,
    model_cache_dir: Path = MODEL_CACHE_DIR,
) -> tuple[np.ndarray, dict]:
    """Validate, embed, and persist chunks with their run manifest.

    Args:
        chunks: Chunk dictionaries containing text and source metadata.
        out_dir: Directory for the NumPy embeddings and JSON manifest.
        model_name: Hugging Face model name or local model path.
        device: Explicit inference device, or ``None`` for automatic selection.
        batch_size: Number of chunks encoded per inference batch.
        chunking_config: Chunk-size settings to record in the manifest.
        model: Optional preloaded model, primarily for reuse or testing.
        model_cache_dir: Directory used to cache model files when loading.

    Returns:
        A tuple containing the embedding matrix and manifest dictionary.

    Raises:
        ValueError: If any chunk exceeds the model's effective token limit.
    """
    chunking_config = chunking_config or {
        "max_chars": MAX_CHARS, "min_chars": MIN_CHARS, "table_max_chars": TABLE_MAX_CHARS,
    }
    model = model or load_model(model_name, device, model_cache_dir)
    limits = token_limits(model)
    token_counts = count_tokens(model, chunks)
    validate_token_budget(chunks, token_counts, limits["effective_max_tokens"])

    embeddings = embed_chunks(model, chunks, batch_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / EMBEDDINGS_FILENAME, embeddings)

    try:
        output_directory = str(out_dir.resolve().relative_to(ROOT))
    except ValueError:
        output_directory = str(out_dir.resolve())
    manifest = build_manifest(
        chunks, token_counts, model, model_name,
        out_dir.name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), output_directory, chunking_config,
    )
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return embeddings, manifest


def load_chunks(chunks_dir: Path) -> list[dict]:
    """Load all chunk JSON Lines files from a directory in filename order."""
    chunks = []
    for path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        with path.open(encoding="utf-8") as fh:
            chunks.extend(json.loads(line) for line in fh if line.strip())
    return chunks


def main() -> None:
    """Run the CLI workflow for embedding an existing chunk directory."""
    parser = argparse.ArgumentParser(
        description="Embed an existing directory of *.chunks.jsonl and save the .npy + manifest next to it.")
    parser.add_argument("chunks_dir", type=Path, help="e.g. data/chunks/<run_timestamp>")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--device", help="cuda | cpu (default: cuda if available)")
    parser.add_argument("--model-cache-dir", type=Path, default=MODEL_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_dir)
    if not chunks:
        raise SystemExit(f"No *.chunks.jsonl found in {args.chunks_dir}")
    embeddings, _ = embed_and_save(
        chunks, args.chunks_dir, args.model, args.device, args.batch_size,
        model_cache_dir=args.model_cache_dir)
    print(embeddings.shape, "->", args.chunks_dir / EMBEDDINGS_FILENAME)


if __name__ == "__main__":
    main()

