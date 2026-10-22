"""Select embedding and reranking backends from the configured execution mode.

This is the only place that decides *where* a model runs. Callers ask for an
embedder or reranker and receive an object implementing the shared interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .huggingface_api import HuggingFaceApiEmbedder, HuggingFaceApiReranker
from .local import LocalEmbedder, LocalReranker
from .types import Embedder, Reranker

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)


def _settings(settings: "Settings | None") -> "Settings":
    if settings is not None:
        return settings
    from config.settings import settings as global_settings
    return global_settings


def _local_device(device: str | None, settings: "Settings") -> str:
    device = device or settings.effective_device
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"DEVICE={device} requested but CUDA is not available")
    return device


def create_embedder(
    model_name: str | None = None,
    *,
    mode: str | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    settings: "Settings | None" = None,
) -> Embedder:
    """Return the embedder for ``mode`` (default: the configured embedding mode)."""
    settings = _settings(settings)
    mode = mode or settings.effective_embedding_mode
    model_name = model_name or settings.EMBEDDING_MODEL
    cache_dir = cache_dir or settings.MODEL_CACHE_DIR
    if mode == "local":
        return LocalEmbedder(model_name, device=_local_device(device, settings), cache_dir=cache_dir)
    if mode == "huggingface_api":
        return HuggingFaceApiEmbedder(
            model_name,
            api_key=settings.HUGGINGFACE_API_KEY,
            provider=settings.HUGGINGFACE_EMBEDDING_PROVIDER,
            timeout=settings.HUGGINGFACE_TIMEOUT,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown embedding execution mode: {mode}")


def create_reranker(
    model_name: str | None = None,
    *,
    mode: str | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    settings: "Settings | None" = None,
) -> Reranker:
    """Return the reranker for ``mode`` (default: the configured reranker mode)."""
    settings = _settings(settings)
    mode = mode or settings.effective_reranker_mode
    model_name = model_name or settings.effective_reranker
    if mode == "local":
        return LocalReranker(
            model_name,
            device=_local_device(device, settings),
            cache_dir=cache_dir or settings.MODEL_CACHE_DIR,
        )
    if mode == "huggingface_api":
        return HuggingFaceApiReranker(
            model_name,
            api_key=settings.HUGGINGFACE_API_KEY,
            url=settings.HUGGINGFACE_RERANKER_URL,
            timeout=settings.HUGGINGFACE_TIMEOUT,
        )
    raise ValueError(f"Unknown reranker execution mode: {mode}")


def check_index_compatibility(
    manifest: dict[str, Any], embedder: Embedder, stored_dimension: int
) -> None:
    """Ensure the active embedder can query embeddings stored for a chunk run.

    Raises:
        ValueError: If the model or vector dimension differs from the stored index.
    """
    indexed = manifest.get("embedding_model", {})
    if indexed.get("name") and indexed["name"] != embedder.model_name:
        raise ValueError(
            f"Index was built with {indexed['name']!r} but the active embedder is "
            f"{embedder.model_name!r}; re-embed the chunk run or change EMBEDDING_MODEL"
        )
    if embedder.dimension != stored_dimension:
        raise ValueError(
            f"Embedder dimension {embedder.dimension} differs from stored embeddings "
            f"dimension {stored_dimension}"
        )
    indexed_backend = indexed.get("backend", "local")
    if indexed_backend != embedder.backend:
        message = (
            f"Index was embedded with the {indexed_backend!r} backend but queries use "
            f"{embedder.backend!r}; same model, but re-embedding is recommended for exact parity"
        )
        logger.warning(message)
        print(f"  WARNING: {message}")
