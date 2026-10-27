"""Local (CUDA/CPU) embedding and reranking backends using sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .types import BGE_QUERY_PREFIX


class LocalEmbedder:
    """SentenceTransformer bi-encoder running on the local device."""

    backend = "local"

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        cache_dir: str | Path,
        query_prefix: str = BGE_QUERY_PREFIX,
        model: Any | None = None,
    ) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device, cache_folder=str(cache_dir))
        self.model = model
        self.model_name = model_name
        self.query_prefix = query_prefix

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            self.query_prefix + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")


class LocalReranker:
    """CrossEncoder reranker running on the local device."""

    backend = "local"

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        cache_dir: str | Path,
        model: Any | None = None,
    ) -> None:
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device=device, cache_folder=str(cache_dir))
        self.model = model
        self.model_name = model_name

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents:
            return np.empty(0, dtype=np.float32)
        return np.asarray(
            self.model.predict([(query, text) for text in documents]), dtype=np.float32
        )
