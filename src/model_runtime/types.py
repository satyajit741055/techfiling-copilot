"""Backend-agnostic interfaces for embedding and reranking models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return float32 row vectors scaled to unit L2 norm."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class Embedder(Protocol):
    """Produces normalized float32 embeddings for queries and documents."""

    backend: str
    model_name: str
    query_prefix: str

    @property
    def dimension(self) -> int: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def max_seq_length(self) -> int: ...

    def encode_query(self, query: str) -> np.ndarray: ...

    def encode_documents(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray: ...


class Reranker(Protocol):
    """Scores query-document pairs; higher means more relevant."""

    backend: str
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...
