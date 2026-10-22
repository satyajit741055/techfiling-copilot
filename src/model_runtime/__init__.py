"""Execution-mode aware model runtime (local CUDA/CPU or Hugging Face API).

Usage:
    from model_runtime import create_embedder, create_reranker

    embedder = create_embedder()   # backend chosen by EMBEDDING/MODEL_EXECUTION_MODE
    reranker = create_reranker()   # backend chosen by RERANKER/MODEL_EXECUTION_MODE
"""

from .factory import check_index_compatibility, create_embedder, create_reranker
from .types import BGE_QUERY_PREFIX, Embedder, Reranker

__all__ = [
    "BGE_QUERY_PREFIX",
    "Embedder",
    "Reranker",
    "check_index_compatibility",
    "create_embedder",
    "create_reranker",
]
