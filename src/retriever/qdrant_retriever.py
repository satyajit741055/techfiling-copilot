"""Qdrant-backed retrieval: server-side dense + BM25 sparse hybrid search with optional reranking.

Drop-in replacement for ``HybridRetrieverWithReranker`` (same ``search`` signature and
``SearchResult`` return type). Nothing is loaded into process memory except the models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from config.logging import get_logger
from model_runtime import Embedder, check_index_compatibility, create_embedder
from retriever.hybrid_retriever import Reranker, SearchResult
from vector_store import QdrantChunkStore

logger = get_logger(__name__)


class QdrantRetriever:
    """Dense, sparse or RRF-hybrid retrieval over a Qdrant collection.

    The RRF fusion weights default to ``(alpha, 1 - alpha)`` so that ``RETRIEVAL_ALPHA``
    keeps its meaning (dense weight) across vector stores. Explicit ``dense_weight`` /
    ``sparse_weight`` override it. This is rank-based fusion, so results are close to, but
    not identical with, the numpy retriever's min-max score fusion.
    """

    def __init__(
        self,
        store: QdrantChunkStore | None = None,
        *,
        mode: str = "hybrid",
        dense_k: int = 50,
        sparse_k: int = 50,
        rrf_k: int = 60,
        alpha: float | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        embedder: Embedder | None = None,
        device: str | None = None,
        model_cache_directory: Any = None,
    ) -> None:
        if alpha is None:
            from config.settings import settings

            alpha = settings.RETRIEVAL_ALPHA
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        dense_weight = alpha if dense_weight is None else dense_weight
        sparse_weight = 1.0 - alpha if sparse_weight is None else sparse_weight
        self.store = store or QdrantChunkStore.from_settings()
        if not self.store.exists() or self.store.count() == 0:
            raise FileNotFoundError(
                f"Qdrant collection {self.store.collection!r} is missing or empty; "
                "run scripts/migrate_npy_to_qdrant.py or the ingestion pipeline"
            )
        self.mode = mode
        self.dense_k, self.sparse_k, self.rrf_k = dense_k, sparse_k, rrf_k
        self.dense_weight, self.sparse_weight = dense_weight, sparse_weight
        self.embedder = embedder or create_embedder(device=device, cache_dir=model_cache_directory)
        self.model_name = self.embedder.model_name
        check_index_compatibility(self.store.index_metadata(), self.embedder, self.store.dimension())
        logger.info(
            f"Qdrant retriever ready: {self.store.collection} ({self.store.count():,} chunks, mode={mode})"
        )

    @property
    def chunk_count(self) -> int:
        return self.store.count()

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        requested = dict(filters or {})
        for key, value in (("doc_id", doc_id), ("section", section), ("type", type_)):
            if value is not None:
                requested[key] = value
        hits = self.store.search(
            query,
            self.embedder.encode_query(query),
            k,
            filters=requested,
            mode=self.mode,
            dense_k=self.dense_k,
            sparse_k=self.sparse_k,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )
        return [SearchResult(score, chunk) for score, chunk in hits]


class QdrantRetrieverWithReranker:
    """Qdrant hybrid candidate retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        store: QdrantChunkStore | None = None,
        *,
        retrieve_k: int = 50,
        rerank_k: int = 5,
        reranker_model: str | None = None,
        device: str | None = None,
        model_cache_directory: Any = None,
        **retriever_kwargs: Any,
    ) -> None:
        if retrieve_k < 1 or rerank_k < 1:
            raise ValueError("retrieve_k and rerank_k must be at least 1")
        self.retrieve_k, self.rerank_k = retrieve_k, rerank_k
        self.retriever = QdrantRetriever(
            store, device=device, model_cache_directory=model_cache_directory, **retriever_kwargs
        )
        self.reranker = Reranker(reranker_model, device=device, model_cache_directory=model_cache_directory)

    @property
    def reranker_model_name(self) -> str:
        return self.reranker.model_name

    @property
    def chunk_count(self) -> int:
        return self.retriever.chunk_count

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> list[SearchResult]:
        k = k if k is not None else self.rerank_k
        candidates = self.retriever.search(
            query, k=max(k, self.retrieve_k), doc_id=doc_id, section=section, type_=type_, filters=filters
        )
        return self.reranker.rerank(query, candidates, k=k)
