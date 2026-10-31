"""Qdrant-backed retrieval: server-side dense + BM25 sparse hybrid search with optional reranking.

Drop-in replacement for ``HybridRetrieverWithReranker`` (same ``search`` signature and
``SearchResult`` return type). Nothing is loaded into process memory except the models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from config.logging import get_logger
from model_runtime import Embedder, check_index_compatibility, create_embedder
from retriever.hybrid_retriever import Reranker, SearchResult
from vector_store import QdrantChunkStore
from vector_store.filters import merge_filters

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
        check_index_compatibility(
            self.store.index_metadata(), self.embedder, self.store.dimension()
        )
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
        """``filters`` accepts scalars, value lists and numeric ranges (``vector_store.filters``)."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        hits = self.store.search(
            query,
            self.embedder.encode_query(query),
            k,
            filters=merge_filters(filters, doc_id=doc_id, section=section, type=type_),
            **self._tuning,
        )
        return [SearchResult(score, chunk) for score, chunk in hits]

    @property
    def _tuning(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dense_k": self.dense_k,
            "sparse_k": self.sparse_k,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Chunks by ``chunk_id`` in request order; duplicates collapsed, unknown ids skipped."""
        return self.store.get_chunks(chunk_ids)

    def search_grouped(
        self,
        query: str,
        group_by: str,
        *,
        groups: Sequence[Any] | None = None,
        k: int = 5,
        group_limit: int = 10,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> dict[Any, list[SearchResult]]:
        """Top ``k`` results per ``group_by`` value; see ``QdrantChunkStore.search_grouped``.

        Pass ``groups`` when the values are known (one search per group, full recall per
        group); omit it to let Qdrant pick the best ``group_limit`` groups in one call.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")
        grouped = self.store.search_grouped(
            query,
            self.embedder.encode_query(query),
            group_by,
            groups=groups,
            group_size=k,
            group_limit=group_limit,
            filters=merge_filters(filters, doc_id=doc_id, section=section, type=type_),
            **self._tuning,
        )
        return {
            group: [SearchResult(score, chunk) for score, chunk in hits]
            for group, hits in grouped.items()
        }


class QdrantRetrieverWithReranker:
    """Qdrant hybrid candidate retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        store: QdrantChunkStore | None = None,
        *,
        retrieve_k: int = 50,
        rerank_k: int = 5,
        reranker_model: str | None = None,
        reranker: Reranker | None = None,
        device: str | None = None,
        model_cache_directory: Any = None,
        **retriever_kwargs: Any,
    ) -> None:
        """``reranker`` reuses an already-loaded cross-encoder (``reranker_model`` is then ignored)."""
        if retrieve_k < 1 or rerank_k < 1:
            raise ValueError("retrieve_k and rerank_k must be at least 1")
        self.retrieve_k, self.rerank_k = retrieve_k, rerank_k
        self.retriever = QdrantRetriever(
            store, device=device, model_cache_directory=model_cache_directory, **retriever_kwargs
        )
        self.reranker = reranker or Reranker(
            reranker_model, device=device, model_cache_directory=model_cache_directory
        )

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
            query,
            k=max(k, self.retrieve_k),
            doc_id=doc_id,
            section=section,
            type_=type_,
            filters=filters,
        )
        return self.reranker.rerank(query, candidates, k=k)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Chunks by ``chunk_id`` in request order (no scoring)."""
        return self.retriever.get_chunks(chunk_ids)

    def search_grouped(
        self,
        query: str,
        group_by: str,
        *,
        groups: Sequence[Any] | None = None,
        k: int | None = None,
        group_limit: int = 10,
        retrieve_k: int | None = None,
        **filter_kwargs: Any,
    ) -> dict[Any, list[SearchResult]]:
        """Grouped hybrid retrieval, then cross-encoder reranking within each group.

        ``k`` (default ``rerank_k``) results per group; ``retrieve_k`` (default the
        pipeline's) candidates per group are reranked, all groups in one reranker call.
        """
        k = k if k is not None else self.rerank_k
        candidates = self.retriever.search_grouped(
            query,
            group_by,
            groups=groups,
            k=max(k, retrieve_k or self.retrieve_k),
            group_limit=group_limit,
            **filter_kwargs,
        )
        return self.reranker.rerank_groups(query, candidates, k=k)
