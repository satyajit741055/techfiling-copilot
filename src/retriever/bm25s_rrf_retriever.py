"""
Dense + BM25 (bm25s) retrieval fused with weighted Reciprocal Rank Fusion (RRF).

Unlike ``HybridRetriever`` (min-max normalized score blending), RRF only uses
each retriever's *rank* for a document, so it is insensitive to score scale:

    rrf_score(d) = dense_weight / (rrf_k + rank_dense(d))
                 + bm25_weight  / (rrf_k + rank_bm25(d))

Documents missing from a retriever's top list contribute 0 for that retriever.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from retriever.hybrid_retriever import HybridRetriever, Reranker, SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[np.ndarray, float]],
    rrf_k: int = 60,
) -> dict[int, float]:
    """Fuse ranked index lists into ``{index: score}`` using weighted RRF.

    Args:
        ranked_lists: ``(indices_in_rank_order, weight)`` pairs.
        rrf_k: Smoothing constant; larger values flatten the rank contribution.
    """
    fused: dict[int, float] = {}
    for indices, weight in ranked_lists:
        for rank, index in enumerate(indices, start=1):
            key = int(index)
            fused[key] = fused.get(key, 0.0) + weight / (rrf_k + rank)
    return fused


class BM25SRRFRetriever(HybridRetriever):
    """Dense and BM25S retrieval combined with weighted reciprocal rank fusion."""

    def __init__(
        self,
        run_directory: str | Path | None = None,
        *,
        dense_k: int = 50,
        bm25_k: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        model: SentenceTransformer | None = None,
        device: str | None = None,
        model_cache_directory: str | Path | None = None,
    ) -> None:
        """Load dense artifacts and the bm25s index for RRF retrieval.

        Args:
            run_directory: Path to chunk directory. If None, uses latest.
            dense_k: Dense candidates fed into fusion.
            bm25_k: BM25 candidates fed into fusion.
            rrf_k: RRF smoothing constant (60 is the standard default).
            dense_weight: Weight of the dense ranking in fusion.
            bm25_weight: Weight of the BM25 ranking in fusion.
            model: Pre-loaded embedding model.
            device: Compute device. If None, uses DEVICE from settings.
            model_cache_directory: Model cache dir. If None, uses settings.
        """
        if dense_k < 1 or bm25_k < 1:
            raise ValueError("dense_k and bm25_k must be at least 1")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("RRF weights must be non-negative")
        if dense_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one RRF weight must be positive")

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        super().__init__(
            run_directory,
            model=model,
            device=device,
            model_cache_directory=model_cache_directory,
        )

    def _print_fusion_config(self) -> None:
        """Log RRF fusion parameters."""
        print(
            f"  RRF: dense_k={self.dense_k}, bm25_k={self.bm25_k}, rrf_k={self.rrf_k}, "
            f"weights=({self.dense_weight}, {self.bm25_weight})"
        )

    @staticmethod
    def _top_indices(scores: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
        """Return ``indices`` of the top-``k`` scores in descending order."""
        result_count = min(k, len(indices))
        if result_count == 0:
            return np.empty(0, dtype=np.int64)
        if result_count == len(indices):
            positions = np.argsort(-scores, kind="stable")
        else:
            candidates = np.argpartition(-scores, result_count - 1)[:result_count]
            positions = candidates[np.argsort(-scores[candidates], kind="stable")]
        return indices[positions]

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
        """Search using dense + BM25 candidate lists fused by RRF.

        Returns:
            List of SearchResult(rrf_score, chunk), sorted by RRF score.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")

        matches = np.asarray(
            self._matching_indices(doc_id=doc_id, section=section, type_=type_, filters=filters),
            dtype=np.int64,
        )
        if len(matches) == 0:
            return []

        dense_scores = self.embeddings[matches] @ self._encode_query(query)
        dense_indices = self._top_indices(dense_scores, matches, max(k, self.dense_k))

        bm25_scores = self.bm25.score_batch(query, matches)
        bm25_indices = self._top_indices(bm25_scores, matches, max(k, self.bm25_k))

        fused = reciprocal_rank_fusion(
            [(dense_indices, self.dense_weight), (bm25_indices, self.bm25_weight)],
            rrf_k=self.rrf_k,
        )
        ranked = sorted(fused, key=lambda index: (-fused[index], index))
        return [SearchResult(fused[index], self.chunks[index]) for index in ranked[:k]]


class BM25SRRFRetrieverWithReranker:
    """BM25S and dense RRF candidate retrieval followed by cross-encoder reranking."""

    def __init__(
        self,
        run_directory: str | Path | None = None,
        *,
        dense_k: int = 50,
        bm25_k: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        retrieve_k: int = 20,
        rerank_k: int = 5,
        reranker_model: str | None = None,
        device: str | None = None,
        model_cache_directory: str | Path | None = None,
    ) -> None:
        """Initialize the RRF candidate retriever and cross-encoder reranker."""
        if retrieve_k < 1 or rerank_k < 1:
            raise ValueError("retrieve_k and rerank_k must be at least 1")
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k

        print("Loading BM25S + dense RRF retriever...")
        self.retriever = BM25SRRFRetriever(
            run_directory,
            dense_k=dense_k,
            bm25_k=bm25_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            device=device,
            model_cache_directory=model_cache_directory,
        )

        print("Loading reranker...")
        self.reranker = Reranker(
            reranker_model,
            device=device,
            model_cache_directory=model_cache_directory,
        )

    @property
    def reranker_model_name(self) -> str:
        """Return the reranker model name for logging."""
        return self.reranker.model_name

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
        """Retrieve RRF candidates, then rerank them with the cross-encoder."""
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

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Return chunks loaded by the underlying RRF retriever."""
        return self.retriever.chunks

    @property
    def run_directory(self) -> Path:
        """Return the active retrieval run directory."""
        return self.retriever.run_directory

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the active retrieval run manifest."""
        return self.retriever.manifest
