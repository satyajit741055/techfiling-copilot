"""
Hybrid Retriever: BM25 + Dense retrieval with optional reranking.

This module provides:
1. BM25Retriever - sparse keyword-based retrieval (via the ``bm25s`` library)
2. HybridRetriever - combines dense (semantic) and BM25 (keyword) scores
3. Reranker - cross-encoder reranking of retrieved results

Rank-based fusion (RRF) lives in ``retriever.bm25s_rrf_retriever``.

The hybrid approach addresses limitations of pure dense retrieval:
- BM25 excels at exact keyword matching (table rows, specific terms)
- Dense excels at semantic similarity (paraphrases, conceptual matches)
- Combining both improves recall and precision on financial documents

Configuration is loaded from centralized settings (config.settings).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import bm25s
import numpy as np

from config.logging import get_logger
from model_runtime import Embedder, check_index_compatibility, create_embedder, create_reranker

logger = get_logger(__name__)


def _get_settings():
    """Lazy import to avoid circular imports."""
    from config.settings import settings

    return settings


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIRECTORY = ROOT / "data" / "chunks"
DEFAULT_MODEL_CACHE_DIRECTORY = ROOT / "models" / "sentence_transformers"

# Legacy constants for backward compatibility - prefer using settings
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_embedding_model() -> str:
    """Get embedding model name from settings."""
    return _get_settings().EMBEDDING_MODEL


def get_reranker_model() -> str:
    """Get reranker model name from settings (environment-aware)."""
    return _get_settings().effective_reranker


def get_model_cache_dir() -> Path:
    """Get model cache directory from settings."""
    return _get_settings().MODEL_CACHE_DIR


def get_device() -> str:
    """Get compute device from settings."""
    return _get_settings().effective_device


class SearchResult(NamedTuple):
    """A single search result with score and chunk data."""

    score: float
    chunk: dict[str, Any]


# =============================================================================
# BM25 Implementation
# =============================================================================


class BM25:
    """
    BM25 (Best Matching 25) ranking function backed by the ``bm25s`` library.

    ``bm25s`` precomputes a sparse document-term score matrix at index time, so
    querying is a vectorized sparse lookup instead of a per-document Python loop.

    Formula (``method="lucene"``):
        score(D, Q) = sum_{i=1}^{n} IDF(q_i) * (f(q_i, D) * (k1 + 1)) /
                      (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
        IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)

    Where:
        - f(q_i, D) = frequency of term q_i in document D
        - |D| = length of document D (in tokens)
        - avgdl = average document length in the corpus
        - k1 = term frequency saturation parameter (default 1.5)
        - b = length normalization parameter (default 0.75)
        - IDF(q_i) = inverse document frequency of term q_i

    Args:
        k1: Term frequency saturation. Higher values give more weight to
            term frequency. Typical range: 1.2-2.0
        b: Length normalization. 0 = no normalization, 1 = full normalization.
            Typical range: 0.5-0.8
        method: bm25s scoring variant ("lucene", "robertson", "atire", "bm25l", "bm25+")
    """

    TOKEN_PATTERN = r"(?u)\b[a-z0-9]+\b"

    def __init__(self, k1: float = 1.5, b: float = 0.75, method: str = "lucene"):
        """Initialize BM25 with term-frequency and length-normalization weights."""
        self.k1 = k1
        self.b = b
        self.method = method
        self.corpus_size = 0
        self.model = bm25s.BM25(k1=k1, b=b, method=method)

    @classmethod
    def tokenize(cls, texts: str | list[str]) -> list[list[str]]:
        """
        Simple tokenization: lowercase, split on non-alphanumeric, no stopwords.

        For financial documents, this preserves:
        - Numbers (416161, 2025)
        - Ticker symbols (AAPL, MSFT)
        - Common terms (sales, revenue, net)
        """
        return bm25s.tokenize(
            texts,
            lower=True,
            token_pattern=cls.TOKEN_PATTERN,
            stopwords=None,
            return_ids=False,
            show_progress=False,
        )

    @property
    def vocab_size(self) -> int:
        """Number of unique terms in the index."""
        return len(self.model.vocab_dict) if self.corpus_size else 0

    def fit(self, corpus: list[str]) -> "BM25":
        """
        Build the BM25 index from a corpus of documents.

        Args:
            corpus: List of document texts to index

        Returns:
            self (for method chaining)
        """
        self.corpus_size = len(corpus)
        self.model.index(self.tokenize(corpus), show_progress=False)
        return self

    def score_batch(
        self, query: str, doc_indices: list[int] | np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute BM25 scores for multiple documents.

        Args:
            query: Query string
            doc_indices: Indices of documents to score. If None, scores all.

        Returns:
            Array of BM25 scores
        """
        query_tokens = self.tokenize(query)[0]
        if query_tokens:
            scores = np.asarray(self.model.get_scores(query_tokens), dtype=np.float64)
        else:
            scores = np.zeros(self.corpus_size, dtype=np.float64)
        return scores if doc_indices is None else scores[np.asarray(doc_indices, dtype=np.int64)]

    def search(
        self, query: str, k: int = 5, doc_indices: list[int] | None = None
    ) -> list[tuple[int, float]]:
        """
        Search for top-k documents matching the query.

        Args:
            query: Query string
            k: Number of results to return
            doc_indices: Subset of document indices to search. If None, searches all.

        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        if doc_indices is None:
            doc_indices = list(range(self.corpus_size))

        scores = self.score_batch(query, doc_indices)

        # Get top-k indices
        if len(scores) <= k:
            top_positions = np.argsort(-scores)
        else:
            top_positions = np.argpartition(-scores, k - 1)[:k]
            top_positions = top_positions[np.argsort(-scores[top_positions])]

        return [(doc_indices[pos], float(scores[pos])) for pos in top_positions]


class BM25Retriever:
    """
    BM25-based sparse retriever for chunk search.

    This retriever uses BM25 (keyword matching) to find relevant chunks.
    It excels at exact term matching, which is useful for:
    - Table lookups ("total net sales", "416,161")
    - Specific financial terms ("research and development expense")
    - Ticker symbols and company names

    Args:
        run_directory: Path to the chunking run directory
    """

    def __init__(
        self,
        run_directory: str | Path | None = None,
    ) -> None:
        """Load chunks from a retrieval run and build their in-memory BM25 index."""
        self.run_directory = self._resolve_run_directory(run_directory)
        self.manifest = self._load_manifest()
        self.chunks = self._load_chunks()

        # Build BM25 index on embed_text (includes breadcrumb + content)
        corpus = [chunk.get("embed_text", chunk.get("text", "")) for chunk in self.chunks]
        self.bm25 = BM25().fit(corpus)

        logger.info(
            f"BM25 index built: {len(self.chunks)} documents, {self.bm25.vocab_size} unique terms"
        )

    @staticmethod
    def _resolve_run_directory(run_directory: str | Path | None) -> Path:
        """Resolve an explicit run directory or select the latest available run."""
        if run_directory is not None:
            resolved = Path(run_directory).expanduser().resolve()
        else:
            candidates = sorted(
                path
                for path in DEFAULT_RUNS_DIRECTORY.glob("*")
                if path.is_dir() and (path / "run_manifest.json").is_file()
            )
            if not candidates:
                raise FileNotFoundError(f"No retrieval run found in {DEFAULT_RUNS_DIRECTORY}")
            resolved = candidates[-1]
        if not resolved.is_dir():
            raise NotADirectoryError(f"Retrieval run does not exist: {resolved}")
        return resolved

    def _load_manifest(self) -> dict[str, Any]:
        """Load metadata for the selected retrieval run."""
        manifest_path = self.run_directory / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _load_chunks(self) -> list[dict[str, Any]]:
        """Load all run chunks in deterministic filename and line order."""
        chunks: list[dict[str, Any]] = []
        for path in sorted(self.run_directory.glob("*.chunks.jsonl")):
            with path.open(encoding="utf-8") as file:
                chunks.extend(json.loads(line) for line in file if line.strip())
        return chunks

    def _matching_indices(
        self,
        *,
        doc_id: str | None,
        section: str | None,
        type_: str | None,
        filters: Mapping[str, Any] | None,
    ) -> list[int]:
        """Return chunk indices satisfying all requested metadata filters."""
        requested = dict(filters or {})
        if doc_id is not None:
            requested["doc_id"] = doc_id
        if section is not None:
            requested["section"] = section
        if type_ is not None:
            requested["type"] = type_
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if all(chunk.get(key) == value for key, value in requested.items())
        ]

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for chunks using BM25.

        Args:
            query: Search query
            k: Number of results to return
            doc_id: Filter by document ID
            section: Filter by section
            type_: Filter by chunk type
            filters: Additional metadata filters

        Returns:
            List of SearchResult(score, chunk) tuples
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        matches = self._matching_indices(
            doc_id=doc_id, section=section, type_=type_, filters=filters
        )
        if not matches:
            return []

        results = self.bm25.search(query, k=k, doc_indices=matches)
        return [SearchResult(score, self.chunks[idx]) for idx, score in results]


# =============================================================================
# Hybrid Retriever
# =============================================================================
class HybridRetriever:
    """
    Hybrid retriever combining dense (semantic) and BM25 (keyword) retrieval.

    The hybrid approach addresses limitations of each method:
    - Dense retrieval: Good at semantic similarity, poor at exact keyword matching
    - BM25 retrieval: Good at exact keywords, poor at paraphrases/synonyms

    Score combination:
        final_score = alpha * normalized_dense + (1 - alpha) * normalized_bm25

    Where:
        - alpha: Weight for dense scores (0.0 = BM25 only, 1.0 = dense only)
        - Scores are min-max normalized to [0, 1] before combination

    Args:
        run_directory: Path to the chunking run directory
        alpha: Weight for dense scores (default 0.5 = equal weight)
        embedder: Pre-built embedder (optional; otherwise created by model_runtime)
        device: Device for a local embedding model ("cuda", "cpu", etc.)
        model_cache_directory: Cache directory for models
    """

    def __init__(
        self,
        run_directory: str | Path | None = None,
        *,
        alpha: float = 0.5,
        embedder: Embedder | None = None,
        device: str | None = None,
        model_cache_directory: str | Path | None = None,
    ) -> None:
        """Load dense artifacts and build BM25 for weighted hybrid retrieval.

        Args:
            run_directory: Path to chunk directory. If None, uses latest.
            alpha: Dense weight (0=BM25 only, 1=dense only).
            embedder: Pre-built embedder. If None, the configured execution mode decides.
            device: Compute device. If None, uses DEVICE from settings.
            model_cache_directory: Model cache dir. If None, uses settings.

        Raises:
            ValueError: If ``alpha`` is outside the inclusive range 0 to 1 or
                loaded artifacts have inconsistent shapes.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        self.run_directory = self._resolve_run_directory(run_directory)
        self.manifest = self._load_manifest()
        self.chunks = self._load_chunks()

        # Load dense embeddings
        self.embeddings = np.load(
            self.run_directory / "embeddings_gpu_base.npy",
            mmap_mode="r",
        )

        # Load the embedding model the index was built with (backend from execution mode)
        self.embedder = embedder or create_embedder(
            device=device,
            cache_dir=model_cache_directory,
        )
        self.model_name = self.embedder.model_name

        # Build BM25 index
        corpus = [chunk.get("embed_text", chunk.get("text", "")) for chunk in self.chunks]
        self.bm25 = BM25().fit(corpus)

        self._validate_artifacts()

        logger.info(f"Loaded {len(self.chunks)} chunks")
        logger.info(f"Dense model: {self.model_name} ({self.embedder.backend})")
        logger.info(f"BM25 index (bm25s): {self.bm25.vocab_size} unique terms")
        self._log_fusion_config()

    def _log_fusion_config(self) -> None:
        """Log how dense and BM25 scores are fused."""
        logger.info(f"Hybrid alpha: {self.alpha} (dense weight)")

    def _encode_query(self, query: str) -> np.ndarray:
        """Embed a query with the BGE retrieval prefix."""
        return self.embedder.encode_query(query)

    @staticmethod
    def _resolve_run_directory(run_directory: str | Path | None) -> Path:
        """Resolve an explicit run directory or select the latest available run."""
        if run_directory is not None:
            resolved = Path(run_directory).expanduser().resolve()
        else:
            candidates = sorted(
                path
                for path in DEFAULT_RUNS_DIRECTORY.glob("*")
                if path.is_dir()
                and (path / "run_manifest.json").is_file()
                and (path / "embeddings_gpu_base.npy").is_file()
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No complete retrieval run found in {DEFAULT_RUNS_DIRECTORY}"
                )
            resolved = candidates[-1]
        if not resolved.is_dir():
            raise NotADirectoryError(f"Retrieval run does not exist: {resolved}")
        return resolved

    def _load_manifest(self) -> dict[str, Any]:
        """Load metadata for the selected retrieval run."""
        manifest_path = self.run_directory / "run_manifest.json"
        with manifest_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _load_chunks(self) -> list[dict[str, Any]]:
        """Load all run chunks in deterministic filename and line order."""
        chunks: list[dict[str, Any]] = []
        for path in sorted(self.run_directory.glob("*.chunks.jsonl")):
            with path.open(encoding="utf-8") as file:
                chunks.extend(json.loads(line) for line in file if line.strip())
        return chunks

    def _validate_artifacts(self) -> None:
        """Ensure stored embeddings form a matrix aligned with loaded chunks.

        Raises:
            ValueError: If embeddings are not two-dimensional or row counts differ.
        """
        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {self.embeddings.shape}")
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                "Chunk and embedding counts differ: "
                f"{len(self.chunks)} chunks versus {len(self.embeddings)} embeddings"
            )
        check_index_compatibility(self.manifest, self.embedder, self.embeddings.shape[1])

    def _matching_indices(
        self,
        *,
        doc_id: str | None,
        section: str | None,
        type_: str | None,
        filters: Mapping[str, Any] | None,
    ) -> list[int]:
        """Return chunk indices satisfying all requested metadata filters."""
        requested = dict(filters or {})
        if doc_id is not None:
            requested["doc_id"] = doc_id
        if section is not None:
            requested["section"] = section
        if type_ is not None:
            requested["type"] = type_
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if all(chunk.get(key) == value for key, value in requested.items())
        ]

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalize scores to [0, 1].

        Handles edge cases:
        - All scores equal: returns 0.5 for all
        - Single score: returns 1.0
        """
        if len(scores) == 0:
            return scores
        if len(scores) == 1:
            return np.array([1.0])

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.full_like(scores, 0.5)

        return (scores - min_score) / (max_score - min_score)

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """
        Search using hybrid dense + BM25 retrieval.

        Args:
            query: Search query
            k: Number of results to return
            doc_id: Filter by document ID
            section: Filter by section
            type_: Filter by chunk type
            filters: Additional metadata filters
            alpha: Override default alpha for this search

        Returns:
            List of SearchResult(score, chunk) tuples, sorted by hybrid score
        """
        if not query.strip():
            raise ValueError("Query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")

        alpha = alpha if alpha is not None else self.alpha

        # Get matching indices
        matches = self._matching_indices(
            doc_id=doc_id, section=section, type_=type_, filters=filters
        )
        if not matches:
            return []

        match_indices = np.asarray(matches, dtype=np.int64)

        # Compute dense scores
        query_embedding = self._encode_query(query)
        dense_scores = self.embeddings[match_indices] @ query_embedding

        # Compute BM25 scores
        bm25_scores = self.bm25.score_batch(query, matches)

        # Normalize both score sets to [0, 1]
        dense_normalized = self._normalize_scores(dense_scores)
        bm25_normalized = self._normalize_scores(bm25_scores)

        # Combine scores
        hybrid_scores = alpha * dense_normalized + (1 - alpha) * bm25_normalized

        # Get top-k
        result_count = min(k, len(match_indices))
        if result_count == len(match_indices):
            ranked_positions = np.argsort(-hybrid_scores)
        else:
            candidates = np.argpartition(-hybrid_scores, result_count - 1)[:result_count]
            ranked_positions = candidates[np.argsort(-hybrid_scores[candidates])]

        return [
            SearchResult(float(hybrid_scores[pos]), self.chunks[match_indices[pos]])
            for pos in ranked_positions[:result_count]
        ]

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        *,
        doc_id: str | None = None,
        section: str | None = None,
        type_: str | None = None,
        filters: Mapping[str, Any] | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with detailed score breakdown.

        Returns results with dense_score, bm25_score, and hybrid_score for analysis.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        alpha = alpha if alpha is not None else self.alpha

        matches = self._matching_indices(
            doc_id=doc_id, section=section, type_=type_, filters=filters
        )
        if not matches:
            return []

        match_indices = np.asarray(matches, dtype=np.int64)

        # Compute scores
        query_embedding = self._encode_query(query)
        dense_scores = self.embeddings[match_indices] @ query_embedding
        bm25_scores = self.bm25.score_batch(query, matches)

        # Normalize
        dense_normalized = self._normalize_scores(dense_scores)
        bm25_normalized = self._normalize_scores(bm25_scores)
        hybrid_scores = alpha * dense_normalized + (1 - alpha) * bm25_normalized

        # Get top-k
        result_count = min(k, len(match_indices))
        if result_count == len(match_indices):
            ranked_positions = np.argsort(-hybrid_scores)
        else:
            candidates = np.argpartition(-hybrid_scores, result_count - 1)[:result_count]
            ranked_positions = candidates[np.argsort(-hybrid_scores[candidates])]

        results = []
        for pos in ranked_positions[:result_count]:
            idx = match_indices[pos]
            results.append(
                {
                    "hybrid_score": float(hybrid_scores[pos]),
                    "dense_score": float(dense_scores[pos]),
                    "dense_normalized": float(dense_normalized[pos]),
                    "bm25_score": float(bm25_scores[pos]),
                    "bm25_normalized": float(bm25_normalized[pos]),
                    **self.chunks[idx],
                }
            )

        return results

    def iter_chunks(self) -> Iterator[dict[str, Any]]:
        """Iterate over loaded chunks in embedding-row order."""
        return iter(self.chunks)


# =============================================================================
# Reranker
# =============================================================================


class Reranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Cross-encoders process query and document together, allowing for
    deeper interaction between them. This produces more accurate relevance
    scores than bi-encoder (dense) similarity.

    Typical workflow:
    1. Retrieve top-20 with hybrid search (fast, approximate)
    2. Rerank with cross-encoder (slower, more accurate)
    3. Return top-5 reranked results

    Args:
        model_name: Cross-encoder model name
        device: Device for a local model ("cuda", "cpu", etc.)
        model_cache_directory: Cache directory for models
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        model_cache_directory: str | Path | None = None,
    ) -> None:
        """Create the configured cross-encoder backend for pairwise reranking.

        Args:
            model_name: Cross-encoder model. If None, uses settings (env-aware).
            device: Compute device. If None, uses DEVICE from settings.
            model_cache_directory: Model cache dir. If None, uses settings.
        """
        self.backend = create_reranker(model_name, device=device, cache_dir=model_cache_directory)
        self._model_name = self.backend.model_name
        logger.info(f"Reranker loaded: {self._model_name} ({self.backend.backend})")

    @property
    def model_name(self) -> str:
        """Return the model name for logging."""
        return self._model_name

    def rerank(
        self,
        query: str,
        results: list[SearchResult] | list[dict[str, Any]],
        k: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original query
            results: Results from initial retrieval (SearchResult or dict with 'text'/'embed_text')
            k: Number of results to return (default: all)

        Returns:
            Reranked list of SearchResult tuples
        """
        if not results:
            return []

        # Extract chunks and texts
        chunks = []
        texts = []
        for r in results:
            if isinstance(r, SearchResult):
                chunks.append(r.chunk)
                texts.append(r.chunk.get("embed_text", r.chunk.get("text", "")))
            else:
                chunks.append(r)
                texts.append(r.get("embed_text", r.get("text", "")))

        # Get cross-encoder scores for each query-document pair
        scores = self.backend.score(query, texts)

        # Sort by score
        ranked_indices = np.argsort(-scores)

        # Return top-k
        if k is not None:
            ranked_indices = ranked_indices[:k]

        return [SearchResult(float(scores[idx]), chunks[idx]) for idx in ranked_indices]


# =============================================================================
# Combined Pipeline
# =============================================================================


class HybridRetrieverWithReranker:
    """
    Complete retrieval pipeline: Hybrid search + Cross-encoder reranking.

    This is the recommended retriever for production use. It combines:
    1. Dense retrieval (semantic similarity)
    2. BM25 retrieval (keyword matching)
    3. Cross-encoder reranking (accurate relevance scoring)

    Args:
        run_directory: Path to the chunking run directory
        alpha: Weight for dense scores in hybrid combination
        retrieve_k: Number of candidates to retrieve before reranking
        rerank_k: Number of results to return after reranking
        reranker_model: Cross-encoder model name
        device: Device for models
        model_cache_directory: Cache directory for models
    """

    def __init__(
        self,
        run_directory: str | Path | None = None,
        *,
        alpha: float = 0.85,
        retrieve_k: int = 50,
        rerank_k: int = 5,
        reranker_model: str | None = None,
        device: str | None = None,
        model_cache_directory: str | Path | None = None,
    ) -> None:
        """Initialize the hybrid candidate retriever and cross-encoder reranker."""
        self.retrieve_k = retrieve_k
        self.rerank_k = rerank_k

        logger.info("Loading hybrid retriever...")
        self.retriever = HybridRetriever(
            run_directory,
            alpha=alpha,
            device=device,
            model_cache_directory=model_cache_directory,
        )

        logger.info("Loading reranker...")
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
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """
        Search with hybrid retrieval + reranking.

        Args:
            query: Search query
            k: Number of final results (default: self.rerank_k)
            doc_id: Filter by document ID
            section: Filter by section
            type_: Filter by chunk type
            filters: Additional metadata filters
            alpha: Override hybrid alpha

        Returns:
            Reranked list of SearchResult tuples
        """
        k = k if k is not None else self.rerank_k

        # Stage 1: Hybrid retrieval
        candidates = self.retriever.search(
            query,
            k=max(k, self.retrieve_k),
            doc_id=doc_id,
            section=section,
            type_=type_,
            filters=filters,
            alpha=alpha,
        )

        if not candidates:
            return []

        # Stage 2: Reranking
        return self.reranker.rerank(query, candidates, k=k)

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Return chunks loaded by the underlying hybrid retriever."""
        return self.retriever.chunks

    @property
    def run_directory(self) -> Path:
        """Return the active retrieval run directory."""
        return self.retriever.run_directory

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the active retrieval run manifest."""
        return self.retriever.manifest


# =============================================================================
# CLI
# =============================================================================


def show(results: list[SearchResult], width: int = 350) -> None:
    """Display search results."""
    for score, chunk in results:
        print(f"{score:.3f} | {chunk['breadcrumb']} [{chunk['type']}]")
        print("      ", chunk["text"][:width].replace("\n", " "))
        print(f"      element_orders: {chunk['element_orders']}")
        print("=" * 80)


def main() -> None:
    """Run dense, BM25, hybrid, or reranked retrieval from the CLI."""
    parser = argparse.ArgumentParser(
        description="Hybrid retriever: BM25 + Dense with optional reranking."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What was Apple's total net sales in 2025?",
    )
    parser.add_argument("-k", type=int, default=5, help="Number of results")
    parser.add_argument("--doc-id", help="Filter by document ID")
    parser.add_argument("--section", help="Filter by section")
    parser.add_argument("--type", dest="type_", help="Filter by chunk type")
    parser.add_argument("--run-directory", help="Path to chunking run")
    parser.add_argument("--device", help="Device for models (cuda, cpu)")
    parser.add_argument(
        "--model-cache-dir",
        default=DEFAULT_MODEL_CACHE_DIRECTORY,
        help="Cache directory for models",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dense weight (default: RETRIEVAL_ALPHA from .env)",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid", "rerank", "rrf", "rrf_rerank"],
        default="hybrid",
        help="Retrieval mode",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    parser.add_argument("--dense-k", type=int, default=50, help="Dense candidates for RRF")
    parser.add_argument("--bm25-k", type=int, default=50, help="BM25 candidates for RRF")
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=None,
        help="Candidates for reranking (default: RETRIEVAL_K from .env)",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Reranker model (default: environment-aware setting from .env)",
    )
    parser.add_argument("--width", type=int, default=350, help="Text display width")
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show detailed score breakdown",
    )
    args = parser.parse_args()
    settings = _get_settings()
    args.alpha = settings.RETRIEVAL_ALPHA if args.alpha is None else args.alpha
    args.retrieve_k = settings.RETRIEVAL_K if args.retrieve_k is None else args.retrieve_k
    args.reranker_model = args.reranker_model or settings.effective_reranker

    if args.mode == "bm25":
        logger.info("Loading BM25 retriever...")
        retriever: Any = BM25Retriever(args.run_directory)
        results = retriever.search(
            args.query,
            k=args.k,
            doc_id=args.doc_id,
            section=args.section,
            type_=args.type_,
        )
        show(results, args.width)

    elif args.mode == "hybrid":
        logger.info("Loading hybrid retriever...")
        retriever = HybridRetriever(
            args.run_directory,
            alpha=args.alpha,
            device=args.device,
            model_cache_directory=args.model_cache_dir,
        )

        if args.show_scores:
            results = retriever.search_with_scores(
                args.query,
                k=args.k,
                doc_id=args.doc_id,
                section=args.section,
                type_=args.type_,
            )
            for r in results:
                print(
                    f"Hybrid: {r['hybrid_score']:.3f} | "
                    f"Dense: {r['dense_score']:.3f} ({r['dense_normalized']:.3f}) | "
                    f"BM25: {r['bm25_score']:.3f} ({r['bm25_normalized']:.3f})"
                )
                print(f"  {r['breadcrumb'][:80]}...")
                print(f"  {r['text'][:200].replace(chr(10), ' ')}...")
                print()
        else:
            results = retriever.search(
                args.query,
                k=args.k,
                doc_id=args.doc_id,
                section=args.section,
                type_=args.type_,
            )
            show(results, args.width)

    elif args.mode == "rerank":
        logger.info("Loading hybrid retriever with reranker...")
        retriever = HybridRetrieverWithReranker(
            args.run_directory,
            alpha=args.alpha,
            retrieve_k=args.retrieve_k,
            rerank_k=args.k,
            reranker_model=args.reranker_model,
            device=args.device,
            model_cache_directory=args.model_cache_dir,
        )
        results = retriever.search(
            args.query,
            k=args.k,
            doc_id=args.doc_id,
            section=args.section,
            type_=args.type_,
        )
        show(results, args.width)

    elif args.mode in ("rrf", "rrf_rerank"):
        from retriever.bm25s_rrf_retriever import (
            BM25SRRFRetriever,
            BM25SRRFRetrieverWithReranker,
        )

        rrf_kwargs = dict(
            dense_k=args.dense_k,
            bm25_k=args.bm25_k,
            rrf_k=args.rrf_k,
            device=args.device,
            model_cache_directory=args.model_cache_dir,
        )
        if args.mode == "rrf":
            retriever = BM25SRRFRetriever(args.run_directory, **rrf_kwargs)
        else:
            retriever = BM25SRRFRetrieverWithReranker(
                args.run_directory,
                retrieve_k=args.retrieve_k,
                rerank_k=args.k,
                reranker_model=args.reranker_model,
                **rrf_kwargs,
            )
        results = retriever.search(
            args.query,
            k=args.k,
            doc_id=args.doc_id,
            section=args.section,
            type_=args.type_,
        )
        show(results, args.width)

    else:  # dense
        # Import DenseRetriever for comparison
        from retriever.retriever import DenseRetriever
        from retriever.retriever import show as dense_show

        logger.info("Loading dense retriever...")
        retriever = DenseRetriever(
            args.run_directory,
            device=args.device,
            model_cache_directory=args.model_cache_dir,
        )
        results = retriever.search(
            args.query,
            k=args.k,
            doc_id=args.doc_id,
            section=args.section,
            type_=args.type_,
        )
        dense_show(results, args.width)


if __name__ == "__main__":
    main()
