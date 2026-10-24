"""
Retrieval Evaluation Module for SEC 10-K RAG System
====================================================

This module provides comprehensive evaluation of the dense retriever against
a golden dataset. It computes standard IR metrics and logs results for
comparison across different configurations.

METRICS COMPUTED
----------------

1. Hit Rate@k (also called Success Rate@k)
   - Definition: Fraction of queries where at least one relevant document
     appears in the top-k results.
   - Formula: (queries with at least 1 hit in top-k) / (total queries)
   - Range: [0, 1], higher is better
   - Use case: "Did we find something useful?" - good for user satisfaction

2. Recall@k
   - Definition: Fraction of all relevant documents that appear in top-k.
   - Formula: |relevant in top-k| / |all relevant|
   - Range: [0, 1], higher is better
   - Use case: "Did we find everything?" - important when completeness matters

3. MRR@k (Mean Reciprocal Rank)
   - Definition: Average of 1/rank for the first relevant result.
   - Formula: mean(1/rank_of_first_relevant) across queries
   - Range: [0, 1], higher is better
   - Use case: "How early does the first good result appear?" - important for
     single-answer questions

4. nDCG@k (Normalized Discounted Cumulative Gain)
   - Definition: Measures ranking quality with graded relevance, normalized
     against the ideal ranking.
   - Formula: DCG@k / IDCG@k where DCG = sum((2^rel - 1) / log2(rank + 1))
   - Range: [0, 1], higher is better
   - Use case: "Is the ranking order good?" - important when relevance varies

5. Precision@k
   - Definition: Fraction of top-k results that are relevant.
   - Formula: |relevant in top-k| / k
   - Range: [0, 1], higher is better
   - Use case: "How clean are the results?" - important when showing few results

RELEVANCE MATCHING
------------------

Chunks are matched to golden answers via `element_orders`. A chunk is considered
relevant if its `element_orders` list intersects with any expected element order
from the golden dataset. This allows stable evaluation across different chunking
configurations since element orders come from the parser, not the chunker.

Relevance grades (from golden dataset):
- 3: Directly answers the question
- 2: Provides supporting evidence
- 1: Provides useful context
- 0: Irrelevant (implicit for non-matching chunks)

USAGE
-----

Command line:
    uv run python eval/dense_retriever_eval.py
    uv run python eval/dense_retriever_eval.py --k 10 --run-directory data/chunks/2026-10-14_15-50-05
    uv run python eval/dense_retriever_eval.py --compare  # Compare all logged runs


OUTPUT
------

Results are saved to `eval/results/<run_timestamp>_eval_results.json` with:
- Configuration (k values, run directory, model, chunking params)
- Per-query results (for debugging retrieval failures)
- Aggregate metrics (for comparison)
- Category breakdowns (to identify weak areas)
- Difficulty breakdowns (to understand performance by query hardness)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retriever.retriever import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_PATH = ROOT / "goldens" / "golden_retriever_dataset_v2.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RelevanceMatch:
    """Represents a relevance match between a retrieved chunk and golden answer."""

    chunk_id: str
    element_orders: list[int]
    matched_orders: list[int]
    relevance_grade: int  # 0-3
    rank: int  # 1-indexed position in results
    score: float  # similarity score from retriever


@dataclass
class QueryResult:
    """Evaluation result for a single query."""

    query_id: str
    query: str
    category: str
    difficulty: str
    doc_id: str | None

    # Retrieval outcomes
    retrieved_count: int
    relevant_retrieved: int
    total_relevant: int

    # Per-query metrics
    hit: bool  # Did we find at least one relevant?
    reciprocal_rank: float  # 1/rank of first relevant, 0 if none
    recall: float  # relevant items found (each credited once) / total_relevant
    precision: float  # retrieved chunks containing a relevant element / retrieved_count
    ndcg: float  # Normalized DCG

    # Details for debugging
    matches: list[RelevanceMatch] = field(default_factory=list)
    top_chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricSummary:
    """Summary statistics for a metric across queries."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int


@dataclass
class EvalResults:
    """Complete evaluation results."""

    # Configuration
    timestamp: str
    run_directory: str
    golden_path: str
    k_values: list[int]
    primary_k: int  # The main k value for headline metrics (typically 5)
    chunking_config: dict[str, Any]
    embedding_model: dict[str, Any]
    total_chunks: int
    total_queries: int

    # Overall metrics at primary_k (headline numbers)
    overall: dict[str, float]

    # Aggregate metrics per k
    metrics_by_k: dict[int, dict[str, float]]

    # Breakdowns
    metrics_by_category: dict[str, dict[int, dict[str, float]]]
    metrics_by_difficulty: dict[str, dict[int, dict[str, float]]]
    metrics_by_doc: dict[str, dict[int, dict[str, float]]]

    # Per-query details (for debugging)
    query_results: dict[int, list[QueryResult]]  # k -> results

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "timestamp": self.timestamp,
            "run_directory": self.run_directory,
            "golden_path": self.golden_path,
            "k_values": self.k_values,
            "primary_k": self.primary_k,
            "chunking_config": self.chunking_config,
            "embedding_model": self.embedding_model,
            "total_chunks": self.total_chunks,
            "total_queries": self.total_queries,
            # Overall metrics at the top for easy access
            "overall": self.overall,
            "metrics_by_k": self.metrics_by_k,
            "metrics_by_category": self.metrics_by_category,
            "metrics_by_difficulty": self.metrics_by_difficulty,
            "metrics_by_doc": self.metrics_by_doc,
            "query_results": {
                k: [asdict(qr) for qr in results]
                for k, results in self.query_results.items()
            },
        }
        return result

# =============================================================================
# Metric Computation Functions
# =============================================================================


def compute_dcg(relevance_scores: list[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at k.

    DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)

    The gain from each result is discounted by its position. Results at
    higher ranks contribute less to the score.

    Args:
        relevance_scores: List of relevance grades (0-3) in rank order
        k: Cutoff position

    Returns:
        DCG score (unbounded, depends on relevance scale)
    """
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        # i is 0-indexed, rank is i+1
        # Discount factor: log2(rank + 1) = log2(i + 2)
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def compute_ndcg(relevance_scores: list[int], ideal_scores: list[int], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.

    nDCG@k = DCG@k / IDCG@k

    IDCG is the DCG of the ideal ranking (all relevant docs sorted by
    relevance grade descending).

    Args:
        relevance_scores: Actual relevance grades in retrieval order
        ideal_scores: All known relevance grades (will be sorted for IDCG)
        k: Cutoff position

    Returns:
        nDCG score in [0, 1], or 0 if no relevant documents exist
    """
    dcg = compute_dcg(relevance_scores, k)

    # IDCG: DCG of ideal ranking (sorted descending)
    ideal_sorted = sorted(ideal_scores, reverse=True)
    idcg = compute_dcg(ideal_sorted, k)

    if idcg == 0:
        return 0.0  # No relevant documents exist
    return dcg / idcg


def compute_reciprocal_rank(relevance_scores: list[int], min_relevance: int = 1) -> float:
    """
    Compute Reciprocal Rank.

    RR = 1 / rank_of_first_relevant

    Args:
        relevance_scores: Relevance grades in rank order
        min_relevance: Minimum grade to count as relevant (default 1)

    Returns:
        Reciprocal rank in (0, 1], or 0 if no relevant found
    """
    for rank, rel in enumerate(relevance_scores, start=1):
        if rel >= min_relevance:
            return 1.0 / rank
    return 0.0


def compute_hit(relevance_scores: list[int], min_relevance: int = 1) -> bool:
    """
    Check if any relevant document was retrieved.

    Args:
        relevance_scores: Relevance grades in rank order
        min_relevance: Minimum grade to count as relevant

    Returns:
        True if at least one relevant document found
    """
    return any(rel >= min_relevance for rel in relevance_scores)


def compute_recall(
    retrieved_relevant: int, total_relevant: int
) -> float:
    """
    Compute Recall.

    Recall = |relevant in retrieved| / |all relevant|

    Args:
        retrieved_relevant: Count of relevant docs in results
        total_relevant: Total count of relevant docs

    Returns:
        Recall in [0, 1], or 0 if no relevant docs exist
    """
    if total_relevant == 0:
        return 0.0
    return retrieved_relevant / total_relevant


def compute_precision(
    retrieved_relevant: int, retrieved_count: int
) -> float:
    """
    Compute Precision.

    Precision = |relevant in retrieved| / |retrieved|

    Args:
        retrieved_relevant: Count of relevant docs in results
        retrieved_count: Total count of retrieved docs

    Returns:
        Precision in [0, 1], or 0 if nothing retrieved
    """
    if retrieved_count == 0:
        return 0.0
    return retrieved_relevant / retrieved_count

# =============================================================================
# Golden Dataset Loader
# =============================================================================


@dataclass
class GoldenQuery:
    """A single query from the golden evaluation dataset."""

    id: str
    query: str
    doc_id: str | None
    category: str
    difficulty: str
    expected_answer: str
    relevant: list[dict[str, Any]]

    def get_relevant_orders(self, doc_id: str | None = None) -> set[int]:
        """
        Get all element orders marked as relevant.

        For cross-document queries, optionally filter by doc_id.
        """
        orders = set()
        for item in self.relevant:
            if item["match_by"] == "element_orders":
                # For cross-document queries, check doc_id match
                item_doc = item.get("doc_id")
                if doc_id is None or item_doc is None or item_doc == doc_id:
                    orders.update(item["value"])
        return orders

    def get_relevance_grade(self, element_orders: list[int], doc_id: str | None = None) -> int:
        """
        Get the highest relevance grade for a set of element orders.

        Args:
            element_orders: Orders from a retrieved chunk
            doc_id: Document ID of the chunk (for cross-document matching)

        Returns:
            Highest matching relevance grade, or 0 if no match
        """
        max_grade = 0
        chunk_orders = set(element_orders)

        for item in self.relevant:
            if item["match_by"] == "element_orders":
                item_doc = item.get("doc_id")
                # For cross-document queries, require doc_id match
                if item_doc is not None and doc_id is not None and item_doc != doc_id:
                    continue

                expected_orders = set(item["value"])
                if chunk_orders & expected_orders:  # Intersection
                    max_grade = max(max_grade, item["relevance"])

        return max_grade

    def score_ranking(self, ranked: list[tuple[list[int], str | None]]) -> tuple[list[int], set[int]]:
        """Grade a ranked list of (element_orders, doc_id), crediting each relevant item at most once.

        An item may list equivalent elements (the same fact in an MD&A table and a note); a second copy
        must not add gain again, otherwise recall double-counts and nDCG can exceed 1.
        Returns per-rank gains and the indices of relevant items found.
        """
        scores, found = [], set()
        for chunk_orders, doc_id in ranked:
            orders = set(chunk_orders or [])
            new = [
                i for i, item in enumerate(self.relevant)
                if i not in found
                and item["match_by"] == "element_orders"
                and (item.get("doc_id") is None or doc_id is None or item["doc_id"] == doc_id)
                and orders & set(item["value"])
            ]
            scores.append(max((self.relevant[i]["relevance"] for i in new), default=0))
            found.update(new)
        return scores, found

    def get_all_relevance_grades(self) -> list[int]:
        """Get all relevance grades from the golden answer (for IDCG)."""
        return [item["relevance"] for item in self.relevant]


def load_golden_dataset(path: Path) -> list[GoldenQuery]:
    """Load golden evaluation dataset from JSONL file."""
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries.append(GoldenQuery(
                    id=data["id"],
                    query=data["query"],
                    doc_id=data.get("doc_id"),
                    category=data["category"],
                    difficulty=data["difficulty"],
                    expected_answer=data["expected_answer"],
                    relevant=data.get("relevant", []),
                ))
    return queries


# =============================================================================
# Evaluator Class
# =============================================================================

class RetrieverEvaluator:
    """
    Evaluates a dense retriever against a golden dataset.

    This class orchestrates the evaluation process:
    1. Loads the golden dataset
    2. Runs queries through the retriever
    3. Matches results to expected answers
    4. Computes metrics
    5. Aggregates and logs results

    Attributes:
        retriever: The DenseRetriever instance to evaluate
        golden_queries: List of golden evaluation queries
        golden_path: Path to the golden dataset file
    """

    def __init__(
        self,
        golden_path: str | Path = DEFAULT_GOLDEN_PATH,
        run_directory: str | Path | None = None,
        device: str | None = None,
    ):
        """
        Initialize the evaluator.

        Args:
            golden_path: Path to golden evaluation dataset (JSONL)
            run_directory: Path to chunking run directory (uses latest if None)
            device: Device for embedding model ("cuda", "cpu", etc.)
        """
        self.golden_path = Path(golden_path)
        self.golden_queries = load_golden_dataset(self.golden_path)

        print(f"Loading retriever from: {run_directory or 'latest run'}")
        self.retriever = DenseRetriever(run_directory, device=device)
        print(f"  Loaded {len(self.retriever.chunks)} chunks")
        print(
            f"  Model: {self.retriever.model_name} "
            f"({self.retriever.embedder.backend})"
        )
        print(f"Loaded {len(self.golden_queries)} golden queries")

    def evaluate_query(
        self,
        golden: GoldenQuery,
        k: int,
    ) -> QueryResult:
        """
        Evaluate a single query.

        Args:
            golden: The golden query to evaluate
            k: Number of results to retrieve

        Returns:
            QueryResult with all metrics and match details
        """
        # Run retrieval
        results = self.retriever.search(
            golden.query,
            k=k,
            doc_id=golden.doc_id,  # Filter by doc if specified
        )

        # Match results to golden answers
        matches = []
        relevance_scores, found_relevant_items = golden.score_ranking(
            [(chunk.get("element_orders", []), chunk.get("doc_id")) for _, chunk in results]
        )

        for rank, (score, chunk) in enumerate(results, start=1):
            chunk_orders = chunk.get("element_orders", [])
            chunk_doc_id = chunk.get("doc_id")

            # Get relevance grade for this chunk
            grade = golden.get_relevance_grade(chunk_orders, chunk_doc_id)

            if grade > 0:
                # Find which orders matched
                matched = []
                for item in golden.relevant:
                    if item["match_by"] == "element_orders":
                        item_doc = item.get("doc_id")
                        if item_doc is None or chunk_doc_id is None or item_doc == chunk_doc_id:
                            matched.extend(
                                o for o in item["value"]
                                if o in chunk_orders
                            )

                matches.append(RelevanceMatch(
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    element_orders=chunk_orders,
                    matched_orders=list(set(matched)),
                    relevance_grade=grade,
                    rank=rank,
                    score=score,
                ))

        # Compute metrics
        total_relevant = len(golden.relevant)
        ideal_scores = golden.get_all_relevance_grades()

        return QueryResult(
            query_id=golden.id,
            query=golden.query,
            category=golden.category,
            difficulty=golden.difficulty,
            doc_id=golden.doc_id,
            retrieved_count=len(results),
            relevant_retrieved=len(found_relevant_items),
            total_relevant=total_relevant,
            hit=compute_hit(relevance_scores),
            reciprocal_rank=compute_reciprocal_rank(relevance_scores),
            recall=compute_recall(len(found_relevant_items), total_relevant),
            precision=compute_precision(len(matches), len(results)),
            ndcg=compute_ndcg(relevance_scores, ideal_scores, k),
            matches=matches,
            top_chunks=[
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "score": float(score),
                    "element_orders": chunk.get("element_orders", []),
                    "breadcrumb": chunk.get("breadcrumb", "")[:100],
                }
                for score, chunk in results[:5]  # Keep top 5 for debugging
            ],
        )

    def evaluate(
        self,
        k_values: list[int] | None = None,
        verbose: bool = True,
    ) -> EvalResults:
        """
        Run full evaluation across all queries and k values.

        Args:
            k_values: List of k values to evaluate (default: [1, 3, 5, 10])
            verbose: Print progress

        Returns:
            EvalResults with all metrics and breakdowns
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Get config from retriever manifest
        manifest = self.retriever.manifest
        chunking_config = manifest.get("chunking_config", {})
        embedding_model = {
            **manifest.get("embedding_model", {}),
            "query_backend": self.retriever.embedder.backend,
        }

        # Storage for results
        query_results: dict[int, list[QueryResult]] = {k: [] for k in k_values}

        # Run evaluation for each k
        max_k = max(k_values)

        if verbose:
            print(f"\nEvaluating {len(self.golden_queries)} queries at k={k_values}...")

        for i, golden in enumerate(self.golden_queries):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(self.golden_queries)} queries")

            # Evaluate at max k, then slice for smaller k values
            result_max_k = self.evaluate_query(golden, max_k)

            for k in k_values:
                if k == max_k:
                    query_results[k].append(result_max_k)
                else:
                    # Re-evaluate at smaller k
                    result_k = self.evaluate_query(golden, k)
                    query_results[k].append(result_k)

        # Aggregate metrics
        metrics_by_k = {}
        for k in k_values:
            results = query_results[k]
            metrics_by_k[k] = self._aggregate_metrics(results)

        # Breakdowns by category
        metrics_by_category = self._compute_breakdown(
            query_results, k_values, "category"
        )

        # Breakdowns by difficulty
        metrics_by_difficulty = self._compute_breakdown(
            query_results, k_values, "difficulty"
        )

        # Breakdowns by document
        metrics_by_doc = self._compute_breakdown(
            query_results, k_values, "doc_id"
        )

        # Determine primary k (prefer 5, else use largest available)
        primary_k = 5 if 5 in k_values else max(k_values)

        # Build overall metrics - the headline numbers at primary_k
        primary_metrics = metrics_by_k.get(primary_k, {})
        overall = {
            "k": primary_k,
            "hit_rate": primary_metrics.get("hit_rate", 0.0),
            "mrr": primary_metrics.get("mrr", 0.0),
            "recall": primary_metrics.get("recall", 0.0),
            "precision": primary_metrics.get("precision", 0.0),
            "ndcg": primary_metrics.get("ndcg", 0.0),
            "positive_queries": primary_metrics.get("positive_queries", 0),
            "negative_queries": primary_metrics.get("negative_queries", 0),
        }

        return EvalResults(
            timestamp=timestamp,
            run_directory=str(self.retriever.run_directory),
            golden_path=str(self.golden_path),
            k_values=k_values,
            primary_k=primary_k,
            chunking_config=chunking_config,
            embedding_model=embedding_model,
            total_chunks=len(self.retriever.chunks),
            total_queries=len(self.golden_queries),
            overall=overall,
            metrics_by_k=metrics_by_k,
            metrics_by_category=metrics_by_category,
            metrics_by_difficulty=metrics_by_difficulty,
            metrics_by_doc=metrics_by_doc,
            query_results=query_results,
        )

    def _aggregate_metrics(self, results: list[QueryResult]) -> dict[str, float]:
        """Compute aggregate metrics from query results."""
        if not results:
            return {}

        # Filter out negative queries for most metrics
        # (negative queries have no relevant docs, so hit/recall/etc. are meaningless)
        positive_results = [r for r in results if r.total_relevant > 0]
        negative_results = [r for r in results if r.total_relevant == 0]

        metrics = {
            "total_queries": len(results),
            "positive_queries": len(positive_results),
            "negative_queries": len(negative_results),
        }

        if positive_results:
            metrics.update({
                "hit_rate": np.mean([r.hit for r in positive_results]),
                "mrr": np.mean([r.reciprocal_rank for r in positive_results]),
                "recall": np.mean([r.recall for r in positive_results]),
                "precision": np.mean([r.precision for r in positive_results]),
                "ndcg": np.mean([r.ndcg for r in positive_results]),
            })

            # Standard deviations
            metrics.update({
                "hit_rate_std": float(np.std([r.hit for r in positive_results])),
                "mrr_std": float(np.std([r.reciprocal_rank for r in positive_results])),
                "recall_std": float(np.std([r.recall for r in positive_results])),
                "precision_std": float(np.std([r.precision for r in positive_results])),
                "ndcg_std": float(np.std([r.ndcg for r in positive_results])),
            })

        # For negative queries, check if retriever correctly returns low-confidence results
        if negative_results:
            # Average score of top result for negative queries
            # (lower is better - we don't want confident wrong answers)
            top_scores = []
            for r in negative_results:
                if r.top_chunks:
                    top_scores.append(r.top_chunks[0]["score"])
            if top_scores:
                metrics["negative_avg_top_score"] = float(np.mean(top_scores))

        return metrics

    def _compute_breakdown(
        self,
        query_results: dict[int, list[QueryResult]],
        k_values: list[int],
        field: str,
    ) -> dict[str, dict[int, dict[str, float]]]:
        """Compute metrics breakdown by a field (category, difficulty, doc_id)."""
        breakdown: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)

        for k in k_values:
            # Group by field value
            groups: dict[str, list[QueryResult]] = defaultdict(list)
            for result in query_results[k]:
                value = getattr(result, field)
                if value is None:
                    value = "cross_document"  # For null doc_id
                groups[value].append(result)

            # Compute metrics for each group
            for value, results in groups.items():
                breakdown[value][k] = self._aggregate_metrics(results)

        return dict(breakdown)

    def save_results(
        self,
        results: EvalResults,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Save evaluation results to JSON file.

        Args:
            results: EvalResults to save
            output_dir: Directory for output (default: eval/results/)

        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = DEFAULT_RESULTS_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_timestamp = Path(results.run_directory).name
        output_path = output_dir / f"{run_timestamp}_eval_results.json"
        version = 2
        while output_path.exists():
            output_path = output_dir / f"{run_timestamp}_v{version}_eval_results.json"
            version += 1

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")
        return output_path

    def print_summary(self, results: EvalResults) -> None:
        """Print a human-readable summary of evaluation results."""
        print("\n" + "=" * 80)
        print("RETRIEVAL EVALUATION SUMMARY")
        print("=" * 80)

        print("\nConfiguration:")
        print(f"  Run directory: {results.run_directory}")
        print(f"  Total chunks: {results.total_chunks}")
        print(f"  Total queries: {results.total_queries} "
              f"({results.overall.get('positive_queries', 0)} positive, "
              f"{results.overall.get('negative_queries', 0)} negative)")
        print(f"  Chunking: MAX_CHARS={results.chunking_config.get('max_chars')}, "
              f"TABLE_MAX_CHARS={results.chunking_config.get('table_max_chars')}")
        print(f"  Model: {results.embedding_model.get('name')}")

        # Overall metrics - the headline numbers
        print("\n" + "-" * 80)
        print(f"OVERALL METRICS (k={results.primary_k})")
        print("-" * 80)
        o = results.overall
        print(f"  Hit Rate:   {o.get('hit_rate', 0):.3f}  "
              f"({o.get('hit_rate', 0)*100:.1f}% of queries found relevant results)")
        print(f"  MRR:        {o.get('mrr', 0):.3f}  "
              f"(first relevant result at ~rank {1/o.get('mrr', 0.001):.1f} on average)")
        print(f"  Recall:     {o.get('recall', 0):.3f}  "
              f"({o.get('recall', 0)*100:.1f}% of all relevant chunks retrieved)")
        print(f"  Precision:  {o.get('precision', 0):.3f}  "
              f"({o.get('precision', 0)*100:.1f}% of retrieved chunks are relevant)")
        print(f"  nDCG:       {o.get('ndcg', 0):.3f}  "
              f"(ranking quality score)")

        # Detailed metrics by k
        print("\nMetrics by k:")
        print(f"{'Metric':<20} | " + " | ".join(f"k={k:>2}" for k in results.k_values))
        print("-" * (22 + 8 * len(results.k_values)))

        metrics_to_show = ["hit_rate", "mrr", "recall", "precision", "ndcg"]
        for metric in metrics_to_show:
            values = []
            for k in results.k_values:
                val = results.metrics_by_k[k].get(metric, 0)
                values.append(f"{val:.3f}")
            print(f"{metric:<20} | " + " | ".join(f"{v:>5}" for v in values))

        # Category breakdown for k=5
        print("\n\nBreakdown by Category (k=5):")
        print("-" * 60)
        k = 5 if 5 in results.k_values else results.k_values[-1]
        for category, metrics in sorted(results.metrics_by_category.items()):
            if k in metrics:
                m = metrics[k]
                print(f"  {category:<25} | "
                      f"Hit={m.get('hit_rate', 0):.2f} "
                      f"MRR={m.get('mrr', 0):.2f} "
                      f"Recall={m.get('recall', 0):.2f} "
                      f"(n={m.get('positive_queries', 0)})")

        # Difficulty breakdown
        print("\nBreakdown by Difficulty (k=5):")
        print("-" * 60)
        for diff in ["easy", "medium", "hard"]:
            if diff in results.metrics_by_difficulty:
                metrics = results.metrics_by_difficulty[diff]
                if k in metrics:
                    m = metrics[k]
                    print(f"  {diff:<25} | "
                          f"Hit={m.get('hit_rate', 0):.2f} "
                          f"MRR={m.get('mrr', 0):.2f} "
                          f"Recall={m.get('recall', 0):.2f} "
                          f"(n={m.get('positive_queries', 0)})")

        print("\n" + "=" * 80)


# =============================================================================
# Comparison Utility
# =============================================================================


def load_eval_results(path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def compare_runs(results_dir: Path | None = None) -> None:
    """
    Compare all evaluation runs in the results directory.

    Prints a table comparing key metrics across runs.
    """
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"No results directory found: {results_dir}")
        return

    result_files = sorted(results_dir.glob("*_eval_results.json"))
    if not result_files:
        print(f"No evaluation results found in {results_dir}")
        return

    print("\n" + "=" * 100)
    print("EVALUATION RUN COMPARISON")
    print("=" * 100)

    # Header
    print(f"\n{'Timestamp':<20} | {'MAX_CHARS':>9} | {'TBL_CHARS':>9} | "
          f"{'Hit@5':>6} | {'MRR@5':>6} | {'Recall@5':>8} | {'Prec@5':>6} | {'nDCG@5':>7} | {'Chunks':>7}")
    print("-" * 100)

    for path in result_files:
        try:
            data = load_eval_results(path)
            ts = data["timestamp"][:16]  # Truncate for display
            config = data.get("chunking_config", {})
            max_chars = config.get("max_chars", "?")
            tbl_chars = config.get("table_max_chars", "?")
            chunks = data.get("total_chunks", "?")

            # Prefer overall metrics (new format), fall back to metrics_by_k
            overall = data.get("overall", {})
            if overall:
                hit = overall.get("hit_rate", 0)
                mrr = overall.get("mrr", 0)
                recall = overall.get("recall", 0)
                precision = overall.get("precision", 0)
                ndcg = overall.get("ndcg", 0)
            else:
                # Fall back to metrics_by_k for older result files
                k_metrics = data.get("metrics_by_k", {})
                k = "5" if "5" in k_metrics else list(k_metrics.keys())[-1] if k_metrics else None
                if k:
                    m = k_metrics[k]
                    hit = m.get("hit_rate", 0)
                    mrr = m.get("mrr", 0)
                    recall = m.get("recall", 0)
                    precision = m.get("precision", 0)
                    ndcg = m.get("ndcg", 0)
                else:
                    hit = mrr = recall = precision = ndcg = 0

            print(f"{ts:<20} | {max_chars:>9} | {tbl_chars:>9} | "
                  f"{hit:>6.3f} | {mrr:>6.3f} | {recall:>8.3f} | {precision:>6.3f} | {ndcg:>7.3f} | {chunks:>7}")

        except Exception as e:
            print(f"Error loading {path.name}: {e}")

    print("\n" + "=" * 100)


# =============================================================================
# Failed Queries Analysis
# =============================================================================


def analyze_failures(
    results: EvalResults,
    k: int = 5,
    min_relevance: int = 3,
) -> None:
    """
    Analyze queries where retrieval failed to find highly relevant documents.

    Args:
        results: Evaluation results
        k: k value to analyze
        min_relevance: Minimum relevance grade to consider as "success"
    """
    print("\n" + "=" * 80)
    print(f"FAILURE ANALYSIS (k={k}, min_relevance={min_relevance})")
    print("=" * 80)

    if k not in results.query_results:
        print(f"No results for k={k}")
        return

    failures = []
    for qr in results.query_results[k]:
        # Skip negative queries
        if qr.total_relevant == 0:
            continue

        # Check if we found a highly relevant result
        found_high_relevance = any(
            m.relevance_grade >= min_relevance for m in qr.matches
        )

        if not found_high_relevance:
            failures.append(qr)

    print(f"\nTotal failures: {len(failures)} / {len(results.query_results[k])}")

    if not failures:
        print("No failures to analyze!")
        return

    # Group by category
    by_category: dict[str, list[QueryResult]] = defaultdict(list)
    for qr in failures:
        by_category[qr.category].append(qr)

    print("\nFailures by category:")
    for cat, qrs in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(qrs)}")

    print("\nSample failures:")
    for qr in failures[:5]:
        print(f"\n  Query: {qr.query[:80]}...")
        print(f"  Category: {qr.category}, Difficulty: {qr.difficulty}")
        print(f"  Expected {qr.total_relevant} relevant, found {qr.relevant_retrieved}")
        if qr.top_chunks:
            print(f"  Top result: score={qr.top_chunks[0]['score']:.3f}, "
                  f"orders={qr.top_chunks[0]['element_orders'][:3]}...")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Run evaluation, run comparison, or failure analysis from the CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate dense retriever against golden dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings
  uv run python eval/dense_retriever_eval.py

  # Evaluate specific run at multiple k values
  uv run python eval/dense_retriever_eval.py --run-directory data/chunks/2026-10-14_15-50-05 -k 1 3 5 10 20

  # Compare all previous runs
  uv run python eval/dense_retriever_eval.py --compare

  # Analyze failures from last run
  uv run python eval/dense_retriever_eval.py --analyze-failures
        """,
    )

    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden evaluation dataset (JSONL)",
    )
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="Path to chunking run directory (uses latest if not specified)",
    )
    parser.add_argument(
        "-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="k values to evaluate (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--device",
        help="Device for embedding model (cuda, cpu, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for output files",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all previous evaluation runs",
    )
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help="Analyze failed queries from evaluation",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )

    args = parser.parse_args()

    if args.compare:
        compare_runs(args.output_dir)
        return

    # Run evaluation
    evaluator = RetrieverEvaluator(
        golden_path=args.golden,
        run_directory=args.run_directory,
        device=args.device,
    )

    results = evaluator.evaluate(k_values=sorted(args.k))
    evaluator.print_summary(results)

    if args.analyze_failures:
        k = 5 if 5 in args.k else args.k[-1]
        analyze_failures(results, k=k)

    if not args.no_save:
        evaluator.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
