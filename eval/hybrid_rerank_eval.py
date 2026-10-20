"""
Hybrid Retrieval Evaluation Module
===================================

This module extends retrieval_eval.py to support:
1. Dense retrieval (baseline)
2. BM25 retrieval (sparse)
3. Hybrid retrieval (dense + BM25)
4. Hybrid + Reranker (full pipeline)

It also supports alpha tuning to find the optimal dense/BM25 weight.

USAGE
-----

# Evaluate hybrid retrieval with default alpha=0.5
uv run python eval/hybrid_rerank_eval.py --mode hybrid

# Evaluate with reranker
uv run python eval/hybrid_rerank_eval.py --mode rerank

# Tune alpha (test multiple values)
uv run python eval/hybrid_rerank_eval.py --tune-alpha

# Compare all runs
uv run python eval/hybrid_rerank_eval.py --compare
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
from retriever.bm25s_rrf_retriever import BM25SRRFRetriever, BM25SRRFRetrieverWithReranker
from retriever.hybrid_retriever import (
    BM25Retriever,
    HybridRetriever,
    HybridRetrieverWithReranker,
)
from retriever.retriever import DenseRetriever, SearchResult

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_PATH = ROOT / "goldens" / "golden_retriever_dataset_v2.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "retriver_results"


# =============================================================================
# Data Classes (same as retrieval_eval.py)
# =============================================================================


@dataclass
class RelevanceMatch:
    """Represents a relevance match between a retrieved chunk and golden answer."""
    chunk_id: str
    element_orders: list[int]
    matched_orders: list[int]
    relevance_grade: int
    rank: int
    score: float


@dataclass
class QueryResult:
    """Evaluation result for a single query."""
    query_id: str
    query: str
    category: str
    difficulty: str
    doc_id: str | None
    retrieved_count: int
    relevant_retrieved: int
    total_relevant: int
    hit: bool
    reciprocal_rank: float
    recall: float
    precision: float
    ndcg: float
    matches: list[RelevanceMatch] = field(default_factory=list)
    top_chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalResults:
    """Complete evaluation results."""
    timestamp: str
    run_directory: str
    golden_path: str
    k_values: list[int]
    primary_k: int
    chunking_config: dict[str, Any]
    embedding_model: dict[str, Any]
    retrieval_config: dict[str, Any]  # NEW: mode, alpha, etc.
    total_chunks: int
    total_queries: int
    overall: dict[str, float]
    metrics_by_k: dict[int, dict[str, float]]
    metrics_by_category: dict[str, dict[int, dict[str, float]]]
    metrics_by_difficulty: dict[str, dict[int, dict[str, float]]]
    metrics_by_doc: dict[str, dict[int, dict[str, float]]]
    query_results: dict[int, list[QueryResult]]

    def to_dict(self) -> dict[str, Any]:
        """Convert nested evaluation results into a JSON-serializable dictionary."""
        result = {
            "timestamp": self.timestamp,
            "run_directory": self.run_directory,
            "golden_path": self.golden_path,
            "k_values": self.k_values,
            "primary_k": self.primary_k,
            "chunking_config": self.chunking_config,
            "embedding_model": self.embedding_model,
            "retrieval_config": self.retrieval_config,
            "total_chunks": self.total_chunks,
            "total_queries": self.total_queries,
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
    """Compute Discounted Cumulative Gain from ranked relevance grades at ``k``."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def compute_ndcg(relevance_scores: list[int], ideal_scores: list[int], k: int) -> float:
    """Compute DCG normalized by the ideal ranking at ``k``."""
    dcg = compute_dcg(relevance_scores, k)
    ideal_sorted = sorted(ideal_scores, reverse=True)
    idcg = compute_dcg(ideal_sorted, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_hit(relevance_scores: list[int]) -> bool:
    """Return whether a ranked result list contains any relevant item."""
    return any(r > 0 for r in relevance_scores)


def compute_reciprocal_rank(relevance_scores: list[int]) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall(relevant_found: int, total_relevant: int) -> float:
    """Return the fraction of all relevant items that were retrieved."""
    return relevant_found / total_relevant if total_relevant > 0 else 0.0


def compute_precision(relevant_found: int, retrieved: int) -> float:
    """Return the fraction of retrieved chunks that contain a relevant element (0-1)."""
    return relevant_found / retrieved if retrieved > 0 else 0.0


# =============================================================================
# Golden Dataset
# =============================================================================


@dataclass
class GoldenQuery:
    """A labeled query and its expected relevant filing elements."""

    id: str
    query: str
    doc_id: str | None
    category: str
    difficulty: str
    expected_answer: str
    relevant: list[dict[str, Any]]

    def is_negative(self) -> bool:
        """Return whether the query intentionally has no relevant answer."""
        return len(self.relevant) == 0

    def get_relevance_grade(self, chunk_orders: list[int], doc_id: str | None = None) -> int:
        """Return the highest relevance grade matching a chunk's element orders."""
        if not chunk_orders:
            return 0
        chunk_orders = set(chunk_orders)
        max_grade = 0
        for item in self.relevant:
            if item["match_by"] == "element_orders":
                item_doc = item.get("doc_id")
                if item_doc is not None and doc_id is not None and item_doc != doc_id:
                    continue
                expected_orders = set(item["value"])
                if chunk_orders & expected_orders:
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
        """Return all expected relevance grades for ideal-ranking metrics."""
        return [item["relevance"] for item in self.relevant]


def load_golden_dataset(path: Path) -> list[GoldenQuery]:
    """Load labeled golden queries from a JSON Lines file."""
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
# Unified Evaluator
# =============================================================================


class HybridEvaluator:
    """
    Evaluates different retrieval modes against a golden dataset.

    Supported modes:
    - "dense": Pure dense retrieval (baseline)
    - "bm25": Pure BM25 retrieval
    - "hybrid": Dense + BM25 combined
    - "rerank": Hybrid + cross-encoder reranking
    - "rrf": Dense + BM25S fused with reciprocal rank fusion
    - "rrf_rerank": RRF + cross-encoder reranking
    """

    def __init__(
        self,
        golden_path: str | Path = DEFAULT_GOLDEN_PATH,
        run_directory: str | Path | None = None,
        mode: str = "hybrid",
        alpha: float = 0.5,
        retrieve_k: int = 20,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        rrf_k: int = 60,
        dense_k: int = 50,
        bm25_k: int = 50,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """Load the golden set and initialize the selected retrieval pipeline.

        Args:
            golden_path: JSON Lines file containing labeled evaluation queries.
            run_directory: Retrieval artifact directory, or ``None`` for latest.
            mode: Retrieval strategy to evaluate.
            alpha: Dense-score weight for normalized hybrid retrieval.
            retrieve_k: Candidate count used before cross-encoder reranking.
            reranker_model: Cross-encoder model used by reranking mode.
            device: Model inference device, such as ``cuda`` or ``cpu``.
            rrf_k: RRF smoothing constant (rrf modes only).
            dense_k: Dense candidates fed into RRF (rrf modes only).
            bm25_k: BM25 candidates fed into RRF (rrf modes only).
            dense_weight: Dense ranking weight in RRF (rrf modes only).
            bm25_weight: BM25 ranking weight in RRF (rrf modes only).

        Raises:
            ValueError: If ``mode`` is unsupported.
        """
        self.golden_path = Path(golden_path)
        self.golden_queries = load_golden_dataset(self.golden_path)
        self.mode = mode
        self.alpha = alpha
        self.retrieve_k = retrieve_k
        self.reranker_model = reranker_model
        self.rrf_config = {
            "rrf_k": rrf_k,
            "dense_k": dense_k,
            "bm25_k": bm25_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
        }

        print(f"Loading retriever (mode={mode}, alpha={alpha})...")

        if mode == "dense":
            self.retriever = DenseRetriever(run_directory, device=device)
        elif mode == "bm25":
            self.retriever = BM25Retriever(run_directory)
        elif mode == "hybrid":
            self.retriever = HybridRetriever(
                run_directory, alpha=alpha, device=device
            )
        elif mode == "rerank":
            self.retriever = HybridRetrieverWithReranker(
                run_directory,
                alpha=alpha,
                retrieve_k=retrieve_k,
                reranker_model=reranker_model,
                device=device,
            )
        elif mode == "rrf":
            self.retriever = BM25SRRFRetriever(run_directory, device=device, **self.rrf_config)
        elif mode == "rrf_rerank":
            self.retriever = BM25SRRFRetrieverWithReranker(
                run_directory,
                retrieve_k=retrieve_k,
                reranker_model=reranker_model,
                device=device,
                **self.rrf_config,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.chunks = self.retriever.chunks
        self.run_directory = self.retriever.run_directory
        self.manifest = self.retriever.manifest

        print(f"  Loaded {len(self.chunks)} chunks")
        print(f"Loaded {len(self.golden_queries)} golden queries")

    def _search(self, query: str, k: int, doc_id: str | None) -> list[SearchResult]:
        """Unified search interface for all retriever types."""
        return self.retriever.search(query, k=k, doc_id=doc_id)

    def evaluate_query(self, golden: GoldenQuery, k: int) -> QueryResult:
        """Evaluate a single query."""
        results = self._search(golden.query, k=k, doc_id=golden.doc_id)

        matches = []
        relevance_scores, found_relevant_items = golden.score_ranking(
            [(chunk.get("element_orders", []), chunk.get("doc_id")) for _, chunk in results]
        )

        for rank, (score, chunk) in enumerate(results, start=1):
            chunk_orders = chunk.get("element_orders", [])
            chunk_doc_id = chunk.get("doc_id")
            grade = golden.get_relevance_grade(chunk_orders, chunk_doc_id)

            if grade > 0:
                matched = []
                for item in golden.relevant:
                    if item["match_by"] == "element_orders":
                        item_doc = item.get("doc_id")
                        if item_doc is None or chunk_doc_id is None or item_doc == chunk_doc_id:
                            matched.extend(o for o in item["value"] if o in chunk_orders)

                matches.append(RelevanceMatch(
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    element_orders=chunk_orders,
                    matched_orders=list(set(matched)),
                    relevance_grade=grade,
                    rank=rank,
                    score=score,
                ))

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
                for score, chunk in results[:5]
            ],
        )

    def evaluate(
        self,
        k_values: list[int] | None = None,
        verbose: bool = True,
    ) -> EvalResults:
        """Run full evaluation."""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        chunking_config = self.manifest.get("chunking_config", {})
        embedding_model = self.manifest.get("embedding_model", {})

        retrieval_config = {
            "mode": self.mode,
            "alpha": self.alpha if self.mode in ("hybrid", "rerank") else None,
            "retrieve_k": self.retrieve_k if self.mode in ("rerank", "rrf_rerank") else None,
            "reranker_model": (
                self.reranker_model if self.mode in ("rerank", "rrf_rerank") else None
            ),
            "rrf": self.rrf_config if self.mode in ("rrf", "rrf_rerank") else None,
        }

        query_results: dict[int, list[QueryResult]] = {k: [] for k in k_values}
        max_k = max(k_values)

        if verbose:
            print(f"\nEvaluating {len(self.golden_queries)} queries at k={k_values}...")

        for i, golden in enumerate(self.golden_queries):
            results_at_max_k = self.evaluate_query(golden, max_k)

            for k in k_values:
                if k == max_k:
                    query_results[k].append(results_at_max_k)
                else:
                    query_results[k].append(self.evaluate_query(golden, k))

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(self.golden_queries)} queries")

        if verbose:
            print(f"  Processed {len(self.golden_queries)}/{len(self.golden_queries)} queries")

        # Aggregate metrics
        metrics_by_k = {}
        for k in k_values:
            metrics_by_k[k] = self._aggregate_metrics(query_results[k])

        # Breakdowns
        metrics_by_category = self._compute_breakdowns(query_results, k_values, "category")
        metrics_by_difficulty = self._compute_breakdowns(query_results, k_values, "difficulty")
        metrics_by_doc = self._compute_breakdowns(query_results, k_values, "doc_id")

        # Overall metrics at k=5
        primary_k = 5 if 5 in k_values else k_values[-1]
        overall = {
            "k": primary_k,
            "hit_rate": metrics_by_k[primary_k]["hit_rate"],
            "mrr": metrics_by_k[primary_k]["mrr"],
            "recall": metrics_by_k[primary_k]["recall"],
            "precision": metrics_by_k[primary_k]["precision"],
            "ndcg": metrics_by_k[primary_k]["ndcg"],
            "positive_queries": metrics_by_k[primary_k]["positive_queries"],
            "negative_queries": metrics_by_k[primary_k]["negative_queries"],
        }

        return EvalResults(
            timestamp=timestamp,
            run_directory=str(self.run_directory),
            golden_path=str(self.golden_path),
            k_values=k_values,
            primary_k=primary_k,
            chunking_config=chunking_config,
            embedding_model=embedding_model,
            retrieval_config=retrieval_config,
            total_chunks=len(self.chunks),
            total_queries=len(self.golden_queries),
            overall=overall,
            metrics_by_k={str(k): v for k, v in metrics_by_k.items()},
            metrics_by_category=metrics_by_category,
            metrics_by_difficulty=metrics_by_difficulty,
            metrics_by_doc=metrics_by_doc,
            query_results=query_results,
        )

    def _aggregate_metrics(self, results: list[QueryResult]) -> dict[str, float]:
        """Aggregate metrics across queries."""
        positive = [r for r in results if r.total_relevant > 0]
        negative = [r for r in results if r.total_relevant == 0]

        if not positive:
            return {
                "total_queries": len(results),
                "positive_queries": 0,
                "negative_queries": len(negative),
            }

        metrics = {
            "total_queries": len(results),
            "positive_queries": len(positive),
            "negative_queries": len(negative),
            "hit_rate": np.mean([r.hit for r in positive]),
            "mrr": np.mean([r.reciprocal_rank for r in positive]),
            "recall": np.mean([r.recall for r in positive]),
            "precision": np.mean([r.precision for r in positive]),
            "ndcg": np.mean([r.ndcg for r in positive]),
            "hit_rate_std": np.std([r.hit for r in positive]),
            "mrr_std": np.std([r.reciprocal_rank for r in positive]),
            "recall_std": np.std([r.recall for r in positive]),
            "precision_std": np.std([r.precision for r in positive]),
            "ndcg_std": np.std([r.ndcg for r in positive]),
        }

        return metrics

    def _compute_breakdowns(
        self,
        query_results: dict[int, list[QueryResult]],
        k_values: list[int],
        group_by: str,
    ) -> dict[str, dict[int, dict[str, float]]]:
        """Compute metrics broken down by a grouping field."""
        breakdowns: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)

        for k in k_values:
            groups: dict[str, list[QueryResult]] = defaultdict(list)
            for r in query_results[k]:
                key = getattr(r, group_by) or "cross_document"
                groups[key].append(r)

            for group_name, group_results in groups.items():
                breakdowns[group_name][k] = self._aggregate_metrics(group_results)

        return dict(breakdowns)

    def print_summary(self, results: EvalResults) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("RETRIEVAL EVALUATION SUMMARY")
        print("=" * 80)

        print("\nConfiguration:")
        print(f"  Run directory: {results.run_directory}")
        print(f"  Total chunks: {results.total_chunks}")
        print(f"  Total queries: {results.total_queries} "
              f"({results.overall['positive_queries']} positive, "
              f"{results.overall['negative_queries']} negative)")

        config = results.chunking_config
        print(f"  Chunking: MAX_CHARS={config.get('max_chars')}, "
              f"TABLE_MAX_CHARS={config.get('table_max_chars')}")

        model = results.embedding_model
        print(f"  Model: {model.get('name')}")

        ret_config = results.retrieval_config
        print(f"  Retrieval mode: {ret_config.get('mode')}")
        if ret_config.get('alpha') is not None:
            print(f"  Alpha: {ret_config.get('alpha')}")
        if ret_config.get('retrieve_k') is not None:
            print(f"  Retrieve-k: {ret_config.get('retrieve_k')}")
        if ret_config.get('reranker_model'):
            print(f"  Reranker: {ret_config.get('reranker_model')}")

        # Overall metrics
        print("\n" + "-" * 80)
        print(f"OVERALL METRICS (k={results.primary_k})")
        print("-" * 80)
        overall = results.overall
        print(f"  Hit Rate:   {overall['hit_rate']:.3f}  "
              f"({overall['hit_rate']*100:.1f}% of queries found relevant results)")
        mrr = overall['mrr']
        avg_rank = 1/mrr if mrr > 0 else float('inf')
        print(f"  MRR:        {mrr:.3f}  "
              f"(first relevant result at ~rank {avg_rank:.1f} on average)")
        print(f"  Recall:     {overall['recall']:.3f}  "
              f"({overall['recall']*100:.1f}% of all relevant chunks retrieved)")
        print(f"  Precision:  {overall['precision']:.3f}  "
              f"({overall['precision']*100:.1f}% of retrieved chunks are relevant)")
        print(f"  nDCG:       {overall['ndcg']:.3f}  (ranking quality score)")

        # Metrics by k
        print("\nMetrics by k:")
        header = f"{'Metric':<20} |"
        for k in results.k_values:
            header += f" k={k:>2} |"
        print(header)
        print("-" * len(header))

        for metric in ["hit_rate", "mrr", "recall", "precision", "ndcg"]:
            row = f"{metric:<20} |"
            for k in results.k_values:
                val = results.metrics_by_k[str(k)].get(metric, 0)
                row += f" {val:>.3f} |"
            print(row)

        # Category breakdown
        print(f"\n\nBreakdown by Category (k={results.primary_k}):")
        print("-" * 60)
        for cat, metrics in sorted(results.metrics_by_category.items()):
            m = metrics.get(results.primary_k) or metrics.get(str(results.primary_k), {})
            if m:
                print(f"  {cat:<25} | Hit={m.get('hit_rate', 0):.2f} "
                      f"MRR={m.get('mrr', 0):.2f} "
                      f"Recall={m.get('recall', 0):.2f} "
                      f"(n={m.get('positive_queries', 0)})")

        # Difficulty breakdown
        print(f"\nBreakdown by Difficulty (k={results.primary_k}):")
        print("-" * 60)
        for diff in ["easy", "medium", "hard"]:
            if diff in results.metrics_by_difficulty:
                by_k = results.metrics_by_difficulty[diff]
                m = by_k.get(results.primary_k) or by_k.get(str(results.primary_k), {})
                if m:
                    print(f"  {diff:<25} | Hit={m.get('hit_rate', 0):.2f} "
                          f"MRR={m.get('mrr', 0):.2f} "
                          f"Recall={m.get('recall', 0):.2f} "
                          f"(n={m.get('positive_queries', 0)})")

        print("\n" + "=" * 80)

    def save_results(self, results: EvalResults, output_dir: Path | None = None) -> Path:
        """Save results to JSON file."""
        if output_dir is None:
            output_dir = DEFAULT_RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Include mode, alpha, and reranker in filename
        mode = results.retrieval_config.get("mode", "unknown")
        alpha = results.retrieval_config.get("alpha")
        alpha_str = f"_a{alpha:.2f}" if alpha is not None else ""
        reranker = results.retrieval_config.get("reranker_model")
        reranker_str = f"_{reranker.split('/')[-1]}" if reranker else ""

        rrf = results.retrieval_config.get("rrf")
        rrf_str = (
            f"_w{rrf['dense_weight']:g}-{rrf['bm25_weight']:g}_k{rrf['rrf_k']}" if rrf else ""
        )

        filename = (
            f"{results.timestamp}_{mode}{alpha_str}{rrf_str}{reranker_str}_eval_results.json"
        )
        output_path = output_dir / filename

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")
        return output_path


# =============================================================================
# Alpha Tuning
# =============================================================================


def tune_alpha(
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    run_directory: str | Path | None = None,
    alpha_values: list[float] | None = None,
    k: int = 5,
    device: str | None = None,
) -> dict[float, dict[str, float]]:
    """
    Find the optimal alpha for hybrid retrieval.

    Tests multiple alpha values and reports metrics for each.

    Args:
        golden_path: Path to golden dataset
        run_directory: Path to chunking run
        alpha_values: List of alpha values to test (default: 0.0 to 1.0 by 0.1)
        k: k value for evaluation
        device: Device for models

    Returns:
        Dictionary mapping alpha -> metrics
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.85, 0.9, 1.0]

    print("\n" + "=" * 80)
    print("ALPHA TUNING")
    print("=" * 80)
    print(f"\nTesting alpha values: {alpha_values}")
    print(f"Evaluation k: {k}")

    results: dict[float, dict[str, float]] = {}

    # Load golden dataset once
    golden_queries = load_golden_dataset(golden_path)

    # Load retriever components once
    print("\nLoading retriever components...")
    from retriever.hybrid_retriever import HybridRetriever

    # Create a base retriever to reuse
    base_retriever = HybridRetriever(
        run_directory, alpha=0.5, device=device
    )

    for alpha in alpha_values:
        print(f"\n--- Alpha = {alpha:.1f} ---")

        # Evaluate with this alpha
        hit_rates = []
        mrrs = []
        recalls = []
        precisions = []

        for golden in golden_queries:
            if golden.is_negative():
                continue

            # Search with this alpha
            search_results = base_retriever.search(
                golden.query, k=k, doc_id=golden.doc_id, alpha=alpha
            )

            # Compute metrics
            relevance_scores, found_items = golden.score_ranking(
                [(chunk.get("element_orders", []), chunk.get("doc_id")) for _, chunk in search_results]
            )

            hit_rates.append(1 if any(r > 0 for r in relevance_scores) else 0)
            mrrs.append(compute_reciprocal_rank(relevance_scores))
            recalls.append(len(found_items) / len(golden.relevant) if golden.relevant else 0)
            relevant_chunks = sum(
                golden.get_relevance_grade(chunk.get("element_orders", []), chunk.get("doc_id")) > 0
                for _, chunk in search_results
            )
            precisions.append(compute_precision(relevant_chunks, len(search_results)))

        metrics = {
            "hit_rate": np.mean(hit_rates),
            "mrr": np.mean(mrrs),
            "recall": np.mean(recalls),
            "precision": np.mean(precisions),
        }
        results[alpha] = metrics

        print(f"  Hit@{k}={metrics['hit_rate']:.3f}  "
              f"MRR={metrics['mrr']:.3f}  "
              f"Recall={metrics['recall']:.3f}  "
              f"Precision={metrics['precision']:.3f}")

    # Find best alpha
    print("\n" + "=" * 80)
    print("ALPHA TUNING RESULTS")
    print("=" * 80)

    print(f"\n{'Alpha':<8} | {'Hit@'+str(k):<8} | {'MRR':<8} | {'Recall':<8} | {'Precision':<8}")
    print("-" * 60)
    for alpha in alpha_values:
        m = results[alpha]
        print(f"{alpha:<8.1f} | {m['hit_rate']:<8.3f} | {m['mrr']:<8.3f} | "
              f"{m['recall']:<8.3f} | {m['precision']:<8.3f}")

    # Best by each metric
    print("\nBest alpha by metric:")
    for metric in ["hit_rate", "mrr", "recall", "precision"]:
        best_alpha = max(alpha_values, key=lambda a: results[a][metric])
        print(f"  {metric}: alpha={best_alpha:.1f} ({results[best_alpha][metric]:.3f})")

    # Overall best (average of normalized metrics)
    def score(alpha):
        """Return the unweighted mean of metrics used to select an alpha."""
        m = results[alpha]
        return (m["hit_rate"] + m["mrr"] + m["recall"] + m["precision"]) / 4

    best_overall = max(alpha_values, key=score)
    print(f"\nRecommended alpha (balanced): {best_overall:.1f}")

    return results


# =============================================================================
# Comparison
# =============================================================================


def load_eval_results(path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def compare_runs(results_dir: Path | None = None) -> None:
    """Compare all evaluation runs."""
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

    width = 145
    print("\n" + "=" * width)
    print("EVALUATION RUN COMPARISON")
    print("=" * width)

    header = (f"\n{'Timestamp':<20} | {'Mode':<10} | {'Alpha':<6} | "
              f"{'Ret-k':<6} | {'Reranker':<22} | {'MaxCh':>5} | {'TblCh':>5} | "
              f"{'Hit@5':>6} | {'MRR@5':>6} | {'Recall@5':>8} | {'Prec@5':>6} | "
              f"{'nDCG@5':>7} | {'Chunks':>7} | Golden")
    print(header)
    print("-" * width)

    for path in result_files:
        try:
            data = load_eval_results(path)
            ts = data["timestamp"][:16]

            # Get retrieval config
            ret_config = data.get("retrieval_config", {})
            mode = ret_config.get("mode", "dense")
            alpha = ret_config.get("alpha")
            alpha_str = f"{alpha:.2f}" if alpha is not None else "-"
            retrieve_k = ret_config.get("retrieve_k")
            retrieve_k_str = f"{retrieve_k}" if retrieve_k is not None else "-"
            reranker = ret_config.get("reranker_model", "")
            reranker_str = reranker.split("/")[-1] if reranker else "-"

            chunks = data.get("total_chunks", "?")
            chunking = data.get("chunking_config") or {}
            max_chars = chunking.get("max_chars", "-")
            table_max_chars = chunking.get("table_max_chars", "-")
            golden = Path(data.get("golden_path") or "?").stem.split("dataset")[-1].lstrip("_") or "v1"

            overall = data.get("overall", {})
            if overall:
                hit = overall.get("hit_rate", 0)
                mrr = overall.get("mrr", 0)
                recall = overall.get("recall", 0)
                precision = overall.get("precision", 0)
                ndcg = overall.get("ndcg", 0)
            else:
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

            print(f"{ts:<20} | {mode:<10} | {alpha_str:<6} | "
                  f"{retrieve_k_str:<6} | {reranker_str:<22} | {max_chars:>5} | {table_max_chars:>5} | "
                  f"{hit:>6.3f} | {mrr:>6.3f} | {recall:>8.3f} | {precision:>6.3f} | "
                  f"{ndcg:>7.3f} | {chunks:>7} | {golden}")

        except Exception as e:
            print(f"Error loading {path.name}: {e}")

    print("\n" + "=" * width)


# =============================================================================
# CLI
# =============================================================================


def main():
    """Run retrieval evaluation, alpha tuning, comparison, or failure analysis."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval with dense, BM25, hybrid, or reranker."
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
    parser.add_argument("--dense-weight", type=float, default=1.0, help="Dense weight for RRF")
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="BM25 weight for RRF")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dense weight for hybrid (0=BM25 only, 1=dense only)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="k values to evaluate",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=20,
        help="Candidates for reranking",
    )
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model name",
    )
    parser.add_argument("--run-directory", help="Path to chunking run")
    parser.add_argument("--golden-path", default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--device", help="Device for models")
    parser.add_argument("--compare", action="store_true", help="Compare all runs")
    parser.add_argument("--tune-alpha", action="store_true", help="Tune alpha")
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help="Analyze failed queries",
    )
    args = parser.parse_args()

    if args.compare:
        compare_runs()
        return

    if args.tune_alpha:
        tune_alpha(
            golden_path=Path(args.golden_path),
            run_directory=args.run_directory,
            device=args.device,
        )
        return

    # Run evaluation
    evaluator = HybridEvaluator(
        golden_path=args.golden_path,
        run_directory=args.run_directory,
        mode=args.mode,
        alpha=args.alpha,
        retrieve_k=args.retrieve_k,
        reranker_model=args.reranker_model,
        device=args.device,
        rrf_k=args.rrf_k,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )

    results = evaluator.evaluate(k_values=args.k)
    evaluator.print_summary(results)
    evaluator.save_results(results)

    if args.analyze_failures:
        print("\n" + "=" * 80)
        print(f"FAILURE ANALYSIS (k={results.primary_k})")
        print("=" * 80)

        failures = []
        for qr in results.query_results[results.primary_k]:
            if qr.total_relevant > 0 and not qr.hit:
                failures.append(qr)

        print(f"\nTotal failures: {len(failures)} / {results.overall['positive_queries']}")

        # Group by category
        by_cat: dict[str, int] = defaultdict(int)
        for f in failures:
            by_cat[f.category] += 1

        print("\nFailures by category:")
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

        print("\nSample failures:")
        for f in failures[:5]:
            print(f"\n  Query: {f.query[:70]}...")
            print(f"  Category: {f.category}, Difficulty: {f.difficulty}")
            print(f"  Expected {f.total_relevant} relevant, found {f.relevant_retrieved}")
            if f.top_chunks:
                top = f.top_chunks[0]
                print(f"  Top result: score={top['score']:.3f}, orders={top['element_orders'][:5]}...")


if __name__ == "__main__":
    main()
