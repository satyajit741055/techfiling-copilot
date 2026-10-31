"""
Qdrant Retrieval Evaluation Module
==================================

Evaluates ``QdrantRetriever`` / ``QdrantRetrieverWithReranker`` against a golden dataset:

1. ``qdrant_dense``: dense vectors only
2. ``qdrant_sparse``: BM25 sparse vectors only
3. ``qdrant``: server-side RRF hybrid (dense + BM25)
4. ``qdrant_rerank``: hybrid + cross-encoder reranking (the app pipeline)

Several collections (index versions) can be evaluated in one run for comparison; models
are loaded once and shared. It also supports alpha (RRF weight) tuning.

USAGE
-----

# Evaluate the configured collection (QDRANT_COLLECTION) with reranking
uv run python eval/hybrid_rerank_eval.py --golden-path goldens/<set>.jsonl --mode qdrant_rerank

# Compare two index versions side by side
uv run python eval/hybrid_rerank_eval.py --golden-path goldens/<set>.jsonl \\
    --collection techfiling_20261031T083307Z techfiling_20261031T083714Z

# Tune alpha (dense RRF weight; sparse weight = 1 - alpha)
uv run python eval/hybrid_rerank_eval.py --golden-path goldens/<set>.jsonl --tune-alpha

# Compare all saved runs
uv run python eval/hybrid_rerank_eval.py --compare
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config.settings import settings
from model_runtime import Embedder
from retriever.hybrid_retriever import Reranker, SearchResult
from retriever.qdrant_retriever import QdrantRetriever, QdrantRetrieverWithReranker
from vector_store import QdrantChunkStore
from vector_store.filters import GROUP_FIELDS, check_group_by, matches, merge_filters

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "eval" / "retriver_results"
CHUNK_RUNS_DIR = ROOT / "data" / "chunks"
MODES = ("qdrant", "qdrant_dense", "qdrant_sparse", "qdrant_rerank")
STORE_MODES = {"qdrant": "hybrid", "qdrant_dense": "dense", "qdrant_sparse": "sparse"}
DEFAULT_GROUP_LIMIT = 3  # same default as the /questions API when groups are not given


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
    group: Any = None


@dataclass
class QueryResult:
    """Evaluation result for a single query.

    For grouped queries ``reciprocal_rank``/``recall``/``ndcg`` are macro-averaged over the
    expected groups that have relevant items, and ``hit`` requires a hit in every such group
    (a comparison needs evidence for each side). ``per_group`` holds each group's metrics.
    """

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
    scope_kind: str = "doc"
    scope: dict[str, Any] = field(default_factory=dict)
    group_by: str | None = None
    groups_expected: list[Any] = field(default_factory=list)
    groups_returned: list[Any] = field(default_factory=list)
    group_recall: float | None = None
    per_group: dict[str, dict[str, float]] = field(default_factory=dict)
    filter_violations: int = 0
    latency_ms: float = 0.0


@dataclass
class QueryRun:
    """Raw retrieval output for one golden query at the largest evaluated k."""

    results: list[SearchResult] | None = None
    grouped: dict[Any, list[SearchResult]] | None = None
    latency_ms: float = 0.0

    def chunks(self) -> list[dict[str, Any]]:
        if self.grouped is None:
            return [chunk for _, chunk in self.results or []]
        return [chunk for hits in self.grouped.values() for _, chunk in hits]


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
    metrics_by_scope: dict[str, dict[int, dict[str, float]]] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)

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
            "metrics_by_scope": self.metrics_by_scope,
            "checks": self.checks,
            "query_results": {
                k: [asdict(qr) for qr in results] for k, results in self.query_results.items()
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


def _latency_stats(latencies_ms: list[float]) -> dict[str, float | None]:
    """Mean / p50 / p95 latency in milliseconds (``None`` when there are no samples)."""
    if not latencies_ms:
        return {"latency_ms_mean": None, "latency_ms_p50": None, "latency_ms_p95": None}
    return {
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
    }


# =============================================================================
# Golden Dataset
# =============================================================================


def doc_group_value(doc_id: str, group_by: str) -> Any:
    """Group value of a filing from its doc_id (``{TICKER}_{FORM}_{FISCAL_YEAR}``)."""
    if group_by == "doc_id":
        return doc_id
    if group_by == "ticker":
        return doc_id.split("_", 1)[0]
    if group_by == "fiscal_year":
        return int(doc_id.rsplit("_", 1)[1])
    raise ValueError(f"group_by must be one of {GROUP_FIELDS}, got {group_by!r}")


@dataclass
class GoldenQuery:
    """A labeled query and its expected relevant filing elements.

    Optional retrieval scope (same semantics as the API / retrievers):
    - ``doc_id``: single filing (merged into ``filters``)
    - ``filters``: scalar, list (match any) or ``{"gte": .., "lte": ..}`` range per field
    - ``group_by`` (+ ``groups``, ``group_limit``): grouped search; ``k`` is per group.
      Without ``groups`` the retriever discovers the top ``group_limit`` groups.
    """

    id: str
    query: str
    doc_id: str | None
    category: str
    difficulty: str
    expected_answer: str
    relevant: list[dict[str, Any]]
    filters: dict[str, Any] | None = None
    group_by: str | None = None
    groups: list[Any] | None = None
    group_limit: int | None = None

    @property
    def scope(self) -> dict[str, Any]:
        """Merged, validated metadata filters for this query."""
        return merge_filters(self.filters, doc_id=self.doc_id)

    @property
    def scope_kind(self) -> str:
        if self.group_by:
            return "grouped"
        if self.filters:
            return "filtered"
        return "doc" if self.doc_id else "open"

    def expected_groups(self) -> list[Any]:
        """Requested ``groups``, else the groups holding relevant items (first-seen order)."""
        if self.groups is not None:
            return list(self.groups)
        return list(
            dict.fromkeys(doc_group_value(item["doc_id"], self.group_by) for item in self.relevant)
        )

    def for_group(self, value: Any) -> GoldenQuery:
        """Copy keeping only the relevant items that belong to group ``value``."""
        return replace(
            self,
            relevant=[
                item
                for item in self.relevant
                if doc_group_value(item["doc_id"], self.group_by) == value
            ],
        )

    def validate(self) -> None:
        """Reject scopes the retrievers would reject, and unassignable grouped items."""
        scope = self.scope
        if self.group_by is None:
            if self.groups is not None or self.group_limit is not None:
                raise ValueError(f"{self.id}: groups/group_limit need group_by")
            return
        check_group_by(self.group_by, scope, self.groups)
        if self.group_limit is not None and self.group_limit < 1:
            raise ValueError(f"{self.id}: group_limit must be at least 1")
        if any(not item.get("doc_id") for item in self.relevant):
            raise ValueError(f"{self.id}: grouped queries need doc_id on every relevant item")

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

    def score_ranking(
        self, ranked: list[tuple[list[int], str | None]]
    ) -> tuple[list[int], set[int]]:
        """Grade a ranked list of (element_orders, doc_id), crediting each relevant item at most once.

        An item may list equivalent elements (the same fact in an MD&A table and a note); a second copy
        must not add gain again, otherwise recall double-counts and nDCG can exceed 1.
        Returns per-rank gains and the indices of relevant items found.
        """
        scores, found = [], set()
        for chunk_orders, doc_id in ranked:
            orders = set(chunk_orders or [])
            new = [
                i
                for i, item in enumerate(self.relevant)
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
                golden = GoldenQuery(
                    id=data["id"],
                    query=data["query"],
                    doc_id=data.get("doc_id"),
                    category=data["category"],
                    difficulty=data["difficulty"],
                    expected_answer=data["expected_answer"],
                    relevant=data.get("relevant", []),
                    filters=data.get("filters"),
                    group_by=data.get("group_by"),
                    groups=data.get("groups"),
                    group_limit=data.get("group_limit"),
                )
                golden.validate()
                queries.append(golden)
    return queries


# =============================================================================
# Unified Evaluator
# =============================================================================


def open_store(collection: str | None = None, client: Any = None) -> QdrantChunkStore:
    """Store for ``collection`` (default ``QDRANT_COLLECTION``), reusing ``client`` if given.

    The embedded on-disk Qdrant can only be opened once per process, so evaluating several
    collections must share one client.
    """
    if client is None:
        return QdrantChunkStore.from_settings(collection=collection)
    return QdrantChunkStore(
        settings.QDRANT_URL, collection or settings.QDRANT_COLLECTION, client=client
    )


def find_chunk_run(collection: str, runs_dir: Path = CHUNK_RUNS_DIR) -> tuple[Path | None, dict]:
    """Chunking run (directory, manifest) that produced ``collection``, if still on disk."""
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("qdrant_collection") or {}).get("name") == collection:
            return manifest_path.parent, manifest
    return None, {}


class HybridEvaluator:
    """
    Evaluates a Qdrant retrieval mode on one collection against a golden dataset.

    Supported modes (``MODES``):
    - "qdrant_dense": dense vectors only
    - "qdrant_sparse": BM25 sparse vectors only
    - "qdrant": RRF hybrid (dense + BM25)
    - "qdrant_rerank": RRF hybrid + cross-encoder reranking
    """

    def __init__(
        self,
        golden_path: str | Path,
        collection: str | None = None,
        mode: str = "qdrant_rerank",
        alpha: float | None = None,
        retrieve_k: int | None = None,
        reranker_model: str | None = None,
        device: str | None = None,
        rrf_k: int = 60,
        dense_k: int = 50,
        bm25_k: int = 50,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        *,
        client: Any = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        retriever: Any = None,
    ):
        """Load the golden set and initialize the selected retrieval pipeline.

        Args:
            golden_path: JSON Lines file containing labeled evaluation queries.
            collection: Qdrant collection (index version); ``None`` uses QDRANT_COLLECTION.
            mode: Retrieval strategy to evaluate (one of ``MODES``).
            alpha: Dense RRF weight (sparse = 1 - alpha) when explicit weights are 1/1.
            retrieve_k: Candidate count used before cross-encoder reranking.
            reranker_model: Cross-encoder model used by reranking mode.
            device: Model inference device, such as ``cuda`` or ``cpu``.
            rrf_k: RRF smoothing constant.
            dense_k: Dense candidates fed into RRF.
            bm25_k: BM25 candidates fed into RRF.
            dense_weight: Dense ranking weight in RRF (overrides alpha unless both are 1).
            bm25_weight: BM25 ranking weight in RRF (overrides alpha unless both are 1).
            client: Existing Qdrant client to share across collections.
            embedder: Already-loaded query embedder to share across collections.
            reranker: Already-loaded reranker to share across collections.
            retriever: Pre-built retriever (tests); skips store and model setup.

        Raises:
            ValueError: If ``mode`` is unsupported.
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode} (expected one of {MODES})")
        self.golden_path = Path(golden_path)
        self.golden_queries = load_golden_dataset(self.golden_path)
        self.mode = mode
        self.alpha = settings.RETRIEVAL_ALPHA if alpha is None else alpha
        self.retrieve_k = settings.RETRIEVAL_K if retrieve_k is None else retrieve_k
        self.reranker_model = reranker_model or settings.effective_reranker
        self.rrf_config = {
            "rrf_k": rrf_k,
            "dense_k": dense_k,
            "bm25_k": bm25_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
        }

        if retriever is None:
            store = open_store(collection, client)
            print(f"Loading retriever (collection={store.collection}, mode={mode})...")
            qdrant_kwargs: dict[str, Any] = {
                "dense_k": dense_k,
                "sparse_k": bm25_k,
                "rrf_k": rrf_k,
                "embedder": embedder,
                "device": device,
            }
            if (dense_weight, bm25_weight) == (1.0, 1.0):
                qdrant_kwargs["alpha"] = self.alpha
            else:
                qdrant_kwargs.update(dense_weight=dense_weight, sparse_weight=bm25_weight)
            if mode == "qdrant_rerank":
                retriever = QdrantRetrieverWithReranker(
                    store,
                    retrieve_k=self.retrieve_k,
                    reranker_model=self.reranker_model,
                    reranker=reranker,
                    **qdrant_kwargs,
                )
            else:
                retriever = QdrantRetriever(store, mode=STORE_MODES[mode], **qdrant_kwargs)
        self.retriever = retriever
        self.base = getattr(retriever, "retriever", retriever)  # QdrantRetriever under a reranker
        self.store = self.base.store
        self.collection = self.store.collection
        self.chunk_count = self.store.count()
        self.chunk_run, run_manifest = find_chunk_run(self.collection)
        self.run_directory = f"qdrant:{self.collection}"
        self.manifest = {**run_manifest, **self.store.index_metadata()}

        print(f"  Loaded {self.chunk_count} chunks (chunk run: {self.chunk_run or 'not found'})")
        print(f"Loaded {len(self.golden_queries)} golden queries")

    @property
    def embedder(self) -> Embedder:
        return self.base.embedder

    @property
    def reranker(self) -> Reranker | None:
        return getattr(self.retriever, "reranker", None)

    def set_alpha(self, alpha: float) -> None:
        """Change the RRF weights in place (dense = alpha, sparse = 1 - alpha)."""
        self.alpha = alpha
        self.base.dense_weight, self.base.sparse_weight = alpha, 1.0 - alpha
        self.rrf_config.update(dense_weight=alpha, bm25_weight=1.0 - alpha)

    def retrieve(self, golden: GoldenQuery, k: int) -> QueryRun:
        """Run the golden query's search (flat or grouped) once, timing it."""
        start = time.perf_counter()
        if golden.group_by:
            grouped = self.retriever.search_grouped(
                golden.query,
                golden.group_by,
                groups=golden.groups,
                k=k,
                group_limit=golden.group_limit or DEFAULT_GROUP_LIMIT,
                filters=golden.scope,
            )
            run = QueryRun(grouped=grouped)
        else:
            run = QueryRun(results=self.retriever.search(golden.query, k=k, filters=golden.scope))
        run.latency_ms = (time.perf_counter() - start) * 1000
        return run

    @staticmethod
    def _score_ranked(
        golden: GoldenQuery, results: list[SearchResult], k: int, group: Any = None
    ) -> dict[str, Any]:
        """Hit / RR / recall / nDCG and relevance matches for one ranked list."""
        relevance_scores, found = golden.score_ranking(
            [(chunk.get("element_orders", []), chunk.get("doc_id")) for _, chunk in results]
        )
        matches = []
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
                matches.append(
                    RelevanceMatch(
                        chunk_id=chunk.get("chunk_id", "unknown"),
                        element_orders=chunk_orders,
                        matched_orders=list(set(matched)),
                        relevance_grade=grade,
                        rank=rank,
                        score=score,
                        group=group,
                    )
                )
        total = len(golden.relevant)
        return {
            "hit": compute_hit(relevance_scores),
            "reciprocal_rank": compute_reciprocal_rank(relevance_scores),
            "recall": compute_recall(len(found), total),
            "ndcg": compute_ndcg(relevance_scores, golden.get_all_relevance_grades(), k),
            "found": len(found),
            "total_relevant": total,
            "retrieved": len(results),
            "matches": matches,
        }

    @staticmethod
    def _top_chunks(results: list[SearchResult], group: Any = None) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": chunk.get("chunk_id"),
                "score": float(score),
                "element_orders": chunk.get("element_orders", []),
                "breadcrumb": chunk.get("breadcrumb", "")[:100],
                **({"group": group} if group is not None else {}),
            }
            for score, chunk in results
        ]

    def evaluate_query(
        self, golden: GoldenQuery, k: int, run: QueryRun | None = None
    ) -> QueryResult:
        """Score one query at ``k`` (``run`` from ``retrieve`` at k >= this k; searched if omitted)."""
        run = run or self.retrieve(golden, k)
        scope = golden.scope
        common = {
            "query_id": golden.id,
            "query": golden.query,
            "category": golden.category,
            "difficulty": golden.difficulty,
            "doc_id": golden.doc_id,
            "total_relevant": len(golden.relevant),
            "scope_kind": golden.scope_kind,
            "scope": scope,
            "latency_ms": run.latency_ms,
        }
        if run.grouped is None:
            results = run.results[:k]
            scored = self._score_ranked(golden, results, k)
            return QueryResult(
                **common,
                retrieved_count=len(results),
                relevant_retrieved=scored["found"],
                hit=scored["hit"],
                reciprocal_rank=scored["reciprocal_rank"],
                recall=scored["recall"],
                precision=compute_precision(len(scored["matches"]), len(results)),
                ndcg=scored["ndcg"],
                matches=scored["matches"],
                top_chunks=self._top_chunks(results[:5]),
                filter_violations=sum(not matches(chunk, scope) for _, chunk in results),
            )

        grouped = {group: hits[:k] for group, hits in run.grouped.items()}
        expected = golden.expected_groups()
        per_group = {
            group: self._score_ranked(golden.for_group(group), grouped.get(group, []), k, group)
            for group in expected
        }
        positive = [scored for scored in per_group.values() if scored["total_relevant"]]

        def macro(metric: str) -> float:
            return float(np.mean([s[metric] for s in positive])) if positive else 0.0

        all_matches = [m for scored in per_group.values() for m in scored["matches"]]
        retrieved = sum(len(hits) for hits in grouped.values())
        returned = [group for group, hits in grouped.items() if hits]
        violations = sum(
            not matches(chunk, scope) or chunk.get(golden.group_by) != group
            for group, hits in grouped.items()
            for _, chunk in hits
        )
        return QueryResult(
            **common,
            retrieved_count=retrieved,
            relevant_retrieved=sum(s["found"] for s in per_group.values()),
            hit=bool(positive) and all(s["hit"] for s in positive),
            reciprocal_rank=macro("reciprocal_rank"),
            recall=macro("recall"),
            precision=compute_precision(len(all_matches), retrieved),
            ndcg=macro("ndcg"),
            matches=all_matches,
            top_chunks=[
                top for group, hits in grouped.items() for top in self._top_chunks(hits[:2], group)
            ],
            group_by=golden.group_by,
            groups_expected=expected,
            groups_returned=returned,
            group_recall=sum(group in returned for group in expected) / len(expected)
            if expected
            else None,
            per_group={
                str(group): {
                    "hit": float(s["hit"]),
                    "reciprocal_rank": s["reciprocal_rank"],
                    "recall": s["recall"],
                    "ndcg": s["ndcg"],
                    "retrieved": s["retrieved"],
                    "total_relevant": s["total_relevant"],
                }
                for group, s in per_group.items()
            },
            filter_violations=violations,
        )

    def check_get_chunks(self, run: QueryRun) -> dict[str, Any]:
        """Re-fetch the run's chunks by id; they must come back identical and in order."""
        chunks = list({chunk["chunk_id"]: chunk for chunk in run.chunks()}.values())
        ids = [chunk["chunk_id"] for chunk in chunks]
        start = time.perf_counter()
        fetched = self.retriever.get_chunks(ids) if ids else []
        latency_ms = (time.perf_counter() - start) * 1000
        ok = [c.get("chunk_id") for c in fetched] == ids and all(
            f.get("text") == c.get("text") for f, c in zip(fetched, chunks)
        )
        return {"chunks": len(ids), "ok": ok, "latency_ms": latency_ms}

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

        reranking = self.mode == "qdrant_rerank"
        fused = self.mode in ("qdrant", "qdrant_rerank")
        retrieval_config = {
            "mode": self.mode,
            "vector_store": "qdrant",
            "collection": self.collection,
            "chunk_run": str(self.chunk_run) if self.chunk_run else None,
            "embedding_backend": getattr(self.embedder, "backend", None),
            "reranker_backend": getattr(getattr(self.reranker, "backend", None), "backend", None)
            if reranking
            else None,
            "alpha": self.alpha if fused else None,
            "retrieve_k": self.retrieve_k if reranking else None,
            "reranker_model": self.reranker_model if reranking else None,
            "rrf": {
                **self.rrf_config,
                "dense_weight": self.base.dense_weight,
                "bm25_weight": self.base.sparse_weight,
            }
            if fused
            else None,
            "bm25": self.manifest.get("bm25") or None,
        }

        query_results: dict[int, list[QueryResult]] = {k: [] for k in k_values}
        max_k = max(k_values)
        get_chunks_runs: list[dict[str, Any]] = []

        if verbose:
            print(f"\nEvaluating {len(self.golden_queries)} queries at k={k_values}...")

        if self.golden_queries:  # warm-up so model/collection loading isn't timed
            self.retriever.search(self.golden_queries[0].query, k=1)

        # Candidate limits (dense_k / sparse_k / retrieve_k) don't depend on k, so one search
        # at the largest k, sliced, gives the same ranking as one search per k.
        for i, golden in enumerate(self.golden_queries):
            run = self.retrieve(golden, max_k)
            for k in k_values:
                query_results[k].append(self.evaluate_query(golden, k, run))
            get_chunks_runs.append(self.check_get_chunks(run))

            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(self.golden_queries)} queries")

        if verbose:
            print(f"  Processed {len(self.golden_queries)}/{len(self.golden_queries)} queries")

        get_chunks_latency = [r["latency_ms"] for r in get_chunks_runs if r["chunks"]]
        checks = {
            "get_chunks": {
                "queries": len(get_chunks_runs),
                "chunks": sum(r["chunks"] for r in get_chunks_runs),
                "failed_queries": sum(not r["ok"] for r in get_chunks_runs),
                **_latency_stats(get_chunks_latency),
            },
        }

        # Aggregate metrics
        metrics_by_k = {}
        for k in k_values:
            metrics_by_k[k] = self._aggregate_metrics(query_results[k])

        # Breakdowns
        metrics_by_category = self._compute_breakdowns(query_results, k_values, "category")
        metrics_by_difficulty = self._compute_breakdowns(query_results, k_values, "difficulty")
        metrics_by_doc = self._compute_breakdowns(query_results, k_values, "doc_id")
        metrics_by_scope = self._compute_breakdowns(query_results, k_values, "scope_kind")

        # Overall metrics at k=5
        primary_k = 5 if 5 in k_values else k_values[-1]
        primary = metrics_by_k[primary_k]
        overall = {
            "k": primary_k,
            **{
                metric: primary.get(metric, 0.0)
                for metric in ("hit_rate", "mrr", "recall", "precision", "ndcg")
            },
            "positive_queries": primary["positive_queries"],
            "negative_queries": primary["negative_queries"],
            "group_recall": primary.get("group_recall"),
            "filter_violations": primary["filter_violations"],
            "latency_ms_p50": primary["latency_ms_p50"],
            "latency_ms_p95": primary["latency_ms_p95"],
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
            total_chunks=self.chunk_count,
            total_queries=len(self.golden_queries),
            overall=overall,
            metrics_by_k={str(k): v for k, v in metrics_by_k.items()},
            metrics_by_category=metrics_by_category,
            metrics_by_difficulty=metrics_by_difficulty,
            metrics_by_doc=metrics_by_doc,
            query_results=query_results,
            metrics_by_scope=metrics_by_scope,
            checks=checks,
        )

    def _aggregate_metrics(self, results: list[QueryResult]) -> dict[str, float]:
        """Aggregate metrics across queries.

        Relevance metrics use positive queries only; latency, filter violations and group
        recall cover every query they apply to.
        """
        positive = [r for r in results if r.total_relevant > 0]
        negative = [r for r in results if r.total_relevant == 0]
        group_recalls = [r.group_recall for r in results if r.group_recall is not None]
        retrieved = sum(r.retrieved_count for r in results)
        violations = sum(r.filter_violations for r in results)

        metrics: dict[str, Any] = {
            "total_queries": len(results),
            "positive_queries": len(positive),
            "negative_queries": len(negative),
            "grouped_queries": sum(r.group_by is not None for r in results),
            "filter_violations": violations,
            "filter_violation_rate": violations / retrieved if retrieved else 0.0,
            **_latency_stats([r.latency_ms for r in results]),
        }
        if group_recalls:
            metrics["group_recall"] = float(np.mean(group_recalls))
        if not positive:
            return metrics

        metrics |= {
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
                # Queries without a doc_id: "cross_document" when unscoped, else their scope kind
                key = getattr(r, group_by) or (
                    "cross_document" if r.scope_kind == "open" else r.scope_kind
                )
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
        print(f"  Collection: {results.retrieval_config.get('collection')}")
        print(f"  Chunk run: {results.retrieval_config.get('chunk_run') or 'not found'}")
        print(f"  Total chunks: {results.total_chunks}")
        print(
            f"  Total queries: {results.total_queries} "
            f"({results.overall['positive_queries']} positive, "
            f"{results.overall['negative_queries']} negative)"
        )

        config = results.chunking_config
        print(
            f"  Chunking: MAX_CHARS={config.get('max_chars')}, "
            f"TABLE_MAX_CHARS={config.get('table_max_chars')}"
        )

        model = results.embedding_model
        print(f"  Model: {model.get('name')}")

        ret_config = results.retrieval_config
        print(f"  Retrieval mode: {ret_config.get('mode')}")
        if ret_config.get("alpha") is not None:
            print(f"  Alpha: {ret_config.get('alpha')}")
        if ret_config.get("retrieve_k") is not None:
            print(f"  Retrieve-k: {ret_config.get('retrieve_k')}")
        if ret_config.get("reranker_model"):
            print(f"  Reranker: {ret_config.get('reranker_model')}")

        # Overall metrics
        print("\n" + "-" * 80)
        print(f"OVERALL METRICS (k={results.primary_k})")
        print("-" * 80)
        overall = results.overall
        print(
            f"  Hit Rate:   {overall['hit_rate']:.3f}  "
            f"({overall['hit_rate'] * 100:.1f}% of queries found relevant results)"
        )
        mrr = overall["mrr"]
        avg_rank = 1 / mrr if mrr > 0 else float("inf")
        print(
            f"  MRR:        {mrr:.3f}  (first relevant result at ~rank {avg_rank:.1f} on average)"
        )
        print(
            f"  Recall:     {overall['recall']:.3f}  "
            f"({overall['recall'] * 100:.1f}% of all relevant chunks retrieved)"
        )
        print(
            f"  Precision:  {overall['precision']:.3f}  "
            f"({overall['precision'] * 100:.1f}% of retrieved chunks are relevant)"
        )
        print(f"  nDCG:       {overall['ndcg']:.3f}  (ranking quality score)")
        if overall.get("group_recall") is not None:
            print(
                f"  Group recall: {overall['group_recall']:.3f}  "
                "(expected groups that returned results; grouped queries only)"
            )
        if overall.get("latency_ms_p50") is not None:
            print(
                f"  Latency:    p50={overall['latency_ms_p50']:.0f} ms  "
                f"p95={overall['latency_ms_p95']:.0f} ms  (per query, end to end)"
            )
        violations = overall.get("filter_violations", 0)
        print(
            f"  Filter violations: {violations}"
            + ("  <-- results outside the query's filters/groups" if violations else "")
        )
        get_chunks = results.checks.get("get_chunks")
        if get_chunks:
            p50 = get_chunks.get("latency_ms_p50")
            print(
                f"  get_chunks: {get_chunks['chunks']} chunks re-fetched, "
                f"{get_chunks['failed_queries']} queries mismatched"
                + (f", p50={p50:.1f} ms" if p50 is not None else "")
            )

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
                print(
                    f"  {cat:<25} | Hit={m.get('hit_rate', 0):.2f} "
                    f"MRR={m.get('mrr', 0):.2f} "
                    f"Recall={m.get('recall', 0):.2f} "
                    f"(n={m.get('positive_queries', 0)})"
                )

        # Difficulty breakdown
        print(f"\nBreakdown by Difficulty (k={results.primary_k}):")
        print("-" * 60)
        for diff in ["easy", "medium", "hard"]:
            if diff in results.metrics_by_difficulty:
                by_k = results.metrics_by_difficulty[diff]
                m = by_k.get(results.primary_k) or by_k.get(str(results.primary_k), {})
                if m:
                    print(
                        f"  {diff:<25} | Hit={m.get('hit_rate', 0):.2f} "
                        f"MRR={m.get('mrr', 0):.2f} "
                        f"Recall={m.get('recall', 0):.2f} "
                        f"(n={m.get('positive_queries', 0)})"
                    )

        # Scope breakdown (doc / filtered / grouped / open)
        print(f"\nBreakdown by Scope (k={results.primary_k}):")
        print("-" * 60)
        for scope_kind, by_k in sorted(results.metrics_by_scope.items()):
            m = by_k.get(results.primary_k) or by_k.get(str(results.primary_k), {})
            if m:
                group_recall = m.get("group_recall")
                p50 = m.get("latency_ms_p50")
                print(
                    f"  {scope_kind:<25} | Hit={m.get('hit_rate', 0):.2f} "
                    f"MRR={m.get('mrr', 0):.2f} "
                    f"Recall={m.get('recall', 0):.2f} "
                    + (f"GroupRecall={group_recall:.2f} " if group_recall is not None else "")
                    + (f"p50={p50:.0f}ms " if p50 is not None else "")
                    + f"(n={m.get('total_queries', 0)})"
                )

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
        rrf_str = f"_w{rrf['dense_weight']:g}-{rrf['bm25_weight']:g}_k{rrf['rrf_k']}" if rrf else ""
        collection = results.retrieval_config.get("collection")
        collection_str = f"_{collection}" if collection else ""

        filename = (
            f"{results.timestamp}_{mode}{collection_str}{alpha_str}{rrf_str}{reranker_str}"
            "_eval_results.json"
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
    evaluator: HybridEvaluator,
    alpha_values: list[float] | None = None,
    k: int = 5,
) -> dict[float, dict[str, float]]:
    """
    Find the optimal RRF alpha (dense weight; sparse = 1 - alpha) on one loaded evaluator.

    Only the fusion weights change between alphas, so models and the collection load once.
    Requires a fused mode (``qdrant`` or ``qdrant_rerank``).

    Args:
        evaluator: Evaluator in a fused mode
        alpha_values: List of alpha values to test (default: 0.0 to 1.0 by 0.1)
        k: k value for evaluation

    Returns:
        Dictionary mapping alpha -> metrics
    """
    if evaluator.mode not in ("qdrant", "qdrant_rerank"):
        raise ValueError(f"Alpha tuning needs a fused mode, got {evaluator.mode}")
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]

    print("\n" + "=" * 80)
    print(f"ALPHA TUNING ({evaluator.collection}, mode={evaluator.mode})")
    print("=" * 80)
    print(f"\nTesting alpha values: {alpha_values}")
    print(f"Evaluation k: {k}")

    results: dict[float, dict[str, float]] = {}

    for alpha in alpha_values:
        print(f"\n--- Alpha = {alpha:.2f} ---")
        evaluator.set_alpha(alpha)
        metrics = evaluator.evaluate(k_values=[k], verbose=False).metrics_by_k[str(k)]
        results[alpha] = metrics

        print(
            f"  Hit@{k}={metrics['hit_rate']:.3f}  "
            f"MRR={metrics['mrr']:.3f}  "
            f"Recall={metrics['recall']:.3f}  "
            f"Precision={metrics['precision']:.3f}"
        )

    # Find best alpha
    print("\n" + "=" * 80)
    print("ALPHA TUNING RESULTS")
    print("=" * 80)

    print(f"\n{'Alpha':<8} | {'Hit@' + str(k):<8} | {'MRR':<8} | {'Recall':<8} | {'Precision':<8}")
    print("-" * 60)
    for alpha in alpha_values:
        m = results[alpha]
        print(
            f"{alpha:<8.2f} | {m['hit_rate']:<8.3f} | {m['mrr']:<8.3f} | "
            f"{m['recall']:<8.3f} | {m['precision']:<8.3f}"
        )

    # Best by each metric
    print("\nBest alpha by metric:")
    for metric in ["hit_rate", "mrr", "recall", "precision"]:
        best_alpha = max(alpha_values, key=lambda a: results[a][metric])
        print(f"  {metric}: alpha={best_alpha:.2f} ({results[best_alpha][metric]:.3f})")

    # Overall best (average of normalized metrics)
    def score(alpha):
        """Return the unweighted mean of metrics used to select an alpha."""
        m = results[alpha]
        return (m["hit_rate"] + m["mrr"] + m["recall"] + m["precision"]) / 4

    best_overall = max(alpha_values, key=score)
    print(f"\nRecommended alpha (balanced): {best_overall:.2f}")

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

    width = 185
    print("\n" + "=" * width)
    print("EVALUATION RUN COMPARISON")
    print("=" * width)

    header = (
        f"\n{'Timestamp':<20} | {'Mode':<13} | {'Collection':<27} | {'Alpha':<6} | "
        f"{'Ret-k':<6} | {'Reranker':<22} | {'MaxCh':>5} | {'TblCh':>5} | {'PHdg':<4} | "
        f"{'Hit@5':>6} | {'MRR@5':>6} | {'Recall@5':>8} | {'Prec@5':>6} | "
        f"{'nDCG@5':>7} | {'Chunks':>7} | Golden"
    )
    print(header)
    print("-" * width)

    for path in result_files:
        try:
            data = load_eval_results(path)
            ts = data["timestamp"][:16]

            # Get retrieval config
            ret_config = data.get("retrieval_config", {})
            mode = ret_config.get("mode", "dense")
            run_directory = str(data.get("run_directory") or "")
            collection = ret_config.get("collection") or (
                run_directory.removeprefix("qdrant:")
                if run_directory.startswith("qdrant:")
                else "-"
            )
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
            pseudo = "yes" if chunking.get("pseudo_headings") else "-"
            golden = (
                Path(data.get("golden_path") or "?").stem.split("dataset")[-1].lstrip("_") or "v1"
            )

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

            print(
                f"{ts:<20} | {mode:<13} | {collection:<27} | {alpha_str:<6} | "
                f"{retrieve_k_str:<6} | {reranker_str:<22} | {max_chars:>5} | {table_max_chars:>5} | {pseudo:<4} | "
                f"{hit:>6.3f} | {mrr:>6.3f} | {recall:>8.3f} | {precision:>6.3f} | "
                f"{ndcg:>7.3f} | {chunks:>7} | {golden}"
            )

        except Exception as e:
            print(f"Error loading {path.name}: {e}")

    print("\n" + "=" * width)


# =============================================================================
# CLI
# =============================================================================


def print_collection_comparison(all_results: list[EvalResults]) -> None:
    """Side-by-side headline metrics for the collections evaluated in this run."""
    k = all_results[0].primary_k
    print("\n" + "=" * 100)
    print(f"COLLECTION COMPARISON (k={k}, mode={all_results[0].retrieval_config['mode']})")
    print("=" * 100)
    print(
        f"{'Collection':<28} | {'MaxCh':>5} | {'TblCh':>5} | {'Chunks':>6} | "
        f"{'Hit':>6} | {'MRR':>6} | {'Recall':>6} | {'Prec':>6} | {'nDCG':>6}"
    )
    print("-" * 100)
    for results in all_results:
        overall, chunking = results.overall, results.chunking_config
        print(
            f"{results.retrieval_config['collection']:<28} | "
            f"{chunking.get('max_chars', '-'):>5} | {chunking.get('table_max_chars', '-'):>5} | "
            f"{results.total_chunks:>6} | {overall.get('hit_rate', 0):>6.3f} | "
            f"{overall.get('mrr', 0):>6.3f} | {overall.get('recall', 0):>6.3f} | "
            f"{overall.get('precision', 0):>6.3f} | {overall.get('ndcg', 0):>6.3f}"
        )
    print("=" * 100)


def main():
    """Run retrieval evaluation, alpha tuning, comparison, or failure analysis."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qdrant retrieval (dense, BM25, RRF hybrid, or hybrid + reranker)."
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="qdrant_rerank",
        help="Retrieval mode (default: qdrant_rerank, the app pipeline)",
    )
    parser.add_argument(
        "--collection",
        nargs="+",
        help="Qdrant collection(s) to evaluate and compare (default: QDRANT_COLLECTION)",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant")
    parser.add_argument("--dense-k", type=int, default=50, help="Dense candidates for RRF")
    parser.add_argument("--bm25-k", type=int, default=50, help="BM25 candidates for RRF")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="Dense weight for RRF")
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="BM25 weight for RRF")
    parser.add_argument(
        "--alpha",
        type=float,
        help="Dense weight for hybrid (default: RETRIEVAL_ALPHA from .env)",
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
        help="Candidates for reranking (default: RETRIEVAL_K from .env)",
    )
    parser.add_argument(
        "--reranker-model",
        help="Reranker model (default: environment-aware setting from .env)",
    )
    parser.add_argument(
        "--golden-path", type=Path, help="Golden dataset (JSONL); required to evaluate"
    )
    parser.add_argument("--device", help="Device for models")
    parser.add_argument("--compare", action="store_true", help="Compare all saved runs")
    parser.add_argument(
        "--tune-alpha", action="store_true", help="Tune alpha (fused modes; first collection)"
    )
    parser.add_argument(
        "--analyze-failures",
        action="store_true",
        help="Analyze failed queries",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    args = parser.parse_args()

    if args.compare:
        compare_runs()
        return
    if args.golden_path is None:
        parser.error("--golden-path is required unless --compare is given")

    collections = args.collection or [None]
    if args.tune_alpha and args.mode not in ("qdrant", "qdrant_rerank"):
        parser.error("--tune-alpha needs --mode qdrant or qdrant_rerank")

    all_results: list[EvalResults] = []
    shared: dict[str, Any] = {}
    for collection in collections:
        evaluator = HybridEvaluator(
            golden_path=args.golden_path,
            collection=collection,
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
            **shared,
        )
        shared = {
            "client": evaluator.store.client,
            "embedder": evaluator.embedder,
            "reranker": evaluator.reranker,
        }
        if args.tune_alpha:
            tune_alpha(evaluator, k=5 if 5 in args.k else max(args.k))
            return

        results = evaluator.evaluate(k_values=args.k)
        evaluator.print_summary(results)
        if not args.no_save:
            evaluator.save_results(results)
        if args.analyze_failures:
            analyze_failures(results)
        all_results.append(results)

    if len(all_results) > 1:
        print_collection_comparison(all_results)


def analyze_failures(results: EvalResults) -> None:
    """Print queries with no relevant result at the primary k."""
    print("\n" + "=" * 80)
    print(f"FAILURE ANALYSIS (k={results.primary_k})")
    print("=" * 80)

    failures = [
        qr
        for qr in results.query_results[results.primary_k]
        if qr.total_relevant > 0 and not qr.hit
    ]
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
