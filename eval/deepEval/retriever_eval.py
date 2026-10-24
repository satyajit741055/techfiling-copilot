"""DeepEval LLM-judge evaluation of the retriever using contextual precision and recall.

This uses the same LLM judge (gpt-4o) as the generator and full-pipeline evaluations,
providing consistent semantic evaluation across all pipeline stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from config.settings import settings

DEFAULT_GOLDEN_PATH = ROOT / "goldens" / "golden_retriever_dataset_v3.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "deepEval" / "retriever_results"
METRIC_NAMES = ("contextual_precision", "contextual_recall")


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load golden queries with relevant chunks."""
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = {"id", "query", "doc_id"} - case.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if not all(isinstance(case[field], str) and case[field].strip() for field in ("id", "query")):
                raise ValueError(f"{path}:{line_number}: id and query must be non-empty strings")
            if case["doc_id"] is not None and not (
                isinstance(case["doc_id"], str) and case["doc_id"].strip()
            ):
                raise ValueError(f"{path}:{line_number}: doc_id must be null or a non-empty string")
            cases.append(case)
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Filter cases by query IDs, categories, difficulty, or limit."""
    if args.query_ids:
        selected = set(args.query_ids)
        cases = [case for case in cases if case["id"] in selected]
        missing = selected - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown query IDs: {sorted(missing)}")
    if args.categories:
        categories = set(args.categories)
        cases = [case for case in cases if case.get("category") in categories]
    if args.difficulties:
        difficulties = set(args.difficulties)
        cases = [case for case in cases if case.get("difficulty") in difficulties]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No golden queries matched the requested filters")
    return cases


def build_metrics(args: argparse.Namespace) -> dict[str, Any]:
    """Build DeepEval metrics with common configuration."""
    selected = set(args.metrics)
    common = {
        "threshold": args.threshold,
        "model": args.judge_model,
        "include_reason": True,
        "async_mode": False,
        "strict_mode": args.strict,
    }
    metrics: dict[str, Any] = {}
    if "contextual_precision" in selected:
        metrics["contextual_precision"] = ContextualPrecisionMetric(**common)
    if "contextual_recall" in selected:
        metrics["contextual_recall"] = ContextualRecallMetric(**common)
    return metrics


def metric_result(metric: Any, test_case: LLMTestCase) -> dict[str, Any]:
    """Extract metric result."""
    metric.measure(test_case)
    score = float(metric.score) if metric.score is not None else None
    return {
        "score": score,
        "passed": bool(metric.is_successful()),
        "threshold": float(metric.threshold),
        "reason": metric.reason,
        "error": metric.error,
        "evaluation_model": metric.evaluation_model,
    }


def score_metrics(
    metrics: dict[str, Any], test_case: LLMTestCase, fail_fast: bool
) -> dict[str, dict[str, Any]]:
    """Score all metrics for a test case."""
    results = {}
    for name, metric in metrics.items():
        try:
            results[name] = metric_result(metric, test_case)
        except Exception as error:
            if fail_fast:
                raise
            results[name] = {
                "score": None,
                "passed": False,
                "threshold": float(metric.threshold),
                "reason": None,
                "error": f"{type(error).__name__}: {error}",
                "evaluation_model": getattr(metric, "evaluation_model", None),
            }
    return results


def aggregate(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    """Aggregate metrics across all cases."""
    successful = [r for r in records if r["status"] == "success"]
    metrics = {}
    for name in metric_names:
        results = [
            r["metrics"][name]
            for r in successful
            if name in r["metrics"] and r["metrics"][name].get("score") is not None
        ]
        metrics[name] = {
            "mean_score": mean(result["score"] for result in results) if results else None,
            "pass_rate": mean(result["passed"] for result in results) if results else None,
            "scored_cases": len(results),
            "metric_error_cases": sum(
                name in r.get("metrics", {}) and r["metrics"][name].get("score") is None
                for r in successful
            ),
        }
    scores = [m["mean_score"] for m in metrics.values() if m["mean_score"] is not None]
    return {
        "cases": len(records),
        "successful_cases": len(successful),
        "error_cases": len(records) - len(successful),
        "mean_score": mean(scores) if scores else None,
        "metrics": metrics,
    }


def breakdown(
    records: list[dict[str, Any]], field: str, metric_names: list[str]
) -> dict[str, dict[str, Any]]:
    """Break down metrics by category, difficulty, or document."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["case"].get(field))].append(record)
    return {name: aggregate(items, metric_names) for name, items in sorted(groups.items())}


def run_evaluation(args: argparse.Namespace) -> Path:
    """Run retriever evaluation with DeepEval metrics."""
    from retriever.hybrid_retriever import HybridRetrieverWithReranker
    from generation.config import GenerationConfig

    cases = select_cases(load_cases(args.golden_path), args)
    config = GenerationConfig()  # Use default retrieval config
    retriever = HybridRetrieverWithReranker(
        run_directory=config.resolve_run_directory(ROOT),
        alpha=config.alpha,
        retrieve_k=config.retrieve_k,
        reranker_model=config.reranker_model,
    )
    metrics = build_metrics(args)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []

    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']} ({case.get('category', 'uncategorized')})")
            try:
                # Retrieve chunks
                results = retriever.search(
                    case["query"],
                    k=args.context_k,
                    doc_id=case["doc_id"],
                )
                retrieval_context = [chunk.get("text", "") for _, chunk in results]

                # Build expected output from relevant chunks
                # For DeepEval, we need the actual relevant text
                relevant_texts = []
                if "relevant" in case:
                    for rel in case["relevant"]:
                        if rel.get("match_by") == "element_orders":
                            # For element_orders, we need to find the actual chunk text
                            # This is a simplification - in practice, you'd map element_orders to chunks
                            relevant_texts.append(f"Element orders: {rel.get('value')}")
                        else:
                            relevant_texts.append(str(rel.get("value", "")))
                expected_output = "\n".join(relevant_texts) if relevant_texts else case.get("expected_answer", "")

                # Create test case
                test_case = LLMTestCase(
                    name=case["id"],
                    input=case["query"],
                    retrieval_context=retrieval_context,
                    expected_output=expected_output,
                    metadata={
                        "doc_id": case["doc_id"],
                        "category": case.get("category"),
                        "difficulty": case.get("difficulty"),
                    },
                )

                # Score metrics
                scores = score_metrics(metrics, test_case, args.fail_fast)
                record = {
                    "status": "success",
                    "case": case,
                    "retrieval_context": retrieval_context,
                    "retrieved_count": len(retrieval_context),
                    "metrics": scores,
                }
            except Exception as error:
                record = {
                    "status": "error",
                    "case": case,
                    "retrieval_context": [],
                    "retrieved_count": 0,
                    "metrics": {},
                    "error": f"{type(error).__name__}: {error}",
                }
                if args.fail_fast:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raise
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    metric_names = list(metrics)
    summary = {
        "run_id": run_dir.name,
        "overall": aggregate(records, metric_names),
        "by_category": breakdown(records, "category", metric_names),
        "by_difficulty": breakdown(records, "difficulty", metric_names),
        "by_doc": breakdown(records, "doc_id", metric_names),
    }
    manifest = {
        "run_id": run_dir.name,
        "timestamp": timestamp,
        "golden_path": str(args.golden_path.resolve()),
        "selected_query_count": len(cases),
        "filters": {
            "query_ids": args.query_ids,
            "categories": args.categories,
            "difficulties": args.difficulties,
            "limit": args.limit,
        },
        "retrieval_config": {
            "run_directory": config.run_directory,
            "alpha": config.alpha,
            "retrieve_k": config.retrieve_k,
            "reranker_model": config.reranker_model,
            "context_k": args.context_k,
        },
        "execution_modes": {
            "embedding": settings.effective_embedding_mode,
            "reranker": settings.effective_reranker_mode,
        },
        "judge_model": args.judge_model,
        "metrics": metric_names,
        "threshold": args.threshold,
        "strict": args.strict,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("RETRIEVER EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nRun directory: {run_dir}")
    print(f"Cases: {summary['overall']['cases']}")
    print(f"Successful: {summary['overall']['successful_cases']}")
    print(f"Errors: {summary['overall']['error_cases']}")
    print(f"\nExecution modes:")
    print(f"  Embedding: {manifest['execution_modes']['embedding']}")
    print(f"  Reranker: {manifest['execution_modes']['reranker']}")
    print(f"\nRetrieval config:")
    print(f"  Run directory: {manifest['retrieval_config']['run_directory']}")
    print(f"  Alpha: {manifest['retrieval_config']['alpha']}")
    print(f"  Retrieve-k: {manifest['retrieval_config']['retrieve_k']}")
    print(f"  Reranker: {manifest['retrieval_config']['reranker_model']}")
    print(f"  Context-k: {manifest['retrieval_config']['context_k']}")
    print(f"\nJudge: {manifest['judge_model']}")
    print(f"Threshold: {manifest['threshold']}")

    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)
    for name, data in summary["overall"]["metrics"].items():
        print(f"  {name}:")
        print(f"    Mean score: {data['mean_score']:.3f}" if data['mean_score'] else "    Mean score: N/A")
        print(f"    Pass rate: {data['pass_rate']*100:.1f}%" if data['pass_rate'] else "    Pass rate: N/A")
        print(f"    Scored cases: {data['scored_cases']}")
        print(f"    Error cases: {data['metric_error_cases']}")

    print("\n" + "-" * 80)
    print("BY CATEGORY")
    print("-" * 80)
    for cat, data in summary["by_category"].items():
        print(f"  {cat}:")
        for name, metric_data in data["metrics"].items():
            score = metric_data.get("mean_score")
            print(f"    {name}: {score:.3f}" if score else f"    {name}: N/A")

    print(f"\nResults saved to: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="DeepEval LLM-judge evaluation of the retriever"
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for results"
    )
    parser.add_argument(
        "--context-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_NAMES,
        default=list(METRIC_NAMES),
        help="Metrics to evaluate"
    )
    parser.add_argument(
        "--query-ids",
        nargs="+",
        help="Specific query IDs to evaluate"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Filter by category"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        help="Filter by difficulty"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of cases"
    )
    parser.add_argument(
        "--judge-model",
        help="DeepEval judge model (default: DEEPEVAL_JUDGE_MODEL from .env)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Passing threshold"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode (binary scoring)"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running evaluation"
    )
    args = parser.parse_args()

    if args.dry_run:
        from config.settings import settings
        from generation.config import GenerationConfig

        config = GenerationConfig()
        print(json.dumps({
            "golden_path": str(args.golden_path.resolve()),
            "selected_queries": len(load_cases(args.golden_path)),
            "context_k": args.context_k,
            "metrics": args.metrics,
            "embedding_mode": settings.effective_embedding_mode,
            "reranker_mode": settings.effective_reranker_mode,
            "retrieval_config": {
                "run_directory": config.run_directory,
                "alpha": config.alpha,
                "retrieve_k": config.retrieve_k,
                "reranker_model": config.reranker_model,
            },
            "judge_model": args.judge_model or settings.DEEPEVAL_JUDGE_MODEL,
            "threshold": args.threshold,
        }, indent=2))
        return

    run_evaluation(args)


if __name__ == "__main__":
    main()
