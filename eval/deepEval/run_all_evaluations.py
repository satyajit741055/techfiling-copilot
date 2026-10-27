"""Run all DeepEval evaluations in one go.

This script runs:
1. Retriever evaluation (contextual precision, contextual recall)
2. Generator evaluation (faithfulness, answer relevancy, answer correctness)
3. Full-pipeline evaluation (contextual relevancy, faithfulness, answer relevancy)

And outputs a consolidated summary.

Usage:
    uv run python eval/deepEval/run_all_evaluations.py --judge-model gpt-4o-mini
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    return result


def load_summary(results_dir: Path, run_id: str) -> dict[str, Any]:
    """Load summary.json from a run directory."""
    summary_path = results_dir / run_id / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_manifest(results_dir: Path, run_id: str) -> dict[str, Any]:
    """Load manifest.json from a run directory."""
    manifest_path = results_dir / run_id / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def get_latest_run_id(results_dir: Path) -> str:
    """Get the most recent run ID from a results directory."""
    run_dirs = sorted(results_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        if run_dir.is_dir() and (run_dir / "summary.json").exists():
            return run_dir.name
    raise ValueError(f"No valid runs found in {results_dir}")


def compare_all():
    """Compare the latest runs across all evaluation types."""
    print("\n" + "=" * 80)
    print("COMPARING LATEST RUNS")
    print("=" * 80)

    results_dir = ROOT / "eval" / "deepEval"
    comparison = {}

    # Retriever
    retriever_dir = results_dir / "retriever_results"
    if retriever_dir.exists():
        try:
            retriever_run_id = get_latest_run_id(retriever_dir)
            retriever_summary = load_summary(retriever_dir, retriever_run_id)
            retriever_manifest = load_manifest(retriever_dir, retriever_run_id)
            comparison["retriever"] = {
                "run_id": retriever_run_id,
                "summary": retriever_summary,
                "manifest": retriever_manifest,
            }
            print(f"Retriever: {retriever_run_id}")
        except ValueError:
            print("Retriever: No runs found")

    # Generator
    generator_dir = results_dir / "generation_results"
    if generator_dir.exists():
        try:
            generator_run_id = get_latest_run_id(generator_dir)
            generator_summary = load_summary(generator_dir, generator_run_id)
            generator_manifest = load_manifest(generator_dir, generator_run_id)
            comparison["generator"] = {
                "run_id": generator_run_id,
                "summary": generator_summary,
                "manifest": generator_manifest,
            }
            print(f"Generator: {generator_run_id}")
        except ValueError:
            print("Generator: No runs found")

    # Full-pipeline
    full_pipeline_dir = results_dir / "full_pipeline_results"
    if full_pipeline_dir.exists():
        try:
            full_pipeline_run_id = get_latest_run_id(full_pipeline_dir)
            full_pipeline_summary = load_summary(full_pipeline_dir, full_pipeline_run_id)
            full_pipeline_manifest = load_manifest(full_pipeline_dir, full_pipeline_run_id)
            comparison["full_pipeline"] = {
                "run_id": full_pipeline_run_id,
                "summary": full_pipeline_summary,
                "manifest": full_pipeline_manifest,
            }
            print(f"Full-pipeline: {full_pipeline_run_id}")
        except ValueError:
            print("Full-pipeline: No runs found")

    # Print comparison table
    print("\n" + "-" * 80)
    print("METRIC COMPARISON")
    print("-" * 80)

    if "retriever" in comparison:
        print("\n### Retriever")
        metrics = comparison["retriever"]["summary"].get("overall", {}).get("metrics", {})
        for name, data in metrics.items():
            print(f"  {name}: {data.get('mean_score', 'N/A'):.3f}")

    if "generator" in comparison:
        print("\n### Generator")
        metrics = comparison["generator"]["summary"].get("overall", {}).get("metrics", {})
        for name, data in metrics.items():
            print(f"  {name}: {data.get('mean_score', 'N/A'):.3f}")

    if "full_pipeline" in comparison:
        print("\n### Full-Pipeline")
        metrics = comparison["full_pipeline"]["summary"].get("overall", {}).get("metrics", {})
        for name, data in metrics.items():
            print(f"  {name}: {data.get('mean_score', 'N/A'):.3f}")
        print(
            f"  Triad Mean: {comparison['full_pipeline']['summary'].get('overall', {}).get('triad_mean_score', 'N/A'):.3f}"
        )

    # Save comparison
    output_dir = ROOT / "eval" / "deepEval" / "consolidated_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"comparison_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "comparison": comparison,
            },
            f,
            indent=2,
        )

    print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-model", default="gpt-4o-mini", help="DeepEval judge model (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--context-k", type=int, default=5, help="Context chunks for full-pipeline (default: 5)"
    )
    parser.add_argument("--skip-retriever", action="store_true", help="Skip retriever evaluation")
    parser.add_argument("--skip-generator", action="store_true", help="Skip generator evaluation")
    parser.add_argument(
        "--skip-full-pipeline", action="store_true", help="Skip full-pipeline evaluation"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare latest runs without running new evaluations"
    )
    args = parser.parse_args()

    if args.compare:
        compare_all()
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"\nDeepEval All Evaluations - {timestamp}")
    print(f"Judge model: {args.judge_model}")
    print(f"Context-k: {args.context_k}")

    results = {}

    # 1. Retriever evaluation
    if not args.skip_retriever:
        print("\n" + "=" * 80)
        print("1. RETRIEVER EVALUATION")
        print("=" * 80)
        cmd = [
            "uv",
            "run",
            "python",
            "eval/deepEval/retriever_eval.py",
            "--judge-model",
            args.judge_model,
        ]
        result = run_command(cmd, ROOT)
        if result.returncode != 0:
            print(f"Retriever evaluation failed with exit code {result.returncode}")
        else:
            retriever_results_dir = ROOT / "eval" / "deepEval" / "retriever_results"
            retriever_run_id = get_latest_run_id(retriever_results_dir)
            retriever_summary = load_summary(retriever_results_dir, retriever_run_id)
            retriever_manifest = load_manifest(retriever_results_dir, retriever_run_id)
            results["retriever"] = {
                "run_id": retriever_run_id,
                "summary": retriever_summary,
                "manifest": retriever_manifest,
            }
            print(f"Retriever evaluation completed: {retriever_run_id}")

    # 2. Generator evaluation
    if not args.skip_generator:
        print("\n" + "=" * 80)
        print("2. GENERATOR EVALUATION")
        print("=" * 80)
        cmd = [
            "uv",
            "run",
            "python",
            "eval/deepEval/generator_eval.py",
            "--judge-model",
            args.judge_model,
        ]
        result = run_command(cmd, ROOT)
        if result.returncode != 0:
            print(f"Generator evaluation failed with exit code {result.returncode}")
        else:
            generator_results_dir = ROOT / "eval" / "deepEval" / "generation_results"
            generator_run_id = get_latest_run_id(generator_results_dir)
            generator_summary = load_summary(generator_results_dir, generator_run_id)
            generator_manifest = load_manifest(generator_results_dir, generator_run_id)
            results["generator"] = {
                "run_id": generator_run_id,
                "summary": generator_summary,
                "manifest": generator_manifest,
            }
            print(f"Generator evaluation completed: {generator_run_id}")

    # 3. Full-pipeline evaluation
    if not args.skip_full_pipeline:
        print("\n" + "=" * 80)
        print("3. FULL-PIPELINE EVALUATION")
        print("=" * 80)
        cmd = [
            "uv",
            "run",
            "python",
            "eval/deepEval/full_pipeline_eval.py",
            "--judge-model",
            args.judge_model,
            "--context-k",
            str(args.context_k),
        ]
        result = run_command(cmd, ROOT)
        if result.returncode != 0:
            print(f"Full-pipeline evaluation failed with exit code {result.returncode}")
        else:
            full_pipeline_results_dir = ROOT / "eval" / "deepEval" / "full_pipeline_results"
            full_pipeline_run_id = get_latest_run_id(full_pipeline_results_dir)
            full_pipeline_summary = load_summary(full_pipeline_results_dir, full_pipeline_run_id)
            full_pipeline_manifest = load_manifest(full_pipeline_results_dir, full_pipeline_run_id)
            results["full_pipeline"] = {
                "run_id": full_pipeline_run_id,
                "summary": full_pipeline_summary,
                "manifest": full_pipeline_manifest,
            }
            print(f"Full-pipeline evaluation completed: {full_pipeline_run_id}")

    # 4. Consolidated summary
    print("\n" + "=" * 80)
    print("CONSOLIDATED SUMMARY")
    print("=" * 80)

    if "retriever" in results:
        print("\n### Retriever Evaluation")
        print(f"Run ID: {results['retriever']['run_id']}")
        overall = results["retriever"]["summary"].get("overall", {})
        metrics = overall.get("metrics", {})
        print(
            f"  Contextual Precision: {metrics.get('contextual_precision', {}).get('mean_score', 'N/A'):.3f}"
        )
        print(
            f"  Contextual Recall: {metrics.get('contextual_recall', {}).get('mean_score', 'N/A'):.3f}"
        )
        print(f"  Mean Score: {overall.get('mean_score', 'N/A'):.3f}")

    if "generator" in results:
        print("\n### Generator Evaluation")
        print(f"Run ID: {results['generator']['run_id']}")
        overall = results["generator"]["summary"].get("overall", {})
        metrics = overall.get("metrics", {})
        print(f"  Faithfulness: {metrics.get('faithfulness', {}).get('mean_score', 'N/A'):.3f}")
        print(
            f"  Answer Relevancy: {metrics.get('answer_relevancy', {}).get('mean_score', 'N/A'):.3f}"
        )
        print(
            f"  Answer Correctness: {metrics.get('answer_correctness', {}).get('mean_score', 'N/A'):.3f}"
        )

    if "full_pipeline" in results:
        print("\n### Full-Pipeline Evaluation")
        print(f"Run ID: {results['full_pipeline']['run_id']}")
        overall = results["full_pipeline"]["summary"].get("overall", {})
        metrics = overall.get("metrics", {})
        print(
            f"  Contextual Relevancy: {metrics.get('contextual_relevancy', {}).get('mean_score', 'N/A'):.3f}"
        )
        print(f"  Faithfulness: {metrics.get('faithfulness', {}).get('mean_score', 'N/A'):.3f}")
        print(
            f"  Answer Relevancy: {metrics.get('answer_relevancy', {}).get('mean_score', 'N/A'):.3f}"
        )
        print(f"  Triad Mean: {overall.get('triad_mean_score', 'N/A'):.3f}")

    # Save consolidated summary
    output_dir = ROOT / "eval" / "deepEval" / "consolidated_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"consolidated_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "judge_model": args.judge_model,
                "context_k": args.context_k,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nConsolidated summary saved to: {output_path}")
    print("\nAll evaluations completed!")


if __name__ == "__main__":
    main()
