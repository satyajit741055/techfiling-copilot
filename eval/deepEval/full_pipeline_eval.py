"""DeepEval RAG-triad evaluation of the full retrieve-then-generate pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from generation import AnswerGenerator, GenerationConfig
from generation.prompts import SYSTEM_PROMPT

DEFAULT_GOLDEN_PATH = ROOT / "goldens" / "golden_retriever_dataset_v2.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "deepEval" / "full_pipeline_results"
METRIC_NAMES = ("contextual_relevancy", "faithfulness", "answer_relevancy")


def load_cases(path: Path) -> list[dict[str, Any]]:
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
    selected = set(args.metrics)
    common = {
        "threshold": args.threshold,
        "model": args.judge_model,
        "include_reason": True,
        "async_mode": False,
        "strict_mode": args.strict,
    }
    metrics: dict[str, Any] = {}
    if "contextual_relevancy" in selected:
        metrics["contextual_relevancy"] = ContextualRelevancyMetric(**common)
    if "faithfulness" in selected:
        metrics["faithfulness"] = FaithfulnessMetric(**common)
    if "answer_relevancy" in selected:
        metrics["answer_relevancy"] = AnswerRelevancyMetric(**common)
    return metrics


def metric_result(metric: Any, test_case: LLMTestCase) -> dict[str, Any]:
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
                "evaluation_model": getattr(metric, "evaluation_model", argsafe_model(metric)),
            }
    return results


def argsafe_model(metric: Any) -> str | None:
    model = getattr(metric, "model", None)
    return model if isinstance(model, str) else getattr(model, "model_name", None)


def aggregate(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    pipeline_successes = [record for record in records if record["status"] == "success"]
    metrics = {}
    for name in metric_names:
        results = [
            record["metrics"][name]
            for record in pipeline_successes
            if name in record["metrics"] and record["metrics"][name].get("score") is not None
        ]
        metrics[name] = {
            "mean_score": mean(result["score"] for result in results) if results else None,
            "pass_rate": mean(result["passed"] for result in results) if results else None,
            "scored_cases": len(results),
            "metric_error_cases": sum(
                name in record.get("metrics", {})
                and record["metrics"][name].get("score") is None
                for record in pipeline_successes
            ),
        }
    scores = [metric["mean_score"] for metric in metrics.values() if metric["mean_score"] is not None]
    return {
        "cases": len(records),
        "successful_pipeline_cases": len(pipeline_successes),
        "pipeline_error_cases": len(records) - len(pipeline_successes),
        "triad_mean_score": mean(scores) if scores else None,
        "prompt_tokens": sum(record["generation"].get("prompt_tokens", 0) for record in pipeline_successes),
        "completion_tokens": sum(
            record["generation"].get("completion_tokens", 0) for record in pipeline_successes
        ),
        "pipeline_latency_seconds": sum(
            record["generation"].get("latency_seconds", 0.0) for record in pipeline_successes
        ),
        "metrics": metrics,
    }


def breakdown(
    records: list[dict[str, Any]], field: str, metric_names: list[str]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["case"].get(field))].append(record)
    return {name: aggregate(items, metric_names) for name, items in sorted(groups.items())}


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def run_evaluation(args: argparse.Namespace) -> Path:
    cases = select_cases(load_cases(args.golden_path), args)
    config = GenerationConfig(
        context_k=args.context_k,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_version=args.prompt_version,
        device=args.device,
    )
    generator = AnswerGenerator(config=config, system_prompt=SYSTEM_PROMPT)
    metrics = build_metrics(args)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir / f"{timestamp}_{args.provider}_{args.model.replace(':', '_')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []

    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']} ({case.get('category', 'uncategorized')})")
            try:
                result = generator.answer(
                    case["query"],
                    doc_id=case["doc_id"],
                    context_k=args.context_k,
                    verbose=args.verbose,
                )
                retrieval_context = [
                    source["text"] for source in result.sources if source.get("text")
                ]
                if not retrieval_context:
                    raise ValueError("Retriever returned no non-empty context chunks")
                actual_output = result.answer.answer or result.answer.explanation
                test_case = LLMTestCase(
                    name=case["id"],
                    input=case["query"],
                    actual_output=actual_output,
                    expected_output=case.get("expected_answer"),
                    retrieval_context=retrieval_context,
                    metadata={
                        "doc_id": case["doc_id"],
                        "category": case.get("category"),
                        "difficulty": case.get("difficulty"),
                    },
                )
                scores = score_metrics(metrics, test_case, args.fail_fast)
                record = {
                    "status": "success",
                    "case": case,
                    "actual_output": actual_output,
                    "retrieval_context": retrieval_context,
                    "retrieved_sources": result.sources,
                    "generation": result.to_dict(),
                    "raw_response": result.raw_response,
                    "formatted_context": result.context,
                    "metrics": scores,
                }
            except Exception as error:
                record = {
                    "status": "error",
                    "case": case,
                    "generation": {},
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
        "generation_config": config.to_dict(),
        "judge_model": args.judge_model,
        "metrics": metric_names,
        "threshold": args.threshold,
        "strict": args.strict,
        "system_prompt_sha256": prompt_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2))
    print(f"Results saved to: {run_dir}")
    return run_dir


def dry_run(args: argparse.Namespace) -> None:
    cases = select_cases(load_cases(args.golden_path), args)
    summary = {
        "golden_path": str(args.golden_path.resolve()),
        "selected_queries": len(cases),
        "query_ids": [case["id"] for case in cases],
        "categories": sorted({str(case.get("category")) for case in cases}),
        "difficulties": sorted({str(case.get("difficulty")) for case in cases}),
        "context_k": args.context_k,
        "metrics": args.metrics,
        "generator": f"{args.provider}/{args.model}",
        "judge_model": args.judge_model,
    }
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "anthropic", "google"])
    parser.add_argument("--model", default="qwen2.5-7b-instruct:latest")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-version", default="v1_strict_grounded")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--metrics", nargs="+", choices=METRIC_NAMES, default=list(METRIC_NAMES))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--query-ids", nargs="+")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--difficulties", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.context_k < 1:
        parser.error("--context-k must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.dry_run:
        dry_run(arguments)
    else:
        run_evaluation(arguments)
