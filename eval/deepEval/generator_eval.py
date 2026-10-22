"""DeepEval-based evaluation of the generator on frozen ideal contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config.settings import settings
from generation import AnswerGenerator, GenerationConfig
from generation.prompts import SYSTEM_PROMPT

DEFAULT_GOLDEN_PATH = ROOT / "goldens" / "golden_generator_isolation_dataset.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "deepEval" / "generation_results"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "answer_correctness")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = {
                "source_qid", "doc_id", "query", "expected_answer", "ideal_context"
            } - case.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
            if not isinstance(case["ideal_context"], list) or not case["ideal_context"]:
                raise ValueError(f"{path}:{line_number}: ideal_context must be a non-empty list")
            if not all(isinstance(context, str) and context for context in case["ideal_context"]):
                raise ValueError(f"{path}:{line_number}: ideal_context values must be non-empty strings")
            cases.append(case)
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def select_cases(
    cases: list[dict[str, Any]], case_ids: list[str] | None, limit: int | None
) -> list[dict[str, Any]]:
    if case_ids:
        selected = set(case_ids)
        cases = [case for case in cases if case["source_qid"] in selected]
        missing = selected - {case["source_qid"] for case in cases}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No golden cases matched the requested filters")
    return cases


def build_chunks(case: dict[str, Any]) -> list[dict[str, Any]]:
    orders = case.get("source_element_order", [])
    return [
        {
            "chunk_id": f"{case['source_qid']}_context_{index}",
            "doc_id": case["doc_id"],
            "type": "golden_context",
            "section": "Generator isolation golden context",
            "element_orders": [orders[index - 1]] if index <= len(orders) else [],
            "text": context,
        }
        for index, context in enumerate(case["ideal_context"], start=1)
    ]


def build_metrics(args: argparse.Namespace) -> dict[str, Any]:
    selected = set(args.metrics)
    metrics: dict[str, Any] = {}
    if "faithfulness" in selected:
        metrics["faithfulness"] = FaithfulnessMetric(
            threshold=args.threshold,
            model=args.judge_model,
            include_reason=True,
            async_mode=False,
            strict_mode=args.strict,
        )
    if "answer_relevancy" in selected:
        metrics["answer_relevancy"] = AnswerRelevancyMetric(
            threshold=args.threshold,
            model=args.judge_model,
            include_reason=True,
            async_mode=False,
            strict_mode=args.strict,
        )
    if "answer_correctness" in selected:
        metrics["answer_correctness"] = GEval(
            name="Answer Correctness",
            criteria=(
                "Determine whether the actual output correctly answers the input when compared "
                "with the expected output. Penalize incorrect entities, dates, periods, units, "
                "values, relationships, unsupported additions, and missing required facts. "
                "Allow concise paraphrases that preserve the expected meaning."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=args.threshold,
            model=args.judge_model,
            async_mode=False,
            strict_mode=args.strict,
        )
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


def aggregate(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    successful = [record for record in records if record["status"] == "success"]
    metrics = {}
    for name in metric_names:
        results = [
            record["metrics"][name]
            for record in successful
            if name in record["metrics"] and record["metrics"][name].get("score") is not None
        ]
        metrics[name] = {
            "mean_score": mean(result["score"] for result in results) if results else None,
            "pass_rate": mean(result["passed"] for result in results) if results else None,
            "scored_cases": len(results),
        }
    return {
        "cases": len(records),
        "successful_cases": len(successful),
        "error_cases": len(records) - len(successful),
        "prompt_tokens": sum(record["generation"].get("prompt_tokens", 0) for record in successful),
        "completion_tokens": sum(
            record["generation"].get("completion_tokens", 0) for record in successful
        ),
        "generation_latency_seconds": sum(
            record["generation"].get("latency_seconds", 0.0) for record in successful
        ),
        "metrics": metrics,
    }


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def run_evaluation(args: argparse.Namespace) -> Path:
    cases = select_cases(load_cases(args.golden_path), args.case_ids, args.limit)
    config = GenerationConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_version=args.prompt_version,
    )
    generator = AnswerGenerator(config=config, system_prompt=SYSTEM_PROMPT)
    metrics = build_metrics(args)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir / re.sub(r"[^\w.-]+", "_", f"{timestamp}_{args.provider}_{args.model}")
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []

    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['source_qid']}")
            try:
                result = generator.generate(
                    case["query"], build_chunks(case), doc_id=case["doc_id"], verbose=args.verbose
                )
                actual_output = result.answer.answer or result.answer.explanation
                test_case = LLMTestCase(
                    name=case["source_qid"],
                    input=case["query"],
                    actual_output=actual_output,
                    expected_output=case["expected_answer"],
                    retrieval_context=case["ideal_context"],
                    metadata={"doc_id": case["doc_id"]},
                )
                scores = {name: metric_result(metric, test_case) for name, metric in metrics.items()}
                record = {
                    "status": "success",
                    "case": case,
                    "actual_output": actual_output,
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

    summary = aggregate(records, list(metrics))
    manifest = {
        "run_id": run_dir.name,
        "timestamp": timestamp,
        "golden_path": str(args.golden_path.resolve()),
        "selected_case_count": len(cases),
        "case_ids": args.case_ids,
        "limit": args.limit,
        "generator_config": config.to_dict(),
        "judge_model": args.judge_model,
        "metrics": list(metrics),
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
    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {run_dir}")
    return run_dir


def dry_run(args: argparse.Namespace) -> None:
    cases = select_cases(load_cases(args.golden_path), args.case_ids, args.limit)
    summary = {
        "golden_path": str(args.golden_path.resolve()),
        "selected_cases": len(cases),
        "case_ids": [case["source_qid"] for case in cases],
        "context_elements": sum(len(case["ideal_context"]) for case in cases),
        "metrics": args.metrics,
        "generator": f"{args.provider}/{args.model}",
        "judge_model": args.judge_model,
    }
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--provider", choices=["ollama", "huggingface", "openai", "anthropic", "google"],
        help="Generator provider (default: from .env execution mode / LLM_PROVIDER)",
    )
    parser.add_argument("--model", help="Generator model (default: provider default from .env)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-version", default="v1_strict_grounded")
    parser.add_argument("--judge-model", help="DeepEval judge (default: DEEPEVAL_JUDGE_MODEL)")
    parser.add_argument("--metrics", nargs="+", choices=METRIC_NAMES, default=list(METRIC_NAMES))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    args.provider = args.provider or settings.effective_llm_provider
    args.model = args.model or settings.model_for_provider(args.provider)
    args.judge_model = args.judge_model or settings.DEEPEVAL_JUDGE_MODEL
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.dry_run:
        dry_run(arguments)
    else:
        run_evaluation(arguments)
