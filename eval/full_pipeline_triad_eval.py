r"""
End-to-End RAG Triad Evaluation for the SEC 10-K RAG System
===========================================================

This harness runs the *full* pipeline (retrieve -> generate) for each golden
query and scores the three legs of the RAG triad on the real output:

    query ----(Context Relevance)----> retrieved context
      \                                       |
       \                               (Groundedness)
        \                                     |
         ----------(Answer Relevance)----> answer

TRIAD LEGS (deterministic, golden-label scoring)
------------------------------------------------

1. Context Relevance  -- Is the retrieved context relevant to the query?
   Scored against the retrieval golden (`golden_retriever_dataset.jsonl`) by
   matching each retrieved chunk's `element_orders` to the graded relevant
   elements. Reuses the IR metric functions from `hybrid_rerank_eval` (hit,
   MRR, recall, precision, nDCG@k).

2. Groundedness       -- Is the answer supported by the retrieved context?
   Reuses `numeric_faithfulness` (numbers in the answer must appear in the
   cited source text) and citation precision/recall from `generation_eval`.

3. Answer Relevance   -- Does the answer actually address the query?
   Reuses `expected_value_coverage` / `lexical_required_fact_coverage` /
   `answer_correct` from `generation_eval`, scored against the structured
   answer gold (`expected_values`, `required_facts`) taken from the `oracle`
   variant of the generation golden (`golden_generation_dataset.jsonl`).

WHY DETERMINISTIC (NOT LLM-AS-JUDGE)
------------------------------------
This is the reference-based counterpart to the TruLens/RAGAS LLM-judge triad.
It reuses scorers that were already validated at the component level, so the
numbers are fully reproducible and need no judge model. An LLM-judge layer can
be added later as a second signal.

DATASET JOIN
------------
The retrieval golden (60 queries) is the driving set -- every query is
genuinely answerable from its filing. Structured answer gold is joined from the
generation golden by `source_qid == <retrieval id>` using the `oracle` variant
(the answerable one that carries the real expected values and facts).

USAGE
-----
    uv run python eval/full_pipeline_triad_eval.py
    uv run python eval/full_pipeline_triad_eval.py --limit 3 --verbose
    uv run python eval/full_pipeline_triad_eval.py --context-k 5 --device cuda
    uv run python eval/full_pipeline_triad_eval.py --query-ids q001 q014
    uv run python eval/full_pipeline_triad_eval.py --compare

OUTPUT
------
One run folder per experiment under `eval/rag_triad_results/`:
    manifest.json  -- config, dataset, prompt hash, environment
    cases.jsonl    -- per-query input / generation / metrics (for debugging)
    summary.json   -- triad headline scores + overall means + breakdowns
"""

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

ROOT = Path(__file__).resolve().parents[1]
# `src` is needed by the imported eval modules; `eval` lets us import them.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from generation import AnswerGenerator, GenerationConfig  # noqa: E402
from generation.prompts import SYSTEM_PROMPT  # noqa: E402

# Reuse the already-validated scorers rather than reimplementing them.
from generation_eval import score_case  # noqa: E402
from hybrid_rerank_eval import (  # noqa: E402
    GoldenQuery,
    compute_hit,
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_reciprocal_rank,
    load_golden_dataset,
)

DEFAULT_RETRIEVAL_GOLDEN = ROOT / "goldens" / "golden_retriever_dataset.jsonl"
DEFAULT_GENERATION_GOLDEN = ROOT / "goldens" / "golden_generation_dataset.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "rag_triad_results"

# Metric keys grouped by triad leg (used for aggregation + headline selection).
CONTEXT_KEYS = ["context_hit", "context_mrr", "context_recall", "context_precision", "context_ndcg"]
GROUNDEDNESS_KEYS = ["numeric_faithfulness", "citation_validity", "citation_precision", "citation_recall"]
ANSWER_KEYS = [
    "answer_correct", "expected_value_coverage", "lexical_required_fact_coverage",
    "abstention_correct", "hallucinated_answer", "false_abstention",
]
HEADLINE = {
    "context_relevance": "context_ndcg",
    "groundedness": "numeric_faithfulness",
    "answer_relevance": "answer_correct",
}


def load_answer_gold(path: Path) -> dict[str, dict[str, Any]]:
    """Load structured answer expectations from the generation golden dataset.

    Reads ``golden_generation_dataset.jsonl``, filters to
    ``context_variant == "oracle"`` (the answerable variant), and returns a
    dict keyed by ``source_qid`` so the retrieval golden can join on it.

    Args:
        path: Path to the generation golden JSONL file.

    Returns:
        Mapping of source_qid -> dict with keys ``answerable``,
        ``expected_answer``, ``expected_values``, ``required_facts``,
        ``gold_chunk_ids``.
    """
    gold: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            if case.get("context_variant") != "oracle":
                continue
            gold[case["source_qid"]] = {
                "answerable": True,
                "expected_answer": case.get("expected_answer"),
                "expected_values": case.get("expected_values", []),
                "required_facts": case.get("required_facts", []),
                "gold_chunk_ids": case.get("gold_chunk_ids", []),
            }
    return gold


def context_relevance_metrics(golden: GoldenQuery, result: Any, k: int) -> dict[str, Any]:
    """Score the Context Relevance leg of the RAG triad.

    Iterates over ``result.sources`` (the chunks the live pipeline actually
    retrieved and formatted), grades each against the golden relevant elements
    via ``element_orders`` intersection, and computes the standard IR metrics.

    Args:
        golden: The labeled retrieval query with expected relevant elements.
        result: A ``GenerationResult`` from ``AnswerGenerator.answer()``.
        k: Cut-off for nDCG calculation (typically ``context_k``).

    Returns:
        Dict containing ``context_hit``, ``context_mrr``, ``context_recall``,
        ``context_precision``, ``context_ndcg``, plus private keys
        ``_retrieved``, ``_total_relevant``, ``_relevant_retrieved`` (popped
        by the caller for per-case logging).
    """
    relevance_scores: list[int] = []
    found_relevant_groups: set[tuple[int, ...]] = set()
    retrieved: list[dict[str, Any]] = []

    for rank, source in enumerate(result.sources, start=1):
        orders = source.get("element_orders", []) or []
        chunk_doc_id = source.get("doc_id")
        grade = golden.get_relevance_grade(orders, chunk_doc_id)
        relevance_scores.append(grade)
        if grade > 0:
            for item in golden.relevant:
                if item["match_by"] != "element_orders":
                    continue
                item_doc = item.get("doc_id")
                if item_doc is not None and chunk_doc_id is not None and item_doc != chunk_doc_id:
                    continue
                if set(orders) & set(item["value"]):
                    found_relevant_groups.add(tuple(item["value"]))
        retrieved.append({
            "rank": rank,
            "source_id": source.get("source_id"),
            "chunk_id": source.get("chunk_id"),
            "doc_id": chunk_doc_id,
            "element_orders": orders,
            "score": source.get("score"),
            "relevance_grade": grade,
        })

    total_relevant = len(golden.relevant)
    found = len(found_relevant_groups)
    ideal_scores = golden.get_all_relevance_grades()
    return {
        "context_hit": compute_hit(relevance_scores),
        "context_mrr": compute_reciprocal_rank(relevance_scores),
        "context_recall": compute_recall(found, total_relevant),
        "context_precision": compute_precision(found, len(result.sources)),
        "context_ndcg": compute_ndcg(relevance_scores, ideal_scores, k),
        "_retrieved": retrieved,
        "_total_relevant": total_relevant,
        "_relevant_retrieved": found,
    }


def build_case(golden: GoldenQuery, answer_gold: dict[str, Any]) -> dict[str, Any]:
    """Merge retrieval golden and answer gold into a single case dict.

    Produces the dict shape that ``generation_eval.score_case()`` expects,
    bridging the retrieval golden's ``GoldenQuery`` fields with the structured
    answer expectations from ``load_answer_gold()``.

    Args:
        golden: Retrieval query with id, query, doc_id, category, difficulty.
        answer_gold: Oracle answer expectations (expected_values, required_facts, etc.).

    Returns:
        A flat dict consumable by both ``score_case()`` and the per-case log.
    """
    return {
        "id": golden.id,
        "query": golden.query,
        "doc_id": golden.doc_id,
        "category": golden.category,
        "difficulty": golden.difficulty,
        "answerable": answer_gold["answerable"],
        "expected_answer": answer_gold["expected_answer"],
        "expected_values": answer_gold["expected_values"],
        "required_facts": answer_gold["required_facts"],
        "gold_chunk_ids": answer_gold["gold_chunk_ids"],
    }


def metric_mean(records: list[dict[str, Any]], key: str) -> float | None:
    """Return the mean of *key* across successful records, or None if empty."""
    values = [
        record["metrics"][key]
        for record in records
        if record["status"] == "success" and record["metrics"].get(key) is not None
    ]
    return mean(values) if values else None


def triad_headlines(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the three headline scores and their mean (the triad score)."""
    legs = {leg: metric_mean(records, key) for leg, key in HEADLINE.items()}
    available = [value for value in legs.values() if value is not None]
    legs["triad_score"] = mean(available) if available else None
    return legs


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all triad metrics plus operational stats across records."""
    successful = [record for record in records if record["status"] == "success"]
    keys = CONTEXT_KEYS + GROUNDEDNESS_KEYS + ANSWER_KEYS
    summary = {key: metric_mean(records, key) for key in keys}
    summary.update({
        "cases": len(records),
        "successful_cases": len(successful),
        "error_cases": len(records) - len(successful),
        "total_latency_seconds": sum(r["generation"].get("latency_seconds", 0.0) for r in successful),
        "prompt_tokens": sum(r["generation"].get("prompt_tokens", 0) for r in successful),
        "completion_tokens": sum(r["generation"].get("completion_tokens", 0) for r in successful),
    })
    summary["average_latency_seconds"] = (
        summary["total_latency_seconds"] / len(successful) if successful else None
    )
    return summary


def breakdown(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    """Group records by a case *field* and aggregate each group separately."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["case"].get(field))].append(record)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def prompt_hash(system_prompt: str) -> str:
    """SHA-256 hex digest of the system prompt for change-detection across runs."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def run_evaluation(args: argparse.Namespace) -> Path:
    """Execute the full RAG triad evaluation.

    Loads golden datasets, initialises the generator, runs every selected query
    through the live retrieve-then-generate pipeline, scores all three triad
    legs, writes streaming ``cases.jsonl``, and saves ``manifest.json`` +
    ``summary.json`` to a timestamped run directory.

    Args:
        args: Parsed CLI arguments (golden paths, model config, filters, etc.).

    Returns:
        Path to the run directory containing the results.
    """
    goldens = load_golden_dataset(args.retrieval_golden)
    answer_gold = load_answer_gold(args.generation_golden)

    missing = [g.id for g in goldens if g.id not in answer_gold]
    if missing:
        raise ValueError(
            f"{len(missing)} retrieval queries have no 'oracle' answer gold: {missing[:5]}..."
        )

    if args.query_ids:
        selected = set(args.query_ids)
        goldens = [g for g in goldens if g.id in selected]
    if args.categories:
        wanted = set(args.categories)
        goldens = [g for g in goldens if g.category in wanted]
    if args.limit is not None:
        goldens = goldens[:args.limit]
    if not goldens:
        raise ValueError("No golden queries matched the requested filters")

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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir / f"{timestamp}_{args.provider}_{args.model.replace(':', '_')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cases_path = run_dir / "cases.jsonl"
    records: list[dict[str, Any]] = []

    with cases_path.open("w", encoding="utf-8") as output:
        for index, golden in enumerate(goldens, start=1):
            case = build_case(golden, answer_gold[golden.id])
            print(f"[{index}/{len(goldens)}] {golden.id} ({golden.category})")
            try:
                result = generator.answer(
                    golden.query,
                    doc_id=golden.doc_id,
                    context_k=args.context_k,
                    verbose=args.verbose,
                )
                context_metrics = context_relevance_metrics(golden, result, args.context_k)
                retrieved = context_metrics.pop("_retrieved")
                total_relevant = context_metrics.pop("_total_relevant")
                relevant_retrieved = context_metrics.pop("_relevant_retrieved")
                metrics = {**context_metrics, **score_case(case, result, args.fact_threshold)}
                record = {
                    "status": "success",
                    "case": case,
                    "generation": {
                        **result.to_dict(),
                        "context": result.context,
                        "raw_response": result.raw_response,
                    },
                    "retrieved": retrieved,
                    "total_relevant": total_relevant,
                    "relevant_retrieved": relevant_retrieved,
                    "metrics": metrics,
                }
            except Exception as error:
                record = {
                    "status": "error",
                    "case": case,
                    "generation": {},
                    "retrieved": [],
                    "metrics": {},
                    "error": f"{type(error).__name__}: {error}",
                }
                if args.fail_fast:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raise
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    manifest = {
        "run_id": run_dir.name,
        "timestamp": timestamp,
        "retrieval_golden": str(args.retrieval_golden),
        "generation_golden": str(args.generation_golden),
        "selected_query_count": len(goldens),
        "filters": {"query_ids": args.query_ids, "categories": args.categories, "limit": args.limit},
        "generation_config": config.to_dict(),
        "context_k": args.context_k,
        "fact_threshold": args.fact_threshold,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": prompt_hash(SYSTEM_PROMPT),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
    }
    summary = {
        "run_id": run_dir.name,
        "triad": triad_headlines(records),
        "overall": aggregate(records),
        "by_category": breakdown(records, "category"),
        "by_difficulty": breakdown(records, "difficulty"),
        "by_doc": breakdown(records, "doc_id"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary)
    print(f"\nResults saved to: {run_dir}")
    return run_dir


def format_metric(value: Any, percent: bool = True) -> str:
    """Format a metric value as a percentage string or decimal, or ``'—'`` if None."""
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%" if percent else f"{float(value):.2f}"


def print_summary(summary: dict[str, Any]) -> None:
    """Print a human-readable triad report to the terminal."""
    triad = summary["triad"]
    overall = summary["overall"]
    print("\n" + "=" * 60)
    print("RAG TRIAD (end-to-end)")
    print("=" * 60)
    print(f"  Context Relevance (nDCG@k)   : {format_metric(triad['context_relevance'])}")
    print(f"  Groundedness (num. faithful) : {format_metric(triad['groundedness'])}")
    print(f"  Answer Relevance (correct)   : {format_metric(triad['answer_relevance'])}")
    print("  " + "-" * 40)
    print(f"  Triad score (mean of 3)      : {format_metric(triad['triad_score'])}")
    print("\n  Context leg  : "
          f"hit={format_metric(overall['context_hit'])}  "
          f"recall={format_metric(overall['context_recall'])}  "
          f"MRR={format_metric(overall['context_mrr'])}")
    print("  Grounded leg : "
          f"cite_valid={format_metric(overall['citation_validity'])}  "
          f"cite_prec={format_metric(overall['citation_precision'])}  "
          f"cite_recall={format_metric(overall['citation_recall'])}")
    print("  Answer leg   : "
          f"value_cov={format_metric(overall['expected_value_coverage'])}  "
          f"fact_cov={format_metric(overall['lexical_required_fact_coverage'])}  "
          f"halluc={format_metric(overall['hallucinated_answer'])}")
    print(f"\n  Cases: {overall['cases']}  (errors: {overall['error_cases']})  "
          f"avg latency: {format_metric(overall['average_latency_seconds'], percent=False)}s")


def comparison_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields that must match for two runs to be fairly compared."""
    return {
        "retrieval_golden": manifest.get("retrieval_golden"),
        "generation_golden": manifest.get("generation_golden"),
        "selected_query_count": manifest.get("selected_query_count"),
        "filters": manifest.get("filters"),
        "context_k": manifest.get("context_k"),
        "system_prompt_sha256": manifest.get("system_prompt_sha256"),
        "fact_threshold": manifest.get("fact_threshold"),
    }


def compare_runs(results_dir: Path) -> Path | None:
    """Load all saved triad runs, print a side-by-side table, and save a comparison JSON.

    Scans *results_dir* for run folders containing ``summary.json``, collects
    their triad headlines and operational stats, detects whether runs are
    directly comparable (same dataset, prompt, config), and writes the
    comparison to a timestamped JSON file.

    Args:
        results_dir: Directory containing one subfolder per evaluation run.

    Returns:
        Path to the saved comparison JSON, or None if no runs found.
    """
    run_dirs = sorted(path.parent for path in results_dir.glob("*/summary.json"))
    rows = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        config = manifest.get("generation_config", {})
        triad = summary.get("triad", {})
        overall = summary.get("overall", {})
        rows.append({
            "run_id": manifest.get("run_id", run_dir.name),
            "provider": config.get("provider"),
            "model": config.get("model"),
            "context_k": manifest.get("context_k"),
            "cases": overall.get("cases"),
            "context_relevance": triad.get("context_relevance"),
            "groundedness": triad.get("groundedness"),
            "answer_relevance": triad.get("answer_relevance"),
            "triad_score": triad.get("triad_score"),
            "average_latency_seconds": overall.get("average_latency_seconds"),
            "comparison_signature": comparison_signature(manifest),
        })
    if not rows:
        print(f"No RAG triad runs found in: {results_dir}")
        return None

    signatures = {json.dumps(row["comparison_signature"], sort_keys=True) for row in rows}
    comparable = len(signatures) == 1
    headers = ["Run", "Model", "k", "Cases", "Context", "Grounded", "Answer", "Triad", "Latency"]
    display_rows = []
    for row in rows:
        display_rows.append([
            row["run_id"],
            f"{row['provider']}/{row['model']}",
            str(row["context_k"]),
            str(row["cases"]),
            format_metric(row["context_relevance"]),
            format_metric(row["groundedness"]),
            format_metric(row["answer_relevance"]),
            format_metric(row["triad_score"]),
            format_metric(row["average_latency_seconds"], percent=False),
        ])
    widths = [max(len(headers[i]), *(len(row[i]) for row in display_rows)) for i in range(len(headers))]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in display_rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    if not comparable:
        print("\nWARNING: Runs are not directly comparable. Dataset, context_k, prompt, or threshold differ.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = results_dir / f"comparison_{timestamp}.json"
    output_path.write_text(
        json.dumps({"timestamp": timestamp, "directly_comparable": comparable, "runs": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nComparison saved to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the triad evaluation harness."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-golden", type=Path, default=DEFAULT_RETRIEVAL_GOLDEN)
    parser.add_argument("--generation-golden", type=Path, default=DEFAULT_GENERATION_GOLDEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "anthropic", "google"])
    parser.add_argument("--model", default="qwen2.5-7b-instruct:latest")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-version", default="v1_strict_grounded")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fact-threshold", type=float, default=0.6)
    parser.add_argument("--query-ids", nargs="+")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare saved RAG triad runs")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.compare:
        compare_runs(arguments.output_dir)
    else:
        run_evaluation(arguments)
