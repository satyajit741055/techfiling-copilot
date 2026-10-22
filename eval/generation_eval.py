"""Isolated evaluation of the answer generator on frozen golden contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.settings import settings
from generation import AnswerGenerator, GenerationConfig
from generation.context import format_context
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_GOLDEN_PATH = ROOT / "goldens" / "golden_generation_dataset.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "generation_results"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
NUMBER_RE = re.compile(
    r"(?P<currency>\$)?\s*(?P<negative>\()?"
    r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>billion|million|thousand|percent|%|basis points?|bps)?\)?",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "were", "with", "year", "fiscal",
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    unit = unit.lower()
    return {"%": "percent", "basis point": "bps", "basis points": "bps"}.get(unit, unit)


def comparable_value(value: float, unit: str | None) -> tuple[float, str | None]:
    unit = normalize_unit(unit)
    if unit == "billion":
        return value * 1000, "million"
    if unit == "thousand":
        return value / 1000, "million"
    return value, unit


def extract_numbers(text: str | None) -> list[dict[str, Any]]:
    values = []
    for match in NUMBER_RE.finditer(text or ""):
        value = float(match.group("value").replace(",", ""))
        if match.group("negative"):
            value = -value
        values.append({
            "value": value,
            "unit": normalize_unit(match.group("unit")),
            "currency": "USD" if match.group("currency") else None,
            "raw": match.group(0).strip(),
        })
    return values


def values_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    expected_value, expected_unit = comparable_value(float(expected["value"]), expected.get("unit"))
    actual_value, actual_unit = comparable_value(float(actual["value"]), actual.get("unit"))
    if expected_unit == "percent":
        return actual_unit == "percent" and abs(expected_value - actual_value) <= 0.1
    if expected_unit and actual_unit and expected_unit != actual_unit:
        return False
    tolerance = 0.005 if expected_value else 0.0
    return math.isclose(expected_value, actual_value, rel_tol=tolerance, abs_tol=1e-9)


def expected_value_coverage(expected: list[dict[str, Any]], answer: str | None) -> float | None:
    if not expected:
        return None
    actual = extract_numbers(answer)
    matched = sum(any(values_match(item, candidate) for candidate in actual) for item in expected)
    return matched / len(expected)


def lexical_fact_coverage(facts: list[str], generated: str) -> float | None:
    if not facts:
        return None
    scores = []
    generated_tokens = set(TOKEN_RE.findall(generated.lower())) - STOPWORDS
    for fact in facts:
        fact_tokens = set(TOKEN_RE.findall(fact.lower())) - STOPWORDS
        scores.append(len(fact_tokens & generated_tokens) / len(fact_tokens) if fact_tokens else 1.0)
    return mean(scores)


def cited_source_ids(citations: list[Any]) -> list[str]:
    return [citation for citation in citations if isinstance(citation, str)]


def citation_metrics(case: dict[str, Any], result: Any) -> dict[str, Any]:
    source_by_id = {source["source_id"]: source for source in result.sources}
    citations = cited_source_ids(result.answer.citations)
    valid = [citation for citation in citations if citation in source_by_id]
    invalid = [citation for citation in citations if citation not in source_by_id]
    gold_ids = set(case["gold_chunk_ids"])
    supporting = [
        citation for citation in valid
        if source_by_id[citation].get("chunk_id") in gold_ids
    ]
    precision = len(supporting) / len(valid) if valid else None
    recall = len({source_by_id[c]["chunk_id"] for c in supporting}) / len(gold_ids) if gold_ids else None
    return {
        "citation_count": len(citations),
        "valid_citation_count": len(valid),
        "invalid_citations": invalid,
        "citation_validity": len(valid) / len(citations) if citations else (1.0 if result.answer.insufficient_context else 0.0),
        "citation_precision": precision,
        "citation_recall": recall,
        "supporting_citations": supporting,
    }


def numeric_faithfulness(result: Any) -> dict[str, Any]:
    source_by_id = {source["source_id"]: source for source in result.sources}
    cited_ids = [citation for citation in cited_source_ids(result.answer.citations) if citation in source_by_id]
    cited_chunk_ids = {source_by_id[citation]["chunk_id"] for citation in cited_ids}
    cited_text = "\n".join(
        block for source, block in zip(result.sources, result.context.split("\n\n<source "))
        if source.get("chunk_id") in cited_chunk_ids
    )
    claims = " ".join(filter(None, [result.answer.answer, result.answer.calculation]))
    claim_numbers = extract_numbers(claims)
    evidence_numbers = extract_numbers(cited_text)
    grounded = [any(values_match(claim, evidence) for evidence in evidence_numbers) for claim in claim_numbers]
    return {
        "numeric_claim_count": len(claim_numbers),
        "grounded_numeric_claim_count": sum(grounded),
        "numeric_faithfulness": sum(grounded) / len(grounded) if grounded else None,
        "ungrounded_numeric_claims": [claim["raw"] for claim, is_grounded in zip(claim_numbers, grounded) if not is_grounded],
    }


def score_case(case: dict[str, Any], result: Any, fact_threshold: float) -> dict[str, Any]:
    predicted_abstention = bool(result.answer.insufficient_context)
    abstention_correct = predicted_abstention == (not case["answerable"])
    generated_text = " ".join(filter(None, [result.answer.answer, result.answer.explanation]))
    value_coverage = expected_value_coverage(case["expected_values"], result.answer.answer)
    fact_coverage = lexical_fact_coverage(case["required_facts"], generated_text)
    if not case["answerable"]:
        answer_correct = predicted_abstention
    elif predicted_abstention:
        answer_correct = False
    elif value_coverage is not None:
        answer_correct = value_coverage == 1.0
    else:
        answer_correct = fact_coverage is not None and fact_coverage >= fact_threshold
    return {
        "answer_correct": answer_correct,
        "expected_value_coverage": value_coverage,
        "lexical_required_fact_coverage": fact_coverage,
        "predicted_abstention": predicted_abstention,
        "abstention_correct": abstention_correct,
        "hallucinated_answer": not case["answerable"] and not predicted_abstention,
        "false_abstention": case["answerable"] and predicted_abstention,
        **citation_metrics(case, result),
        **numeric_faithfulness(result),
    }


def metric_mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record["metrics"][key] for record in records if record["metrics"].get(key) is not None]
    return mean(values) if values else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "answer_correct", "expected_value_coverage", "lexical_required_fact_coverage",
        "abstention_correct", "hallucinated_answer", "false_abstention",
        "citation_validity", "citation_precision", "citation_recall", "numeric_faithfulness",
    ]
    successful = [record for record in records if record["status"] == "success"]
    summary = {key: metric_mean(successful, key) for key in keys}
    summary.update({
        "cases": len(records),
        "successful_cases": len(successful),
        "error_cases": len(records) - len(successful),
        "prompt_tokens": sum(record["generation"].get("prompt_tokens", 0) for record in successful),
        "completion_tokens": sum(record["generation"].get("completion_tokens", 0) for record in successful),
        "total_latency_seconds": sum(record["generation"].get("latency_seconds", 0.0) for record in successful),
    })
    summary["average_latency_seconds"] = (
        summary["total_latency_seconds"] / len(successful) if successful else None
    )
    return summary


def breakdown(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["case"].get(field))].append(record)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def prompt_hash(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def run_evaluation(args: argparse.Namespace) -> Path:
    cases = load_cases(args.golden_path)
    if args.variants:
        selected = set(args.variants)
        cases = [case for case in cases if case["context_variant"] in selected]
    if args.case_ids:
        selected_ids = set(args.case_ids)
        cases = [case for case in cases if case["id"] in selected_ids]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No golden cases matched the requested filters")

    config = GenerationConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_version=args.prompt_version,
    )
    generator = AnswerGenerator(config=config, system_prompt=SYSTEM_PROMPT)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = args.output_dir / re.sub(r"[^\w.-]+", "_", f"{timestamp}_{args.provider}_{args.model}")
    run_dir.mkdir(parents=True, exist_ok=False)
    cases_path = run_dir / "cases.jsonl"
    records = []

    with cases_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            chunks = [entry["chunk"] for entry in case["context"]]
            context, _ = format_context(chunks)
            user_prompt = build_user_prompt(case["query"], context)
            print(f"[{index}/{len(cases)}] {case['id']}")
            try:
                result = generator.generate(case["query"], chunks, doc_id=case["doc_id"], verbose=args.verbose)
                record = {
                    "status": "success",
                    "case": {key: value for key, value in case.items() if key != "context"},
                    "input": {
                        "chunk_ids": [entry["chunk_id"] for entry in case["context"]],
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "formatted_context": result.context,
                    },
                    "generation": {
                        **result.to_dict(),
                        "context": result.context,
                        "raw_response": result.raw_response,
                    },
                    "metrics": score_case(case, result, args.fact_threshold),
                }
            except Exception as error:
                record = {
                    "status": "error",
                    "case": {key: value for key, value in case.items() if key != "context"},
                    "input": {
                        "chunk_ids": [entry["chunk_id"] for entry in case["context"]],
                        "system_prompt": SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "formatted_context": context,
                    },
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

    manifest = {
        "run_id": run_dir.name,
        "timestamp": timestamp,
        "golden_path": str(args.golden_path),
        "selected_case_count": len(cases),
        "filters": {"variants": args.variants, "case_ids": args.case_ids, "limit": args.limit},
        "generation_config": config.to_dict(),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": prompt_hash(SYSTEM_PROMPT),
        "fact_threshold": args.fact_threshold,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "command": sys.argv,
    }
    summary = {
        "run_id": run_dir.name,
        "overall": aggregate(records),
        "by_variant": breakdown(records, "context_variant"),
        "by_category": breakdown(records, "category"),
        "by_difficulty": breakdown(records, "difficulty"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Results saved to: {run_dir}")
    return run_dir


def comparison_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "golden_path": manifest.get("golden_path"),
        "selected_case_count": manifest.get("selected_case_count"),
        "filters": manifest.get("filters"),
        "system_prompt_sha256": manifest.get("system_prompt_sha256"),
        "fact_threshold": manifest.get("fact_threshold"),
    }


def format_metric(value: Any, percent: bool = True) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%" if percent else f"{float(value):.2f}"


def compare_runs(results_dir: Path) -> Path | None:
    run_dirs = sorted(path.parent for path in results_dir.glob("*/summary.json"))
    rows = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        config = manifest.get("generation_config", {})
        overall = summary.get("overall", {})
        rows.append({
            "run_id": manifest.get("run_id", run_dir.name),
            "provider": config.get("provider"),
            "model": config.get("model"),
            "prompt_version": config.get("prompt_version"),
            "temperature": config.get("temperature"),
            "cases": overall.get("cases"),
            "answer_correct": overall.get("answer_correct"),
            "numeric_faithfulness": overall.get("numeric_faithfulness"),
            "abstention_correct": overall.get("abstention_correct"),
            "hallucinated_answer": overall.get("hallucinated_answer"),
            "false_abstention": overall.get("false_abstention"),
            "citation_precision": overall.get("citation_precision"),
            "citation_validity": overall.get("citation_validity"),
            "average_latency_seconds": overall.get("average_latency_seconds"),
            "prompt_tokens": overall.get("prompt_tokens"),
            "completion_tokens": overall.get("completion_tokens"),
            "error_cases": overall.get("error_cases"),
            "comparison_signature": comparison_signature(manifest),
        })
    if not rows:
        print(f"No generator evaluation runs found in: {results_dir}")
        return None

    signatures = {json.dumps(row["comparison_signature"], sort_keys=True) for row in rows}
    comparable = len(signatures) == 1
    headers = ["Run", "Model", "Cases", "Correct", "Faithful", "Abstain", "Halluc.", "Cite prec.", "Latency"]
    display_rows = []
    for row in rows:
        display_rows.append([
            row["run_id"],
            f"{row['provider']}/{row['model']}",
            str(row["cases"]),
            format_metric(row["answer_correct"]),
            format_metric(row["numeric_faithfulness"]),
            format_metric(row["abstention_correct"]),
            format_metric(row["hallucinated_answer"]),
            format_metric(row["citation_precision"]),
            format_metric(row["average_latency_seconds"], percent=False),
        ])
    widths = [max(len(headers[index]), *(len(row[index]) for row in display_rows)) for index in range(len(headers))]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in display_rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    if not comparable:
        print("\nWARNING: Runs are not directly comparable. Dataset filters, prompt hash, or metric threshold differ.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = results_dir / f"comparison_{timestamp}.json"
    output = {
        "timestamp": timestamp,
        "directly_comparable": comparable,
        "compatibility_fields": [
            "golden_path", "selected_case_count", "filters", "system_prompt_sha256", "fact_threshold",
        ],
        "runs": rows,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nComparison saved to: {output_path}")
    return output_path


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
    parser.add_argument("--fact-threshold", type=float, default=0.6)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Compare saved generator evaluation runs")
    args = parser.parse_args()
    args.provider = args.provider or settings.effective_llm_provider
    args.model = args.model or settings.model_for_provider(args.provider)
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.compare:
        compare_runs(arguments.output_dir)
    else:
        run_evaluation(arguments)
