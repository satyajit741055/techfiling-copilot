"""Numeric accuracy of research chat with the facts store off and on (M4).

Usage:
    uv run python scripts/numeric_eval.py [--facts off|postgres|both] [--only ID,ID]

Runs every question in eval/facts/numeric_golden.jsonl through the same AgentService
that serves /chat (configured Qdrant + agent LLM), once per facts mode, and grades the
answer text against the reported figures:

- exact match: every expected figure (and computed %) appears in the answer, no
  forbidden value (segment questions must not answer with the company total) and no
  other-year column value instead of the expected one;
- wrong period: an expected figure is missing and a value from another fiscal-year
  column of the same table row is stated instead (the 211,915-for-FY2024 error);
- invalid / dropped: the agent validator rejected the answer or dropped claims.

Writes one JSONL row per question and mode plus a summary JSON to eval/facts/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GOLDEN = ROOT / "eval" / "facts" / "numeric_golden.jsonl"
_NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-−])?\$?(?P<digits>\d[\d,]*(?:\.\d+)?)\s*(?P<pct>%)?"
    r"(?:\s*(?P<word>thousand|million|billion|trillion|bn)\b)?",
    re.IGNORECASE,
)
_MILLIONS = {"thousand": 0.001, "million": 1.0, "billion": 1_000.0, "bn": 1_000.0}
_MILLIONS["trillion"] = 1_000_000.0
_MAX_ROUNDING = 0.005  # a rounded figure may differ from the table by at most 0.5%


def _numbers(text: str) -> list[tuple[float, int, str | None, bool]]:
    """(value, decimals, scale word, is percent) for every number in ``text``."""
    found = []
    for match in _NUMBER.finditer(text or ""):
        digits = match["digits"].rstrip(",")
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        word = (match["word"] or "").lower() or None
        decimals = len(digits.partition(".")[2])
        found.append((value, decimals, word, bool(match["pct"])))
    return found


def _states_figure(text: str, figure: dict[str, Any], value: float | None = None) -> bool:
    """Whether ``text`` states a reported figure, as printed or rounded with a scale word."""
    target = abs(value if value is not None else figure["value"])
    for number, decimals, word, percent in _numbers(text):
        if percent:
            continue
        if figure["unit"] != "USD" or figure.get("scale") != 6:
            if abs(number - target) <= 0.5 * 10 ** -max(decimals, 2) + 1e-9:
                return True
            continue
        millions = number * _MILLIONS.get(word or "million", 1.0)
        tolerance = 0.5 * 10**-decimals * _MILLIONS.get(word or "million", 1.0)
        if abs(millions - target) <= min(tolerance, _MAX_ROUNDING * target) + 1e-9:
            return True
    return False


def _states_percent(text: str, expected: float) -> bool:
    return any(
        percent and abs(number - abs(expected)) <= 0.5 * 10**-decimals + 0.01
        for number, decimals, _, percent in _numbers(text)
    )


def grade(case: dict[str, Any], answer: str | None) -> dict[str, Any]:
    """Grade one answer text against a golden case.

    For questions that ask for a percentage (growth, CAGR, margin), the figures are the
    calculation's inputs: the answer need not state them, but stating another year's
    column instead still counts as a wrong period.
    """
    text = answer or ""
    facts = case.get("facts", [])
    percents = case.get("percents", [])
    expected_values = {abs(item["value"]) for item in facts}
    asked = [] if percents else facts
    unstated = [item for item in facts if not _states_figure(text, item)]
    missing = [item for item in unstated if item in asked]
    wrong_period = [
        f"{item['ticker']} FY{item['fiscal_year']} {item['metric']}: {other:g}"
        for item in unstated
        for other in item.get("other_columns", [])
        if abs(other) not in expected_values and _states_figure(text, item, other)
    ]
    percents_missing = [p for p in percents if not _states_percent(text, p)]
    forbidden = [
        value
        for value in case.get("forbidden", [])
        if _states_figure(text, {"value": value, "unit": "USD", "scale": 6})
    ]
    graded = bool(facts or percents or case.get("forbidden"))
    return {
        "graded": graded,
        "exact": graded and not (missing or percents_missing or forbidden or wrong_period),
        "facts_hit": len(asked) - len(missing),
        "facts_total": len(asked),
        "missing": [f"{i['ticker']} FY{i['fiscal_year']} {i['metric']}" for i in missing],
        "percents_missing": percents_missing,
        "wrong_period": wrong_period,
        "forbidden": forbidden,
    }


def _answer_row(case: dict[str, Any], events: list[Any], seconds: float) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_name.setdefault(event.event, []).append(event.data)
    answer = (by_name.get("answer") or [{}])[0]
    error = (by_name.get("error") or [None])[0]
    done = (by_name.get("done") or [{}])[0]
    text = answer.get("answer")
    if error:
        outcome = f"error:{error['code']}"
    elif by_name.get("clarification"):
        outcome = "clarification"
    else:
        outcome = "answer" if text else "abstain"
    tools = [step.get("tool") for step in answer.get("trajectory", [])]
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "outcome": outcome,
        "route": answer.get("route"),
        "answer": text,
        "is_valid": answer.get("is_valid"),
        "dropped_claims": len(answer.get("dropped_claims", [])),
        "facts_used": "query_financials" in tools,
        "searches": sum(tool not in {"query_financials", "worker_llm"} for tool in tools),
        "llm_calls": done.get("llm_calls", 0),
        "prompt_tokens": done.get("prompt_tokens", 0),
        "completion_tokens": done.get("completion_tokens", 0),
        "latency_seconds": round(seconds, 2),
        **grade(case, text),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [row for row in rows if row["graded"]]
    numeric = [row for row in graded if row["facts_total"]]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return round(sum(bool(row[key]) for row in items) / len(items), 3) if items else 0.0

    return {
        "questions": len(rows),
        "graded": len(graded),
        "exact_match": rate(graded, "exact"),
        "fact_recall": round(
            sum(row["facts_hit"] for row in numeric)
            / max(1, sum(row["facts_total"] for row in numeric)),
            3,
        ),
        "wrong_period_rate": rate(graded, "wrong_period"),
        "forbidden_violations": sum(bool(row["forbidden"]) for row in graded),
        "invalid_answers": sum(row["is_valid"] is False for row in rows),
        "dropped_claims": sum(row["dropped_claims"] for row in rows),
        "abstained": sum(row["outcome"] == "abstain" for row in rows),
        "errors": sum(row["outcome"].startswith("error") for row in rows),
        "facts_used": sum(row["facts_used"] for row in rows),
        "mean_llm_calls": round(statistics.fmean(row["llm_calls"] for row in rows), 2),
        "mean_searches": round(statistics.fmean(row["searches"] for row in rows), 2),
        "median_latency_seconds": round(statistics.median(r["latency_seconds"] for r in rows), 1),
    }


async def _run(cases: list[dict[str, Any]], service: Any, mode: str) -> list[dict[str, Any]]:
    from agent.service import ChatTurn

    rows = []
    for case in cases:
        turn = ChatTurn(
            thread_id=uuid.uuid4(),
            owner="numeric-eval",
            message=case["question"],
            request_id=f"numeric-{mode}-{case['id']}",
        )
        start = time.perf_counter()
        events = [event async for event in service.stream(turn)]
        row = _answer_row(case, events, time.perf_counter() - start)
        rows.append(row)
        verdict = "-" if not row["graded"] else ("PASS" if row["exact"] else "FAIL")
        issues = row["missing"] + row["wrong_period"] + [f"{p}%" for p in row["percents_missing"]]
        print(
            f"[{mode}] {row['id']:<24} {verdict:<4} {row['outcome']:<10} "
            f"route={row['route'] or '-':<10} facts={row['facts_used']!s:<5} "
            f"llm={row['llm_calls']:<2} {row['latency_seconds']:>6.1f}s"
            + (f"  wrong-period: {row['wrong_period']}" if row["wrong_period"] else "")
            + (f"  missing: {issues}" if issues and not row["wrong_period"] else ""),
            flush=True,
        )
    return rows


def _facts_store(settings: Any) -> Any:
    from facts.store import FactsStore
    from persistence.facts import read_facts

    records, load = read_facts(settings.HISTORY_DB_URL or "")
    if not records or load is None or load.collection != settings.QDRANT_COLLECTION:
        raise SystemExit("financial_facts is empty or linked to another collection")
    return FactsStore(records, load.collection)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--facts", choices=["off", "postgres", "both"], default="both")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "facts" / "results")
    args = parser.parse_args()

    from agent.service import AgentService
    from config.settings import settings
    from generation import AnswerGenerator, GenerationConfig

    cases = [json.loads(line) for line in args.golden.read_text(encoding="utf-8").splitlines()]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [case for case in cases if case["id"] in wanted]
    modes = ["off", "postgres"] if args.facts == "both" else [args.facts]

    print("Loading retriever, reranker and filing catalog...", flush=True)
    generator = AnswerGenerator(GenerationConfig(), root=ROOT)
    filings = generator.retriever.retriever.store.list_filings()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for mode in modes:
        config = settings.model_copy(update={"FACTS_STORE": mode})
        facts = _facts_store(config) if mode == "postgres" else None
        service = AgentService.from_generator(generator, filings, config=config, facts=facts)
        print(f"\nfacts={mode} · {service.provider}/{service.model} · {len(cases)} questions")
        rows = asyncio.run(_run(cases, service, mode))
        with (args.out / f"{stamp}_facts-{mode}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps({**row, "facts_mode": mode, "model": service.model}) + "\n")
        summaries[mode] = {"model": service.model, **summarize(rows)}

    path = args.out / f"{stamp}_summary.json"
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    keys = [key for key in next(iter(summaries.values())) if key != "model"]
    print(f"\n{'metric':<24}" + "".join(f"{mode:>12}" for mode in summaries))
    for key in keys:
        print(f"{key:<24}" + "".join(f"{summaries[m][key]!s:>12}" for m in summaries))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
