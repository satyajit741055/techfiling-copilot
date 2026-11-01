"""Smoke-run the research agent on 5 representative questions.

Usage:
    uv run python scripts/agent_smoke.py [--max-tool-steps 0|2|6] [--only ID,ID] [--out DIR]

Runs against the configured vector store (Docker Qdrant) and agent LLM (e.g. local
Ollama), through the same AgentService that serves /chat. Prints route, outcome, coverage,
validity, tool calls and latency per question and writes one JSONL row per turn to
eval/agent_smoke/. --max-tool-steps overrides AGENT_MAX_TOOL_STEPS for a quick
deterministic / hybrid / ReAct-style sanity check (the full ablation is M3).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.service import AgentService, ChatTurn  # noqa: E402
from config.settings import settings  # noqa: E402

# "thread" groups turns into one conversation; "text" answers the previous clarification.
QUESTIONS: list[dict[str, Any]] = [
    {"id": "single_fact", "message": "What was Apple's total net sales in fiscal 2025?"},
    {
        "id": "cross_company",
        "message": "Compare research and development expenses for Apple and Microsoft "
        "in their latest fiscal year.",
    },
    {"id": "ambiguous", "thread": "A", "message": "What was revenue last year?"},
    {"id": "clarify_answer", "thread": "A", "text": "Microsoft"},
    {"id": "not_in_corpus", "message": "What was Nvidia's revenue in fiscal 2024?"},
]


def _summary(case: dict[str, Any], events: list[Any], seconds: float) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_name.setdefault(event.event, []).append(event.data)
    answer = (by_name.get("answer") or [None])[0]
    clarification = (by_name.get("clarification") or [None])[0]
    error = (by_name.get("error") or [None])[0]
    done = (by_name.get("done") or [{}])[0]
    if error:
        outcome = f"error:{error['code']}"
    elif clarification:
        outcome = "clarification"
    else:
        outcome = "answer" if answer and answer.get("answer") else "abstain"
    return {
        "id": case["id"],
        "input": case.get("message") or case.get("text"),
        "outcome": outcome,
        "route": answer["route"] if answer else None,
        "is_valid": answer["is_valid"] if answer else None,
        "coverage": [
            f"{cell['ticker']} FY{cell['fiscal_year']}:{cell['status']}"
            for cell in (answer or {}).get("coverage", [])
        ],
        "assumptions": (answer or {}).get("assumptions", []),
        "answer": (answer or {}).get("answer"),
        "claims": len((answer or {}).get("claims", [])),
        "dropped_claims": (answer or {}).get("dropped_claims", []),
        "validation_errors": (answer or {}).get("validation_errors", []),
        "clarification": clarification,
        "steps": [e["label"] for e in by_name.get("step", []) if e["status"] == "started"],
        "trajectory": (answer or {}).get("trajectory", []),
        "tool_calls": done.get("tool_calls", 0),
        "llm_calls": done.get("llm_calls", 0),
        "prompt_tokens": done.get("prompt_tokens", 0),
        "completion_tokens": done.get("completion_tokens", 0),
        "latency_seconds": round(seconds, 2),
    }


async def run(cases: list[dict[str, Any]], service: AgentService) -> list[dict[str, Any]]:
    threads: dict[str, uuid.UUID] = {}
    rows = []
    for case in cases:
        thread = threads.setdefault(case.get("thread") or case["id"], uuid.uuid4())
        turn = ChatTurn(
            thread_id=thread,
            owner="smoke",
            message=case.get("message"),
            text=case.get("text"),
            request_id=f"smoke-{case['id']}",
        )
        start = time.perf_counter()
        events = [event async for event in service.stream(turn)]
        row = _summary(case, events, time.perf_counter() - start)
        rows.append(row)
        coverage = ", ".join(row["coverage"]) or "-"
        print(
            f"{row['id']:<18} {row['outcome']:<24} route={row['route'] or '-':<10} "
            f"valid={row['is_valid']!s:<5} tools={row['tool_calls']:<2} "
            f"llm={row['llm_calls']:<2} {row['latency_seconds']:>6.1f}s  {coverage}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--max-tool-steps", type=int, help="override AGENT_MAX_TOOL_STEPS")
    parser.add_argument("--only", help="comma-separated question ids to run")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "agent_smoke")
    args = parser.parse_args()

    from generation import AnswerGenerator, GenerationConfig

    config = settings
    if args.max_tool_steps is not None:
        config = settings.model_copy(update={"AGENT_MAX_TOOL_STEPS": args.max_tool_steps})
    cases = QUESTIONS
    if args.only:
        wanted = set(args.only.split(","))
        cases = [case for case in QUESTIONS if case["id"] in wanted]

    print("Loading retriever, reranker and filing catalog...", flush=True)
    generator = AnswerGenerator(GenerationConfig(), root=ROOT)
    filings = generator.retriever.retriever.store.list_filings()
    service = AgentService.from_generator(generator, filings, config=config)
    steps = config.AGENT_MAX_TOOL_STEPS
    print(
        f"Agent {service.provider}/{service.model} · {len(filings)} filings · "
        f"max_tool_steps={steps}\n",
        flush=True,
    )

    rows = asyncio.run(run(cases, service))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = args.out / f"{stamp}_steps{steps}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({**row, "max_tool_steps": steps, "model": service.model}) + "\n")
    latencies = sorted(row["latency_seconds"] for row in rows)
    errors = sum(row["outcome"].startswith("error") for row in rows)
    print(
        f"\n{len(rows)} turns · {errors} errors · median {latencies[len(latencies) // 2]:.1f}s · "
        f"max {latencies[-1]:.1f}s · wrote {path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
