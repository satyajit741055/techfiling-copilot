"""Aggregate QueryRecord dicts into dashboard-ready latency, cost and reliability summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

LATENCY_FIELDS = (
    "retrieval_seconds",
    "generation_seconds",
    "llm_seconds",
    "total_seconds",
    "client_e2e_seconds",
)


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile (same as numpy's default), pct in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(v) for v in values if v is not None]
    return {
        "count": len(data),
        "mean": mean(data) if data else None,
        "min": min(data) if data else None,
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": max(data) if data else None,
    }


def split_cold_warm(records: list[dict[str, Any]], warmup: int) -> tuple[list, list]:
    """Cold = flagged by the service (model load / first requests) or within the first `warmup`
    records of the run. Everything else is warm."""
    cold: list[dict[str, Any]] = []
    warm: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        (cold if record.get("is_cold") or index < warmup else warm).append(record)
    return cold, warm


def latency_section(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "success"]
    return {field: distribution(r.get(field) for r in ok if field in r) for field in LATENCY_FIELDS}


def cost_section(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "success"]
    n = len(ok)
    total = sum(r.get("total_cost_usd", 0.0) for r in ok)
    return {
        "billed_queries": n,
        "total_cost_usd": total,
        "cost_per_query_usd": total / n if n else None,
        "cost_per_query_distribution": distribution(r.get("total_cost_usd", 0.0) for r in ok),
        "input_cost_usd": sum(r.get("input_cost_usd", 0.0) for r in ok),
        "output_cost_usd": sum(r.get("output_cost_usd", 0.0) for r in ok),
        "infra_cost_usd": sum(r.get("infra_cost_usd", 0.0) for r in ok),
        "cost_per_1k_queries_usd": total / n * 1000 if n else None,
        "prompt_tokens": distribution(r.get("prompt_tokens", 0) for r in ok),
        "completion_tokens": distribution(r.get("completion_tokens", 0) for r in ok),
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in ok),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in ok),
    }


def reliability_section(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    success = sum(r.get("status") == "success" for r in records)
    timeouts = sum(r.get("status") == "timeout" for r in records)
    errors = n - success
    with_retry = sum(r.get("retries", 0) > 0 for r in records)
    total_retries = sum(r.get("retries", 0) for r in records)
    by_type: dict[str, int] = defaultdict(int)
    by_stage: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("status") != "success":
            by_type[r.get("error_type") or "unknown"] += 1
            by_stage[r.get("error_stage") or "unknown"] += 1

    def rate(count: int) -> float | None:
        return count / n if n else None

    successes = [r for r in records if r.get("status") == "success"]
    return {
        "requests": n,
        "success": success,
        "failed": errors,
        "timeouts": timeouts,
        "success_rate": rate(success),
        "error_rate": rate(errors),
        "timeout_rate": rate(timeouts),
        "queries_with_retry": with_retry,
        "retry_rate": rate(with_retry),
        "total_retries": total_retries,
        "retries_per_request": total_retries / n if n else None,
        "validation_failure_rate": (
            sum(r.get("validation_passed") is False for r in successes) / len(successes)
            if successes
            else None
        ),
        "insufficient_context_rate": (
            sum(bool(r.get("insufficient_context")) for r in successes) / len(successes)
            if successes
            else None
        ),
        "errors_by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "errors_by_stage": dict(by_stage),
    }


def agent_section(records: list[dict[str, Any]]) -> dict[str, Any]:
    """/chat turns: outcome mix, latency, per-node time, loop counts, validity and cost."""
    n = len(records)
    finished = [r for r in records if r.get("status") == "success"]
    answers = [r for r in finished if r.get("outcome") == "answer"]
    nodes: dict[str, list[float]] = defaultdict(list)
    for r in finished:
        for node, seconds in (r.get("node_seconds") or {}).items():
            nodes[node].append(seconds)
    by: dict[str, dict[str, int]] = {"status": {}, "outcome": {}, "route": {}}
    for r in records:
        for key, counts in by.items():
            value = str(r.get(key) or "none")
            counts[value] = counts.get(value, 0) + 1
    total_cost = sum(r.get("total_cost_usd", 0.0) for r in finished)
    return {
        "turns": n,
        "by_status": by["status"],
        "by_outcome": by["outcome"],
        "by_route": by["route"],
        "total_seconds": distribution(r.get("total_seconds") for r in finished),
        "node_seconds": {node: distribution(values) for node, values in sorted(nodes.items())},
        "llm_calls": distribution(r.get("llm_calls", 0) for r in finished),
        "tool_calls": distribution(r.get("tool_calls", 0) for r in finished),
        "replans": distribution(r.get("replans", 0) for r in finished),
        "repairs": distribution(r.get("repairs", 0) for r in finished),
        "invalid_answer_rate": (
            sum(r.get("is_valid") is False for r in answers) / len(answers) if answers else None
        ),
        "abstain_rate": (
            sum(bool(r.get("insufficient_context")) for r in answers) / len(answers)
            if answers
            else None
        ),
        "total_cost_usd": total_cost,
        "cost_per_turn_usd": total_cost / len(finished) if finished else None,
        "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in finished),
        "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in finished),
    }


def summarize(records: list[dict[str, Any]], warmup: int = 2) -> dict[str, Any]:
    """Full summary. Latency percentiles are reported for warm requests, cold requests are
    reported separately; cost and reliability cover all requests.

    /chat turns (``record_type == "agent"``) are summarised in their own ``agent`` section
    and left out of the /questions numbers.
    """
    agent = [r for r in records if r.get("record_type") == "agent"]
    records = [r for r in records if r.get("record_type") != "agent"]
    cold, warm = split_cold_warm(records, warmup)
    inits = [r["retriever_init_seconds"] for r in records if r.get("retriever_init_seconds")]
    return {
        "requests": len(records),
        "warmup_requests": warmup,
        "latency": {
            "warm": {"requests": len(warm), **latency_section(warm)},
            "cold": {
                "requests": len(cold),
                "retriever_init_seconds": distribution(inits),
                **latency_section(cold),
            },
            "all": {"requests": len(records), **latency_section(records)},
        },
        "cost": cost_section(records),
        "reliability": reliability_section(records),
        "agent": agent_section(agent),
    }
