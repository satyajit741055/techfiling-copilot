"""Offline tests for latency/cost/reliability instrumentation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config.settings import settings
from generation.generator import AnswerGenerator
from generation.schemas import LLMResponse
from observability import JsonlMetricsWriter, QueryTracker, compute_cost, percentile, summarize
from observability.records import is_timeout_error
from observability.summary import split_cold_warm


def test_percentile_matches_linear_interpolation():
    data = list(range(1, 101))
    assert percentile(data, 50) == pytest.approx(50.5)
    assert percentile(data, 95) == pytest.approx(95.05)
    assert percentile(data, 99) == pytest.approx(99.01)
    assert percentile([], 50) is None
    assert percentile([3.0], 99) == 3.0


def test_cost_tokens_and_infra():
    cost = compute_cost(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        local_compute_seconds=1800,
        compute_cost_per_hour=2.0,
    )
    assert cost.input_cost_usd == pytest.approx(0.15)
    assert cost.output_cost_usd == pytest.approx(0.30)
    assert cost.infra_cost_usd == pytest.approx(1.0)
    assert cost.total_cost_usd == pytest.approx(1.45)
    override = compute_cost(
        provider="ollama",
        model="x",
        prompt_tokens=1_000_000,
        completion_tokens=0,
        input_price_per_1m=1.5,
    )
    assert override.total_cost_usd == pytest.approx(1.5)


def test_timeout_detection_walks_exception_chain():
    class ReadTimeout(Exception):
        pass

    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = ReadTimeout()
    assert is_timeout_error(TimeoutError())
    assert is_timeout_error(wrapped)
    assert not is_timeout_error(ValueError("nope"))


def _record(status="success", total=1.0, cold=False, retries=0, **extra):
    return {
        "status": status,
        "total_seconds": total,
        "retrieval_seconds": total / 2,
        "generation_seconds": total / 2,
        "llm_seconds": total / 2,
        "is_cold": cold,
        "retries": retries,
        "total_cost_usd": 0.01,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        **extra,
    }


def test_summary_separates_cold_and_computes_rates():
    records = [_record(total=30.0, cold=True)] + [_record(total=float(i)) for i in range(1, 9)]
    records += [_record("timeout", error_type="ReadTimeout"), _record("error", retries=2)]
    summary = summarize(records, warmup=0)
    assert summary["latency"]["cold"]["requests"] == 1
    assert summary["latency"]["cold"]["total_seconds"]["p50"] == 30.0
    assert summary["latency"]["warm"]["total_seconds"]["max"] == 8.0
    rel = summary["reliability"]
    assert rel["requests"] == 11
    assert rel["success_rate"] == pytest.approx(9 / 11)
    assert rel["timeout_rate"] == pytest.approx(1 / 11)
    assert rel["error_rate"] == pytest.approx(2 / 11)
    assert rel["retry_rate"] == pytest.approx(1 / 11)
    assert summary["cost"]["cost_per_query_usd"] == pytest.approx(0.01)
    cold, warm = split_cold_warm([_record(), _record(), _record()], warmup=2)
    assert (len(cold), len(warm)) == (2, 1)


class _Retriever:
    def search(self, question, k, doc_id=None):
        return [
            SimpleNamespace(
                chunk={
                    "chunk_id": "c1",
                    "doc_id": "AAPL_10-K_2025",
                    "type": "text",
                    "section_title": "MD&A",
                    "element_orders": [1],
                    "text": "Net sales were $100.",
                },
                score=0.9,
            )
        ]


class _LLM:
    provider = "ollama"
    model = "fake"
    temperature = 0.0
    max_tokens = 10

    def __init__(self, failures=0, exc=RuntimeError("boom")):
        self.calls = 0
        self.failures = failures
        self.exc = exc

    def generate(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        content = json.dumps(
            {
                "answer": "$100",
                "explanation": "stated",
                "citations": ["S1"],
                "calculation": None,
                "insufficient_context": False,
            }
        )
        return LLMResponse(
            content=content,
            model="fake",
            prompt_tokens=50,
            completion_tokens=5,
            latency_seconds=0.01,
        )


def _generator(llm):
    return AnswerGenerator(retriever=_Retriever(), llm_client=llm, system_prompt="sys")


def test_tracker_records_components_and_writes_jsonl(tmp_path):
    tracker = QueryTracker(JsonlMetricsWriter(tmp_path), cold_requests=1)
    _, first = tracker.run(_generator(_LLM()), "Q?", doc_id="AAPL_10-K_2025")
    _, second = tracker.run(_generator(_LLM()), "Q?", doc_id="AAPL_10-K_2025")
    assert first.status == "success" and first.is_cold and not second.is_cold
    assert first.retrieval_seconds >= 0 and first.generation_seconds >= 0
    assert first.total_seconds >= first.retrieval_seconds + first.generation_seconds
    assert (first.prompt_tokens, first.completion_tokens) == (50, 5)
    lines = next(tmp_path.glob("queries-*.jsonl")).read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["status"] == "success"


def test_retries_are_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(settings, "LLM_RETRY_BACKOFF_SECONDS", 0.0)
    tracker = QueryTracker(JsonlMetricsWriter(tmp_path), cold_requests=0)
    _, record = tracker.run(_generator(_LLM(failures=2)), "Q?", doc_id=None)
    assert record.status == "success" and record.llm_attempts == 3 and record.retries == 2


def test_exhausted_retries_and_timeouts_are_classified(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(settings, "LLM_RETRY_BACKOFF_SECONDS", 0.0)
    tracker = QueryTracker(JsonlMetricsWriter(tmp_path), cold_requests=0)
    with pytest.raises(TimeoutError) as info:
        tracker.run(_generator(_LLM(failures=5, exc=TimeoutError("slow"))), "Q?", doc_id=None)
    record = info.value.query_record
    assert record.status == "timeout" and record.error_stage == "generation"
    assert record.llm_attempts == 2 and record.retries == 1
    with pytest.raises(RuntimeError) as info:
        tracker.run(_generator(_LLM(failures=5)), "Q?", doc_id=None)
    assert info.value.query_record.status == "error"
