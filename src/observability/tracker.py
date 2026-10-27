"""Instrumented execution of the retrieve -> generate pipeline.

Used by the API (RAGService) and the benchmark so production and evaluation numbers share one
code path.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .pricing import compute_cost
from .records import (
    JsonlMetricsWriter,
    QueryRecord,
    get_request_id,
    is_timeout_error,
    set_last_record,
    utc_now_iso,
)


class QueryTracker:
    def __init__(self, writer: JsonlMetricsWriter | None = None, cold_requests: int | None = None):
        from config.settings import settings

        self.writer = writer
        self.cold_requests = (
            settings.METRICS_COLD_REQUESTS if cold_requests is None else cold_requests
        )
        self._seq = 0
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> QueryTracker:
        from config.settings import ROOT, settings

        writer = None
        if settings.METRICS_ENABLED:
            directory = settings.METRICS_DIR
            from pathlib import Path

            path = Path(directory)
            writer = JsonlMetricsWriter(path if path.is_absolute() else ROOT / path)
        return cls(writer)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def run(
        self,
        generator: Any,
        question: str,
        *,
        doc_id: str | None,
        context_k: int | None = None,
        request_id: str | None = None,
    ) -> tuple[Any, QueryRecord]:
        """Run the pipeline and return (GenerationResult, record).

        On failure the record is persisted, attached as `error.query_record`, and the error
        is re-raised so callers keep their existing error handling.
        """
        from config.settings import settings

        seq = self._next_seq()
        record = QueryRecord(
            timestamp=utc_now_iso(),
            request_id=request_id or get_request_id(),
            doc_id=doc_id,
            status="success",
            process_seq=seq,
            provider=generator.llm_client.provider,
            model=generator.llm_client.model,
        )
        start = time.perf_counter()
        stage = "retrieval"
        try:
            if getattr(generator, "_retriever", True) is None:
                init_start = time.perf_counter()
                generator.retriever
                record.retriever_init_seconds = time.perf_counter() - init_start
            record.is_cold = seq <= self.cold_requests or record.retriever_init_seconds > 0

            retrieval_start = time.perf_counter()
            chunks = generator.retrieve(question, doc_id=doc_id, context_k=context_k)
            record.retrieval_seconds = time.perf_counter() - retrieval_start

            stage = "generation"
            generation_start = time.perf_counter()
            result = generator.generate(question, chunks, doc_id=doc_id)
            record.generation_seconds = time.perf_counter() - generation_start
        except Exception as error:
            record.total_seconds = time.perf_counter() - start
            record.status = "timeout" if is_timeout_error(error) else "error"
            record.error_type = type(error).__name__
            record.error_stage = stage
            record.error_message = str(error)[:500]
            record.llm_attempts = getattr(error, "llm_attempts", 0)
            record.retries = max(record.llm_attempts - 1, 0)
            self._finish(record)
            error.query_record = record  # type: ignore[attr-defined]
            raise

        record.total_seconds = time.perf_counter() - start
        record.llm_seconds = result.latency_seconds
        record.llm_attempts = result.llm_attempts
        record.retries = max(result.llm_attempts - 1, 0)
        record.prompt_tokens = result.prompt_tokens
        record.completion_tokens = result.completion_tokens
        record.context_k = result.context_k
        record.validation_passed = result.answer.is_valid
        record.insufficient_context = result.answer.insufficient_context

        local_seconds = 0.0
        if "local" in (settings.effective_embedding_mode, settings.effective_reranker_mode):
            local_seconds += record.retrieval_seconds
        if record.provider == "ollama":
            local_seconds += record.llm_seconds
        cost = compute_cost(
            provider=record.provider or "",
            model=record.model or "",
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            local_compute_seconds=local_seconds,
            input_price_per_1m=settings.LLM_INPUT_PRICE_PER_1M,
            output_price_per_1m=settings.LLM_OUTPUT_PRICE_PER_1M,
            compute_cost_per_hour=settings.LOCAL_COMPUTE_COST_PER_HOUR,
        )
        record.input_cost_usd = cost.input_cost_usd
        record.output_cost_usd = cost.output_cost_usd
        record.infra_cost_usd = cost.infra_cost_usd
        record.total_cost_usd = cost.total_cost_usd
        self._finish(record)
        return result, record

    def _finish(self, record: QueryRecord) -> None:
        set_last_record(record)
        if self.writer is not None:
            self.writer.write(record)
