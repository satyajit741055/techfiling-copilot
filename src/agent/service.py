"""AgentService: run or resume one /chat turn, stream its events, record and persist it.

The graph runs in its own task and reports through a queue, so the stream can enforce the
turn timeout, notice a client disconnect between events and cancel the run. History and
metrics are written after the turn ends, outside the graph, so nodes stay pure and
resumable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from config.logging import bind_log_context, reset_log_context
from config.settings import ROOT, Settings, settings
from observability import AgentRecord, JsonlMetricsWriter, compute_cost
from observability.records import is_timeout_error, utc_now_iso
from observability.tracing import tracing
from persistence import HistoryStore, NullHistoryStore, TurnRecord

from .catalog import FilingCatalog
from .events import NODE_LABELS, ChatEvent, UsageCallback, clarification_payload, completed_detail
from .evidence import EvidenceRegistry, TurnEvidence
from .graph import build_graph, graph_config
from .llm import AgentLLM, get_chat_model
from .nodes import NodeDependencies
from .schemas import ChatAnswer, ResolvedScope, UIFilters, Usage
from .tools import AgentTools
from .trace import TurnTrace

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0
_OPTION_PREFIX = "opt_"
ERROR_DETAILS = {
    "clarification_expired": "This clarification has expired. Please send your question again.",
    "invalid_choice": "That option is not available. Choose one of the listed options.",
    "thread_busy": "This conversation is still answering the previous message.",
    "timeout": "The research took too long. Try a narrower question.",
    "agent_error": "Unable to answer right now. Please try again shortly.",
}


@dataclass(frozen=True)
class ChatTurn:
    """One /chat request: a new ``message``, or a clarification answer (option or text)."""

    thread_id: UUID
    owner: str
    message: str | None = None
    option_id: str | None = None
    text: str | None = None
    filters: UIFilters | None = None
    history_user: UUID | None = None
    request_id: str | None = None


@dataclass
class _Run:
    started: float
    last_mark: float
    request_id: str | None = None
    turn_id: str | None = None
    question: str = ""
    graph_ran: bool = False
    final: ChatAnswer | None = None
    scope: ResolvedScope | None = None
    clarification: dict[str, Any] | None = None
    status: str = "success"
    error_code: str | None = None
    error_type: str | None = None
    node_seconds: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    tool_events: int = 0
    replans: int = 0
    repairs: int = 0
    usage: Usage = field(default_factory=Usage)
    totals_done: bool = False
    trace: TurnTrace | None = None

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def fail(self, code: str, status: str = "error", error: BaseException | None = None) -> None:
        self.status, self.error_code = status, code
        self.error_type = type(error).__name__ if error is not None else None


_END = object()


@dataclass(frozen=True)
class _Failure:
    error: Exception


class AgentService:
    def __init__(
        self,
        graph: Any,
        catalog: FilingCatalog,
        *,
        evidence: TurnEvidence | None = None,
        history: HistoryStore | None = None,
        writer: JsonlMetricsWriter | None = None,
        timeout_seconds: float = 180.0,
        max_replans: int = 2,
        provider: str = "",
        model: str = "",
        input_price_per_1m: float | None = None,
        output_price_per_1m: float | None = None,
        poll_seconds: float = POLL_SECONDS,
        trace_dir: Path | None = None,
        trace_mode: str = "all",
    ) -> None:
        self.graph = graph
        self.catalog = catalog
        self.evidence = evidence
        self.history: HistoryStore = history or NullHistoryStore()
        self.writer = writer
        self.timeout_seconds = timeout_seconds
        self.max_replans = max_replans
        self.provider, self.model = provider, model
        self._prices = (input_price_per_1m, output_price_per_1m)
        self._poll_seconds = poll_seconds
        self.trace_dir, self.trace_mode = trace_dir, trace_mode
        self._active: set[str] = set()

    @classmethod
    def from_generator(
        cls,
        generator: Any,
        filings: Iterable[Mapping[str, Any]],
        *,
        history: HistoryStore | None = None,
        writer: JsonlMetricsWriter | None = None,
        config: Settings | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> AgentService:
        """Build the agent on the generator's already-loaded retriever and reranker."""
        from retriever.qdrant_retriever import QdrantRetrieverWithReranker

        config = config or settings
        retriever = generator.retriever
        if not isinstance(retriever, QdrantRetrieverWithReranker):
            raise RuntimeError("Research chat needs the Qdrant retriever (VECTOR_STORE=qdrant)")
        catalog = FilingCatalog(filings)
        tools = AgentTools(catalog, retriever, retriever.retriever.store, EvidenceRegistry())
        deps = NodeDependencies(
            catalog=catalog,
            llm=AgentLLM(get_chat_model(config=config), config.AGENT_LLM_CONCURRENCY),
            tools=tools,
            generator=generator,
            max_tasks=config.AGENT_MAX_TASKS,
            max_replans=config.AGENT_MAX_REPLANS,
            max_tool_steps=config.AGENT_MAX_TOOL_STEPS,
            task_k=config.AGENT_TASK_K,
            max_context_chunks=config.AGENT_MAX_CONTEXT_CHUNKS,
            min_evidence_score=config.AGENT_MIN_EVIDENCE_SCORE,
            retrieval_concurrency=config.AGENT_RETRIEVAL_CONCURRENCY,
        )
        return cls(
            build_graph(deps, checkpointer),
            catalog,
            evidence=deps.evidence,
            history=history,
            writer=writer,
            timeout_seconds=config.AGENT_TIMEOUT_SECONDS,
            max_replans=config.AGENT_MAX_REPLANS,
            provider=config.effective_agent_provider,
            model=config.effective_agent_model,
            input_price_per_1m=config.LLM_INPUT_PRICE_PER_1M,
            output_price_per_1m=config.LLM_OUTPUT_PRICE_PER_1M,
            trace_dir=ROOT / config.AGENT_TRACE_DIR if config.AGENT_TRACE != "off" else None,
            trace_mode=config.AGENT_TRACE,
        )

    async def stream(
        self,
        turn: ChatTurn,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Events for one turn: thread, step/tool..., answer | clarification | error, done.

        Never raises: failures become an ``error`` event without internals. If the client
        disconnects (or the consumer stops iterating) the graph run is cancelled.
        """
        now = time.perf_counter()
        run = _Run(started=now, last_mark=now, request_id=turn.request_id)
        key = f"{turn.owner}:{turn.thread_id}"
        config = graph_config(key, self.max_replans)
        usage = UsageCallback()
        callbacks: list[Any] = [usage]
        config["callbacks"] = callbacks
        # Every log line of this turn (graph nodes and retrieval threads included) carries
        # the thread and turn ids next to the request id bound by the middleware.
        log_tokens = bind_log_context(thread_id=str(turn.thread_id))
        finished = locked = False
        try:
            if key in self._active:
                run.fail("thread_busy")
                graph_input = None
            else:
                self._active.add(key)
                locked = True
                graph_input = await self._prepare(turn, run, config)
            log_tokens += bind_log_context(turn_id=run.turn_id)
            run.trace = self._start_trace(turn, run)
            if run.trace is not None:
                callbacks.append(run.trace)
            yield ChatEvent("thread", {"thread_id": str(turn.thread_id), "turn_id": run.turn_id})
            if graph_input is not None:
                run.graph_ran = True
                # aclosing: if our consumer stops, the inner generator (and the graph task)
                # is closed now rather than whenever it is garbage collected.
                async with contextlib.aclosing(
                    self._run_graph(graph_input, config, run, is_disconnected)
                ) as events:
                    async for event in events:
                        yield event
            if run.status == "cancelled":
                return
            self._totals(run, usage)
            if run.final is not None and run.error_code is None:
                run.final = run.final.model_copy(update={"usage": run.usage})
                yield self._sent(run, ChatEvent("answer", run.final.model_dump(mode="json")))
            elif run.clarification is None and run.error_code is None:
                run.fail("agent_error")
            if run.error_code is not None:
                yield self._sent(
                    run,
                    ChatEvent(
                        "error",
                        {
                            "code": run.error_code,
                            "detail": ERROR_DETAILS.get(
                                run.error_code, ERROR_DETAILS["agent_error"]
                            ),
                            "correlation_id": turn.request_id,
                        },
                    ),
                )
            if run.graph_ran:
                await self._persist(turn, run)
            finished = True
            self._record(turn, run)
            done = ChatEvent(
                "done",
                {
                    "latency_ms": run.elapsed_ms(),
                    "prompt_tokens": run.usage.prompt_tokens,
                    "completion_tokens": run.usage.completion_tokens,
                    "cost_usd": run.usage.cost_usd,
                    "llm_calls": run.usage.llm_calls,
                    "tool_calls": run.usage.tool_calls,
                },
            )
            self._sent(run, done)
            self._finish(turn, run)
            yield done
        finally:
            if locked:
                self._active.discard(key)
            if self.evidence is not None and run.turn_id:
                self.evidence.release(run.turn_id)
            if not finished:
                if run.status == "success":
                    run.status = "cancelled"
                self._totals(run, usage)
                self._record(turn, run)
                self._finish(turn, run)
            reset_log_context(log_tokens)

    async def _prepare(self, turn: ChatTurn, run: _Run, config: Any) -> Any:
        """Graph input for a new message, or the resume command; None on a resume error."""
        if turn.message is not None:
            run.turn_id, run.question = uuid.uuid4().hex, turn.message
            return {"question": turn.message, "turn_id": run.turn_id, "ui_filters": turn.filters}
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values or {}
        pending = snapshot.interrupts
        if not pending:
            # MemorySaver is lost on restart, and a newer message replaces the pause.
            run.fail("clarification_expired")
            return None
        run.turn_id, run.question = values.get("turn_id"), str(values.get("question") or "")
        choice: str | None = turn.text
        if turn.option_id is not None:
            options = list(pending[0].value.get("options") or [])
            index = turn.option_id.removeprefix(_OPTION_PREFIX)
            if not index.isdigit() or not 1 <= int(index) <= len(options):
                run.fail("invalid_choice")
                return None
            choice = str(options[int(index) - 1])
        return Command(resume=choice or "")

    async def _run_graph(
        self,
        graph_input: Any,
        config: Any,
        run: _Run,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> AsyncGenerator[ChatEvent, None]:
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def pump() -> None:
            sink = run.trace.add if run.trace is not None else None
            try:
                with tracing(sink):
                    async for item in self.graph.astream(
                        graph_input, config, stream_mode=["updates", "custom"]
                    ):
                        queue.put_nowait(item)
            except Exception as error:
                queue.put_nowait(_Failure(error))
            else:
                queue.put_nowait(_END)

        task = asyncio.create_task(pump())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds
        next_check = loop.time() + self._poll_seconds
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    run.fail("timeout", status="timeout")
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), min(self._poll_seconds, remaining))
                except TimeoutError:
                    item = None
                if is_disconnected is not None and loop.time() >= next_check:
                    next_check = loop.time() + self._poll_seconds
                    if await is_disconnected():
                        run.status = "cancelled"
                        return
                if item is None:
                    continue
                if item is _END:
                    return
                if isinstance(item, _Failure):
                    timed_out = is_timeout_error(item.error)
                    logger.error(
                        f"Chat turn failed [correlation_id={run.request_id}]: "
                        f"{type(item.error).__name__}: {item.error}",
                        exc_info=item.error,
                    )
                    if run.trace is not None:
                        run.trace.failure(item.error)
                    run.fail(
                        "timeout" if timed_out else "agent_error",
                        status="timeout" if timed_out else "error",
                        error=item.error,
                    )
                    return
                mode, payload = item
                if run.trace is not None:
                    run.trace.stream_item(mode, payload)
                for event in self._events(mode, payload, run):
                    yield event
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def _events(self, mode: str, payload: Any, run: _Run) -> list[ChatEvent]:
        if mode == "custom":
            if not isinstance(payload, Mapping):
                return []
            data = {key: value for key, value in payload.items() if key != "type"}
            if payload.get("type") == "tool":
                run.tool_events += 1
                return [ChatEvent("tool", data)]
            if payload.get("type") == "step":
                return [
                    ChatEvent("step", {**data, "status": "started", "elapsed_ms": run.elapsed_ms()})
                ]
            return []
        events: list[ChatEvent] = []
        for node, update in (payload or {}).items():
            if node == "__interrupt__":
                run.clarification = clarification_payload(update[0].value)
                events.append(ChatEvent("clarification", run.clarification))
                continue
            now = time.perf_counter()
            run.node_seconds[node] += now - run.last_mark
            run.last_mark = now
            if isinstance(update, Mapping):
                run.final = update.get("final") or run.final
                run.scope = update.get("scope") or run.scope
                run.replans = update.get("replans", run.replans)
                run.repairs = update.get("repairs", run.repairs)
            events.append(
                ChatEvent(
                    "step",
                    {
                        "node": node,
                        "status": "completed",
                        "label": NODE_LABELS.get(node, node),
                        "detail": completed_detail(node, update),
                        "elapsed_ms": run.elapsed_ms(),
                    },
                )
            )
        return events

    def _totals(self, run: _Run, usage: UsageCallback) -> None:
        """LLM usage from the callback plus the single-filing generator's own counts.

        Runs once per turn: afterwards ``run.final.usage`` already holds the totals, so a
        second pass (stream closed after the answer) would count the callback twice.
        """
        if run.totals_done:
            return
        run.totals_done = True
        own = run.final.usage if run.final is not None else Usage()
        prompt = usage.prompt_tokens + own.prompt_tokens
        completion = usage.completion_tokens + own.completion_tokens
        cost = compute_cost(
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            input_price_per_1m=self._prices[0],
            output_price_per_1m=self._prices[1],
        ).total_cost_usd
        run.usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            llm_calls=usage.llm_calls + own.llm_calls,
            tool_calls=own.tool_calls if run.final is not None else run.tool_events,
            cost_usd=round(cost, 8),
        )

    async def _persist(self, turn: ChatTurn, run: _Run) -> None:
        if turn.history_user is None or not self.history.enabled or not run.question:
            self._trace(
                run, "history", "history not saved (store disabled or no owner)", saved=False
            )
            return
        final = run.final if run.error_code is None else None
        if run.error_code is not None:
            status = "error"
        elif run.clarification is not None:
            status = "clarification"
        else:
            status = "refused" if final is not None and final.route == "refuse" else "answered"
        try:
            record = TurnRecord(
                thread_id=turn.thread_id,
                user_id=turn.history_user,
                turn_id=run.turn_id,
                question=run.question,
                status=status,
                route=final.route if final else (run.scope.route if run.scope else None),
                answer_markdown=final.answer if final else None,
                answer=(
                    final.model_dump(mode="json", exclude={"trajectory"})
                    if final
                    else ({"clarification": run.clarification} if run.clarification else None)
                ),
                citations=[source.model_dump(mode="json") for source in final.sources]
                if final
                else None,
                assumptions=final.assumptions if final else None,
                coverage=[cell.model_dump(mode="json") for cell in final.coverage]
                if final
                else None,
                scope=run.scope.model_dump(mode="json") if run.scope else None,
                is_valid=final.is_valid if final else None,
                validation_errors=final.validation_errors if final else None,
                trajectory=[step.model_dump(mode="json") for step in final.trajectory]
                if final
                else None,
                model=self.model or None,
                prompt_tokens=run.usage.prompt_tokens,
                completion_tokens=run.usage.completion_tokens,
                latency_ms=run.elapsed_ms(),
                cost_usd=Decimal(str(run.usage.cost_usd)),
            )
        except Exception as error:
            logger.warning(f"History record for turn {run.turn_id} is invalid ({error!r})")
            self._trace(run, "history", f"history record invalid: {error!r}", saved=False)
            return
        saved = await asyncio.to_thread(self.history.save_turn, record)
        self._trace(run, "history", f"history saved={saved} status={status}", saved=saved)

    @staticmethod
    def _outcome(run: _Run) -> str | None:
        if run.clarification is not None and run.error_code is None:
            return "clarification"
        if run.final is not None and run.error_code is None:
            return {"refuse": "refused", "onboard": "onboard"}.get(run.final.route, "answer")
        return None

    def _record(self, turn: ChatTurn, run: _Run) -> None:
        if self.writer is None:
            return
        final = run.final if run.error_code is None else None
        outcome = self._outcome(run)
        self._trace(run, "metrics", "metrics record written", outcome=outcome)
        self.writer.write(
            AgentRecord(
                timestamp=utc_now_iso(),
                request_id=turn.request_id,
                thread_id=str(turn.thread_id),
                turn_id=run.turn_id,
                status=run.status,
                outcome=outcome,
                route=final.route if final else (run.scope.route if run.scope else None),
                error_type=run.error_type,
                error_code=run.error_code,
                total_seconds=time.perf_counter() - run.started,
                node_seconds={node: round(value, 4) for node, value in run.node_seconds.items()},
                llm_calls=run.usage.llm_calls,
                tool_calls=run.usage.tool_calls,
                replans=run.replans,
                repairs=run.repairs,
                is_valid=final.is_valid if final else None,
                insufficient_context=(final.answer is None) if final else None,
                claims=len(final.claims) if final else 0,
                dropped_claims=len(final.dropped_claims) if final else 0,
                coverage=dict(Counter(cell.status for cell in final.coverage)) if final else {},
                provider=self.provider or None,
                model=self.model or None,
                prompt_tokens=run.usage.prompt_tokens,
                completion_tokens=run.usage.completion_tokens,
                total_cost_usd=run.usage.cost_usd,
            )
        )

    # -- Logging and debug trace ----------------------------------------------------------

    @staticmethod
    def _trace(run: _Run, kind: str, summary: str, **data: Any) -> None:
        if run.trace is not None:
            run.trace.add(kind, summary, **data)

    def _start_trace(self, turn: ChatTurn, run: _Run) -> TurnTrace | None:
        """Log the turn start; with AGENT_TRACE also open its trace (rejected turns too)."""
        kind = (
            "message"
            if turn.message is not None
            else ("resume option" if turn.option_id is not None else "resume text")
        )
        filters = turn.filters.model_dump(mode="json") if turn.filters else None
        rejected = f" rejected={run.error_code}" if run.error_code else ""
        summary = f"Chat turn started: {kind} filters={filters}{rejected}"
        logger.info(summary)
        if self.trace_dir is None:
            return None
        trace = TurnTrace(
            self.trace_dir,
            turn_id=run.turn_id,
            thread_id=str(turn.thread_id),
            request_id=turn.request_id,
            question=run.question,
            model=f"{self.provider}/{self.model}",
            log_events=self.trace_mode == "all",
        )
        trace.add(
            "request",
            summary,
            logging.DEBUG,  # already logged above
            request=kind,
            message=turn.message,
            option_id=turn.option_id,
            text=turn.text,
            filters=filters,
            owner=turn.owner,
            rejected=run.error_code,
        )
        return trace

    def _sent(self, run: _Run, event: ChatEvent) -> ChatEvent:
        """Trace what the client receives (the full answer is in the trace summary)."""
        data = event.data
        if event.event == "answer":
            data = {
                key: data.get(key) for key in ("route", "answer", "is_valid", "validation_errors")
            }
        self._trace(
            run, "sent", f"sent {event.event} {_preview(data)}", event=event.event, data=data
        )
        return event

    @staticmethod
    def _needs_attention(run: _Run) -> bool:
        """Failed, timed out, cancelled, rejected, invalid, empty or trimmed answers."""
        final = run.final
        if run.status != "success" or run.error_code is not None:
            return True
        return (
            final is not None
            and final.route not in {"refuse", "onboard"}
            and (not final.is_valid or final.answer is None or bool(final.dropped_claims))
        )

    def _finish(self, turn: ChatTurn, run: _Run) -> None:
        """One summary line per turn (always) and, by AGENT_TRACE mode, the trace file."""
        trace, run.trace = run.trace, None
        path = None
        if trace is not None and (self.trace_mode == "all" or self._needs_attention(run)):
            path = trace.write(
                {
                    "status": run.status,
                    "error_code": run.error_code,
                    "error_type": run.error_type,
                    "resumed": turn.message is None,
                    "total_seconds": round(time.perf_counter() - run.started, 3),
                    "node_seconds": {
                        node: round(value, 3) for node, value in run.node_seconds.items()
                    },
                    "replans": run.replans,
                    "repairs": run.repairs,
                    "usage": run.usage,
                    "scope": run.scope,
                    "clarification": run.clarification,
                    "final": run.final,
                }
            )
        final = run.final if run.error_code is None else None
        quality = (
            f" valid={final.is_valid} answered={final.answer is not None} "
            f"claims={len(final.claims)} dropped={len(final.dropped_claims)} "
            f"coverage={dict(Counter(cell.status for cell in final.coverage))}"
            if final is not None
            else ""
        )
        error = f" error={run.error_code}({run.error_type or '-'})" if run.error_code else ""
        logger.log(
            logging.INFO if run.status == "success" and not run.error_code else logging.WARNING,
            f"Chat turn finished: status={run.status} outcome={self._outcome(run)} "
            f"route={final.route if final else (run.scope.route if run.scope else None)} "
            f"{time.perf_counter() - run.started:.1f}s llm={run.usage.llm_calls} "
            f"tools={run.usage.tool_calls} tokens={run.usage.prompt_tokens}/"
            f"{run.usage.completion_tokens} replans={run.replans} repairs={run.repairs}"
            f"{quality}{error}" + (f" trace={path}" if path else ""),
        )


def _preview(data: Any, limit: int = 160) -> str:
    text = " ".join(str(data).split())
    return text if len(text) <= limit else text[:limit] + "..."
