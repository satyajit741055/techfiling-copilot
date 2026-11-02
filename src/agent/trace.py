"""Opt-in call-by-call trace of one /chat turn (AGENT_TRACE=all|errors), for debugging.

``TurnTrace`` is a LangChain callback (passed in the graph run config, like the usage
callback), so it sees every agent LLM call with its node, full prompt, raw output, error,
timing and tokens without touching the nodes. It is also the ``observability.tracing``
sink for the turn, so retrieval, the single-filing generator and code-side decisions
(grounding, context selection, repairs) add their own events. ``AgentService`` feeds it
the request, the streamed step/tool events, each node's state update, failures and what
was sent back. The JSON file is written when the turn ends, including failed, timed-out
and cancelled turns; its name and every log line carry the turn and request ids.

Traces contain prompts, filing text and answers: keep them out of version control.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from .events import completed_detail

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 200


def _default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    return repr(value)


def _plain(value: Any) -> Any:
    """A JSON-safe copy, taken now so later state changes cannot alter the record."""
    return json.loads(json.dumps(value, default=_default, ensure_ascii=False))


def _preview(text: Any) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "..."


def _message(message: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": getattr(message, "type", type(message).__name__),
        "content": getattr(message, "content", message),
    }
    for key in ("tool_calls", "tool_call_id", "name"):
        if value := getattr(message, key, None):
            data[key] = value
    return data


class TurnTrace(BaseCallbackHandler):
    """Collects one turn's events; ``write`` saves them as JSON in ``directory``.

    With ``log_events`` every event is also logged as one line (AGENT_TRACE=all);
    otherwise only warnings and errors are logged (AGENT_TRACE=errors).
    """

    run_inline = True

    def __init__(
        self,
        directory: Path,
        *,
        turn_id: str | None,
        thread_id: str,
        request_id: str | None = None,
        question: str = "",
        model: str = "",
        log_events: bool = True,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.turn_id = turn_id or ""
        self.request_id = request_id or ""
        self.log_events = log_events
        self.started_at = datetime.now()
        self._started = time.perf_counter()
        self.header = {
            "turn_id": turn_id,
            "thread_id": thread_id,
            "request_id": request_id,
            "question": question,
            "model": model,
            "started_at": self.started_at.isoformat(timespec="seconds"),
        }
        self.events: list[dict[str, Any]] = []
        self.llm_calls = 0
        self._calls: dict[UUID, tuple[int, str | None, float]] = {}
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        stamp = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        name = self.turn_id[:8] or f"req-{self.request_id[:8] or 'none'}"
        return self.directory / f"{stamp}_{name}.json"

    def add(self, kind: str, summary: str, level: int = logging.INFO, **data: Any) -> None:
        """Record one event (also the ``observability.tracing`` sink for other layers)."""
        plain = _plain(data)
        with self._lock:
            elapsed = round(time.perf_counter() - self._started, 3)
            self.events.append({"seq": len(self.events) + 1, "t": elapsed, "kind": kind, **plain})
        if self.log_events or level >= logging.WARNING:
            logger.log(level, f"trace +{elapsed:.1f}s {summary}")

    # -- LLM calls (LangChain callbacks) --------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        prompt = [_message(message) for message in (messages[0] if messages else [])]
        with self._lock:
            self.llm_calls += 1
            call = self.llm_calls
            self._calls[run_id] = (call, node, time.perf_counter())
        chars = sum(len(str(message["content"])) for message in prompt)
        self.add(
            "llm_start",
            f"LLM #{call} start  node={node} messages={len(prompt)} prompt_chars={chars:,}",
            call=call,
            node=node,
            params=invocation_params or {},
            messages=prompt,
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        call, node, started = self._calls.pop(run_id, (0, None, time.perf_counter()))
        generations = getattr(response, "generations", None) or [[]]
        generation = generations[0][0] if generations and generations[0] else None
        message = getattr(generation, "message", None)
        content = message.content if message is not None else getattr(generation, "text", "")
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        usage = dict(getattr(message, "usage_metadata", None) or {})
        seconds = round(time.perf_counter() - started, 3)
        tools = f" tool_calls={[item['name'] for item in tool_calls]}" if tool_calls else ""
        self.add(
            "llm_end",
            f"LLM #{call} done   node={node} {seconds:.1f}s "
            f"tokens={usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')}"
            f"{tools} output={_preview(content)!r}",
            call=call,
            node=node,
            seconds=seconds,
            usage=usage,
            output=content,
            tool_calls=tool_calls,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        call, node, started = self._calls.pop(run_id, (0, None, time.perf_counter()))
        self.add(
            "llm_error",
            f"LLM #{call} FAILED node={node}: {type(error).__name__}: {_preview(error)}",
            logging.WARNING,
            call=call,
            node=node,
            seconds=round(time.perf_counter() - started, 3),
            error=f"{type(error).__name__}: {error}",
        )

    # -- Graph stream (fed by AgentService) -----------------------------------------------

    def stream_item(self, mode: str, payload: Any) -> None:
        if mode == "custom" and isinstance(payload, Mapping):
            if payload.get("type") == "tool":
                # The retrieval event (from the tool itself) already logged the hits.
                error = f" ERROR={payload['error']}" if payload.get("error") else ""
                self.add(
                    "tool",
                    f"tool   {payload.get('task_id')} {payload.get('tool')} -> "
                    f"{payload.get('n_results')} results{error}",
                    logging.WARNING if error else logging.DEBUG,
                    **dict(payload),
                )
            elif payload.get("type") == "step":
                detail = f" ({payload['detail']})" if payload.get("detail") else ""
                self.add(
                    "step",
                    f"step   {payload.get('node')}: {payload.get('label')}{detail}",
                    node=payload.get("node"),
                    label=payload.get("label"),
                    detail=payload.get("detail"),
                )
            return
        for node, update in (payload or {}).items():
            if node == "__interrupt__":
                value = getattr(update[0], "value", update) if update else update
                self.add("clarification", "clarification requested", value=value)
                continue
            detail = completed_detail(node, update)
            keys = sorted(update) if isinstance(update, Mapping) else []
            self.add(
                "node",
                f"node   {node} done{f' ({detail})' if detail else ''} updated={keys}",
                node=node,
                update=update,
            )

    def failure(self, error: BaseException) -> None:
        self.add(
            "error",
            f"turn FAILED: {type(error).__name__}: {_preview(error)}",
            logging.ERROR,
            error=f"{type(error).__name__}: {error}",
            traceback="".join(traceback.format_exception(error)),
        )

    def write(self, summary: Mapping[str, Any]) -> Path | None:
        """Save the trace; never raises (a trace must not break the turn)."""
        self.add("end", f"turn end status={summary.get('status')}", status=summary.get("status"))
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {**self.header, "summary": _plain(dict(summary)), "events": self.events},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as error:
            logger.warning(f"Could not write agent trace {self.path}: {error!r}")
            return None
        logger.info(f"trace written to {self.path}")
        return self.path
