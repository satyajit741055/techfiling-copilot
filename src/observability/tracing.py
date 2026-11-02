"""Context-local hook for debug trace events.

A traced chat turn installs a sink with ``tracing(sink)`` around its graph run; any code
running inside it (graph nodes, retrieval threads, the generator) can then call
``trace_event``. Context variables follow asyncio tasks and ``asyncio.to_thread``. Outside
a traced turn ``trace_event`` is a no-op, so the calls cost nothing in normal operation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

TraceSink = Callable[..., None]

_SINK: ContextVar[TraceSink | None] = ContextVar("trace_sink", default=None)


def tracing_enabled() -> bool:
    return _SINK.get() is not None


def trace_event(kind: str, summary: str, **data: Any) -> None:
    """Add ``kind`` with a one-line ``summary`` and JSON-able ``data`` to the active trace."""
    sink = _SINK.get()
    if sink is not None:
        sink(kind, summary, **data)


def trace_retrieval(
    label: str,
    chunks: Sequence[Mapping[str, Any]],
    scores: Sequence[float | None] | None = None,
    seconds: float | None = None,
    **data: Any,
) -> None:
    """One ``retrieval`` event: every hit with its filing, section, table and score."""
    if not tracing_enabled():
        return
    score_list = list(scores) if scores is not None else [None] * len(chunks)
    hits = [
        {
            "rank": rank,
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "ticker": chunk.get("ticker"),
            "fiscal_year": chunk.get("fiscal_year"),
            "type": chunk.get("type"),
            "section": chunk.get("section_title") or chunk.get("section"),
            "table_id": chunk.get("table_id"),
            "score": round(float(score), 4) if score is not None else None,
            "chars": len(str(chunk.get("text", ""))),
        }
        for rank, (chunk, score) in enumerate(zip(chunks, score_list), start=1)
    ]
    top = ", ".join(
        f"{hit['chunk_id']}" + (f" ({hit['score']:.3f})" if hit["score"] is not None else "")
        for hit in hits[:3]
    )
    took = f" {seconds:.2f}s" if seconds is not None else ""
    trace_event(
        "retrieval",
        f"retrieval {label}{took} -> {len(hits)} hits{': ' + top if top else ''}",
        seconds=round(seconds, 3) if seconds is not None else None,
        hits=hits,
        **data,
    )


@contextmanager
def tracing(sink: TraceSink | None) -> Iterator[None]:
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)
