"""SSE event model for /chat, node labels, and the per-turn LLM usage callback."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler

EventName = Literal["thread", "step", "tool", "clarification", "answer", "error", "done"]

# Label for a node's "completed" step; "started" labels come from the nodes themselves.
NODE_LABELS = {
    "analyze": "Understood the question",
    "resolve_scope": "Scope resolved",
    "refuse": "Outside the indexed filings",
    "onboard": "Overview",
    "clarify": "Clarification applied",
    "simple_rag": "Single-filing answer",
    "lookup_facts": "Reported figures checked",
    "plan": "Searches planned",
    "execute": "Search finished",
    "check_coverage": "Coverage checked",
    "synthesize": "Answer drafted",
    "validate": "Citations verified",
    "finalize": "Answer ready",
}


@dataclass(frozen=True)
class ChatEvent:
    event: EventName
    data: dict[str, Any]


def completed_detail(node: str, update: Any) -> str | None:
    """Short detail for a finished node, from its state update."""
    if not isinstance(update, Mapping):
        return None
    if node == "execute":
        results = update.get("results") or []
        return (
            "; ".join(
                f"{result.ticker}"
                + (f" FY{result.fiscal_year}" if result.fiscal_year is not None else "")
                + f": {result.status}"
                for result in results
            )
            or None
        )
    if node == "plan":
        return f"{len(update.get('tasks') or [])} searches"
    if node == "lookup_facts":
        facts = update.get("facts") or []
        return f"{len(facts)} figure(s)" if facts else "none used"
    if node == "validate" and (report := update.get("validation")) is not None:
        return "valid" if report.is_valid else f"{len(report.errors)} issue(s)"
    return None


def clarification_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    """The interrupt value from ``clarify`` as the client-facing clarification event."""
    options = [str(option) for option in request.get("options") or []]
    return {
        "question": str(request.get("question") or ""),
        "options": [
            {"id": f"opt_{index}", "label": label} for index, label in enumerate(options, start=1)
        ],
        "allow_free_text": bool(request.get("allow_free_text", True)),
    }


class UsageCallback(BaseCallbackHandler):
    """Counts LLM calls and tokens for one turn (passed in the graph run config).

    Unlike ``UsageMetadataCallbackHandler`` it does not need ``model_name`` in the response
    metadata, which Ollama does not set.
    """

    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _started(self) -> None:
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self._started()

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._started()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        prompt = completion = 0
        for generations in getattr(response, "generations", None) or []:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    prompt += int(usage.get("input_tokens", 0) or 0)
                    completion += int(usage.get("output_tokens", 0) or 0)
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
