"""LangGraph assembly for the filing-research agent."""

from __future__ import annotations

from collections.abc import Hashable
from functools import partial
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel

from . import schemas
from .nodes import (
    NodeDependencies,
    analyze_node,
    check_coverage_node,
    clarify_node,
    execute_node,
    finalize_node,
    lookup_facts_node,
    onboard_node,
    plan_node,
    refuse_node,
    resolve_scope_node,
    route_after_facts,
    simple_rag_node,
    synthesize_node,
    validate_node,
)
from .state import AgentState, ResultsUpdate, WorkerState

RECURSION_LIMIT = 40
_SCOPE_ROUTES = {
    "refuse": "refuse",
    "onboard": "onboard",
    "clarify": "clarify",
    "simple_rag": "simple_rag",
    "research": "plan",
}


def route_after_scope(state: AgentState) -> str:
    scope = state.get("scope")
    if scope is None:
        raise ValueError("resolve_scope must set scope before routing")
    return _SCOPE_ROUTES[scope.route]


def route_after_plan(state: AgentState) -> list[Send] | str:
    """Fan out one ``execute`` worker per task; with no tasks go straight to coverage."""
    tasks = state.get("tasks") or []
    if not tasks:
        return "check_coverage"
    turn_id = str(state.get("turn_id") or "")
    return [
        Send("execute", WorkerState(task=task, question=state["question"], turn_id=turn_id))
        for task in tasks
    ]


def route_after_coverage(state: AgentState) -> str:
    """Re-plan while cells are ``missing``; ``check_coverage`` stops marking them when exhausted."""
    coverage = state.get("coverage") or []
    return "plan" if any(cell.status == "missing" for cell in coverage) else "synthesize"


def route_after_validate(state: AgentState) -> str:
    """One repair pass: back to ``synthesize`` after the first failed validation only."""
    validation = state.get("validation")
    if validation is not None and not validation.is_valid and state.get("repairs", 0) == 1:
        return "synthesize"
    return "finalize"


def checkpoint_serializer() -> JsonPlusSerializer:
    """Serializer that only revives the agent's own state models from checkpoints."""
    models = [
        (value.__module__, value.__name__)
        for value in vars(schemas).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == schemas.__name__
    ]
    return JsonPlusSerializer(
        allowed_msgpack_modules=[*models, (ResultsUpdate.__module__, ResultsUpdate.__name__)]
    )


def build_graph(
    deps: NodeDependencies, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the research graph. A checkpointer is required for clarification interrupts.

    Defaults to an in-memory checkpointer with the agent-model allow-list.
    """
    graph = StateGraph(AgentState)
    graph.add_node("analyze", partial(analyze_node, deps=deps))
    graph.add_node("resolve_scope", partial(resolve_scope_node, deps=deps))
    graph.add_node("refuse", partial(refuse_node, deps=deps))
    graph.add_node("onboard", partial(onboard_node, deps=deps))
    graph.add_node("clarify", partial(clarify_node, deps=deps))
    graph.add_node("simple_rag", partial(simple_rag_node, deps=deps))
    graph.add_node("plan", partial(plan_node, deps=deps))
    graph.add_node("execute", partial(execute_node, deps=deps), input_schema=WorkerState)
    graph.add_node("check_coverage", partial(check_coverage_node, deps=deps))
    graph.add_node("synthesize", partial(synthesize_node, deps=deps))
    graph.add_node("validate", partial(validate_node, deps=deps))
    graph.add_node("finalize", partial(finalize_node, deps=deps))

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "resolve_scope")
    if deps.facts is None:
        graph.add_conditional_edges(
            "resolve_scope", route_after_scope, list(_SCOPE_ROUTES.values())
        )
    else:
        # Filing questions look up reported figures first (facts store configured only).
        graph.add_node("lookup_facts", partial(lookup_facts_node, deps=deps))
        targets: dict[Hashable, str] = {target: target for target in _SCOPE_ROUTES.values()}
        targets.update(simple_rag="lookup_facts", plan="lookup_facts")
        graph.add_conditional_edges("resolve_scope", route_after_scope, targets)
        graph.add_conditional_edges(
            "lookup_facts", route_after_facts, ["synthesize", "simple_rag", "plan"]
        )
    graph.add_edge("clarify", "resolve_scope")
    for terminal in ("refuse", "onboard", "simple_rag", "finalize"):
        graph.add_edge(terminal, END)
    graph.add_conditional_edges("plan", route_after_plan, ["execute", "check_coverage"])
    graph.add_edge("execute", "check_coverage")
    graph.add_conditional_edges("check_coverage", route_after_coverage, ["plan", "synthesize"])
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges("validate", route_after_validate, ["synthesize", "finalize"])
    if checkpointer is None:
        checkpointer = MemorySaver(serde=checkpoint_serializer())
    return graph.compile(checkpointer=checkpointer)


def recursion_limit(max_replans: int = 2) -> int:
    """40 steps covers the default 2 re-plans; each extra re-plan adds plan → execute → coverage."""
    return RECURSION_LIMIT + 3 * max(0, max_replans - 2)


def graph_config(thread_id: str, max_replans: int = 2) -> RunnableConfig:
    """Run config for one conversation thread (same ``thread_id`` to resume)."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit(max_replans),
    }
