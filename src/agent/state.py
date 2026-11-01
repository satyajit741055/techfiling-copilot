"""LangGraph state types and reducers for the research agent."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from .schemas import (
    Analysis,
    ChatAnswer,
    CoverageCell,
    GroundedAnswer,
    ResolvedScope,
    Task,
    TaskResult,
    TurnSummary,
    UIFilters,
    ValidationReport,
)


class ResultsUpdate(StrEnum):
    RESET = "__reset_agent_results__"


RESET = ResultsUpdate.RESET


def merge_or_reset(
    current: list[TaskResult] | None,
    update: list[TaskResult] | ResultsUpdate | None,
) -> list[TaskResult]:
    """Append parallel worker results, or clear them at the start of a new turn."""
    if update == RESET:
        return []
    return [*(current or []), *(update or [])]


class AgentState(TypedDict, total=False):
    turns: Annotated[list[TurnSummary], operator.add]
    turn_id: str
    question: str
    ui_filters: UIFilters | None
    analysis: Analysis | None
    scope: ResolvedScope | None
    clarifications_this_turn: int
    tasks: list[Task]
    results: Annotated[list[TaskResult], merge_or_reset]
    replans: int
    coverage: list[CoverageCell]
    draft: GroundedAnswer | None
    validation: ValidationReport | None
    repairs: int
    final: ChatAnswer | None


class WorkerState(TypedDict):
    task: Task
    question: str
    turn_id: str
