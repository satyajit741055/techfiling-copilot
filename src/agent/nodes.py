"""Composable node implementations for the filing-research graph."""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from generation.generator import AnswerGenerator

from .catalog import FilingCatalog
from .evidence import EvidenceRegistry, TurnEvidence
from .llm import AgentLLM
from .schemas import (
    Ambiguity,
    Analysis,
    ChatAnswer,
    Clarification,
    CoverageCell,
    GroundedAnswer,
    GroundedClaim,
    ResearchPlan,
    Source,
    Task,
    TaskResult,
    ToolStep,
    TurnSummary,
    Usage,
    ValidationReport,
    YearSelection,
)
from .scope import resolve_scope
from .state import RESET, AgentState, WorkerState
from .tools import AgentTools
from .validation import drop_invalid_claims, validate_grounded_answer

_PREVIEW_CHARS = 300
_PENDING_STATUSES = {"pending", "missing"}
_YEAR_MENTION = re.compile(r"(?:\bFY\s*)?(?<!\d)((?:19|20)\d{2})(?!\d)", re.IGNORECASE)
_LIST_SEPARATOR = re.compile(r"\s*(?:,|/|&|\band\b)\s*", re.IGNORECASE)


@dataclass
class NodeDependencies:
    catalog: FilingCatalog
    llm: AgentLLM
    tools: AgentTools
    evidence: TurnEvidence = field(default_factory=TurnEvidence)
    generator: AnswerGenerator | None = None
    max_tasks: int = 10
    max_replans: int = 2
    max_tool_steps: int = 2
    task_k: int = 5
    max_context_chunks: int = 24
    min_evidence_score: float | None = None
    retrieval_concurrency: int = 2
    _retrieval_semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._retrieval_semaphore = asyncio.Semaphore(self.retrieval_concurrency)

    async def retrieve(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        async with self._retrieval_semaphore:
            return await asyncio.to_thread(function, *args, **kwargs)

    def registry(self, state: Mapping[str, Any]) -> EvidenceRegistry:
        return self.evidence.get(_turn_id(state))

    def tools_for(self, state: Mapping[str, Any]) -> AgentTools:
        return self.tools.bind(self.registry(state))


def _turn_id(state: Mapping[str, Any]) -> str:
    return str(state.get("turn_id") or "")


def emit(event: dict[str, Any]) -> None:
    """Send a progress event to ``stream_mode="custom"``; a no-op outside a graph run."""
    try:
        writer = get_stream_writer()
    except (RuntimeError, KeyError):
        return
    writer(event)


def _step(node: str, label: str, detail: str | None = None) -> None:
    emit({"type": "step", "node": node, "label": label, "detail": detail})


def _filings_label(tickers: Sequence[str], years: Sequence[int]) -> str:
    years_text = ", ".join(f"FY{year}" for year in years) or "no filings"
    return f"{', '.join(tickers) or 'no companies'} · {years_text}"


def _analysis_messages(state: AgentState) -> list[BaseMessage]:
    history = "\n".join(
        f"Q: {turn.question}\nA: {turn.answer}" for turn in state.get("turns", [])[-3:]
    )
    return [
        SystemMessage(content="Classify the filing question. Use only the supplied conversation."),
        HumanMessage(content=f"Conversation:\n{history}\n\nQuestion: {state['question']}"),
    ]


async def analyze_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    _step("analyze", "Understanding the question")
    # Without a turn id every turn would share the "" evidence registry.
    turn_id = _turn_id(state) or uuid.uuid4().hex
    deps.evidence.release(turn_id)
    analysis = await deps.llm.structured(Analysis, _analysis_messages(state))
    return {
        "turn_id": turn_id,
        "analysis": analysis,
        "scope": None,
        "tasks": [],
        "results": RESET,
        "replans": 0,
        "coverage": [],
        "draft": None,
        "validation": None,
        "repairs": 0,
        "final": None,
        "clarifications_this_turn": 0,
    }


def resolve_scope_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    analysis = state.get("analysis")
    if analysis is None:
        raise ValueError("analyze must run before resolve_scope")
    scope = resolve_scope(
        analysis,
        deps.catalog,
        ui_filters=state.get("ui_filters"),
        turns=state.get("turns", []),
        clarifications_this_turn=state.get("clarifications_this_turn", 0),
        max_tasks=deps.max_tasks,
    )
    if scope.route not in {"refuse", "onboard"}:
        _step(
            "resolve_scope",
            f"Scope: {_filings_label(scope.tickers, scope.fiscal_years)}",
            "; ".join(scope.assumptions) or None,
        )
    return {"scope": scope, "coverage": scope.coverage}


def refuse_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    answer = (
        f"I can only answer questions grounded in the indexed SEC filings. "
        f"Available coverage: {deps.catalog.describe()}."
    )
    return {"final": ChatAnswer(answer=answer, route="refuse", coverage=state.get("coverage", []))}


def onboard_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    answer = (
        f"I research and compare the indexed 10-K filings for {deps.catalog.describe()}. "
        "Ask about a disclosed metric, trend, risk, or company comparison."
    )
    return {"final": ChatAnswer(answer=answer, route="onboard")}


def _choice_text(choice: Any) -> str:
    if isinstance(choice, Mapping):
        choice = choice.get("choice", choice.get("value", ""))
    if isinstance(choice, (list, tuple)):
        return ", ".join(str(item) for item in choice)
    return str(choice or "").strip()


def apply_clarification(
    analysis: Analysis,
    clarification: Clarification,
    choice: Any,
    catalog: FilingCatalog,
) -> Analysis:
    """Apply a clarification answer (option or free text) to the analysis."""
    text = _choice_text(choice)
    update: dict[str, Any] = {"ambiguity": Ambiguity()}
    if not text:
        return analysis.model_copy(update=update)
    years = sorted({int(year) for year in _YEAR_MENTION.findall(text)})
    if years:
        update["years"] = YearSelection(explicit=years)
    remainder = _YEAR_MENTION.sub(" ", text)
    parts = [part.strip() for part in _LIST_SEPARATOR.split(remainder) if part.strip()]
    companies = [part for part in parts if catalog.resolve_company(part)]
    if clarification.dimension == "company":
        if text.casefold() == "all":
            companies = catalog.tickers
        elif not companies:
            companies = parts
    if companies:
        update["companies"] = companies
    elif clarification.dimension in {"metric", "other"} and parts:
        update["topics"] = [*analysis.topics, text]
    return analysis.model_copy(update=update)


def clarify_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    scope = state.get("scope")
    clarification = scope.clarification if scope else None
    if clarification is None:
        raise ValueError("clarify requires a resolved clarification")
    analysis = state.get("analysis")
    if analysis is None:
        raise ValueError("clarify requires analysis")
    options = list(clarification.options[:4])
    if (
        clarification.dimension == "company"
        and len(deps.catalog.tickers) <= deps.max_tasks
        and "All" not in options
    ):
        options.append("All")
    choice = interrupt(
        {"question": clarification.question, "options": options, "allow_free_text": True}
    )
    return {
        "analysis": apply_clarification(analysis, clarification, choice, deps.catalog),
        "clarifications_this_turn": state.get("clarifications_this_turn", 0) + 1,
    }


async def simple_rag_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    if deps.generator is None:
        raise ValueError("simple_rag requires an AnswerGenerator")
    scope = state.get("scope")
    if scope is None or len(scope.cells) != 1:
        raise ValueError("simple_rag requires exactly one scope cell")
    cell = scope.cells[0]
    _step("simple_rag", f"Searching {cell.ticker} FY{cell.fiscal_year}")
    result = await deps.retrieve(
        deps.generator.answer,
        state["question"],
        filters={"ticker": cell.ticker, "fiscal_year": cell.fiscal_year},
    )
    generated = result.answer
    grounded = bool(generated.answer) and generated.is_valid and not generated.insufficient_context
    claims = (
        [
            GroundedClaim(
                text=generated.answer,
                citation_ids=generated.citations,
                ticker=cell.ticker,
                fiscal_year=cell.fiscal_year,
            )
        ]
        if grounded
        else []
    )
    sources = [
        Source(
            source_id=source["source_id"],
            chunk_id=source["chunk_id"],
            ticker=source["ticker"],
            fiscal_year=source["fiscal_year"],
            period_end=source.get("period_end"),
            section=source.get("section_title") or source.get("section"),
            table_id=source.get("table_id"),
            units=source.get("units"),
            text=source.get("text", ""),
        )
        for source in result.sources
    ]
    coverage = [
        CoverageCell(
            ticker=cell.ticker,
            fiscal_year=cell.fiscal_year,
            status="found" if grounded else "not_disclosed",
        )
    ]
    final = ChatAnswer(
        answer=generated.answer if grounded else None,
        claims=claims,
        coverage=coverage,
        assumptions=scope.assumptions,
        sources=sources,
        is_valid=generated.is_valid,
        validation_errors=generated.validation_errors,
        route="simple_rag",
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            llm_calls=result.llm_attempts,
        ),
    )
    turn = TurnSummary(
        question=state["question"],
        tickers=[cell.ticker],
        fiscal_years=[cell.fiscal_year],
        answer=final.answer or "",
    )
    return {"coverage": coverage, "final": final, "turns": [turn]}


def _plan_messages(
    state: AgentState, units: Sequence[str], max_items: int, previous: Sequence[Task]
) -> list[BaseMessage]:
    analysis = state.get("analysis")
    sub_questions = analysis.sub_questions if analysis else []
    lines = [
        f"Question: {state['question']}",
        f"Filings to research: {', '.join(units)}",
        f"Sub-questions: {sub_questions or 'derive from the question'}",
        f"Return at most {max_items} items, each with a short search query.",
        "Set section only to an exact 10-K section title; leave it null when unsure.",
    ]
    if previous:
        queries = sorted({task.query for task in previous})
        lines.append(
            f"These queries found nothing usable: {queries}. Propose different queries or sections."
        )
    return [
        SystemMessage(content="Plan distinct filing search queries for the listed filings."),
        HumanMessage(content="\n".join(lines)),
    ]


async def plan_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    scope = state.get("scope")
    if scope is None:
        raise ValueError("plan requires resolved scope")
    status = {
        (cell.ticker, cell.fiscal_year): cell.status
        for cell in state.get("coverage") or scope.coverage
    }
    missing = [
        cell
        for cell in scope.cells
        if status.get((cell.ticker, cell.fiscal_year), "pending") in _PENDING_STATUSES
    ]
    previous = state.get("tasks", [])
    replans = state.get("replans", 0) + bool(previous)
    units: list[tuple[str, int | None, list[int]]]
    if scope.collapse_years:
        units = [
            (ticker, None, years)
            for ticker in scope.tickers
            if (years := [cell.fiscal_year for cell in missing if cell.ticker == ticker])
        ]
    else:
        units = [(cell.ticker, cell.fiscal_year, []) for cell in missing]
    if not units:
        return {"tasks": [], "replans": replans}

    max_items = max(1, min(3, deps.max_tasks // len(units)))
    labels = [
        f"{ticker} FY{year}" if year is not None else f"{ticker} FY{', FY'.join(map(str, years))}"
        for ticker, year, years in units
    ]
    plan = await deps.llm.structured(
        ResearchPlan, _plan_messages(state, labels, max_items, previous)
    )
    offset = len(state.get("results", []))
    tasks = [
        Task(
            task_id=f"T{offset + index}",
            ticker=ticker,
            fiscal_year=year,
            fiscal_years=years,
            sub_question=item.sub_question,
            query=item.query,
            section=item.section,
        )
        for index, ((ticker, year, years), item) in enumerate(
            itertools.product(units, plan.items[:max_items]), start=1
        )
    ]
    tasks = tasks[: deps.max_tasks]
    _step("plan", f"Planning {len(tasks)} searches", "Re-plan" if replans else None)
    return {"tasks": tasks, "replans": replans}


def _worker_messages(
    question: str, task: Task, chunks: Sequence[Mapping[str, Any]], max_steps: int
) -> list[BaseMessage]:
    years = task.fiscal_years or ([task.fiscal_year] if task.fiscal_year is not None else [])
    previews = "\n".join(
        json.dumps(
            {
                "chunk_id": chunk.get("chunk_id"),
                "section": chunk.get("section_title") or chunk.get("section"),
                "type": chunk.get("type"),
                "table_id": chunk.get("table_id"),
                "text": str(chunk.get("text", ""))[:_PREVIEW_CHARS],
            },
            ensure_ascii=False,
        )
        for chunk in chunks
    )
    return [
        SystemMessage(
            content=(
                "You check search hits for one filing-research task. Call read_neighbors when "
                "a passage is cut off, read_table when a table is incomplete, or search_filings "
                "with a refined query when the hits miss the sub-question. Call done when the "
                f"hits are sufficient. Use at most {max_steps} tool calls."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\nSub-question: {task.sub_question}\n"
                f"Filing scope: {task.ticker} FY{', FY'.join(map(str, years))}\n"
                f"Hits:\n{previews}"
            )
        ),
    ]


async def execute_node(state: WorkerState, deps: NodeDependencies) -> dict[str, Any]:
    task = state["task"]
    tools = deps.tools_for(state)
    search_args: dict[str, Any] = {
        "query": task.query,
        "fiscal_years": task.fiscal_years
        or ([task.fiscal_year] if task.fiscal_year is not None else []),
        "k": deps.task_k,
        "group_by": "fiscal_year" if task.fiscal_years else None,
    }
    trajectory: list[ToolStep] = []
    chunks: list[dict[str, Any]] = []

    def record(step: ToolStep) -> None:
        trajectory.append(step)
        emit(
            {
                "type": "tool",
                "task_id": task.task_id,
                "tool": step.tool,
                "args": step.args,
                "n_results": len(step.chunk_ids),
                "ticker": task.ticker,
                "fiscal_year": task.fiscal_year,
                "fiscal_years": task.fiscal_years,
                "error": step.error,
            }
        )

    try:
        output = await deps.retrieve(
            tools.search_filings, task, section=task.section, **search_args
        )
        record(output.step)
        if not output.chunks and task.section:
            output = await deps.retrieve(tools.search_filings, task, **search_args)
            record(output.step)
        chunks.extend(output.chunks)
    except Exception as error:
        record(ToolStep(tool="search_filings", args=search_args, error=repr(error)))
        result = TaskResult(
            task_id=task.task_id,
            ticker=task.ticker,
            fiscal_year=task.fiscal_year,
            sub_question=task.sub_question,
            status="error",
            trajectory=trajectory,
        )
        return {"results": [result]}

    fallback = False
    if chunks and deps.max_tool_steps:
        follow_ups = {
            "search_filings": tools.search_filings,
            "read_neighbors": tools.read_neighbors,
            "read_table": tools.read_table,
        }
        try:
            model = deps.llm.model.bind_tools(tools.langchain_tools(task))
            response = await deps.llm.invoke_model(
                model, _worker_messages(state["question"], task, chunks, deps.max_tool_steps)
            )
        except Exception as error:
            fallback = True
            record(ToolStep(tool="worker_llm", error=type(error).__name__))
        else:
            calls = [
                call
                for call in getattr(response, "tool_calls", None) or []
                if call.get("name") in follow_ups
            ]
            for call in calls[: deps.max_tool_steps]:
                name, args = call["name"], call.get("args") or {}
                try:
                    follow_up = await deps.retrieve(follow_ups[name], task, **args)
                except Exception as error:
                    record(
                        ToolStep(
                            tool=name,
                            args=args if isinstance(args, dict) else {},
                            error=f"{type(error).__name__}: {error}"[:500],
                        )
                    )
                    continue
                record(follow_up.step)
                chunks.extend(follow_up.chunks)

    registry = deps.registry(state)
    chunk_ids = list(dict.fromkeys(chunk["chunk_id"] for chunk in chunks))
    scores = [
        entry.best_score
        for chunk_id in chunk_ids
        if (entry := registry.get(chunk_id)) is not None and entry.best_score is not None
    ]
    best_score = max(scores, default=None)
    usable = bool(chunk_ids) and (
        deps.min_evidence_score is None
        or (best_score is not None and best_score >= deps.min_evidence_score)
    )
    result = TaskResult(
        task_id=task.task_id,
        ticker=task.ticker,
        fiscal_year=task.fiscal_year,
        sub_question=task.sub_question,
        chunk_ids=chunk_ids,
        best_score=best_score,
        status="found" if usable else "missing",
        trajectory=trajectory,
        llm_fallback=fallback,
    )
    return {"results": [result]}


def check_coverage_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    scope = state.get("scope")
    if scope is None:
        raise ValueError("coverage requires scope")
    registry = deps.registry(state)
    found: set[tuple[str, int]] = set()
    for result in state.get("results", []):
        if result.status != "found":
            continue
        if result.fiscal_year is not None:
            found.add((result.ticker, result.fiscal_year))
            continue
        for chunk_id in result.chunk_ids:
            entry = registry.get(chunk_id)
            if entry and entry.chunk.get("fiscal_year") is not None:
                found.add((result.ticker, int(entry.chunk["fiscal_year"])))
    # No tasks means the planner had nothing left to search, so a re-plan cannot help.
    exhausted = state.get("replans", 0) >= deps.max_replans or not state.get("tasks")
    coverage = [
        cell.model_copy(
            update={
                "status": "found"
                if (cell.ticker, cell.fiscal_year) in found
                else ("not_disclosed" if exhausted else "missing")
            }
        )
        if cell.status != "not_in_corpus"
        else cell
        for cell in scope.coverage
    ]
    searchable = [cell for cell in coverage if cell.status != "not_in_corpus"]
    found_count = sum(cell.status == "found" for cell in searchable)
    _step("check_coverage", f"Coverage {found_count}/{len(searchable)}")
    return {"coverage": coverage}


def _synthesis_messages(
    state: AgentState, context: str, validation: ValidationReport | None
) -> list[BaseMessage]:
    cells = "\n".join(
        f"- {cell.ticker} FY{cell.fiscal_year}: {cell.status}" for cell in state.get("coverage", [])
    )
    rules = (
        "Answer only from the sources and cite every claim with inline [S#] ids. "
        "Each claim needs citation_ids, an excerpt copied verbatim (at most 300 characters) "
        "from a cited source, and the ticker and fiscal_year of the cited filing. Never mix "
        "companies or years in one claim. Return exactly one coverage entry per listed cell: "
        "keep not_in_corpus, use found only when a claim supports the cell, otherwise "
        "not_disclosed. Any derived number needs a calculation with cited inputs and a simple "
        "arithmetic expression."
    )
    prompt = f"Question: {state['question']}\nCoverage cells:\n{cells}\n\nSources:\n{context}"
    if validation is not None and not validation.is_valid:
        errors = "\n".join(f"- {error}" for error in validation.errors)
        prompt += f"\n\nThe previous draft failed validation; fix these errors:\n{errors}"
    return [SystemMessage(content=rules), HumanMessage(content=prompt)]


async def synthesize_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    _step("synthesize", "Writing answer", "Repair pass" if state.get("validation") else None)
    registry = deps.registry(state)
    registry.assign_source_ids(deps.max_context_chunks)
    draft = await deps.llm.structured(
        GroundedAnswer,
        _synthesis_messages(state, registry.format_context(), state.get("validation")),
    )
    return {"draft": draft}


def validate_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    """Validate the draft; request one repair, then drop invalid content and re-validate.

    ``repairs`` counts failed validations: after the first failure the graph returns to
    ``synthesize``; after the second the draft is cleaned and the turn finalizes.
    """
    draft, scope = state.get("draft"), state.get("scope")
    if draft is None or scope is None:
        raise ValueError("validate requires a draft and scope")
    _step("validate", "Verifying citations")
    entries = deps.registry(state).source_entries()
    report = validate_grounded_answer(draft, scope, entries, deps.catalog)
    repairs = state.get("repairs", 0)
    if report.is_valid or repairs < 1:
        return {"validation": report, "repairs": repairs + (not report.is_valid)}

    dropped_claims = list(report.dropped_claims)
    dropped_calculations = list(report.dropped_calculations)
    for _ in range(len(draft.claims) + len(draft.calculations) + 1):
        draft = drop_invalid_claims(draft, report, scope.coverage)
        report = validate_grounded_answer(draft, scope, entries, deps.catalog)
        dropped_claims.extend(report.dropped_claims)
        dropped_calculations.extend(report.dropped_calculations)
        if report.is_valid or not (report.dropped_claims or report.dropped_calculations):
            break
    report = report.model_copy(
        update={
            "dropped_claims": list(dict.fromkeys(dropped_claims)),
            "dropped_calculations": list(dict.fromkeys(dropped_calculations)),
        }
    )
    return {"draft": draft, "validation": report, "repairs": repairs + 1}


def finalize_node(state: AgentState, deps: NodeDependencies) -> dict[str, Any]:
    draft, scope = state.get("draft"), state.get("scope")
    if draft is None or scope is None:
        raise ValueError("finalize requires a draft and scope")
    validation = state.get("validation")
    drafted = {(cell.ticker, cell.fiscal_year): cell.status for cell in draft.coverage}
    coverage = [
        cell
        if cell.status == "not_in_corpus"
        else cell.model_copy(
            update={"status": drafted.get((cell.ticker, cell.fiscal_year), "not_disclosed")}
        )
        for cell in scope.coverage
    ]
    trajectory = [step for result in state.get("results", []) for step in result.trajectory]
    final = ChatAnswer(
        answer=draft.answer,
        claims=draft.claims,
        coverage=coverage,
        assumptions=scope.assumptions,
        calculations=draft.calculations,
        sources=deps.registry(state).sources(),
        is_valid=validation.is_valid if validation else False,
        validation_errors=validation.errors if validation else ["Draft was not validated"],
        dropped_claims=validation.dropped_claims if validation else [],
        route="research",
        trajectory=trajectory,
        usage=Usage(tool_calls=sum(step.tool != "worker_llm" for step in trajectory)),
    )
    deps.evidence.release(_turn_id(state))
    turn = TurnSummary(
        question=state["question"],
        tickers=scope.tickers,
        fiscal_years=scope.fiscal_years,
        answer=final.answer or "",
    )
    return {"final": final, "turns": [turn]}
