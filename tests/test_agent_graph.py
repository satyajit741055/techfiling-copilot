from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.types import Command, Send

from agent.catalog import FilingCatalog
from agent.evidence import EvidenceRegistry
from agent.graph import (
    build_graph,
    graph_config,
    route_after_coverage,
    route_after_plan,
    route_after_scope,
    route_after_validate,
)
from agent.nodes import NodeDependencies
from agent.schemas import (
    Analysis,
    CoverageCell,
    GroundedClaim,
    ResearchPlan,
    ResolvedScope,
    SynthesisDraft,
    Task,
    ValidationReport,
)
from agent.tools import AgentTools
from retriever.hybrid_retriever import SearchResult


def chunk(ticker: str, year: int, text: str) -> dict[str, Any]:
    return {
        "chunk_id": f"{ticker}_10-K_{year}_c0000",
        "doc_id": f"{ticker}_10-K_{year}",
        "ticker": ticker,
        "fiscal_year": year,
        "section": "Financial Statements",
        "type": "text",
        "text": text,
    }


CHUNKS = [
    chunk("AAPL", 2024, "Net sales were $90 million."),
    chunk("AAPL", 2025, "Net sales were $100 million."),
    chunk("MSFT", 2025, "Revenue was $200 million."),
]


class FakeRetriever:
    def __init__(self, good_queries: set[str] | None = None) -> None:
        self.good_queries = good_queries

    def search(self, query: str, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        if self.good_queries is not None and query not in self.good_queries:
            return []
        return [
            SearchResult(0.9, item)
            for item in CHUNKS
            if item["ticker"] in filters["ticker"] and item["fiscal_year"] in filters["fiscal_year"]
        ][:k]


class FakeStore:
    def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [item for item in CHUNKS if item["chunk_id"] in chunk_ids]

    def find_chunks(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return []


class DoneModel:
    """Worker model that always decides the search hits are sufficient."""

    def bind_tools(self, tools: list[Any]) -> DoneModel:
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        return SimpleNamespace(tool_calls=[{"name": "done", "args": {}}])


class NoToolsModel:
    def bind_tools(self, tools: list[Any]) -> Any:
        raise NotImplementedError("tool calling unsupported")


class ScriptedLLM:
    def __init__(self, responses: dict[type, list[Any]], model: Any = None) -> None:
        self.responses = {schema: list(items) for schema, items in responses.items()}
        self.model = model or DoneModel()
        self.calls: Counter[str] = Counter()

    async def structured(self, schema: type[Any], messages: list[Any]) -> Any:
        self.calls[schema.__name__] += 1
        return self.responses[schema].pop(0)

    async def invoke_model(self, model: Any, messages: list[Any]) -> Any:
        return await model.ainvoke(messages)


def make_deps(
    llm: ScriptedLLM, retriever: FakeRetriever | None = None, generator: Any = None
) -> NodeDependencies:
    catalog = FilingCatalog(
        {
            "doc_id": item["doc_id"],
            "ticker": item["ticker"],
            "company": {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}[item["ticker"]],
            "fiscal_year": item["fiscal_year"],
        }
        for item in CHUNKS
    )
    tools = AgentTools(catalog, retriever or FakeRetriever(), FakeStore(), EvidenceRegistry())  # type: ignore[arg-type]
    return NodeDependencies(catalog=catalog, llm=llm, tools=tools, generator=generator)  # type: ignore[arg-type]


def analysis(question_type: str = "single_fact", **values: Any) -> Analysis:
    return Analysis(
        intent="filing_question", scope="specific", question_type=question_type, **values
    )


def plan(query: str) -> ResearchPlan:
    return ResearchPlan(items=[{"sub_question": "Revenue", "query": query}])


def claim(ticker: str, source_id: str, text: str, excerpt: str) -> GroundedClaim:
    return GroundedClaim(
        text=text, citation_ids=[source_id], excerpt=excerpt, ticker=ticker, fiscal_year=2025
    )


def cell(ticker: str, status: str, year: int = 2025) -> CoverageCell:
    return CoverageCell(ticker=ticker, fiscal_year=year, status=status)  # type: ignore[arg-type]


COMPARISON = SynthesisDraft(
    answer="Apple net sales were $100 million [S1]; Microsoft revenue was $200 million [S2].",
    claims=[
        claim("AAPL", "S1", "Apple net sales were $100 million.", "Net sales were $100 million"),
        claim("MSFT", "S2", "Microsoft revenue was $200 million.", "Revenue was $200 million"),
    ],
)
APPLE = SynthesisDraft(
    answer="Apple net sales were $100 million [S1].",
    claims=[
        claim("AAPL", "S1", "Apple net sales were $100 million.", "Net sales were $100 million")
    ],
)


def simple_generator() -> Any:
    generated = SimpleNamespace(
        answer="Net sales were $100 million [S1].",
        citations=["S1"],
        is_valid=True,
        insufficient_context=False,
        validation_errors=[],
    )
    result = SimpleNamespace(
        answer=generated,
        sources=[{"source_id": "S1", **CHUNKS[1]}],
        prompt_tokens=10,
        completion_tokens=5,
        llm_attempts=1,
    )
    return SimpleNamespace(answer=lambda question, **kwargs: result)


def scope_with(route: str) -> ResolvedScope:
    return ResolvedScope(route=route)  # type: ignore[arg-type]


def test_routing_functions() -> None:
    assert route_after_scope({"scope": scope_with("research")}) == "plan"
    assert route_after_scope({"scope": scope_with("clarify")}) == "clarify"
    tasks = [
        Task(task_id=f"T{index}", ticker="AAPL", fiscal_year=2025, sub_question="Q", query="q")
        for index in (1, 2)
    ]
    sends = route_after_plan({"tasks": tasks, "question": "Q", "turn_id": "t1"})
    assert isinstance(sends, list) and all(isinstance(send, Send) for send in sends)
    assert [send.arg["task"].task_id for send in sends] == ["T1", "T2"]
    assert sends[0].node == "execute" and sends[0].arg["turn_id"] == "t1"
    assert route_after_plan({"tasks": [], "question": "Q"}) == "check_coverage"
    assert route_after_coverage({"coverage": [cell("AAPL", "missing")]}) == "plan"
    assert route_after_coverage({"coverage": [cell("AAPL", "not_disclosed")]}) == "synthesize"
    invalid = ValidationReport(is_valid=False, errors=["x"])
    assert route_after_validate({"validation": invalid, "repairs": 1}) == "synthesize"
    assert route_after_validate({"validation": invalid, "repairs": 2}) == "finalize"
    assert route_after_validate({"validation": ValidationReport(is_valid=True)}) == "finalize"


def test_recursion_limit_grows_with_replans() -> None:
    assert graph_config("t")["recursion_limit"] == 40
    assert graph_config("t", max_replans=12)["recursion_limit"] == 70


@pytest.mark.asyncio
async def test_missing_turn_id_gets_a_unique_one() -> None:
    refused = Analysis(
        intent="general_knowledge", scope="general_knowledge", question_type="qualitative"
    )
    graph = build_graph(make_deps(ScriptedLLM({Analysis: [refused, refused]})))
    first = await graph.ainvoke({"question": "What is EBITDA?"}, graph_config("a"))
    second = await graph.ainvoke({"question": "What is EBITDA?"}, graph_config("b"))
    assert first["turn_id"] and second["turn_id"] and first["turn_id"] != second["turn_id"]


@pytest.mark.asyncio
async def test_refuse_and_onboard_routes() -> None:
    refused = Analysis(
        intent="general_knowledge", scope="general_knowledge", question_type="qualitative"
    )
    greeting = Analysis(intent="greeting", scope="specific", question_type="qualitative")
    graph = build_graph(make_deps(ScriptedLLM({Analysis: [refused, greeting]})))
    config = graph_config("thread")

    first = await graph.ainvoke({"question": "What is EBITDA?", "turn_id": "t1"}, config)
    second = await graph.ainvoke({"question": "Hi", "turn_id": "t2"}, config)
    assert first["final"].route == "refuse"
    assert "AAPL" in first["final"].answer
    assert second["final"].route == "onboard"


@pytest.mark.asyncio
async def test_simple_rag_route() -> None:
    llm = ScriptedLLM({Analysis: [analysis(companies=["AAPL"])]})
    graph = build_graph(make_deps(llm, generator=simple_generator()))
    state = await graph.ainvoke({"question": "Apple revenue?", "turn_id": "t1"}, graph_config("t"))
    assert state["final"].route == "simple_rag"
    assert state["final"].coverage[0].status == "found"
    assert state["turns"][0].tickers == ["AAPL"]


@pytest.mark.asyncio
async def test_fan_out_streams_progress_and_next_turn_resets_results() -> None:
    llm = ScriptedLLM(
        {
            Analysis: [
                analysis("cross_company", companies=["Apple", "Microsoft"]),
                analysis(companies=["AAPL"]),
            ],
            ResearchPlan: [plan("revenue")],
            SynthesisDraft: [COMPARISON],
        }
    )
    deps = make_deps(llm, generator=simple_generator())
    graph = build_graph(deps)
    config = graph_config("thread")

    updates: list[dict[str, Any]] = []
    custom: list[dict[str, Any]] = []
    async for mode, payload in graph.astream(
        {"question": "Compare revenue", "turn_id": "t1"}, config, stream_mode=["updates", "custom"]
    ):
        (updates if mode == "updates" else custom).append(payload)

    assert sum("execute" in update for update in updates) == 2
    labels = [event["label"] for event in custom if event["type"] == "step"]
    assert "Scope: AAPL, MSFT · FY2025" in labels
    assert "Planning 2 searches" in labels
    assert "Coverage 2/2" in labels
    tools = [event for event in custom if event["type"] == "tool"]
    assert sorted((event["ticker"], event["n_results"]) for event in tools) == [
        ("AAPL", 1),
        ("MSFT", 1),
    ]

    state = (await graph.aget_state(config)).values
    final = state["final"]
    assert final.route == "research"
    assert final.is_valid, final.validation_errors
    assert [(item.ticker, item.status) for item in final.coverage] == [
        ("AAPL", "found"),
        ("MSFT", "found"),
    ]
    assert [source.source_id for source in final.sources] == ["S1", "S2"]
    assert final.usage.tool_calls == 2
    assert len(deps.evidence) == 0

    second = await graph.ainvoke({"question": "Apple revenue?", "turn_id": "t2"}, config)
    assert second["results"] == []
    assert [turn.tickers for turn in second["turns"]] == [["AAPL", "MSFT"], ["AAPL"]]


@pytest.mark.asyncio
async def test_replan_then_success_with_one_repair_pass() -> None:
    invalid = APPLE.model_copy(
        update={"claims": [claim("AAPL", "S1", "Apple net sales were $100 million.", "invented")]}
    )
    llm = ScriptedLLM(
        {
            Analysis: [analysis("trend", companies=["AAPL"])],
            ResearchPlan: [plan("bad query"), plan("net sales")],
            SynthesisDraft: [invalid, APPLE],
        }
    )
    graph = build_graph(make_deps(llm, FakeRetriever({"net sales"})))
    state = await graph.ainvoke({"question": "Revenue trend?", "turn_id": "t1"}, graph_config("t"))

    assert llm.calls == {"Analysis": 1, "ResearchPlan": 2, "SynthesisDraft": 2}
    assert state["replans"] == 1
    assert [task.task_id for task in state["tasks"]] == ["T2"]
    assert state["final"].is_valid
    assert state["final"].coverage[0].status == "found"


@pytest.mark.asyncio
async def test_repair_that_abstains_keeps_valid_claims_from_the_first_draft() -> None:
    partly_invalid = COMPARISON.model_copy(
        update={
            "claims": [
                COMPARISON.claims[0],
                claim("MSFT", "S2", "Microsoft revenue was $200 million.", "invented"),
            ]
        }
    )
    abstain = SynthesisDraft(answer="", claims=[], insufficient_context=True)
    llm = ScriptedLLM(
        {
            Analysis: [analysis("cross_company", companies=["Apple", "Microsoft"])],
            ResearchPlan: [plan("revenue")],
            SynthesisDraft: [partly_invalid, abstain],
        }
    )
    graph = build_graph(make_deps(llm))
    state = await graph.ainvoke({"question": "Compare revenue", "turn_id": "t1"}, graph_config("t"))

    final = state["final"]
    assert llm.calls["SynthesisDraft"] == 2
    assert final.is_valid, final.validation_errors
    assert final.answer == "Apple net sales were $100 million. [S1]"
    assert [(item.ticker, item.status) for item in final.coverage] == [
        ("AAPL", "found"),
        ("MSFT", "not_disclosed"),
    ]
    assert final.dropped_claims == ["Microsoft revenue was $200 million."]


@pytest.mark.asyncio
async def test_replans_exhausted_marks_not_disclosed() -> None:
    abstain = SynthesisDraft(answer="", claims=[], insufficient_context=True)
    llm = ScriptedLLM(
        {
            Analysis: [analysis("trend", companies=["AAPL"])],
            ResearchPlan: [plan("a"), plan("b"), plan("c")],
            SynthesisDraft: [abstain],
        }
    )
    graph = build_graph(make_deps(llm, FakeRetriever(set())))
    state = await graph.ainvoke({"question": "Revenue trend?", "turn_id": "t1"}, graph_config("t"))

    assert llm.calls["ResearchPlan"] == 3
    assert state["replans"] == 2
    assert state["final"].coverage[0].status == "not_disclosed"
    assert state["final"].answer is None
    assert state["final"].is_valid


@pytest.mark.asyncio
async def test_clarify_interrupts_and_resumes() -> None:
    llm = ScriptedLLM({Analysis: [analysis()]})
    graph = build_graph(make_deps(llm, generator=simple_generator()))
    config = graph_config("thread")

    paused = await graph.ainvoke({"question": "What was revenue?", "turn_id": "t1"}, config)
    request = paused["__interrupt__"][0].value
    assert request["options"] == ["AAPL", "MSFT", "All"]
    assert request["allow_free_text"]

    resumed = await graph.ainvoke(Command(resume="Apple"), config)
    assert resumed["final"].route == "simple_rag"
    assert resumed["clarifications_this_turn"] == 1
    assert resumed["scope"].tickers == ["AAPL"]
    assert llm.calls["Analysis"] == 1


@pytest.mark.asyncio
async def test_worker_without_tool_calling_falls_back_to_search_hits() -> None:
    llm = ScriptedLLM(
        {
            Analysis: [analysis("cross_company", companies=["AAPL", "MSFT"])],
            ResearchPlan: [plan("revenue")],
            SynthesisDraft: [COMPARISON],
        },
        model=NoToolsModel(),
    )
    graph = build_graph(make_deps(llm))
    state = await graph.ainvoke({"question": "Compare", "turn_id": "t1"}, graph_config("t"))

    assert all(result.llm_fallback for result in state["results"])
    assert {result.status for result in state["results"]} == {"found"}
    assert state["final"].is_valid
