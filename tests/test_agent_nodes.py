from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent.catalog import FilingCatalog
from agent.evidence import EvidenceRegistry
from agent.nodes import (
    NodeDependencies,
    analyze_node,
    apply_clarification,
    check_coverage_node,
    execute_node,
    finalize_node,
    onboard_node,
    plan_node,
    refuse_node,
    resolve_scope_node,
    simple_rag_node,
    validate_node,
)
from agent.schemas import (
    Analysis,
    Clarification,
    CoverageCell,
    GroundedAnswer,
    GroundedClaim,
    ResearchPlan,
    ResolvedScope,
    Task,
    TaskResult,
    ValidationReport,
)
from agent.tools import AgentTools
from retriever.hybrid_retriever import SearchResult


def chunk(chunk_id: str, ticker: str = "AAPL", year: int = 2025, **extra: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"{ticker}_10-K_{year}",
        "ticker": ticker,
        "fiscal_year": year,
        "section": "Financial Statements",
        "type": "text",
        "text": f"Net sales were $100 million ({chunk_id}).",
        **extra,
    }


CHUNKS = [chunk("AAPL_10-K_2025_c0000"), chunk("AAPL_10-K_2025_c0001")]


class FakeStore:
    def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        by_id = {item["chunk_id"]: item for item in CHUNKS}
        return [by_id[item] for item in chunk_ids if item in by_id]

    def find_chunks(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return []


class FakeRetriever:
    def search(self, query: str, k: int, **kwargs: Any) -> list[SearchResult]:
        filters = kwargs["filters"]
        return [
            SearchResult(0.9, item)
            for item in CHUNKS
            if item["ticker"] in filters["ticker"]
            and item["fiscal_year"] in filters["fiscal_year"]
            and filters.get("section", item["section"]) == item["section"]
        ][:k]


class FakeBoundModel:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, tool_calls: list[dict[str, Any]]) -> None:
        self.bound = FakeBoundModel(SimpleNamespace(tool_calls=tool_calls))

    def bind_tools(self, tools: list[Any]) -> FakeBoundModel:
        return self.bound


class StubLLM:
    def __init__(self, responses: list[Any], model: Any = None) -> None:
        self.responses = responses
        self.model = model
        self.messages: list[Any] = []

    async def structured(self, schema: type[Any], messages: list[Any]) -> Any:
        self.messages = messages
        return self.responses.pop(0)

    async def invoke_model(self, model: Any, messages: list[Any]) -> Any:
        return await model.ainvoke(messages)


def make_catalog(*cells: tuple[str, int]) -> FilingCatalog:
    names = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}
    return FilingCatalog(
        {
            "doc_id": f"{ticker}_10-K_{year}",
            "ticker": ticker,
            "company": names[ticker],
            "fiscal_year": year,
        }
        for ticker, year in cells
    )


@pytest.fixture
def dependencies() -> NodeDependencies:
    catalog = make_catalog(("AAPL", 2025))
    return NodeDependencies(
        catalog=catalog,
        llm=StubLLM([]),  # type: ignore[arg-type]
        tools=AgentTools(catalog, FakeRetriever(), FakeStore(), EvidenceRegistry()),  # type: ignore[arg-type]
    )


def research_scope(*cells: tuple[str, int], collapse: bool = False) -> ResolvedScope:
    return ResolvedScope(
        tickers=sorted({ticker for ticker, _ in cells}),
        fiscal_years=sorted({year for _, year in cells}),
        cells=[{"ticker": ticker, "fiscal_year": year} for ticker, year in cells],
        coverage=[{"ticker": ticker, "fiscal_year": year} for ticker, year in cells],
        route="research",
        collapse_years=collapse,
    )


@pytest.mark.asyncio
async def test_analyze_resets_turn_state(dependencies: NodeDependencies) -> None:
    analysis = Analysis(
        intent="filing_question",
        scope="specific",
        question_type="single_fact",
        companies=["AAPL"],
    )
    stale = dependencies.evidence.get("t1")
    dependencies.llm = StubLLM([analysis])  # type: ignore[assignment]
    update = await analyze_node(
        {"question": "Revenue?", "turns": [], "turn_id": "t1"}, dependencies
    )
    assert update["analysis"] == analysis
    assert update["results"].value == "__reset_agent_results__"
    assert update["replans"] == 0
    assert dependencies.evidence.get("t1") is not stale


def test_scope_refuse_and_onboard_nodes(dependencies: NodeDependencies) -> None:
    analysis = Analysis(
        intent="filing_question",
        scope="specific",
        question_type="single_fact",
        companies=["AAPL"],
    )
    update = resolve_scope_node({"analysis": analysis}, dependencies)
    assert update["scope"].route == "simple_rag"
    refused = refuse_node({"coverage": []}, dependencies)["final"]
    onboarded = onboard_node({}, dependencies)["final"]
    assert refused.route == "refuse"
    assert "AAPL" in refused.answer
    assert onboarded.route == "onboard"


def test_clarification_respects_dimension() -> None:
    catalog = make_catalog(("AAPL", 2025), ("MSFT", 2025))
    analysis = Analysis(
        intent="filing_question",
        scope="specific",
        question_type="trend",
        companies=["AAPL"],
        ambiguity={"is_ambiguous": True, "dimension": "year"},
    )
    company = Clarification(dimension="company", question="Which?")
    year = Clarification(dimension="year", question="Which year?")
    metric = Clarification(dimension="metric", question="Which metric?")

    companies = apply_clarification(analysis, company, "Apple and Microsoft", catalog)
    everyone = apply_clarification(analysis, company, {"choice": "All"}, catalog)
    years = apply_clarification(analysis, year, "FY2024", catalog)
    topic = apply_clarification(analysis, metric, "Gross margin", catalog)

    assert companies.companies == ["Apple", "Microsoft"]
    assert everyone.companies == ["AAPL", "MSFT"]
    assert years.companies == ["AAPL"]
    assert years.years.explicit == [2024]
    assert not years.ambiguity.is_ambiguous
    assert topic.companies == ["AAPL"]
    assert topic.topics == ["Gross margin"]


@pytest.mark.asyncio
async def test_simple_rag_records_turn_and_withholds_invalid_answer(
    dependencies: NodeDependencies,
) -> None:
    generated = SimpleNamespace(
        answer="$999 million",
        citations=["S1"],
        is_valid=False,
        insufficient_context=False,
        validation_errors=["Invalid citation: S1"],
    )
    result = SimpleNamespace(
        answer=generated, sources=[], prompt_tokens=10, completion_tokens=5, llm_attempts=1
    )
    dependencies.generator = SimpleNamespace(answer=lambda question, **kwargs: result)  # type: ignore[assignment]
    scope = research_scope(("AAPL", 2025)).model_copy(update={"route": "simple_rag"})

    update = await simple_rag_node({"question": "Revenue?", "scope": scope}, dependencies)
    assert update["final"].answer is None
    assert update["final"].claims == []
    assert update["coverage"][0].status == "not_disclosed"
    assert update["turns"][0].tickers == ["AAPL"]
    assert update["turns"][0].fiscal_years == [2025]


@pytest.mark.asyncio
async def test_plan_expands_items_over_scope_cells(dependencies: NodeDependencies) -> None:
    analysis = Analysis(
        intent="filing_question",
        scope="specific",
        question_type="trend",
        companies=["AAPL"],
    )
    scope = resolve_scope_node({"analysis": analysis}, dependencies)["scope"]
    dependencies.llm = StubLLM(
        [ResearchPlan(items=[{"sub_question": "Revenue", "query": "net sales"}])]
    )  # type: ignore[assignment]
    update = await plan_node(
        {"question": "Revenue trend?", "scope": scope, "results": [], "tasks": []},
        dependencies,
    )
    assert len(update["tasks"]) == 1
    assert update["tasks"][0].ticker == "AAPL"
    assert update["tasks"][0].query == "net sales"


@pytest.mark.asyncio
async def test_replan_targets_missing_cells_with_unique_ids_and_item_cap(
    dependencies: NodeDependencies,
) -> None:
    dependencies.max_tasks = 2
    scope = research_scope(("AAPL", 2024), ("AAPL", 2025))
    items = [{"sub_question": f"Q{index}", "query": f"query {index}"} for index in range(3)]
    dependencies.llm = StubLLM([ResearchPlan(items=items)])  # type: ignore[assignment]
    previous = Task(task_id="T1", ticker="AAPL", fiscal_year=2024, sub_question="Q", query="old")
    results = [
        TaskResult(task_id=f"T{index}", ticker="AAPL", sub_question="Q", status="missing")
        for index in (1, 2)
    ]
    coverage = [
        CoverageCell(ticker="AAPL", fiscal_year=2024, status="missing"),
        CoverageCell(ticker="AAPL", fiscal_year=2025, status="found"),
    ]

    update = await plan_node(
        {
            "question": "Revenue trend?",
            "scope": scope,
            "coverage": coverage,
            "results": results,
            "tasks": [previous],
        },
        dependencies,
    )
    assert [(task.task_id, task.fiscal_year) for task in update["tasks"]] == [
        ("T3", 2024),
        ("T4", 2024),
    ]
    assert update["replans"] == 1
    assert "old" in dependencies.llm.messages[-1].content  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_collapsed_plan_skips_tickers_without_missing_years(
    dependencies: NodeDependencies,
) -> None:
    scope = research_scope(("AAPL", 2025), ("MSFT", 2025), collapse=True)
    dependencies.llm = StubLLM(
        [ResearchPlan(items=[{"sub_question": "Revenue", "query": "net sales"}])]
    )  # type: ignore[assignment]
    coverage = [
        CoverageCell(ticker="AAPL", fiscal_year=2025, status="found"),
        CoverageCell(ticker="MSFT", fiscal_year=2025, status="missing"),
    ]
    update = await plan_node(
        {"question": "Compare", "scope": scope, "coverage": coverage, "tasks": [], "results": []},
        dependencies,
    )
    assert [(task.ticker, task.fiscal_years) for task in update["tasks"]] == [("MSFT", [2025])]


@pytest.mark.asyncio
async def test_worker_retries_without_section_and_survives_bad_tool_call(
    dependencies: NodeDependencies,
) -> None:
    model = FakeChatModel(
        [
            {"name": "read_table", "args": {"table_id": "t", "chunk_id": "c"}},
            {
                "name": "read_neighbors",
                "args": {"chunk_id": "AAPL_10-K_2025_c0000", "before": 0, "after": 1},
            },
        ]
    )
    dependencies.llm = StubLLM([], model=model)  # type: ignore[assignment]
    task = Task(
        task_id="T1",
        ticker="AAPL",
        fiscal_year=2025,
        sub_question="Net sales",
        query="net sales",
        section="Invented Section",
    )

    update = await execute_node(
        {"task": task, "question": "Revenue?", "turn_id": "t1"}, dependencies
    )
    result = update["results"][0]
    assert [step.tool for step in result.trajectory] == [
        "search_filings",
        "search_filings",
        "read_table",
        "read_neighbors",
    ]
    assert result.trajectory[0].chunk_ids == []
    assert result.trajectory[2].error
    assert result.status == "found"
    assert not result.llm_fallback
    assert "Net sales" in model.bound.messages[-1].content
    assert len(dependencies.evidence.get("t1")) == 2
    assert len(dependencies.evidence.get("t2")) == 0


@pytest.mark.asyncio
async def test_worker_respects_min_evidence_score(dependencies: NodeDependencies) -> None:
    dependencies.max_tool_steps = 0
    dependencies.min_evidence_score = 0.95
    task = Task(task_id="T1", ticker="AAPL", fiscal_year=2025, sub_question="Q", query="q")
    update = await execute_node({"task": task, "question": "Q", "turn_id": "t1"}, dependencies)
    assert update["results"][0].status == "missing"


def evidence_turn(dependencies: NodeDependencies) -> None:
    registry = dependencies.evidence.get("t1")
    registry.register(
        [chunk("AAPL_10-K_2025_c0000")], task_id="T1", tool="search_filings", scores=[0.9]
    )
    registry.assign_source_ids(5)


def draft_with_invalid_claim() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Net sales were $100 million [S1]; services grew 12% [S1].",
        claims=[
            GroundedClaim(
                text="Net sales were $100 million.",
                citation_ids=["S1"],
                excerpt="Net sales were $100 million",
                ticker="AAPL",
                fiscal_year=2025,
            ),
            GroundedClaim(
                text="Services grew 12%.",
                citation_ids=["S1"],
                excerpt="Services grew 12%",
                ticker="AAPL",
                fiscal_year=2025,
            ),
        ],
        coverage=[CoverageCell(ticker="AAPL", fiscal_year=2025, status="found")],
    )


def test_validate_requests_repair_then_drops_and_revalidates(
    dependencies: NodeDependencies,
) -> None:
    evidence_turn(dependencies)
    scope = research_scope(("AAPL", 2025))
    state = {"draft": draft_with_invalid_claim(), "scope": scope, "turn_id": "t1"}

    first = validate_node({**state, "repairs": 0}, dependencies)
    assert "draft" not in first
    assert not first["validation"].is_valid
    assert first["repairs"] == 1

    second = validate_node({**state, "repairs": 1}, dependencies)
    assert second["validation"].is_valid, second["validation"].errors
    assert second["validation"].dropped_claims == ["Services grew 12%."]
    assert second["draft"].answer == "Net sales were $100 million. [S1]"
    assert second["repairs"] == 2


def test_coverage_and_finalize(dependencies: NodeDependencies) -> None:
    analysis = Analysis(
        intent="filing_question",
        scope="specific",
        question_type="trend",
        companies=["AAPL"],
    )
    scope = resolve_scope_node({"analysis": analysis}, dependencies)["scope"]
    result = TaskResult(
        task_id="T1",
        ticker="AAPL",
        fiscal_year=2025,
        sub_question="Revenue",
        status="found",
    )
    coverage = check_coverage_node(
        {"scope": scope, "results": [result], "replans": 0, "turn_id": "t1"}, dependencies
    )["coverage"]
    draft = GroundedAnswer(answer=None, coverage=coverage, insufficient_context=True)
    evidence_turn(dependencies)
    final = finalize_node(
        {
            "question": "Revenue?",
            "scope": scope,
            "coverage": coverage,
            "draft": draft,
            "validation": ValidationReport(is_valid=True),
            "results": [result],
            "turn_id": "t1",
        },
        dependencies,
    )
    assert final["final"].route == "research"
    assert final["final"].coverage[0].status == "found"
    assert [source.source_id for source in final["final"].sources] == ["S1"]
    assert final["turns"][0].tickers == ["AAPL"]
    assert len(dependencies.evidence) == 0

    unvalidated = finalize_node({"question": "Q", "scope": scope, "draft": draft}, dependencies)
    assert not unvalidated["final"].is_valid
