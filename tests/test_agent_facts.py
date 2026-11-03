"""Facts fast path, pinned evidence, code calculations and the period check (offline)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

import api.dependencies as dependencies
import persistence.facts as persistence_facts
from agent.catalog import FilingCatalog
from agent.evidence import EvidenceEntry, EvidenceRegistry
from agent.graph import build_graph, graph_config
from agent.nodes import NodeDependencies, _figures_text
from agent.schemas import (
    Analysis,
    Calculation,
    CalculationInput,
    CoverageCell,
    FactValue,
    GroundedAnswer,
    GroundedClaim,
    ResearchPlan,
    ResolvedScope,
    ScopeCell,
    SynthesisDraft,
)
from agent.tools import AgentTools
from agent.validation import ground_computed_claims, validate_grounded_answer
from facts.records import FactRecord
from facts.store import FactsStore
from persistence.facts import FactsLoad
from retriever.hybrid_retriever import SearchResult
from tests.test_agent_graph import ScriptedLLM

REVENUE = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
RD = "us-gaap:ResearchAndDevelopmentExpense"
UNITS = "(In millions, except per share amounts)"
MSFT_REVENUE_ROW = "Revenue > Total revenue | 245,122 | 211,915 | 198,270"
MSFT_RD_ROW = "Research and development | 29,510 | 27,195 | 24,512"
# Total net sales per Apple filing: (own year, prior year, two years earlier)
AAPL_SALES = {
    2023: (383_285, 394_328, 365_817),
    2024: (391_035, 383_285, 394_328),
    2025: (416_161, 391_035, 383_285),
}


def _row(values: tuple[int, ...]) -> str:
    return "Total net sales | " + " | ".join(f"{value:,}" for value in values)


def _chunk(ticker: str, year: int, number: int, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "chunk_id": f"{ticker}_10-K_{year}_c{number:04d}",
        "doc_id": f"{ticker}_10-K_{year}",
        "ticker": ticker,
        "fiscal_year": year,
        "section": "Financial Statements",
        "type": "text",
        "text": text,
        **extra,
    }


def _table(ticker: str, year: int, *rows: str) -> dict[str, Any]:
    header = f"| Year | {year} | {year - 1} | {year - 2} |\n|---|---|---|---|\n"
    body = "\n".join(f"| {row} |" for row in rows)
    return _chunk(
        ticker,
        year,
        100,
        f"**INCOME STATEMENTS** {UNITS}\n\n{header}{body}",
        type="table",
        table_id=f"{ticker}_10-K_{year}_t001",
        units=UNITS,
    )


MSFT_TABLE = _table("MSFT", 2024, MSFT_REVENUE_ROW, MSFT_RD_ROW)
AAPL_TABLES = {year: _table("AAPL", year, _row(values)) for year, values in AAPL_SALES.items()}
CHUNKS = [
    MSFT_TABLE,
    _chunk("MSFT", 2024, 1, "Microsoft competes in cloud services."),
    *AAPL_TABLES.values(),
    *(
        _chunk("AAPL", year, 1, "Risk factors include supply chain disruption.")
        for year in AAPL_SALES
    ),
]


def _record(
    chunk: dict[str, Any], concept: str, fiscal_year: int, shown: float, excerpt: str
) -> FactRecord:
    filing_year = int(chunk["fiscal_year"])
    return FactRecord(
        doc_id=str(chunk["doc_id"]),
        fact_id=f"{concept}-{fiscal_year}-{filing_year}",
        ticker=str(chunk["ticker"]),
        filing_fiscal_year=filing_year,
        fiscal_year=fiscal_year,
        concept=concept,
        period_start=f"{fiscal_year - 1}-07-01",
        period_end=f"{fiscal_year}-06-30",
        instant=False,
        unit="USD",
        value=shown * 1e6,
        shown=shown,
        scale=6,
        chunk_id=str(chunk["chunk_id"]),
        table_id=str(chunk["table_id"]),
        row_label=excerpt.split(" | ")[0],
        excerpt=excerpt,
    )


def _records() -> list[FactRecord]:
    records = [
        _record(MSFT_TABLE, REVENUE, 2024 - offset, value, MSFT_REVENUE_ROW)
        for offset, value in enumerate((245_122, 211_915, 198_270))
    ] + [
        _record(MSFT_TABLE, RD, 2024 - offset, value, MSFT_RD_ROW)
        for offset, value in enumerate((29_510, 27_195, 24_512))
    ]
    for year, values in AAPL_SALES.items():
        records += [
            _record(AAPL_TABLES[year], REVENUE, year - offset, value, _row(values))
            for offset, value in enumerate(values)
        ]
    return records


STORE = FactsStore(_records(), collection="test")


class Retriever:
    def search(self, query: str, k: int, filters: dict[str, Any]) -> list[SearchResult]:
        return [
            SearchResult(0.9, item)
            for item in CHUNKS
            if item["type"] == "text"
            and item["ticker"] in filters["ticker"]
            and item["fiscal_year"] in filters["fiscal_year"]
        ][:k]


class Store:
    def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [item for item in CHUNKS if item["chunk_id"] in chunk_ids]

    def find_chunks(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return []


def _catalog() -> FilingCatalog:
    names = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"}
    filings = {(item["ticker"], item["fiscal_year"]) for item in CHUNKS}
    return FilingCatalog(
        {
            "doc_id": f"{ticker}_10-K_{year}",
            "ticker": ticker,
            "company": names[ticker],
            "fiscal_year": year,
        }
        for ticker, year in sorted(filings)
    )


def _failing_generator() -> Any:
    def answer(question: str, **kwargs: Any) -> Any:
        raise AssertionError("simple_rag must not run")

    return SimpleNamespace(answer=answer)


def _deps(
    llm: ScriptedLLM, *, facts: FactsStore | None = STORE, generator: Any = None, **kwargs: Any
) -> NodeDependencies:
    catalog = _catalog()
    tools = AgentTools(catalog, Retriever(), Store(), EvidenceRegistry())  # type: ignore[arg-type]
    return NodeDependencies(
        catalog=catalog,
        llm=llm,  # type: ignore[arg-type]
        tools=tools,
        generator=generator,
        facts=facts,
        **kwargs,
    )


def _analysis(
    question_type: str, companies: list[str], topics: list[str], **years: Any
) -> Analysis:
    return Analysis(
        intent="filing_question",
        scope="specific",
        question_type=question_type,
        companies=companies,
        years=years,
        topics=topics,
    )


def _claim(ticker: str, year: int, source_id: str, text: str, excerpt: str) -> GroundedClaim:
    return GroundedClaim(
        text=text, citation_ids=[source_id], excerpt=excerpt, ticker=ticker, fiscal_year=year
    )


MSFT_ANSWER = SynthesisDraft(
    answer="Microsoft's total revenue was $245,122 million in fiscal 2024 [S1].",
    claims=[
        _claim(
            "MSFT",
            2024,
            "S1",
            "Microsoft's total revenue was $245,122 million in fiscal 2024.",
            MSFT_REVENUE_ROW,
        )
    ],
)


@pytest.mark.asyncio
async def test_numeric_single_filing_question_is_answered_from_facts_without_retrieval() -> None:
    llm = ScriptedLLM(
        {
            Analysis: [_analysis("single_fact", ["MSFT"], ["total revenue"], explicit=[2024])],
            SynthesisDraft: [MSFT_ANSWER],
        }
    )
    graph = build_graph(_deps(llm, generator=_failing_generator()))
    state = await graph.ainvoke(
        {"question": "What was Microsoft's revenue in fiscal 2024?", "turn_id": "t1"},
        graph_config("t"),
    )

    assert llm.calls == {"Analysis": 1, "SynthesisDraft": 1}
    final = state["final"]
    assert final.is_valid, final.validation_errors
    assert final.route == "research"
    assert [source.chunk_id for source in final.sources] == [MSFT_TABLE["chunk_id"]]
    assert [(fact.metric, fact.shown) for fact in state["facts"]] == [("revenue", 245_122)]
    assert [step.tool for step in final.trajectory] == ["query_financials"]
    assert final.coverage[0].status == "found"


@pytest.mark.asyncio
async def test_analysis_without_topics_falls_back_to_scanning_the_question() -> None:
    llm = ScriptedLLM(
        {
            Analysis: [_analysis("single_fact", ["Microsoft"], [], explicit=[2024])],
            SynthesisDraft: [MSFT_ANSWER],
        }
    )
    graph = build_graph(_deps(llm, generator=_failing_generator()))
    state = await graph.ainvoke(
        {"question": "What was Microsoft's total revenue in fiscal 2024?", "turn_id": "t1"},
        graph_config("t"),
    )
    assert llm.calls == {"Analysis": 1, "SynthesisDraft": 1}
    assert state["final"].is_valid and state["facts"][0].shown == 245_122


@pytest.mark.asyncio
async def test_trend_question_gets_code_calculations_that_pass_validation() -> None:
    first, last = AAPL_SALES[2023][0], AAPL_SALES[2025][0]
    cagr = ((last / first) ** (1 / 2) - 1) * 100
    claims = [
        _claim(
            "AAPL",
            year,
            f"S{index}",
            f"Apple's total net sales were ${AAPL_SALES[year][0]:,} million in fiscal {year}.",
            _row(AAPL_SALES[year]),
        )
        for index, year in enumerate(AAPL_SALES, start=1)
    ]
    # Small models copy the excerpt from the prompt's computed-values list (no source has it).
    claims.append(
        _claim(
            "AAPL",
            2023,
            "S1",
            f"Net sales grew at a compound annual rate of {cagr:.2f}% from fiscal 2023.",
            f"AAPL Total revenue CAGR % FY2023 to FY2025 = {cagr:.2f}% [S3, S1]",
        )
    )
    draft = SynthesisDraft(
        answer=" ".join(f"{claim.text} [{claim.citation_ids[0]}]" for claim in claims),
        claims=claims,
    )
    llm = ScriptedLLM(
        {
            Analysis: [
                _analysis("trend", ["AAPL"], ["revenue growth"], range={"gte": 2023, "lte": 2025})
            ],
            SynthesisDraft: [draft],
        }
    )
    graph = build_graph(_deps(llm))
    state = await graph.ainvoke(
        {"question": "How did Apple's revenue grow from 2023 to 2025?", "turn_id": "t1"},
        graph_config("t"),
    )

    final = state["final"]
    assert "ResearchPlan" not in llm.calls
    assert final.is_valid, final.validation_errors
    descriptions = [calculation.description for calculation in final.calculations]
    assert "AAPL Total revenue CAGR % FY2023 to FY2025" in descriptions
    assert "AAPL Total revenue growth % FY2024 to FY2025" in descriptions
    computed = {item.description: item for item in final.calculations}
    assert computed["AAPL Total revenue CAGR % FY2023 to FY2025"].result == pytest.approx(cagr)
    assert "**" in computed["AAPL Total revenue CAGR % FY2023 to FY2025"].expression
    assert [item.status for item in final.coverage] == ["found", "found", "found"]
    cagr_claim = final.claims[-1]
    assert (cagr_claim.citation_ids, cagr_claim.fiscal_year) == (["S3"], 2025)
    assert cagr_claim.excerpt == _row(AAPL_SALES[2025])
    assert final.dropped_claims == []


def test_computed_claim_citing_every_input_filing_cites_the_latest_row() -> None:
    entries = {
        "S1": EvidenceEntry(chunk=AAPL_TABLES[2024], source_id="S1"),
        "S2": EvidenceEntry(chunk=AAPL_TABLES[2025], source_id="S2"),
    }
    old, new = AAPL_SALES[2024][0], AAPL_SALES[2025][0]
    growth = Calculation(
        description="AAPL Total revenue growth % FY2024 to FY2025",
        expression=f"({new} - {old}) / {old} * 100",
        result=(new - old) / old * 100,
        inputs=[
            CalculationInput(value=new, citation_id="S2"),
            CalculationInput(value=old, citation_id="S1"),
        ],
    )
    text = f"Apple's net sales grew {(new - old) / old * 100:.2f}% in fiscal 2025."
    claim = GroundedClaim(
        text=text,
        citation_ids=["S2", "S1"],
        excerpt=_row(AAPL_SALES[2025]),
        ticker="AAPL",
        fiscal_year=2025,
    )
    plain = claim.model_copy(update={"text": "Apple's net sales were $416,161 million."})
    grounded = ground_computed_claims([claim, plain], [growth], entries)
    assert grounded[0].citation_ids == ["S2"] and grounded[0].excerpt == _row(AAPL_SALES[2025])
    assert grounded[1] == plain


@pytest.mark.asyncio
async def test_mixed_question_pins_figures_and_still_runs_the_workers() -> None:
    abstain = SynthesisDraft(answer="", claims=[], insufficient_context=True)
    llm = ScriptedLLM(
        {
            Analysis: [
                _analysis(
                    "trend",
                    ["AAPL"],
                    ["revenue", "risk factors"],
                    range={"gte": 2024, "lte": 2025},
                )
            ],
            ResearchPlan: [ResearchPlan(items=[{"sub_question": "Risks", "query": "risks"}])],
            SynthesisDraft: [abstain],
        }
    )
    graph = build_graph(_deps(llm))
    state = await graph.ainvoke(
        {"question": "Apple revenue and main risk factors?", "turn_id": "t1"}, graph_config("t")
    )

    assert llm.calls["ResearchPlan"] == 1
    assert sum(result.task_id.startswith("T") for result in state["results"]) == 2
    chunk_ids = {source.chunk_id for source in state["final"].sources}
    assert {AAPL_TABLES[2024]["chunk_id"], AAPL_TABLES[2025]["chunk_id"]} <= chunk_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("topic", "reason"),
    [("iPhone revenue", "segment topic matches no metric"), ("SG&A", "no reported figure")],
)
async def test_unmatched_or_missing_metric_keeps_the_single_filing_route(
    topic: str, reason: str
) -> None:
    generated = SimpleNamespace(
        answer="Not found.",
        citations=[],
        is_valid=True,
        insufficient_context=True,
        validation_errors=[],
    )
    result = SimpleNamespace(
        answer=generated, sources=[], prompt_tokens=1, completion_tokens=1, llm_attempts=1
    )
    generator = SimpleNamespace(answer=lambda question, **kwargs: result)
    llm = ScriptedLLM({Analysis: [_analysis("single_fact", ["MSFT"], [topic], explicit=[2024])]})
    state = await build_graph(_deps(llm, generator=generator)).ainvoke(
        {"question": f"Microsoft {topic} in 2024?", "turn_id": "t1"}, graph_config("t")
    )
    assert state["final"].route == "simple_rag", reason
    assert state.get("facts") == []


@pytest.mark.asyncio
async def test_facts_off_builds_the_graph_without_lookup_facts() -> None:
    graph = build_graph(_deps(ScriptedLLM({}), facts=None))
    assert "lookup_facts" not in graph.get_graph().nodes
    assert "lookup_facts" in build_graph(_deps(ScriptedLLM({}))).get_graph().nodes


@pytest.mark.asyncio
async def test_wrong_column_draft_is_repaired() -> None:
    wrong = SynthesisDraft(
        answer="Microsoft's total revenue was $211,915 million [S1].",
        claims=[
            _claim(
                "MSFT",
                2024,
                "S1",
                "Microsoft's total revenue was $211,915 million.",
                MSFT_REVENUE_ROW,
            )
        ],
    )
    llm = ScriptedLLM(
        {
            Analysis: [_analysis("single_fact", ["MSFT"], ["revenue"], explicit=[2024])],
            SynthesisDraft: [wrong, MSFT_ANSWER],
        }
    )
    graph = build_graph(_deps(llm))
    config = graph_config("t")
    state = await graph.ainvoke({"question": "MSFT revenue 2024?", "turn_id": "t1"}, config)

    assert llm.calls["SynthesisDraft"] == 2
    assert state["first_draft"].claims[0].text.endswith("$211,915 million.")
    assert state["final"].is_valid
    assert "245,122" in (state["final"].answer or "")


def _validate(text: str, facts: FactsStore | None = STORE) -> list[str]:
    entry = EvidenceEntry(chunk=MSFT_TABLE, source_id="S1")
    claim = _claim("MSFT", 2024, "S1", text, MSFT_REVENUE_ROW)
    scope = ResolvedScope(
        tickers=["MSFT"],
        fiscal_years=[2024],
        cells=[ScopeCell(ticker="MSFT", fiscal_year=2024)],
        coverage=[CoverageCell(ticker="MSFT", fiscal_year=2024)],
        route="research",
    )
    answer = GroundedAnswer(
        answer=f"{text} [S1]",
        claims=[claim],
        coverage=[CoverageCell(ticker="MSFT", fiscal_year=2024, status="found")],
    )
    return validate_grounded_answer(answer, scope, {"S1": entry}, _catalog(), facts).errors


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ("Microsoft revenue was $245,122 million.", True),
        ("Microsoft revenue was $245.1 billion.", True),
        ("Microsoft revenue was $211,915 million.", False),
        ("Microsoft revenue was $211.9 billion.", False),
        ("Revenue was $211,915 million in fiscal 2023.", True),
        ("Revenue rose from $211,915 million in FY23.", True),
        ("Revenue rose from the prior year's $211,915 million.", True),
        ("Revenue was $198,270 million in the prior year.", False),
    ],
)
def test_period_check_rejects_another_years_column_unless_named(text: str, valid: bool) -> None:
    errors = _validate(text)
    assert not errors if valid else any("FY20" in error and "column" in error for error in errors)


def test_period_check_needs_the_facts_store() -> None:
    assert _validate("Microsoft revenue was $211,915 million.", facts=None) == []


def test_pinned_rows_survive_a_small_context_budget() -> None:
    registry = EvidenceRegistry()
    others = [_chunk("MSFT", 2024, number, f"Passage {number}.") for number in (1, 3, 5)]
    registry.register(others, task_id="T1", tool="search", scores=[0.99, 0.98, 0.97])
    registry.register([MSFT_TABLE], task_id="facts", tool="query_financials", pinned=True)
    kept = [source.chunk_id for source in registry.assign_source_ids(2)]
    assert MSFT_TABLE["chunk_id"] in kept and len(kept) == 2


def test_figures_block_lists_restated_values_and_computed_values() -> None:
    fact = FactValue(
        metric="eps_diluted",
        label="Diluted EPS",
        concept="us-gaap:EarningsPerShareDiluted",
        ticker="TSLA",
        fiscal_year=2021,
        shown=4.90,
        value=4.90,
        unit="USD/share",
        scale=0,
        chunk_id="c1",
        excerpt="Diluted | 4.90",
        restated_shown=1.63,
        restated_in=2023,
    )
    growth = Calculation(
        description="TSLA Diluted EPS growth % FY2020 to FY2021",
        expression="(4.9 - 0.75) / 0.75 * 100",
        result=(4.9 - 0.75) / 0.75 * 100,
        inputs=[
            CalculationInput(value=4.9, citation_id="S4"),
            CalculationInput(value=0.75, citation_id="S4"),
        ],
    )
    text = _figures_text([fact], [growth], {"c1": "S4"})
    assert 'TSLA FY2021 Diluted EPS: 4.90 (USD per share) [S4], row: "Diluted | 4.90"' in text
    assert "the FY2023 10-K later reports 1.63" in text
    assert "- TSLA Diluted EPS growth % FY2020 to FY2021 = 553.33% [S4]" in text
    assert _figures_text([fact], [growth], {}) == ""


SETTINGS = SimpleNamespace(
    FACTS_STORE="postgres", HISTORY_DB_URL="postgresql://u@127.0.0.1:1/db", QDRANT_COLLECTION="c"
)


def test_startup_fails_open_when_the_database_is_unreachable(monkeypatch, caplog) -> None:
    def unreachable(conninfo: str) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(persistence_facts, "read_facts", unreachable)
    with caplog.at_level(logging.WARNING, logger="api.dependencies"):
        assert dependencies._load_facts_store(SETTINGS) is None
    assert "Facts store DISABLED" in caplog.text


@pytest.mark.parametrize(
    ("records", "load", "message"),
    [
        ([], None, "empty"),
        (_records(), FactsLoad(collection="other", chunk_run="r", facts=1, linked=1), "other"),
    ],
)
def test_startup_disables_facts_for_empty_or_mismatched_loads(
    monkeypatch, caplog, records, load, message
) -> None:
    monkeypatch.setattr(persistence_facts, "read_facts", lambda conninfo: (records, load))
    with caplog.at_level(logging.WARNING, logger="api.dependencies"):
        assert dependencies._load_facts_store(SETTINGS) is None
    assert message in caplog.text


def test_startup_loads_matching_facts(monkeypatch) -> None:
    load = FactsLoad(collection="c", chunk_run="r", facts=1, linked=1)
    monkeypatch.setattr(persistence_facts, "read_facts", lambda conninfo: (_records(), load))
    store = dependencies._load_facts_store(SETTINGS)
    assert store is not None and len(store) == len(_records())
    assert dependencies._load_facts_store(SimpleNamespace(FACTS_STORE="off")) is None
