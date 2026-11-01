from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent.catalog import FilingCatalog
from agent.evidence import EvidenceRegistry, TurnEvidence
from agent.schemas import Task
from agent.tools import AgentTools, ReadNeighborsArgs, SearchFilingsArgs
from retriever.hybrid_retriever import SearchResult


def chunk(
    chunk_id: str,
    ticker: str = "AAPL",
    year: int = 2025,
    *,
    table_id: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": f"{ticker}_10-K_{year}",
        "ticker": ticker,
        "fiscal_year": year,
        "period_end": f"{year}-09-27",
        "section": "Financial Statements",
        "type": "table" if table_id else "text",
        "table_id": table_id,
        "text": chunk_id,
    }


class FakeStore:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = {item["chunk_id"]: item for item in chunks}

    def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [self.chunks[item] for item in chunk_ids if item in self.chunks]

    def find_chunks(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(
            [
                item
                for item in self.chunks.values()
                if all(item.get(key) == value for key, value in filters.items())
            ],
            key=lambda item: item["chunk_id"],
        )


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, k: int, **kwargs: Any) -> list[SearchResult]:
        self.calls.append({"query": query, "k": k, **kwargs})
        filters = kwargs["filters"]
        return [
            result
            for result in self.results
            if result.chunk["ticker"] in filters["ticker"]
            and result.chunk["fiscal_year"] in filters["fiscal_year"]
        ][:k]

    def search_grouped(self, *args: Any, **kwargs: Any) -> dict[Any, list[SearchResult]]:
        raise AssertionError("not expected")


@pytest.fixture
def tools() -> tuple[AgentTools, FakeRetriever, EvidenceRegistry]:
    chunks = [
        chunk("AAPL_10-K_2025_c0000"),
        chunk("AAPL_10-K_2025_c0001", table_id="AAPL_10-K_2025_t001"),
        chunk("AAPL_10-K_2025_c0002", table_id="AAPL_10-K_2025_t001"),
        chunk("MSFT_10-K_2025_c0000", ticker="MSFT"),
    ]
    catalog = FilingCatalog(
        [
            {
                "doc_id": "AAPL_10-K_2025",
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "fiscal_year": 2025,
            },
            {
                "doc_id": "MSFT_10-K_2025",
                "ticker": "MSFT",
                "company": "Microsoft Corporation",
                "fiscal_year": 2025,
            },
        ]
    )
    retriever = FakeRetriever([SearchResult(0.9, item) for item in chunks])
    evidence = EvidenceRegistry()
    return AgentTools(catalog, retriever, FakeStore(chunks), evidence), retriever, evidence  # type: ignore[arg-type]


def task() -> Task:
    return Task(
        task_id="T1",
        ticker="AAPL",
        fiscal_year=2025,
        sub_question="Revenue",
        query="revenue",
    )


def test_search_clamps_scope_and_registers_evidence(
    tools: tuple[AgentTools, FakeRetriever, EvidenceRegistry],
) -> None:
    agent_tools, retriever, evidence = tools
    output = agent_tools.search_filings(
        task(), query="revenue", tickers=["AAPL", "MSFT"], fiscal_years=[2025], k=2
    )

    assert output.step.scope_clamped
    assert retriever.calls[0]["filters"] == {"ticker": ["AAPL"], "fiscal_year": [2025]}
    assert output.step.chunk_ids == ["AAPL_10-K_2025_c0000", "AAPL_10-K_2025_c0001"]
    assert len(evidence) == 2
    assert evidence.get("AAPL_10-K_2025_c0000").best_score == 0.9  # type: ignore[union-attr]


def test_out_of_scope_search_does_not_retrieve(
    tools: tuple[AgentTools, FakeRetriever, EvidenceRegistry],
) -> None:
    agent_tools, retriever, _ = tools
    output = agent_tools.search_filings(
        task(), query="revenue", tickers=["MSFT"], fiscal_years=[2025]
    )

    assert output.chunks == []
    assert output.step.scope_clamped
    assert output.step.error
    assert retriever.calls == []


def test_neighbors_and_table_stay_in_worker_scope(
    tools: tuple[AgentTools, FakeRetriever, EvidenceRegistry],
) -> None:
    agent_tools, _, evidence = tools
    neighbors = agent_tools.read_neighbors(
        task(), chunk_id="AAPL_10-K_2025_c0001", before=1, after=1
    )
    table = agent_tools.read_table(task(), chunk_id="AAPL_10-K_2025_c0001")

    assert neighbors.step.chunk_ids == [
        "AAPL_10-K_2025_c0000",
        "AAPL_10-K_2025_c0001",
        "AAPL_10-K_2025_c0002",
    ]
    assert table.step.chunk_ids == ["AAPL_10-K_2025_c0001", "AAPL_10-K_2025_c0002"]
    assert len(evidence.get("AAPL_10-K_2025_c0001").provenance) == 2  # type: ignore[union-attr]


def test_expansions_inherit_anchor_score_and_bind_isolates_registry(
    tools: tuple[AgentTools, FakeRetriever, EvidenceRegistry],
) -> None:
    agent_tools, _, evidence = tools
    agent_tools.search_filings(task(), query="revenue", k=1)
    agent_tools.read_neighbors(task(), chunk_id="AAPL_10-K_2025_c0000", before=0, after=1)
    assert evidence.get("AAPL_10-K_2025_c0001").best_score == 0.9  # type: ignore[union-attr]

    other = EvidenceRegistry()
    agent_tools.bind(other).read_table(task(), table_id="AAPL_10-K_2025_t001")
    assert len(other) == 2
    assert len(evidence) == 2


def test_turn_evidence_is_isolated_and_bounded() -> None:
    turns = TurnEvidence(max_turns=2)
    first = turns.get("t1")
    assert turns.get("t1") is first
    assert turns.get("t2") is not first
    turns.get("t3")
    assert len(turns) == 2
    assert turns.get("t1") is not first
    turns.release("t1")
    assert len(turns) == 1


def test_tool_argument_caps() -> None:
    with pytest.raises(ValidationError):
        SearchFilingsArgs(query="x", k=11)
    with pytest.raises(ValidationError):
        ReadNeighborsArgs(chunk_id="AAPL_10-K_2025_c0001", before=3)


def test_evidence_source_ids_are_stable_and_budget_keeps_each_cell() -> None:
    first = EvidenceRegistry()
    second = EvidenceRegistry()
    aapl = chunk("AAPL_10-K_2025_c0001")
    msft = chunk("MSFT_10-K_2025_c0001", ticker="MSFT")
    for registry, ordered in ((first, [msft, aapl]), (second, [aapl, msft])):
        for item in ordered:
            registry.register(
                [item],
                task_id=f"T-{item['ticker']}",
                tool="search_filings",
                scores=[0.8],
                sub_question="Revenue",
            )

    assert [source.chunk_id for source in first.assign_source_ids(2)] == [
        source.chunk_id for source in second.assign_source_ids(2)
    ]
    assert {source.ticker for source in first.sources()} == {"AAPL", "MSFT"}
    assert '<source id="S1"' in first.format_context()
