"""Offline tests for agent state, filing catalog, and deterministic scope resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import RESET, FilingCatalog, merge_or_reset, resolve_scope  # noqa: E402
from agent.schemas import (  # noqa: E402
    Ambiguity,
    Analysis,
    TaskResult,
    TurnSummary,
    UIFilters,
    YearRange,
    YearSelection,
)


@pytest.fixture
def catalog() -> FilingCatalog:
    return FilingCatalog(
        {
            "doc_id": f"{ticker}_10-K_{year}",
            "ticker": ticker,
            "company": company,
            "fiscal_year": year,
            "period_end": f"{year}-09-30",
        }
        for ticker, company in (("AAPL", "Apple Inc."), ("MSFT", "MICROSOFT CORPORATION"))
        for year in range(2023, 2026)
    )


def analysis(**overrides) -> Analysis:
    values = {
        "intent": "filing_question",
        "scope": "specific",
        "question_type": "single_fact",
        "companies": ["AAPL"],
        "sub_questions": ["What was revenue?"],
    }
    values.update(overrides)
    return Analysis.model_validate(values)


def test_catalog_resolves_aliases_and_describes_coverage(catalog):
    assert catalog.tickers == ["AAPL", "MSFT"]
    assert catalog.years_by_ticker("aapl") == [2023, 2024, 2025]
    assert catalog.latest_year("MSFT") == 2025
    assert catalog.resolve_company("Apple Inc.") == "AAPL"
    assert catalog.resolve_company("microsoft") == "MSFT"
    assert catalog.describe() == "AAPL, MSFT · FY2023–FY2025"


def test_catalog_rejects_duplicate_company_year():
    records = [
        {"doc_id": "a", "ticker": "AAPL", "company": "Apple", "fiscal_year": 2025},
        {"doc_id": "b", "ticker": "AAPL", "company": "Apple", "fiscal_year": 2025},
    ]
    with pytest.raises(ValueError, match="Duplicate filing"):
        FilingCatalog(records)


def test_results_reducer_appends_and_resets():
    first = TaskResult(
        task_id="t1",
        ticker="AAPL",
        fiscal_year=2025,
        sub_question="Revenue",
        status="found",
    )
    second = first.model_copy(update={"task_id": "t2"})
    assert merge_or_reset([first], [second]) == [first, second]
    assert merge_or_reset([first], RESET) == []


def test_refuse_and_onboard_routes_do_not_create_cells(catalog):
    refused = resolve_scope(
        analysis(intent="general_knowledge", scope="general_knowledge"), catalog
    )
    onboarded = resolve_scope(analysis(intent="greeting", companies=[]), catalog)
    assert refused.route == "refuse" and refused.cells == []
    assert onboarded.route == "onboard" and onboarded.cells == []


def test_ambiguous_company_clarifies_once_then_defaults_to_all(catalog):
    ambiguous = analysis(
        companies=[],
        ambiguity=Ambiguity(
            is_ambiguous=True,
            dimension="company",
            reason="Company missing",
        ),
    )
    first = resolve_scope(ambiguous, catalog)
    second = resolve_scope(ambiguous, catalog, clarifications_this_turn=1)
    assert first.route == "clarify"
    assert first.clarification and first.clarification.options == ["AAPL", "MSFT"]
    assert second.route == "research"
    assert second.tickers == ["AAPL", "MSFT"]
    assert "after clarification" in second.assumptions[0]


def test_ui_filters_override_analysis_and_skip_clarification(catalog):
    ambiguous = analysis(
        companies=[],
        ambiguity={"is_ambiguous": True, "dimension": "company", "reason": "missing"},
    )
    scope = resolve_scope(
        ambiguous,
        catalog,
        ui_filters=UIFilters(tickers=["Microsoft"], fiscal_years=[2024]),
    )
    assert scope.route == "simple_rag"
    assert [(cell.ticker, cell.fiscal_year) for cell in scope.cells] == [("MSFT", 2024)]

    years_only = resolve_scope(
        ambiguous,
        catalog,
        ui_filters=UIFilters(fiscal_years=[2024]),
    )
    assert years_only.route == "research"
    assert years_only.tickers == ["AAPL", "MSFT"]


def test_history_resolves_context_dependent_company(catalog):
    current = analysis(companies=[], scope="context_dependent", uses_history=True)
    previous = TurnSummary(question="AAPL revenue", tickers=["AAPL"], fiscal_years=[2024])
    scope = resolve_scope(current, catalog, turns=[previous])
    assert scope.tickers == ["AAPL"]
    assert scope.cells[0].fiscal_year == 2025
    assert "previous turn" in scope.assumptions[0]


def test_missing_year_uses_latest_and_states_assumption(catalog):
    scope = resolve_scope(analysis(), catalog)
    assert scope.route == "simple_rag"
    assert scope.fiscal_years == [2025]
    assert scope.assumptions == ["No year specified; used latest indexed 10-K (AAPL FY2025)"]


def test_explicit_and_relative_years_resolve_against_catalog(catalog):
    explicit = resolve_scope(
        analysis(
            question_type="trend",
            years=YearSelection(range=YearRange(gte=2023, lte=2024)),
        ),
        catalog,
    )
    relative = resolve_scope(
        analysis(question_type="trend", years=YearSelection(relative="last_n", n=2)),
        catalog,
    )
    assert explicit.fiscal_years == [2023, 2024]
    assert relative.fiscal_years == [2024, 2025]
    assert not relative.assumptions


def test_unknown_company_and_year_are_reported_without_losing_valid_scope(catalog):
    scope = resolve_scope(
        analysis(companies=["AAPL", "NVIDIA"], years={"explicit": [2022, 2025]}), catalog
    )
    assert scope.route == "simple_rag"
    assert scope.not_in_corpus == ["NVIDIA"]
    assert scope.cells[0].fiscal_year == 2025
    assert [(cell.fiscal_year, cell.status) for cell in scope.coverage] == [
        (2022, "not_in_corpus"),
        (2025, "pending"),
    ]


def test_corpus_wide_scope_uses_all_tickers(catalog):
    scope = resolve_scope(
        analysis(
            scope="corpus_wide",
            companies=[],
            question_type="cross_company",
            years={"explicit": [2025]},
        ),
        catalog,
    )
    assert scope.route == "research"
    assert scope.tickers == ["AAPL", "MSFT"]
    assert len(scope.cells) == 2
    assert "all indexed companies" in scope.assumptions[0]


def test_fanout_cap_collapses_matrix_by_year(catalog):
    scope = resolve_scope(
        analysis(
            scope="corpus_wide",
            companies=[],
            question_type="matrix",
            years={"explicit": [2023, 2024, 2025]},
            sub_questions=["Revenue", "Margin"],
        ),
        catalog,
        max_tasks=4,
    )
    assert scope.route == "research"
    assert scope.collapse_years is True


def test_fanout_cap_clarifies_non_collapsible_request(catalog):
    scope = resolve_scope(
        analysis(
            question_type="trend",
            years={"explicit": [2023, 2024, 2025]},
            sub_questions=["Revenue", "Margin"],
        ),
        catalog,
        max_tasks=2,
    )
    assert scope.route == "clarify"
    assert scope.clarification and "too broad" in scope.clarification.question


def test_fanout_cap_narrows_instead_of_clarifying_twice(catalog):
    scope = resolve_scope(
        analysis(
            question_type="trend",
            years={"explicit": [2023, 2024, 2025]},
            sub_questions=["Revenue", "Margin"],
        ),
        catalog,
        clarifications_this_turn=1,
        max_tasks=4,
    )
    assert scope.route == "research"
    assert [(cell.ticker, cell.fiscal_year) for cell in scope.cells] == [
        ("AAPL", 2024),
        ("AAPL", 2025),
    ]
    assert [cell.fiscal_year for cell in scope.coverage] == [2024, 2025]
    assert scope.fiscal_years == [2024, 2025]
    assert any("search limit" in assumption for assumption in scope.assumptions)
