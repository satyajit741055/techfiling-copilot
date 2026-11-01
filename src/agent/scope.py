"""Pure, deterministic scope resolution for filing questions."""

from __future__ import annotations

from collections.abc import Sequence

from .catalog import FilingCatalog
from .schemas import (
    Analysis,
    Clarification,
    CoverageCell,
    ResolvedScope,
    Route,
    ScopeCell,
    TurnSummary,
    UIFilters,
)

_REFUSE_INTENTS = {"general_knowledge", "out_of_scope"}
_ONBOARD_INTENTS = {"greeting", "capabilities"}
_SIMPLE_TYPES = {"single_fact", "qualitative"}
_COLLAPSIBLE_TYPES = {"matrix", "cross_company"}


def _unique[T](values: Sequence[T]) -> list[T]:
    return list(dict.fromkeys(values))


def _company_scope(
    analysis: Analysis,
    catalog: FilingCatalog,
    ui_filters: UIFilters | None,
    turns: Sequence[TurnSummary],
) -> tuple[list[str], list[str], list[str]]:
    assumptions: list[str] = []
    mentions = ui_filters.tickers if ui_filters and ui_filters.tickers else analysis.companies
    tickers: list[str] = []
    unknown: list[str] = []
    for mention in mentions:
        ticker = catalog.resolve_company(mention)
        if ticker:
            tickers.append(ticker)
        else:
            unknown.append(mention)
    tickers = _unique(tickers)
    unknown = _unique(unknown)

    if not mentions and analysis.scope == "corpus_wide":
        tickers = catalog.tickers
        assumptions.append(
            f"Corpus-wide question; used all indexed companies ({', '.join(tickers)})"
        )
    elif (
        not mentions and (analysis.scope == "context_dependent" or analysis.uses_history) and turns
    ):
        tickers = [ticker for ticker in turns[-1].tickers if ticker in catalog.tickers]
        if tickers:
            assumptions.append(f"Reused companies from the previous turn ({', '.join(tickers)})")
    return tickers, unknown, assumptions


def _requested_years(analysis: Analysis, available: list[int]) -> tuple[list[int], bool]:
    years = analysis.years
    if years.explicit:
        return _unique(years.explicit), True
    if years.range:
        return [year for year in range(years.range.gte, years.range.lte + 1)], True
    if years.relative == "latest":
        return available[-1:] if available else [], True
    if years.relative == "last_year":
        return available[-2:-1] if len(available) >= 2 else [], True
    if years.relative == "last_n":
        return available[-(years.n or 1) :], True
    return available[-1:] if available else [], False


def _clarification(analysis: Analysis, catalog: FilingCatalog) -> Clarification:
    dimension = analysis.ambiguity.dimension or "company"
    if dimension == "company":
        return Clarification(
            dimension="company",
            question="Which company should I research?",
            options=catalog.tickers,
        )
    return Clarification(
        dimension=dimension,
        question=analysis.ambiguity.reason or f"Please clarify the {dimension}.",
        options=analysis.ambiguity.options,
    )


def resolve_scope(
    analysis: Analysis,
    catalog: FilingCatalog,
    *,
    ui_filters: UIFilters | None = None,
    turns: Sequence[TurnSummary] = (),
    clarifications_this_turn: int = 0,
    max_tasks: int = 10,
) -> ResolvedScope:
    """Resolve LLM analysis into catalog-backed cells, assumptions, and a graph route."""
    if analysis.intent in _REFUSE_INTENTS or analysis.scope in _REFUSE_INTENTS:
        return ResolvedScope(route="refuse")
    if analysis.intent in _ONBOARD_INTENTS:
        return ResolvedScope(route="onboard")

    tickers, unknown, assumptions = _company_scope(analysis, catalog, ui_filters, turns)
    ui_filters_set = bool(ui_filters and (ui_filters.tickers or ui_filters.fiscal_years))
    if not tickers and not unknown and (ui_filters_set or clarifications_this_turn > 0):
        tickers = catalog.tickers
        assumptions.append(
            f"No company selected after clarification; used all indexed companies "
            f"({', '.join(tickers)})"
        )
    should_ask = (
        not ui_filters_set
        and clarifications_this_turn == 0
        and (
            (not tickers and not unknown)
            or (
                analysis.ambiguity.is_ambiguous
                and analysis.ambiguity.dimension in {"company", "metric"}
            )
        )
    )
    if should_ask:
        return ResolvedScope(
            tickers=tickers,
            assumptions=assumptions,
            not_in_corpus=unknown,
            route="clarify",
            clarification=_clarification(analysis, catalog),
        )
    if not tickers:
        return ResolvedScope(
            assumptions=assumptions,
            not_in_corpus=unknown,
            route="refuse",
        )

    cells: list[ScopeCell] = []
    coverage: list[CoverageCell] = []
    all_years: list[int] = []
    used_default_year = False
    for ticker in tickers:
        available = catalog.years_by_ticker(ticker)
        if ui_filters and ui_filters.fiscal_years:
            requested, explicit = _unique(ui_filters.fiscal_years), True
        else:
            requested, explicit = _requested_years(analysis, available)
        used_default_year |= not explicit
        all_years.extend(requested)
        for year in requested:
            if catalog.has(ticker, year):
                cells.append(ScopeCell(ticker=ticker, fiscal_year=year))
                coverage.append(CoverageCell(ticker=ticker, fiscal_year=year))
            else:
                coverage.append(
                    CoverageCell(
                        ticker=ticker,
                        fiscal_year=year,
                        status="not_in_corpus",
                        detail=f"Available years: {', '.join(map(str, available))}",
                        available_years=available,
                    )
                )

    if used_default_year and cells:
        selected = ", ".join(f"{cell.ticker} FY{cell.fiscal_year}" for cell in cells)
        assumptions.append(f"No year specified; used latest indexed 10-K ({selected})")
    if unknown:
        assumptions.append(f"Not indexed: {', '.join(unknown)}")
    if not cells:
        return ResolvedScope(
            tickers=tickers,
            fiscal_years=sorted(set(all_years)),
            assumptions=assumptions,
            coverage=coverage,
            not_in_corpus=unknown,
            route="refuse",
        )

    sub_question_count = max(1, len(analysis.sub_questions))
    task_count = len(cells) * sub_question_count
    collapse_years = False
    if task_count > max_tasks:
        if analysis.question_type in _COLLAPSIBLE_TYPES or analysis.scope == "corpus_wide":
            collapsed_count = len(tickers) * sub_question_count
            if collapsed_count <= max_tasks:
                collapse_years = True
            else:
                return ResolvedScope(
                    tickers=tickers,
                    fiscal_years=sorted(set(all_years)),
                    cells=cells,
                    assumptions=assumptions,
                    coverage=coverage,
                    not_in_corpus=unknown,
                    route="clarify",
                    clarification=Clarification(
                        dimension="other",
                        question="This request is too broad. Please narrow the companies, years, or topics.",
                    ),
                )
        else:
            return ResolvedScope(
                tickers=tickers,
                fiscal_years=sorted(set(all_years)),
                cells=cells,
                assumptions=assumptions,
                coverage=coverage,
                not_in_corpus=unknown,
                route="clarify",
                clarification=Clarification(
                    dimension="other",
                    question="This request is too broad. Please narrow the companies, years, or topics.",
                ),
            )

    route: Route = (
        "simple_rag"
        if len(cells) == 1 and analysis.question_type in _SIMPLE_TYPES and sub_question_count == 1
        else "research"
    )
    return ResolvedScope(
        tickers=tickers,
        fiscal_years=sorted(set(all_years)),
        cells=cells,
        assumptions=assumptions,
        coverage=coverage,
        not_in_corpus=unknown,
        route=route,
        collapse_years=collapse_years,
    )
