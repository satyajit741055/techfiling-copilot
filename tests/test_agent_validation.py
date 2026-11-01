from __future__ import annotations

from agent.catalog import FilingCatalog
from agent.evidence import EvidenceRegistry
from agent.schemas import (
    Calculation,
    CalculationInput,
    CoverageCell,
    GroundedAnswer,
    GroundedClaim,
    ResolvedScope,
)
from agent.validation import drop_invalid_claims, normalize_excerpt, validate_grounded_answer


def setup_context() -> tuple[FilingCatalog, ResolvedScope, EvidenceRegistry]:
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
    scope = ResolvedScope(
        tickers=["AAPL"],
        fiscal_years=[2025],
        cells=[{"ticker": "AAPL", "fiscal_year": 2025}],
        coverage=[{"ticker": "AAPL", "fiscal_year": 2025}],
        route="research",
    )
    evidence = EvidenceRegistry()
    evidence.register(
        [
            {
                "chunk_id": "AAPL_10-K_2025_c0001",
                "doc_id": "AAPL_10-K_2025",
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "text": "Net sales were $100 million in 2025.",
            },
            {
                "chunk_id": "MSFT_10-K_2025_c0001",
                "doc_id": "MSFT_10-K_2025",
                "ticker": "MSFT",
                "fiscal_year": 2025,
                "text": "Revenue was $200 million in 2025.",
            },
        ],
        task_id="T1",
        tool="search_filings",
        scores=[0.9, 0.8],
    )
    evidence.assign_source_ids(2)
    return catalog, scope, evidence


def valid_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="Apple reported net sales of $100 million [S1].",
        claims=[
            GroundedClaim(
                text="Apple reported net sales of $100 million.",
                citation_ids=["S1"],
                excerpt="Net sales were $100 million in 2025.",
                ticker="AAPL",
                fiscal_year=2025,
            )
        ],
        coverage=[CoverageCell(ticker="AAPL", fiscal_year=2025, status="found")],
    )


def test_valid_grounded_answer_passes() -> None:
    catalog, scope, evidence = setup_context()
    report = validate_grounded_answer(valid_answer(), scope, evidence.source_entries(), catalog)
    assert report.is_valid
    assert report.errors == []


def test_unknown_inline_and_claim_citations_fail() -> None:
    catalog, scope, evidence = setup_context()
    answer = valid_answer().model_copy(deep=True)
    answer.answer = "Unsupported [S9]."
    answer.claims[0].citation_ids = ["S9"]

    report = validate_grounded_answer(answer, scope, evidence.source_entries(), catalog)
    assert not report.is_valid
    assert any("unknown sources" in error for error in report.errors)


def test_uncited_nonverbatim_and_cross_scope_claims_fail() -> None:
    catalog, scope, evidence = setup_context()
    uncited = valid_answer().model_copy(deep=True)
    uncited.claims[0].citation_ids = []
    cross_scope = valid_answer().model_copy(deep=True)
    cross_scope.claims[0].ticker = "MSFT"
    cross_scope.claims[0].excerpt = "invented excerpt"

    uncited_report = validate_grounded_answer(uncited, scope, evidence.source_entries(), catalog)
    cross_report = validate_grounded_answer(cross_scope, scope, evidence.source_entries(), catalog)
    assert any("no citation" in error for error in uncited_report.errors)
    assert any("scope does not match" in error for error in cross_report.errors)
    assert any("not verbatim" in error for error in cross_report.errors)


def ratio_calculation(**overrides) -> Calculation:
    values = {
        "description": "Apple sales as a share of Microsoft revenue",
        "expression": "100 / 200 * 100",
        "result": 50,
        "inputs": [
            CalculationInput(value=100, citation_id="S1"),
            CalculationInput(value=200, citation_id="S2"),
        ],
    }
    values.update(overrides)
    return Calculation.model_validate(values)


def test_unsupported_number_fails_but_calculation_result_passes() -> None:
    catalog, scope, evidence = setup_context()
    unsupported = valid_answer().model_copy(deep=True)
    unsupported.claims[0].text = "Apple's net sales of $100 million were 50% of Microsoft's."
    unsupported.answer = "Apple's net sales were 50% of Microsoft's [S1]."
    supported = unsupported.model_copy(deep=True)
    supported.calculations = [ratio_calculation()]

    failed = validate_grounded_answer(unsupported, scope, evidence.source_entries(), catalog)
    passed = validate_grounded_answer(supported, scope, evidence.source_entries(), catalog)
    assert any("unsupported numbers" in error for error in failed.errors)
    assert passed.is_valid, passed.errors


def test_calculation_result_and_literals_are_verified() -> None:
    catalog, scope, evidence = setup_context()
    entries = evidence.source_entries()
    wrong_result = valid_answer().model_copy(
        update={"calculations": [ratio_calculation(result=60)]}
    )
    uncited_literal = valid_answer().model_copy(
        update={
            "calculations": [
                ratio_calculation(
                    expression="100 + 50", result=150, inputs=[ratio_calculation().inputs[0]]
                )
            ]
        }
    )
    not_arithmetic = valid_answer().model_copy(
        update={"calculations": [ratio_calculation(expression="__import__('os')")]}
    )

    wrong_report = validate_grounded_answer(wrong_result, scope, entries, catalog)
    literal_report = validate_grounded_answer(uncited_literal, scope, entries, catalog)
    arithmetic_report = validate_grounded_answer(not_arithmetic, scope, entries, catalog)
    assert any("does not match" in error for error in wrong_report.errors)
    assert wrong_report.dropped_calculations == [ratio_calculation().description]
    assert any("uncited values (50)" in error for error in literal_report.errors)
    assert any("not simple arithmetic" in error for error in arithmetic_report.errors)


def test_number_parsing_ignores_forms_and_year_ranges_and_handles_millions() -> None:
    catalog, scope, _ = setup_context()
    evidence = EvidenceRegistry()
    evidence.register(
        [
            {
                "chunk_id": "AAPL_10-K_2025_c0001",
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "text": "Total net sales were $1,234,567 for 2024-2025 per the 10-K.",
            }
        ],
        task_id="T1",
        tool="search_filings",
        scores=[0.9],
    )
    evidence.assign_source_ids(1)
    answer = GroundedAnswer(
        answer="Per the 10-K, 2024-2025 net sales were $1,234,567.00 [S1].",
        claims=[
            GroundedClaim(
                text="Per the 10-K, 2024-2025 net sales were $1,234,567.00.",
                citation_ids=["S1"],
                excerpt="Total net sales were $1,234,567 for 2024-2025",
                ticker="AAPL",
                fiscal_year=2025,
            )
        ],
        coverage=[CoverageCell(ticker="AAPL", fiscal_year=2025, status="found")],
        calculations=[
            Calculation(
                description="Sales in dollars",
                expression="1,234,567 * 1000",
                result=1234567000,
                inputs=[CalculationInput(value=1234567, citation_id="S1")],
            )
        ],
    )

    report = validate_grounded_answer(answer, scope, evidence.source_entries(), catalog)
    assert report.is_valid, report.errors


def test_scaled_claim_numbers_match_table_units() -> None:
    catalog, scope, _ = setup_context()
    evidence = EvidenceRegistry()
    evidence.register(
        [
            {
                "chunk_id": "AAPL_10-K_2025_c0001",
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "units": "In millions, except per-share amounts",
                "text": "| Total net sales | 94,830 | 88,000 |",
            }
        ],
        task_id="T1",
        tool="read_table",
        scores=[0.9],
    )
    evidence.assign_source_ids(1)

    def answer(text: str) -> GroundedAnswer:
        return GroundedAnswer(
            answer=f"{text} [S1]",
            claims=[
                GroundedClaim(
                    text=text,
                    citation_ids=["S1"],
                    excerpt="Total net sales | 94,830",
                    ticker="AAPL",
                    fiscal_year=2025,
                )
            ],
            coverage=[CoverageCell(ticker="AAPL", fiscal_year=2025, status="found")],
        )

    entries = evidence.source_entries()
    for supported in ("Net sales were $94.83 billion.", "Net sales were about $94.8 billion."):
        report = validate_grounded_answer(answer(supported), scope, entries, catalog)
        assert report.is_valid, report.errors
    wrong = validate_grounded_answer(
        answer("Net sales were $95.9 billion."), scope, entries, catalog
    )
    assert any("unsupported numbers ($95.9)" in error for error in wrong.errors)
    repeated = validate_grounded_answer(
        answer("Net sales were 94.83 billion, not 94.83 thousand."), scope, entries, catalog
    )
    assert any("unsupported numbers (94.83)" in error for error in repeated.errors)


def test_scaled_rounding_is_capped_at_five_percent() -> None:
    catalog, scope, _ = setup_context()
    evidence = EvidenceRegistry()
    evidence.register(
        [
            {
                "chunk_id": "AAPL_10-K_2025_c0001",
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "units": "In millions",
                "text": "| Total net sales | 94,830 |\n| Other income | 960 |\n| Interest | 600 |",
            }
        ],
        task_id="T1",
        tool="read_table",
        scores=[0.9],
    )
    evidence.assign_source_ids(1)
    entries = evidence.source_entries()

    def report(text: str, excerpt: str) -> list[str]:
        claim = GroundedClaim(
            text=text, citation_ids=["S1"], excerpt=excerpt, ticker="AAPL", fiscal_year=2025
        )
        answer = GroundedAnswer(
            answer=f"{text} [S1]",
            claims=[claim],
            coverage=[CoverageCell(ticker="AAPL", fiscal_year=2025, status="found")],
        )
        return validate_grounded_answer(answer, scope, entries, catalog).errors

    assert report("Net sales were $95 billion.", "Total net sales | 94,830") == []
    assert report("Other income was $1.0 billion.", "Other income | 960") == []
    assert report("Other income was $1 billion.", "Other income | 960") == []
    assert any(
        "unsupported numbers ($1)" in error
        for error in report("Interest was $1 billion.", "Interest | 600")
    )


def test_uncited_prose_numbers_and_pending_coverage_fail() -> None:
    catalog, scope, evidence = setup_context()
    answer = valid_answer().model_copy(deep=True)
    answer.answer = "Apple reported net sales of $100 million and margin of 46.2% [S1]."
    answer.coverage[0].status = "pending"

    report = validate_grounded_answer(answer, scope, evidence.source_entries(), catalog)
    assert any("not backed by a valid claim: 46.2%" in error for error in report.errors)
    assert any("non-final status" in error for error in report.errors)


def test_excerpt_ignores_missing_pipe_at_table_row_break() -> None:
    source = "| R&D | 34,550 | 10% | 29,915 |\n| Percentage of total net sales | 8% |"
    excerpt = "R&D | 34,550 | 10% | 29,915\n| Percentage of total net sales"
    assert normalize_excerpt(excerpt) in normalize_excerpt(source)


def test_long_excerpt_is_truncated_not_rejected() -> None:
    claim = GroundedClaim(text="x", excerpt="a" * 400, ticker="AAPL", fiscal_year=2025)
    assert len(claim.excerpt) == 300


def test_coverage_must_be_complete_and_catalog_consistent() -> None:
    catalog, scope, evidence = setup_context()
    missing = valid_answer().model_copy(deep=True)
    missing.coverage = []
    wrong = valid_answer().model_copy(deep=True)
    wrong.coverage[0].status = "not_in_corpus"

    missing_report = validate_grounded_answer(missing, scope, evidence.source_entries(), catalog)
    wrong_report = validate_grounded_answer(wrong, scope, evidence.source_entries(), catalog)
    assert any("missing required cells" in error for error in missing_report.errors)
    assert any("incorrectly marks" in error for error in wrong_report.errors)


def test_invalid_claims_are_dropped_and_answer_abstains() -> None:
    answer = valid_answer()
    cleaned = drop_invalid_claims(
        answer,
        validate_grounded_answer(
            answer.model_copy(update={"claims": []}),
            ResolvedScope(
                coverage=[CoverageCell(ticker="AAPL", fiscal_year=2025)], route="research"
            ),
            {},
            FilingCatalog([]),
        ).model_copy(update={"dropped_claims": [answer.claims[0].text]}),
    )

    assert cleaned.claims == []
    assert cleaned.answer is None
    assert cleaned.insufficient_context
    assert cleaned.coverage[0].status == "not_disclosed"


def test_drop_rebuilds_prose_and_coverage_from_required_cells() -> None:
    catalog, scope, evidence = setup_context()
    answer = valid_answer().model_copy(deep=True)
    answer.answer = "Apple reported net sales of $100 million [S1], and $999 million in services."
    answer.coverage = []
    answer.calculations = [ratio_calculation(result=60)]
    report = validate_grounded_answer(answer, scope, evidence.source_entries(), catalog)
    assert not report.is_valid

    cleaned = drop_invalid_claims(answer, report, scope.coverage)
    assert cleaned.answer == "Apple reported net sales of $100 million. [S1]"
    assert cleaned.calculations == []
    assert [(cell.ticker, cell.status) for cell in cleaned.coverage] == [("AAPL", "found")]
    assert validate_grounded_answer(cleaned, scope, evidence.source_entries(), catalog).is_valid
