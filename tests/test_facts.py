"""
Financial facts (M4): building chunk-linked records, metric matching, as-reported lookups
with restatements, the calculator, and the Postgres round trip (HISTORY_TEST_DB_URL).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import psycopg
import pytest

from facts import FactRecord, build_facts, chunk_run_for_collection
from facts.build import _Filing, fiscal_year_of, normalize_unit
from facts.calculator import CalcInput, cagr_pct, change, growth_pct, margin_pct, total
from facts.metrics import company_words, match_metrics, metric, normalize_topic, scan_question
from facts.store import FactsStore
from persistence.facts import FactsLoad, read_facts, replace_facts
from persistence.migrations import apply_migrations
from tests.test_history import _temp_schema, live

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "data" / "parsed"
CHUNK_RUN = ROOT / "data" / "chunks" / "2026-10-31_19-00-27"
real_data = pytest.mark.skipif(
    not (PARSED / "MSFT_10-K_2024.facts.jsonl").exists() or not CHUNK_RUN.exists(),
    reason="ingestion output (data/parsed, data/chunks) not present",
)


# =============================================================================
# BUILD (synthetic filing)
# =============================================================================


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _fact(fact_id: str, concept: str, shown: float, start: str | None, end: str, **extra) -> dict:
    row = {
        "id": fact_id,
        "concept": concept,
        "context": "c",
        "unit": "U_USD",
        "scale": 6,
        "sign": 1,
        "shown": shown,
        "value": shown * 1e6,
        "period_start": start,
        "period_end": end,
        "instant": start is None,
        "dims": {},
        "doc_id": "ABC_10-K_2024",
    }
    return {**row, **extra}


@pytest.fixture()
def filing(tmp_path: Path) -> tuple[Path, Path]:
    parsed, chunks = tmp_path / "parsed", tmp_path / "chunks"
    parsed.mkdir()
    chunks.mkdir()
    (parsed / "ABC_10-K_2024.document.json").write_text(
        json.dumps(
            {
                "doc_id": "ABC_10-K_2024",
                "ticker": "abc",
                "fiscal_year": 2024,
                "period_end": "2024-06-30",
            }
        ),
        encoding="utf-8",
    )
    revenue = "us-gaap:Revenues"
    _write_jsonl(
        parsed / "ABC_10-K_2024.facts.jsonl",
        [
            _fact("f1", revenue, 120.0, "2023-07-01", "2024-06-30"),
            _fact("f2", revenue, 100.0, "2022-07-01", "2023-06-30"),
            # The same figure rounded in prose: the table value wins.
            _fact("f3", revenue, 0.1, "2023-07-01", "2024-06-30", scale=9, value=1e8),
            _fact("f4", revenue, 30.0, "2024-04-01", "2024-06-30"),  # quarter: dropped
            _fact(
                "f5",
                revenue,
                70.0,
                "2023-07-01",
                "2024-06-30",
                dims={"srt:ProductOrServiceAxis": "x"},
            ),  # segment: dropped
            _fact("f6", "dei:EntityPublicFloat", 5.0, None, "2023-12-31"),  # dei: dropped
            _fact("f7", "us-gaap:Assets", 900.0, None, "2024-06-30"),  # not in a table
            _fact("f8", "us-gaap:OtherExpense", 0.0, "2023-07-01", "2024-06-30"),  # dash
        ],
    )
    cell = lambda ids: {"text": "", "ctype": "NUMBER", "value": None, "fact_ids": ids}  # noqa: E731
    table = {
        "table_id": "ABC_10-K_2024_t001",
        "kind": "financial",
        "rows": [
            {"label": "Revenue", "path": ["Revenue"], "row_type": "heading", "cells": []},
            {
                "label": "Total revenue",
                "path": ["Revenue", "Total revenue"],
                "row_type": "data",
                "cells": [cell(["f1"]), cell(["f2"])],
            },
            {
                "label": "Other",
                "path": ["Other"],
                "row_type": "data",
                "cells": [cell(["f8"]), cell([])],
            },
        ],
    }
    _write_jsonl(parsed / "ABC_10-K_2024.elements.jsonl", [{"order": 7, "table": table}])
    _write_jsonl(
        chunks / "ABC_10-K_2024.chunks.jsonl",
        [
            {
                "chunk_id": "ABC_10-K_2024_c0001",
                "table_id": "ABC_10-K_2024_t001",
                "text": "**INCOME** (In millions)\n\n| Year | 2024 | 2023 |\n|---|---|---|\n"
                "| Revenue > Total revenue | 120 | 100 |\n| Other | — | 4 |",
            }
        ],
    )
    return parsed, chunks


def test_build_links_table_rows_and_assigns_period_fiscal_years(filing):
    records = {record.fact_id: record for record in build_facts(*filing)}
    assert set(records) == {"f1", "f2", "f7", "f8"}
    current, prior = records["f1"], records["f2"]
    assert (current.ticker, current.filing_fiscal_year, current.fiscal_year) == ("ABC", 2024, 2024)
    assert prior.fiscal_year == 2023 and prior.filing_fiscal_year == 2024
    assert current.unit == "USD" and current.value == 120e6 and current.shown == 120
    assert current.chunk_id == "ABC_10-K_2024_c0001"
    assert current.excerpt == prior.excerpt == "Revenue > Total revenue | 120 | 100"
    assert records["f8"].linked  # zero printed as a dash
    assert not records["f7"].linked and records["f7"].instant


def test_fiscal_year_of_uses_matching_or_offset_filing():
    filings = [
        _Filing("X_2021", "X", 2021, date(2021, 9, 25)),
        _Filing("X_2022", "X", 2022, date(2022, 9, 24)),
    ]
    assert fiscal_year_of(date(2022, 9, 24), filings) == 2022
    assert fiscal_year_of(date(2022, 9, 30), filings) == 2022  # within 10 days
    assert fiscal_year_of(date(2019, 9, 28), filings) == 2019


@pytest.mark.parametrize(
    ("raw", "unit"),
    [
        ("usd", "USD"),
        ("U_iso4217USD", "USD"),
        ("U_UnitedStatesOfAmericaDollarsShare", "USD/share"),
        ("usdPerShare", "USD/share"),
        ("U_xbrlishares", "shares"),
        ("U_pure", "pure"),
        (None, "unknown"),
    ],
)
def test_normalize_unit(raw, unit):
    assert normalize_unit(raw) == unit


def test_chunk_run_for_collection(tmp_path: Path):
    for name, collection in (("run_a", "c1"), ("run_b", "c2")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_manifest.json").write_text(
            json.dumps({"qdrant_collection": {"name": collection}}), encoding="utf-8"
        )
    assert chunk_run_for_collection(tmp_path, "c2") == tmp_path / "run_b"
    assert chunk_run_for_collection(tmp_path, "nope") is None


# =============================================================================
# METRICS
# =============================================================================


def test_normalize_topic_drops_possessives_filler_and_punctuation():
    assert normalize_topic("Apple's total net sales") == "net sales"
    assert normalize_topic("Apple Inc.'s R&D spending") == "r&d spending"
    assert normalize_topic("Total R&D expense") == "r&d expense"
    assert normalize_topic("the reported operating income (loss)") == "operating income loss"


@pytest.mark.parametrize(
    ("topics", "keys", "growth", "margin"),
    [
        (["total net sales"], ["revenue"], False, False),
        (["R&D expense", "net income"], ["rd_expense", "net_income"], False, False),
        (["revenue growth"], ["revenue"], True, False),
        (["growth in operating income"], ["operating_income"], True, False),
        (["operating margin"], ["operating_income"], False, True),
        (["diluted EPS"], ["eps_diluted"], False, False),
        (["capital expenditures"], ["capex"], False, False),
    ],
)
def test_match_metrics(topics, keys, growth, margin):
    match = match_metrics(topics)
    assert [item.key for item in match.metrics] == keys
    assert not match.unmatched
    assert (match.growth, match.margin) == (growth, margin)


@pytest.mark.parametrize(
    "topic",
    [
        "iPhone revenue",
        "cloud revenue",
        "risk factors",
        "supply chain risks",
        "revenue by geography",
    ],
)
def test_segment_and_qualitative_topics_do_not_match(topic):
    match = match_metrics([topic])
    assert match.metrics == [] and match.unmatched == [topic]


def test_mixed_topics_report_the_unmatched_part():
    match = match_metrics(["revenue", "main risk factors"])
    assert [item.key for item in match.metrics] == ["revenue"]
    assert match.unmatched == ["main risk factors"]


def test_company_words_and_calculation_only_topics():
    ignore = company_words("Tesla", "TSLA", "Tesla, Inc.")
    match = match_metrics(["Tesla revenue", "compound annual growth rate"], ignore)
    assert [item.key for item in match.metrics] == ["revenue"]
    assert not match.unmatched and match.growth
    assert match_metrics(["Tesla revenue"]).unmatched == ["Tesla revenue"]


GOLDEN_FILE = Path(__file__).resolve().parents[1] / "eval" / "facts" / "numeric_golden.jsonl"
COMPANIES = company_words(
    "Apple", "AAPL", "Apple Inc.", "Microsoft", "MSFT", "Microsoft Corporation", "Tesla", "TSLA"
)


@pytest.mark.parametrize(
    "case",
    [json.loads(line) for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines()],
    ids=lambda case: case["id"],
)
def test_question_scan_on_the_golden_set(case):
    """Without analysis topics, every purely numeric golden question resolves to exactly
    its metrics; segment questions match nothing (MSFT SG&A matches, then has no fact)."""
    match = scan_question(case["question"], COMPANIES)
    expected = list(dict.fromkeys(item["metric"] for item in case["facts"]))
    if case["category"] == "margin":
        expected = [key for key in expected if key != "revenue"]
    if case["category"] == "not_covered":
        expected = ["sga"]
    assert [item.key for item in match.metrics] == expected
    assert match.margin is (case["category"] == "margin")


@pytest.mark.parametrize(
    "question",
    [
        "What was Apple's revenue and what are its main risk factors?",
        "What were Apple's iPhone net sales in fiscal 2025?",
        "How did Microsoft's cloud revenue change?",
        "Why did Tesla's net income fall in 2025?",
    ],
)
def test_question_scan_matches_nothing_when_other_words_remain(question):
    match = scan_question(question, COMPANIES)
    assert match.metrics == [] and match.unmatched == [question]


# =============================================================================
# STORE
# =============================================================================


def _record(
    fiscal_year: int, filing_year: int, shown: float, concept: str = "us-gaap:Revenues", **extra
) -> FactRecord:
    base = FactRecord(
        doc_id=f"ABC_10-K_{filing_year}",
        fact_id=f"f{fiscal_year}{filing_year}{concept[-3:]}",
        ticker="ABC",
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
        chunk_id=f"ABC_10-K_{filing_year}_c0001",
        table_id="t",
        row_label="Revenue",
        excerpt=f"Revenue | {shown:,.0f}",
    )
    return replace(base, **extra)


def test_lookup_returns_the_as_reported_value_and_flags_a_restatement():
    store = FactsStore(
        [
            _record(2023, 2023, 100),
            _record(2023, 2024, 100),
            _record(2023, 2025, 104),
            _record(2024, 2024, 120),
        ]
    )
    found = store.lookup("abc", 2023, metric("revenue"))
    assert found is not None
    assert found.record.filing_fiscal_year == 2023 and found.record.shown == 100
    assert found.restated is not None and found.restated.filing_fiscal_year == 2025
    assert store.lookup("ABC", 2024, metric("revenue")).restated is None


def test_lookup_needs_a_linked_own_filing_fact_and_falls_through_aliases():
    unlinked = _record(2024, 2024, 120, chunk_id=None, excerpt=None)
    later_column = _record(2024, 2025, 120)
    assert FactsStore([unlinked, later_column]).lookup("ABC", 2024, metric("revenue")) is None
    alias = _record(
        2024, 2024, 120, concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    )
    store = FactsStore([alias])
    assert store.lookup("ABC", 2024, metric("revenue")).record.concept == alias.concept


def test_lookup_rejects_a_unit_mismatch_and_indexes_facts_by_chunk():
    shares = _record(2024, 2024, 7.0, concept="us-gaap:EarningsPerShareDiluted", unit="shares")
    store = FactsStore([shares, _record(2024, 2024, 120)])
    assert store.lookup("ABC", 2024, metric("eps_diluted")) is None
    assert len(store.facts_in_chunk("ABC_10-K_2024_c0001")) == 2


# =============================================================================
# CALCULATOR
# =============================================================================

OLD = CalcInput("FY2023 revenue", 211915, "c1")
NEW = CalcInput("FY2024 revenue", 245122, "c1")


def test_change_and_growth_use_printed_values():
    assert change(OLD, NEW, "d").expression == "245122 - 211915"
    assert change(OLD, NEW, "d").result == 33207
    growth = growth_pct(OLD, NEW, "d")
    assert growth.expression == "(245122 - 211915) / 211915 * 100"
    assert growth.result == pytest.approx(15.6699, rel=1e-4)
    assert growth_pct(CalcInput("x", 0, "c"), NEW, "d") is None


def test_cagr_margin_and_sum():
    cagr = cagr_pct(CalcInput("FY2021", 168088, "c0"), NEW, 3, "d")
    assert cagr.expression == "((245122 / 168088) ** (1 / 3) - 1) * 100"
    assert cagr.result == pytest.approx(13.40, abs=0.01)
    margin = margin_pct(CalcInput("op income", 109433, "c1"), NEW, "d")
    assert margin.result == pytest.approx(44.64, abs=0.01)
    assert total([OLD, NEW], "d").expression == "211915 + 245122"
    assert total([OLD], "d") is None


def test_signed_inputs_keep_losses_negative():
    loss = CalcInput("FY2019 net loss", 862, "c", signed=-862)
    assert change(loss, CalcInput("FY2020 net income", 721, "c"), "d").expression == "721 - -862"


# =============================================================================
# REAL DATA (ingestion output)
# =============================================================================


@pytest.fixture(scope="module")
def real_store() -> FactsStore:
    return FactsStore(build_facts(PARSED, CHUNK_RUN))


@real_data
def test_real_data_known_values_and_rows(real_store):
    msft = real_store.lookup("MSFT", 2024, metric("revenue"))
    assert msft.record.shown == 245122 and msft.record.value == 245_122_000_000
    assert msft.record.excerpt == "Revenue > Total revenue | 245,122 | 211,915 | 198,270"
    aapl = real_store.lookup("AAPL", 2025, metric("revenue"))
    assert aapl.record.shown == 416161
    tsla = real_store.lookup("TSLA", 2023, metric("revenue"))
    assert tsla.record.shown == 96773 and "Total revenues" in tsla.record.row_label
    assert real_store.lookup("MSFT", 2024, metric("eps_diluted")).record.shown == 11.8


@real_data
def test_real_data_curated_metric_coverage(real_store):
    gaps = real_store.coverage(["AAPL", "MSFT", "TSLA"], range(2021, 2026))
    msft = [f"MSFT FY{year}" for year in range(2021, 2026)]
    tsla = [f"TSLA FY{year}" for year in range(2021, 2026)]
    expected = {
        "sga": msft,
        "operating_expenses": msft[:4],
        "dividends_paid": tsla,
        "share_repurchases": tsla,
    }
    assert {key: cells for key, cells in gaps.items() if cells} == expected
    assert real_store.linked / len(real_store) >= 0.85


@real_data
def test_real_data_flags_the_tesla_stock_split_restatement(real_store):
    eps = real_store.lookup("TSLA", 2021, metric("eps_diluted"))
    assert eps.record.shown == 4.9
    assert eps.restated is not None and eps.restated.shown == pytest.approx(1.63)


# =============================================================================
# POSTGRES ROUND TRIP (HISTORY_TEST_DB_URL)
# =============================================================================


@live
def test_live_replace_and_read_facts(filing):
    records = build_facts(*filing)
    with _temp_schema() as conninfo:
        with psycopg.connect(conninfo, autocommit=True) as conn:
            apply_migrations(conn)
        assert read_facts(conninfo) == ([], None)
        load = FactsLoad(collection="c1", chunk_run="run", facts=len(records), linked=3)
        replace_facts(conninfo, records, load)
        replace_facts(conninfo, records, load.model_copy(update={"collection": "c2"}))
        stored, latest = read_facts(conninfo)
    assert sorted(stored, key=lambda r: r.fact_id) == sorted(records, key=lambda r: r.fact_id)
    assert latest is not None and latest.collection == "c2"
