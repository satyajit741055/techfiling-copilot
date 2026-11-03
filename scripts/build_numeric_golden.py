"""Build eval/facts/numeric_golden.jsonl (the numeric eval set) from the facts store.

Usage:
    uv run python scripts/build_numeric_golden.py [--out PATH] [--db-url URL]

Expected values come from the financial_facts table (fill it with scripts/load_facts.py):
each figure is the value reported in that fiscal year's own 10-K, with the table row
that prints it ("row"), the other year columns of that row ("other_columns", used to
detect wrong-period answers) and any later restatement. Percentages (growth, CAGR,
margin) are computed from the shown values. Prints every case with its row so the
values can be checked against the filings. Edit the case list below to change the set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facts.metrics import metric  # noqa: E402
from facts.store import FactsStore  # noqa: E402


def _fact(store: FactsStore, ticker: str, year: int, key: str) -> dict[str, Any]:
    found = store.lookup(ticker, year, metric(key))
    if found is None:
        raise SystemExit(f"No citable reported figure for {ticker} FY{year} {key}")
    record = found.record
    out: dict[str, Any] = {
        "ticker": ticker,
        "fiscal_year": year,
        "metric": key,
        "value": record.shown,
        "unit": record.unit,
        "scale": record.scale,
        "other_columns": sorted(
            {
                item.shown
                for item in store.facts_in_chunk(record.chunk_id or "")
                if item.concept == record.concept and item.fiscal_year != year
            }
        ),
        "row": record.excerpt,
    }
    if found.restated:
        out["restated_value"] = found.restated.shown
        out["restated_in"] = found.restated.filing_fiscal_year
    return out


def _growth(old: dict[str, Any], new: dict[str, Any]) -> float:
    return round((new["value"] - old["value"]) / old["value"] * 100, 2)


def _cagr(first: dict[str, Any], last: dict[str, Any]) -> float:
    years = last["fiscal_year"] - first["fiscal_year"]
    return round(((last["value"] / first["value"]) ** (1 / years) - 1) * 100, 2)


def _pct(part: dict[str, Any], whole: dict[str, Any]) -> float:
    return round(part["value"] / whole["value"] * 100, 2)


def build_cases(store: FactsStore) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def fact(ticker: str, year: int, key: str) -> dict[str, Any]:
        return _fact(store, ticker, year, key)

    def add(
        case_id: str,
        category: str,
        question: str,
        facts: list[dict[str, Any]] | None = None,
        percents: list[float] | None = None,
        forbidden: list[float] | None = None,
    ) -> None:
        cases.append(
            {
                "id": case_id,
                "category": category,
                "question": question,
                "facts": facts or [],
                "percents": percents or [],
                "forbidden": forbidden or [],
            }
        )

    single = [
        (
            "aapl_rev_2025",
            "What were Apple's total net sales in fiscal 2025?",
            "AAPL",
            2025,
            "revenue",
        ),
        (
            "msft_rev_2024",
            "What was Microsoft's total revenue in fiscal 2024?",
            "MSFT",
            2024,
            "revenue",
        ),
        ("msft_rev_last_year", "What was Microsoft's revenue last year?", "MSFT", 2024, "revenue"),
        (
            "tsla_rev_2024",
            "What were Tesla's total revenues in fiscal 2024?",
            "TSLA",
            2024,
            "revenue",
        ),
        (
            "msft_rev_2021",
            "What was Microsoft's total revenue in fiscal 2021?",
            "MSFT",
            2021,
            "revenue",
        ),
        ("aapl_ni_2024", "What was Apple's net income in fiscal 2024?", "AAPL", 2024, "net_income"),
        (
            "msft_ni_2025",
            "What was Microsoft's net income in fiscal 2025?",
            "MSFT",
            2025,
            "net_income",
        ),
        ("tsla_ni_2023", "What was Tesla's net income in fiscal 2023?", "TSLA", 2023, "net_income"),
        (
            "aapl_rd_2023",
            "How much did Apple spend on research and development in fiscal 2023?",
            "AAPL",
            2023,
            "rd_expense",
        ),
        (
            "msft_rd_2022",
            "What was Microsoft's R&D expense in fiscal 2022?",
            "MSFT",
            2022,
            "rd_expense",
        ),
        (
            "tsla_rd_2025",
            "What was Tesla's research and development expense in fiscal 2025?",
            "TSLA",
            2025,
            "rd_expense",
        ),
        (
            "aapl_eps_2025",
            "What was Apple's diluted earnings per share in fiscal 2025?",
            "AAPL",
            2025,
            "eps_diluted",
        ),
        (
            "msft_opinc_2023",
            "What was Microsoft's operating income in fiscal 2023?",
            "MSFT",
            2023,
            "operating_income",
        ),
        (
            "tsla_ocf_2022",
            "What was Tesla's net cash provided by operating activities in fiscal 2022?",
            "TSLA",
            2022,
            "operating_cash_flow",
        ),
        (
            "aapl_capex_2024",
            "What were Apple's capital expenditures in fiscal 2024?",
            "AAPL",
            2024,
            "capex",
        ),
        (
            "msft_assets_2025",
            "What were Microsoft's total assets at the end of fiscal 2025?",
            "MSFT",
            2025,
            "total_assets",
        ),
        (
            "tsla_cash_2024",
            "How much cash and cash equivalents did Tesla have at the end of fiscal 2024?",
            "TSLA",
            2024,
            "cash",
        ),
        (
            "aapl_gp_2022",
            "What was Apple's gross profit in fiscal 2022?",
            "AAPL",
            2022,
            "gross_profit",
        ),
    ]
    for case_id, question, ticker, year, key in single:
        add(case_id, "single_fact", question, [fact(ticker, year, key)])

    add(
        "rd_aapl_msft_2024",
        "cross_company",
        "Compare research and development expenses for Apple and Microsoft in fiscal 2024.",
        [fact("AAPL", 2024, "rd_expense"), fact("MSFT", 2024, "rd_expense")],
    )
    add(
        "rev_all_2023",
        "cross_company",
        "What was total revenue for Apple, Microsoft and Tesla in fiscal 2023?",
        [fact(ticker, 2023, "revenue") for ticker in ("AAPL", "MSFT", "TSLA")],
    )
    add(
        "ni_aapl_tsla_2025",
        "cross_company",
        "Compare Apple's and Tesla's net income in fiscal 2025.",
        [fact("AAPL", 2025, "net_income"), fact("TSLA", 2025, "net_income")],
    )

    add(
        "aapl_rev_2021_2023",
        "trend",
        "What were Apple's total net sales from fiscal 2021 to fiscal 2023?",
        [fact("AAPL", year, "revenue") for year in (2021, 2022, 2023)],
    )
    income = [fact("MSFT", year, "net_income") for year in (2024, 2025)]
    add(
        "msft_ni_growth",
        "trend",
        "How much did Microsoft's net income grow from fiscal 2024 to fiscal 2025, in percent?",
        income,
        [_growth(*income)],
    )
    first, last = fact("TSLA", 2021, "revenue"), fact("TSLA", 2024, "revenue")
    add(
        "tsla_rev_cagr",
        "trend",
        "What was the compound annual growth rate of Tesla's revenue from fiscal 2021 to "
        "fiscal 2024?",
        [first, last],
        [_cagr(first, last)],
    )

    gross, revenue = fact("AAPL", 2025, "gross_profit"), fact("AAPL", 2025, "revenue")
    add(
        "aapl_gm_2025",
        "margin",
        "What was Apple's gross margin percentage in fiscal 2025?",
        [gross, revenue],
        [_pct(gross, revenue)],
    )
    operating, revenue = fact("MSFT", 2024, "operating_income"), fact("MSFT", 2024, "revenue")
    add(
        "msft_om_2024",
        "margin",
        "What was Microsoft's operating margin in fiscal 2024?",
        [operating, revenue],
        [_pct(operating, revenue)],
    )

    add(
        "tsla_eps_2021_restated",
        "restated",
        "What was Tesla's diluted EPS in fiscal 2021?",
        [fact("TSLA", 2021, "eps_diluted")],
    )
    add("msft_sga_2024", "not_covered", "What was Microsoft's SG&A expense in fiscal 2024?")
    add(
        "aapl_iphone_2025",
        "segment",
        "What were Apple's iPhone net sales in fiscal 2025?",
        forbidden=[fact("AAPL", 2025, "revenue")["value"]],
    )
    add(
        "msft_cloud_2024",
        "segment",
        "What was revenue for Microsoft's Intelligent Cloud segment in fiscal 2024?",
        forbidden=[fact("MSFT", 2024, "revenue")["value"]],
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "eval" / "facts" / "numeric_golden.jsonl"
    )
    parser.add_argument("--db-url", help="Postgres URL (default: HISTORY_DB_URL from .env)")
    args = parser.parse_args()

    from config.settings import settings
    from persistence.facts import read_facts

    records, load = read_facts(args.db_url or settings.HISTORY_DB_URL or "")
    if not records:
        raise SystemExit("financial_facts is empty (run scripts/load_facts.py)")
    cases = build_cases(FactsStore(records, load.collection if load else None))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case) + "\n")
    for case in cases:
        print(case["id"], case["percents"], case["forbidden"])
        for item in case["facts"]:
            print(
                f"   {item['ticker']} FY{item['fiscal_year']} {item['metric']}={item['value']} "
                f"other={item['other_columns']} restated={item.get('restated_value')} "
                f"| {item['row']}"
            )
    print(f"{len(cases)} cases -> {args.out}")


if __name__ == "__main__":
    main()
