"""In-memory filing catalog used for deterministic scope resolution."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

_MANUAL_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
}
_COMPANY_SUFFIXES = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|plc)\b", re.IGNORECASE
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_company(value: str) -> str:
    """Normalize a ticker or company mention for alias matching."""
    normalized = _NON_ALNUM.sub(" ", value.casefold()).strip()
    return " ".join(_COMPANY_SUFFIXES.sub(" ", normalized).split())


class Filing(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    doc_id: str
    ticker: str
    company: str
    fiscal_year: int
    period_end: str | None = None


class FilingCatalog:
    """Validated filing metadata and company-alias lookup built once at startup."""

    def __init__(self, filings: Iterable[Mapping[str, Any]]):
        records = [Filing.model_validate(record) for record in filings]
        by_cell: dict[tuple[str, int], Filing] = {}
        for record in records:
            ticker = record.ticker.upper()
            key = (ticker, record.fiscal_year)
            if key in by_cell and by_cell[key].doc_id != record.doc_id:
                raise ValueError(f"Duplicate filing for {ticker} FY{record.fiscal_year}")
            by_cell[key] = record.model_copy(update={"ticker": ticker})

        self._by_cell = by_cell
        self._filings = sorted(by_cell.values(), key=lambda item: (item.ticker, item.fiscal_year))
        years: defaultdict[str, list[int]] = defaultdict(list)
        aliases: dict[str, str] = {}
        for filing in self._filings:
            years[filing.ticker].append(filing.fiscal_year)
            for alias in (filing.ticker, filing.company, normalize_company(filing.company)):
                normalized = normalize_company(alias)
                if normalized:
                    existing = aliases.get(normalized)
                    if existing and existing != filing.ticker:
                        raise ValueError(f"Company alias {alias!r} maps to multiple tickers")
                    aliases[normalized] = filing.ticker
        for alias, ticker in _MANUAL_ALIASES.items():
            if ticker in years:
                aliases[normalize_company(alias)] = ticker

        self._years = {ticker: sorted(set(values)) for ticker, values in years.items()}
        self._aliases = aliases

    @property
    def tickers(self) -> list[str]:
        return sorted(self._years)

    def years_by_ticker(self, ticker: str) -> list[int]:
        return list(self._years.get(ticker.upper(), []))

    def latest_year(self, ticker: str) -> int | None:
        years = self.years_by_ticker(ticker)
        return years[-1] if years else None

    def has(self, ticker: str, fiscal_year: int) -> bool:
        return (ticker.upper(), fiscal_year) in self._by_cell

    def resolve_company(self, mention: str) -> str | None:
        return self._aliases.get(normalize_company(mention))

    def filings(self) -> list[Filing]:
        return list(self._filings)

    def describe(self) -> str:
        if not self._filings:
            return "No filings are currently indexed"
        years = sorted({filing.fiscal_year for filing in self._filings})
        year_text = f"FY{years[0]}" if len(years) == 1 else f"FY{years[0]}–FY{years[-1]}"
        return f"{', '.join(self.tickers)} · {year_text}"
