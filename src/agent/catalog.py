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

    def find_companies(self, text: str) -> list[str]:
        """Indexed companies named in free text, in order of first mention.

        Names match as whole words after normalisation ("Apple's" -> apple); tickers only
        when written in capitals ("MSFT"), so short tickers do not match ordinary words.
        """
        normalized = f" {normalize_company(text)} "
        positions: dict[str, int] = {}
        for alias, ticker in self._aliases.items():
            if alias == ticker.casefold():
                match = re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", text)
                position = match.start() if match else -1
            else:
                position = normalized.find(f" {alias} ") if len(alias) >= 3 else -1
            if position >= 0:
                positions[ticker] = min(position, positions.get(ticker, position))
        return sorted(positions, key=positions.__getitem__)

    def filings(self) -> list[Filing]:
        return list(self._filings)

    def company_name(self, ticker: str) -> str:
        latest = self.latest_year(ticker)
        filing = self._by_cell.get((ticker.upper(), latest)) if latest is not None else None
        return filing.company if filing else ticker.upper()

    def example_questions(self) -> list[str]:
        """Three questions that the indexed filings can answer (for refusals and onboarding)."""
        tickers = self.tickers
        if not tickers:
            return []
        first, second = tickers[0], tickers[1 % len(tickers)]
        latest = self.latest_year(first)
        years = self.years_by_ticker(first)
        examples = [f"What was {self.company_name(first)}'s total revenue in FY{latest}?"]
        if second != first:
            examples.append(
                f"Compare research and development spending for {self.company_name(first)} "
                f"and {self.company_name(second)}."
            )
        if len(years) > 1:
            examples.append(
                f"How did {self.company_name(first)}'s main risk factors change from "
                f"FY{years[-2]} to FY{years[-1]}?"
            )
        examples.append(f"What risk factors did {self.company_name(tickers[-1])} report?")
        return examples[:3]

    def describe(self) -> str:
        if not self._filings:
            return "No filings are currently indexed"
        years = sorted({filing.fiscal_year for filing in self._filings})
        year_text = f"FY{years[0]}" if len(years) == 1 else f"FY{years[0]}–FY{years[-1]}"
        return f"{', '.join(self.tickers)} · {year_text}"
