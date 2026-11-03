"""In-memory index over fact records: as-reported lookups and facts per chunk."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .metrics import METRICS, Metric
from .records import FactRecord


@dataclass(frozen=True)
class FactLookup:
    """The as-reported value of a metric for one filing, plus a later restatement if any."""

    metric: Metric
    record: FactRecord
    restated: FactRecord | None = None


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


class FactsStore:
    def __init__(self, records: Iterable[FactRecord], collection: str | None = None) -> None:
        self.collection = collection
        self._records = list(records)
        self._by_cell: defaultdict[tuple[str, str, int], list[FactRecord]] = defaultdict(list)
        self._by_chunk: defaultdict[str, list[FactRecord]] = defaultdict(list)
        for record in self._records:
            self._by_cell[(record.ticker, record.concept, record.fiscal_year)].append(record)
            if record.chunk_id:
                self._by_chunk[record.chunk_id].append(record)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def linked(self) -> int:
        return sum(record.linked for record in self._records)

    def lookup(self, ticker: str, fiscal_year: int, metric: Metric) -> FactLookup | None:
        """Value reported in the fiscal year's own 10-K, from the first alias concept that
        has a citable (chunk-linked) fact; a later 10-K's differing value is ``restated``."""
        ticker = ticker.upper()
        for concept in metric.concepts:
            candidates = self._by_cell.get((ticker, concept, fiscal_year), [])
            own = [
                record
                for record in candidates
                if record.filing_fiscal_year == fiscal_year
                and record.linked
                and record.unit == metric.unit
            ]
            if not own:
                continue
            record = own[0]
            later = sorted(
                (
                    item
                    for item in candidates
                    if item.filing_fiscal_year > fiscal_year
                    and item.unit == record.unit
                    and not _same(item.value, record.value)
                ),
                key=lambda item: item.filing_fiscal_year,
            )
            return FactLookup(metric, record, later[-1] if later else None)
        return None

    def facts_in_chunk(self, chunk_id: str) -> list[FactRecord]:
        return list(self._by_chunk.get(chunk_id, []))

    def coverage(self, tickers: Iterable[str], years: Iterable[int]) -> dict[str, list[str]]:
        """metric key -> cells (``TICKER FYyyyy``) with no citable as-reported value."""
        cells = [(ticker, year) for ticker in tickers for year in years]
        return {
            metric.key: [
                f"{ticker} FY{year}"
                for ticker, year in cells
                if self.lookup(ticker, year, metric) is None
            ]
            for metric in METRICS
        }
