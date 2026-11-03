"""One reported XBRL figure, normalised and linked to the table row that shows it."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FactRecord:
    doc_id: str
    fact_id: str
    ticker: str
    filing_fiscal_year: int
    fiscal_year: int  # fiscal year of the period, not of the filing
    concept: str
    period_start: str | None
    period_end: str
    instant: bool
    unit: str  # USD | shares | USD/share | pure | other (lowercased raw unit)
    value: float  # true value: sign * shown * 10**scale
    shown: float  # as printed in the document
    scale: int
    chunk_id: str | None = None
    table_id: str | None = None
    row_label: str | None = None
    excerpt: str | None = None  # the table row as it appears in the chunk text

    @property
    def linked(self) -> bool:
        return self.chunk_id is not None and self.excerpt is not None


COLUMNS = tuple(field.name for field in fields(FactRecord))
