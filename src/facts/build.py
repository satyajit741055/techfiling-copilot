"""Build normalised, chunk-linked fact records from parser and chunker output.

Inputs (all written by ingestion):
- ``<parsed_dir>/<doc_id>.document.json``: ticker, fiscal year and period end per filing.
- ``<parsed_dir>/<doc_id>.facts.jsonl``: every ``ix:nonFraction`` fact with its context.
- ``<parsed_dir>/<doc_id>.elements.jsonl``: tables whose cells carry the ids of the facts
  they display.
- ``<chunks_dir>/<doc_id>.chunks.jsonl``: table chunks, one table row per line.

Only non-dimensional numeric facts (no segment/geography axes) with an annual duration
or an instant are kept; duplicates within one filing collapse to the most precise value.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from .records import FactRecord

ANNUAL_DAYS = (330, 380)
SAME_PERIOD_DAYS = 10
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_DASHES = {"—", "–", "-"}
_GENERIC_LABELS = {"", "total", "totals", "net", "total net", "subtotal"}


@dataclass(frozen=True)
class _Filing:
    doc_id: str
    ticker: str
    fiscal_year: int
    period_end: date


@dataclass(frozen=True)
class _Cell:
    table_id: str
    label: str  # row path as rendered in chunk text ("Revenue > Total revenue")
    index: int  # position among the row's value cells
    cells: int  # number of value cells in the row


@dataclass(frozen=True)
class _RowLine:
    chunk_id: str
    line: str
    values: tuple[str, ...]


def normalize_unit(unit: str | None) -> str:
    """``usd``/``U_USD``/``U_iso4217USD`` -> ``USD``; per-share, share and pure units likewise."""
    raw = (unit or "").casefold()
    dollars = "usd" in raw or "dollar" in raw  # "U_UnitedStatesOfAmericaDollarsShare"
    if dollars and "share" in raw:
        return "USD/share"
    if dollars:
        return "USD"
    if "share" in raw:
        return "shares"
    if "pure" in raw:
        return "pure"
    return raw or "unknown"


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _filings(parsed_dir: Path) -> dict[str, _Filing]:
    filings: dict[str, _Filing] = {}
    for path in sorted(parsed_dir.glob("*.document.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        filings[document["doc_id"]] = _Filing(
            doc_id=document["doc_id"],
            ticker=str(document["ticker"]).upper(),
            fiscal_year=int(document["fiscal_year"]),
            period_end=date.fromisoformat(document["period_end"]),
        )
    if not filings:
        raise FileNotFoundError(f"No *.document.json files in {parsed_dir}")
    return filings


def fiscal_year_of(period_end: date, filings: Iterable[_Filing]) -> int:
    """Fiscal year of a period: the filing whose period ends within 10 days, otherwise the
    nearest filing's year minus the number of years between them."""
    nearest = min(filings, key=lambda filing: abs((filing.period_end - period_end).days))
    days = (nearest.period_end - period_end).days
    if abs(days) <= SAME_PERIOD_DAYS:
        return nearest.fiscal_year
    return nearest.fiscal_year - round(days / 365.25)


def _annual_or_instant(fact: dict[str, Any]) -> bool:
    if fact.get("instant"):
        return True
    if not fact.get("period_start") or not fact.get("period_end"):
        return False
    days = (date.fromisoformat(fact["period_end"]) - date.fromisoformat(fact["period_start"])).days
    return ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]


def _keep(fact: dict[str, Any]) -> bool:
    return (
        not fact.get("dims")
        and fact.get("value") is not None
        and fact.get("shown") is not None
        and bool(fact.get("period_end"))
        and not str(fact.get("concept", "")).startswith("dei:")
        and _annual_or_instant(fact)
    )


def _table_cells(elements_path: Path) -> dict[str, _Cell]:
    """fact id -> the data-row cell that displays it."""
    cells: dict[str, _Cell] = {}
    for element in _read_jsonl(elements_path):
        table = element.get("table")
        if not table:
            continue
        for row in table.get("rows", []):
            if row.get("row_type") != "data":
                continue
            label = " > ".join(row.get("path") or [row.get("label", "")]).replace("|", "\\|")
            row_cells = row.get("cells", [])
            for index, cell in enumerate(row_cells):
                for fact_id in cell.get("fact_ids", []):
                    cells.setdefault(
                        fact_id, _Cell(table["table_id"], label, index, len(row_cells))
                    )
    return cells


def _row_lines(chunks_path: Path) -> dict[tuple[str, str], list[_RowLine]]:
    """(table_id, row label) -> table rows as they appear in chunk text."""
    rows: defaultdict[tuple[str, str], list[_RowLine]] = defaultdict(list)
    if not chunks_path.exists():
        return rows
    for chunk in _read_jsonl(chunks_path):
        table_id = chunk.get("table_id")
        if not table_id:
            continue
        for line in str(chunk.get("text", "")).splitlines():
            line = line.strip()
            if not (line.startswith("| ") and line.endswith(" |")) or line.startswith("|---"):
                continue
            inner = line[2:-2]
            # Pipes inside labels are escaped ("\|"), so " | " only separates cells.
            label, *values = inner.split(" | ")
            rows[(table_id, label)].append(_RowLine(chunk["chunk_id"], inner, tuple(values)))
    return rows


def _number(text: str) -> float | None:
    match = _NUMBER.search(text.replace("(", "-"))
    return abs(float(match.group(0).replace(",", ""))) if match else None


def _link(
    fact: dict[str, Any], cell: _Cell | None, rows: dict[tuple[str, str], list[_RowLine]]
) -> tuple[str, str, str, str] | None:
    """(chunk_id, table_id, row label, row line) for the fact, or None when no chunk shows it."""
    if cell is None:
        return None
    shown = abs(float(fact["shown"]))
    for row in rows.get((cell.table_id, cell.label), []):
        if len(row.values) != cell.cells:
            continue
        text = row.values[cell.index].strip()
        number = 0.0 if text in _DASHES else _number(text)
        if number is not None and abs(number - shown) <= 1e-9 * max(1.0, shown):
            return row.chunk_id, cell.table_id, cell.label, row.line
    return None


def _precision_key(record: FactRecord) -> tuple[int, int, int]:
    """Linked facts first, then the most precise (lowest scale), then rows whose label
    names the line item ("Revenues > Total revenues" rather than a bare "Total")."""
    leaf = (record.row_label or "").rsplit(" > ", 1)[-1].strip().casefold()
    return (0 if record.linked else 1, record.scale, 1 if leaf in _GENERIC_LABELS else 0)


def build_facts(parsed_dir: Path, chunks_dir: Path) -> list[FactRecord]:
    """Fact records for every filing in ``parsed_dir``, linked to chunks in ``chunks_dir``."""
    filings = _filings(parsed_dir)
    by_ticker: defaultdict[str, list[_Filing]] = defaultdict(list)
    for filing in filings.values():
        by_ticker[filing.ticker].append(filing)

    records: list[FactRecord] = []
    for doc_id, filing in sorted(filings.items()):
        facts_path = parsed_dir / f"{doc_id}.facts.jsonl"
        if not facts_path.exists():
            continue
        cells = _table_cells(parsed_dir / f"{doc_id}.elements.jsonl")
        rows = _row_lines(chunks_dir / f"{doc_id}.chunks.jsonl")
        best: dict[tuple[str, str | None, str], FactRecord] = {}
        for fact in _read_jsonl(facts_path):
            if not _keep(fact):
                continue
            period_end = date.fromisoformat(fact["period_end"])
            record = FactRecord(
                doc_id=doc_id,
                fact_id=str(fact["id"]),
                ticker=filing.ticker,
                filing_fiscal_year=filing.fiscal_year,
                fiscal_year=fiscal_year_of(period_end, by_ticker[filing.ticker]),
                concept=str(fact["concept"]),
                period_start=None if fact.get("instant") else fact.get("period_start"),
                period_end=fact["period_end"],
                instant=bool(fact.get("instant")),
                unit=normalize_unit(fact.get("unit")),
                value=float(fact["value"]),
                shown=abs(float(fact["shown"])),
                scale=int(fact.get("scale") or 0),
            )
            if link := _link(fact, cells.get(record.fact_id), rows):
                chunk_id, table_id, label, line = link
                record = replace(
                    record, chunk_id=chunk_id, table_id=table_id, row_label=label, excerpt=line
                )
            key = (record.concept, record.period_start, record.period_end)
            if key not in best or _precision_key(record) < _precision_key(best[key]):
                best[key] = record
        records.extend(best.values())
    return records


def chunk_run_for_collection(chunks_root: Path, collection: str) -> Path | None:
    """The chunk run whose manifest records ``collection`` (the newest when several do)."""
    matches = []
    for manifest_path in chunks_root.glob("*/run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("qdrant_collection") or {}).get("name") == collection:
            matches.append(manifest_path.parent)
    return max(matches, default=None)
