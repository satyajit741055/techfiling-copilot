"""Parse SEC 10-K Inline-XBRL HTML into structured elements (heading / paragraph / table).

Logic ported from exploration/00_explore_src.py. Writes, per document:
  <doc_id>.document.json, <doc_id>.elements.jsonl, <doc_id>.facts.jsonl and quality_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from config.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw_html"
OUT_DIR = ROOT / "data" / "parsed"

CONFIG = {
    "boilerplate_min_repeats": 5,
    "boilerplate_max_len": 80,
    "heading_max_len": 150,
    "title_lookback": 4,
    "max_header_rows": 6,
    "nbsp_pt": 2.5,
    "indent_tolerance_pt": 1.0,
    "numeric_table_ratio": 0.3,
    "note_min_len": 100,
    "low_confidence": 0.6,
}
DASHES = {"—", "–", "-", "−", "--"}
VALUE_TYPES = {"NUMBER", "DASH", "YEAR"}
BOLD_WEIGHTS = {"700", "800", "900", "bold", "bolder"}
BLOCK_TAGS = ["div", "p", "table", "h1", "h2", "h3", "h4", "h5", "h6", "li", "center"]
SCALE_TO_UNITS = {3: "in thousands", 6: "in millions", 9: "in billions"}

WS_RE = re.compile(r"\s+")
UNITS_RE = re.compile(r"\b(?:in|dollars in|amounts in|\$ in)\s+(millions|thousands|billions)\b", re.I)
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.*)$", re.I)
PART_RE = re.compile(r"^part\s+(iv|i{1,3})\b\.?\s*(.*)$", re.I)
YEAR_RE = re.compile(r"^(?:fiscal\s+|fy\s*)?(?:19|20)\d{2}$", re.I)
DATE_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|\b(year|years|quarter|quarters|months|weeks)\s+ended\b",
    re.I,
)
CURRENCY_RE = re.compile(r"^(?:US\$|C\$|A\$|\$|€|£|¥)$")
NUM_CORE_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?$|^\.\d+$")
UNITS_ONLY_RE = re.compile(r"^\(?\s*(?:\$|dollars|amounts)?\s*in\s+(?:millions|thousands|billions)\b[^)]*\)?$", re.I)
MAGNITUDES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
MAGNITUDE_RE = re.compile(r"(thousand|million|billion|trillion)s?$", re.I)
PAGE_NUM_RE = re.compile(r"^(?:page\s*)?(?:\d{1,3}|[ivxlc]{1,6})$|^[-–—]\s*\d{1,3}\s*[-–—]$", re.I)


@dataclass
class XbrlFact:
    id: str
    concept: str
    context: str
    unit: str | None
    scale: int
    sign: int
    shown: float | None  # absolute number as printed in the document
    value: float | None  # true value = sign * shown * 10**scale


@dataclass
class Cell:
    text: str
    ctype: str  # NUMBER | DASH | YEAR | DATE | TEXT | EMPTY ...
    value: float | None = None
    is_percent: bool = False
    currency: str | None = None
    fact_ids: list[str] = field(default_factory=list)


@dataclass
class Column:
    name: str
    period_start: str | None = None
    period_end: str | None = None


@dataclass
class Row:
    label: str
    level: int
    path: list[str]
    row_type: str  # data | heading | note
    is_total: bool
    cells: list[Cell]  # aligned with Table.columns[1:]


@dataclass
class Table:
    table_id: str
    kind: str  # financial | textual | text_block | layout | toc
    title: str | None = None
    units: str | None = None
    intro: str | None = None
    columns: list[Column] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    period_axis: str | None = None
    quality: dict = field(default_factory=dict)
    markdown: str = ""


@dataclass
class Element:
    doc_id: str
    order: int
    type: str  # heading | paragraph | table
    part: str | None
    section: str | None
    section_title: str | None
    text: str | None = None
    table: Table | None = None


@dataclass
class Document:
    doc_id: str
    company: str
    ticker: str | None
    cik: str | None
    form: str
    fiscal_year: int | None
    period_end: str | None
    source_file: str
    elements: list[Element] = field(default_factory=list)

def clean(s: str | None) -> str:
    """Collapse whitespace and non-breaking spaces into single spaces and strip the result."""
    return WS_RE.sub(" ", (s or "").replace("\xa0", " ")).strip()

def style_of(tag) -> dict:
    """Parse a tag's inline ``style`` attribute into a {property: value} dict (lowercased)."""
    out = {}
    for part in (tag.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip().lower()
    return out

def to_pt(v: str) -> float:
    """Convert a CSS length (px/em/in/pt) to points, used to compare indentation."""
    m = re.match(r"(-?[\d.]+)\s*(pt|px|em|in)?", v or "")
    if not m:
        return 0.0
    n, unit = float(m.group(1)), m.group(2) or "px"
    return {"px": n * 0.75, "em": n * 10, "in": n * 72}.get(unit, n)

def padding_left_pt(st: dict) -> float:
    """Return the left padding in points from a style dict, handling the shorthand ``padding``."""
    if "padding-left" in st:
        return to_pt(st["padding-left"])
    parts = st.get("padding", "").split()
    idx = {1: 0, 2: 1, 3: 1, 4: 3}.get(len(parts))
    return to_pt(parts[idx]) if idx is not None else 0.0

def is_bold(tag) -> bool:
    """True only if every piece of visible text in the tag is bold (via <b>/<strong> or font-weight)."""
    nodes = [s for s in tag.find_all(string=True) if clean(s)]
    if not nodes:
        return False
    for s in nodes:
        p, bold = s.parent, False
        while p is not None:
            if p.name in ("b", "strong") or style_of(p).get("font-weight") in BOLD_WEIGHTS:
                bold = True
                break
            if p is tag:
                break
            p = p.parent
        if not bold:
            return False
    return True

def load(path: Path) -> BeautifulSoup:
    """Read an HTML filing from disk and parse it into a BeautifulSoup tree (lxml backend)."""
    return BeautifulSoup(path.read_bytes(), "lxml")


def doc_metadata(soup: BeautifulSoup, path: Path) -> dict:
    """Extract document-level metadata (company, ticker, CIK, form, fiscal year, period end) from DEI tags."""
    dei = {}
    for t in soup.find_all("ix:nonnumeric"):
        name = t.get("name", "")
        if name.startswith("dei:") and name not in dei:
            dei[name] = clean(t.get_text())
    company = dei.get("dei:EntityRegistrantName") or path.stem.split("__")[0].replace("_", " ")
    ticker = dei.get("dei:TradingSymbol")
    form = dei.get("dei:DocumentType", "10-K")
    fy = dei.get("dei:DocumentFiscalYearFocus", "")
    return {
        "doc_id": f"{ticker or re.sub(r'\W+', '_', company)}_{form}_{fy}",
        "company": company,
        "ticker": ticker,
        "cik": dei.get("dei:EntityCentralIndexKey"),
        "form": form,
        "fiscal_year": int(fy) if fy.isdigit() else None,
        "period_end": dei.get("dei:DocumentPeriodEndDate"),
        "source_file": path.name,
    }

def parse_contexts(soup: BeautifulSoup) -> dict:
    """Map each XBRL context id to its reporting period (start/end/instant) and dimensions."""
    out = {}
    for c in soup.find_all("xbrli:context"):
        def get(name):
            t = c.find(name)
            return t.get_text(strip=True) if t else None

        inst = get("xbrli:instant")
        out[c.get("id")] = {
            "start": get("xbrli:startdate"),
            "end": inst or get("xbrli:enddate"),
            "instant": inst is not None,
            "dims": {m.get("dimension"): clean(m.get_text()) for m in c.find_all(["xbrldi:explicitmember", "xbrldi:typedmember"])},
        }
    return out



def parse_fact(t) -> XbrlFact:
    """Turn one ``ix:nonfraction`` tag into an XbrlFact, decoding its shown vs. true (scaled, signed) value."""
    txt, fmt = clean(t.get_text()), t.get("format", "")
    scale, sign = int(t.get("scale") or 0), (-1 if t.get("sign") == "-" else 1)
    if "zero" in fmt or txt in DASHES:
        shown = 0.0
    else:
        core = re.sub(r"[^\d.,]", "", txt)
        core = core.replace(".", "").replace(",", ".") if "comma-decimal" in fmt else core.replace(",", "")
        try:
            shown = float(core)
        except ValueError:
            shown = None
    value = None if shown is None else sign * shown * 10**scale
    return XbrlFact(t.get("id"), t.get("name"), t.get("contextref"), t.get("unitref"), scale, sign, shown, value)



def facts_records(facts: dict, contexts: dict, doc_id: str) -> list[dict]:
    """Flatten facts joined with their context period/dims into plain dicts for the .facts.jsonl output."""
    rows = []
    for f in facts.values():
        ctx = contexts.get(f.context, {})
        rows.append({**asdict(f), "period_start": ctx.get("start"), "period_end": ctx.get("end"),
                     "instant": ctx.get("instant"), "dims": ctx.get("dims") or {}, "doc_id": doc_id})
    return rows



def strip_noise(soup: BeautifulSoup) -> Counter:
    """Remove hidden/absolutely-positioned blocks and normalize <br> and block tags; return counts of what was removed."""
    removed = Counter()
    for t in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        removed["hidden_blocks"] += 1
        t.decompose()
    for t in soup.find_all(style=re.compile(r"position\s*:\s*absolute", re.I)):
        if t.parent is not None:
            removed["page_header_footer_blocks"] += 1
            t.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" ")
    for block in soup.find_all(["div", "p", "li"]):  # stacked divs inside one cell must not glue words together
        block.append(" ")
    return removed


def norm_key(t: str) -> str:
    """Normalize a line for boilerplate matching by lowercasing and masking digits as ``#``."""
    return re.sub(r"\d+", "#", t.lower())


def find_boilerplate(texts: list[str]) -> set[str]:
    """Find short lines that repeat often enough (e.g. running headers/footers) to be treated as boilerplate."""
    counts = Counter(norm_key(t) for t in texts if len(t) <= CONFIG["boilerplate_max_len"])
    return {k for k, n in counts.items() if n >= CONFIG["boilerplate_min_repeats"]}

def is_boilerplate(text: str, boilerplate: set[str], bold: bool = False) -> bool:
    """Decide whether a line should be dropped as boilerplate/page-number, keeping meaningful units and Item/Part headings."""
    if PAGE_NUM_RE.match(text):
        return True
    if UNITS_RE.search(text):  # "(In millions)" repeats a lot but carries meaning
        return False
    if bold and (ITEM_RE.fullmatch(text) or PART_RE.fullmatch(text)):
        return False
    return len(text) <= CONFIG["boilerplate_max_len"] and norm_key(text) in boilerplate

def iter_blocks(body):
    """Yield the leaf block elements (and top-level tables) of the document in reading order."""
    for el in body.find_all(BLOCK_TAGS):
        in_table = el.find_parent("table") is not None
        if el.name == "table":
            if not in_table:
                yield el
        elif not in_table and el.find(BLOCK_TAGS) is None and clean(el.get_text("")):
            yield el


def detect_section(text: str, bold: bool):
    """Detect whether a bold line starts a new ``Item`` or ``Part`` section and return its label/title."""
    if not bold or len(text) > CONFIG["heading_max_len"]:
        return None
    if m := ITEM_RE.match(text):
        return "item", f"Item {m.group(1).upper()}", clean(m.group(2)) or None
    if m := PART_RE.match(text):
        return "part", f"Part {m.group(1).upper()}", clean(m.group(2)) or None
    return None



@dataclass
class RawCell:
    text: str
    start: int
    span: int
    bold: bool
    indent_pt: float
    fact_ids: list[str]
    has_link: bool
    ctype: str = ""


def raw_rows(table) -> list[list[RawCell]]:
    """Flatten an HTML ``<table>`` into a rectangular grid of RawCells.

    Walks the direct ``<tr>`` rows of this table (skipping nested tables), expanding
    ``colspan`` so each cell knows its starting column and span, and ``rowspan`` by
    carrying a copy of the cell down into the rows it covers (the carried copies drop
    the fact ids so spanned values are not double-counted). For every cell it also
    records left indentation (CSS padding + leading non-breaking spaces, in points),
    whether the text is bold, the ids of any XBRL ``ix:nonfraction`` facts inside it,
    and whether it contains an in-document anchor link.

    Args:
        table: The BeautifulSoup ``<table>`` tag to flatten.

    Returns:
        A list of rows, each a list of RawCells sorted left-to-right by start column.
    """
    rows, carried = [], defaultdict(dict)  # row -> {col: copy of a rowspan cell from above}
    trs = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    for r, tr in enumerate(trs):
        col, cells = 0, []
        for td in tr.find_all(["td", "th"], recursive=False):
            while col in carried[r]:
                cells.append(carried[r].pop(col))
                col = cells[-1].start + cells[-1].span
            cs, rs = max(1, int(td.get("colspan") or 1)), max(1, int(td.get("rowspan") or 1))
            raw = td.get_text("")
            lead = raw.strip(" \t\r\n")
            nbsp = len(lead) - len(lead.lstrip("\xa0"))
            text = clean(raw)
            inner_pad = max((padding_left_pt(style_of(d)) for d in td.find_all(["div", "p"])), default=0.0)
            cells.append(RawCell(
                text=text, start=col, span=cs, bold=bool(text) and is_bold(td),
                indent_pt=padding_left_pt(style_of(td)) + inner_pad + nbsp * CONFIG["nbsp_pt"],
                fact_ids=[f.get("id") for f in td.find_all("ix:nonfraction")],
                has_link=td.find("a", href=re.compile(r"^#")) is not None,
            ))
            for rr in range(r + 1, r + rs):  # fill down: the spanned rows show this text too (no fact ids: no double count)
                c = cells[-1]
                carried[rr][col] = RawCell(c.text, c.start, c.span, c.bold, c.indent_pt, [], c.has_link)
            col += cs
        cells += carried.pop(r, {}).values()
        rows.append(sorted(cells, key=lambda c: c.start))
    return rows



def parse_number(t: str):
    """-> (value, is_percent, currency, is_dash)"""
    s = t.replace("−", "-").replace(" ", "")
    m = re.search(r"US\$|C\$|A\$|\$|€|£|¥", s)
    cur = m.group(0) if m else None
    if cur:
        s = s.replace(cur, "")
    pct = "%" in s
    s = s.replace("%", "")
    neg = s.startswith("(")
    s = s.strip("()")
    if s in DASHES:
        return 0.0, pct, cur, True
    if s.startswith("-"):
        neg, s = True, s[1:]
    mult = 1.0
    if m := MAGNITUDE_RE.search(s):  # "$2.0 trillion"
        mult, s = MAGNITUDES[m.group(1).lower()], s[: m.start()]
    if NUM_CORE_RE.match(s):
        v = float(s.replace(",", "")) * mult
        return (-v if neg else v), pct, cur, False
    return None, pct, cur, False


def cell_type(t: str) -> str:
    """Classify a cell's text into a type (NUMBER, DASH, YEAR, DATE, CURRENCY, TEXT, EMPTY, ...)."""
    if not t:
        return "EMPTY"
    if CURRENCY_RE.match(t):
        return "CURRENCY"
    if t in ("(", "$(", "($"):
        return "OPEN"
    if t in (")", "%", ")%", "%)"):
        return "SUFFIX"
    if YEAR_RE.match(t):
        return "YEAR"
    v, _, _, dash = parse_number(t)
    if dash:
        return "DASH"
    if v is not None:
        return "NUMBER"
    if len(t) <= 60 and DATE_RE.search(t):
        return "DATE"
    return "TEXT"


def merge_fragments(cells: list[RawCell]) -> list[RawCell]:
    """Glue split cell fragments back together (e.g. ``$`` + ``1,234`` + ``)``) into single value cells."""
    out: list[RawCell] = []
    for c in (c for c in cells if c.text):
        c.ctype = cell_type(c.text)
        p = out[-1] if out else None
        glue = p is not None and (
            (p.ctype in ("CURRENCY", "OPEN") and c.ctype in VALUE_TYPES | {"CURRENCY", "OPEN"})
            or (c.ctype == "SUFFIX" and p.ctype in VALUE_TYPES)
        )
        if glue:
            end = max(p.start + p.span, c.start + c.span)
            p.text, p.span = p.text + c.text, end - p.start
            p.fact_ids += c.fact_ids
            p.ctype = cell_type(p.text)
            continue
        out.append(c)
    return out



def classify(merged: list[list[RawCell]]) -> str:
    """Classify a whole table as layout / toc / financial / textual based on links, facts, and numeric density."""
    cells = [c for r in merged for c in r]
    if not cells:
        return "layout"
    if sum(c.has_link for c in cells) >= 0.5 * len(cells):
        return "toc"
    if any(c.fact_ids for c in cells):
        return "financial"
    lstart = label_start(merged)  # ignore the label column: exhibit numbers like "3.1" are not data
    values = [c for c in cells if c.start != lstart]
    numeric = sum(c.ctype == "NUMBER" for c in values)  # dashes alone (e.g. "—" trading symbol) don't make a table numeric
    return "financial" if values and numeric / len(values) >= CONFIG["numeric_table_ratio"] else "textual"


def label_start(merged) -> int:
    """Infer the grid column where row labels begin (the most common first-cell position of data rows)."""
    starts = [r[0].start for r in merged if len(r) > 1 and any(c.ctype in ("NUMBER", "DASH") or c.fact_ids for c in r[1:])]
    return Counter(starts).most_common(1)[0][0] if starts else 0


def header_row_count(merged, lstart: int) -> int:
    """Count how many leading rows form the table header before the first data row."""
    n = 0
    for i, cells in enumerate(merged[: CONFIG["max_header_rows"]]):
        if not cells:
            n = i + 1
            continue
        if any(c.fact_ids or c.ctype == "NUMBER" for c in cells):
            break
        units_only = len(cells) == 1 and UNITS_RE.search(cells[0].text)
        label_empty = cells[0].start > lstart
        multi = len(cells) >= 2 and (all(c.bold for c in cells) or any(c.ctype in ("YEAR", "DATE") for c in cells[1:]))
        if units_only or label_empty or multi:
            n = i + 1
            continue
        break
    return n



def merge_intervals(iv) -> list[list[int]]:
    """Merge overlapping [start, end) intervals into a sorted, non-overlapping list."""
    groups: list[list[int]] = []
    for s, e in sorted(iv):
        if groups and s < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], e)
        else:
            groups.append([s, e])
    return groups


def column_groups(data_rows, lstart: int) -> list[list[int]]:
    """Derive the table's logical columns as a set of non-overlapping start/end intervals.

    Uses only multi-cell data rows (single-cell rows are headings/notes). It first builds
    "primary" columns from value-bearing cells — numbers, dashes, XBRL facts, and the label
    column at ``lstart`` — merging their spans. It then adds any text cells (descriptions,
    dates, "Not applicable") that don't overlap a primary column as their own columns.
    If nothing qualifies, it falls back to merging the spans of every cell.

    Args:
        data_rows: The table's data rows as lists of RawCells.
        lstart: Grid column where row labels start (from ``label_start``).

    Returns:
        A sorted list of ``[start, end)`` intervals, one per logical column.
    """
    multi = [c for r in data_rows if len(r) > 1 for c in r]  # single-cell rows are headings/notes
    primary = merge_intervals([(c.start, c.start + c.span) for c in multi
                               if c.ctype in ("NUMBER", "DASH") or c.fact_ids or c.start == lstart])
    # text columns (descriptions, dates, "Not applicable") that don't overlap a value column get their own column
    rest = [(c.start, c.start + c.span) for c in multi
            if not any(c.start < ge and gs < c.start + c.span for gs, ge in primary)]
    groups = sorted(primary + merge_intervals(rest))
    return groups or merge_intervals([(c.start, c.start + c.span) for r in data_rows for c in r])


def project(cells, groups, is_header: bool):
    """Assign each cell to the column group(s) its span overlaps.

    A cell that overlaps no group is snapped to the nearest one (counted as a conflict).
    Header cells may legitimately span several columns, so all overlapped groups are kept;
    data cells are forced into a single group (extra overlaps, or two data cells landing in
    the same group, are counted as conflicts). The conflict count later feeds the table
    quality score.

    Args:
        cells: The RawCells of one row.
        groups: Column intervals from ``column_groups``.
        is_header: Whether this row is part of the header (allows multi-column spans).

    Returns:
        A tuple ``(projected, conflicts)`` where ``projected`` is a list of cell lists,
        one per column group, and ``conflicts`` is the number of alignment problems found.
    """
    out, conflicts = [[] for _ in groups], 0
    for c in cells:
        s, e = c.start, c.start + c.span
        hits = [j for j, (gs, ge) in enumerate(groups) if s < ge and gs < e]
        if not hits:
            hits = [min(range(len(groups)), key=lambda j: min(abs(groups[j][0] - e), abs(groups[j][1] - s)))]
            conflicts += 1
        if not is_header and len(hits) > 1:
            conflicts += 1
        for j in hits if is_header else hits[:1]:
            if out[j] and not is_header:
                conflicts += 1
            out[j].append(c)
    return out, conflicts


def column_names(header_rows, groups) -> list[str]:
    """Build each column's name by stacking the header-row fragments that project onto that column group."""
    parts = [[] for _ in groups]
    for cells in header_rows:
        cells = [c for c in cells if not UNITS_ONLY_RE.match(c.text)]  # drop "(in millions)", keep "Number of RSUs (in thousands)"
        proj, _ = project(cells, groups, is_header=True)
        for j, cs in enumerate(proj):
            for c in cs:
                if not parts[j] or parts[j][-1] != c.text:
                    parts[j].append(c.text)
    return [clean(" ".join(p)) for p in parts]



def to_cell(cs: list[RawCell]) -> Cell:
    """Combine the raw cells projected onto one column into a typed Cell (value, percent, currency, fact ids)."""
    if not cs:
        return Cell("", "EMPTY")
    text = clean(" ".join(c.text for c in cs))
    ctype = cell_type(text) if len(cs) > 1 else cs[0].ctype
    v, pct, cur, _ = parse_number(text) if ctype in ("NUMBER", "DASH") else (None, False, None, False)
    if ctype == "YEAR":
        v = float(text[-4:])
    return Cell(text, ctype, v, pct, cur, [f for c in cs for f in c.fact_ids])


TOTAL_RE = re.compile(r"^(sub)?total\b", re.I)
FOOTNOTE_RE = re.compile(r"\((?:\d{1,2}|[a-z])\)|\*+$", re.I)


def norm_label(s: str) -> str:
    """Normalize a row label for comparison by stripping footnote markers and non-alphanumerics."""
    return re.sub(r"[^a-z0-9 ]", "", FOOTNOTE_RE.sub("", s).lower()).strip()


def closes_group(total_label: str, heading: str) -> bool:
    """True if a ``Total ...`` row matches (and therefore closes) a given open heading group."""
    rest = re.sub(r"^(sub)?total\s*", "", norm_label(total_label))
    return rest == "" or rest == norm_label(heading)


@dataclass
class OpenGroup:
    indent: float
    label: str
    mode: str | None = None  # "nested" (children indented deeper) or "flat" (children at the same indent or less)
    child_indent: float = 0.0  # indent of the first child; a flat group closes on a row shallower than this
    n_data: int = 0


def build_rows(data_rows, groups):
    """Turn projected data rows into typed Rows with an inferred heading hierarchy.

    For each row it projects the cells onto the column groups, reads the label (column 0)
    and the value cells, and decides the row type:
      - ``note``  — no values and a long label (a footnote/explanatory sentence).
      - ``heading`` — a label with no values (opens a group).
      - ``data`` — has values; an empty label continues the previous data row.

    Heading nesting is tracked with a stack keyed on left indentation. The first child of a
    heading fixes whether that group is "nested" (children indented deeper) or "flat"
    (children at the same/lesser indent), which controls when the group closes. ``Total``/
    ``Subtotal`` rows close the matching open group even when not indented, and a bare
    ``Total`` with no group to close is treated as a grand total. Each row gets a breadcrumb
    ``path`` built from the open headings plus its own label.

    Args:
        data_rows: The table's data rows as lists of RawCells.
        groups: Column intervals from ``column_groups``.

    Returns:
        A tuple ``(rows, conflicts)`` of the built Rows and the accumulated projection
        conflict count (used in scoring).
    """
    projected, conflicts = [], 0
    for cells in data_rows:
        if cells:
            proj, k = project(cells, groups, is_header=False)
            projected.append(proj)
            conflicts += k if len(cells) > 1 else 0
    tol = CONFIG["indent_tolerance_pt"]
    rows, stack = [], []  # open heading groups
    for proj in projected:
        label_cells = proj[0]
        label = clean(" ".join(c.text for c in label_cells)).rstrip(":").strip()
        values = [to_cell(cs) for cs in proj[1:]]
        has_values = any(v.ctype != "EMPTY" for v in values)
        if not has_values and len(label) >= CONFIG["note_min_len"]:
            rows.append(Row(label, 0, [label], "note", False, values))
            continue
        if not label and any(v.ctype in ("NUMBER", "DASH") for v in values) and rows and rows[-1].row_type == "data":
            label = rows[-1].label  # continuation row: Apple prints "Operating leases" once for 2 lines
        row_type = "heading" if label and not has_values else "data"
        indent = label_cells[0].indent_pt if label_cells else math.inf  # unlabeled row continues the group
        is_total = bool(TOTAL_RE.match(label))
        bare = is_total and closes_group(label, "")  # "Total", "Subtotal", "Total (2)(3)"
        owner = None
        if is_total:  # a bare total only closes a group that has something to add up
            owner = next((i for i in range(len(stack) - 1, -1, -1)
                          if closes_group(label, stack[i].label) and (not bare or stack[i].n_data >= 2)), None)
        if owner is not None:  # "Total X" belongs to open group X even when it isn't indented deeper
            del stack[owner + 1:]
        elif bare:  # bare total with no group to close: grand total
            stack.clear()
        elif row_type == "heading":
            while stack and indent - stack[-1].indent <= tol:
                stack.pop()
        else:
            if stack and stack[-1].mode is None:  # the first child always belongs to its heading and sets the style
                stack[-1].mode = "nested" if indent - stack[-1].indent > tol else "flat"
                stack[-1].child_indent = indent
            while stack and ((stack[-1].mode == "nested" and indent - stack[-1].indent <= tol)
                             or (stack[-1].mode == "flat" and indent < stack[-1].child_indent - tol)):
                stack.pop()
        path = [s.label for s in stack] + ([label] if label else [])
        rows.append(Row(label, len(stack), path, row_type, is_total, values))
        if row_type == "heading":
            stack.append(OpenGroup(indent, label))
        else:
            if stack:
                stack[-1].n_data += 1
            if owner is not None:
                stack.pop()
    return rows, conflicts



def is_units_line(t: str) -> bool:
    """True if a line is purely a units notice like ``(in millions)``."""
    return bool(UNITS_RE.search(t)) and t.startswith("(") and t.endswith(")")


def table_context(prev: list[tuple[str, bool]], header_text: str, fact_list: list[XbrlFact], company: str):
    """Infer a table's title, units, and intro sentence from its surrounding text.

    Looks back a few lines before the table (nearest first). Units are taken from a
    ``(in millions)``-style line, else from the header text, else inferred from the most
    common scale of the table's USD facts. The title is the nearest bold, heading-length
    line that is not a units line or the company name. The intro is a non-bold sentence
    sitting between the title and the table.

    Args:
        prev: Recent ``(text, is_bold)`` lines preceding the table.
        header_text: Concatenated text of the table's header rows.
        fact_list: XBRL facts referenced by the table (used to guess units from scale).
        company: Registrant name, so it isn't mistaken for the table title.

    Returns:
        A tuple ``(title, units, intro)``; any element may be ``None`` if not found.
    """
    cands = prev[-CONFIG["title_lookback"]:][::-1]  # nearest first
    units = next((t if is_units_line(t) else m.group(0) for t, _ in cands if (m := UNITS_RE.search(t))), None)
    if not units and (m := UNITS_RE.search(header_text)):
        units = next((p for p in re.findall(r"\([^()]*\)", header_text) if UNITS_RE.search(p)), m.group(0))
    if not units:
        scales = Counter(f.scale for f in fact_list if f.unit and f.unit.lower() == "usd")
        units = SCALE_TO_UNITS.get(scales.most_common(1)[0][0]) if scales else None
    usable = [(t, b) for t, b in cands if not is_units_line(t) and norm_label(t) != norm_label(company)]
    title = next((t for t, b in usable if b and len(t) <= CONFIG["heading_max_len"]), None)
    before_title = usable[: next((i for i, (t, _) in enumerate(usable) if t == title), len(usable))]
    intro = next((t for t, b in before_title[:2] if not b and len(t) > 40), None)  # only text between title and table
    return title, units, intro



def purity(groups: dict) -> float | None:
    """Fraction of items that fall in the dominant category per group, averaged across groups (a consistency score)."""
    total = sum(sum(c.values()) for c in groups.values())
    return None if not total else sum(c.most_common(1)[0][1] for c in groups.values()) / total


def validate(rows: list[Row], columns: list[Column], facts: dict, contexts: dict) -> dict:
    """Cross-check the parsed table against its underlying XBRL facts.

    For every cell backed by a single numeric fact it checks that the parsed value matches
    the fact's shown/true value and that the sign agrees. It also tallies, per row and per
    column, the reporting periods and concepts of the facts to measure "purity" (how
    consistently each row/column maps to one period/concept). From the column period tallies
    it decides the period axis (columns vs rows), fills in each column's period, and — as a
    side effect — can rename or sanity-check column headers against their period year.

    Args:
        rows: The table's Rows (mutated only via their cells' fact ids).
        columns: The table's Columns; their ``period_start/end`` and ``name`` may be set here.
        facts: Mapping of fact id → XbrlFact.
        contexts: Mapping of context id → period/dimension info.

    Returns:
        A metrics dict (``n_fact_cells``, ``value_match``, ``sign_match``, ``period_axis``,
        ``period_purity``, ``concept_purity``, ``header_period_agreement``); keys are omitted
        or ``None`` when there are no fact-backed cells to measure.
    """
    by_col_p, by_row_p, by_col_c, by_row_c = (defaultdict(Counter) for _ in range(4))
    n = n_value = value_ok = sign_ok = 0
    for i, r in enumerate(rows):
        for j, c in enumerate(r.cells, start=1):
            f = facts.get(c.fact_ids[0]) if c.fact_ids else None
            if f is None:
                continue
            n += 1
            # only numeric cells: ranges ("4.70-5.75%", 2 facts) and prose ("20 million vehicles delivered") are kept verbatim
            if len(c.fact_ids) == 1 and f.shown is not None and c.ctype in ("NUMBER", "DASH"):
                n_value += 1
                if c.value is not None and any(math.isclose(abs(c.value), abs(x), rel_tol=1e-9, abs_tol=1e-9)
                                               for x in (f.shown, f.value)):
                    value_ok += 1
            if c.value is not None and (c.value == 0 or (c.value < 0) == (f.sign < 0)):
                sign_ok += 1
            ctx = contexts.get(f.context, {})
            p = (ctx.get("start"), ctx.get("end"))
            by_col_p[j][p] += 1
            by_row_p[i][p] += 1
            by_col_c[j][f.concept] += 1
            by_row_c[i][f.concept] += 1
    if not n:
        return {"n_fact_cells": 0}
    col_p, row_p = purity(by_col_p), purity(by_row_p)
    n_periods = len({p for cnt in by_col_p.values() for p in cnt})
    axis = None if n_periods == 1 else ("columns" if col_p >= row_p else "rows")
    agree = checked = 0
    for j, cnt in by_col_p.items():
        (start, end), _ = cnt.most_common(1)[0]
        if axis == "columns" or n_periods == 1:
            columns[j].period_start, columns[j].period_end = start, end
        if axis == "columns" and end and columns[j].name:
            checked += 1
            agree += end[:4] in columns[j].name
        elif axis == "columns" and end:
            columns[j].name = f"{start} to {end}" if start else f"As of {end}"
    return {
        "n_fact_cells": n,
        "value_match": round(value_ok / n_value, 3) if n_value else None,
        "sign_match": round(sign_ok / n, 3),
        "period_axis": axis,
        "period_purity": round(max(col_p, row_p), 3),
        "concept_purity": round(max(purity(by_row_c), purity(by_col_c)), 3),
        "header_period_agreement": round(agree / checked, 3) if checked else None,
    }


def score(kind, n_header, title, units, conflicts, n_cells, v) -> dict:
    """Compute a 0-1 confidence score for a parsed table, with human-readable reasons.

    Starts at 1.0 and subtracts penalties: a financial table missing a header (-0.3) or units
    (-0.1), any table missing a title (-0.1), cell-alignment conflicts (scaled by their share
    of cells, capped at -0.4), and validation shortfalls (weighted ``value_match``,
    ``period_purity``, ``concept_purity``, ``header_period_agreement`` below 1.0).

    Args:
        kind: Table kind ("financial", "textual", ...).
        n_header: Number of header rows detected.
        title / units: Inferred title and units (``None`` if missing).
        conflicts: Alignment conflict count from row/column projection.
        n_cells: Total data cells (denominator for the conflict penalty).
        v: Validation metrics dict from ``validate``.

    Returns:
        ``v`` merged with ``score`` (rounded, floored at 0), a ``low_confidence`` flag, and a
        ``reasons`` list naming each penalty applied.
    """
    s, reasons = 1.0, []
    if kind == "financial" and n_header == 0:
        s -= 0.3
        reasons.append("no_header")
    if kind == "financial" and not units:
        s -= 0.1
        reasons.append("no_units")
    if not title:
        s -= 0.1
        reasons.append("no_title")
    if conflicts:
        s -= min(0.4, 2 * conflicts / max(1, n_cells))
        reasons.append(f"conflicts={conflicts}")
    if v.get("n_fact_cells"):
        for key, w in (("value_match", 0.4), ("period_purity", 0.3), ("concept_purity", 0.2), ("header_period_agreement", 0.2)):
            if v.get(key) is not None and v[key] < 1:
                s -= w * (1 - v[key])
                reasons.append(f"{key}={v[key]}")
    s = round(max(0.0, s), 3)
    return {**v, "score": s, "low_confidence": s < CONFIG["low_confidence"], "reasons": reasons}



def fmt_value(c: Cell) -> str:
    """Format a cell's value for Markdown output, preserving decimals, percent sign, and em-dash for blanks."""
    if c.ctype == "DASH":
        return "—"
    if c.value is None or c.ctype == "YEAR":
        return c.text.replace("|", "\\|")
    m = re.search(r"\.(\d+)", c.text)
    return f"{c.value:,.{len(m.group(1)) if m else 0}f}" + ("%" if c.is_percent else "")


def to_markdown(t: Table) -> str:
    """Render a parsed Table as a Markdown table (with title, units, breadcrumb labels, and footnotes)."""
    units = t.units if not t.units or t.units.startswith("(") else f"({t.units})"
    head = f"**{t.title or 'Table'}**" + (f" {units}" if units and t.units not in (t.title or "") else "")
    if t.kind == "text_block":
        return "\n".join(r.label for r in t.rows)
    names = [t.columns[0].name or "Line item"] + [c.name or f"Column {j}" for j, c in enumerate(t.columns[1:], 1)]
    lines = [head, "", "| " + " | ".join(n.replace("|", "\\|") for n in names) + " |", "|" + "---|" * len(names)]
    notes = []
    for r in t.rows:
        if r.row_type == "note":
            notes.append(r.label)
        elif r.row_type == "data":
            label = " > ".join(r.path).replace("|", "\\|")
            lines.append("| " + " | ".join([label] + [fmt_value(c) for c in r.cells]) + " |")
    return "\n".join(lines + ([""] + [f"_Note: {n}_" for n in notes] if notes else []))



def parse_table(tag, table_id, prev_text, meta, facts, contexts) -> Table:
    """Run the full parsing pipeline for a single HTML table.

    Flattens the table (``raw_rows`` + ``merge_fragments``) and classifies it; layout/toc
    tables return early as empty shells. Otherwise it locates the label column and header
    rows, derives column groups, builds typed rows, and — for single-column tables — returns
    a ``text_block``. For real multi-column tables it names the columns, validates them
    against XBRL facts, infers title/units/intro, scores quality, and renders Markdown.

    Args:
        tag: The BeautifulSoup ``<table>`` tag.
        table_id: Stable id assigned to this table (``<doc_id>_tNNN``).
        prev_text: Recent ``(text, is_bold)`` lines preceding the table (for title/units).
        meta: Document metadata (uses ``company``).
        facts: Mapping of fact id → XbrlFact.
        contexts: Mapping of context id → period/dimension info.

    Returns:
        A Table whose ``kind`` reflects how it was handled (layout/toc/text_block/financial/
        textual), populated with columns, rows, quality metrics, and Markdown as applicable.
    """
    merged = [merge_fragments(r) for r in raw_rows(tag)]
    kind = classify(merged)
    if kind in ("layout", "toc"):
        return Table(table_id, kind)
    lstart = label_start(merged)
    n_header = header_row_count(merged, lstart)
    if not any(merged[n_header:]):  # every row looked like a header: it's a small text grid, keep all rows as data
        n_header = 0
    header_rows, data_rows = merged[:n_header], merged[n_header:]
    groups = column_groups([r for r in data_rows if r], lstart)
    if not groups:
        return Table(table_id, "layout")
    rows, conflicts = build_rows(data_rows, groups)
    if len(groups) == 1:
        return Table(table_id, "text_block", rows=rows, markdown="\n".join(r.label for r in rows))
    columns = [Column(n) for n in column_names(header_rows, groups)]
    v = validate(rows, columns, facts, contexts)
    fact_list = [facts[f] for r in rows for c in r.cells for f in c.fact_ids if f in facts]
    header_text = " ".join(c.text for r in header_rows for c in r)
    title, units, intro = table_context(prev_text, header_text, fact_list, meta["company"])
    n_cells = sum(len(r) for r in data_rows)
    t = Table(table_id, kind, title, units, intro, columns, rows, v.get("period_axis"),
              score(kind, n_header, title, units, conflicts, n_cells, v))
    t.markdown = to_markdown(t)
    return t


def parse_document(path: Path):
    """Parse one 10-K filing end-to-end into an ordered Document.

    Loads the HTML, reads document metadata, XBRL contexts and facts, strips noise, and walks
    the block elements in reading order. Tables are parsed via ``parse_table`` (layout/toc are
    dropped, text blocks become paragraphs). Text blocks are filtered for boilerplate, tracked
    for Part/Item section changes and pending section titles, and emitted as ``heading`` or
    ``paragraph`` elements. Each element carries the current part/section/section_title so the
    downstream chunker can keep context.

    Args:
        path: Path to the filing's HTML file.

    Returns:
        A tuple ``(doc, facts, contexts, removed)``: the populated Document, the fact and
        context maps, and a Counter of how many blocks were removed by category.
    """
    soup = load(path)
    meta = doc_metadata(soup, path)
    contexts = parse_contexts(soup)
    facts = {f.id: f for f in map(parse_fact, soup.find_all("ix:nonfraction"))}
    removed = strip_noise(soup)
    blocks = list(iter_blocks(soup.body))
    boilerplate = find_boilerplate([clean(b.get_text("")) for b in blocks if b.name != "table"])

    doc = Document(**meta)
    part, section, section_title = "Front Matter", "Cover Page", "Cover Page"
    pending_title, prev_text, n_tables = False, [], 0

    def add(etype, text=None, table=None):
        doc.elements.append(Element(meta["doc_id"], len(doc.elements), etype, part, section, section_title, text, table))

    for b in blocks:
        if b.name == "table":
            n_tables += 1
            t = parse_table(b, f"{meta['doc_id']}_t{n_tables:03d}", prev_text, meta, facts, contexts)
            prev_text = []
            if t.kind in ("layout", "toc"):
                removed[f"{t.kind}_tables"] += 1
            elif t.kind == "text_block":
                add("paragraph", text=t.markdown)
            else:
                add("table", table=t)
            continue
        text = clean(b.get_text(""))
        bold = is_bold(b)
        if is_boilerplate(text, boilerplate, bold):
            removed["boilerplate_lines"] += 1
            continue
        sec = detect_section(text, bold)
        if sec and sec[0] == "part":
            part, section, section_title, etype = sec[1], None, sec[1], "heading"
        elif sec:
            section, section_title, pending_title = sec[1], sec[2], sec[2] is None
            etype = "heading"
        elif pending_title and bold and len(text) <= CONFIG["heading_max_len"]:
            section_title, pending_title, etype = text, False, "heading"
        else:
            etype = "heading" if bold and len(text) <= CONFIG["heading_max_len"] else "paragraph"
        add(etype, text=text)
        prev_text.append((text, bold))
    return doc, facts, contexts, removed



def save(doc: Document, facts: dict, contexts: dict, out_dir: Path = OUT_DIR) -> None:
    """Write a document's metadata (.document.json), elements (.elements.jsonl), and facts (.facts.jsonl) to disk."""
    meta = {k: v for k, v in asdict(doc).items() if k != "elements"}
    (out_dir / f"{doc.doc_id}.document.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(out_dir / f"{doc.doc_id}.elements.jsonl", "w", encoding="utf-8") as fh:
        for e in doc.elements:
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    with open(out_dir / f"{doc.doc_id}.facts.jsonl", "w", encoding="utf-8") as fh:
        for row in facts_records(facts, contexts, doc.doc_id):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


QUALITY_COLUMNS = ["doc_id", "table_id", "section", "kind", "title", "units", "rows", "cols", "score",
                   "n_fact_cells", "value_match", "period_purity", "concept_purity",
                   "header_period_agreement", "reasons"]


def quality_rows(doc: Document) -> list[dict]:
    """Flatten each table's quality metrics into rows for the quality_report.csv summary."""
    return [{"doc_id": doc.doc_id, "table_id": e.table.table_id, "section": e.section, "kind": e.table.kind,
             "title": e.table.title, "units": e.table.units, "rows": len(e.table.rows),
             "cols": len(e.table.columns),
             **{k: e.table.quality.get(k) for k in QUALITY_COLUMNS[8:14]},
             "reasons": ";".join(e.table.quality.get("reasons", []))}
            for e in doc.elements if e.table]


def parse_all(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR, files: list[Path] | None = None) -> list[Document]:
    """Parse and save every filing (or the given files), write the combined quality_report.csv, and return the documents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs, rows = [], []
    for path in files or sorted(raw_dir.glob("*.html")):
        doc, facts, contexts, removed = parse_document(path)
        save(doc, facts, contexts, out_dir)
        docs.append(doc)
        rows += quality_rows(doc)
        types = Counter(e.type for e in doc.elements)
        logger.info(f"{doc.doc_id}: {dict(types)} xbrl_facts={len(facts)} removed={dict(removed)}")
    with open(out_dir / "quality_report.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUALITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return docs


def main() -> None:
    """CLI entry point: parse the HTML files given on the command line (or the whole raw dir) into the output dir."""
    parser = argparse.ArgumentParser(description="Parse 10-K Inline-XBRL HTML filings into elements.jsonl.")
    parser.add_argument("files", nargs="*", type=Path, help="HTML files (default: every *.html in --raw-dir)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    parse_all(args.raw_dir, args.out_dir, args.files or None)


if __name__ == "__main__":
    main()
