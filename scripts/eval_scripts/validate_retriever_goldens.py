#!/usr/bin/env python3
"""
Validate the retrieval golden dataset against the parsed elements AND the raw SEC HTML.

Source integrity (parsed vs raw HTML):
- element orders are unique and contiguous (0..N-1)
- every cited element's text (paragraph) / first data row (table) is found verbatim in the raw HTML
- cited elements appear in the HTML in the same relative order as their `order` values

Golden rules:
- schema, unique sequential ids, category/difficulty vocab
- every cited order exists for its doc_id; cross_document queries have doc_id=null and per-item doc_ids
- at least one relevance=3 item (positives); no heading as relevance=3; negatives have no items
- every number in expected_answer is present in a cited element (or listed in `derived_values`)
- cross_section items span >= 2 sections; risk_factor has a relevance=3 item in Item 1A
- warnings: unlabeled near-duplicate paragraphs/tables of cited elements, unlabeled tables that
  contain a numeric answer, distribution vs. the generation prompt

Usage:
    uv run python scripts/eval_scripts/validate_retriever_goldens.py
    uv run python scripts/eval_scripts/validate_retriever_goldens.py --golden goldens/golden_retriever_dataset.jsonl
    uv run python scripts/eval_scripts/validate_retriever_goldens.py --strict  # exit 1 on warnings
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parents[2]
PARSED_DIR = ROOT / "data" / "parsed"
RAW_DIR = ROOT / "data" / "raw_html"
DEFAULT_GOLDEN = ROOT / "goldens" / "golden_retriever_dataset_v2.jsonl"

CATEGORIES = {
    "exact_numeric": 10,
    "trend_explanation": 8,
    "risk_factor": 8,
    "accounting_policy": 8,
    "business_description": 6,
    "cross_section": 6,
    "cross_document": 6,
    "negative": 4,
    "table_structure": 4,
}
DIFFICULTY = {"easy": 20, "medium": 25, "hard": 15}
REQUIRED_SECTIONS = ["Item 1", "Item 1A", "Item 7", "Item 8"]
OPTIONAL_SECTIONS = ["Item 1C", "Item 2", "Item 15"]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,'][a-z0-9]+)*")
NUM_RE = re.compile(r"(?<![\w.])\$?\(?-?\d[\d,]*(?:\.\d+)?%?")
DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s+\d{1,2},?)?(?:\s+\d{4})?"
)
DUP_JACCARD = 0.6


def tokens(s: str | None) -> list[str]:
    s = (s or "").replace("\xa0", " ").replace("\u2019", "'").replace("\u2018", "'").lower()
    return TOKEN_RE.findall(s)


def element_text(e: dict) -> str:
    t = e.get("table")
    return (t.get("markdown") or "") if t else (e.get("text") or "")


def norm_number(raw: str) -> str | None:
    """'$416,161' -> '416161', '46.9%' -> '46.9%', '$0.26' -> '0.26'; years and tiny ints -> None."""
    s = raw.strip("$()").replace(",", "")
    pct = s.endswith("%")
    s = s.rstrip("%").lstrip("-")
    if not s or not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return None
    if not pct and "." not in s and (len(s) < 2 or (len(s) == 4 and 1990 <= int(s) <= 2035)):
        return None
    return s + ("%" if pct else "")


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in NUM_RE.findall(DATE_RE.sub(" ", text or "")):
        n = norm_number(m)
        if n:
            out.add(n)
            out.add(n.rstrip("%"))
    return out


def load_docs() -> dict[str, list[dict]]:
    docs = {}
    for p in sorted(PARSED_DIR.glob("*.elements.jsonl")):
        doc_id = p.name.removesuffix(".elements.jsonl")
        docs[doc_id] = [json.loads(line) for line in p.open(encoding="utf-8")]
    return docs


def load_html_tokens(doc_id: str) -> str:
    meta = json.loads((PARSED_DIR / f"{doc_id}.document.json").read_text(encoding="utf-8"))
    soup = BeautifulSoup((RAW_DIR / meta["source_file"]).read_bytes(), "lxml")
    for t in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        t.decompose()
    for tag in soup.body.find_all(["div", "p", "li", "td", "th", "br", "tr"]):
        tag.append(" ")
    return " " + " ".join(tokens(soup.body.get_text(""))) + " "


def element_probe(e: dict) -> str | None:
    """Token string that must appear verbatim in the HTML for this element."""
    if t := e.get("table"):
        for r in t.get("rows", []):
            toks = tokens(r["label"]) + [x for c in r["cells"] for x in tokens(c["text"])]
            if len(toks) >= 3:
                return " " + " ".join(toks) + " "
        return None
    toks = tokens(e.get("text"))
    return " " + " ".join(toks) + " " if len(toks) >= 3 else None


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, qid: str, msg: str):
        self.errors.append(f"[{qid}] {msg}")

    def warn(self, qid: str, msg: str):
        self.warnings.append(f"[{qid}] {msg}")


def check_sources(docs: dict[str, list[dict]], cited: dict[str, set[int]], rep: Report) -> None:
    for doc_id, els in docs.items():
        if [e["order"] for e in els] != list(range(len(els))):
            rep.err(doc_id, "element orders are not contiguous 0..N-1")
        if not cited.get(doc_id):
            continue
        html = load_html_tokens(doc_id)
        last_pos, last_order = -1, -1
        for e in els:
            if e["order"] not in cited[doc_id] or not (probe := element_probe(e)):
                continue
            pos = html.find(probe, last_pos + 1)
            if pos == -1:
                pos = html.find(probe)
                if pos == -1:
                    rep.err(doc_id, f"#{e['order']} ({e['type']}) not found verbatim in raw HTML")
                    continue
                rep.err(
                    doc_id, f"#{e['order']} appears in HTML before #{last_order} (order mismatch)"
                )
            last_pos, last_order = pos, e["order"]


def check_query(g: dict, idx: int, by_doc: dict[str, dict[int, dict]], docs, rep: Report) -> None:
    qid = g.get("id", f"line{idx + 1}")
    for key in ("id", "query", "doc_id", "category", "difficulty", "expected_answer", "relevant"):
        if key not in g:
            rep.err(qid, f"missing field {key}")
            return
    if qid != f"q{idx + 1:03d}":
        rep.err(qid, f"id should be q{idx + 1:03d}")
    if g["category"] not in CATEGORIES:
        rep.err(qid, f"unknown category {g['category']}")
    if g["difficulty"] not in DIFFICULTY:
        rep.err(qid, f"unknown difficulty {g['difficulty']}")
    if re.search(r"\b(order|element)\s+\d+", g["query"], re.I):
        rep.err(qid, "query mentions internal identifiers")

    rel = g["relevant"]
    if g["category"] == "negative":
        if rel:
            rep.err(qid, "negative query must have relevant=[]")
        if g["expected_answer"] != "Not disclosed in these filings":
            rep.err(qid, "negative expected_answer must be 'Not disclosed in these filings'")
        return
    if g["category"] == "cross_document":
        if g["doc_id"] is not None:
            rep.err(qid, "cross_document must have doc_id=null")
        if len({r.get("doc_id") for r in rel}) < 2:
            rep.err(qid, "cross_document must cite at least 2 doc_ids")
    elif g["doc_id"] not in by_doc:
        rep.err(qid, f"unknown doc_id {g['doc_id']}")
        return

    cited, sections, seen = [], set(), set()
    for r in rel:
        d = r.get("doc_id") or g["doc_id"]
        if r.get("match_by") != "element_orders" or not r.get("value"):
            rep.err(qid, f"bad relevant item {r}")
            continue
        if r.get("relevance") not in (1, 2, 3):
            rep.err(qid, f"relevance must be 1-3: {r}")
        if g["doc_id"] and r.get("doc_id") not in (None, g["doc_id"]):
            rep.err(qid, f"item doc_id {r.get('doc_id')} != query doc_id")
        for o in r["value"]:
            if (d, o) in seen:
                rep.err(qid, f"{d}#{o} cited in more than one item")
            seen.add((d, o))
            e = by_doc.get(d, {}).get(o)
            if e is None:
                rep.err(qid, f"{d}#{o} does not exist")
                continue
            cited.append((r, e))
            sections.add(e["section"])
            if r["relevance"] == 3 and e["type"] == "heading":
                rep.err(qid, f"{d}#{o} is a heading but has relevance 3")
            if e.get("table") and e["table"]["table_id"] not in r.get("reason", ""):
                rep.warn(
                    qid, f"{d}#{o} is table {e['table']['table_id']} but reason does not cite it"
                )
    if not any(r["relevance"] == 3 for r in rel):
        rep.err(qid, "no relevance=3 item")

    cited_numbers = (
        set().union(*(numbers_in(element_text(e)) for _, e in cited)) if cited else set()
    )
    derived = (
        set().union(*(numbers_in(x) for x in g.get("derived_values", [])))
        if g.get("derived_values")
        else set()
    )
    for n in sorted(numbers_in(g["expected_answer"]) - derived):
        if n not in cited_numbers:
            rep.err(qid, f"expected_answer number '{n}' not found in any cited element")

    if g["category"] == "cross_section" and len(sections) < 2:
        rep.err(qid, f"cross_section cites only {sections}")
    if g["category"] == "risk_factor" and not any(
        r["relevance"] == 3 and e["section"] == "Item 1A" for r, e in cited
    ):
        rep.err(qid, "risk_factor has no relevance=3 element in Item 1A")

    # coverage warnings: unlabeled equivalents
    for d in {r.get("doc_id") or g["doc_id"] for r in rel}:
        labeled = {o for (dd, o) in seen if dd == d}
        sigs = {
            e["order"]: set(tokens(element_text(e)))
            for r, e in cited
            if e["doc_id"] == d and len(tokens(element_text(e))) >= 8
        }
        answer_nums = {
            n
            for n in numbers_in(g["expected_answer"])
            if len(n.replace(".", "")) >= 4 and not n.endswith("%")
        }
        for e in docs[d]:
            if e["order"] in labeled or e["type"] == "heading":
                continue
            toks = set(tokens(element_text(e)))
            for o, sig in sigs.items():
                if (
                    toks
                    and len(sig & toks) / len(sig | toks) >= DUP_JACCARD
                    and e["type"] == by_doc[d][o]["type"]
                ):
                    rep.warn(
                        qid,
                        f"{d}#{e['order']} [{e['section']}] near-duplicate of cited #{o} but unlabeled",
                    )
                    break
            else:
                if e.get("table") and g["category"] in (
                    "exact_numeric",
                    "cross_document",
                    "table_structure",
                ):
                    hits = answer_nums & numbers_in(element_text(e))
                    if hits:
                        rep.warn(
                            qid,
                            f"{d}#{e['order']} table {e['table']['table_id']} contains answer value(s) {sorted(hits)} but unlabeled",
                        )


def distribution(gold: list[dict], by_doc, rep: Report) -> list[str]:
    lines = []
    cats = Counter(g["category"] for g in gold)
    diffs = Counter(g["difficulty"] for g in gold)
    comp = Counter(g["doc_id"] for g in gold if g["category"] not in ("negative", "cross_document"))
    primary = Counter()
    sec = Counter()
    for g in gold:
        types, secs = set(), set()
        for r in g["relevant"]:
            d = r.get("doc_id") or g["doc_id"]
            for o in r["value"]:
                e = by_doc[d].get(o)
                if e:
                    secs.add(e["section"])
                    if r["relevance"] == 3:
                        types.add(e["type"])
        primary.update(types)
        sec.update(secs)
    lines.append(f"categories : {dict(cats)}")
    lines.append(f"difficulty : {dict(diffs)} (target {DIFFICULTY})")
    lines.append(f"companies  : {dict(comp)} (excl. cross_document/negative)")
    lines.append(f"rel-3 type : {dict(primary)} (target >=20 table, >=20 paragraph)")
    lines.append(f"sections   : {dict(sorted(sec.items(), key=lambda x: str(x[0])))}")
    for c, n in CATEGORIES.items():
        if cats[c] != n:
            rep.warn("dataset", f"category {c}: {cats[c]} (target {n})")
    for s in REQUIRED_SECTIONS:
        if sec[s] < 2:
            rep.warn("dataset", f"section {s} cited by {sec[s]} queries (target >=2)")
    if not any(sec[s] for s in OPTIONAL_SECTIONS):
        rep.warn("dataset", f"no query cites any of {OPTIONAL_SECTIONS}")
    for t in ("table", "paragraph"):
        if primary[t] < 20:
            rep.warn(
                "dataset", f"only {primary[t]} queries have a {t} as relevance-3 (target >=20)"
            )
    if len(gold) != sum(CATEGORIES.values()):
        rep.warn("dataset", f"{len(gold)} queries (target {sum(CATEGORIES.values())})")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--skip-html", action="store_true", help="skip raw HTML checks (faster)")
    ap.add_argument(
        "--all-elements",
        action="store_true",
        help="HTML-check every parsed element, not only cited ones",
    )
    args = ap.parse_args()

    docs = load_docs()
    by_doc = {d: {e["order"]: e for e in els} for d, els in docs.items()}
    gold = [json.loads(line) for line in args.golden.open(encoding="utf-8") if line.strip()]
    rep = Report()

    if len({g.get("id") for g in gold}) != len(gold):
        rep.err("dataset", "duplicate ids")
    for i, g in enumerate(gold):
        check_query(g, i, by_doc, docs, rep)
    lines = distribution(gold, by_doc, rep)

    if not args.skip_html:
        cited: dict[str, set[int]] = {}
        for g in gold:
            for r in g["relevant"]:
                cited.setdefault(r.get("doc_id") or g["doc_id"], set()).update(r["value"])
        if args.all_elements:
            cited = {d: {e["order"] for e in els} for d, els in docs.items()}
        check_sources(docs, cited, rep)

    print(f"Golden: {args.golden}  ({len(gold)} queries)")
    print("\n".join(lines))
    print(f"\nERRORS ({len(rep.errors)})")
    print("\n".join(f"  {e}" for e in rep.errors) or "  none")
    print(f"\nWARNINGS ({len(rep.warnings)})")
    print("\n".join(f"  {w}" for w in rep.warnings) or "  none")
    return 1 if rep.errors or (args.strict and rep.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
