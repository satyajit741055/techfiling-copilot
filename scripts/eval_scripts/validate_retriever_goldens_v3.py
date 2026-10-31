#!/usr/bin/env python3
"""
Validate the 15-filing retrieval golden dataset (AAPL/MSFT/TSLA x FY2021-FY2025).

Companion of goldens/retriever_golden_prompt_15filings.md. Errors must be 0; every warning must
be reviewed (fixed, or explained as a false positive).

Checks
- schema, sequential ids, vocabularies (scope_type, category, difficulty)
- retrieval scope: doc_id / filters / group_by / groups / group_limit are valid for the retrievers,
  match the scope_type, and every relevant item's filing lies inside the scope
- multi-filing queries: every in-scope filing (every group) has a relevance-3 item
- every cited order exists; no element in two items; no heading at relevance 3
- every item carries a verbatim `evidence` quote found in one of its elements (anti-hallucination)
- every number in expected_answer appears in a cited element (or in derived_values)
- category rules (risk_factor -> Item 1A, cross_section -> >= 2 sections in one filing, ...)
- cited elements appear verbatim, in order, in the raw SEC HTML
- warnings: unlabeled duplicates of cited facts, open-corpus answers that are not unique to one
  filing, negative search terms that do occur, query wording, distribution vs. targets
- the eval loader (eval/hybrid_rerank_eval.py) accepts the file

Usage:
    uv run python scripts/eval_scripts/validate_retriever_goldens_v3.py --golden goldens/golden_retriever_15filings.jsonl
    uv run python scripts/eval_scripts/validate_retriever_goldens_v3.py --golden <file> --partial   # while building
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from validate_retriever_goldens import (  # noqa: E402
    PARSED_DIR,
    Report,
    check_sources,
    element_text,
    load_docs,
    numbers_in,
    tokens,
)

from vector_store.filters import check_group_by, matches, merge_filters  # noqa: E402

DEFAULT_GOLDEN = ROOT / "goldens" / "golden_retriever_15filings.jsonl"
NEGATIVE_ANSWER = "Not disclosed in these filings"
COMPANIES = {"AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla"}
YEARS = range(2021, 2026)
FILTER_KEYS = {"doc_id", "ticker", "fiscal_year", "form"}
MAX_GROUPS = 6
DEFAULT_GROUP_LIMIT = 3

SCOPE_TYPES = {
    "single_doc": 69,
    "open_corpus": 6,
    "cross_year": 12,
    "cross_company": 9,
    "cross_company_year": 3,
    "negative": 6,
}
MULTI = {"cross_year", "cross_company", "cross_company_year"}
CONTENT = {  # single_doc + open_corpus
    "exact_numeric": 15,
    "trend_explanation": 12,
    "risk_factor": 12,
    "accounting_policy": 9,
    "business_description": 9,
    "cross_section": 9,
    "table_structure": 9,
}
CATEGORIES = set(CONTENT) | {"negative"}
DIFFICULTY = {"easy": 30, "medium": 45, "hard": 30}
PER_FILING = 5
MECHANISMS = {
    "cross_year": {"grouped_fiscal_year": 6, "range": 3, "list": 3},
    "cross_company": {"grouped_ticker": 5, "grouped_doc_id": 2, "list": 2},
    "cross_company_year": {"grouped_ticker": 1, "grouped_doc_id": 1, "discovery": 1},
}
REQUIRED = (
    "id",
    "query",
    "scope_type",
    "doc_id",
    "category",
    "difficulty",
    "expected_answer",
    "relevant",
)
OPTIONAL = (
    "filters",
    "group_by",
    "groups",
    "group_limit",
    "derived_values",
    "negative_search_terms",
    "notes",
)
ITEM_KEYS = {"match_by", "value", "relevance", "doc_id", "evidence", "reason"}
INTERNAL_ID_RE = re.compile(r"\b(order|element)\s*#?\d+|_t\d{3}\b|10-K_20\d\d|doc_id|chunk", re.I)
MIN_EVIDENCE_TOKENS = 4


def load_meta() -> dict[str, dict]:
    return {
        p.name.removesuffix(".document.json"): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(PARSED_DIR.glob("*.document.json"))
    }


def mechanism(g: dict) -> str:
    if g.get("group_by"):
        return "discovery" if g.get("groups") is None else f"grouped_{g['group_by']}"
    filters = g.get("filters") or {}
    if any(isinstance(v, dict) for v in filters.values()):
        return "range"
    if any(isinstance(v, list) for v in filters.values()):
        return "list"
    return "scalar" if filters else "none"


def contains_tokens(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


class Validator:
    def __init__(self, docs: dict[str, list[dict]], meta: dict[str, dict], rep: Report):
        self.docs, self.meta, self.rep = docs, meta, rep
        self.by_doc = {d: {e["order"]: e for e in els} for d, els in docs.items()}
        self.toks = {
            d: {e["order"]: tokens(element_text(e)) for e in els} for d, els in docs.items()
        }

    # ---------------------------------------------------------------- scope
    def in_scope(self, qid: str, g: dict) -> list[str] | None:
        """Filings the query's retrieval scope can return, or None if the scope is invalid."""
        filters, group_by, groups = g.get("filters"), g.get("group_by"), g.get("groups")
        if filters is not None and (not isinstance(filters, dict) or not filters):
            self.rep.err(qid, "filters must be a non-empty object or null")
            return None
        if filters and (bad := set(filters) - FILTER_KEYS):
            self.rep.err(qid, f"filter keys {sorted(bad)} not allowed (use {sorted(FILTER_KEYS)})")
        try:
            scope = merge_filters(filters, doc_id=g.get("doc_id"))
            if group_by is not None:
                check_group_by(group_by, scope, groups)
            elif groups is not None or g.get("group_limit") is not None:
                raise ValueError("groups/group_limit need group_by")
        except (ValueError, TypeError) as exc:
            self.rep.err(qid, f"invalid scope: {exc}")
            return None
        if group_by:
            if groups is not None:
                if len(groups) > MAX_GROUPS:
                    self.rep.err(qid, f"{len(groups)} groups (max {MAX_GROUPS})")
                if len(set(map(str, groups))) != len(groups):
                    self.rep.err(qid, "duplicate group values")
                valid = {self.meta[d][group_by] for d in self.meta}
                for value in groups:
                    if value not in valid:
                        self.rep.err(
                            qid,
                            f"group value {value!r} ({type(value).__name__}) is not a {group_by} in the corpus",
                        )
            limit = g.get("group_limit")
            if limit is not None and (not isinstance(limit, int) or limit < 1):
                self.rep.err(qid, "group_limit must be a positive integer")
        for key, value in (filters or {}).items():
            values = (
                value.values()
                if isinstance(value, dict)
                else value
                if isinstance(value, list)
                else [value]
            )
            if key == "fiscal_year" and not all(isinstance(v, int) and v in YEARS for v in values):
                self.rep.err(
                    qid, f"fiscal_year filter values must be integers in 2021-2025: {value}"
                )
            if key == "ticker" and not all(v in COMPANIES for v in values):
                self.rep.err(qid, f"ticker filter values must be in {sorted(COMPANIES)}: {value}")
            if key == "doc_id" and not all(v in self.meta for v in values):
                self.rep.err(qid, f"unknown doc_id in filters: {value}")
        docs = [
            d
            for d, m in self.meta.items()
            if matches(m, scope) and (not group_by or groups is None or m[group_by] in groups)
        ]
        if not docs:
            self.rep.err(qid, "scope matches no filing")
            return None
        return docs

    def check_scope_type(self, qid: str, g: dict, docs: list[str]) -> None:
        st, doc_id, mech = g["scope_type"], g["doc_id"], mechanism(g)
        if st == "single_doc":
            if doc_id not in self.meta or g.get("filters") or g.get("group_by"):
                self.rep.err(qid, "single_doc needs a valid doc_id and no filters/group_by")
        elif st == "open_corpus":
            if doc_id is not None or g.get("filters") or g.get("group_by"):
                self.rep.err(qid, "open_corpus needs doc_id=null and no filters/group_by")
        elif st == "negative":
            if g.get("filters") or g.get("group_by"):
                self.rep.err(qid, "negative queries are scoped by doc_id only (or fully open)")
        else:
            if doc_id is not None:
                self.rep.err(qid, f"{st} needs doc_id=null (scope via filters/group_by)")
            tickers = {self.meta[d]["ticker"] for d in docs}
            years = {self.meta[d]["fiscal_year"] for d in docs}
            want = {
                "cross_year": (
                    len(tickers) == 1 and len(years) >= 2,
                    "one ticker and >= 2 fiscal years",
                ),
                "cross_company": (
                    len(tickers) >= 2 and len(years) == 1,
                    ">= 2 tickers and one fiscal year",
                ),
                "cross_company_year": (
                    len(tickers) >= 2 and len(years) >= 2,
                    ">= 2 tickers and >= 2 years",
                ),
            }[st]
            if not want[0]:
                self.rep.err(qid, f"{st} scope must cover {want[1]}; it covers {sorted(docs)}")
            if mech not in MECHANISMS[st]:
                self.rep.err(qid, f"{st} must use one of {sorted(MECHANISMS[st])}, got {mech}")
            if (
                st == "cross_year"
                and mech in ("range", "list")
                and not isinstance((g.get("filters") or {}).get("fiscal_year"), (dict, list))
            ):
                self.rep.err(
                    qid, "cross_year range/list queries must put the range/list on fiscal_year"
                )
            if (
                st == "cross_company_year"
                and mech == "grouped_ticker"
                and not isinstance((g.get("filters") or {}).get("fiscal_year"), dict)
            ):
                self.rep.err(
                    qid, "cross_company_year grouped_ticker query must use a fiscal_year range"
                )
            if mech == "discovery":
                groups = {self.meta[d][g["group_by"]] for d in docs}
                if len(groups) > (g.get("group_limit") or DEFAULT_GROUP_LIMIT):
                    self.rep.err(qid, f"discovery scope has {len(groups)} groups > group_limit")

    # ---------------------------------------------------------------- items
    def check_items(self, qid: str, g: dict, docs: list[str]) -> list[tuple[dict, dict]]:
        cited, seen = [], set()
        for r in g["relevant"]:
            if not isinstance(r, dict):
                self.rep.err(qid, f"relevant item is not an object: {r}")
                continue
            if extra := set(r) - ITEM_KEYS:
                self.rep.warn(qid, f"unknown item keys {sorted(extra)}")
            d = r.get("doc_id")
            if r.get("match_by") != "element_orders":
                self.rep.err(qid, f"match_by must be 'element_orders': {r}")
            if r.get("relevance") not in (1, 2, 3):
                self.rep.err(qid, f"relevance must be 1, 2 or 3: {r.get('relevance')}")
            if not isinstance(r.get("reason"), str) or not r["reason"].strip():
                self.rep.err(qid, f"{d}: item needs a non-empty reason")
            values = r.get("value")
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(o, int) for o in values)
            ):
                self.rep.err(qid, f"{d}: value must be a non-empty list of integer orders")
                continue
            if d not in self.meta:
                self.rep.err(qid, f"item doc_id {d!r} missing or unknown (required on every item)")
                continue
            if d not in docs:
                self.rep.err(qid, f"item filing {d} is outside the query's retrieval scope")
            elements = []
            for o in values:
                if (d, o) in seen:
                    self.rep.err(qid, f"{d}#{o} cited in more than one item")
                seen.add((d, o))
                e = self.by_doc[d].get(o)
                if e is None:
                    self.rep.err(qid, f"{d}#{o} does not exist")
                    continue
                elements.append(e)
                cited.append((r, e))
                if r.get("relevance") == 3 and e["type"] == "heading":
                    self.rep.err(qid, f"{d}#{o} is a heading but has relevance 3")
                if e.get("table") and e["table"]["table_id"] not in r.get("reason", ""):
                    self.rep.warn(
                        qid,
                        f"{d}#{o} is table {e['table']['table_id']} but reason does not name it",
                    )
            evidence = r.get("evidence")
            ev = tokens(evidence) if isinstance(evidence, str) else []
            if len(ev) < MIN_EVIDENCE_TOKENS:
                self.rep.err(
                    qid,
                    f"{d} {values}: evidence must be a verbatim quote of >= {MIN_EVIDENCE_TOKENS} words",
                )
            elif elements and not any(
                contains_tokens(self.toks[d][e["order"]], ev) for e in elements
            ):
                self.rep.err(
                    qid,
                    f"{d} {values}: evidence not found verbatim in any of the item's elements: {evidence[:80]!r}",
                )
        return cited

    # ---------------------------------------------------------------- query
    def check_query(self, g: dict, idx: int) -> None:
        rep = self.rep
        qid = g.get("id", f"line{idx + 1}")
        missing = [k for k in REQUIRED if k not in g]
        if missing:
            rep.err(qid, f"missing fields {missing}")
            return
        if extra := set(g) - set(REQUIRED) - set(OPTIONAL):
            rep.warn(qid, f"unknown fields {sorted(extra)}")
        if qid != f"q{idx + 1:03d}":
            rep.err(qid, f"id should be q{idx + 1:03d}")
        st, cat = g["scope_type"], g["category"]
        if st not in SCOPE_TYPES:
            rep.err(qid, f"unknown scope_type {st}")
            return
        if cat not in CATEGORIES:
            rep.err(qid, f"unknown category {cat}")
        if g["difficulty"] not in DIFFICULTY:
            rep.err(qid, f"unknown difficulty {g['difficulty']}")
        if INTERNAL_ID_RE.search(g["query"]):
            rep.err(qid, "query mentions internal identifiers")
        if not isinstance(g["relevant"], list):
            rep.err(qid, "relevant must be a list")
            return

        docs = self.in_scope(qid, g)
        if docs is None:
            return
        self.check_scope_type(qid, g, docs)
        self.check_wording(qid, g, docs)

        if st == "negative" or cat == "negative":
            if st != "negative" or cat != "negative":
                rep.err(qid, "negative queries need scope_type=negative and category=negative")
            if g["relevant"]:
                rep.err(qid, "negative query must have relevant=[]")
            if g["expected_answer"] != NEGATIVE_ANSWER:
                rep.err(qid, f"negative expected_answer must be exactly {NEGATIVE_ANSWER!r}")
            self.check_negative_terms(qid, g, docs)
            return
        if not g["relevant"]:
            rep.err(qid, "positive query has no relevant items")
            return

        cited = self.check_items(qid, g, docs)
        grade3 = defaultdict(list)
        for r, e in cited:
            if r.get("relevance") == 3:
                grade3[e["doc_id"]].append(e)
        if not grade3:
            rep.err(qid, "no relevance=3 item")
        if st in MULTI:
            for d in docs:
                if d not in grade3:
                    rep.err(
                        qid,
                        f"in-scope filing {d} has no relevance-3 item (every filing/group needs evidence)",
                    )
        if st == "open_corpus" and len({e["doc_id"] for _, e in cited}) != 1:
            rep.err(qid, "open_corpus evidence must come from exactly one filing")

        cited_numbers = (
            set().union(*(numbers_in(element_text(e)) for _, e in cited)) if cited else set()
        )
        derived = (
            set().union(*(numbers_in(x) for x in g.get("derived_values") or []))
            if g.get("derived_values")
            else set()
        )
        for n in sorted(numbers_in(g["expected_answer"]) - derived):
            if n not in cited_numbers:
                rep.err(qid, f"expected_answer number '{n}' not found in any cited element")
        if cat == "exact_numeric" and not numbers_in(g["expected_answer"]):
            rep.err(qid, "exact_numeric expected_answer contains no number")

        if cat == "cross_section":
            secs = defaultdict(set)
            for _, e in cited:
                secs[e["doc_id"]].add(e["section"])
            if not any(len(s) >= 2 for s in secs.values()):
                rep.err(
                    qid, f"cross_section needs >= 2 sections within one filing, got {dict(secs)}"
                )
        if cat == "risk_factor":
            for d, els in grade3.items():
                if not any(e["section"] == "Item 1A" for e in els):
                    rep.err(qid, f"risk_factor: {d} has no relevance-3 element in Item 1A")
        if cat == "table_structure" and not any(
            e["type"] == "table" for els in grade3.values() for e in els
        ):
            rep.err(qid, "table_structure needs a table at relevance 3")

        self.coverage_warnings(qid, g, cited)
        if st == "open_corpus":
            self.uniqueness_warnings(qid, g, cited)

    def check_wording(self, qid: str, g: dict, docs: list[str]) -> None:
        """The query must name every company it covers and, for one filing, its fiscal year."""
        st, q = g["scope_type"], g["query"].lower()
        if st == "open_corpus":
            docs = [
                r["doc_id"]
                for r in g["relevant"]
                if isinstance(r, dict) and r.get("doc_id") in self.meta
            ]
        elif st == "negative" and g["doc_id"] is None:
            return
        for ticker in sorted({self.meta[d]["ticker"] for d in docs}):
            if COMPANIES[ticker].lower() not in q and ticker.lower() not in q:
                self.rep.warn(qid, f"query does not name {COMPANIES[ticker]}")
        if st in ("single_doc", "open_corpus", "negative"):
            for y in sorted({self.meta[d]["fiscal_year"] for d in docs}):
                if str(y) not in q:
                    self.rep.warn(qid, f"query does not state fiscal year {y}")

    def check_negative_terms(self, qid: str, g: dict, docs: list[str]) -> None:
        terms = g.get("negative_search_terms")
        if (
            not isinstance(terms, list)
            or len(terms) < 4
            or not all(isinstance(t, str) and t.strip() for t in terms)
        ):
            self.rep.err(qid, "negative query needs negative_search_terms (>= 4 non-empty strings)")
            return
        for term in terms:
            needle = tokens(term)
            hits = [
                f"{d}#{o}"
                for d in docs
                for o, toks in self.toks[d].items()
                if needle and contains_tokens(toks, needle)
            ]
            if hits:
                self.rep.warn(
                    qid,
                    f"negative term {term!r} occurs in {len(hits)} element(s), e.g. {hits[:4]}; confirm none answers the query",
                )

    def coverage_warnings(self, qid: str, g: dict, cited: list[tuple[dict, dict]]) -> None:
        answer_nums = {
            n
            for n in numbers_in(g["expected_answer"])
            if len(n.replace(".", "")) >= 4 and not n.endswith("%")
        }
        labeled = {(e["doc_id"], e["order"]) for _, e in cited}
        for d in sorted({e["doc_id"] for _, e in cited}):
            sigs = {
                e["order"]: set(self.toks[d][e["order"]])
                for _, e in cited
                if e["doc_id"] == d and len(self.toks[d][e["order"]]) >= 8
            }
            for e in self.docs[d]:
                if (d, e["order"]) in labeled or e["type"] == "heading":
                    continue
                toks = set(self.toks[d][e["order"]])
                dup = next(
                    (
                        o
                        for o, sig in sigs.items()
                        if toks
                        and e["type"] == self.by_doc[d][o]["type"]
                        and len(sig & toks) / len(sig | toks) >= 0.6
                    ),
                    None,
                )
                if dup is not None:
                    self.rep.warn(
                        qid,
                        f"{d}#{e['order']} [{e['section']}] near-duplicate of cited #{dup} but unlabeled",
                    )
                elif e.get("table") and g["category"] in ("exact_numeric", "table_structure"):
                    if hits := answer_nums & numbers_in(element_text(e)):
                        self.rep.warn(
                            qid,
                            f"{d}#{e['order']} table {e['table']['table_id']} contains answer value(s) {sorted(hits)} but unlabeled",
                        )

    def uniqueness_warnings(self, qid: str, g: dict, cited: list[tuple[dict, dict]]) -> None:
        own = {e["doc_id"] for _, e in cited}
        probes = [
            tokens(r["evidence"])
            for r in g["relevant"]
            if r.get("relevance") == 3 and isinstance(r.get("evidence"), str)
        ]
        answer_nums = {
            n for n in numbers_in(g["expected_answer"]) if len(n.replace(".", "").rstrip("%")) >= 4
        }
        for d in self.docs:
            if d in own:
                continue
            for e in self.docs[d]:
                toks = self.toks[d][e["order"]]
                if any(len(p) >= MIN_EVIDENCE_TOKENS and contains_tokens(toks, p) for p in probes):
                    self.rep.warn(
                        qid,
                        f"open_corpus evidence also appears in {d}#{e['order']}; the answer must be unique to one filing",
                    )
                    break
                if hits := answer_nums & numbers_in(element_text(e)):
                    self.rep.warn(
                        qid,
                        f"open_corpus answer value(s) {sorted(hits)} also appear in {d}#{e['order']}; the answer must be unique to one filing",
                    )
                    break


def item_docs(g: dict, meta: dict[str, dict]) -> set[str]:
    return {
        r.get("doc_id")
        for r in g.get("relevant") or []
        if isinstance(r, dict) and r.get("doc_id") in meta
    }


def distribution(
    gold: list[dict],
    meta: dict[str, dict],
    by_doc: dict[str, dict[int, dict]],
    rep: Report,
    partial: bool,
) -> list[str]:
    st = Counter(g.get("scope_type") for g in gold)
    diffs = Counter(g.get("difficulty") for g in gold)
    single = [g for g in gold if g.get("scope_type") in ("single_doc", "open_corpus")]
    cats = Counter(g.get("category") for g in single)
    per_filing, open_by_ticker = Counter(), Counter()
    for g in single:
        docs = item_docs(g, meta) if g["scope_type"] == "open_corpus" else {g.get("doc_id")}
        per_filing.update(docs)
        if g["scope_type"] == "open_corpus":
            open_by_ticker.update(meta[d]["ticker"] for d in docs)
    mechs = {s: Counter(mechanism(g) for g in gold if g.get("scope_type") == s) for s in MECHANISMS}
    cross_year_tickers, cross_company_years = Counter(), set()
    for g in gold:
        if g.get("scope_type") == "cross_year":
            cross_year_tickers.update({meta[d]["ticker"] for d in item_docs(g, meta)})
        elif g.get("scope_type") == "cross_company":
            cross_company_years.update(meta[d]["fiscal_year"] for d in item_docs(g, meta))
    neg_scoped = sum(1 for g in gold if g.get("scope_type") == "negative" and g.get("doc_id"))
    grade3_types, sections = Counter(), Counter()
    for g in gold:
        types, secs = set(), set()
        for r in g.get("relevant") or []:
            if not isinstance(r, dict) or not isinstance(r.get("value"), list):
                continue
            for o in r["value"]:
                e = by_doc.get(r.get("doc_id"), {}).get(o)
                if e:
                    secs.add(e["section"])
                    if r.get("relevance") == 3:
                        types.add(e["type"])
        grade3_types.update(types)
        sections.update(secs)

    lines = [
        f"scope_type : {dict(st)} (target {SCOPE_TYPES})",
        f"categories : {dict(cats)} (single_doc+open_corpus; target {CONTENT})",
        f"difficulty : {dict(diffs)} (target {DIFFICULTY})",
        f"per filing : {dict(sorted(per_filing.items(), key=lambda x: str(x[0])))} (target {PER_FILING} each)",
        f"mechanisms : { {s: dict(c) for s, c in mechs.items()} }",
        f"rel-3 type : {dict(grade3_types)} (target >= 30 table, >= 30 paragraph)",
        f"sections   : {dict(sorted(sections.items(), key=lambda x: str(x[0])))}",
    ]
    if partial:
        return lines

    def want(name: str, got: int, target: int) -> None:
        if got != target:
            rep.warn("dataset", f"{name}: {got} (target {target})")

    want("total queries", len(gold), sum(SCOPE_TYPES.values()))
    for k, n in SCOPE_TYPES.items():
        want(f"scope_type {k}", st[k], n)
    for k, n in CONTENT.items():
        want(f"category {k}", cats[k], n)
    for k, n in DIFFICULTY.items():
        want(f"difficulty {k}", diffs[k], n)
    for d in meta:
        want(f"single_doc+open_corpus queries for {d}", per_filing[d], PER_FILING)
    for t in COMPANIES:
        want(f"open_corpus queries for {t}", open_by_ticker[t], 2)
        want(f"cross_year queries for {t}", cross_year_tickers[t], 4)
    for s, targets in MECHANISMS.items():
        for m, n in targets.items():
            want(f"{s} mechanism {m}", mechs[s][m], n)
    if missing := set(YEARS) - cross_company_years:
        rep.warn("dataset", f"cross_company queries miss fiscal years {sorted(missing)}")
    want("negative queries scoped to a doc_id", neg_scoped, 4)
    for t in ("table", "paragraph"):
        if grade3_types[t] < 30:
            rep.warn(
                "dataset",
                f"only {grade3_types[t]} queries have a {t} at relevance 3 (target >= 30)",
            )
    for s in ("Item 1", "Item 1A", "Item 7", "Item 8"):
        if sections[s] < 5:
            rep.warn("dataset", f"section {s} cited by {sections[s]} queries (target >= 5)")
    if sum(sections[s] for s in ("Item 1C", "Item 2", "Item 7A", "Item 15")) < 3:
        rep.warn("dataset", "fewer than 3 queries cite Item 1C / Item 2 / Item 7A / Item 15")
    queries = Counter(g.get("query", "").strip().lower() for g in gold)
    for q, n in queries.items():
        if n > 1:
            rep.err("dataset", f"duplicate query text ({n}x): {q[:80]}")
    return lines


def check_eval_loader(path: Path, rep: Report) -> None:
    try:
        from hybrid_rerank_eval import load_golden_dataset

        load_golden_dataset(path)
    except Exception as exc:  # noqa: BLE001 - report any loader failure
        rep.err("dataset", f"eval/hybrid_rerank_eval.py cannot load the file: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument(
        "--partial", action="store_true", help="skip distribution targets (while building)"
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--skip-html", action="store_true", help="skip raw HTML checks (faster)")
    ap.add_argument(
        "--skip-loader", action="store_true", help="skip the eval loader check (faster)"
    )
    args = ap.parse_args()

    docs, meta = load_docs(), load_meta()
    rep = Report()
    gold = []
    for n, line in enumerate(args.golden.open(encoding="utf-8"), start=1):
        if line.strip():
            try:
                gold.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rep.err(f"line{n}", f"invalid JSON: {exc}")
    if len({g.get("id") for g in gold}) != len(gold):
        rep.err("dataset", "duplicate ids")

    validator = Validator(docs, meta, rep)
    for i, g in enumerate(gold):
        validator.check_query(g, i)
    lines = distribution(gold, meta, validator.by_doc, rep, args.partial)

    if not args.skip_html:
        cited: dict[str, set[int]] = {}
        for g in gold:
            for r in g.get("relevant", []):
                if (
                    isinstance(r, dict)
                    and r.get("doc_id") in docs
                    and isinstance(r.get("value"), list)
                ):
                    cited.setdefault(r["doc_id"], set()).update(
                        o for o in r["value"] if isinstance(o, int)
                    )
        check_sources(docs, cited, rep)
    if not args.skip_loader and not rep.errors:
        check_eval_loader(args.golden, rep)

    print(f"Golden: {args.golden}  ({len(gold)} queries){'  [partial]' if args.partial else ''}")
    print("\n".join(lines))
    print(f"\nERRORS ({len(rep.errors)})")
    print("\n".join(f"  {e}" for e in rep.errors) or "  none")
    print(f"\nWARNINGS ({len(rep.warnings)})")
    print("\n".join(f"  {w}" for w in rep.warnings) or "  none")
    return 1 if rep.errors or (args.strict and rep.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
