"""Check how much of each raw 10-K HTML file survived parsing into data/parsed/, and give a confidence score.

Deliberately independent of the parser (exploration/00_explore_src.py): it re-reads the HTML with its own
minimal logic, so a parser bug cannot hide itself by passing its own test.

Five checks per document, combined into one confidence score:
  words     - multiset recall of visible words (after removing content that is excluded *by design*)
  blocks    - share of HTML text blocks (paragraphs/headings) found verbatim in parsed paragraphs
  facts     - share of visible XBRL numeric facts present in the parsed output (linked to a cell, or in text)
  structure - are tables *arranged* correctly? (works for tables with and without XBRL)
                rows:    every parsed row equals exactly one HTML row, in order (no merged/split/dropped cells)
                lost:    HTML rows not in the parsed table must be header rows (their words appear in the header)
                columns: all values of one parsed column come from one physical HTML column (no shifted values)
                totals:  every "Total ..." row equals the sum of the rows above it in its group (hierarchy check)
  tables    - mean parser quality score of the parsed tables

Usage:
    uv run python scripts/check_coverage.py              # report + exit code 1 on FAIL
    uv run python scripts/check_coverage.py --strict     # exit code 1 on WARN too (for CI)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parents[1]
BLOCK_TAGS = ["div", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "center"]
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,'][a-z0-9]+)*")
UNITS_RE = re.compile(r"\bin\s+(millions|thousands|billions)\b", re.I)
PAGE_NUM_RE = re.compile(r"^(?:page\s*)?(?:\d{1,3}|[ivxlc]{1,6})$|^[-–—]\s*\d{1,3}\s*[-–—]$", re.I)

WEIGHTS = {"words": 0.25, "blocks": 0.15, "facts": 0.25, "structure": 0.25, "tables": 0.10}
PASS = {"confidence": 0.95, "words": 0.97, "facts": 0.99, "structure": 0.95}
TABLE_MATCH_MIN = 0.5  # Jaccard of row labels needed to pair an HTML table with a parsed table
FAIL = {"confidence": 0.85, "facts": 0.95}
REPEAT_MIN, REPEAT_MAX_LEN = 5, 80  # page header/footer lines: short and repeated


def clean(s: str | None) -> str:
    return WS_RE.sub(" ", (s or "").replace("\xa0", " ").replace("’", "'").replace("‘", "'")).strip()


def tokens(s: str | None) -> list[str]:
    return TOKEN_RE.findall(clean(s).lower())


def norm(s: str | None) -> str:
    return " ".join(tokens(s))


def top_level_tables(soup):
    return [t for t in soup.body.find_all("table") if t.find_parent("table") is None]


def leaf_blocks(soup):
    for el in soup.body.find_all(BLOCK_TAGS):
        if el.find_parent("table") is None and el.find(BLOCK_TAGS + ["table"]) is None and clean(el.get_text("")):
            yield el


# ---------------------------------------------------------------- HTML side
def load_html(path: Path):
    """Return (soup with by-design exclusions removed, Counter of excluded tokens, exclusion stats, all-visible tokens)."""
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for t in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):  # hidden XBRL header: never content
        t.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" ")
    for tag in soup.body.find_all(BLOCK_TAGS + ["td", "th"]):  # word boundaries at blocks/cells only, not inside
        tag.append(" ")                                           # inline spans ("jurisd<span>ictions</span>")
    all_visible = Counter(tokens(soup.body.get_text("")))

    excluded, stats = Counter(), Counter()
    for t in soup.find_all(style=re.compile(r"position\s*:\s*absolute", re.I)):  # page header/footer containers
        if t.parent is not None:
            excluded.update(tokens(t.get_text("")))
            stats["page_furniture_blocks"] += 1
            t.decompose()
    for tb in top_level_tables(soup):  # table of contents: mostly internal "#" links
        cells = [td for td in tb.find_all("td") if clean(td.get_text(""))]
        if cells and sum(td.find("a", href=re.compile(r"^#")) is not None for td in cells) >= 0.5 * len(cells):
            excluded.update(tokens(tb.get_text("")))
            stats["toc_tables"] += 1
            tb.decompose()
    blocks = list(leaf_blocks(soup))
    texts = [clean(b.get_text("")) for b in blocks]
    repeats = Counter(re.sub(r"\d+", "#", t.lower()) for t in texts if len(t) <= REPEAT_MAX_LEN)
    for b, t in zip(blocks, texts):
        repeated = len(t) <= REPEAT_MAX_LEN and repeats[re.sub(r"\d+", "#", t.lower())] >= REPEAT_MIN
        if PAGE_NUM_RE.match(t) or (repeated and not UNITS_RE.search(t)):
            excluded.update(tokens(t))
            stats["repeated_or_page_number_lines"] += 1
            b.decompose()
    return soup, excluded, stats, all_visible


# ---------------------------------------------------------------- parsed side
def load_parsed(parsed_dir: Path, doc_id: str):
    elements = [json.loads(line) for line in open(parsed_dir / f"{doc_id}.elements.jsonl", encoding="utf-8")]
    strings, para_norms, label_norms, cell_facts, scores, tables = [], [], [], {}, [], []
    for e in elements:
        if e["text"]:
            strings.append(e["text"])
            para_norms.append(norm(e["text"]))
        if t := e["table"]:
            tables.append(t)
            if t.get("quality", {}).get("score") is not None:
                scores.append((t["table_id"], t["quality"]["score"], t["quality"].get("low_confidence", False)))
            strings += [c["name"] for c in t["columns"]] + [t.get("units") or ""]  # units may come from a header cell
            for r in t["rows"]:
                strings.append(r["label"])
                label_norms.append(norm(r["label"]))
                for c in r["cells"]:
                    strings.append(c["text"])
                    for fid in c["fact_ids"]:
                        cell_facts[fid] = c["text"]
    return {
        "tokens": Counter(tok for s in strings for tok in tokens(s)),
        "para_blob": " | ".join(para_norms),
        "text_blob": " | ".join(para_norms + label_norms),
        "cell_facts": cell_facts,
        "table_scores": scores,
        "tables": tables,
    }


# ---------------------------------------------------------------- checks
def check_words(html_tokens: Counter, parsed_tokens: Counter, top_n: int):
    total = sum(html_tokens.values())
    covered = sum(min(n, parsed_tokens[w]) for w, n in html_tokens.items())
    missing = Counter({w: n - parsed_tokens[w] for w, n in html_tokens.items() if n > parsed_tokens[w]})
    return covered / total if total else 1.0, total, covered, missing.most_common(top_n)


def check_blocks(soup, parsed, top_n: int):
    blocks = [clean(b.get_text("")) for b in leaf_blocks(soup)]
    missing = [t for t in blocks if norm(t) and norm(t) not in parsed["para_blob"]]
    return 1 - len(missing) / len(blocks) if blocks else 1.0, len(blocks), [m[:160] for m in missing[:top_n]]


def check_facts(soup, parsed, top_n: int):
    status, lost = Counter(), []
    for f in soup.find_all("ix:nonfraction"):
        fid, shown = f.get("id"), clean(f.get_text(""))
        digits = re.sub(r"\D", "", shown)
        if fid in parsed["cell_facts"]:
            ok = digits in re.sub(r"\D", "", parsed["cell_facts"][fid]) or shown in ("—", "-", "–")
            status["linked_value_ok" if ok else "linked_value_mismatch"] += 1
            continue
        container = f.find_parent("td") or f.find_parent(BLOCK_TAGS)
        if container is not None and norm(container.get_text("")) in parsed["text_blob"]:
            status["in_text_only"] += 1  # present (e.g. inside a row label) but not linked to a cell
            continue
        status["missing"] += 1
        if len(lost) < top_n:
            lost.append({"id": fid, "concept": f.get("name"), "shown": shown,
                         "context": clean(container.get_text(" ") if container else "")[:160]})
    total = sum(status.values())
    found = total - status["missing"] - status["linked_value_mismatch"]
    return found / total if total else 1.0, total, dict(status), lost


# ---------------------------------------------------------------- structure (table arrangement)
def html_grid(table) -> list[list[tuple[int, list[str]]]]:
    """Rows of (physical right edge, tokens) for non-empty cells, with colspan/rowspan expanded.

    The right edge, not the start: numbers are right-aligned, and "$" may sit in its own cell in one row
    but share a colspan=2 cell in the next, so starts differ (4 vs 3) while the right edge is (nearly) stable.
    Column check: a physical position must be claimed by ONE parsed column. A shifted value shows up as the
    same position claimed by two parsed columns; harmless edge drift (5 vs 6 within one column) does not."""
    rows, carried = [], defaultdict(dict)  # rowspan cells are filled down: the spanned rows show that text too
    for r, tr in enumerate(tr for tr in table.find_all("tr") if tr.find_parent("table") is table):
        col, cells = 0, []
        for td in tr.find_all(["td", "th"], recursive=False):
            while col in carried[r]:
                span, toks = carried[r].pop(col)
                if toks:
                    cells.append((col + span, toks))
                col += span
            cs, rs = max(1, int(td.get("colspan") or 1)), max(1, int(td.get("rowspan") or 1))
            toks = tokens(td.get_text(""))
            if toks:
                cells.append((col + cs, toks))
            for rr in range(r + 1, r + rs):
                carried[rr][col] = (cs, toks)
            col += cs
        cells += [(c + span, toks) for c, (span, toks) in carried.pop(r, {}).items() if toks]
        rows.append(sorted(cells))
    return rows


def row_tokens(row: dict) -> list[str]:
    return tokens(row["label"]) + [t for c in row["cells"] for t in tokens(c["text"])]


def check_rows_and_columns(grid, table: dict) -> dict:
    hseq = [[t for _, toks in cells for t in toks] for cells in grid]
    used, pos, matched, unmatched = set(), 0, 0, []
    col_obs, unaligned = defaultdict(Counter), 0
    prows = [r for r in table["rows"] if row_tokens(r)]
    inherited, prev_label = 0, None
    for r in prows:
        ptoks, n_label = row_tokens(r), len(tokens(r["label"]))
        k = next((k for k in range(pos, len(hseq)) if hseq[k] == ptoks), None)
        if k is None:  # also look backwards once, in case of reordering
            k = next((k for k in range(len(hseq)) if k not in used and hseq[k] == ptoks), None)
        if k is None and r["label"] and r["label"] == prev_label:  # label inherited from the row above
            k = next((k for k in range(pos, len(hseq)) if hseq[k] == ptoks[n_label:]), None)
            inherited, n_label = inherited + (k is not None), 0
        prev_label = r["label"]
        if k is None:
            unmatched.append(" ".join(ptoks)[:120])
            continue
        matched += 1
        used.add(k)
        pos = k + 1
        cells, i, consumed = grid[k], 0, 0
        while i < len(cells) and consumed < n_label:
            consumed += len(cells[i][1])
            i += 1
        pcells = [(j, tokens(c["text"])) for j, c in enumerate(r["cells"]) if tokens(c["text"])]
        hcells = cells[i:]
        if len(hcells) == len(pcells) and all(h[1] == p[1] for h, p in zip(hcells, pcells)):
            for (phys, _), (j, _) in zip(hcells, pcells):
                col_obs[phys][j] += 1  # which parsed columns claim this physical position?
        else:
            unaligned += 1  # one parsed cell = several HTML cells (or vice versa): can't attribute columns
    header_vocab = set(tokens(" ".join([c["name"] for c in table["columns"]] +
                                       [table.get("title") or "", table.get("units") or "", table.get("intro") or ""])))
    lost = [" ".join(hseq[k])[:120] for k in range(len(hseq))
            if hseq[k] and k not in used and not set(hseq[k]) <= header_vocab]
    obs = sum(sum(c.values()) for c in col_obs.values())
    return {
        "parsed_rows": len(prows), "matched_rows": matched, "unmatched_rows": unmatched, "inherited_label_rows": inherited,
        "html_rows": sum(1 for s in hseq if s), "lost_html_rows": lost,
        "column_obs": obs, "column_pure": sum(c.most_common(1)[0][1] for c in col_obs.values()), "unaligned_rows": unaligned,
    }


def parse_value(text: str):
    s = clean(text).replace("$", "").replace(",", "").replace(" ", "").replace("−", "-")
    if not s:
        return None
    if "%" in s:
        return "pct"
    if s in ("—", "–", "-", "--"):
        return 0.0
    neg = s.startswith("(") or s.startswith("-")
    try:
        v = float(s.strip("()-"))
    except ValueError:
        return "text"
    return -v if neg else v


def decimals(text: str) -> int:
    m = re.search(r"\.(\d+)", text)
    return len(m.group(1)) if m else 0


def match_suffix(items, total_vals, total_texts):
    """Find k so that the last k items sum to the total in as many numeric columns as possible.

    Returns (k, columns matched, columns checked); k=0 if no column matched, None if nothing to check.
    Scored per column because some columns are never additive (average price, %, rates)."""
    cols = [j for j, v in enumerate(total_vals) if isinstance(v, float)]
    if not items or not cols:
        return None
    best = (0, 0, len(cols))
    for k in [*range(2, len(items) + 1), 1]:  # prefer real sums (k>=2); k=1 is a "total" of one row
        suffix, hits = items[-k:], 0
        for j in cols:
            vals = [x[j] for x in suffix if j < len(x) and isinstance(x[j], float)]
            tol = 0.5 * 10 ** -decimals(total_texts[j]) * (k + 1) + 1e-9
            hits += bool(vals) and abs(sum(vals) - total_vals[j]) <= tol
        if hits > best[1]:
            best = (k, hits, len(cols))
        if hits == len(cols):
            break
    return best


def check_totals(table: dict):
    """-> (column checks passed, column checks done, failed totals)

    Each parent path holds an ordered list of entries: a row's values, or a child group (heading).
    A child group counts as its closing total if it has one, else as the sum of its own entries."""
    entries, closed = defaultdict(list), {}
    passed, checked, failed = 0, 0, []
    ncols = len(table["columns"]) - 1

    def resolve(entry):
        kind, x = entry
        if kind == "row":
            return x
        if x in closed:
            return closed[x]
        vecs = [resolve(e) for e in entries[x]]
        out = []
        for j in range(ncols):
            nums = [v[j] for v in vecs if j < len(v) and isinstance(v[j], float)]
            out.append(sum(nums) if nums else None)
        return out

    for r in table["rows"]:
        path = tuple(r["path"])
        parent = path[:-1]
        if r["row_type"] == "heading":
            entries[parent].append(("group", path))
            continue
        if r["row_type"] != "data":
            continue
        texts = [c["text"] for c in r["cells"]]
        vals = [parse_value(t) for t in texts]
        label = re.sub(r"\((?:\d{1,2}|[a-z])\)", "", r["label"], flags=re.I)  # footnote markers: "Total (2)(3)", "(a)"
        if re.match(r"(sub)?total\b", label, re.I) and any(isinstance(v, float) for v in vals):
            res = match_suffix([resolve(e) for e in entries[parent]], vals, texts)
            if res is not None:
                k, hits, n = res
                passed, checked = passed + hits, checked + n
                if hits < n:
                    failed.append(f"{table['table_id']}: {' > '.join(r['path'])} ({hits}/{n} columns add up)")
                if k:
                    del entries[parent][-k:]
            entries[parent].append(("row", vals))
            rest = re.sub(r"^(sub)?total\s*", "", norm(label))
            if parent and rest in ("", norm(re.sub(r"\((?:\d{1,2}|[a-z])\)", "", parent[-1], flags=re.I))):  # closing total stands for the whole group one level up
                closed[parent] = vals
        else:
            entries[parent].append(("row", vals))
    return passed, checked, failed


def check_structure(soup, parsed, top_n: int):
    html_tables = [(t, html_grid(t)) for t in top_level_tables(soup)]
    html_tables = [(t, g) for t, g in html_tables if any(g)]
    sig = lambda labels: {x for x in labels if x}
    html_sigs = [sig(" ".join(cells[0][1]) for cells in g if cells) for _, g in html_tables]
    html_rows = [{" ".join(t for _, toks in cells for t in toks) for cells in g if cells} for _, g in html_tables]
    agg, problems, totals_pass, totals_checked, totals_fail = Counter(), [], 0, 0, []
    used = set()
    for t in parsed["tables"]:
        prow = [s for s in (" ".join(row_tokens(r)) for r in t["rows"]) if s]
        psig = sig(norm(r["label"]) for r in t["rows"])
        # exact row text is the main signal (labels repeat across similar tables); label overlap breaks ties
        cands = [(sum(s in html_rows[i] for s in prow),
                  len(psig & html_sigs[i]) / len(psig | html_sigs[i]) if psig | html_sigs[i] else 0, -i)
                 for i in range(len(html_tables)) if i not in used]
        hits, jac, neg_i = max(cands, default=(0, 0, None))
        i = -neg_i if neg_i is not None else None
        if hits < 0.5 * len(prow) and jac < TABLE_MATCH_MIN:
            agg["parsed_tables_unpaired"] += 1
            problems.append({"table_id": t["table_id"], "issue": "no matching HTML table"})
            continue
        used.add(i)
        res = check_rows_and_columns(html_tables[i][1], t)
        for key in ("parsed_rows", "matched_rows", "html_rows", "column_obs", "column_pure", "unaligned_rows",
                    "inherited_label_rows"):
            agg[key] += res[key]
        agg["lost_html_rows"] += len(res["lost_html_rows"])
        p, n, f = check_totals(t)
        totals_pass, totals_checked = totals_pass + p, totals_checked + n
        totals_fail += f
        if res["unmatched_rows"] or res["lost_html_rows"] or res["column_pure"] < res["column_obs"]:
            problems.append({"table_id": t["table_id"], "title": t.get("title"),
                             "unmatched_rows": res["unmatched_rows"][:3], "lost_html_rows": res["lost_html_rows"][:3],
                             "column_purity": round(res["column_pure"] / res["column_obs"], 3) if res["column_obs"] else None})
    unpaired_html = [g for i, (_, g) in enumerate(html_tables) if i not in used
                     and norm(" ".join(" ".join(t for _, toks in cells for t in toks) for cells in g)) not in parsed["para_blob"]]
    n_checked = totals_checked
    parts = {
        "rows": agg["matched_rows"] / agg["parsed_rows"] if agg["parsed_rows"] else 1.0,
        "lost": 1 - agg["lost_html_rows"] / agg["html_rows"] if agg["html_rows"] else 1.0,
        "columns": agg["column_pure"] / agg["column_obs"] if agg["column_obs"] else 1.0,
        "totals": totals_pass / n_checked if n_checked else 1.0,
        "tables_paired": 1 - (len(unpaired_html) + agg["parsed_tables_unpaired"]) / max(1, len(html_tables)),
    }
    score = sum(parts.values()) / len(parts)
    detail = {**{k: round(v, 4) for k, v in parts.items()}, **dict(agg),
              "html_tables": len(html_tables), "html_tables_unpaired": len(unpaired_html),
              "totals_checked": n_checked, "totals_passed": totals_pass, "totals_failed": totals_fail[:top_n],
              "problem_tables": problems[:top_n]}
    return score, detail


def check_document(html_path: Path, parsed_dir: Path, doc_id: str, top_n: int) -> dict:
    soup, excluded, excl_stats, all_visible = load_html(html_path)
    parsed = load_parsed(parsed_dir, doc_id)
    html_tokens = all_visible - excluded

    raw_words, *_ = check_words(all_visible, parsed["tokens"], 0)
    words, w_total, w_covered, w_missing = check_words(html_tokens, parsed["tokens"], top_n)
    blocks, b_total, b_missing = check_blocks(soup, parsed, top_n)
    facts, f_total, f_status, f_lost = check_facts(soup, parsed, top_n)
    structure, s_detail = check_structure(soup, parsed, top_n)
    scores = parsed["table_scores"]
    tables = sum(s for _, s, _ in scores) / len(scores) if scores else 1.0

    parts = {"words": words, "blocks": blocks, "facts": facts, "structure": structure, "tables": tables}
    confidence = sum(WEIGHTS[k] * v for k, v in parts.items())
    if confidence < FAIL["confidence"] or facts < FAIL["facts"]:
        verdict = "FAIL"
    elif confidence >= PASS["confidence"] and all(parts[k] >= PASS[k] for k in ("words", "facts", "structure")):
        verdict = "PASS"
    else:
        verdict = "WARN"
    return {
        "doc_id": doc_id,
        "source_file": html_path.name,
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "scores": {k: round(v, 4) for k, v in parts.items()},
        "words": {"coverage_all_visible": round(raw_words, 4), "coverage_adjusted": round(words, 4),
                  "html_tokens_adjusted": w_total, "covered": w_covered, "excluded_by_design_tokens": sum(excluded.values()),
                  "excluded_by_design": dict(excl_stats), "top_missing": w_missing},
        "blocks": {"total": b_total, "missing_examples": b_missing},
        "facts": {"visible": f_total, **f_status, "missing_examples": f_lost},
        "structure": s_detail,
        "tables": {"parsed": len(scores), "mean_score": round(tables, 4),
                   "low_confidence": [tid for tid, _, low in scores if low]},
    }


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw_html")
    ap.add_argument("--parsed-dir", type=Path, default=ROOT / "data" / "parsed")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path (default: <parsed-dir>/coverage_report.json)")
    ap.add_argument("--top", type=int, default=10, help="how many missing examples to report per check")
    ap.add_argument("--strict", action="store_true", help="exit 1 on WARN as well as FAIL")
    args = ap.parse_args()

    docs = {json.loads(p.read_text(encoding="utf-8"))["source_file"]: json.loads(p.read_text(encoding="utf-8"))["doc_id"]
            for p in args.parsed_dir.glob("*.document.json")}
    reports = []
    for html_path in sorted(args.raw_dir.glob("*.htm*")):
        if html_path.name not in docs:
            reports.append({"source_file": html_path.name, "verdict": "FAIL", "confidence": 0.0,
                            "error": "no parsed output found for this file"})
            continue
        reports.append(check_document(html_path, args.parsed_dir, docs[html_path.name], args.top))

    out = args.out or args.parsed_dir / "coverage_report.json"
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["words", "blocks", "facts", "structure", "tables"]
    print(f"{'document':<18} {'verdict':<7} {'confidence':>10} " + " ".join(f"{c:>9}" for c in cols))
    for r in reports:
        if "error" in r:
            print(f"{r['source_file']:<18} FAIL    {r['error']}")
            continue
        print(f"{r['doc_id']:<18} {r['verdict']:<7} {r['confidence']:>10.1%} "
              + " ".join(f"{r['scores'][c]:>9.1%}" for c in cols))
    for r in reports:
        if "error" in r:
            continue
        print(f"\n--- {r['doc_id']}")
        w, f = r["words"], r["facts"]
        print(f"  words : {w['coverage_adjusted']:.2%} of content ({w['coverage_all_visible']:.2%} of all visible text); "
              f"{w['excluded_by_design_tokens']} tokens excluded by design {w['excluded_by_design']}")
        print(f"  facts : {f['visible']} visible | linked ok {f.get('linked_value_ok', 0)} | in text only {f.get('in_text_only', 0)} "
              f"| value mismatch {f.get('linked_value_mismatch', 0)} | missing {f.get('missing', 0)}")
        s = r["structure"]
        print(f"  struct: rows matched {s.get('matched_rows', 0)}/{s.get('parsed_rows', 0)} | lost HTML rows {s.get('lost_html_rows', 0)} "
              f"| column purity {s['columns']:.2%} ({s.get('column_obs', 0)} values) | unaligned rows {s.get('unaligned_rows', 0)} "
              f"| totals {s['totals_passed']}/{s['totals_checked']} | HTML tables {s['html_tables']} "
              f"(unpaired {s['html_tables_unpaired']}, parsed unpaired {s.get('parsed_tables_unpaired', 0)})")
        for t in s["totals_failed"]:
            print(f"  total does not add up: {t}")
        for p in s["problem_tables"]:
            print(f"  table issue : {p}")
        print(f"  tables: {r['tables']['parsed']} parsed, mean score {r['tables']['mean_score']:.3f}, "
              f"low confidence: {r['tables']['low_confidence'] or 'none'}")
        if w["top_missing"]:
            print("  top missing words:", ", ".join(f"{t}×{n}" for t, n in w["top_missing"]))
        for m in r["blocks"]["missing_examples"]:
            print(f"  missing block: {m}")
        for m in f["missing_examples"]:
            print(f"  missing fact : {m['id']} {m['concept']} = {m['shown']} | {m['context']}")
    print(f"\nreport written to {out}")

    bad = {"FAIL", "WARN"} if args.strict else {"FAIL"}
    return 1 if any(r["verdict"] in bad for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
