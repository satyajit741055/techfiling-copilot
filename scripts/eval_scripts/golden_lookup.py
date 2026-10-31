"""Lookup helper for building retrieval goldens (read-only).

uv run python scripts/eval_scripts/golden_lookup.py show AAPL_10-K_2023 357            # full element JSON (text / markdown)
uv run python scripts/eval_scripts/golden_lookup.py around AAPL_10-K_2023 357 3        # neighbours +-3 (one line each)
uv run python scripts/eval_scripts/golden_lookup.py find "Total net sales" --doc AAPL_10-K_2023
uv run python scripts/eval_scripts/golden_lookup.py find "416,161" --ticker AAPL       # every filing of one company
uv run python scripts/eval_scripts/golden_lookup.py find "digital assets"               # all 15 filings
uv run python scripts/eval_scripts/golden_lookup.py toc MSFT_10-K_2024 "Item 1A"        # headings of a section
"""

import argparse
import json
import re
from pathlib import Path

PARSED = Path(__file__).resolve().parents[2] / "data" / "parsed"


def load(doc_id):
    return [
        json.loads(line) for line in (PARSED / f"{doc_id}.elements.jsonl").open(encoding="utf-8")
    ]


def text(e):
    return (e["table"] or {}).get("markdown") or e["text"] or ""


def line(e, width=160):
    tid = (e["table"] or {}).get("table_id", "")
    body = re.sub(r"\s+", " ", text(e))[:width]
    return f"{e['doc_id']}#{e['order']} [{e['type']}|{e['section']}|{tid}] {body}"


def docs(args):
    names = sorted(p.name.removesuffix(".elements.jsonl") for p in PARSED.glob("*.elements.jsonl"))
    if getattr(args, "doc", None):
        return [args.doc]
    if getattr(args, "ticker", None):
        return [d for d in names if d.startswith(args.ticker + "_")]
    return names


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("show")
s.add_argument("doc")
s.add_argument("order", type=int)
s = sub.add_parser("around")
s.add_argument("doc")
s.add_argument("order", type=int)
s.add_argument("n", type=int, nargs="?", default=3)
s = sub.add_parser("find")
s.add_argument("needle")
s.add_argument("--doc")
s.add_argument("--ticker")
s.add_argument("--max", type=int, default=60)
s = sub.add_parser("toc")
s.add_argument("doc")
s.add_argument("section")
args = ap.parse_args()

if args.cmd == "show":
    e = next(e for e in load(args.doc) if e["order"] == args.order)
    print(json.dumps({k: v for k, v in e.items() if k != "table"}, ensure_ascii=False, indent=1))
    if e["table"]:
        print(
            "table_id:",
            e["table"]["table_id"],
            "| title:",
            e["table"]["title"],
            "| units:",
            e["table"]["units"],
        )
        print(e["table"]["markdown"])
elif args.cmd == "around":
    for e in load(args.doc):
        if abs(e["order"] - args.order) <= args.n:
            print(("> " if e["order"] == args.order else "  ") + line(e))
elif args.cmd == "find":
    needle, hits = args.needle.lower(), 0
    for d in docs(args):
        for e in load(d):
            if needle in text(e).lower():
                hits += 1
                if hits <= args.max:
                    print(line(e))
    print(f"-- {hits} element(s) contain {args.needle!r}")
elif args.cmd == "toc":
    for e in load(args.doc):
        if e["section"] == args.section and e["type"] == "heading":
            print(line(e, 120))
