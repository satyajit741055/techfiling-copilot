#!/usr/bin/env python3
"""
Validate the generator-isolation golden dataset against the parsed filing elements.

Each dataset record supplies a query, an expected answer and an ideal context copied verbatim
from parsed elements in data/parsed/*.elements.jsonl. This validator proves schema correctness
and source identity; it does not prove semantic entailment of the expected answer.

Checks:
- file exists, valid UTF-8, no blank lines, one complete JSON object per line (no trailing content)
- required fields present with the expected types; no unexpected fields
- exactly 20 records (1..20 with --partial), IDs exactly q001..qNNN in order, no duplicate IDs
- non-empty doc_id / query / expected_answer / ideal_context / source_element_order
- ideal_context and source_element_order have equal length, at most two elements
- context values are strings; orders are integers (not bool), unique and ascending
- every (doc_id, order) exists in the parsed data and belongs to the record's document
- source element type is heading / paragraph / table with non-empty canonical context
- decoded ideal_context[i] exactly equals the canonical context of source_element_order[i]
  (heading/paragraph -> element["text"], table -> element["table"]["markdown"])
- no duplicate normalized queries and no exact duplicate records
- parsed source files are valid and contain no duplicate (doc_id, order) keys

Usage:
    uv run python scripts/eval_scripts/validate_generator_isolation_dataset.py
    uv run python scripts/eval_scripts/validate_generator_isolation_dataset.py path/to/dataset.jsonl
    uv run python scripts/eval_scripts/validate_generator_isolation_dataset.py --partial
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARSED_DIR = ROOT / "data" / "parsed"
DEFAULT_DATASET = ROOT / "goldens" / "golden_generator_isolation_dataset.jsonl"

EXPECTED_RECORDS = 20
MAX_CONTEXT_ELEMENTS = 2
SUPPORTED_TYPES = {"heading", "paragraph", "table"}
REQUIRED_FIELDS = {
    "source_qid": str,
    "doc_id": str,
    "query": str,
    "expected_answer": str,
    "ideal_context": list,
    "source_element_order": list,
}

Key = tuple[str, int]


class LoadError(Exception):
    """Raised when the dataset or source files cannot be loaded safely."""


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def err(self, rule: str, msg: str, *, line: int | None = None, qid: str | None = None,
            doc_id: str | None = None, order: int | None = None) -> None:
        loc = [f"line {line}" if line is not None else None,
               f"qid={qid}" if qid else None,
               f"doc_id={doc_id}" if doc_id else None,
               f"order={order}" if order is not None else None]
        prefix = ", ".join(x for x in loc if x)
        self.errors.append(f"[{rule}] {prefix + ': ' if prefix else ''}{msg}")


def is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def canonical_context(e: dict) -> str | None:
    if e.get("type") == "table":
        t = e.get("table")
        return t.get("markdown") if isinstance(t, dict) else None
    return e.get("text")


def load_sources(parsed_dir: Path) -> dict[Key, dict]:
    """Index all parsed elements by (doc_id, order). Raises LoadError on any source problem."""
    files = sorted(parsed_dir.glob("*.elements.jsonl"))
    if not files:
        raise LoadError(f"no source files matching {parsed_dir / '*.elements.jsonl'}")
    index: dict[Key, dict] = {}
    problems: list[str] = []
    for path in files:
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as ex:
            problems.append(f"{path.name}: invalid UTF-8 ({ex})")
            continue
        for n, raw in enumerate(text.split("\n"), 1):
            if not raw.strip():
                continue
            try:
                e = json.loads(raw)
            except json.JSONDecodeError as ex:
                problems.append(f"{path.name} line {n}: invalid JSON ({ex})")
                continue
            if not isinstance(e, dict):
                problems.append(f"{path.name} line {n}: not a JSON object")
                continue
            doc_id, order = e.get("doc_id"), e.get("order")
            if not isinstance(doc_id, str) or not doc_id or not is_int(order):
                problems.append(f"{path.name} line {n}: missing/invalid doc_id or order")
                continue
            if (doc_id, order) in index:
                problems.append(f"{path.name} line {n}: duplicate source key ({doc_id}, {order})")
                continue
            index[(doc_id, order)] = e
    if problems:
        raise LoadError("invalid source files:\n  " + "\n  ".join(problems))
    return index


def parse_dataset(path: Path, rep: Report) -> list[tuple[int, dict]]:
    """Return (line_number, record) for every line that parses to a JSON object."""
    if not path.exists():
        raise LoadError(f"dataset file does not exist: {path}")
    if not path.is_file():
        raise LoadError(f"dataset path is not a file: {path}")
    raw_lines = path.read_bytes().split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()  # trailing newline at end of file
    decoder = json.JSONDecoder()
    records: list[tuple[int, dict]] = []
    for n, raw in enumerate(raw_lines, 1):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as ex:
            rep.err("invalid_utf8", f"line is not valid UTF-8 ({ex})", line=n)
            continue
        if line.endswith("\r"):
            line = line[:-1]
        if not line.strip():
            rep.err("blank_line", "blank line", line=n)
            continue
        stripped = line.lstrip()
        try:
            obj, end = decoder.raw_decode(stripped)
        except json.JSONDecodeError as ex:
            rep.err("invalid_json", str(ex), line=n)
            continue
        if stripped[end:].strip():
            rep.err("trailing_content", f"extra content after JSON value at column "
                    f"{len(line) - len(stripped) + end + 1}: {stripped[end:end + 40]!r}", line=n)
            continue
        if not isinstance(obj, dict):
            rep.err("not_object", f"line is a JSON {type(obj).__name__}, expected object", line=n)
            continue
        records.append((n, obj))
    return records


def first_diff(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return (f"first difference at char {i}: dataset={a[i:i + 30]!r} source={b[i:i + 30]!r} "
            f"(len dataset={len(a)}, source={len(b)})")


def validate_record(n: int, rec: dict, index: dict[Key, dict], rep: Report) -> None:
    qid = rec.get("source_qid") if isinstance(rec.get("source_qid"), str) else None
    doc_id = rec.get("doc_id") if isinstance(rec.get("doc_id"), str) else None
    ctx = dict(line=n, qid=qid, doc_id=doc_id)

    type_ok = True
    for field, typ in REQUIRED_FIELDS.items():
        if field not in rec:
            rep.err("missing_field", f"missing required field '{field}'", **ctx)
            type_ok = False
        elif not isinstance(rec[field], typ):
            rep.err("wrong_type", f"field '{field}' must be {typ.__name__}, "
                    f"got {type(rec[field]).__name__}", **ctx)
            type_ok = False
    for field in sorted(set(rec) - set(REQUIRED_FIELDS)):
        rep.err("unexpected_field", f"unexpected field '{field}'", **ctx)
    if not type_ok:
        return

    for field in ("doc_id", "query", "expected_answer", "source_qid"):
        if not rec[field].strip():
            rep.err(f"empty_{field}", f"'{field}' is empty", **ctx)
    contexts, orders = rec["ideal_context"], rec["source_element_order"]
    if not contexts:
        rep.err("empty_ideal_context", "'ideal_context' is empty", **ctx)
    if not orders:
        rep.err("empty_source_element_order", "'source_element_order' is empty", **ctx)
    if len(contexts) != len(orders):
        rep.err("length_mismatch", f"ideal_context has {len(contexts)} items but "
                f"source_element_order has {len(orders)}", **ctx)
    if len(contexts) > MAX_CONTEXT_ELEMENTS or len(orders) > MAX_CONTEXT_ELEMENTS:
        rep.err("too_many_elements", f"more than {MAX_CONTEXT_ELEMENTS} context elements", **ctx)

    for i, c in enumerate(contexts):
        if not isinstance(c, str):
            rep.err("non_string_context", f"ideal_context[{i}] is {type(c).__name__}", **ctx)
    int_orders = []
    for i, o in enumerate(orders):
        if is_int(o):
            int_orders.append(o)
        else:
            rep.err("non_integer_order", f"source_element_order[{i}]={o!r} is not an integer", **ctx)
    if len(int_orders) == len(orders):
        dups = sorted(o for o, k in Counter(orders).items() if k > 1)
        if dups:
            rep.err("duplicate_order", f"duplicate source orders {dups}", **ctx)
        elif orders != sorted(orders):
            rep.err("orders_not_ascending", f"source orders {orders} are not ascending", **ctx)

    if not doc_id or not doc_id.strip():
        return
    known_docs = {d for d, _ in index}
    if doc_id not in known_docs:
        rep.err("unknown_doc_id", f"doc_id not found in parsed sources {sorted(known_docs)}", **ctx)
    for i, (c, o) in enumerate(zip(contexts, orders)):
        if not is_int(o):
            continue
        e = index.get((doc_id, o))
        if e is None:
            other = [d for d in sorted(known_docs) if d != doc_id and isinstance(c, str)
                     and (index.get((d, o)) or {}) and canonical_context(index[(d, o)]) == c]
            if other:
                rep.err("cross_document_context", f"ideal_context[{i}] matches element from "
                        f"different document {other[0]}", **ctx, order=o)
            else:
                rep.err("missing_reference", f"({doc_id}, {o}) not found in parsed sources",
                        **ctx, order=o)
            continue
        if e.get("doc_id") != doc_id:
            rep.err("cross_document_context", f"element doc_id {e.get('doc_id')!r} differs "
                    "from record doc_id", **ctx, order=o)
        etype = e.get("type")
        if etype not in SUPPORTED_TYPES:
            rep.err("unsupported_element_type", f"source element type {etype!r} not in "
                    f"{sorted(SUPPORTED_TYPES)}", **ctx, order=o)
            continue
        source = canonical_context(e)
        if not isinstance(source, str) or not source.strip():
            rep.err("empty_source_context", f"canonical context of {etype} element is null/empty",
                    **ctx, order=o)
            continue
        if not isinstance(c, str):
            continue
        if c != source:
            elsewhere = [k for k, v in index.items() if k[0] != doc_id and canonical_context(v) == c]
            if elsewhere:
                rep.err("cross_document_context", f"ideal_context[{i}] matches element "
                        f"{elsewhere[0][0]} order {elsewhere[0][1]}, not the record's document",
                        **ctx, order=o)
            else:
                rep.err("context_mismatch", f"ideal_context[{i}] != canonical {etype} context; "
                        + first_diff(c, source), **ctx, order=o)


def normalize_query(q: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", q.lower()).split())


def validate_dataset(records: list[tuple[int, dict]], index: dict[Key, dict], rep: Report,
                     partial: bool) -> None:
    count = len(records)
    if partial:
        if not 1 <= count <= EXPECTED_RECORDS:
            rep.err("record_count", f"partial mode requires 1-{EXPECTED_RECORDS} valid records, "
                    f"found {count}")
    elif count != EXPECTED_RECORDS:
        rep.err("record_count", f"exactly {EXPECTED_RECORDS} records required, found {count}")

    for n, rec in records:
        validate_record(n, rec, index, rep)

    qids = [(n, rec.get("source_qid")) for n, rec in records]
    for q, k in Counter(q for _, q in qids if isinstance(q, str)).items():
        if k > 1:
            lines = [n for n, x in qids if x == q]
            rep.err("duplicate_source_qid", f"source_qid {q!r} appears {k} times (lines {lines})")
    for pos, (n, q) in enumerate(qids, 1):
        expected = f"q{pos:03d}"
        if q != expected:
            rep.err("qid_sequence", f"source_qid {q!r} at record {pos}; expected {expected!r}",
                    line=n, qid=q if isinstance(q, str) else None)

    seen_q: dict[str, int] = {}
    for n, rec in records:
        q = rec.get("query")
        if isinstance(q, str) and q.strip():
            nq = normalize_query(q)
            if nq in seen_q:
                rep.err("duplicate_query", f"normalized query duplicates line {seen_q[nq]}",
                        line=n, qid=rec.get("source_qid"))
            else:
                seen_q[nq] = n

    seen_r: dict[str, int] = {}
    for n, rec in records:
        sig = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        if sig in seen_r:
            rep.err("duplicate_record", f"exact duplicate of line {seen_r[sig]}",
                    line=n, qid=rec.get("source_qid"))
        else:
            seen_r[sig] = n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET,
                    help=f"dataset JSONL path (default: {DEFAULT_DATASET.relative_to(ROOT)})")
    ap.add_argument("--partial", action="store_true",
                    help=f"allow 1-{EXPECTED_RECORDS} records (IDs still sequential from q001)")
    ap.add_argument("--parsed-dir", type=Path, default=PARSED_DIR,
                    help=f"directory with *.elements.jsonl (default: {PARSED_DIR.relative_to(ROOT)})")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="backslashreplace")
    dataset = args.dataset.resolve()

    rep = Report()
    try:
        index = load_sources(args.parsed_dir)
        records = parse_dataset(dataset, rep)
    except LoadError as ex:
        print(f"FATAL: {ex}")
        return 2
    validate_dataset(records, index, rep, args.partial)

    mode = "partial" if args.partial else "full"
    if rep.errors:
        print(f"Dataset: {dataset}  ({len(records)} parsed records, {mode} mode)")
        print(f"VALIDATION FAILED: {len(rep.errors)} error(s)")
        print("\n".join(f"  {e}" for e in rep.errors))
        return 1
    docs = Counter(rec["doc_id"] for _, rec in records)
    n_ctx = sum(len(rec["ideal_context"]) for _, rec in records)
    print(f"VALIDATION PASSED ({mode} mode): {dataset}")
    print(f"  records: {len(records)}  context elements: {n_ctx}  "
          f"source elements indexed: {len(index)}")
    print(f"  by doc_id: {dict(sorted(docs.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
