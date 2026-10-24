"""Structure-aware chunking of parsed 10-K elements.

  1. never cross an Item boundary
  2. pack paragraphs under the same heading up to a size budget
  3. tables are atomic (split by rows only when large, repeating title + header)
  4. every chunk carries a breadcrumb so it is self-contained
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PARSED_DIR = ROOT / "data" / "parsed"

# Sizes include breadcrumb + text. [ Need to adjust according to Embedder Cotext window ]
MAX_CHARS = 1200
MIN_CHARS = 200
TABLE_MAX_CHARS = 900

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

# Short sub-headings the parser emits as paragraphs ("Intelligent Cloud", "Greater China",
# "Revenue Recognition"); starting a chunk at them keeps a subsection's text together.
PSEUDO_HEADING_MAX_CHARS = 80
PSEUDO_HEADING_END = re.compile(r"[.:;,?!)]$")
PSEUDO_HEADING_START = re.compile(r"^[(\d•\-/]")


def looks_like_heading(e: dict, nxt: dict | None) -> bool:
    """Return whether a paragraph is a short sub-heading directly followed by body content."""
    text = (e.get("text") or "").strip()
    return (
        e["type"] == "paragraph" and nxt is not None
        and nxt["type"] in ("paragraph", "table")
        and (nxt["part"], nxt["section"]) == (e["part"], e["section"])
        and 0 < len(text) <= PSEUDO_HEADING_MAX_CHARS
        and not PSEUDO_HEADING_END.search(text) and not PSEUDO_HEADING_START.match(text)
    )


def load_elements(doc_id: str, parsed_dir: Path = PARSED_DIR) -> list[dict]:
    """Load parsed elements for a document from its JSON Lines file.

    Args:
        doc_id: Document identifier used as the input filename prefix.
        parsed_dir: Directory containing parsed element files.

    Returns:
        Parsed, non-empty JSON objects in file order.
    """
    with (parsed_dir / f"{doc_id}.elements.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def list_doc_ids(parsed_dir: Path = PARSED_DIR) -> list[str]:
    """Return sorted document IDs discovered in the parsed-data directory."""
    return sorted(p.name.removesuffix(".elements.jsonl") for p in parsed_dir.glob("*.elements.jsonl"))


def split_long(text: str, max_chars: int) -> list[str]:
    """Split oversized text at sentence boundaries into size-limited pieces.

    A single sentence longer than ``max_chars`` remains intact.
    """
    parts, cur = [], ""
    for s in SENT_SPLIT.split(text):
        if cur and len(cur) + len(s) + 1 > max_chars:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts

def concise_table_intro(intro: str | None, max_chars: int = 300) -> str:
    """Extract the latest complete sentences from a table introduction.

    Args:
        intro: Full introductory text, or ``None`` when unavailable.
        max_chars: Preferred maximum length of the returned context.

    Returns:
        Trailing sentences in their original order, or an empty string. One
        sentence may exceed ``max_chars`` to avoid truncating its meaning.
    """
    intro = (intro or "").strip()
    if not intro:
        return ""
    sentences = [s.strip() for s in SENT_SPLIT.split(intro) if s.strip()]
    if not sentences:
        return intro[-max_chars:]
    selected, total = [], 0
    for sentence in reversed(sentences):
        additional = len(sentence) + (1 if selected else 0)
        if selected and total + additional > max_chars:
            break
        selected.append(sentence)
        total += additional
    return " ".join(reversed(selected))

def split_table_markdown(md: str, max_chars: int) -> list[str]:
    """Split a Markdown table by rows while repeating its header in each piece.

    Args:
        md: Complete Markdown table, including its separator row.
        max_chars: Preferred maximum size of each table piece.

    Returns:
        Table pieces in row order. A header or individual row may cause a piece
        to exceed ``max_chars`` because rows are never split.

    Raises:
        StopIteration: If the Markdown does not contain a separator row.
    """
    if len(md) <= max_chars:
        return [md]
    lines = md.split("\n")
    sep = next(i for i, l in enumerate(lines) if re.fullmatch(r"\|[-| ]+\|", l))
    head, rows = lines[: sep + 1], lines[sep + 1:]
    pieces, cur = [], []
    for r in rows:
        if cur and len("\n".join(head + cur + [r])) > max_chars:
            pieces.append("\n".join(head + cur))
            cur = []
        cur.append(r)
    if cur:
        pieces.append("\n".join(head + cur))
    return pieces

def breadcrumb(e: dict, headings: list[str]) -> str:
    """Build a searchable document breadcrumb using the two latest headings."""
    parts = [e["doc_id"].replace("_", " "), e.get("part"), e.get("section"), e.get("section_title")]
    return " > ".join([p for p in parts if p] + headings[-2:])


def chunk_document(
    els: list[dict], max_chars: int = MAX_CHARS, table_max_chars: int = TABLE_MAX_CHARS,
    pseudo_headings: bool = True,
    min_chars: int = MIN_CHARS,
) -> list[dict]:
    """Convert ordered parsed elements into section-aware text and table chunks.

    Args:
        els: Parsed elements in source-document order.
        max_chars: Preferred maximum text size per chunk.
        table_max_chars: Preferred maximum Markdown table size per chunk.
        pseudo_headings: Start a new chunk at short heading-like paragraphs (see
            ``looks_like_heading``) once the current chunk holds at least ``min_chars``.
        min_chars: Minimum buffered text before a pseudo-heading may start a new chunk.

    Returns:
        Chunk dictionaries with source metadata, breadcrumbs, and embedding text.
        Chunks never cross part/section boundaries; oversized sentences and table
        rows remain atomic and may exceed their preferred limits.
    """
    chunks, buf, buf_orders, headings = [], [], [], []
    state = {"key": None, "first": None}

    def emit(type_, text, orders, ref, extra=None):
        """Create a chunk from buffered content and append it to the result."""
        c = {
            "chunk_id": f"{ref['doc_id']}_c{len(chunks):04d}",
            "doc_id": ref["doc_id"], "part": ref.get("part"), "section": ref.get("section"),
            "section_title": ref.get("section_title"), "type": type_,
            "headings": list(headings[-2:]), "element_orders": orders,
            "text": text, "breadcrumb": breadcrumb(ref, headings),
        }
        c["embed_text"] = f"{c['breadcrumb']}\n\n{text}"
        c.update(extra or {})
        chunks.append(c)

    def flush(ref):
        """Emit accumulated text against a reference element, then clear buffers."""
        if buf:
            emit("text", "\n\n".join(buf), buf_orders[:], ref)
        buf.clear()
        buf_orders.clear()

    for i, e in enumerate(els):
        if (pseudo_headings and sum(map(len, buf)) >= min_chars
                and looks_like_heading(e, els[i + 1] if i + 1 < len(els) else None)):
            flush(e)  # sub-heading after real content = new topic; heading text stays in the body
        key = (e["part"], e["section"])
        if key != state["key"]:  # new Item -> hard boundary
            if state["first"]:
                flush(state["first"])
            headings.clear()
            state["key"] = key
        state["first"] = e
        ref = e

        if e["type"] == "heading":
            if buf:
                flush(ref)  # heading after body text = new topic
            headings.append(e["text"])
            continue

        if e["type"] == "table":
            flush(ref)
            t = e["table"]
            full_intro = t.get("intro")
            table_context = concise_table_intro(full_intro)
            for piece in split_table_markdown(t["markdown"], table_max_chars):
                text = f"{table_context}\n\n{piece}" if table_context else piece
                emit("table", text, [e["order"]], ref,
                     {"table_id": t["table_id"], "table_title": t.get("title"), "units": t.get("units"),
                      "intro": full_intro, "table_context": table_context})
            continue

        text = (e["text"] or "").strip()
        if not text:
            continue
        for piece in (split_long(text, max_chars) if len(text) > max_chars else [text]):
            if buf and sum(map(len, buf)) + len(piece) > max_chars:
                flush(ref)
            buf.append(piece)
            buf_orders.append(e["order"])
    if state["first"]:
        flush(state["first"])
    return chunks


def merge_tiny(chunks: list[dict], min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> list[dict]:
    """Merge undersized text chunks forward within the same document section.

    Args:
        chunks: Ordered chunk dictionaries to compact.
        min_chars: Text length below which a preceding chunk is considered tiny.
        max_chars: Maximum combined text length allowed for a merge.

    Returns:
        Compacted chunks with sequentially reassigned chunk IDs. Input chunk
        dictionaries retained in the result are modified in place.
    """
    out = []
    for c in chunks:
        p = out[-1] if out else None
        if (p and p["type"] == "text" and c["type"] == "text" and c["section"] == p["section"]
                and c["doc_id"] == p["doc_id"]
                and len(p["text"]) < min_chars and len(p["text"]) + len(c["text"]) <= max_chars):
            p["text"] += "\n\n" + c["text"]
            p["element_orders"] += c["element_orders"]
            p["embed_text"] = f"{p['breadcrumb']}\n\n{p['text']}"
        else:
            out.append(c)
    for i, c in enumerate(out):  # renumber ids
        c["chunk_id"] = f"{c['doc_id']}_c{i:04d}"
    return out


def chunk_doc_id(
    doc_id: str,
    parsed_dir: Path = PARSED_DIR,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    table_max_chars: int = TABLE_MAX_CHARS,
    pseudo_headings: bool = True,
) -> list[dict]:
    """Load, chunk, and compact one parsed document.

    Args:
        doc_id: Identifier of the document to process.
        parsed_dir: Directory containing the document's element file.
        max_chars: Preferred maximum text chunk size.
        min_chars: Threshold used to identify tiny text chunks.
        table_max_chars: Preferred maximum table chunk size.
        pseudo_headings: Start new chunks at short heading-like paragraphs.

    Returns:
        Final chunk dictionaries ready to be written or embedded.
    """
    chunks = chunk_document(load_elements(doc_id, parsed_dir), max_chars, table_max_chars, pseudo_headings, min_chars)
    return merge_tiny(chunks, min_chars, max_chars)


def write_chunks(chunks: list[dict], path: Path) -> None:
    """Write chunk dictionaries as UTF-8 JSON Lines to ``path``."""
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the command-line workflow for chunking every parsed document."""
    parser = argparse.ArgumentParser(description="Chunk parsed 10-K elements (no embedding).")
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="directory for <doc_id>.chunks.jsonl (default: data/chunks/<timestamp>)",
    )
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--table-max-chars", type=int, default=TABLE_MAX_CHARS)
    parser.add_argument("--pseudo-headings", action=argparse.BooleanOptionalAction, default=True,
                        help="start new chunks at short heading-like paragraphs (default: on)")
    args = parser.parse_args()

    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.out_dir = ROOT / "data" / "chunks" / timestamp
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for doc_id in list_doc_ids(args.parsed_dir):
        chunks = chunk_doc_id(doc_id, args.parsed_dir, args.max_chars, args.min_chars, args.table_max_chars,
                              args.pseudo_headings)
        write_chunks(chunks, args.out_dir / f"{doc_id}.chunks.jsonl")
        total += len(chunks)
        print(doc_id, len(chunks))
    print("total", total)


if __name__ == "__main__":
    main()