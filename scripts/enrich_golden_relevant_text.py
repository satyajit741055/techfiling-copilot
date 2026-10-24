"""Add a `relevant_text` field to each golden retrieval case by resolving its
element_orders to chunk text.

Usage:
    uv run python scripts/enrich_golden_relevant_text.py \
        --golden goldens/golden_retriever_dataset_v3.jsonl \
        --chunks-dir data/chunks/2026-10-24_12-46-23 \
        --output goldens/golden_retriever_dataset_v3_enriched.jsonl
"""

import argparse
import json
from pathlib import Path

SEPARATOR = "\n\n---\n\n"


def _orders(value) -> list[int]:
    return json.loads(value) if isinstance(value, str) else list(value or [])


def load_chunks(chunks_dir: Path) -> dict[str, list[dict]]:
    chunks: dict[str, list[dict]] = {}
    for path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunk["element_orders"] = set(_orders(chunk["element_orders"]))
                    chunks.setdefault(chunk["doc_id"], []).append(chunk)
    return chunks


def enrich(case: dict, chunks: dict[str, list[dict]]) -> dict:
    selected: dict[str, dict] = {}
    missing: list[str] = []
    for rel in case.get("relevant", []):
        if rel.get("match_by") != "element_orders":
            continue
        doc_id = rel.get("doc_id") or case.get("doc_id")
        doc_ids = [doc_id] if doc_id else list(chunks)
        wanted = set(_orders(rel["value"]))
        found: set[int] = set()
        for d in doc_ids:
            for chunk in chunks.get(d, []):
                hit = chunk["element_orders"] & wanted
                if hit:
                    selected[chunk["chunk_id"]] = chunk
                    found |= hit
        missing += [f"{doc_id or '*'}:{o}" for o in sorted(wanted - found)]
    if missing:
        print(f"[warn] {case['id']}: element_orders not found in any chunk: {missing}")
    ordered = sorted(selected.values(), key=lambda c: (c["doc_id"], c["chunk_id"]))
    cross_doc = case.get("doc_id") is None
    texts = [f"[{c['doc_id']}]\n{c['text']}" if cross_doc else c["text"] for c in ordered]
    return {**case, "relevant_text": SEPARATOR.join(texts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", type=Path, default=Path("goldens/golden_retriever_dataset_v3.jsonl"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks/2026-10-24_12-46-23"))
    parser.add_argument("--output", type=Path, default=Path("goldens/golden_retriever_dataset_v3_enriched.jsonl"))
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_dir)
    with args.golden.open(encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    enriched = [enrich(case, chunks) for case in cases]
    with args.output.open("w", encoding="utf-8") as f:
        for case in enriched:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Wrote {len(enriched)} cases to {args.output}")


if __name__ == "__main__":
    main()
