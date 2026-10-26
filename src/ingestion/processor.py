"""End-to-end ingestion: raw 10-K HTML -> parsed elements -> chunks -> embeddings (.npy).

Output: data/chunks/<run_timestamp>/
    <doc_id>.chunks.jsonl, embeddings_gpu_base.npy, run_manifest.json
(the layout read by retriver/retriever.py)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from config.logging import get_logger
from ingestion.chunking.chunker import (
    MAX_CHARS,
    MIN_CHARS,
    PARSED_DIR,
    TABLE_MAX_CHARS,
    chunk_doc_id,
    list_doc_ids,
    write_chunks,
)
from ingestion.chunking.embedder import CHUNKS_DIR, MODEL_CACHE_DIR, embed_and_save
from ingestion.parser.parser import RAW_DIR, parse_all

logger = get_logger(__name__)


def process(
    raw_dir: Path = RAW_DIR,
    parsed_dir: Path = PARSED_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int = 64,
    skip_parse: bool = False,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    table_max_chars: int = TABLE_MAX_CHARS,
    model_cache_dir: Path = MODEL_CACHE_DIR,
    pseudo_headings: bool = True,
) -> Path:
    if not skip_parse:
        parse_all(raw_dir, parsed_dir)

    doc_ids = list_doc_ids(parsed_dir)
    if not doc_ids:
        raise FileNotFoundError(f"No *.elements.jsonl in {parsed_dir}; run without --skip-parse")

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = chunks_dir / run_timestamp
    out_dir.mkdir(parents=True, exist_ok=False)

    all_chunks = []
    for doc_id in doc_ids:
        chunks = chunk_doc_id(doc_id, parsed_dir, max_chars, min_chars, table_max_chars, pseudo_headings)
        write_chunks(chunks, out_dir / f"{doc_id}.chunks.jsonl")
        all_chunks += chunks
        logger.info(f"{doc_id}: {len(chunks)} chunks")
    logger.info(f"Total chunks: {len(all_chunks)}")

    embeddings, manifest = embed_and_save(
        all_chunks, out_dir, model_name, device, batch_size,
        {"max_chars": max_chars, "min_chars": min_chars, "table_max_chars": table_max_chars,
         "pseudo_headings": pseudo_headings},
        model_cache_dir=model_cache_dir,
    )
    logger.info(f"Embeddings shape: {embeddings.shape} -> {out_dir}")

    from config.settings import settings

    if settings.VECTOR_STORE == "qdrant":
        from vector_store import QdrantChunkStore, load_filing_metadata

        store = QdrantChunkStore.from_settings()
        n = store.index_chunks(
            all_chunks, embeddings, manifest["embedding_model"],
            filings=load_filing_metadata(parsed_dir, doc_ids),
        )
        logger.info(f"Indexed {n} chunks into Qdrant collection {store.collection!r}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, chunk and embed 10-K filings into data/chunks/<run>/.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--model", help="Embedding model (default: EMBEDDING_MODEL from .env)")
    parser.add_argument("--device", help="cuda | cpu (default: cuda if available)")
    parser.add_argument("--model-cache-dir", type=Path, default=MODEL_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-parse", action="store_true", help="reuse existing data/parsed/*.elements.jsonl")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--table-max-chars", type=int, default=TABLE_MAX_CHARS)
    parser.add_argument("--pseudo-headings", action=argparse.BooleanOptionalAction, default=True,
                        help="start new chunks at short heading-like paragraphs (default: on)")
    args = parser.parse_args()
    process(args.raw_dir, args.parsed_dir, args.chunks_dir, args.model, args.device, args.batch_size,
            args.skip_parse, args.max_chars, args.min_chars, args.table_max_chars, args.model_cache_dir,
            args.pseudo_headings)


if __name__ == "__main__":
    main()
