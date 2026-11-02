"""End-to-end ingestion: raw 10-K HTML -> parsed elements -> chunks -> embeddings.

Output: data/chunks/<run_timestamp>/
    <doc_id>.chunks.jsonl, run_manifest.json, and embeddings_gpu_base.npy when
    VECTOR_STORE=numpy (the layout read by retriever/retriever.py). With
    VECTOR_STORE=qdrant the embeddings go into a new timestamped collection instead.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
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

REQUIRED_METADATA_FIELDS = ("ticker", "fiscal_year", "form", "period_end")
REQUIRED_STRUCTURAL_FIELDS = ("chunk_id", "doc_id", "text", "embed_text", "type")


def validate_ingestion(
    doc_ids: list[str],
    all_chunks: list[dict],
    expected_tickers: list[str] | None = None,
    expected_years: list[int] | None = None,
) -> None:
    """Validate document and chunk invariants before embedding.

    Args:
        doc_ids: Document IDs from the current parse run.
        all_chunks: All chunks generated for the current run.
        expected_tickers: Tickers that must each have every expected fiscal year.
        expected_years: Fiscal years that must exist for every expected ticker.

    Raises:
        ValueError: If any invariant fails.
    """
    # Document-level checks
    duplicates = sorted(d for d, n in Counter(doc_ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate document IDs in current run: {', '.join(duplicates)}")

    # Per-chunk field checks run first so later checks can index fields safely
    for chunk in all_chunks:
        cid = chunk.get("chunk_id", "<missing chunk_id>")
        missing = [f for f in REQUIRED_STRUCTURAL_FIELDS if f not in chunk]
        if missing:
            raise ValueError(f"Chunk {cid} missing structural field: {', '.join(missing)}")
        missing = [f for f in REQUIRED_METADATA_FIELDS if chunk.get(f) is None]
        if missing:
            raise ValueError(f"Chunk {cid} missing required field: {', '.join(missing)}")
        fy = chunk["fiscal_year"]
        if isinstance(fy, bool) or not isinstance(fy, int):
            raise ValueError(f"Chunk {cid} has non-integer fiscal_year: {fy!r}")

    # chunk_id uniqueness (primary key)
    chunk_duplicates = sorted(
        c for c, n in Counter(c["chunk_id"] for c in all_chunks).items() if n > 1
    )
    if chunk_duplicates:
        raise ValueError(f"Duplicate chunk IDs: {', '.join(chunk_duplicates[:20])}")

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in all_chunks:
        by_doc[chunk["doc_id"]].append(chunk)

    missing_chunks = set(doc_ids) - set(by_doc)
    if missing_chunks:
        raise ValueError(f"Documents with no chunks: {', '.join(sorted(missing_chunks))}")
    unknown_docs = set(by_doc) - set(doc_ids)
    if unknown_docs:
        raise ValueError(
            f"Chunks for documents not in current run: {', '.join(sorted(unknown_docs))}"
        )

    doc_keys: dict[tuple, str] = {}
    for doc_id in doc_ids:
        doc_chunks = by_doc[doc_id]
        actual = sorted(c["chunk_id"] for c in doc_chunks)
        for i, cid in enumerate(actual):
            if cid != f"{doc_id}_c{i:04d}":
                raise ValueError(
                    f"Chunk suffixes not sequential for {doc_id}: expected {doc_id}_c{i:04d}, got {cid}"
                )

        # Filing metadata must be identical on every chunk of a document
        metas = {tuple(c[f] for f in REQUIRED_METADATA_FIELDS) for c in doc_chunks}
        if len(metas) > 1:
            raise ValueError(
                f"Inconsistent filing metadata across chunks of {doc_id}: {sorted(metas)}"
            )

        # (ticker, form, fiscal_year) must identify exactly one document
        first = doc_chunks[0]
        key = (first["ticker"], first["form"], first["fiscal_year"])
        if key in doc_keys:
            raise ValueError(
                f"Duplicate (ticker, form, fiscal_year) key {key}: {doc_keys[key]} and {doc_id}"
            )
        doc_keys[key] = doc_id

    if (expected_tickers is None) != (expected_years is None):
        raise ValueError("expected_tickers and expected_years must be provided together")
    if expected_tickers is not None and expected_years is not None:
        covered = {(str(ticker).upper(), fiscal_year) for ticker, _, fiscal_year in doc_keys}
        expected = {
            (ticker.strip().upper(), fiscal_year)
            for ticker in expected_tickers
            for fiscal_year in expected_years
        }
        missing_cells = sorted(expected - covered)
        if missing_cells:
            formatted = ", ".join(f"{ticker} FY{year}" for ticker, year in missing_cells)
            raise ValueError(f"Missing expected ticker/fiscal-year filings: {formatted}")

    logger.info(f"Ingestion validation passed: {len(doc_ids)} documents, {len(all_chunks)} chunks")


def process(
    raw_dir: Path = RAW_DIR,
    parsed_dir: Path = PARSED_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int = 64,
    skip_parse: bool = False,
    skip_embed: bool = False,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    table_max_chars: int = TABLE_MAX_CHARS,
    model_cache_dir: Path = MODEL_CACHE_DIR,
    pseudo_headings: bool = True,
    expected_tickers: list[str] | None = None,
    expected_years: list[int] | None = None,
) -> Path:
    if not skip_parse:
        docs = parse_all(raw_dir, parsed_dir)
        doc_ids = [doc.doc_id for doc in docs]
    else:
        doc_ids = list_doc_ids(parsed_dir)

    if not doc_ids:
        raise FileNotFoundError(f"No *.elements.jsonl in {parsed_dir}; run without --skip-parse")

    chunks_by_doc = {}
    for doc_id in doc_ids:
        chunks_by_doc[doc_id] = chunk_doc_id(
            doc_id, parsed_dir, max_chars, min_chars, table_max_chars, pseudo_headings
        )
        logger.info(f"{doc_id}: {len(chunks_by_doc[doc_id])} chunks")
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    logger.info(f"Total chunks: {len(all_chunks)}")

    # Validate invariants before anything is written, embedded or indexed
    validate_ingestion(doc_ids, all_chunks, expected_tickers, expected_years)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = chunks_dir / run_timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    for doc_id, chunks in chunks_by_doc.items():
        write_chunks(chunks, out_dir / f"{doc_id}.chunks.jsonl")

    if skip_embed:
        logger.info(f"Skipping embedding and Qdrant indexing (chunk-only run) -> {out_dir}")
        return out_dir

    from config.settings import settings

    use_qdrant = settings.VECTOR_STORE == "qdrant"
    embeddings, manifest = embed_and_save(
        all_chunks,
        out_dir,
        model_name,
        device,
        batch_size,
        {
            "max_chars": max_chars,
            "min_chars": min_chars,
            "table_max_chars": table_max_chars,
            "pseudo_headings": pseudo_headings,
        },
        model_cache_dir=model_cache_dir,
        parsed_dir=parsed_dir,
        skip_npy=use_qdrant,
    )
    logger.info(f"Embeddings shape: {embeddings.shape} -> {out_dir}")

    if use_qdrant:
        from vector_store import QdrantChunkStore, load_filing_metadata

        # Create a timestamped collection name
        collection_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        collection_name = f"techfiling_{collection_timestamp}"

        # Create a new store with the timestamped collection
        store = QdrantChunkStore.from_settings(settings, collection=collection_name)

        # Fail if collection already exists (should not happen with timestamp)
        if store.exists():
            raise ValueError(f"Collection {collection_name!r} already exists; timestamp collision")

        n = store.index_chunks(
            all_chunks,
            embeddings,
            manifest["embedding_model"],
            filings=load_filing_metadata(parsed_dir, doc_ids),
        )
        logger.info(f"Indexed {n} chunks into Qdrant collection {collection_name!r}")

        # Verify point count matches chunk count
        if n != len(all_chunks):
            raise ValueError(f"Qdrant point count {n} does not match chunk count {len(all_chunks)}")

        # Verify all filings appear in list_filings
        filings = store.list_filings()
        filing_doc_ids = {f["doc_id"] for f in filings}
        missing_filings = set(doc_ids) - filing_doc_ids
        if missing_filings:
            raise ValueError(f"Missing filings in Qdrant: {', '.join(sorted(missing_filings))}")

        # Write collection name to manifest
        manifest["qdrant_collection"] = {
            "name": collection_name,
            "created_at": collection_timestamp,
            "point_count": n,
            "location": str(settings.QDRANT_PATH) if settings.QDRANT_PATH else settings.QDRANT_URL,
        }
        store.client.close()
        (out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            f"Updated manifest with Qdrant collection {collection_name!r}; "
            f"set QDRANT_COLLECTION={collection_name} to serve it "
            f"(currently {settings.QDRANT_COLLECTION!r})"
        )
    return out_dir


def _parse_tickers(value: str) -> list[str]:
    tickers = [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]
    if not tickers:
        raise argparse.ArgumentTypeError("provide at least one comma-separated ticker")
    return tickers


def _parse_years(value: str) -> list[int]:
    years: list[int] = []
    try:
        for item in value.split(","):
            item = item.strip()
            if "-" in item:
                start, end = (int(part) for part in item.split("-", 1))
                if start > end:
                    raise ValueError
                years.extend(range(start, end + 1))
            elif item:
                years.append(int(item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated years or inclusive ranges, e.g. 2021-2025"
        ) from exc
    if not years:
        raise argparse.ArgumentTypeError("provide at least one fiscal year")
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse, chunk and embed 10-K filings into data/chunks/<run>/."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--model", help="Embedding model (default: EMBEDDING_MODEL from .env)")
    parser.add_argument("--device", help="cuda | cpu (default: cuda if available)")
    parser.add_argument("--model-cache-dir", type=Path, default=MODEL_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-parse", action="store_true", help="reuse existing data/parsed/*.elements.jsonl"
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="skip embedding and Qdrant indexing (chunk-only run)",
    )
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--table-max-chars", type=int, default=TABLE_MAX_CHARS)
    parser.add_argument(
        "--expect-tickers",
        type=_parse_tickers,
        help="comma-separated tickers required in the corpus, e.g. AAPL,MSFT,TSLA",
    )
    parser.add_argument(
        "--expect-years",
        type=_parse_years,
        help="required fiscal years, e.g. 2021-2025 or 2021,2023,2025",
    )
    parser.add_argument(
        "--pseudo-headings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start new chunks at short heading-like paragraphs (default: on)",
    )
    args = parser.parse_args()
    process(
        raw_dir=args.raw_dir,
        parsed_dir=args.parsed_dir,
        chunks_dir=args.chunks_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        skip_parse=args.skip_parse,
        skip_embed=args.skip_embed,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        table_max_chars=args.table_max_chars,
        model_cache_dir=args.model_cache_dir,
        pseudo_headings=args.pseudo_headings,
        expected_tickers=args.expect_tickers,
        expected_years=args.expect_years,
    )


if __name__ == "__main__":
    main()
