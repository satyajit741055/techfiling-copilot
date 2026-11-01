"""Copy a Qdrant collection into a new one, reusing its dense vectors and recomputing BM25.

Use to tune the sparse (BM25) encoding: no re-embedding is needed. The new collection
records the BM25 average chunk length it was built with in its metadata.

Usage:
    uv run python scripts/reindex_qdrant_collection.py [--source NAME] [--target NAME]
        [--target-url URL] [--avg-doc-tokens source | 256 | measured]

Defaults come from .env (QDRANT_PATH / QDRANT_URL, QDRANT_COLLECTION). The target defaults
to a new timestamped name; it must not exist yet. Use --target-url to copy from an
embedded source store to a Qdrant server without recomputing dense embeddings.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.settings import settings  # noqa: E402
from vector_store import QdrantChunkStore  # noqa: E402
from vector_store.qdrant_store import DENSE, FILING_KEY  # noqa: E402
from vector_store.sparse import AVG_DOC_TOKENS, average_tokens  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default=settings.QDRANT_COLLECTION)
    parser.add_argument(
        "--target",
        default=f"techfiling_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument(
        "--target-url",
        help="write to a different Qdrant server instead of the source store",
    )
    parser.add_argument(
        "--avg-doc-tokens",
        default="source",
        help="BM25 avgdl: a number, 'measured' for the mean chunk length, or 'source' to keep "
        f"the source collection's value (legacy collections: {AVG_DOC_TOKENS}) "
        "(default: %(default)s)",
    )
    args = parser.parse_args()

    source = QdrantChunkStore.from_settings(settings, collection=args.source)
    if not source.exists():
        raise SystemExit(f"Source collection {args.source!r} does not exist")
    if args.target_url:
        target = QdrantChunkStore(
            args.target_url,
            args.target,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
        )
    else:
        target = QdrantChunkStore(settings.QDRANT_URL, args.target, client=source.client, path=None)
        target.local = source.local
    if target.exists():
        raise SystemExit(f"Target collection {args.target!r} already exists")

    chunks, vectors, filings = [], [], {}
    offset = None
    while True:
        points, offset = source.client.scroll(
            args.source, limit=1000, offset=offset, with_payload=True, with_vectors=[DENSE]
        )
        for p in points:
            payload = p.payload or {}
            chunks.append(source._chunk(payload))
            vectors.append(p.vector[DENSE])
            if payload.get(FILING_KEY):
                filings[payload["doc_id"]] = payload[FILING_KEY]
        if offset is None:
            break

    order = sorted(range(len(chunks)), key=lambda i: chunks[i]["chunk_id"])
    chunks = [chunks[i] for i in order]
    matrix = np.asarray([vectors[i] for i in order], dtype=np.float32)
    if args.avg_doc_tokens == "measured":
        avg_doc_tokens = average_tokens([target._sparse_text(c) for c in chunks])
    elif args.avg_doc_tokens == "source":
        avg_doc_tokens = source.avg_doc_tokens()
    else:
        avg_doc_tokens = float(args.avg_doc_tokens)

    n = target.index_chunks(
        chunks,
        matrix,
        source.index_metadata()["embedding_model"],
        replace_documents=False,
        filings=filings,
        avg_doc_tokens=avg_doc_tokens,
    )
    if target.count() != source.count() or n != source.count():
        raise SystemExit(f"Point count mismatch: source {source.count()}, target {target.count()}")
    print(
        f"Reindexed {n} chunks: {args.source!r} -> {args.target!r} "
        f"(BM25 avg_doc_tokens={target.avg_doc_tokens()}, "
        f"was {source.avg_doc_tokens()})\n"
        f"Set QDRANT_COLLECTION={args.target} to serve it."
    )
    if target.client is not source.client:
        target.client.close()
    source.client.close()


if __name__ == "__main__":
    main()
