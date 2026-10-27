"""Load an existing run (data/chunks/<run>/) into Qdrant without re-embedding.

Usage:
    uv run python scripts/migrate_npy_to_qdrant.py [run_directory] [--url URL] [--collection NAME]
        [--parsed-dir DIR] [--keep-existing]

Defaults come from .env (RUN_DIRECTORY, QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY).
Each document in the run replaces its previous chunks, so the script is safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.settings import settings  # noqa: E402
from ingestion.chunking.embedder import EMBEDDINGS_FILENAME, MANIFEST_FILENAME, load_chunks  # noqa: E402
from vector_store import QdrantChunkStore, load_filing_metadata  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_directory", nargs="?", type=Path, default=settings.get_run_directory())
    parser.add_argument("--url", default=settings.QDRANT_URL)
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=settings.DATA_DIR / "parsed",
        help="folder with <doc_id>.document.json, stored with the chunks as the filing catalog",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="upsert only; do not delete a document's old chunks first",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.run_directory)
    embeddings = np.load(args.run_directory / EMBEDDINGS_FILENAME)
    manifest = json.loads((args.run_directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if len(chunks) != len(embeddings):
        raise SystemExit(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings in {args.run_directory}"
        )

    store = QdrantChunkStore(
        args.url, args.collection, api_key=settings.QDRANT_API_KEY, timeout=settings.QDRANT_TIMEOUT
    )
    filings = load_filing_metadata(
        args.parsed_dir, list(dict.fromkeys(c["doc_id"] for c in chunks))
    )
    store.index_chunks(
        chunks,
        embeddings,
        manifest["embedding_model"],
        replace_documents=not args.keep_existing,
        filings=filings,
    )
    print(
        f"Indexed {len(chunks)} chunks into {args.collection!r}; collection now has {store.count()} points "
        f"for documents: {store.doc_ids()}"
    )
    if len(filings) < len({c["doc_id"] for c in chunks}):
        print(
            f"Warning: filing metadata found for only {len(filings)} document(s) in {args.parsed_dir}"
        )


if __name__ == "__main__":
    main()
