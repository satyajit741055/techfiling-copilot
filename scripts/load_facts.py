"""Build the financial facts table from parser and chunker output and load it into Postgres.

Usage:
    uv run python scripts/load_facts.py [--chunks-dir data/chunks/<run>] [--collection NAME]
                                        [--parsed-dir data/parsed] [--db-url URL] [--dry-run]

Facts are linked to the table-row chunks of the chunk run behind the Qdrant collection the
API uses (QDRANT_COLLECTION; the run whose run_manifest.json names it). Run it after
promoting a new ingestion run, and apply migrations first (scripts/apply_migrations.py).
The table is replaced in one transaction; the API reads it at startup (FACTS_STORE=postgres).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facts import build_facts, chunk_run_for_collection  # noqa: E402
from facts.store import FactsStore  # noqa: E402
from persistence.facts import FactsLoad, replace_facts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parsed-dir", type=Path, default=ROOT / "data" / "parsed")
    parser.add_argument("--chunks-dir", type=Path, help="chunk run (default: run of --collection)")
    parser.add_argument("--collection", help="Qdrant collection (default: QDRANT_COLLECTION)")
    parser.add_argument("--db-url", help="Postgres URL (default: HISTORY_DB_URL from .env)")
    parser.add_argument("--dry-run", action="store_true", help="build and report, do not load")
    args = parser.parse_args()

    collection, url = args.collection, args.db_url
    if not collection or (not url and not args.dry_run):
        from config.settings import settings

        collection = collection or settings.QDRANT_COLLECTION
        url = url or settings.HISTORY_DB_URL
    chunks_dir = args.chunks_dir or chunk_run_for_collection(ROOT / "data" / "chunks", collection)
    if chunks_dir is None:
        raise SystemExit(
            f"No chunk run manifest names collection {collection!r}; pass --chunks-dir"
        )

    records = build_facts(args.parsed_dir, chunks_dir)
    store = FactsStore(records, collection)
    tickers = sorted({record.ticker for record in records})
    years = sorted({record.filing_fiscal_year for record in records})
    print(f"collection {collection}  chunk run {chunks_dir.name}")
    print(
        f"{len(records)} facts, {store.linked} linked to a table row ({store.linked / len(records):.0%})"
    )
    for key, missing in store.coverage(tickers, years).items():
        print(f"  {key:<22} {'all filings' if not missing else 'missing ' + ', '.join(missing)}")
    if args.dry_run:
        return
    if not url:
        raise SystemExit("Set HISTORY_DB_URL in .env or pass --db-url")
    load = FactsLoad(
        collection=collection, chunk_run=chunks_dir.name, facts=len(records), linked=store.linked
    )
    try:
        replace_facts(url, records, load)
    except psycopg.Error as exc:
        hint = " (run scripts/apply_migrations.py first?)" if exc.sqlstate == "42P01" else ""
        raise SystemExit(f"Load failed{hint}: {type(exc).__name__}: {exc}") from None
    print(f"loaded {len(records)} facts")


if __name__ == "__main__":
    main()
