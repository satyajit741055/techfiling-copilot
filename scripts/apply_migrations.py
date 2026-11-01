"""Create or update the conversation-history tables.

Usage:
    uv run python scripts/apply_migrations.py [--supabase] [--db-url URL]

Applies db/migrations/*.sql (plain Postgres) once each. --supabase also applies
db/supabase/*.sql (Row Level Security); use it only against a Supabase database.
The database URL defaults to HISTORY_DB_URL (or SUPABASE_DB_URL) from .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from persistence.history import with_connect_timeout  # noqa: E402
from persistence.migrations import apply_migrations, migration_files  # noqa: E402


def _failed(supabase: bool, exc: psycopg.Error) -> str:
    hint = " (--supabase needs a Supabase database)" if supabase else ""
    return f"Migration failed{hint}: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db-url", help="Postgres URL (default: HISTORY_DB_URL from .env)")
    parser.add_argument(
        "--supabase", action="store_true", help="also apply the Supabase-only RLS policies"
    )
    args = parser.parse_args()

    url = args.db_url
    if not url:
        # Only now: loading the full settings validates unrelated keys (providers, etc.),
        # which must not block a run that passes --db-url.
        from config.settings import settings

        url = settings.HISTORY_DB_URL
    if not url:
        raise SystemExit("Set HISTORY_DB_URL in .env or pass --db-url")

    try:
        # prepare_threshold=None: the Supabase pooler (transaction mode) rejects prepared
        # statements.
        conninfo = with_connect_timeout(url, seconds=10)
        with psycopg.connect(conninfo, autocommit=True, prepare_threshold=None) as conn:
            applied = apply_migrations(conn, include_supabase=args.supabase)
    except psycopg.ProgrammingError as exc:
        if exc.sqlstate is None:
            # Client-side URL parse error; libpq's message can quote the password.
            raise SystemExit("The database URL is not a valid Postgres connection URL") from None
        raise SystemExit(_failed(args.supabase, exc)) from None
    except psycopg.Error as exc:
        raise SystemExit(_failed(args.supabase, exc)) from None

    total = len(migration_files(args.supabase))
    for name in applied:
        print(f"applied  {name}")
    print(f"{len(applied)} applied, {total - len(applied)} already up to date")


if __name__ == "__main__":
    main()
