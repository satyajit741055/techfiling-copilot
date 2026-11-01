"""Apply the SQL files in db/ once each, recording them in a schema_migrations table.

db/migrations/ is plain Postgres and runs everywhere (Docker Postgres, Supabase).
db/supabase/ needs Supabase (auth.uid() in RLS policies) and runs only when asked.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

DB_DIR = Path(__file__).resolve().parents[2] / "db"
PORTABLE_DIR = "migrations"
SUPABASE_DIR = "supabase"
# Arbitrary key for pg_advisory_xact_lock, shared by every migration run.
LOCK_KEY = 815_302_009


def migration_files(
    include_supabase: bool = False, db_dir: Path = DB_DIR
) -> list[tuple[str, Path]]:
    """(name, path) in apply order: portable files first, then the Supabase files."""
    folders = [PORTABLE_DIR] + ([SUPABASE_DIR] if include_supabase else [])
    return [
        (f"{folder}/{path.name}", path)
        for folder in folders
        for path in sorted((db_dir / folder).glob("*.sql"))
    ]


def apply_migrations(
    conn: psycopg.Connection, include_supabase: bool = False, db_dir: Path = DB_DIR
) -> list[str]:
    """Apply pending files, each in its own transaction. Needs an autocommit connection.

    Returns the names applied in this run; files applied before are skipped. Every
    transaction takes a transaction-level advisory lock (session locks do not survive the
    Supabase transaction pooler), so concurrent runs wait and then skip what the other
    run applied instead of applying a file twice.
    """
    if not conn.autocommit:
        raise ValueError("apply_migrations needs a connection opened with autocommit=True")
    with conn.transaction():
        conn.execute("select pg_advisory_xact_lock(%s)", (LOCK_KEY,))
        conn.execute(
            "create table if not exists schema_migrations ("
            "name text primary key, applied_at timestamptz not null default now())"
        )
    done = []
    for name, path in migration_files(include_supabase, db_dir):
        with conn.transaction():
            conn.execute("select pg_advisory_xact_lock(%s)", (LOCK_KEY,))
            if conn.execute("select 1 from schema_migrations where name = %s", (name,)).fetchone():
                continue
            # No parameters, so psycopg sends the file as one multi-statement query.
            conn.execute(path.read_text(encoding="utf-8").encode())
            conn.execute("insert into schema_migrations (name) values (%s)", (name,))
        done.append(name)
    return done
