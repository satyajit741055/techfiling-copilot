"""Read and replace the ``financial_facts`` table (db/migrations/002_financial_facts.sql).

The table is small (~10k rows), so the API reads it once at startup into an in-memory
store; the loader script replaces it in one transaction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple
from datetime import date
from typing import Any

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

from facts.records import COLUMNS, FactRecord

from .history import with_connect_timeout


class FactsLoad(BaseModel):
    collection: str
    chunk_run: str
    facts: int
    linked: int


def _connect(conninfo: str, **kwargs: Any) -> psycopg.Connection[Any]:
    try:
        conninfo = with_connect_timeout(conninfo)
    except psycopg.ProgrammingError:
        # libpq's parse error can quote the URL, password included.
        raise ValueError("HISTORY_DB_URL is not a valid Postgres connection URL") from None
    # prepare_threshold=None: the Supabase pooler (transaction mode) rejects prepared statements.
    return psycopg.connect(conninfo, prepare_threshold=None, **kwargs)


def _record(row: dict[str, Any]) -> FactRecord:
    for name in ("period_start", "period_end"):
        if isinstance(row[name], date):
            row[name] = row[name].isoformat()
    return FactRecord(**row)


def read_facts(conninfo: str) -> tuple[list[FactRecord], FactsLoad | None]:
    """All fact rows and the newest load record (None when the table was never loaded)."""
    with _connect(conninfo) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "select collection, chunk_run, facts, linked from financial_facts_loads "
            "order by id desc limit 1"
        )
        load = cur.fetchone()
        cur.execute(f"select {', '.join(COLUMNS)} from financial_facts")
        records = [_record(row) for row in cur.fetchall()]
    return records, FactsLoad(**load) if load else None


def replace_facts(conninfo: str, records: Sequence[FactRecord], load: FactsLoad) -> None:
    """Replace every fact row and record the load, atomically."""
    with _connect(conninfo) as conn, conn.transaction():
        conn.execute("delete from financial_facts")
        with conn.cursor().copy(f"copy financial_facts ({', '.join(COLUMNS)}) from stdin") as copy:
            for record in records:
                copy.write_row(astuple(record))
        conn.execute(
            "insert into financial_facts_loads (collection, chunk_run, facts, linked) "
            "values (%s, %s, %s, %s)",
            (load.collection, load.chunk_run, load.facts, load.linked),
        )
