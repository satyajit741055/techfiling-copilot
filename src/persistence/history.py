"""Conversation history: threads and their messages, one row per finished turn.

Plain Postgres, so the same store runs on Docker Postgres in development and on the
Supabase pooler later; only HISTORY_DB_URL changes. The store is synchronous (psycopg
pool, parameterised SQL only): async callers use ``asyncio.to_thread``. This also avoids
psycopg's async mode, which cannot run on the Windows default (Proactor) event loop.

Writes are best-effort: ``save_turn`` logs and counts a failure and returns False rather
than raising, so a database outage never breaks an answer.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol
from uuid import UUID

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

TITLE_CHARS = 80
MAX_THREADS = 200
MAX_MESSAGES = 500
POOL_TIMEOUT_SECONDS = 5.0
CONNECT_TIMEOUT_SECONDS = 5

TurnStatus = Literal["answered", "clarification", "refused", "error"]
JSON_COLUMNS = (
    "answer",
    "citations",
    "assumptions",
    "coverage",
    "scope",
    "validation_errors",
    "trajectory",
)


def _without_nul(value: Any) -> Any:
    """Drop NUL characters from strings, also inside lists and dicts."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_without_nul(item) for item in value]
    if isinstance(value, dict):
        return {_without_nul(key): _without_nul(item) for key, item in value.items()}
    return value


class TurnRecord(BaseModel):
    """Everything saved for one finished turn (answer, clarification, refusal or error)."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _strip_nul(cls, data: Any) -> Any:
        # Postgres text and jsonb reject NUL, so one stray NUL would fail every write.
        return _without_nul(data)

    thread_id: UUID
    user_id: UUID
    turn_id: str | None = None
    question: str = Field(min_length=1)
    status: TurnStatus
    route: str | None = None
    answer_markdown: str | None = None
    answer: dict[str, Any] | None = None
    citations: list[Any] | None = None
    assumptions: list[Any] | None = None
    coverage: list[Any] | None = None
    scope: dict[str, Any] | None = None
    is_valid: bool | None = None
    validation_errors: list[Any] | None = None
    trajectory: list[Any] | None = None
    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0)


class ThreadSummary(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class StoredMessage(BaseModel):
    """A saved turn as shown to its owner (no trajectory, tokens or cost)."""

    id: UUID
    turn_id: str | None = None
    question: str
    status: TurnStatus
    route: str | None = None
    answer_markdown: str | None = None
    answer: dict[str, Any] | None = None
    citations: list[Any] | None = None
    assumptions: list[Any] | None = None
    coverage: list[Any] | None = None
    scope: dict[str, Any] | None = None
    is_valid: bool | None = None
    validation_errors: list[Any] | None = None
    created_at: datetime


class HistoryStore(Protocol):
    enabled: bool

    def open(self) -> None: ...

    def close(self) -> None: ...

    def save_turn(self, record: TurnRecord) -> bool: ...

    def list_threads(self, user_id: UUID, limit: int = 50) -> list[ThreadSummary]: ...

    def list_messages(
        self, user_id: UUID, thread_id: UUID, limit: int = MAX_MESSAGES
    ) -> list[StoredMessage] | None:
        """The thread's latest ``limit`` turns, oldest first; None if not the user's."""
        ...


class NullHistoryStore:
    """HISTORY_STORE=none: nothing is saved and nothing is found."""

    enabled = False

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def save_turn(self, record: TurnRecord) -> bool:
        return False

    def list_threads(self, user_id: UUID, limit: int = 50) -> list[ThreadSummary]:
        return []

    def list_messages(
        self, user_id: UUID, thread_id: UUID, limit: int = MAX_MESSAGES
    ) -> list[StoredMessage] | None:
        return None


_UPSERT_THREAD = """
insert into threads (id, user_id, title) values (%s, %s, %s)
on conflict (id) do update set updated_at = now()
where threads.user_id = excluded.user_id
returning id
"""

_MESSAGE_COLUMNS = (
    "thread_id",
    "user_id",
    "turn_id",
    "question",
    "status",
    "route",
    "answer_markdown",
    "answer",
    "citations",
    "assumptions",
    "coverage",
    "scope",
    "is_valid",
    "validation_errors",
    "trajectory",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "cost_usd",
)
_INSERT_MESSAGE = (
    f"insert into messages ({', '.join(_MESSAGE_COLUMNS)}) "
    f"values ({', '.join(['%s'] * len(_MESSAGE_COLUMNS))})"
)
_SELECT_MESSAGES = f"""
select * from (
    select {", ".join(StoredMessage.model_fields)} from messages
    where thread_id = %s and user_id = %s
    order by created_at desc, id desc
    limit %s
) latest
order by created_at, id
"""


def thread_title(question: str) -> str:
    """First question, on one line, cut at TITLE_CHARS."""
    text = " ".join(question.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1].rstrip() + "…"


def with_connect_timeout(conninfo: str, seconds: int = CONNECT_TIMEOUT_SECONDS) -> str:
    """Add connect_timeout unless the URL sets one.

    Without it a connection attempt can wait forever, e.g. on Windows "localhost" tries
    ::1 first and never notices that nothing listens there.
    """
    if "connect_timeout" in conninfo_to_dict(conninfo):
        return conninfo
    return make_conninfo(conninfo, connect_timeout=seconds)


def _clamp(limit: int, maximum: int) -> int:
    return max(1, min(limit, maximum))


def _error_summary(exc: Exception) -> str:
    """Error text that is safe to log.

    Server errors keep only the primary message: str(exc) also carries DETAIL, which for
    constraint errors quotes the whole failing row (question, answer, ...).
    """
    diag = getattr(exc, "diag", None)
    primary = getattr(diag, "message_primary", None)
    sqlstate = getattr(exc, "sqlstate", None)
    code = f" [{sqlstate}]" if sqlstate else ""
    return f"{type(exc).__name__}{code}: {primary or exc}"


class PostgresHistoryStore:
    """HISTORY_STORE=postgres over a psycopg connection pool."""

    enabled = True

    def __init__(self, conninfo: str, max_size: int = 5, timeout: float = POOL_TIMEOUT_SECONDS):
        try:
            conninfo = with_connect_timeout(conninfo)
        except psycopg.ProgrammingError:
            # libpq's parse error can quote the URL, password included.
            raise ValueError("HISTORY_DB_URL is not a valid Postgres connection URL") from None
        # prepare_threshold=None: the Supabase pooler (transaction mode) rejects
        # prepared statements. check_connection replaces connections the pooler dropped.
        self._pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=max_size,
            timeout=timeout,
            open=False,
            check=ConnectionPool.check_connection,
            kwargs={"prepare_threshold": None},
            name="history",
        )
        self._lock = threading.Lock()
        self.failed_writes = 0

    def open(self) -> None:
        """Start connecting in the background; startup does not wait for the database."""
        self._pool.open(wait=False)

    def close(self) -> None:
        self._pool.close()

    def save_turn(self, record: TurnRecord) -> bool:
        """Save one turn, creating the thread on its first turn. Never raises."""
        try:
            with self._pool.connection() as conn, conn.transaction():
                owned = conn.execute(
                    _UPSERT_THREAD,
                    (record.thread_id, record.user_id, thread_title(record.question)),
                ).fetchone()
                if owned is None:
                    # The id belongs to another user's thread: never write into it.
                    raise PermissionError("thread belongs to another user")
                conn.execute(_INSERT_MESSAGE, self._message_params(record))
            return True
        except Exception as exc:
            with self._lock:
                self.failed_writes += 1
                failures = self.failed_writes
            logger.warning(
                f"History write failed for thread {record.thread_id} "
                f"({_error_summary(exc)}); total failed writes: {failures}"
            )
            return False

    def list_threads(self, user_id: UUID, limit: int = 50) -> list[ThreadSummary]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "select id, title, created_at, updated_at from threads "
                "where user_id = %s order by updated_at desc, id limit %s",
                (user_id, _clamp(limit, MAX_THREADS)),
            )
            return [ThreadSummary(**row) for row in cur.fetchall()]

    def list_messages(
        self, user_id: UUID, thread_id: UUID, limit: int = MAX_MESSAGES
    ) -> list[StoredMessage] | None:
        """The latest ``limit`` messages, oldest first; None if the thread is not the user's."""
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "select 1 from threads where id = %s and user_id = %s", (thread_id, user_id)
            )
            if cur.fetchone() is None:
                return None
            cur.execute(_SELECT_MESSAGES, (thread_id, user_id, _clamp(limit, MAX_MESSAGES)))
            return [StoredMessage(**row) for row in cur.fetchall()]

    @staticmethod
    def _message_params(record: TurnRecord) -> list[Any]:
        values = record.model_dump(mode="json")
        return [
            Jsonb(values[name])
            if name in JSON_COLUMNS and values[name] is not None
            # UUIDs and Decimal keep their Python types for the uuid / numeric columns.
            else getattr(record, name)
            for name in _MESSAGE_COLUMNS
        ]


def create_history_store(settings: Any) -> HistoryStore:
    """Store for HISTORY_STORE: Postgres when configured, otherwise the null store."""
    if settings.HISTORY_STORE == "postgres":
        return PostgresHistoryStore(settings.HISTORY_DB_URL, max_size=settings.HISTORY_DB_POOL_SIZE)
    return NullHistoryStore()
