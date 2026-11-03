"""Research-chat graph checkpoints in Postgres (AGENT_CHECKPOINTER=postgres).

Clarification pauses and the follow-up context (``turns``) then survive a restart. The
saver is LangGraph's synchronous PostgresSaver with its async methods run in a worker
thread: psycopg's async mode cannot run on the Windows default (Proactor) event loop,
the same reason the history store is synchronous.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from .history import POOL_TIMEOUT_SECONDS, with_connect_timeout

# PostgresSaver runs every call under one lock, so more connections would sit idle.
POOL_MAX_SIZE = 2
OPEN_TIMEOUT_SECONDS = 10.0


class ThreadedPostgresSaver(PostgresSaver):
    """PostgresSaver usable from async graphs: each async method runs its sync twin."""

    pool: ConnectionPool[psycopg.Connection[DictRow]]

    def __init__(
        self, pool: ConnectionPool[psycopg.Connection[DictRow]], serde: SerializerProtocol
    ) -> None:
        super().__init__(pool, serde=serde)
        self.pool = pool

    def open(self, timeout: float = OPEN_TIMEOUT_SECONDS) -> None:
        """Connect (raises if the database is unreachable) and create or migrate the tables."""
        self.pool.open(wait=True, timeout=timeout)
        self.setup()

    def close(self) -> None:
        self.pool.close()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        await asyncio.to_thread(self.prune, thread_ids, strategy=strategy)


def create_checkpointer(conninfo: str, serde: SerializerProtocol) -> ThreadedPostgresSaver:
    """Saver over a not-yet-opened pool; call ``open()`` before use and ``close()`` after."""
    try:
        conninfo = with_connect_timeout(conninfo)
    except psycopg.ProgrammingError:
        # libpq's parse error can quote the URL, password included.
        raise ValueError("HISTORY_DB_URL is not a valid Postgres connection URL") from None
    # autocommit + dict_row are required by PostgresSaver; prepare_threshold=None because
    # the Supabase pooler (transaction mode) rejects prepared statements.
    pool: ConnectionPool[psycopg.Connection[DictRow]] = ConnectionPool(
        conninfo,
        connection_class=psycopg.Connection[DictRow],
        min_size=1,
        max_size=POOL_MAX_SIZE,
        timeout=POOL_TIMEOUT_SECONDS,
        open=False,
        check=ConnectionPool.check_connection,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
        name="checkpoints",
    )
    return ThreadedPostgresSaver(pool, serde)
