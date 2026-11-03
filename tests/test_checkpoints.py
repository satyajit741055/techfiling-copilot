"""
Postgres checkpointer (AGENT_CHECKPOINTER=postgres): the threaded saver, startup wiring
and pruning offline; against a live database (HISTORY_TEST_DB_URL) clarification pauses
and follow-up context surviving a restart. Each live test uses its own temporary schema.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
import pytest
from langgraph.checkpoint.memory import MemorySaver

import api.dependencies as dependencies
from agent.graph import build_graph, checkpoint_serializer, graph_config
from agent.schemas import Analysis
from agent.service import AgentService
from config.settings import settings
from persistence import ThreadedPostgresSaver, create_checkpointer
from tests.test_agent_graph import ScriptedLLM, analysis, make_deps, simple_generator
from tests.test_chat import REFUSED, FakeHistory, FakeWriter, collect, names, only, turn
from tests.test_history import _temp_schema, live

UNREACHABLE = "postgresql://u:p@127.0.0.1:1/x"
FOLLOW_UP = Analysis(
    intent="filing_question",
    scope="context_dependent",
    question_type="single_fact",
    uses_history=True,
)


class RecordingLLM(ScriptedLLM):
    """Scripted LLM that keeps the prompt text of every structured call."""

    def __init__(self, responses: dict[type, list[Any]]) -> None:
        super().__init__(responses)
        self.prompts: list[str] = []

    async def structured(self, schema: type[Any], messages: list[Any]) -> Any:
        self.prompts.append("\n".join(str(message.content) for message in messages))
        return await super().structured(schema, messages)


class PruneRecorder(MemorySaver):
    def __init__(self, error: Exception | None = None) -> None:
        super().__init__(serde=checkpoint_serializer())
        self.pruned: list[list[str]] = []
        self.error = error

    async def aprune(self, thread_ids: Any, *, strategy: str = "keep_latest") -> None:
        if self.error is not None:
            raise self.error
        self.pruned.append(list(thread_ids))


def service_with(llm: Any, checkpointer: Any) -> AgentService:
    deps = make_deps(llm, generator=simple_generator())
    return AgentService(
        build_graph(deps, checkpointer),
        deps.catalog,
        evidence=deps.evidence,
        history=FakeHistory(),
        writer=FakeWriter(),  # type: ignore[arg-type]
        provider="ollama",
        model="fake-model",
        poll_seconds=0.02,
    )


# =============================================================================
# OFFLINE
# =============================================================================


@pytest.mark.asyncio
async def test_async_methods_run_the_sync_methods_in_a_worker_thread(monkeypatch) -> None:
    saver = create_checkpointer(UNREACHABLE, checkpoint_serializer())  # pool not opened
    calls: list[tuple[str, int]] = []

    def record(name: str, result: Any = None) -> Any:
        def method(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, threading.get_ident()))
            return result

        return method

    monkeypatch.setattr(saver, "get_tuple", record("get_tuple"))
    monkeypatch.setattr(saver, "list", record("list", iter(["a", "b"])))
    monkeypatch.setattr(saver, "put", record("put", {"configurable": {}}))
    monkeypatch.setattr(saver, "put_writes", record("put_writes"))
    monkeypatch.setattr(saver, "delete_thread", record("delete_thread"))
    monkeypatch.setattr(saver, "prune", record("prune"))
    config = graph_config("owner:thread")

    assert await saver.aget_tuple(config) is None
    assert [item async for item in saver.alist(config)] == ["a", "b"]
    assert await saver.aput(config, {}, {}, {}) == {"configurable": {}}  # type: ignore[arg-type]
    await saver.aput_writes(config, [], "task")
    await saver.adelete_thread("owner:thread")
    await saver.aprune(["owner:thread"])

    assert [name for name, _ in calls] == [
        "get_tuple",
        "list",
        "put",
        "put_writes",
        "delete_thread",
        "prune",
    ]
    assert threading.get_ident() not in {ident for _, ident in calls}
    saver.close()


def test_malformed_url_error_does_not_echo_the_password() -> None:
    with pytest.raises(ValueError, match="not a valid Postgres connection URL") as info:
        create_checkpointer("postgresql://u:secret pw@127.0.0.1/db", checkpoint_serializer())
    assert "secret" not in str(info.value) and info.value.__cause__ is None


def test_unreachable_database_disables_research_chat_and_closes_the_pool(
    monkeypatch, caplog
) -> None:
    closed: list[bool] = []

    def refuse(self: ThreadedPostgresSaver, timeout: float = 0) -> None:
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(settings, "AGENT_CHECKPOINTER", "postgres")
    monkeypatch.setattr(settings, "HISTORY_DB_URL", UNREACHABLE)
    monkeypatch.setattr(ThreadedPostgresSaver, "open", refuse)
    monkeypatch.setattr(ThreadedPostgresSaver, "close", lambda self: closed.append(True))
    monkeypatch.setattr(dependencies, "_agent_service", object())

    with caplog.at_level(logging.ERROR, logger="api.dependencies"):
        dependencies.init_agent()

    assert dependencies.get_agent_service() is None and dependencies._checkpointer is None
    assert closed == [True]
    assert "Research chat DISABLED: OperationalError" in caplog.text
    assert ":p@" not in caplog.text


@pytest.mark.asyncio
async def test_finished_turns_prune_the_thread_but_a_clarification_pause_does_not() -> None:
    saver = PruneRecorder()
    service = service_with(ScriptedLLM({Analysis: [analysis(), REFUSED]}), saver)
    thread = uuid.uuid4()

    await collect(service, turn(thread, message="What was revenue?"))
    assert saver.pruned == []
    await collect(service, turn(thread, option_id="opt_1"))
    await collect(service, turn(thread, message="What is EBITDA?"))
    assert saver.pruned == [[f"local:{thread}"]] * 2


@pytest.mark.asyncio
async def test_a_failed_prune_never_fails_the_turn(caplog) -> None:
    service = service_with(
        ScriptedLLM({Analysis: [analysis(companies=["AAPL"])]}),
        PruneRecorder(RuntimeError("database gone")),
    )
    with caplog.at_level(logging.WARNING, logger="agent.service"):
        events = await collect(service, turn(message="Apple revenue?"))
    assert names(events) == ["thread", "progress", "answer", "done"]
    assert "Checkpoint prune failed: RuntimeError: database gone" in caplog.text


# =============================================================================
# LIVE POSTGRES (HISTORY_TEST_DB_URL)
# =============================================================================


@contextmanager
def live_saver(conninfo: str) -> Iterator[ThreadedPostgresSaver]:
    """One 'server process': a fresh pool and saver; closing it is the restart."""
    saver = create_checkpointer(conninfo, checkpoint_serializer())
    saver.open()
    try:
        yield saver
    finally:
        saver.close()


def checkpoint_count(conninfo: str, key: str) -> int:
    with psycopg.connect(conninfo) as conn:
        row = conn.execute("select count(*) from checkpoints where thread_id = %s", (key,))
        return int(row.fetchone()[0])  # type: ignore[index]


@live
@pytest.mark.asyncio
async def test_live_clarification_survives_a_restart() -> None:
    thread = uuid.uuid4()
    with _temp_schema() as conninfo:
        with live_saver(conninfo) as saver:
            first = await collect(
                service_with(ScriptedLLM({Analysis: [analysis()]}), saver),
                turn(thread, message="What was revenue?"),
            )
        assert names(first) == ["thread", "progress", "clarification", "done"]

        with live_saver(conninfo) as saver:
            second = await collect(
                service_with(ScriptedLLM({}), saver), turn(thread, option_id="opt_1")
            )

    assert names(second) == ["thread", "progress", "answer", "done"]
    assert only(second, "thread")[0]["turn_id"] == only(first, "thread")[0]["turn_id"]
    answer = only(second, "answer")[0]
    assert answer["route"] == "simple_rag" and answer["coverage"][0]["ticker"] == "AAPL"


@live
@pytest.mark.asyncio
async def test_live_follow_up_context_reaches_the_llm_after_a_restart() -> None:
    thread = uuid.uuid4()
    with _temp_schema() as conninfo:
        with live_saver(conninfo) as saver:
            await collect(
                service_with(ScriptedLLM({Analysis: [analysis(companies=["MSFT"])]}), saver),
                turn(thread, message="Microsoft revenue?"),
            )

        with live_saver(conninfo) as saver:
            llm = RecordingLLM({Analysis: [FOLLOW_UP]})
            events = await collect(
                service_with(llm, saver), turn(thread, message="And their revenue?")
            )
            other = RecordingLLM({Analysis: [FOLLOW_UP]})
            await collect(
                service_with(other, saver),
                turn(thread, owner="someone-else", message="And their revenue?"),
            )

    assert "Q: Microsoft revenue?\n(companies: MSFT; years: FY2025)" in llm.prompts[0]
    assert only(events, "answer")[0]["coverage"][0]["ticker"] == "MSFT"
    # Checkpoints are namespaced by owner: the same thread id carries no context for others.
    assert "Conversation:\n(none)" in other.prompts[0]


@live
@pytest.mark.asyncio
async def test_live_each_thread_keeps_one_checkpoint_with_every_turn() -> None:
    thread = uuid.uuid4()
    key = f"local:{thread}"
    with _temp_schema() as conninfo, live_saver(conninfo) as saver:
        service = service_with(
            ScriptedLLM({Analysis: [analysis(companies=["AAPL"]), analysis(companies=["MSFT"])]}),
            saver,
        )
        await collect(service, turn(thread, message="Apple revenue?"))
        await collect(service, turn(thread, message="Microsoft revenue?"))

        assert checkpoint_count(conninfo, key) == 1
        state = await service.graph.aget_state(graph_config(key))
        assert [t.tickers for t in state.values["turns"]] == [["AAPL"], ["MSFT"]]
