"""
Conversation history: settings, owner resolution, migrations, the Postgres store and the
/threads endpoints. Offline by default; the live Postgres tests run only when
HISTORY_TEST_DB_URL is set (each test uses its own temporary schema, dropped afterwards).
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient
from jose import jwt
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import ValidationError

from api.auth import history_user_id
from api.config import APISettings, api_settings
from api.dependencies import get_history_store
from api.main import app
from api.routers.history import _not_found
from persistence import (
    NullHistoryStore,
    PostgresHistoryStore,
    StoredMessage,
    ThreadSummary,
    TurnRecord,
    create_history_store,
)
from persistence.history import TITLE_CHARS, _error_summary, thread_title, with_connect_timeout
from persistence.migrations import apply_migrations, migration_files

DEV_USER = uuid.UUID("11111111-1111-4111-8111-111111111111")
OTHER_USER = uuid.UUID("22222222-2222-4222-8222-222222222222")
THREAD = uuid.UUID("33333333-3333-4333-8333-333333333333")
NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)
SECRET = "test_jwt_secret_for_testing"


def _record(**overrides) -> TurnRecord:
    values = {
        "thread_id": THREAD,
        "user_id": DEV_USER,
        "turn_id": "t1",
        "question": "Compare Apple and Microsoft R&D",
        "status": "answered",
        "route": "research",
        "answer_markdown": "Apple spent more [S1].",
        "answer": {"answer": "Apple spent more [S1].", "is_valid": True},
        "citations": [{"source_id": "S1", "chunk_id": "AAPL_10-K_2025_c0001"}],
        "assumptions": ["Latest fiscal year"],
        "coverage": [{"ticker": "AAPL", "fiscal_year": 2025, "status": "found"}],
        "scope": {"tickers": ["AAPL", "MSFT"], "fiscal_years": [2025]},
        "is_valid": True,
        "validation_errors": [],
        "trajectory": [{"tool": "search_filings"}],
        "model": "fake-model",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "latency_ms": 1200,
        "cost_usd": Decimal("0.000125"),
    }
    return TurnRecord(**{**values, **overrides})


# =============================================================================
# MODELS AND SETTINGS
# =============================================================================


def test_turn_record_rejects_unknown_status_fields_and_negative_counts():
    with pytest.raises(ValidationError):
        _record(status="pending")
    with pytest.raises(ValidationError):
        _record(unexpected=True)
    with pytest.raises(ValidationError):
        _record(prompt_tokens=-1)


def test_turn_record_strips_nul_characters_everywhere():
    record = _record(
        question="What\x00 was revenue?",
        answer={"text\x00": ["a\x00b", {"k": "c\x00"}]},
        assumptions=["x\x00"],
    )
    assert record.question == "What was revenue?"
    assert record.answer == {"text": ["ab", {"k": "c"}]}
    assert record.assumptions == ["x"]
    with pytest.raises(ValidationError):
        _record(question="\x00")  # nothing left after stripping


def test_logged_database_error_omits_detail_with_row_data():
    class CheckViolation(Exception):
        sqlstate = "23514"
        diag = SimpleNamespace(message_primary='new row violates check constraint "c"')

        def __str__(self) -> str:
            return self.diag.message_primary + "\nDETAIL:  Failing row contains (SECRET QUESTION)."

    summary = _error_summary(CheckViolation())
    assert "23514" in summary and "check constraint" in summary
    assert "SECRET" not in summary
    assert _error_summary(PermissionError("thread belongs to another user")).endswith(
        "thread belongs to another user"
    )


def test_malformed_db_url_error_does_not_echo_the_password():
    with pytest.raises(ValueError, match="not a valid Postgres connection URL") as info:
        PostgresHistoryStore("postgresql://u:secret pw@127.0.0.1/db")
    assert "secret" not in str(info.value) and info.value.__cause__ is None


def test_not_found_is_a_fresh_exception_each_time():
    assert _not_found() is not _not_found()


def test_thread_title_is_one_line_and_truncated():
    assert thread_title("  What was\nApple's   revenue? ") == "What was Apple's revenue?"
    long = thread_title("x" * 200)
    assert len(long) == TITLE_CHARS and long.endswith("…")


def test_connect_timeout_is_added_unless_the_url_sets_one():
    added = conninfo_to_dict(with_connect_timeout("postgresql://u:p@127.0.0.1:5432/db"))
    assert added["connect_timeout"] == "5" and added["host"] == "127.0.0.1"
    kept = with_connect_timeout("postgresql://u:p@h/db?connect_timeout=30&sslmode=require")
    assert conninfo_to_dict(kept)["connect_timeout"] == "30"
    assert conninfo_to_dict(kept)["sslmode"] == "require"


def test_create_history_store_follows_setting():
    assert isinstance(create_history_store(SimpleNamespace(HISTORY_STORE="none")), NullHistoryStore)
    store = create_history_store(
        SimpleNamespace(
            HISTORY_STORE="postgres",
            HISTORY_DB_URL="postgresql://u:p@127.0.0.1:1/x",
            HISTORY_DB_POOL_SIZE=2,
        )
    )
    assert isinstance(store, PostgresHistoryStore) and store.enabled


def test_null_store_saves_and_finds_nothing():
    store = NullHistoryStore()
    assert not store.enabled
    assert store.save_turn(_record()) is False
    assert store.list_threads(DEV_USER) == []
    assert store.list_messages(DEV_USER, THREAD) is None


def test_dev_user_setting(monkeypatch):
    monkeypatch.delenv("HISTORY_DEV_USER_ID", raising=False)
    assert APISettings(_env_file=None, HISTORY_DEV_USER_ID=" ").HISTORY_DEV_USER_ID is None
    assert (
        APISettings(_env_file=None, HISTORY_DEV_USER_ID=str(DEV_USER)).HISTORY_DEV_USER_ID
        == DEV_USER
    )
    with pytest.raises(ValidationError):
        APISettings(_env_file=None, HISTORY_DEV_USER_ID="not-a-uuid")
    with pytest.raises(ValidationError, match="HISTORY_DEV_USER_ID is for local development"):
        APISettings(
            _env_file=None,
            ENVIRONMENT="prod",
            API_ACCESS_KEY="k" * 32,
            CORS_ORIGINS="https://app.example.com",
            HISTORY_DEV_USER_ID=str(DEV_USER),
        )


def test_history_user_id_uses_dev_user_without_auth_and_jwt_subject_with_auth(monkeypatch):
    request = SimpleNamespace(state=SimpleNamespace(user_id=str(OTHER_USER)))
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", DEV_USER)
    assert history_user_id(request) == DEV_USER  # auth off (conftest default)

    monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", SECRET)
    assert history_user_id(request) == OTHER_USER  # dev user ignored once auth is on
    assert history_user_id(SimpleNamespace(state=SimpleNamespace())) is None
    assert history_user_id(SimpleNamespace(state=SimpleNamespace(user_id="u1"))) is None


# =============================================================================
# MIGRATIONS
# =============================================================================


def test_migration_files_portable_first_supabase_only_on_request():
    portable = [name for name, _ in migration_files()]
    assert portable == ["migrations/001_threads_messages.sql"]
    with_rls = [name for name, _ in migration_files(include_supabase=True)]
    assert with_rls == portable + ["supabase/001_rls.sql"]
    assert "auth.uid()" not in migration_files()[0][1].read_text(encoding="utf-8")


def test_apply_migrations_requires_autocommit():
    with pytest.raises(ValueError, match="autocommit"):
        apply_migrations(SimpleNamespace(autocommit=False))  # type: ignore[arg-type]


# =============================================================================
# BEST-EFFORT WRITES
# =============================================================================


def test_save_turn_never_raises_when_database_is_down():
    store = PostgresHistoryStore("postgresql://u:p@127.0.0.1:1/x?connect_timeout=1", timeout=0.5)
    store.open()
    try:
        assert store.save_turn(_record()) is False
        assert store.save_turn(_record(turn_id="t2")) is False
        assert store.failed_writes == 2
    finally:
        store.close()


# =============================================================================
# /threads ENDPOINTS (fake store, no database)
# =============================================================================


class FakeStore:
    enabled = True

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple] = []

    def list_threads(self, user_id, limit=50):
        self.calls.append(("threads", user_id, limit))
        if self.error:
            raise self.error
        return [ThreadSummary(id=THREAD, title="Compare", created_at=NOW, updated_at=NOW)]

    def list_messages(self, user_id, thread_id, limit=500):
        self.calls.append(("messages", user_id, thread_id))
        if self.error:
            raise self.error
        if user_id != DEV_USER or thread_id != THREAD:
            return None
        return [
            StoredMessage(id=uuid.uuid4(), question="Compare", status="answered", created_at=NOW)
        ]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        from api.rate_limit import in_memory_limiter

        in_memory_limiter.requests.clear()


def _use(store) -> None:
    app.dependency_overrides[get_history_store] = lambda: store


def test_endpoints_404_when_history_is_off(client, monkeypatch):
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", DEV_USER)
    _use(NullHistoryStore())
    assert client.get("/threads").status_code == 404
    assert client.get(f"/threads/{THREAD}/messages").status_code == 404


def test_endpoints_404_without_auth_or_dev_user(client, monkeypatch):
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", None)
    store = FakeStore()
    _use(store)
    assert client.get("/threads").status_code == 404
    assert store.calls == []


def test_dev_user_lists_threads_and_messages(client, monkeypatch):
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", DEV_USER)
    store = FakeStore()
    _use(store)

    threads = client.get("/threads", params={"limit": 5})
    assert threads.status_code == 200
    assert threads.json()["threads"][0]["id"] == str(THREAD)
    assert store.calls[0] == ("threads", DEV_USER, 5)

    messages = client.get(f"/threads/{THREAD}/messages")
    assert messages.status_code == 200
    body = messages.json()
    assert body["thread_id"] == str(THREAD) and body["messages"][0]["status"] == "answered"
    assert "trajectory" not in body["messages"][0]


def test_unknown_or_foreign_thread_is_404_and_bad_ids_are_422(client, monkeypatch):
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", DEV_USER)
    _use(FakeStore())
    assert client.get(f"/threads/{uuid.uuid4()}/messages").status_code == 404
    assert client.get("/threads/not-a-uuid/messages").status_code == 422
    assert client.get("/threads", params={"limit": 0}).status_code == 422


def test_database_errors_are_503_without_details(client, monkeypatch):
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", DEV_USER)
    _use(FakeStore(error=psycopg.OperationalError("password authentication failed for x")))
    for path in ("/threads", f"/threads/{THREAD}/messages"):
        response = client.get(path)
        assert response.status_code == 503
        assert "password" not in response.text


def test_signed_in_user_owns_history_and_guests_are_rejected(client, monkeypatch):
    monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", SECRET)
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", OTHER_USER)
    store = FakeStore()
    _use(store)

    assert client.get("/threads").status_code == 401
    token = jwt.encode({"sub": str(DEV_USER), "aud": "authenticated"}, SECRET, algorithm="HS256")
    response = client.get("/threads", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert store.calls == [("threads", DEV_USER, 50)]


# =============================================================================
# LIVE POSTGRES (HISTORY_TEST_DB_URL)
# =============================================================================

LIVE_URL = os.environ.get("HISTORY_TEST_DB_URL")
live = pytest.mark.skipif(not LIVE_URL, reason="HISTORY_TEST_DB_URL not set")


@contextmanager
def _temp_schema():
    """Conninfo whose search_path is a new schema, dropped on exit."""
    schema = f"history_test_{uuid.uuid4().hex[:12]}"
    admin_url = with_connect_timeout(LIVE_URL)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f'create schema "{schema}"'.encode())
        try:
            yield make_conninfo(admin_url, options=f"-c search_path={schema}")
        finally:
            admin.execute(f'drop schema "{schema}" cascade'.encode())


def _migrate(conninfo: str) -> list[str]:
    with psycopg.connect(conninfo, autocommit=True) as conn:
        return apply_migrations(conn)


@pytest.fixture
def live_store():
    with _temp_schema() as conninfo:
        assert _migrate(conninfo) == ["migrations/001_threads_messages.sql"]
        assert _migrate(conninfo) == []  # second run is a no-op
        store = PostgresHistoryStore(conninfo, max_size=2)
        store.open()
        try:
            yield store
        finally:
            store.close()


@live
def test_live_concurrent_migration_runs_apply_each_file_once():
    with _temp_schema() as conninfo, ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(_migrate, [conninfo] * 4))
    assert sorted(name for run in runs for name in run) == ["migrations/001_threads_messages.sql"]


@live
def test_live_save_and_read_back(live_store):
    assert live_store.save_turn(_record())
    assert live_store.save_turn(
        _record(turn_id="t2", question="And Tesla?", status="clarification", answer=None)
    )

    [thread] = live_store.list_threads(DEV_USER)
    assert thread.id == THREAD and thread.title == "Compare Apple and Microsoft R&D"
    assert thread.updated_at >= thread.created_at

    messages = live_store.list_messages(DEV_USER, THREAD)
    assert [m.turn_id for m in messages] == ["t1", "t2"]
    first = messages[0]
    assert first.answer == {"answer": "Apple spent more [S1].", "is_valid": True}
    assert first.coverage == [{"ticker": "AAPL", "fiscal_year": 2025, "status": "found"}]
    assert messages[1].answer is None and messages[1].status == "clarification"


@live
def test_live_limit_keeps_the_latest_turns_in_order(live_store):
    for turn in ("t1", "t2", "t3"):
        assert live_store.save_turn(_record(turn_id=turn))
    assert [m.turn_id for m in live_store.list_messages(DEV_USER, THREAD, limit=2)] == ["t2", "t3"]


@live
def test_live_nul_characters_no_longer_fail_the_write(live_store):
    assert live_store.save_turn(_record(question="Q\x00?", answer={"a": "x\x00y"}))
    [message] = live_store.list_messages(DEV_USER, THREAD)
    assert message.question == "Q?" and message.answer == {"a": "xy"}
    assert live_store.failed_writes == 0


@live
def test_live_other_users_cannot_read_or_write_a_thread(live_store):
    assert live_store.save_turn(_record())
    assert live_store.save_turn(_record(user_id=OTHER_USER, turn_id="intruder")) is False
    assert live_store.failed_writes == 1
    assert live_store.list_threads(OTHER_USER) == []
    assert live_store.list_messages(OTHER_USER, THREAD) is None
    assert [m.turn_id for m in live_store.list_messages(DEV_USER, THREAD)] == ["t1"]
    assert live_store.list_messages(DEV_USER, uuid.uuid4()) is None


@live
def test_live_threads_newest_first(live_store):
    second = uuid.uuid4()
    assert live_store.save_turn(_record())
    assert live_store.save_turn(_record(thread_id=second, question="Newer thread"))
    assert [t.id for t in live_store.list_threads(DEV_USER)] == [second, THREAD]
    assert live_store.save_turn(_record(turn_id="t3"))  # activity moves THREAD to the top
    assert [t.id for t in live_store.list_threads(DEV_USER)] == [THREAD, second]
