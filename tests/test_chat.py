"""
/chat: AgentService event streams (every route, clarification resume, expiry, timeout,
errors, disconnect/cancel, persistence, metrics) and the SSE endpoint (validation, wire
format, auth, rate limit). Offline: scripted LLMs and fake retrieval from the graph tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jose import jwt
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent.events import UsageCallback, clarification_payload
from agent.graph import build_graph
from agent.llm import AgentLLM
from agent.schemas import Analysis, ResearchPlan, SynthesisDraft, UIFilters
from agent.service import AgentService, ChatTurn
from api.config import api_settings
from api.dependencies import get_agent_service
from api.main import app
from api.routers.chat import CHAT_LIMIT
from observability import summarize
from tests.test_agent_graph import (
    COMPARISON,
    ScriptedLLM,
    analysis,
    make_deps,
    plan,
    simple_generator,
)

USER = uuid.UUID("11111111-1111-4111-8111-111111111111")
SECRET = "test_jwt_secret_for_testing"
REFUSED = Analysis(
    intent="general_knowledge", scope="general_knowledge", question_type="qualitative"
)
GREETING = Analysis(intent="greeting", scope="specific", question_type="qualitative")


class FakeHistory:
    enabled = True

    def __init__(self) -> None:
        self.records: list[Any] = []

    def save_turn(self, record: Any) -> bool:
        self.records.append(record)
        return True


class FakeWriter:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def write(self, record: Any) -> None:
        self.records.append(record)


class SlowLLM(ScriptedLLM):
    """Blocks in analyze until cancelled; records the cancellation."""

    def __init__(self) -> None:
        super().__init__({})
        self.cancelled = asyncio.Event()

    async def structured(self, schema: type[Any], messages: list[Any]) -> Any:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def make_service(llm: Any, *, timeout: float = 30.0, **deps_kwargs: Any) -> AgentService:
    deps = make_deps(llm, **deps_kwargs)
    return AgentService(
        build_graph(deps),
        deps.catalog,
        evidence=deps.evidence,
        history=FakeHistory(),
        writer=FakeWriter(),  # type: ignore[arg-type]
        timeout_seconds=timeout,
        provider="ollama",
        model="fake-model",
        poll_seconds=0.02,
    )


def turn(thread: uuid.UUID | None = None, owner: str = "local", **values: Any) -> ChatTurn:
    return ChatTurn(thread_id=thread or uuid.uuid4(), owner=owner, history_user=USER, **values)


async def collect(service: AgentService, chat_turn: ChatTurn, **kwargs: Any) -> list[Any]:
    return [event async for event in service.stream(chat_turn, **kwargs)]


def names(events: list[Any]) -> list[str]:
    """Event names with consecutive step/tool events collapsed."""
    out: list[str] = []
    for event in events:
        name = "progress" if event.event in {"step", "tool"} else event.event
        if not out or out[-1] != name:
            out.append(name)
    return out


def only(events: list[Any], name: str) -> list[dict[str, Any]]:
    return [event.data for event in events if event.event == name]


# =============================================================================
# SERVICE: ROUTES
# =============================================================================


@pytest.mark.asyncio
async def test_simple_rag_stream_order_usage_and_persistence() -> None:
    service = make_service(
        ScriptedLLM({Analysis: [analysis(companies=["AAPL"])]}), generator=simple_generator()
    )
    events = await collect(service, turn(message="Apple revenue?", request_id="corr-1"))

    assert names(events) == ["thread", "progress", "answer", "done"]
    thread = only(events, "thread")[0]
    answer = only(events, "answer")[0]
    assert answer["route"] == "simple_rag" and answer["coverage"][0]["status"] == "found"
    done = only(events, "done")[0]
    assert (done["prompt_tokens"], done["completion_tokens"], done["llm_calls"]) == (10, 5, 1)
    assert answer["usage"]["prompt_tokens"] == 10

    steps = only(events, "step")
    assert {"node": "analyze", "status": "completed"}.items() <= steps[1].items()
    assert all(isinstance(step["elapsed_ms"], int) for step in steps)

    (record,) = service.history.records  # type: ignore[attr-defined]
    assert record.status == "answered" and record.turn_id == thread["turn_id"]
    assert record.question == "Apple revenue?" and record.citations[0]["source_id"] == "S1"
    assert record.scope["tickers"] == ["AAPL"] and record.model == "fake-model"

    (metric,) = service.writer.records  # type: ignore[union-attr]
    assert metric.status == "success" and metric.outcome == "answer"
    assert metric.request_id == "corr-1" and "simple_rag" in metric.node_seconds


@pytest.mark.asyncio
async def test_refusal_lists_coverage_and_examples_and_is_saved_as_refused() -> None:
    service = make_service(ScriptedLLM({Analysis: [REFUSED]}))
    events = await collect(service, turn(message="What is EBITDA?"))

    answer = only(events, "answer")[0]
    assert answer["route"] == "refuse"
    assert "AAPL, MSFT" in answer["answer"] and "For example:" in answer["answer"]
    assert service.history.records[0].status == "refused"  # type: ignore[attr-defined]
    assert service.writer.records[0].outcome == "refused"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_research_fan_out_streams_tools_and_records_metrics() -> None:
    llm = ScriptedLLM(
        {
            Analysis: [analysis("cross_company", companies=["Apple", "Microsoft"])],
            ResearchPlan: [plan("revenue")],
            SynthesisDraft: [COMPARISON],
        }
    )
    service = make_service(llm)
    events = await collect(service, turn(message="Compare revenue"))

    assert names(events) == ["thread", "progress", "answer", "done"]
    tools = only(events, "tool")
    assert sorted(tool["ticker"] for tool in tools) == ["AAPL", "MSFT"]
    started = [step["label"] for step in only(events, "step") if step["status"] == "started"]
    assert "Planning 2 searches" in started and "Verifying citations" in started
    executed = [
        step["detail"]
        for step in only(events, "step")
        if step["node"] == "execute" and step["status"] == "completed"
    ]
    assert sorted(executed) == ["AAPL FY2025: found", "MSFT FY2025: found"]
    answer = only(events, "answer")[0]
    assert answer["is_valid"] and [s["source_id"] for s in answer["sources"]] == ["S1", "S2"]
    assert only(events, "done")[0]["tool_calls"] == 2

    (metric,) = service.writer.records  # type: ignore[union-attr]
    assert metric.route == "research" and metric.tool_calls == 2 and metric.is_valid
    assert metric.coverage == {"found": 2} and metric.replans == 0 and metric.repairs == 0
    assert {"plan", "execute", "synthesize", "validate", "finalize"} <= metric.node_seconds.keys()
    summary = summarize([metric.to_dict()])
    assert summary["agent"]["turns"] == 1 and summary["reliability"]["requests"] == 0


@pytest.mark.asyncio
async def test_llm_tokens_are_counted_through_the_run_callbacks() -> None:
    reply = AIMessage(
        content='{"intent": "greeting", "scope": "specific", "question_type": "qualitative"}',
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    deps_llm = AgentLLM(GenericFakeChatModel(messages=iter([reply] * 3)))
    service = make_service(ScriptedLLM({}))
    service.graph = build_graph(make_deps(deps_llm))  # type: ignore[arg-type]
    events = await collect(service, turn(message="Hi"))

    assert only(events, "answer")[0]["route"] == "onboard"
    done = only(events, "done")[0]
    assert done["llm_calls"] >= 1 and done["prompt_tokens"] >= 7 and done["completion_tokens"] >= 3


def test_usage_callback_sums_usage_metadata() -> None:
    callback = UsageCallback()
    callback.on_chat_model_start()
    message = AIMessage(
        content="", usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
    )
    callback.on_llm_end(type("R", (), {"generations": [[type("G", (), {"message": message})()]]})())
    assert (callback.llm_calls, callback.prompt_tokens, callback.completion_tokens) == (1, 4, 2)


# =============================================================================
# SERVICE: CLARIFICATION
# =============================================================================


@pytest.mark.asyncio
async def test_clarification_then_resume_by_option_continues_the_same_turn() -> None:
    service = make_service(ScriptedLLM({Analysis: [analysis()]}), generator=simple_generator())
    thread = uuid.uuid4()
    first = await collect(service, turn(thread, message="What was revenue?"))

    assert names(first) == ["thread", "progress", "clarification", "done"]
    clarification = only(first, "clarification")[0]
    assert clarification["options"] == [
        {"id": "opt_1", "label": "AAPL"},
        {"id": "opt_2", "label": "MSFT"},
        {"id": "opt_3", "label": "All"},
    ]
    assert clarification["allow_free_text"] is True

    second = await collect(service, turn(thread, option_id="opt_1"))
    assert names(second) == ["thread", "progress", "answer", "done"]
    assert only(second, "thread")[0]["turn_id"] == only(first, "thread")[0]["turn_id"]
    assert only(second, "answer")[0]["route"] == "simple_rag"

    saved = service.history.records  # type: ignore[attr-defined]
    assert [record.status for record in saved] == ["clarification", "answered"]
    assert saved[0].answer == {"clarification": clarification}
    assert {record.question for record in saved} == {"What was revenue?"}
    assert [m.outcome for m in service.writer.records] == ["clarification", "answer"]  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_resume_with_free_text() -> None:
    service = make_service(ScriptedLLM({Analysis: [analysis()]}), generator=simple_generator())
    thread = uuid.uuid4()
    await collect(service, turn(thread, message="What was revenue?"))
    events = await collect(service, turn(thread, text="Apple"))
    assert only(events, "answer")[0]["coverage"][0]["ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_resume_errors_expired_invalid_option_and_other_owner() -> None:
    service = make_service(ScriptedLLM({Analysis: [analysis()]}), generator=simple_generator())

    expired = await collect(service, turn(option_id="opt_1", request_id="corr-9"))
    assert names(expired) == ["thread", "error", "done"]
    error = only(expired, "error")[0]
    assert error["code"] == "clarification_expired" and error["correlation_id"] == "corr-9"
    assert service.history.records == []  # type: ignore[attr-defined]

    thread = uuid.uuid4()
    await collect(service, turn(thread, message="What was revenue?"))
    invalid = await collect(service, turn(thread, option_id="opt_9"))
    assert only(invalid, "error")[0]["code"] == "invalid_choice"

    # Checkpoints are namespaced by owner: another user's resume does not find the pause.
    foreign = await collect(service, turn(thread, owner="someone-else", option_id="opt_1"))
    assert only(foreign, "error")[0]["code"] == "clarification_expired"
    still_pending = await collect(service, turn(thread, option_id="opt_1"))
    assert only(still_pending, "answer")[0]["route"] == "simple_rag"


@pytest.mark.asyncio
async def test_new_message_replaces_a_pending_clarification() -> None:
    service = make_service(
        ScriptedLLM({Analysis: [analysis(), analysis(companies=["AAPL"])]}),
        generator=simple_generator(),
    )
    thread = uuid.uuid4()
    await collect(service, turn(thread, message="What was revenue?"))
    events = await collect(service, turn(thread, message="Apple revenue?"))
    assert only(events, "answer")[0]["route"] == "simple_rag"
    expired = await collect(service, turn(thread, option_id="opt_1"))
    assert only(expired, "error")[0]["code"] == "clarification_expired"


@pytest.mark.asyncio
async def test_ui_filters_skip_clarification_and_reset_on_the_next_turn() -> None:
    service = make_service(
        ScriptedLLM({Analysis: [analysis(), analysis()]}), generator=simple_generator()
    )
    thread = uuid.uuid4()
    chips = UIFilters(tickers=["AAPL"], fiscal_years=[2025])
    first = await collect(service, turn(thread, message="What was revenue?", filters=chips))
    assert only(first, "answer")[0]["route"] == "simple_rag"
    second = await collect(service, turn(thread, message="What was revenue?"))
    assert names(second)[-2:] == ["clarification", "done"]


def test_clarification_payload_numbers_options() -> None:
    payload = clarification_payload({"question": "Which?", "options": ["A", "B"]})
    assert payload["options"][1] == {"id": "opt_2", "label": "B"} and payload["allow_free_text"]


# =============================================================================
# SERVICE: FAILURES, TIMEOUT, CANCELLATION
# =============================================================================


@pytest.mark.asyncio
async def test_graph_failure_is_an_error_event_without_internals() -> None:
    service = make_service(ScriptedLLM({}))  # no scripted Analysis -> KeyError in analyze
    events = await collect(service, turn(message="Apple revenue?"))

    assert names(events) == ["thread", "progress", "error", "done"]
    error = only(events, "error")[0]
    assert error["code"] == "agent_error" and "KeyError" not in json.dumps(error)
    assert service.history.records[0].status == "error"  # type: ignore[attr-defined]
    metric = service.writer.records[0]  # type: ignore[union-attr]
    assert (metric.status, metric.error_type) == ("error", "KeyError")


@pytest.mark.asyncio
async def test_timeout_cancels_the_run_and_reports_it() -> None:
    llm = SlowLLM()
    service = make_service(llm, timeout=0.1)
    events = await collect(service, turn(message="Apple revenue?"))

    assert names(events) == ["thread", "progress", "error", "done"]
    assert only(events, "error")[0]["code"] == "timeout"
    assert llm.cancelled.is_set()
    assert service.writer.records[0].status == "timeout"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_client_disconnect_cancels_the_graph() -> None:
    llm = SlowLLM()
    service = make_service(llm)

    async def disconnected() -> bool:
        return True

    events = await collect(service, turn(message="Apple revenue?"), is_disconnected=disconnected)
    assert "done" not in names(events)
    assert llm.cancelled.is_set()
    assert service.history.records == []  # type: ignore[attr-defined]
    assert service.writer.records[0].status == "cancelled"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_consumer_closing_the_stream_cancels_the_graph() -> None:
    llm = SlowLLM()
    service = make_service(llm)
    stream = service.stream(turn(message="Apple revenue?"))
    assert (await anext(stream)).event == "thread"
    assert (await anext(stream)).event == "step"
    await stream.aclose()
    await asyncio.wait_for(llm.cancelled.wait(), 1)
    assert service.writer.records[0].status == "cancelled"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_closing_the_stream_after_the_answer_does_not_double_count_usage() -> None:
    reply = AIMessage(
        content='{"intent": "greeting", "scope": "specific", "question_type": "qualitative"}',
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    service = make_service(ScriptedLLM({}))
    service.graph = build_graph(make_deps(AgentLLM(GenericFakeChatModel(messages=iter([reply])))))
    stream = service.stream(turn(message="Hi"))
    async for event in stream:
        if event.event == "answer":
            sent = event.data["usage"]
            break
    await stream.aclose()

    (metric,) = service.writer.records  # type: ignore[union-attr]
    assert metric.status == "cancelled"
    assert (metric.prompt_tokens, metric.completion_tokens, metric.llm_calls) == (
        sent["prompt_tokens"],
        sent["completion_tokens"],
        sent["llm_calls"],
    )
    assert metric.prompt_tokens == 7


@pytest.mark.asyncio
async def test_second_message_on_a_busy_thread_is_rejected() -> None:
    llm = SlowLLM()
    service = make_service(llm, timeout=0.3)
    thread = uuid.uuid4()
    first = asyncio.create_task(collect(service, turn(thread, message="Apple revenue?")))
    await asyncio.sleep(0.05)
    busy = await collect(service, turn(thread, message="Again?"))
    assert only(busy, "error")[0]["code"] == "thread_busy"
    await first


@pytest.mark.asyncio
async def test_nothing_is_saved_without_a_history_owner() -> None:
    service = make_service(ScriptedLLM({Analysis: [GREETING]}))
    await collect(service, ChatTurn(thread_id=uuid.uuid4(), owner="local", message="Hi"))
    assert service.history.records == []  # type: ignore[attr-defined]


# =============================================================================
# SERVICE: TRACE (AGENT_TRACE)
# =============================================================================


def read_trace(directory: Any) -> dict[str, Any]:
    (path,) = directory.glob("*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def kinds(trace: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [event for event in trace["events"] if event["kind"] == kind]


@pytest.mark.asyncio
async def test_trace_records_steps_tools_and_node_updates_of_a_research_turn(
    tmp_path, caplog
) -> None:
    llm = ScriptedLLM(
        {
            Analysis: [analysis("cross_company", companies=["Apple", "Microsoft"])],
            ResearchPlan: [plan("revenue")],
            SynthesisDraft: [COMPARISON],
        }
    )
    service = make_service(llm)
    service.trace_dir = tmp_path
    with caplog.at_level(logging.INFO, logger="agent.trace"):
        await collect(service, turn(message="Compare revenue", request_id="corr-7"))

    trace = read_trace(tmp_path)
    assert trace["question"] == "Compare revenue" and trace["request_id"] == "corr-7"
    assert trace["summary"]["status"] == "success" and trace["summary"]["final"]["is_valid"]
    tools = kinds(trace, "tool")
    assert sorted(tool["ticker"] for tool in tools) == ["AAPL", "MSFT"]
    assert all(tool["args"]["query"] for tool in tools)
    nodes = [event["node"] for event in kinds(trace, "node")]
    assert nodes[:3] == ["analyze", "resolve_scope", "plan"] and nodes[-1] == "finalize"
    synthesize = next(event for event in kinds(trace, "node") if event["node"] == "synthesize")
    assert synthesize["update"]["draft"]["claims"]
    assert [event["seq"] for event in trace["events"]] == list(range(1, len(trace["events"]) + 1))
    assert any("retrieval T1 search_filings" in record.message for record in caplog.records)
    assert any("written to" in record.message for record in caplog.records)

    # Retrieval detail: every hit with its filing and reranker score, plus timing.
    retrievals = kinds(trace, "retrieval")
    assert sorted(event["task_id"] for event in retrievals) == ["T1", "T2"]
    hit = retrievals[0]["hits"][0]
    assert hit["score"] == 0.9 and hit["chunk_id"].endswith("_c0000") and hit["fiscal_year"]
    assert retrievals[0]["seconds"] >= 0 and retrievals[0]["filters"]["ticker"]

    # Code decisions: which evidence became S1/S2 for synthesis.
    (context,) = [e for e in kinds(trace, "decision") if e["decision"] == "context_selection"]
    assert [source["source_id"] for source in context["sources"]] == ["S1", "S2"]
    assert context["dropped"] == []

    # Around the agent: the request, what the client received, history and metrics.
    (request,) = kinds(trace, "request")
    assert request["message"] == "Compare revenue" and request["request"] == "message"
    assert [event["event"] for event in kinds(trace, "sent")] == ["answer", "done"]
    assert kinds(trace, "history")[0]["saved"] is True and kinds(trace, "metrics")


@pytest.mark.asyncio
async def test_trace_records_each_llm_call_with_node_prompt_and_raw_output(tmp_path) -> None:
    reply = AIMessage(
        content='{"intent": "greeting", "scope": "specific", "question_type": "qualitative"}',
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    service = make_service(ScriptedLLM({}))
    service.graph = build_graph(  # type: ignore[arg-type]
        make_deps(AgentLLM(GenericFakeChatModel(messages=iter([reply] * 3))))
    )
    service.trace_dir = tmp_path
    await collect(service, turn(message="Hi"))

    trace = read_trace(tmp_path)
    start, end = kinds(trace, "llm_start")[0], kinds(trace, "llm_end")[0]
    assert (start["node"], start["call"]) == ("analyze", 1)
    assert start["messages"][-1]["role"] == "human"
    assert start["messages"][-1]["content"].endswith("Question: Hi")
    assert start["messages"][0]["role"] == "system" and "Indexed filings" in str(start["messages"])
    assert (end["node"], end["call"]) == ("analyze", 1) and '"greeting"' in end["output"]
    assert end["usage"]["input_tokens"] == 7 and end["seconds"] >= 0


@pytest.mark.asyncio
async def test_failed_turn_still_writes_its_trace_with_the_traceback(tmp_path) -> None:
    service = make_service(ScriptedLLM({}))  # no scripted Analysis -> KeyError in analyze
    service.trace_dir = tmp_path
    await collect(service, turn(message="Apple revenue?"))

    trace = read_trace(tmp_path)
    assert (trace["summary"]["status"], trace["summary"]["error_type"]) == ("error", "KeyError")
    (error,) = kinds(trace, "error")
    assert error["error"].startswith("KeyError") and "Traceback" in error["traceback"]
    assert kinds(trace, "sent")[0]["data"]["code"] == "agent_error"


@pytest.mark.asyncio
async def test_errors_mode_writes_traces_only_for_turns_that_need_attention(tmp_path) -> None:
    service = make_service(ScriptedLLM({Analysis: [GREETING]}))
    service.trace_dir, service.trace_mode = tmp_path, "errors"
    await collect(service, turn(message="Hi"))
    assert not list(tmp_path.glob("*.json"))

    service = make_service(ScriptedLLM({}))  # fails in analyze
    service.trace_dir, service.trace_mode = tmp_path, "errors"
    await collect(service, turn(message="Apple revenue?"))
    assert read_trace(tmp_path)["summary"]["status"] == "error"


@pytest.mark.asyncio
async def test_rejected_turn_is_traced_and_logged(tmp_path, caplog) -> None:
    service = make_service(ScriptedLLM({}))
    service.trace_dir = tmp_path
    with caplog.at_level(logging.INFO, logger="agent.service"):
        await collect(service, turn(option_id="opt_1"))  # nothing pending to resume

    trace = read_trace(tmp_path)
    assert trace["summary"]["error_code"] == "clarification_expired"
    assert trace["turn_id"] is None and "_req-" in next(tmp_path.glob("*.json")).name
    assert kinds(trace, "request")[0]["rejected"] == "clarification_expired"
    finished = [r.message for r in caplog.records if r.message.startswith("Chat turn finished")]
    assert finished and "error=clarification_expired" in finished[0]


@pytest.mark.asyncio
async def test_every_log_line_of_a_turn_carries_its_ids() -> None:
    from config.logging import LogContextFilter, log_context

    service = make_service(ScriptedLLM({Analysis: [GREETING]}))
    handler = logging.Handler()
    handler.addFilter(LogContextFilter())
    records: list[logging.LogRecord] = []
    handler.emit = records.append  # type: ignore[method-assign]
    logging.getLogger("agent").addHandler(handler)
    try:
        with log_context(request_id="req-12345678-abcd"):
            events = await collect(service, turn(message="Hi"))
    finally:
        logging.getLogger("agent").removeHandler(handler)

    turn_id = only(events, "thread")[0]["turn_id"]
    (finished,) = [r for r in records if r.getMessage().startswith("Chat turn finished")]
    assert (finished.request_id, finished.turn_id) == ("req-12345678-abcd", turn_id)  # type: ignore[attr-defined]
    assert finished.log_ctx == f"[req=req-1234 turn={turn_id[:8]}] "  # type: ignore[attr-defined]


def test_json_log_lines_include_correlation_ids_and_extra_fields() -> None:
    from config.logging import JSONFormatter, LogContextFilter, log_context

    record = logging.LogRecord("api.middleware", logging.INFO, "x.py", 1, "POST /chat", (), None)
    record.status_code = 200
    with log_context(request_id="r1", turn_id="t1"):
        LogContextFilter().filter(record)
    line = json.loads(JSONFormatter().format(record))
    assert (line["request_id"], line["turn_id"], line["status_code"]) == ("r1", "t1", 200)
    assert "log_ctx" not in line and "thread_id" not in line


def test_agent_trace_accepts_modes_and_booleans() -> None:
    from config.settings import Settings

    assert [
        Settings(AGENT_TRACE=value).AGENT_TRACE for value in ("true", "false", "errors", "")
    ] == [  # type: ignore[arg-type]
        "all",
        "off",
        "errors",
        "off",
    ]


# =============================================================================
# API
# =============================================================================


def parse_sse(text: str) -> list[tuple[str, Any]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        name = next((line[7:] for line in lines if line.startswith("event: ")), None)
        data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
        if name:
            events.append((name, json.loads(data)))
    return events


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        from api.rate_limit import in_memory_limiter

        in_memory_limiter.requests.clear()


def use(service: AgentService | None) -> None:
    app.dependency_overrides[get_agent_service] = lambda: service


def test_chat_is_503_when_the_agent_is_unavailable(client) -> None:
    use(None)
    response = client.post("/chat", json={"message": "Apple revenue?"})
    assert response.status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": "Apple?", "resume": {"text": "Apple"}, "thread_id": str(uuid.uuid4())},
        {"resume": {"option_id": "opt_1"}},
        {"thread_id": str(uuid.uuid4()), "resume": {"option_id": "opt_1", "text": "Apple"}},
        {"thread_id": str(uuid.uuid4()), "resume": {"option_id": "drop table"}},
        {"message": "   "},
        {"message": "x" * 2001},
        {"message": "Apple?", "filters": {"tickers": ["A", "B", "C", "D", "E", "F", "G"]}},
        {"message": "Apple?", "thread_id": "not-a-uuid"},
        {"message": "Apple?", "extra": True},
    ],
)
def test_chat_request_validation(client, body) -> None:
    use(make_service(ScriptedLLM({})))
    assert client.post("/chat", json=body).status_code == 422


def test_chat_filters_are_checked_against_the_catalog(client) -> None:
    use(make_service(ScriptedLLM({})))
    unknown = client.post("/chat", json={"message": "Rev?", "filters": {"tickers": ["NVDA"]}})
    assert unknown.status_code == 422 and "NVDA" in unknown.json()["detail"]
    year = client.post(
        "/chat", json={"message": "Rev?", "filters": {"tickers": ["MSFT"], "fiscal_years": [2024]}}
    )
    assert year.status_code == 422 and "2024" in year.json()["detail"]


def test_chat_streams_server_sent_events(client) -> None:
    service = make_service(ScriptedLLM({Analysis: [analysis()]}), generator=simple_generator())
    use(service)
    response = client.post(
        "/chat",
        json={"message": "Revenue?", "filters": {"tickers": ["aapl"], "fiscal_years": [2025]}},
        headers={"X-Correlation-ID": "corr-42"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    assert events[0][0] == "thread" and events[-1][0] == "done"
    assert [name for name, _ in events if name in {"answer", "clarification", "error"}] == [
        "answer"
    ]
    thread_id = events[0][1]["thread_id"]
    assert uuid.UUID(thread_id)
    assert service.writer.records[0].request_id == "corr-42"  # type: ignore[union-attr]
    # Auth is off and HISTORY_DEV_USER_ID is unset in tests: nothing is saved.
    assert service.history.records == []  # type: ignore[attr-defined]


def test_chat_resume_over_http(client, monkeypatch) -> None:
    monkeypatch.setattr(api_settings, "HISTORY_DEV_USER_ID", USER)
    service = make_service(ScriptedLLM({Analysis: [analysis()]}), generator=simple_generator())
    use(service)
    first = parse_sse(client.post("/chat", json={"message": "What was revenue?"}).text)
    thread_id = first[0][1]["thread_id"]
    assert [name for name, _ in first][-2:] == ["clarification", "done"]
    second = parse_sse(
        client.post("/chat", json={"thread_id": thread_id, "resume": {"option_id": "opt_2"}}).text
    )
    answer = next(data for name, data in second if name == "answer")
    assert answer["coverage"][0]["ticker"] == "MSFT"
    saved = service.history.records  # type: ignore[attr-defined]
    assert [r.status for r in saved] == ["clarification", "answered"]
    assert {r.user_id for r in saved} == {USER} and {str(r.thread_id) for r in saved} == {thread_id}


def test_chat_requires_sign_in_when_auth_is_on(client, monkeypatch) -> None:
    monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", SECRET)
    service = make_service(ScriptedLLM({Analysis: [GREETING]}))
    use(service)
    assert client.post("/chat", json={"message": "Hi"}).status_code == 401
    token = jwt.encode({"sub": str(USER), "aud": "authenticated"}, SECRET, algorithm="HS256")
    response = client.post(
        "/chat", json={"message": "Hi"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert [record.user_id for record in service.history.records] == [USER]  # type: ignore[attr-defined]


def test_chat_is_rate_limited(client) -> None:
    service = make_service(ScriptedLLM({Analysis: [GREETING] * (CHAT_LIMIT + 1)}))
    use(service)
    statuses = [
        client.post("/chat", json={"message": "Hi"}).status_code for _ in range(CHAT_LIMIT + 1)
    ]
    assert statuses[:-1] == [200] * CHAT_LIMIT and statuses[-1] == 429
