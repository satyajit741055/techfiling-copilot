"""
Chat Router.

POST /chat runs one research-agent turn and streams its progress as Server-Sent Events:

    thread → step / tool … → answer | clarification | error → done

Request errors (validation, unknown chips, auth, rate limit, agent unavailable) are
normal JSON responses before the stream starts. Once streaming, failures arrive as an
``error`` event without internals.
"""

import logging
from collections.abc import AsyncIterable
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from agent.schemas import UIFilters
from agent.service import AgentService, ChatTurn
from api.auth import history_user_id
from api.config import api_settings
from api.dependencies import get_agent_service
from api.rate_limit import rate_limit_dependency
from api.schemas import ChatFilters, ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()

CHAT_LIMIT = api_settings.RATE_LIMIT_CHAT_PER_MINUTE or api_settings.RATE_LIMIT_QUESTIONS_PER_MINUTE


def _check_filters(service: AgentService, filters: ChatFilters) -> None:
    catalog = service.catalog
    unknown = [ticker for ticker in filters.tickers if ticker not in catalog.tickers]
    tickers = filters.tickers or catalog.tickers
    missing = [
        year
        for year in filters.fiscal_years
        if not any(catalog.has(ticker, year) for ticker in tickers)
    ]
    if unknown or missing:
        problems = []
        if unknown:
            problems.append(f"unknown tickers {unknown}")
        if missing:
            problems.append(f"fiscal years not indexed {missing}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid filters: {'; '.join(problems)}. Coverage: {catalog.describe()}",
        )


def prepare_turn(
    request: Request,
    body: Annotated[ChatRequest, Body()],
    service: AgentService | None = Depends(get_agent_service),
) -> tuple[AgentService, ChatTurn]:
    """Validate the request against the catalog before the stream starts."""
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research chat is unavailable right now. Please try again later.",
        )
    if body.filters is not None:
        _check_filters(service, body.filters)
    filters = (
        UIFilters(tickers=body.filters.tickers, fiscal_years=body.filters.fiscal_years)
        if body.filters is not None and (body.filters.tickers or body.filters.fiscal_years)
        else None
    )
    owner_id = history_user_id(request)
    # Checkpoints are namespaced by owner, so a thread id alone never opens another
    # user's conversation state.
    owner = str(owner_id or getattr(request.state, "user_id", None) or "local")
    turn = ChatTurn(
        thread_id=body.thread_id or uuid4(),
        owner=owner,
        message=body.message,
        option_id=body.resume.option_id if body.resume else None,
        text=body.resume.text if body.resume else None,
        filters=filters,
        history_user=owner_id,
        request_id=getattr(request.state, "correlation_id", None),
    )
    return service, turn


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    summary="Research chat (streamed)",
    description=(
        "Ask a question across the indexed 10-K filings, or answer a pending clarification "
        "(`resume` with the same `thread_id`). Streams `thread`, `step`, `tool`, then "
        "`answer` | `clarification` | `error`, and always ends with `done`."
    ),
    responses={
        200: {"description": "text/event-stream of chat events"},
        401: {"description": "Sign-in required"},
        422: {"description": "Invalid request or filters"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Research chat unavailable"},
    },
    dependencies=[
        Depends(
            rate_limit_dependency(
                limit=api_settings.RATE_LIMIT_GLOBAL_PER_MINUTE, period=60, scope="global"
            )
        ),
        Depends(rate_limit_dependency(limit=CHAT_LIMIT, period=60, scope="ip")),
    ],
)
async def chat(
    request: Request,
    prepared: tuple[AgentService, ChatTurn] = Depends(prepare_turn),
) -> AsyncIterable[ServerSentEvent]:
    service, turn = prepared
    async for event in service.stream(turn, request.is_disconnected):
        yield ServerSentEvent(event=event.event, data=event.data)
