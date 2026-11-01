"""
History Router.

Read-only access to the current user's saved conversation threads.
Returns 404 when history is off, when there is no history owner (auth disabled and no
HISTORY_DEV_USER_ID), or when the thread does not belong to the caller.
"""

import logging
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from api.auth import history_user_id
from api.config import api_settings
from api.dependencies import get_history_store
from api.rate_limit import rate_limit_dependency
from api.schemas import ThreadMessagesResponse, ThreadsResponse
from persistence import HistoryStore
from persistence.history import MAX_MESSAGES, MAX_THREADS

logger = logging.getLogger(__name__)

router = APIRouter()

RATE_LIMITS = [
    Depends(
        rate_limit_dependency(
            limit=api_settings.RATE_LIMIT_GLOBAL_PER_MINUTE, period=60, scope="global"
        )
    ),
    Depends(
        rate_limit_dependency(
            limit=api_settings.RATE_LIMIT_DEFAULT_PER_MINUTE, period=60, scope="ip"
        )
    ),
]


def _not_found() -> HTTPException:
    # A fresh instance each time: re-raising one shared exception keeps growing its
    # traceback (and the request frames it references) on every request.
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


def _owner(request: Request, store: HistoryStore) -> UUID:
    user_id = history_user_id(request)
    if not store.enabled or user_id is None:
        raise _not_found()
    return user_id


def _unavailable(exc: psycopg.Error) -> HTTPException:
    logger.warning(f"History read failed ({type(exc).__name__})")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="History is temporarily unavailable. Please try again.",
    )


@router.get(
    "/threads",
    response_model=ThreadsResponse,
    summary="List conversation threads",
    description="The current user's threads, most recently updated first",
    responses={
        404: {"description": "History is not enabled"},
        429: {"description": "Rate limit exceeded"},
    },
    dependencies=RATE_LIMITS,
)
def list_threads(
    request: Request,
    limit: int = Query(default=50, ge=1, le=MAX_THREADS),
    store: HistoryStore = Depends(get_history_store),
) -> ThreadsResponse:
    user_id = _owner(request, store)
    try:
        return ThreadsResponse(threads=store.list_threads(user_id, limit=limit))
    except psycopg.Error as exc:
        raise _unavailable(exc) from exc


@router.get(
    "/threads/{thread_id}/messages",
    response_model=ThreadMessagesResponse,
    summary="List a thread's messages",
    description="The latest saved turns of one of the current user's threads, oldest first",
    responses={
        404: {"description": "Thread not found"},
        429: {"description": "Rate limit exceeded"},
    },
    dependencies=RATE_LIMITS,
)
def list_thread_messages(
    thread_id: UUID,
    request: Request,
    limit: int = Query(default=MAX_MESSAGES, ge=1, le=MAX_MESSAGES),
    store: HistoryStore = Depends(get_history_store),
) -> ThreadMessagesResponse:
    user_id = _owner(request, store)
    try:
        messages = store.list_messages(user_id, thread_id, limit=limit)
    except psycopg.Error as exc:
        raise _unavailable(exc) from exc
    if messages is None:
        raise _not_found()
    return ThreadMessagesResponse(thread_id=thread_id, messages=messages)
