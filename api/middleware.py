"""
API Middleware.

Custom middleware for request tracking, correlation IDs, and logging.
"""

import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from api.auth import optional_get_current_user

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add correlation IDs to all requests for tracing and user tracking.

    - Generates a correlation ID if not present in headers
    - Adds it to request state for use in endpoints
    - Includes it in response headers
    - Logs it for distributed tracing
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

        # Add to request state
        request.state.correlation_id = correlation_id

        # Log request with correlation ID
        logger.info(
            f"Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else None,
            }
        )

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        # Expose the caller's remaining quota (set by the rate limit dependency)
        rate_limit = getattr(request.state, "rate_limit", None)
        if rate_limit:
            response.headers["X-RateLimit-Limit"] = str(rate_limit["limit"])
            response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limit["remaining"]))
            response.headers["X-RateLimit-Reset"] = str(rate_limit["reset"])

        # Log response with correlation ID
        logger.info(
            f"Request completed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
            }
        )

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detailed request/response logging.

    Logs request details including IP, user agent, and timing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import time

        start_time = time.time()

        # Log request details
        logger.debug(
            f"Incoming request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log response details
        logger.debug(
            f"Outgoing response",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            }
        )

        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract user information from JWT and add to request state.

    This enables per-user rate limiting without requiring authentication on every endpoint.
    If no valid token is provided, the request continues without user info (rate limiting falls back to IP).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Try to extract user from Authorization header
        authorization = request.headers.get("Authorization")

        if authorization and authorization.startswith("Bearer "):
            try:
                from api.auth import verify_supabase_token
                token = authorization[7:]  # Remove "Bearer " prefix
                # Threadpool: the first call may fetch the JWKS over the network
                user = await run_in_threadpool(verify_supabase_token, token)
                # Add user info to request state for rate limiting
                request.state.user_id = user.user_id
                request.state.user_email = user.email
                logger.debug(f"Authenticated request from user: {user.user_id}")
            except Exception as e:
                # Auth failed, but continue without user info (rate limiting will use IP)
                logger.debug(f"Auth middleware: failed to verify token: {e}")

        # Process request
        return await call_next(request)
