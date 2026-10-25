"""
Rate Limiting with Redis Backend.

Implements distributed rate limiting using slowapi with Redis storage.
Supports per-IP and global rate limits with in-memory fallback for development.

Features:
- Redis-backed distributed rate limiting (production)
- In-memory fallback for development (no Redis required)
- Per-IP rate limiting for all endpoints
- Global rate limiting for expensive operations
- Configurable limits via environment variables
"""

import logging
from collections import defaultdict
from typing import Literal

from fastapi import Request, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.config import api_settings

logger = logging.getLogger(__name__)


# =============================================================================
# IN-MEMORY FALLBACK (DEV ONLY)
# =============================================================================

class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter for development when Redis is unavailable.

    Uses a sliding window counter per IP address.
    """

    def __init__(self):
        self.requests = defaultdict(list)  # IP -> list of timestamps

    def is_allowed(self, key: str, limit: int, period: int) -> tuple[bool, dict]:
        """
        Check if request is allowed.

        Args:
            key: Unique identifier (e.g., IP address)
            limit: Maximum requests allowed
            period: Time period in seconds

        Returns:
            (allowed, info) tuple where info contains remaining/reset time
        """
        import time

        now = time.time()
        window_start = now - period

        # Clean old requests
        self.requests[key] = [ts for ts in self.requests[key] if ts > window_start]

        # Check limit
        current_count = len(self.requests[key])
        if current_count >= limit:
            reset_time = int(self.requests[key][0] + period)
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset": reset_time,
            }

        # Add current request
        self.requests[key].append(now)
        return True, {
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset": int(now + period),
        }


# In-memory limiter instance (used only if Redis fails)
in_memory_limiter = InMemoryRateLimiter()


# =============================================================================
# RATE LIMITER SETUP
# =============================================================================

def get_redis_storage():
    """
    Get Redis storage URL or None for in-memory fallback.
    """
    redis_url = api_settings.RATE_LIMIT_REDIS_URL
    if not redis_url:
        logger.info("Rate limit Redis URL not configured, using in-memory storage (development mode)")
        return None

    try:
        # Test connection
        import redis
        client = redis.from_url(redis_url)
        client.ping()
        logger.info(f"Rate limiting using Redis at {redis_url}")
        return redis_url
    except Exception as e:
        logger.warning(f"Failed to connect to Redis for rate limiting: {e}. Falling back to in-memory storage.")
        return None


# Create limiter with moving window strategy (more accurate than fixed window)
redis_url = get_redis_storage()
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url if redis_url else "memory://",
    strategy="moving-window",  # More accurate than fixed-window
    default_limits=[],  # No default limits; we use custom dependencies
)


# =============================================================================
# CUSTOM RATE LIMIT CHECK
# =============================================================================

def check_rate_limit(
    request: Request,
    limit: int,
    period: int,
    key_func: callable = get_remote_address,
    scope: Literal["ip", "global"] = "ip",
) -> None:
    """
    Check rate limit and raise HTTPException if exceeded.

    Args:
        request: FastAPI request object
        limit: Maximum requests allowed
        period: Time period in seconds
        key_func: Function to extract rate limit key (default: IP address)
        scope: Rate limit scope ("ip" or "global")

    Raises:
        HTTPException: If rate limit is exceeded
    """
    key = key_func(request)

    # Use slowapi if Redis is available, otherwise use in-memory fallback
    if redis_url:
        try:
            # Use slowapi's internal check
            limiter._check_request_limit(request, limit=limit, period=period)
        except RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded for {scope}={key}: {limit}/{period}s")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "scope": scope,
                    "limit": limit,
                    "period": period,
                    "retry_after": e.retry_after,
                },
                headers={"Retry-After": str(e.retry_after)},
            )
    else:
        # Fallback to in-memory limiter
        allowed, info = in_memory_limiter.is_allowed(key, limit, period)
        if not allowed:
            logger.warning(f"Rate limit exceeded for {scope}={key}: {limit}/{period}s")
            retry_after = max(0, info["reset"] - int(__import__("time").time()))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "scope": scope,
                    "limit": limit,
                    "period": period,
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )


# =============================================================================
# FASTAPI DEPENDENCY
# =============================================================================

def rate_limit_dependency(
    limit: int,
    period: int,
    scope: Literal["ip", "global"] = "ip",
):
    """
    FastAPI dependency for rate limiting.

    Usage:
        @router.get("/endpoint")
        @router.dependencies([Depends(rate_limit_dependency(10, 60))])
        async def endpoint():
            ...

    Args:
        limit: Maximum requests allowed
        period: Time period in seconds
        scope: Rate limit scope ("ip" or "global")
    """

    async def check(request: Request):
        check_rate_limit(request, limit, period, scope=scope)

    return check
