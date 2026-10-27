"""
Rate Limiting with Redis Backend.

Implements distributed rate limiting using slowapi with Redis storage.
Supports per-IP, per-user, and global rate limits with in-memory fallback for development.

Features:
- Redis-backed distributed rate limiting (production)
- In-memory fallback for development (no Redis required)
- Per-IP rate limiting (fallback when no auth)
- Per-user rate limiting (when JWT is present)
- Global rate limiting for expensive operations
- Configurable limits via environment variables
"""

import logging
import math
import threading
import time
from collections import defaultdict
from typing import Literal

from fastapi import HTTPException, Request, status
from limits import RateLimitItemPerSecond
from limits.storage import storage_from_string
from limits.strategies import MovingWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from api.config import api_settings

logger = logging.getLogger(__name__)


# =============================================================================
# RATE LIMIT KEY FUNCTIONS
# =============================================================================

def get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key based on authentication status.

    Priority:
    1. User ID from JWT (if authenticated)
    2. IP address (fallback for unauthenticated requests)

    This enables per-user rate limiting when auth is enabled,
    while falling back to per-IP for backward compatibility.
    """
    # Try to get user ID from request state (set by auth middleware)
    if hasattr(request.state, "user_id") and request.state.user_id:
        return f"user:{request.state.user_id}"

    # Fallback to IP address
    return f"ip:{get_remote_address(request)}"


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
        self._lock = threading.Lock()

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
        with self._lock:
            return self._is_allowed(key, limit, period)

    def _is_allowed(self, key: str, limit: int, period: int) -> tuple[bool, dict]:
        now = time.time()
        window_start = now - period

        # Clean old requests
        self.requests[key] = [ts for ts in self.requests[key] if ts > window_start]
        if len(self.requests) > 10_000:
            for stale in [k for k, v in self.requests.items() if not v or v[-1] <= window_start]:
                del self.requests[stale]

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
    key_func=get_rate_limit_key,
    storage_uri=redis_url if redis_url else "memory://",
    strategy="moving-window",  # More accurate than fixed-window
    default_limits=[],  # No default limits; we use custom dependencies
)
moving_window = MovingWindowRateLimiter(storage_from_string(redis_url if redis_url else "memory://"))


# =============================================================================
# CUSTOM RATE LIMIT CHECK
# =============================================================================

def check_rate_limit(
    request: Request,
    limit: int,
    period: int,
    key_func: callable = get_rate_limit_key,
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
    key = "global" if scope == "global" else key_func(request)

    outcome = None
    if redis_url:
        try:
            item = RateLimitItemPerSecond(limit, period)
            scoped_key = f"{scope}:{key}"
            allowed = moving_window.hit(item, scoped_key)
            stats = moving_window.get_window_stats(item, scoped_key)
            outcome = allowed, stats.remaining, math.ceil(stats.reset_time)
        except Exception as exc:
            logger.warning(f"Redis rate limiter unavailable ({type(exc).__name__}); using in-memory limits")
    if outcome is None:
        allowed, info = in_memory_limiter.is_allowed(f"{scope}:{limit}/{period}:{key}", limit, period)
        outcome = allowed, info["remaining"], info["reset"]
    allowed, remaining, reset = outcome

    request.state.rate_limit = {"limit": limit, "remaining": remaining, "reset": reset}
    retry_after = 0 if allowed else max(1, reset - int(time.time()))

    if not allowed:
        logger.warning(f"Rate limit exceeded for {key}: {limit}/{period}s")
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
        await run_in_threadpool(check_rate_limit, request, limit, period, scope=scope)

    return check
