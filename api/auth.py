"""
Authentication with Supabase JWT.

Validates Supabase JWT tokens from the Authorization header and extracts user information.
This enables per-user rate limiting and user tracking.
"""

import logging
import threading
import time
from typing import Annotated

import httpx
from fastapi import Header, HTTPException, Request, status
from jose import JWTError, jwt
from pydantic import BaseModel

from api.config import api_settings

logger = logging.getLogger(__name__)


ASYMMETRIC_ALGORITHMS = ("ES256", "RS256")
JWKS_TTL_SECONDS = 600
JWKS_TIMEOUT_SECONDS = 5

_jwks_cache: dict = {"keys": None, "fetched_at": 0.0}
_jwks_lock = threading.Lock()


class AuthUser(BaseModel):
    """Authenticated user information extracted from JWT."""

    user_id: str
    email: str | None = None
    role: str = "authenticated"


def _fetch_jwks() -> list[dict]:
    """Fetch the project's public signing keys from Supabase."""
    url = f"{api_settings.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    response = httpx.get(url, timeout=JWKS_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("keys", [])


def _get_signing_key(kid: str | None) -> dict:
    """
    Return the public JWK matching `kid`, using a TTL cache.

    An unknown kid triggers one refresh (key rotation), rate-limited to avoid
    letting forged tokens force a fetch on every request.
    """

    def find(keys):
        return next((k for k in keys or [] if k.get("kid") == kid), None)

    with _jwks_lock:
        now = time.time()
        fresh = (
            _jwks_cache["keys"] is not None and now - _jwks_cache["fetched_at"] < JWKS_TTL_SECONDS
        )
        key = find(_jwks_cache["keys"]) if fresh else None
        if key:
            return key
        if not fresh or now - _jwks_cache["fetched_at"] > 30:
            try:
                _jwks_cache.update(keys=_fetch_jwks(), fetched_at=now)
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch Supabase JWKS: {e}")
                raise JWTError("Unable to fetch signing keys") from e
            key = find(_jwks_cache["keys"])
        if not key:
            raise JWTError(f"No signing key found for kid={kid}")
        return key


def verify_supabase_token(token: str) -> AuthUser:
    """
    Verify a Supabase JWT token and extract user information.

    Args:
        token: The JWT token from the Authorization header

    Returns:
        AuthUser with user_id, email, and role

    Raises:
        HTTPException: If token is invalid or verification fails
    """
    if not api_settings.supabase_auth_enabled:
        logger.warning("Neither SUPABASE_URL nor SUPABASE_JWT_SECRET set, authentication disabled")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication not configured (SUPABASE_URL / SUPABASE_JWT_SECRET not set)",
        )

    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg")
        if alg == "HS256" and api_settings.SUPABASE_JWT_SECRET:
            key, algorithms = api_settings.SUPABASE_JWT_SECRET, ["HS256"]
        elif alg in ASYMMETRIC_ALGORITHMS and api_settings.SUPABASE_URL:
            key, algorithms = _get_signing_key(header.get("kid")), [alg]
        else:
            raise JWTError(f"Unsupported or unconfigured token algorithm: {alg}")

        payload = jwt.decode(token, key, algorithms=algorithms, audience="authenticated")

        # Extract user information
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
            )

        email = payload.get("email")
        role = payload.get("role", "authenticated")

        logger.debug(f"Authenticated user: {user_id} (email: {email})")
        return AuthUser(user_id=user_id, email=email, role=role)

    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from e


def require_user(request: Request) -> None:
    """
    Router-level dependency: require a signed-in user on protected endpoints.

    AuthMiddleware has already verified any Bearer token and set request.state.user_id.
    When Supabase auth is not configured (local dev) this check is skipped.
    """
    if not api_settings.supabase_auth_enabled or getattr(request.state, "user_id", None):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Please sign in to continue.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    authorization: Annotated[str | None, Header(description="Bearer token")] = None,
) -> AuthUser:
    """
    FastAPI dependency to get the current authenticated user.

    Extracts the Bearer token from the Authorization header and verifies it.

    Args:
        authorization: The Authorization header value (format: "Bearer <token>")

    Returns:
        AuthUser with user information

    Raises:
        HTTPException: If no token provided or token is invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format (expected 'Bearer <token>')",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:]  # Remove "Bearer " prefix
    return verify_supabase_token(token)


def optional_get_current_user(
    authorization: Annotated[str | None, Header(description="Bearer token")] = None,
) -> AuthUser | None:
    """
    Optional authentication dependency that returns None if no token provided.

    Useful for endpoints that work with or without authentication.

    Args:
        authorization: The Authorization header value

    Returns:
        AuthUser if token is valid, None if no token provided
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        return None

    token = authorization[7:]
    try:
        return verify_supabase_token(token)
    except HTTPException:
        # Return None on any auth error instead of raising
        return None
