"""
Authentication with Supabase JWT.

Validates Supabase JWT tokens from the Authorization header and extracts user information.
This enables per-user rate limiting and user tracking.
"""

import logging
from typing import Annotated

from fastapi import Header, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel

from api.config import api_settings

logger = logging.getLogger(__name__)


class AuthUser(BaseModel):
    """Authenticated user information extracted from JWT."""

    user_id: str
    email: str | None = None
    role: str = "authenticated"


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
    if not api_settings.SUPABASE_JWT_SECRET:
        logger.warning("SUPABASE_JWT_SECRET not set, authentication disabled")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication not configured (SUPABASE_JWT_SECRET not set)",
        )

    try:
        # Decode and verify the JWT
        payload = jwt.decode(
            token,
            api_settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )

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
