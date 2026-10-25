"""
API Security.

Shared-secret authentication for the data endpoints. The key is sent in the
X-API-Key header by the frontend's server-side proxy (never from the browser).
"""

import hmac

from fastapi import Header, HTTPException

from api.config import api_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency enforcing the API key.

    When API_ACCESS_KEY is unset (allowed in dev only) the check is skipped.
    """
    expected = api_settings.API_ACCESS_KEY
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
