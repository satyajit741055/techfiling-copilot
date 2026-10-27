"""
API Configuration.

Handles API-specific settings (host, port, CORS, auth) separate from RAG pipeline settings.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Project root (techfiling-copilot/)
ROOT = Path(__file__).resolve().parents[1]

DEV_CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
MIN_API_KEY_LENGTH = 24


class APISettings(BaseSettings):
    """
    API-specific settings loaded from environment variables.

    These are separate from the RAG pipeline settings in src/config/settings.py.
    """

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
    )

    ENVIRONMENT: Literal["dev", "prod"] = Field(
        default="dev",
        description="Running environment: 'dev' or 'prod' (same variable as the pipeline settings)",
    )

    # Server settings
    API_HOST: str = Field(default="0.0.0.0", description="Host to bind the API server")
    API_PORT: int = Field(default=8000, description="Port to bind the API server")

    # CORS settings: comma-separated list or JSON array
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEV_CORS_ORIGINS),
        description="Allowed CORS origins. Must be set explicitly in prod; '*' is rejected in prod.",
    )

    # Authentication
    API_ACCESS_KEY: str | None = Field(
        default=None,
        description="Shared secret required in the X-API-Key header on /questions and /filings. "
        "Required when ENVIRONMENT=prod; auth is disabled in dev when unset.",
    )
    SUPABASE_JWT_SECRET: str | None = Field(
        default=None,
        description="Legacy HS256 JWT secret. Only needed if your Supabase project still signs tokens with "
        "the shared secret; projects using asymmetric keys (ES256/RS256) only need SUPABASE_URL.",
    )
    SUPABASE_URL: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"),
        description="Supabase project URL. Used to fetch the public signing keys (JWKS) that verify "
        "ES256/RS256 tokens. Falls back to NEXT_PUBLIC_SUPABASE_URL.",
    )

    # Rate limiting
    RATE_LIMIT_REDIS_URL: str | None = Field(
        default=None,
        description="Redis URL for distributed rate limiting (e.g., redis://localhost:6379/0). "
        "If not set, uses in-memory storage (development only).",
    )
    RATE_LIMIT_DEFAULT_PER_MINUTE: int = Field(
        default=60, description="Default rate limit per IP per minute for all endpoints"
    )
    RATE_LIMIT_QUESTIONS_PER_MINUTE: int = Field(
        default=2,
        description="Rate limit per IP per minute for /questions endpoint (expensive LLM calls)",
    )
    RATE_LIMIT_FILINGS_PER_MINUTE: int = Field(
        default=30, description="Rate limit per IP per minute for /filings endpoint"
    )
    RATE_LIMIT_GLOBAL_PER_MINUTE: int = Field(
        default=100, description="Global rate limit across all IPs per minute (system protection)"
    )

    # Paths
    PARSED_DATA_DIR: Path = Field(
        default=ROOT / "data" / "parsed",
        description="Directory containing parsed document metadata",
    )

    @field_validator("ENVIRONMENT", mode="before")
    @classmethod
    def _normalize_environment(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        """Accept 'a,b' or '["a","b"]'; trim whitespace and trailing slashes."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                value = json.loads(text)
            else:
                value = text.split(",")
        if isinstance(value, list):
            return [str(item).strip().rstrip("/") for item in value if str(item).strip()]
        return value

    @field_validator(
        "API_ACCESS_KEY",
        "SUPABASE_JWT_SECRET",
        "SUPABASE_URL",
        "RATE_LIMIT_REDIS_URL",
        mode="before",
    )
    @classmethod
    def _empty_key_is_unset(cls, value: object) -> object:
        """Treat empty API_ACCESS_KEY / SUPABASE_JWT_SECRET / RATE_LIMIT_REDIS_URL as not set."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def _validate_production(self) -> "APISettings":
        if self.ENVIRONMENT != "prod":
            return self
        problems = []
        if not self.API_ACCESS_KEY:
            problems.append("API_ACCESS_KEY is required when ENVIRONMENT=prod")
        elif len(self.API_ACCESS_KEY) < MIN_API_KEY_LENGTH:
            problems.append(
                f"API_ACCESS_KEY must be at least {MIN_API_KEY_LENGTH} characters in prod"
            )
        if (
            not self.CORS_ORIGINS
            or "*" in self.CORS_ORIGINS
            or self.CORS_ORIGINS == DEV_CORS_ORIGINS
        ):
            problems.append(
                "CORS_ORIGINS must be set explicitly to your frontend origin(s) in prod "
                "(no '*', no localhost defaults)"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def supabase_auth_enabled(self) -> bool:
        """True when a JWT secret or project URL is configured."""
        return bool(self.SUPABASE_JWT_SECRET or self.SUPABASE_URL)

    @property
    def is_prod(self) -> bool:
        """True when running with ENVIRONMENT=prod."""
        return self.ENVIRONMENT == "prod"

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return ROOT


# Global settings instance
api_settings = APISettings()
