"""
API Configuration.

Handles API-specific settings (host, port, CORS) separate from RAG pipeline settings.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root (techfiling-copilot/)
ROOT = Path(__file__).resolve().parents[1]


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
    )
    
    # Server settings
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    API_PORT: int = Field(
        default=8000,
        description="Port to bind the API server"
    )
    
    # CORS settings - use ["*"] to allow all origins
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins for frontend"
    )
    
    # Paths
    PARSED_DATA_DIR: Path = Field(
        default=ROOT / "data" / "parsed",
        description="Directory containing parsed document metadata"
    )
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return ROOT


# Global settings instance
api_settings = APISettings()
