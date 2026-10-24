"""
Application Settings with Pydantic.

Centralized configuration loaded from environment variables and .env file.
Supports dev/prod environments with sensible defaults.

Environment Variables:
    ENVIRONMENT: "dev" or "prod" (default: "dev")

    # Model execution (where embedding, reranking and generation run)
    MODEL_EXECUTION_MODE: "local" | "huggingface_api" (default: "local")
    EMBEDDING_EXECUTION_MODE / RERANKER_EXECUTION_MODE / GENERATION_EXECUTION_MODE:
        optional per-component overrides of MODEL_EXECUTION_MODE

    # LLM Configuration
    LLM_PROVIDER: "ollama" | "huggingface" | "openai" | "anthropic" | "google"
    LLM_MODEL: Model name for the selected provider
    LLM_TEMPERATURE: Generation temperature (default: 0.0)
    LLM_MAX_TOKENS: Maximum response tokens (default: 1024)

    # API Keys (required based on provider)
    HUGGINGFACE_API_KEY: Hugging Face Inference API key
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
    GOOGLE_API_KEY: Google AI API key

    # Hardware
    DEVICE: "cuda" | "cpu" | "auto" (default: "auto")

    # Embedding & Retrieval
    EMBEDDING_MODEL: Sentence transformer model (default: BAAI/bge-base-en-v1.5)
    RERANKER_MODEL: Cross-encoder model for reranking

    # Paths
    DATA_DIR: Base data directory
    MODEL_CACHE_DIR: Directory for cached models
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""
    DEV = "dev"
    PROD = "prod"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


ExecutionMode = Literal["local", "huggingface_api"]

# Project root (techfiling-copilot/)
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    The .env file is automatically loaded from the project root.
    """

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================
    ENVIRONMENT: Environment = Field(
        default=Environment.DEV,
        description="Application environment: 'dev' or 'prod'"
    )

    # =========================================================================
    # MODEL EXECUTION MODE
    # =========================================================================
    MODEL_EXECUTION_MODE: ExecutionMode = Field(
        default="local",
        description="Where models run: 'local' (CUDA/CPU + Ollama) or 'huggingface_api'"
    )
    EMBEDDING_EXECUTION_MODE: ExecutionMode | None = Field(
        default=None,
        description="Override MODEL_EXECUTION_MODE for the embedding model"
    )
    RERANKER_EXECUTION_MODE: ExecutionMode | None = Field(
        default=None,
        description="Override MODEL_EXECUTION_MODE for the reranker"
    )
    GENERATION_EXECUTION_MODE: ExecutionMode | None = Field(
        default=None,
        description="Override MODEL_EXECUTION_MODE for answer generation"
    )

    # =========================================================================
    # LLM CONFIGURATION
    # =========================================================================
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="LLM provider to use"
    )

    # Model names per provider (with sensible defaults)
    LLM_MODEL: str = Field(
        default="",
        description="LLM model name (auto-selected based on provider if empty)"
    )

    # Default models per provider
    OLLAMA_MODEL: str = Field(
        default="qwen2.5-7b-instruct:latest",
        description="Default Ollama model"
    )
    HUGGINGFACE_MODEL: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Default Hugging Face generation model"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model"
    )
    ANTHROPIC_MODEL: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Default Anthropic model"
    )
    GOOGLE_MODEL: str = Field(
        default="gemini-1.5-pro",
        description="Default Google model"
    )

    # Generation parameters
    LLM_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Generation temperature (0 for deterministic)"
    )
    LLM_MAX_TOKENS: int = Field(
        default=1024,
        ge=1,
        le=32768,
        description="Maximum response tokens"
    )

    # =========================================================================
    # API KEYS
    # =========================================================================
    HUGGINGFACE_API_KEY: str | None = Field(
        default=None,
        description="Hugging Face Inference API key"
    )
    OPENAI_API_KEY: str | None = Field(
        default=None,
        description="OpenAI API key"
    )
    ANTHROPIC_API_KEY: str | None = Field(
        default=None,
        description="Anthropic API key"
    )
    GOOGLE_API_KEY: str | None = Field(
        default=None,
        description="Google AI API key"
    )

    # =========================================================================
    # OLLAMA CONFIGURATION
    # =========================================================================
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )

    # =========================================================================
    # HUGGING FACE CONFIGURATION
    # =========================================================================
    HUGGINGFACE_PROVIDER: str = Field(
        default="auto",
        description="Inference provider for generation ('auto' picks an available provider)"
    )
    HUGGINGFACE_EMBEDDING_PROVIDER: str = Field(
        default="hf-inference",
        description="Inference provider for embeddings"
    )
    HUGGINGFACE_RERANKER_URL: str = Field(
        default="https://router.huggingface.co/hf-inference/models/{model}",
        description="Reranker endpoint; '{model}' is replaced by the reranker model. "
                    "URLs ending in '/rerank' use the TEI rerank format."
    )
    HUGGINGFACE_TIMEOUT: float = Field(
        default=120.0,
        gt=0,
        description="Timeout in seconds for Hugging Face API requests"
    )

    # =========================================================================
    # EVALUATION
    # =========================================================================
    DEEPEVAL_JUDGE_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Judge model used by DeepEval metrics"
    )

    # =========================================================================
    # HARDWARE CONFIGURATION
    # =========================================================================
    DEVICE: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Compute device: 'cuda', 'cpu', or 'auto'"
    )

    # =========================================================================
    # EMBEDDING & RETRIEVAL MODELS
    # =========================================================================
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Sentence transformer model for embeddings"
    )
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    RERANKER_MODEL_PROD: str = Field(
        default="cross-encoder/ettin-reranker-150m-v1",
        description="Higher-quality reranker for production"
    )

    # =========================================================================
    # RETRIEVAL PARAMETERS
    # =========================================================================
    RETRIEVAL_ALPHA: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Dense weight in hybrid retrieval (0=BM25 only, 1=dense only)"
    )
    RETRIEVAL_K: int = Field(
        default=50,
        ge=1,
        description="Number of candidates for reranking"
    )
    CONTEXT_K: int = Field(
        default=5,
        ge=1,
        description="Number of chunks passed to LLM"
    )

    # =========================================================================
    # PATHS
    # =========================================================================
    DATA_DIR: Path = Field(
        default=ROOT / "data",
        description="Base data directory"
    )
    MODEL_CACHE_DIR: Path = Field(
        default=ROOT / "models" / "sentence_transformers",
        description="Directory for cached sentence transformer models"
    )
    CHUNKS_DIR: Path = Field(
        default=ROOT / "data" / "chunks",
        description="Directory containing chunk data"
    )
    RUN_DIRECTORY: str = Field(
        default="data/chunks/2026-10-24_12-46-23",
        description="Active chunking run directory (relative to project root)"
    )

    # =========================================================================
    # LOGGING
    # =========================================================================
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == Environment.DEV

    @property
    def is_prod(self) -> bool:
        """Check if running in production mode."""
        return self.ENVIRONMENT == Environment.PROD

    @property
    def effective_device(self) -> str:
        """Get the effective compute device."""
        if self.DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE

    @property
    def effective_embedding_mode(self) -> str:
        """Execution mode for the embedding model."""
        return self.EMBEDDING_EXECUTION_MODE or self.MODEL_EXECUTION_MODE

    @property
    def effective_reranker_mode(self) -> str:
        """Execution mode for the reranker."""
        return self.RERANKER_EXECUTION_MODE or self.MODEL_EXECUTION_MODE

    @property
    def effective_generation_mode(self) -> str:
        """Execution mode for answer generation."""
        return self.GENERATION_EXECUTION_MODE or self.MODEL_EXECUTION_MODE

    @property
    def effective_llm_provider(self) -> str:
        """LLM provider: Hugging Face in API mode, otherwise LLM_PROVIDER."""
        if self.effective_generation_mode == "huggingface_api":
            return LLMProvider.HUGGINGFACE.value
        return self.LLM_PROVIDER.value

    def model_for_provider(self, provider: str) -> str:
        """Model name for a provider: LLM_MODEL if set, else the provider default."""
        if self.LLM_MODEL:
            return self.LLM_MODEL
        provider_defaults = {
            LLMProvider.OLLAMA: self.OLLAMA_MODEL,
            LLMProvider.HUGGINGFACE: self.HUGGINGFACE_MODEL,
            LLMProvider.OPENAI: self.OPENAI_MODEL,
            LLMProvider.ANTHROPIC: self.ANTHROPIC_MODEL,
            LLMProvider.GOOGLE: self.GOOGLE_MODEL,
        }
        return provider_defaults.get(LLMProvider(provider), self.OLLAMA_MODEL)

    @property
    def effective_model(self) -> str:
        """Get the effective LLM model name based on provider."""
        return self.model_for_provider(self.effective_llm_provider)

    @property
    def effective_reranker(self) -> str:
        """Get the effective reranker model based on environment."""
        if self.is_prod:
            return self.RERANKER_MODEL_PROD
        return self.RERANKER_MODEL

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return ROOT

    def get_run_directory(self) -> Path:
        """Get the absolute path to the active run directory."""
        return ROOT / self.RUN_DIRECTORY

    # =========================================================================
    # VALIDATORS
    # =========================================================================
    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """Validate that required API keys are present for the selected provider."""
        provider = LLMProvider(self.effective_llm_provider)
        uses_hf_api = provider == LLMProvider.HUGGINGFACE or "huggingface_api" in (
            self.effective_embedding_mode, self.effective_reranker_mode
        )

        if uses_hf_api and not self.HUGGINGFACE_API_KEY:
            raise ValueError(
                "HUGGINGFACE_API_KEY is required when any model uses the Hugging Face API"
            )
        if provider == LLMProvider.OPENAI and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        elif provider == LLMProvider.ANTHROPIC and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
        elif provider == LLMProvider.GOOGLE and not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when using Google provider")

        return self

    @field_validator(
        "EMBEDDING_EXECUTION_MODE", "RERANKER_EXECUTION_MODE", "GENERATION_EXECUTION_MODE",
        mode="before",
    )
    @classmethod
    def blank_mode_is_unset(cls, v: str | None) -> str | None:
        """Treat an empty override (e.g. ``RERANKER_EXECUTION_MODE=``) as not set."""
        return v or None

    @field_validator("DATA_DIR", "MODEL_CACHE_DIR", "CHUNKS_DIR", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            path = Path(v)
            # If relative, resolve against project root
            if not path.is_absolute():
                return ROOT / path
            return path
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Global settings instance
settings = get_settings()
