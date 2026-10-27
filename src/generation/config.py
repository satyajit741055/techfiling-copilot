"""
Generation Configuration.

Separates fixed retrieval parameters (from evaluation) from variable generation parameters.
Now loads defaults from centralized settings (config.settings).

Usage:
    from generation.config import GenerationConfig

    # Uses defaults from .env / centralized settings
    config = GenerationConfig()

    # Override specific values
    config = GenerationConfig(context_k=10, temperature=0.1)

    # Create from settings explicitly
    config = GenerationConfig.from_settings()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings


def _get_settings():
    """Lazy import to avoid circular imports."""
    from config.settings import settings

    return settings


@dataclass
class GenerationConfig:
    """
    Configuration for the generation pipeline.

    All defaults are loaded from centralized settings (config.settings).
    You can override any value by passing it to the constructor.

    Attributes:
        run_directory: Path to the chunking run directory
        alpha: Dense weight in hybrid retrieval (fixed from evaluation)
        retrieve_k: Candidates for reranking (fixed from evaluation)
        reranker_model: Cross-encoder model name (fixed from evaluation)
        context_k: Number of chunks passed to LLM (variable for experiments)
        provider: LLM provider name
        model: LLM model name
        temperature: Generation temperature (0 for deterministic)
        max_tokens: Maximum response length
        prompt_version: Version string for tracking prompt experiments
        device: Device for embedding/reranking models
    """

    # ========== FIXED (from retrieval evaluation) ==========
    run_directory: str = field(default_factory=lambda: _get_settings().RUN_DIRECTORY)
    alpha: float = field(default_factory=lambda: _get_settings().RETRIEVAL_ALPHA)
    retrieve_k: int = field(default_factory=lambda: _get_settings().RETRIEVAL_K)
    reranker_model: str = field(default_factory=lambda: _get_settings().effective_reranker)

    # ========== VARIABLE (generation experiments) ==========
    context_k: int = field(default_factory=lambda: _get_settings().CONTEXT_K)

    # LLM settings
    provider: str = field(default_factory=lambda: _get_settings().effective_llm_provider)
    model: str = ""  # empty -> default model for the selected provider
    temperature: float = field(default_factory=lambda: _get_settings().LLM_TEMPERATURE)
    max_tokens: int = field(default_factory=lambda: _get_settings().LLM_MAX_TOKENS)

    # Tracking
    prompt_version: str = "v2_complete_precise"

    # Hardware
    device: str = field(default_factory=lambda: _get_settings().effective_device)

    def __post_init__(self) -> None:
        """Resolve the model from the provider when not given explicitly."""
        self.model = self.model or _get_settings().model_for_provider(self.provider)

    @classmethod
    def from_settings(cls, settings: "Settings | None" = None) -> "GenerationConfig":
        """
        Create a GenerationConfig from settings.

        Args:
            settings: Settings instance (uses global settings if None)

        Returns:
            GenerationConfig with all values from settings
        """
        if settings is None:
            settings = _get_settings()

        return cls(
            run_directory=settings.RUN_DIRECTORY,
            alpha=settings.RETRIEVAL_ALPHA,
            retrieve_k=settings.RETRIEVAL_K,
            reranker_model=settings.effective_reranker,
            context_k=settings.CONTEXT_K,
            provider=settings.effective_llm_provider,
            model=settings.effective_model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            device=settings.effective_device,
        )

    def resolve_run_directory(self, root: Path | str) -> str:
        """Resolve run_directory relative to project root."""
        root = Path(root)
        return str(root / self.run_directory)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "run_directory": self.run_directory,
            "alpha": self.alpha,
            "retrieve_k": self.retrieve_k,
            "reranker_model": self.reranker_model,
            "context_k": self.context_k,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_version": self.prompt_version,
            "device": self.device,
        }
