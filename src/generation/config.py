"""
Generation Configuration.

Separates fixed retrieval parameters (from evaluation) from variable generation parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationConfig:
    """
    Configuration for the generation pipeline.
    
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
    run_directory: str = "data/chunks/2026-10-16_17-03-28"
    alpha: float = 0.85
    retrieve_k: int = 50
    reranker_model: str = "cross-encoder/ettin-reranker-150m-v1"
    
    # ========== VARIABLE (generation experiments) ==========
    context_k: int = 5
    
    # LLM settings
    provider: str = "ollama"
    model: str = "qwen2.5-7b-instruct:latest"
    temperature: float = 0.0
    max_tokens: int = 1024
    
    # Tracking
    prompt_version: str = "v1_strict_grounded"
    
    # Hardware
    device: str = "cuda"
    
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
