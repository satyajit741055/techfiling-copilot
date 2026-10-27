"""
Data classes for generation pipeline.

Defines structured types for LLM responses, generated answers, and complete results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """
    Standardized response from any LLM provider.

    Attributes:
        content: Raw text content from the model
        model: Model name used
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        latency_seconds: Time taken for generation
        raw_response: Original provider-specific response object
    """

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    raw_response: Any = None


@dataclass
class GeneratedAnswer:
    """
    Structured answer from the LLM.

    Attributes:
        answer: The direct answer to the question (None if abstaining)
        explanation: Brief explanation of how the answer was found
        citations: List of source IDs cited (e.g., ["S1", "S2"])
        calculation: Formula shown if answer requires calculation
        insufficient_context: True if model abstained due to missing info
        is_valid: Whether the answer passed validation
        validation_errors: List of validation error messages
    """

    answer: str | None
    explanation: str
    citations: list[str]
    calculation: str | None
    insufficient_context: bool

    # Validation results (filled by validate_answer())
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GenerationResult:
    """
    Complete result from the generation pipeline.

    Includes input, output, sources, and metadata for full traceability.

    Attributes:
        question: The input question
        doc_id: Document filter used (if any)
        answer: The generated answer
        sources: List of source metadata for each chunk used
        model: LLM model name
        context_k: Number of context chunks used
        prompt_tokens: Input token count
        completion_tokens: Output token count
        latency_seconds: Generation latency
        timestamp: ISO timestamp of generation
        context: Formatted context string (for debugging)
        raw_response: Raw LLM response (for debugging)
    """

    # Input
    question: str
    doc_id: str | None

    # Generated answer
    answer: GeneratedAnswer

    # Sources used
    sources: list[dict[str, Any]]

    # Metadata
    model: str
    context_k: int
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    timestamp: str

    # Raw data for debugging
    context: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "doc_id": self.doc_id,
            "answer": self.answer.to_dict(),
            "sources": self.sources,
            "model": self.model,
            "context_k": self.context_k,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_seconds": self.latency_seconds,
            "timestamp": self.timestamp,
        }
