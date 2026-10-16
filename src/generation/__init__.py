"""
Answer Generation Module for SEC 10-K RAG System.

This module provides grounded answer generation using retrieved filing evidence.

Components:
    - config: Generation configuration
    - context: Context formatting with source IDs
    - prompts: System and user prompt templates
    - schemas: Data classes for answers and results
    - llm_client: Multi-provider LLM abstraction
    - validation: Citation and answer validation
    - generator: Main AnswerGenerator pipeline
"""

from .config import GenerationConfig
from .schemas import LLMResponse, GeneratedAnswer, GenerationResult
from .context import format_source, format_context
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .llm_client import LLMClient
from .validation import parse_llm_response, validate_answer
from .generator import AnswerGenerator

__all__ = [
    # Config
    "GenerationConfig",
    # Schemas
    "LLMResponse",
    "GeneratedAnswer",
    "GenerationResult",
    # Context
    "format_source",
    "format_context",
    # Prompts
    "SYSTEM_PROMPT",
    "build_user_prompt",
    # LLM
    "LLMClient",
    # Validation
    "parse_llm_response",
    "validate_answer",
    # Generator
    "AnswerGenerator",
]
