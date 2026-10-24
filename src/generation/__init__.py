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
from .context import format_context, format_source
from .generator import AnswerGenerator
from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPTS, build_user_prompt, get_system_prompt
from .schemas import GeneratedAnswer, GenerationResult, LLMResponse
from .validation import parse_llm_response, validate_answer

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
    "SYSTEM_PROMPTS",
    "build_user_prompt",
    "get_system_prompt",
    # LLM
    "LLMClient",
    # Validation
    "parse_llm_response",
    "validate_answer",
    # Generator
    "AnswerGenerator",
]
