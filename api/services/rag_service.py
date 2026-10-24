"""
RAG Service.

Thin wrapper around AnswerGenerator that bridges API schemas to pipeline objects.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from api.schemas.questions import QuestionRequest, QuestionResponse

if TYPE_CHECKING:
    from generation import AnswerGenerator


class RAGService:
    """
    Service layer wrapping the AnswerGenerator.

    Provides a clean interface between API routes and the RAG pipeline.
    Future enhancements: caching, logging, multi-turn context.
    """

    def __init__(self, generator: "AnswerGenerator"):
        """
        Initialize the RAG service.

        Args:
            generator: Initialized AnswerGenerator instance
        """
        self.generator = generator

    def answer(self, request: QuestionRequest) -> QuestionResponse:
        """
        Answer a question using the RAG pipeline.

        Args:
            request: Validated question request

        Returns:
            QuestionResponse with answer, sources, and metadata
        """
        # Build kwargs
        kwargs = {
            "question": request.question,
            "doc_id": request.doc_id,
        }

        # Only pass context_k if explicitly provided
        if request.context_k is not None:
            kwargs["context_k"] = request.context_k

        # Call the pipeline
        result = self.generator.answer(**kwargs)

        # Convert to API response
        return QuestionResponse.from_generation_result(result)

    @property
    def model_name(self) -> str:
        """Get the name of the loaded LLM model."""
        return self.generator.llm_client.model
