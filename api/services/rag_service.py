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
from observability import QueryTracker

if TYPE_CHECKING:
    from generation import AnswerGenerator


class RAGService:
    """
    Service layer wrapping the AnswerGenerator.

    Provides a clean interface between API routes and the RAG pipeline.
    Future enhancements: caching, logging, multi-turn context.
    """

    def __init__(self, generator: "AnswerGenerator", tracker: "QueryTracker | None" = None):
        """
        Initialize the RAG service.

        Args:
            generator: Initialized AnswerGenerator instance
            tracker: Latency/cost/reliability tracker (defaults to one configured from settings)
        """
        self.generator = generator
        self.tracker = tracker or QueryTracker.from_settings()

    def answer(self, request: QuestionRequest) -> QuestionResponse:
        """
        Answer a question using the RAG pipeline.

        Args:
            request: Validated question request

        Returns:
            QuestionResponse with answer, sources, and metadata
        """
        result, _ = self.tracker.run(
            self.generator,
            request.question,
            doc_id=request.doc_id,
            context_k=request.context_k,
        )
        return QuestionResponse.from_generation_result(result)

    @property
    def model_name(self) -> str:
        """Get the name of the loaded LLM model."""
        return self.generator.llm_client.model
