"""
Question/Answer Schemas.

Pydantic models for the /questions, /filings, and /health endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# REQUEST MODELS
# =============================================================================


class QuestionRequest(BaseModel):
    """Request body for POST /questions."""

    question: str = Field(
        ..., min_length=3, max_length=2000, description="The question to ask about the filing"
    )
    doc_id: str = Field(..., description="Filing ID to query, e.g., 'AAPL_10-K_2025'")
    context_k: int | None = Field(
        None, ge=1, le=20, description="Number of context chunks to use (default: from settings)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What was Apple's total net sales in fiscal 2025?",
                "doc_id": "AAPL_10-K_2025",
            }
        }
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class SourceInfo(BaseModel):
    """Information about a source chunk used in the answer."""

    source_id: str = Field(..., description="Source ID, e.g., 'S1', 'S2'")
    doc_id: str = Field(..., description="Document ID")
    chunk_type: str = Field(..., description="Type of chunk: 'text' or 'table'")
    score: float = Field(..., description="Reranker relevance score")
    elements: list[int] = Field(..., description="Element IDs for traceability")
    text: str = Field(..., description="The chunk text content")

    # Optional metadata
    section: str | None = Field(None, description="Filing section")
    section_title: str | None = Field(
        None, description="Filing section title, e.g., 'Legal Proceedings'"
    )
    table_id: str | None = Field(None, description="Table ID if chunk is a table")
    breadcrumb: str | None = Field(None, description="Full path: Doc > Part > Section > Title")


class AnswerInfo(BaseModel):
    """The generated answer with validation status."""

    answer: str | None = Field(..., description="The direct answer (None if abstaining)")
    explanation: str = Field(..., description="How the answer was found")
    citations: list[str] = Field(..., description="Source IDs cited, e.g., ['S1', 'S2']")
    calculation: str | None = Field(None, description="Formula if calculation required")
    insufficient_context: bool = Field(..., description="True if model abstained")
    is_valid: bool = Field(True, description="Whether answer passed validation")
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation error messages"
    )


class QuestionResponse(BaseModel):
    """Response for POST /questions."""

    # Input
    question: str = Field(..., description="The input question")
    doc_id: str | None = Field(..., description="Document filter used")

    # Generated answer
    answer: AnswerInfo = Field(..., description="The generated answer")

    # Sources used
    sources: list[SourceInfo] = Field(..., description="Source chunks used")

    # Metadata
    model: str = Field(..., description="LLM model used")
    context_k: int = Field(..., description="Number of context chunks")
    prompt_tokens: int = Field(..., description="Input token count")
    completion_tokens: int = Field(..., description="Output token count")
    latency_seconds: float = Field(..., description="Generation latency")
    timestamp: str = Field(..., description="ISO timestamp of generation")

    @classmethod
    def from_generation_result(cls, result: Any) -> "QuestionResponse":
        """
        Create a QuestionResponse from a GenerationResult.

        Args:
            result: GenerationResult from the generation pipeline

        Returns:
            QuestionResponse ready for API serialization
        """
        # Convert answer
        answer_dict = result.answer.to_dict()
        answer_info = AnswerInfo(**answer_dict)

        # Convert sources
        sources = []
        for src in result.sources:
            # Map element_orders to elements (API uses 'elements', pipeline uses 'element_orders')
            elements = src.get("elements") or src.get("element_orders") or []
            sources.append(
                SourceInfo(
                    source_id=src.get("source_id", ""),
                    doc_id=src.get("doc_id", ""),
                    chunk_type=src.get("type", "text"),  # Pipeline uses 'type', not 'chunk_type'
                    score=src.get("score") or 0.0,
                    elements=elements,
                    text=src.get("text", ""),
                    section=src.get("section") or src.get("section_title"),
                    section_title=src.get("section_title"),
                    table_id=src.get("table_id"),
                    breadcrumb=src.get("breadcrumb"),
                )
            )

        return cls(
            question=result.question,
            doc_id=result.doc_id,
            answer=answer_info,
            sources=sources,
            model=result.model,
            context_k=result.context_k,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_seconds=result.latency_seconds,
            timestamp=result.timestamp,
        )


# =============================================================================
# FILINGS MODELS
# =============================================================================


class FilingInfo(BaseModel):
    """Information about an available filing."""

    doc_id: str = Field(..., description="Unique document ID")
    company: str = Field(..., description="Company name")
    ticker: str = Field(..., description="Stock ticker symbol")
    form: str = Field(..., description="SEC form type, e.g., '10-K'")
    fiscal_year: int = Field(..., description="Fiscal year")
    period_end: str = Field(..., description="Period end date")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doc_id": "AAPL_10-K_2025",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "form": "10-K",
                "fiscal_year": 2025,
                "period_end": "September 27, 2025",
            }
        }
    )


class FilingsResponse(BaseModel):
    """Response for GET /filings."""

    filings: list[FilingInfo] = Field(..., description="List of available filings")


# =============================================================================
# HEALTH MODELS
# =============================================================================


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="Service status")
    environment: str = Field(..., description="Running environment (dev/prod)")
    model: str = Field(..., description="Loaded LLM model")
    filings_loaded: int = Field(..., description="Number of filings available")


class DependencyCheck(BaseModel):
    """Result of a single readiness check."""

    status: str = Field(..., description="'ok' or 'fail'")
    detail: str = Field(..., description="Short human-readable outcome (never contains secrets)")


class ReadinessResponse(BaseModel):
    """Response for GET /health/ready."""

    status: str = Field(..., description="'ready' when every check passes, otherwise 'not_ready'")
    checks: dict[str, DependencyCheck] = Field(..., description="Per-dependency results")
