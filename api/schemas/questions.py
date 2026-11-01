"""
Question/Answer Schemas.

Pydantic models for the /questions, /filings, and /health endpoints.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# =============================================================================
# REQUEST MODELS
# =============================================================================

MAX_GROUPS = 6
MAX_GROUPED_CONTEXT = 30  # chunks per group x groups; default CONTEXT_K=5 x 6 groups fits


class YearRange(BaseModel):
    """Inclusive fiscal-year range; at least one bound is required."""

    gte: int | None = Field(None, ge=1990, le=2100, description="First fiscal year")
    lte: int | None = Field(None, ge=1990, le=2100, description="Last fiscal year")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_bounds(self) -> "YearRange":
        if self.gte is None and self.lte is None:
            raise ValueError("fiscal_year range needs 'gte' and/or 'lte'")
        if self.gte is not None and self.lte is not None and self.gte > self.lte:
            raise ValueError("fiscal_year range has gte > lte")
        return self


class RetrievalFilters(BaseModel):
    """Chunk metadata filters. A list matches any of its values."""

    ticker: str | list[str] | None = Field(None, description="e.g. 'AAPL' or ['AAPL', 'MSFT']")
    fiscal_year: int | list[int] | YearRange | None = Field(
        None, description="e.g. 2024, [2023, 2024] or {'gte': 2022, 'lte': 2024}"
    )
    form: str | None = Field(None, description="SEC form, e.g. '10-K'")
    section: str | list[str] | None = Field(None, description="e.g. 'Item 7'")
    type: Literal["text", "table"] | None = Field(None, description="Chunk type")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_lists(self) -> "RetrievalFilters":
        if not self.model_dump(exclude_none=True):
            raise ValueError("filters must set at least one field")
        for name in ("ticker", "fiscal_year", "section"):
            value = getattr(self, name)
            if isinstance(value, list) and not 1 <= len(value) <= 20:
                raise ValueError(f"{name} list must have 1-20 values")
        if isinstance(self.ticker, list):
            self.ticker = [t.upper() for t in self.ticker]
        elif self.ticker is not None:
            self.ticker = self.ticker.upper()
        return self

    def to_filters(self) -> dict[str, Any]:
        """Retriever ``filters`` dict (ranges become ``{"gte": .., "lte": ..}``)."""
        return self.model_dump(exclude_none=True)


class QuestionRequest(BaseModel):
    """Request body for POST /questions.

    Scope the search with ``doc_id``, ``filters`` and/or ``group_by``; at least one is
    required. With ``group_by``, ``context_k`` is the number of chunks per group.
    """

    question: str = Field(
        ..., min_length=3, max_length=2000, description="The question to ask about the filing"
    )
    doc_id: str | None = Field(None, description="Filing ID to query, e.g., 'AAPL_10-K_2025'")
    filters: RetrievalFilters | None = Field(None, description="Metadata filters")
    group_by: Literal["doc_id", "ticker", "fiscal_year"] | None = Field(
        None, description="Retrieve and rerank separately per filing, ticker or fiscal year"
    )
    groups: list[str | int] | None = Field(
        None,
        min_length=1,
        max_length=MAX_GROUPS,
        description="Group values to compare, e.g. ['AAPL', 'MSFT']. Omit to let retrieval pick.",
    )
    group_limit: int = Field(
        3, ge=1, le=MAX_GROUPS, description="Maximum groups when 'groups' is omitted"
    )
    context_k: int | None = Field(
        None, ge=1, le=20, description="Number of context chunks to use (default: from settings)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What was Apple's total net sales in fiscal 2025?",
                    "doc_id": "AAPL_10-K_2025",
                },
                {
                    "question": "How did R&D expense compare between Apple and Microsoft?",
                    "filters": {"fiscal_year": 2024},
                    "group_by": "ticker",
                    "groups": ["AAPL", "MSFT"],
                },
            ]
        }
    )

    @model_validator(mode="after")
    def _check_scope(self) -> "QuestionRequest":
        if self.doc_id is None and self.filters is None and self.group_by is None:
            raise ValueError("Provide at least one of 'doc_id', 'filters' or 'group_by'")
        if self.groups is not None:
            if self.group_by is None:
                raise ValueError("'groups' requires 'group_by'")
            filters = self.filters.to_filters() if self.filters else {}
            if self.group_by in filters or (self.group_by == "doc_id" and self.doc_id):
                raise ValueError(f"Pass {self.group_by!r} either in 'groups' or as a filter")
            if self.group_by == "fiscal_year":
                try:
                    self.groups = [int(g) for g in self.groups]
                except ValueError as error:
                    raise ValueError("fiscal_year groups must be integers") from error
            else:
                self.groups = [str(g) for g in self.groups]
                if self.group_by == "ticker":
                    self.groups = [g.upper() for g in self.groups]
        if self.group_by is not None and self.context_k is not None:
            group_count = len(set(self.groups)) if self.groups else self.group_limit
            if self.context_k * group_count > MAX_GROUPED_CONTEXT:
                raise ValueError(
                    f"context_k x groups must be at most {MAX_GROUPED_CONTEXT} chunks, "
                    f"got {self.context_k} x {group_count}"
                )
        return self

    def retrieval_options(self) -> dict[str, Any]:
        """Keyword arguments for ``AnswerGenerator.retrieve()`` beyond doc_id/context_k."""
        options: dict[str, Any] = {}
        if self.filters is not None:
            options["filters"] = self.filters.to_filters()
        if self.group_by is not None:
            options.update(group_by=self.group_by, groups=self.groups, group_limit=self.group_limit)
        return options


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
    ticker: str | None = Field(None, description="Company ticker")
    fiscal_year: int | None = Field(None, description="Filing fiscal year")
    period_end: str | None = Field(None, description="Filing period end date")
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
                    ticker=src.get("ticker"),
                    fiscal_year=src.get("fiscal_year"),
                    period_end=src.get("period_end"),
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
                "period_end": "2025-09-27",
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
