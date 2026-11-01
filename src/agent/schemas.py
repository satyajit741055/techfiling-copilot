"""Structured models shared by the research-agent graph."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Intent = Literal[
    "greeting",
    "capabilities",
    "filing_question",
    "general_knowledge",
    "out_of_scope",
]
ScopeKind = Literal[
    "specific", "corpus_wide", "context_dependent", "general_knowledge", "out_of_scope"
]
QuestionType = Literal[
    "single_fact", "qualitative", "trend", "cross_company", "matrix", "qualitative_change"
]
Route = Literal["refuse", "onboard", "clarify", "simple_rag", "research"]
CoverageStatus = Literal["pending", "found", "missing", "not_disclosed", "not_in_corpus"]


class AgentModel(BaseModel):
    """Strict base model for graph-owned data."""

    model_config = ConfigDict(extra="forbid")


class YearRange(AgentModel):
    gte: int
    lte: int

    @model_validator(mode="after")
    def ordered(self) -> "YearRange":
        if self.gte > self.lte:
            raise ValueError("year range gte must not exceed lte")
        return self


class YearSelection(AgentModel):
    explicit: list[int] = Field(default_factory=list)
    range: YearRange | None = None
    relative: Literal["latest", "last_year", "last_n"] | None = None
    n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def relative_count(self) -> "YearSelection":
        if self.relative == "last_n" and self.n is None:
            raise ValueError("n is required when relative is last_n")
        return self


class Ambiguity(AgentModel):
    is_ambiguous: bool = False
    dimension: Literal["company", "year", "metric", "other"] | None = None
    reason: str = ""
    options: list[str] = Field(default_factory=list)


class Analysis(AgentModel):
    intent: Intent
    scope: ScopeKind
    question_type: QuestionType
    companies: list[str] = Field(default_factory=list)
    years: YearSelection = Field(default_factory=YearSelection)
    topics: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list, max_length=3)
    ambiguity: Ambiguity = Field(default_factory=Ambiguity)
    uses_history: bool = False


class UIFilters(AgentModel):
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)


class ScopeCell(AgentModel):
    ticker: str
    fiscal_year: int


class CoverageCell(ScopeCell):
    status: CoverageStatus = "pending"
    detail: str | None = None
    available_years: list[int] = Field(default_factory=list)


class Clarification(AgentModel):
    dimension: Literal["company", "year", "metric", "other"]
    question: str
    options: list[str] = Field(default_factory=list)


class ResolvedScope(AgentModel):
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    cells: list[ScopeCell] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    coverage: list[CoverageCell] = Field(default_factory=list)
    not_in_corpus: list[str] = Field(default_factory=list)
    route: Route
    clarification: Clarification | None = None
    collapse_years: bool = False


class Task(AgentModel):
    task_id: str
    ticker: str
    fiscal_year: int | None = None
    fiscal_years: list[int] = Field(default_factory=list)
    sub_question: str
    query: str
    section: str | None = None


class ToolStep(AgentModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(default_factory=list)
    scope_clamped: bool = False
    error: str | None = None


class TaskResult(AgentModel):
    task_id: str
    ticker: str
    fiscal_year: int | None = None
    sub_question: str
    chunk_ids: list[str] = Field(default_factory=list)
    best_score: float | None = None
    status: Literal["found", "missing", "error"]
    trajectory: list[ToolStep] = Field(default_factory=list)
    llm_fallback: bool = False


class GroundedClaim(AgentModel):
    text: str
    citations: list[str] = Field(default_factory=list)


class GroundedAnswer(AgentModel):
    answer: str | None = None
    claims: list[GroundedClaim] = Field(default_factory=list)
    calculations: list[str] = Field(default_factory=list)
    insufficient_context: bool = False


class ValidationReport(AgentModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)


class Source(AgentModel):
    source_id: str
    chunk_id: str
    ticker: str
    fiscal_year: int
    period_end: str | None = None
    section: str | None = None
    table_id: str | None = None
    units: str | None = None
    text: str


class Usage(AgentModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0


class ChatAnswer(AgentModel):
    answer: str | None
    claims: list[GroundedClaim] = Field(default_factory=list)
    coverage: list[CoverageCell] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    calculations: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)
    route: Route
    trajectory: list[ToolStep] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class TurnSummary(AgentModel):
    question: str
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    answer: str = ""
