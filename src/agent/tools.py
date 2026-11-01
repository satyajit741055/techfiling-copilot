"""Guarded retrieval tools for research workers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import Field, ValidationInfo, field_validator, model_validator

from retriever.qdrant_retriever import QdrantRetrieverWithReranker
from vector_store import QdrantChunkStore, neighbor_ids

from .catalog import FilingCatalog
from .evidence import EvidenceRegistry
from .schemas import AgentModel, Task, ToolStep, YearRange


class SearchFilingsArgs(AgentModel):
    query: str = Field(min_length=1)
    tickers: list[str] = Field(default_factory=list, max_length=10)
    fiscal_years: list[int] = Field(default_factory=list, max_length=10)
    year_range: YearRange | None = None
    section: str | None = None
    type: Literal["text", "table"] | None = None
    k: int = Field(default=5, ge=1, le=10)
    group_by: Literal["doc_id", "ticker", "fiscal_year"] | None = None

    @field_validator("type", "group_by", mode="before")
    @classmethod
    def unknown_option_is_unset(cls, value: Any, info: ValidationInfo) -> Any:
        """Small models put things like "10-K" here; treat unknown options as unset."""
        allowed = {"type": {"text", "table"}, "group_by": {"doc_id", "ticker", "fiscal_year"}}
        return value if value in allowed[str(info.field_name)] else None

    @model_validator(mode="after")
    def exclusive_year_filters(self) -> "SearchFilingsArgs":
        if self.fiscal_years and self.year_range:
            raise ValueError("Pass fiscal_years or year_range, not both")
        return self


class ReadChunksArgs(AgentModel):
    chunk_ids: list[str] = Field(min_length=1, max_length=10)


class ReadNeighborsArgs(AgentModel):
    chunk_id: str
    before: int = Field(default=1, ge=0, le=2)
    after: int = Field(default=1, ge=0, le=2)


class ReadTableArgs(AgentModel):
    table_id: str | None = None
    chunk_id: str | None = None

    @model_validator(mode="after")
    def some_identifier(self) -> "ReadTableArgs":
        # Both is fine (table_id wins); the in-scope filter still applies to the result.
        if not self.table_id and not self.chunk_id:
            raise ValueError("Pass table_id or chunk_id")
        return self


class DoneArgs(AgentModel):
    reason: str = "Evidence is sufficient"


class ToolOutput(AgentModel):
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    step: ToolStep


class AgentTools:
    """Plain tool implementations with task-scope enforcement and evidence registration."""

    def __init__(
        self,
        catalog: FilingCatalog,
        retriever: QdrantRetrieverWithReranker,
        store: QdrantChunkStore,
        evidence: EvidenceRegistry,
    ) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.store = store
        self.evidence = evidence

    def bind(self, evidence: EvidenceRegistry) -> AgentTools:
        """Same backends, registering into another turn's evidence registry."""
        return AgentTools(self.catalog, self.retriever, self.store, evidence)

    def list_filings(self) -> list[dict[str, Any]]:
        return [filing.model_dump() for filing in self.catalog.filings()]

    @staticmethod
    def _scope(task: Task) -> tuple[list[str], list[int]]:
        years = task.fiscal_years or ([task.fiscal_year] if task.fiscal_year is not None else [])
        return [task.ticker], years

    @staticmethod
    def _clamp(values: Sequence[Any], allowed: Sequence[Any]) -> tuple[list[Any], bool]:
        requested = list(dict.fromkeys(values))
        permitted = list(dict.fromkeys(allowed))
        if not requested:
            return permitted, False
        clamped = [value for value in requested if value in permitted]
        return clamped, clamped != requested

    def _register(
        self,
        task: Task,
        tool: str,
        chunks: list[dict[str, Any]],
        scores: list[float | None] | None = None,
        anchors: Sequence[str] = (),
    ) -> list[str]:
        if scores is None and anchors:
            anchor = self.evidence.anchor_score(anchors)
            scores = [anchor] * len(chunks)
        return self.evidence.register(
            chunks,
            task_id=task.task_id,
            tool=tool,
            scores=scores,
            sub_question=task.sub_question,
        )

    def search_filings(self, task: Task, **kwargs: Any) -> ToolOutput:
        args = SearchFilingsArgs.model_validate(kwargs)
        allowed_tickers, allowed_years = self._scope(task)
        # Company names ("Apple") resolve to tickers before clamping to the task.
        requested_tickers = [self.catalog.resolve_company(name) or name for name in args.tickers]
        tickers, ticker_clamped = self._clamp(requested_tickers, allowed_tickers)
        requested_years = args.fiscal_years
        if args.year_range:
            requested_years = list(range(args.year_range.gte, args.year_range.lte + 1))
        years, year_clamped = self._clamp(requested_years, allowed_years)
        filters: dict[str, Any] = {"ticker": tickers, "fiscal_year": years}
        if args.section:
            filters["section"] = args.section
        if args.type:
            filters["type"] = args.type
        clamped = ticker_clamped or year_clamped
        if not tickers or not years:
            step = ToolStep(
                tool="search_filings",
                args=args.model_dump(mode="json"),
                scope_clamped=clamped,
                error="Requested scope is outside the worker task",
            )
            return ToolOutput(step=step)
        if args.group_by:
            groups = years if args.group_by == "fiscal_year" else None
            group_filters = dict(filters)
            if groups is not None:
                group_filters.pop("fiscal_year")
            grouped = self.retriever.search_grouped(
                args.query,
                args.group_by,
                groups=groups,
                k=args.k,
                filters=group_filters,
            )
            results = [result for hits in grouped.values() for result in hits]
        else:
            results = self.retriever.search(args.query, k=args.k, filters=filters)
        chunks = [dict(result.chunk) for result in results]
        scores = [float(result.score) for result in results]
        chunk_ids = self._register(task, "search_filings", chunks, scores)
        return ToolOutput(
            chunks=chunks,
            step=ToolStep(
                tool="search_filings",
                args=args.model_dump(mode="json"),
                chunk_ids=chunk_ids,
                scope_clamped=clamped,
            ),
        )

    def _in_scope(
        self, task: Task, chunks: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool]:
        tickers, years = self._scope(task)
        accepted = [
            chunk
            for chunk in chunks
            if chunk.get("ticker") in tickers and chunk.get("fiscal_year") in years
        ]
        return accepted, len(accepted) != len(chunks)

    def read_chunks(self, task: Task, **kwargs: Any) -> ToolOutput:
        args = ReadChunksArgs.model_validate(kwargs)
        chunks, clamped = self._in_scope(task, self.store.get_chunks(args.chunk_ids))
        chunk_ids = self._register(task, "read_chunks", chunks)
        return ToolOutput(
            chunks=chunks,
            step=ToolStep(
                tool="read_chunks",
                args=args.model_dump(mode="json"),
                chunk_ids=chunk_ids,
                scope_clamped=clamped,
            ),
        )

    def read_neighbors(self, task: Task, **kwargs: Any) -> ToolOutput:
        args = ReadNeighborsArgs.model_validate(kwargs)
        requested = neighbor_ids(args.chunk_id, args.before, args.after)
        chunks, clamped = self._in_scope(task, self.store.get_chunks(requested))
        chunk_ids = self._register(task, "read_neighbors", chunks, anchors=[args.chunk_id])
        return ToolOutput(
            chunks=chunks,
            step=ToolStep(
                tool="read_neighbors",
                args=args.model_dump(mode="json"),
                chunk_ids=chunk_ids,
                scope_clamped=clamped,
            ),
        )

    def read_table(self, task: Task, **kwargs: Any) -> ToolOutput:
        args = ReadTableArgs.model_validate(kwargs)
        table_id = args.table_id
        if not table_id and args.chunk_id:
            chunks = self.store.get_chunks([args.chunk_id])
            table_id = chunks[0].get("table_id") if chunks else None
        if not table_id:
            return ToolOutput(
                step=ToolStep(
                    tool="read_table",
                    args=args.model_dump(mode="json"),
                    error="Chunk does not belong to a table",
                )
            )
        chunks, clamped = self._in_scope(task, self.store.find_chunks({"table_id": table_id}))
        anchors = [chunk["chunk_id"] for chunk in chunks] + [args.chunk_id or ""]
        chunk_ids = self._register(task, "read_table", chunks, anchors=anchors)
        return ToolOutput(
            chunks=chunks,
            step=ToolStep(
                tool="read_table",
                args=args.model_dump(mode="json"),
                chunk_ids=chunk_ids,
                scope_clamped=clamped,
            ),
        )

    def langchain_tools(self, task: Task) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                name="search_filings",
                description="Search only within this worker's assigned filing scope.",
                func=lambda **kwargs: self.search_filings(task, **kwargs).model_dump(mode="json"),
                args_schema=SearchFilingsArgs,
            ),
            StructuredTool.from_function(
                name="read_neighbors",
                description="Read nearby chunks from the same filing.",
                func=lambda **kwargs: self.read_neighbors(task, **kwargs).model_dump(mode="json"),
                args_schema=ReadNeighborsArgs,
            ),
            StructuredTool.from_function(
                name="read_table",
                description="Read all chunks belonging to one table.",
                func=lambda **kwargs: self.read_table(task, **kwargs).model_dump(mode="json"),
                args_schema=ReadTableArgs,
            ),
            StructuredTool.from_function(
                name="done",
                description="Stop when the available evidence is sufficient.",
                func=lambda **kwargs: DoneArgs.model_validate(kwargs).model_dump(mode="json"),
                args_schema=DoneArgs,
            ),
        ]
