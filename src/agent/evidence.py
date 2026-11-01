"""Turn-local evidence collection and deterministic source assignment."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from generation.context import format_source

from .schemas import Source


@dataclass(frozen=True)
class Provenance:
    task_id: str
    tool: str
    rank: int
    score: float | None = None
    sub_question: str = ""


@dataclass
class EvidenceEntry:
    chunk: dict[str, Any]
    provenance: list[Provenance] = field(default_factory=list)
    best_score: float | None = None
    source_id: str | None = None


class EvidenceRegistry:
    """Deduplicated chunks and provenance for one agent turn.

    Workers register from ``asyncio.to_thread`` threads, so mutations hold a lock.
    """

    def __init__(self) -> None:
        self._entries: dict[str, EvidenceEntry] = {}
        self._assigned = False
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._assigned = False

    def anchor_score(self, chunk_ids: Iterable[str]) -> float | None:
        """Best known score among ``chunk_ids``, inherited by neighbour/table expansions."""
        scores = [
            entry.best_score
            for chunk_id in chunk_ids
            if (entry := self._entries.get(chunk_id)) is not None and entry.best_score is not None
        ]
        return max(scores, default=None)

    def register(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        task_id: str,
        tool: str,
        scores: Iterable[float | None] | None = None,
        sub_question: str = "",
    ) -> list[str]:
        chunk_list = [dict(chunk) for chunk in chunks]
        score_list = list(scores) if scores is not None else [None] * len(chunk_list)
        if len(score_list) != len(chunk_list):
            raise ValueError("scores must have one value per chunk")
        if any(not chunk.get("chunk_id") for chunk in chunk_list):
            raise ValueError("Evidence chunks require chunk_id")
        registered: list[str] = []
        with self._lock:
            if self._assigned:
                raise RuntimeError("Evidence cannot be registered after source IDs are assigned")
            for rank, (chunk, score) in enumerate(zip(chunk_list, score_list), start=1):
                chunk_id = str(chunk["chunk_id"])
                entry = self._entries.setdefault(chunk_id, EvidenceEntry(chunk=chunk))
                entry.provenance.append(Provenance(task_id, tool, rank, score, sub_question))
                if score is not None and (entry.best_score is None or score > entry.best_score):
                    entry.best_score = score
                registered.append(chunk_id)
        return registered

    def get(self, chunk_id: str) -> EvidenceEntry | None:
        return self._entries.get(chunk_id)

    def entries(self) -> dict[str, EvidenceEntry]:
        return dict(self._entries)

    @staticmethod
    def _document_order(entry: EvidenceEntry) -> str:
        return str(entry.chunk.get("chunk_id", ""))

    @staticmethod
    def _priority(entry: EvidenceEntry) -> tuple[str, int, str, float, str]:
        provenance = min(
            entry.provenance,
            key=lambda item: (item.sub_question, item.task_id, item.rank),
        )
        chunk = entry.chunk
        return (
            str(chunk.get("ticker", "")),
            int(chunk.get("fiscal_year") or 0),
            provenance.sub_question,
            -(entry.best_score if entry.best_score is not None else float("-inf")),
            str(chunk.get("chunk_id", "")),
        )

    def _select(self, max_chunks: int) -> list[EvidenceEntry]:
        if max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")
        entries = list(self._entries.values())
        if len(entries) <= max_chunks:
            return entries
        by_cell: defaultdict[tuple[str, int], list[EvidenceEntry]] = defaultdict(list)
        for entry in entries:
            cell = (str(entry.chunk.get("ticker", "")), int(entry.chunk.get("fiscal_year") or 0))
            by_cell[cell].append(entry)
        selected: list[EvidenceEntry] = []
        for cell in sorted(by_cell):
            if len(selected) == max_chunks:
                break
            selected.append(min(by_cell[cell], key=self._priority))
        selected_ids = {entry.chunk["chunk_id"] for entry in selected}
        remaining = sorted(
            (entry for entry in entries if entry.chunk["chunk_id"] not in selected_ids),
            key=self._priority,
        )
        selected.extend(remaining[: max_chunks - len(selected)])
        return selected

    def assign_source_ids(self, max_chunks: int) -> list[Source]:
        with self._lock:
            if not self._assigned:
                selected = sorted(self._select(max_chunks), key=self._priority)
                selected_ids = {entry.chunk["chunk_id"] for entry in selected}
                for chunk_id in list(self._entries):
                    if chunk_id not in selected_ids:
                        del self._entries[chunk_id]
                for index, entry in enumerate(selected, start=1):
                    entry.source_id = f"S{index}"
                self._assigned = True
        return self.sources()

    def sources(self) -> list[Source]:
        assigned = [entry for entry in self._entries.values() if entry.source_id]
        return [
            Source(
                source_id=entry.source_id or "",
                chunk_id=str(entry.chunk["chunk_id"]),
                ticker=str(entry.chunk["ticker"]),
                fiscal_year=int(entry.chunk["fiscal_year"]),
                period_end=entry.chunk.get("period_end"),
                section=entry.chunk.get("section_title") or entry.chunk.get("section"),
                table_id=entry.chunk.get("table_id"),
                units=entry.chunk.get("units"),
                text=str(entry.chunk.get("text", "")),
            )
            for entry in sorted(assigned, key=lambda item: int((item.source_id or "S0")[1:]))
        ]

    def source_entries(self) -> dict[str, EvidenceEntry]:
        return {
            entry.source_id: entry
            for entry in self._entries.values()
            if entry.source_id is not None
        }

    def format_context(self) -> str:
        return "\n\n".join(
            format_source(int(source.source_id[1:]), self._entries[source.chunk_id].chunk)
            for source in self.sources()
        )


class TurnEvidence:
    """One ``EvidenceRegistry`` per turn, so concurrent conversations never share evidence.

    Registries are created lazily and released when a turn finalizes; the oldest are
    evicted beyond ``max_turns`` so turns that fail mid-graph cannot leak memory.
    """

    def __init__(self, max_turns: int = 128) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_turns = max_turns
        self._registries: dict[str, EvidenceRegistry] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._registries)

    def get(self, turn_id: str) -> EvidenceRegistry:
        with self._lock:
            registry = self._registries.get(turn_id)
            if registry is None:
                registry = self._registries[turn_id] = EvidenceRegistry()
                while len(self._registries) > self.max_turns:
                    self._registries.pop(next(iter(self._registries)))
            return registry

    def reset(self, turn_id: str) -> EvidenceRegistry:
        self.release(turn_id)
        return self.get(turn_id)

    def release(self, turn_id: str) -> None:
        with self._lock:
            self._registries.pop(turn_id, None)
