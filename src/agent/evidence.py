"""Turn-local evidence collection and deterministic source assignment."""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from generation.context import format_source

from .schemas import Source

_CHUNK_POSITION = re.compile(r"^(?P<doc>.+)_c(?P<number>\d+)$")


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
    # Rows holding reported figures (facts store): always kept in the synthesis context.
    pinned: bool = False


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
        pinned: bool = False,
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
                entry.pinned |= pinned
                if score is not None and (entry.best_score is None or score > entry.best_score):
                    entry.best_score = score
                registered.append(chunk_id)
        return registered

    def get(self, chunk_id: str) -> EvidenceEntry | None:
        return self._entries.get(chunk_id)

    def entries(self) -> dict[str, EvidenceEntry]:
        return dict(self._entries)

    @staticmethod
    def _document_order(entry: EvidenceEntry) -> tuple[str, int, str]:
        chunk_id = str(entry.chunk.get("chunk_id", ""))
        match = _CHUNK_POSITION.match(chunk_id)
        return (match["doc"], int(match["number"]), "") if match else (chunk_id, -1, chunk_id)

    @staticmethod
    def _cell(entry: EvidenceEntry) -> tuple[str, int]:
        return str(entry.chunk.get("ticker", "")), int(entry.chunk.get("fiscal_year") or 0)

    @staticmethod
    def _score(entry: EvidenceEntry) -> float:
        return entry.best_score if entry.best_score is not None else float("-inf")

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

    def _groups(self) -> list[list[EvidenceEntry]]:
        """Entries joined by a shared ``table_id`` or adjacent chunks of the same filing."""
        parent = {chunk_id: chunk_id for chunk_id in self._entries}

        def find(chunk_id: str) -> str:
            while parent[chunk_id] != chunk_id:
                parent[chunk_id] = parent[parent[chunk_id]]
                chunk_id = parent[chunk_id]
            return chunk_id

        def union(left: str, right: str) -> None:
            parent[find(left)] = find(right)

        tables: dict[str, str] = {}
        positions: dict[tuple[str, int], str] = {}
        for chunk_id, entry in self._entries.items():
            if table_id := entry.chunk.get("table_id"):
                union(chunk_id, tables.setdefault(str(table_id), chunk_id))
            if match := _CHUNK_POSITION.match(chunk_id):
                positions[(match["doc"], int(match["number"]))] = chunk_id
        for (doc, number), chunk_id in positions.items():
            if (following := positions.get((doc, number + 1))) is not None:
                union(chunk_id, following)
        groups: defaultdict[str, list[EvidenceEntry]] = defaultdict(list)
        for chunk_id, entry in self._entries.items():
            groups[find(chunk_id)].append(entry)
        return [sorted(group, key=self._document_order) for group in groups.values()]

    def _group_priority(self, group: list[EvidenceEntry]) -> tuple[str, int, str, float, str]:
        return min(self._priority(entry) for entry in group)

    def _trim(self, group: list[EvidenceEntry], limit: int) -> list[EvidenceEntry]:
        best = sorted(group, key=lambda entry: (-self._score(entry), self._document_order(entry)))
        return sorted(best[:limit], key=self._document_order)

    def _select(self, max_chunks: int) -> list[list[EvidenceEntry]]:
        """Context budget: pinned entries (reported-figure rows) always, then the best group
        of every remaining cell, then whole groups by score.

        Returns groups in source-id order (cell, sub-question, score), each group in
        document order so neighbours and table parts stay together.
        """
        if max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")
        groups = sorted(self._groups(), key=self._group_priority)
        if sum(map(len, groups)) <= max_chunks:
            return groups
        # Pinned entries never count against the budget's trimming; their unpinned group
        # neighbours compete like any other evidence (dropped here when over budget).
        selected: dict[int, list[EvidenceEntry]] = {
            index: [entry for entry in group if entry.pinned]
            for index, group in enumerate(groups)
            if any(entry.pinned for entry in group)
        }
        used = sum(map(len, selected.values()))
        cells = {self._cell(group[0]) for group in selected.values()}
        cells_left = len({self._cell(group[0]) for group in groups} - cells)
        for index, group in enumerate(groups):
            if index in selected:
                continue
            cell = self._cell(group[0])
            if cell in cells or used >= max_chunks:
                continue
            cells.add(cell)
            cells_left -= 1
            # Reserve one slot for every cell still waiting so a large group cannot starve it.
            room = max(1, max_chunks - used - cells_left)
            selected[index] = group if len(group) <= room else self._trim(group, room)
            used += len(selected[index])
        by_score = sorted(
            (index for index in range(len(groups)) if index not in selected),
            key=lambda index: (
                -max(map(self._score, groups[index])),
                self._group_priority(groups[index]),
            ),
        )
        for index in by_score:
            if len(groups[index]) <= max_chunks - used:
                selected[index] = groups[index]
                used += len(groups[index])
        return [selected[index] for index in sorted(selected)]

    def assign_source_ids(self, max_chunks: int) -> list[Source]:
        with self._lock:
            if not self._assigned:
                selected = [entry for group in self._select(max_chunks) for entry in group]
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
