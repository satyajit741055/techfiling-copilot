"""Chunk metadata filters shared by the Qdrant store and the in-memory retrievers.

A filter maps a payload field to one of:

- a scalar: exact match, e.g. ``{"ticker": "AAPL"}``
- a list, tuple or set: match any value, e.g. ``{"ticker": ["AAPL", "MSFT"]}``
- a mapping with ``gt``/``gte``/``lt``/``lte``: numeric range, e.g.
  ``{"fiscal_year": {"gte": 2022, "lte": 2024}}``

All fields must match (AND). ``None`` values are dropped, so optional arguments can be
passed straight through.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from typing import Any

from qdrant_client import models

RANGE_OPERATORS = {"gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}
GROUP_FIELDS = ("doc_id", "ticker", "fiscal_year")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate(key: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        unknown = set(value) - set(RANGE_OPERATORS)
        if not value or unknown:
            raise ValueError(
                f"Range filter on {key!r} needs keys from {sorted(RANGE_OPERATORS)}, got {sorted(value)}"
            )
        if not all(_is_number(bound) for bound in value.values()):
            raise ValueError(f"Range filter on {key!r} needs numeric bounds, got {dict(value)}")
        return dict(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        if not value:
            raise ValueError(f"Filter on {key!r} has an empty value list")
        return list(value)
    return value


def merge_filters(filters: Mapping[str, Any] | None = None, **fields: Any) -> dict[str, Any]:
    """Validated filters plus explicit keyword fields (which take precedence); ``None`` dropped."""
    merged = {
        **(filters or {}),
        **{key: value for key, value in fields.items() if value is not None},
    }
    return {key: _validate(key, value) for key, value in merged.items() if value is not None}


def matches(chunk: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """Whether ``chunk`` satisfies already-merged ``filters``."""
    for key, expected in filters.items():
        actual = chunk.get(key)
        if isinstance(expected, dict):
            if not _is_number(actual) or not all(
                RANGE_OPERATORS[op](actual, bound) for op, bound in expected.items()
            ):
                return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def matching_indices(chunks: Iterable[Mapping[str, Any]], filters: Mapping[str, Any]) -> list[int]:
    return [index for index, chunk in enumerate(chunks) if matches(chunk, filters)]


def to_qdrant_filter(filters: Mapping[str, Any] | None) -> models.Filter | None:
    """Qdrant ``Filter`` for already-merged ``filters`` (``None`` when there are none)."""
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if isinstance(value, dict):
            conditions.append(models.FieldCondition(key=key, range=models.Range(**value)))
        elif isinstance(value, list):
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
        else:
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=conditions)


def check_sequence(name: str, values: Any) -> None:
    """Reject a bare string (it would be iterated character by character) or mapping."""
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a list of values, not {type(values).__name__}")


def check_group_by(group_by: str, filters: Mapping[str, Any], groups: Any) -> None:
    if group_by not in GROUP_FIELDS:
        raise ValueError(f"group_by must be one of {GROUP_FIELDS}, got {group_by!r}")
    if groups is not None:
        check_sequence("groups", groups)
        if not groups:
            raise ValueError("groups must not be empty")
        if group_by in filters:
            raise ValueError(f"Pass {group_by!r} either as groups or as a filter, not both")
