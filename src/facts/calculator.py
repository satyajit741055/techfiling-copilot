"""Deterministic calculations over reported figures.

Expressions use the values as printed in the cited table rows (e.g. ``245122``, not
``245122000000``), so the answer validator can find every input in a cited source and
re-evaluate the expression.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Operation = Literal["change", "growth_pct", "cagr_pct", "margin_pct", "ratio", "sum"]


@dataclass(frozen=True)
class CalcInput:
    label: str  # "AAPL FY2024 Total revenue"
    value: float  # as printed in the table row
    chunk_id: str
    signed: float | None = None  # value with its sign, when it differs from the printed one

    @property
    def number(self) -> float:
        return self.value if self.signed is None else self.signed


@dataclass(frozen=True)
class CalcResult:
    op: Operation
    description: str
    expression: str
    result: float
    inputs: tuple[CalcInput, ...]


def _literal(value: float) -> str:
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def change(old: CalcInput, new: CalcInput, description: str) -> CalcResult:
    expression = f"{_literal(new.number)} - {_literal(old.number)}"
    return CalcResult("change", description, expression, new.number - old.number, (new, old))


def growth_pct(old: CalcInput, new: CalcInput, description: str) -> CalcResult | None:
    if old.number <= 0:
        return None
    expression = f"({_literal(new.number)} - {_literal(old.number)}) / {_literal(old.number)} * 100"
    result = (new.number - old.number) / old.number * 100
    return CalcResult("growth_pct", description, expression, result, (new, old))


def cagr_pct(first: CalcInput, last: CalcInput, years: int, description: str) -> CalcResult | None:
    if years < 1 or first.number <= 0 or last.number <= 0:
        return None
    expression = (
        f"(({_literal(last.number)} / {_literal(first.number)}) ** (1 / {years}) - 1) * 100"
    )
    result = ((last.number / first.number) ** (1 / years) - 1) * 100
    return CalcResult("cagr_pct", description, expression, result, (last, first))


def margin_pct(part: CalcInput, whole: CalcInput, description: str) -> CalcResult | None:
    if whole.number == 0:
        return None
    expression = f"{_literal(part.number)} / {_literal(whole.number)} * 100"
    return CalcResult(
        "margin_pct", description, expression, part.number / whole.number * 100, (part, whole)
    )


def ratio(numerator: CalcInput, denominator: CalcInput, description: str) -> CalcResult | None:
    if denominator.number == 0:
        return None
    expression = f"{_literal(numerator.number)} / {_literal(denominator.number)}"
    return CalcResult(
        "ratio",
        description,
        expression,
        numerator.number / denominator.number,
        (numerator, denominator),
    )


def total(items: Sequence[CalcInput], description: str) -> CalcResult | None:
    if len(items) < 2:
        return None
    expression = " + ".join(_literal(item.number) for item in items)
    return CalcResult(
        "sum", description, expression, sum(item.number for item in items), tuple(items)
    )
