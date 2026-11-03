"""Fail-closed validation for grounded research answers."""

from __future__ import annotations

import ast
import math
import operator
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from facts.records import FactRecord
from facts.store import FactsStore
from generation.validation import normalize_to_millions

from .catalog import FilingCatalog
from .evidence import EvidenceEntry
from .schemas import (
    MAX_EXCERPT_CHARS,
    Calculation,
    CoverageCell,
    GroundedAnswer,
    GroundedClaim,
    ResolvedScope,
    ValidationReport,
)

_CITATION = re.compile(r"\[(S\d+)]")
_EXCERPT_BREAK = re.compile(r"\s*\n\s*")
_CITATION_GROUP = re.compile(r"\s*\[S\d+(?:\s*,\s*S\d+)*]")
_CITATION_RUN = re.compile(r"(?:\s*\[S\d+(?:\s*,\s*S\d+)*])+")
_SOURCE_ID = re.compile(r"S\d+")
_NUMBER = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
_FORM_SUFFIX = re.compile(r"-[A-Za-z]")
_UNIT_WORD = re.compile(r"\s*(?:thousand|million|billion|trillion)\b", re.IGNORECASE)
_UNIT_LABEL = re.compile(r"\b(thousand|million|billion)s?\b", re.IGNORECASE)
_TRANSLATION = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
_EXPRESSION_TRANSLATION = str.maketrans(
    {"×": "*", "÷": "/", "−": "-", "$": None, ",": None, "%": None}
)
_PRIOR_YEAR = re.compile(
    r"\b(?:prior|previous|preceding)\s+(?:fiscal\s+)?year\b|\byear\s+(?:earlier|before)\b",
    re.IGNORECASE,
)
_FINAL_STATUSES = {"found", "not_disclosed", "not_in_corpus"}
_SCALE_CONSTANTS = (100.0, 1_000.0, 1_000_000.0, 1_000_000_000.0)
_MAX_EXPRESSION_CHARS = 200
_MAX_ROUNDING = 0.05
_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def normalize_excerpt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(_TRANSLATION)
    normalized = re.sub(r"\s*\|(?:\s*\|)*\s*", " | ", normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True)
class _Number:
    raw: str
    value: Decimal
    decimals: int
    financial: bool
    unit: str | None = None

    def millions(self, default_unit: str | None = None) -> tuple[float, float] | None:
        """Value and rounding tolerance in millions, when the scale is known."""
        unit = self.unit or default_unit
        if unit is None or "%" in self.raw:
            return None
        tolerance = 0.5 * 10.0**-self.decimals
        return normalize_to_millions(float(self.value), unit), normalize_to_millions(
            tolerance, unit
        )


def _key(value: Decimal | float) -> str:
    decimal = value if isinstance(value, Decimal) else Decimal(repr(float(value)))
    return format(abs(decimal).normalize(), "f")


def _parse_numbers(text: str) -> list[_Number]:
    """Numbers in ``text``, excluding years, ``S#`` ids and form names such as ``10-K``."""
    found: list[_Number] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        if _FORM_SUFFIX.match(text, match.end()):
            continue
        digits = raw.strip("$%").rstrip(",")
        has_symbol = "$" in raw or "%" in raw
        if not has_symbol and digits.isdigit() and len(digits) == 4 and 1900 <= int(digits) <= 2099:
            continue
        try:
            value = Decimal(digits.replace(",", ""))
        except InvalidOperation:
            continue
        word = _UNIT_WORD.match(text, match.end())
        financial = has_symbol or "," in digits or "." in digits or word is not None
        unit = _unit(word.group(0)) if word else None
        found.append(_Number(raw, value, len(digits.partition(".")[2]), financial, unit))
    return found


def _number_keys(text: str) -> set[str]:
    return {_key(number.value) for number in _parse_numbers(text)}


def _unit(text: str | None) -> str | None:
    """``thousand``/``million``/``billion`` named in a table units label or after a number."""
    match = _UNIT_LABEL.search(text or "")
    return match.group(1).lower() if match else None


def _scaled_values(excerpt: str, units: Sequence[str | None]) -> list[float]:
    """Excerpt numbers in millions, scaled by their own unit word or a table units label."""
    excerpt_numbers = _parse_numbers(excerpt)
    return [
        scaled[0]
        for default in {None, *map(_unit, units)}
        for number in excerpt_numbers
        if (scaled := number.millions(default)) is not None
    ]


def _scaled_match(number: _Number, excerpt_values: Sequence[float]) -> bool:
    """Whether a claim number with a scale word equals an excerpt number in millions.

    Lets "$94.83 billion" match "94,830" from a table labelled "(in millions)"; the
    tolerance is half a unit of the claim's last digit, so rounding is allowed, but
    never more than 5% of the source value ("$1 billion" does not match 600 million).
    """
    claimed = number.millions() if number.unit else None
    return claimed is not None and any(
        abs(claimed[0] - value) <= min(claimed[1], _MAX_ROUNDING * abs(value)) + 1e-9
        for value in excerpt_values
    )


def _matches_result(number: _Number, results: Sequence[float]) -> bool:
    value = float(number.value)
    return any(
        math.isclose(value, round(abs(result), number.decimals), abs_tol=1e-9)
        or math.isclose(value, abs(result), rel_tol=1e-9)
        for result in results
    )


def _is_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _is_one(node: ast.AST) -> bool:
    return _is_constant(node) and float(node.value) == 1.0  # type: ignore[attr-defined]


def _evaluate(expression: str) -> tuple[float | None, list[float]]:
    """Evaluate ``+ - * /`` arithmetic over numeric literals; ``None`` if anything else.

    Growth rates (CAGR) also allow ``(a / b) ** (1 / n) - 1``: a power of a positive base
    whose exponent is a constant fraction of at most 1 in size. Literals inside the
    exponent and the ``1`` subtracted from the power are structural, not cited values.
    """
    literals: list[float] = []

    def constant(node: ast.AST) -> float:
        """Exponents: constants and ``/`` between constants only; not recorded as literals."""
        if _is_constant(node):
            return float(node.value)  # type: ignore[attr-defined]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return constant(node.left) / constant(node.right)
        raise ValueError("unsupported exponent")

    def visit(node: ast.AST) -> float:
        if _is_constant(node):
            value = float(node.value)  # type: ignore[attr-defined]
            literals.append(abs(value))
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = visit(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = visit(node.left), constant(node.right)
            if base <= 0 or not 0 < abs(exponent) <= 1:
                raise ValueError("unsupported power")
            return float(base**exponent)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.Add, ast.Sub))
            and isinstance(node.left, ast.BinOp)
            and isinstance(node.left.op, ast.Pow)
            and _is_one(node.right)
        ):
            return _OPERATORS[type(node.op)](visit(node.left), 1.0)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("unsupported expression")

    cleaned = expression.translate(_EXPRESSION_TRANSLATION)
    if len(cleaned) > _MAX_EXPRESSION_CHARS:
        return None, literals
    try:
        return visit(ast.parse(cleaned, mode="eval").body), literals
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None, literals


def _calculation_errors(
    calculation: Calculation, source_entries: dict[str, EvidenceEntry]
) -> list[str]:
    errors: list[str] = []
    if not calculation.inputs:
        errors.append(f"Calculation has no cited inputs: {calculation.description}")
    for item in calculation.inputs:
        entry = source_entries.get(item.citation_id)
        if entry is None:
            errors.append(f"Calculation input cites unknown source: {item.citation_id}")
        elif _key(item.value) not in _number_keys(str(entry.chunk.get("text", ""))):
            errors.append(f"Calculation input {_key(item.value)} is absent from {item.citation_id}")
    evaluated, literals = _evaluate(calculation.expression)
    if evaluated is None:
        errors.append(f"Calculation expression is not simple arithmetic: {calculation.expression}")
        return errors
    if not math.isclose(evaluated, calculation.result, rel_tol=5e-3, abs_tol=5e-3):
        errors.append(
            f"Calculation result {_key(calculation.result)} does not match "
            f"{calculation.expression} = {_key(evaluated)}"
        )
    inputs = [abs(item.value) for item in calculation.inputs]
    uncited = [
        literal
        for literal in literals
        if not any(math.isclose(literal, value, rel_tol=1e-9) for value in inputs)
        and literal not in _SCALE_CONSTANTS
    ]
    if uncited:
        errors.append(
            f"Calculation uses uncited values ({', '.join(map(_key, uncited))}): "
            f"{calculation.expression}"
        )
    return errors


def _verbatim_in(excerpt: str, entries: Sequence[EvidenceEntry]) -> bool:
    normalized = normalize_excerpt(excerpt)
    return bool(normalized) and any(
        normalized in normalize_excerpt(str(entry.chunk.get("text", ""))) for entry in entries
    )


def _filing(entry: EvidenceEntry) -> tuple[object, object]:
    return entry.chunk.get("ticker"), entry.chunk.get("fiscal_year")


def _verbatim_passage(
    excerpt: str, numbers: set[str], entries: Sequence[EvidenceEntry]
) -> str | None:
    """``excerpt`` if verbatim in ``entries``, else its verbatim passage holding the most
    claim numbers (then the longest), else ``None``."""
    if _verbatim_in(excerpt, entries):
        return excerpt
    passages = [
        passage.strip()
        for passage in _EXCERPT_BREAK.split(excerpt)
        if _verbatim_in(passage, entries)
    ]
    return max(
        passages,
        key=lambda passage: (len(numbers & _number_keys(passage)), len(passage)),
        default=None,
    )


def ground_claims(
    claims: Sequence[GroundedClaim], source_entries: dict[str, EvidenceEntry]
) -> list[GroundedClaim]:
    """Code-owned provenance for model claims, applied before validation.

    Only claims whose citations all come from one filing of the claim's own company are
    touched; a claim citing another company's filing (or several filings) is left as is,
    so validation rejects it.

    - The claim takes the cited filing's fiscal year: a prior-year column of a FY2025
      10-K is a FY2025-filing claim.
    - An excerpt that stitches passages together is cut to the passage that is verbatim
      in a cited source, so the number checks only see text the source contains.
    - An excerpt copied from a different source of the same filing (e.g. the income
      statement instead of the cited segment table) adds that source to the citations.

    Excerpts that no source of the filing contains stay unchanged and invalid.
    """
    grounded: list[GroundedClaim] = []
    for claim in claims:
        cited = [source_entries[sid] for sid in claim.citation_ids if sid in source_entries]
        filings = {_filing(entry) for entry in cited}
        ticker, year = next(iter(filings)) if len(filings) == 1 else (None, None)
        if ticker != claim.ticker or year is None:
            grounded.append(claim)
            continue
        update: dict[str, object] = {}
        if int(str(year)) != claim.fiscal_year:
            update["fiscal_year"] = int(str(year))
        numbers = _number_keys(claim.text)
        passage = _verbatim_passage(claim.excerpt, numbers, cited)
        if passage is None:
            for source_id, entry in source_entries.items():
                if source_id in claim.citation_ids or _filing(entry) != (ticker, year):
                    continue
                if (passage := _verbatim_passage(claim.excerpt, numbers, [entry])) is not None:
                    update["citation_ids"] = [*claim.citation_ids, source_id]
                    break
        if passage is not None and passage != claim.excerpt:
            update["excerpt"] = passage
        grounded.append(claim.model_copy(update=update) if update else claim)
    return grounded


def _names_year(text: str, year: int, claim_year: int) -> bool:
    """Whether ``text`` says which fiscal year a figure belongs to ("2023", "FY23",
    "prior year" for the year before the filing's)."""
    if re.search(rf"(?<!\d){year}(?!\d)|\bFY\s?'?{year % 100:02d}\b", text, re.IGNORECASE):
        return True
    return year == claim_year - 1 and bool(_PRIOR_YEAR.search(text))


def _fact_matches(number: _Number, record: FactRecord) -> bool:
    """A claim number equals a fact as printed, or (with a scale word) its true value."""
    if "%" in number.raw:
        return False
    if _key(number.value) == _key(record.shown):
        return True
    claimed = number.millions() if number.unit else None
    value = abs(record.value) / 1_000_000
    return (
        claimed is not None
        and abs(claimed[0] - value) <= min(claimed[1], _MAX_ROUNDING * value) + 1e-9
    )


def _period_errors(
    claim: GroundedClaim, entries: Sequence[EvidenceEntry], facts: FactsStore
) -> list[str]:
    """Claim numbers read from another fiscal year's column of a cited table.

    A FY2024 10-K table also prints FY2023 and FY2022 columns; a claim about FY2024 that
    states the FY2023 value is rejected unless the claim names that year.
    """
    records = [
        record
        for entry in entries
        for record in facts.facts_in_chunk(str(entry.chunk.get("chunk_id", "")))
    ]
    if not records:
        return []
    errors: list[str] = []
    for number in _parse_numbers(claim.text):
        matching = [record for record in records if _fact_matches(number, record)]
        if not matching or any(record.fiscal_year == claim.fiscal_year for record in matching):
            continue
        years = sorted({record.fiscal_year for record in matching})
        if any(_names_year(claim.text, year, claim.fiscal_year) for year in years):
            continue
        errors.append(
            f"Claim uses the FY{years[-1]} column of the FY{claim.fiscal_year} filing "
            f"({number.raw}) without naming that year: {claim.text}"
        )
    return errors


def _row_with(text: str, value: float) -> str | None:
    """The line of ``text`` (a table row, pipes trimmed) that prints ``value``."""
    for line in text.splitlines():
        if _key(value) in _number_keys(line):
            return line.strip().strip("|").strip()[:MAX_EXCERPT_CHARS]
    return None


def ground_computed_claims(
    claims: Sequence[GroundedClaim],
    calculations: Sequence[Calculation],
    source_entries: dict[str, EvidenceEntry],
) -> list[GroundedClaim]:
    """Cite code-computed values to the table row they were computed from.

    A claim that states a code calculation's result ("CAGR of 21.98%") often copies its
    excerpt from the prompt's computed-values list, which no source contains, or cites
    every input filing at once (a claim may cite only one filing). When either happens
    and a claim number equals the result of one of ``calculations`` (code-owned, inputs
    already verified), the claim cites the calculation's latest input of the claim's
    company instead, with that input's row as the excerpt. The calculation itself is
    still validated against all of its inputs.
    """
    grounded: list[GroundedClaim] = []
    for claim in claims:
        cited = [source_entries[sid] for sid in claim.citation_ids if sid in source_entries]
        numbers = _parse_numbers(claim.text)
        calculation = next(
            (
                item
                for item in calculations
                if any(_matches_result(number, [item.result]) for number in numbers)
            ),
            None,
        )
        one_filing = len({_filing(entry) for entry in cited}) == 1
        if calculation is None or (one_filing and _verbatim_in(claim.excerpt, cited)):
            grounded.append(claim)
            continue
        inputs = [
            (item, source_entries[item.citation_id])
            for item in calculation.inputs
            if item.citation_id in source_entries
            and source_entries[item.citation_id].chunk.get("ticker") == claim.ticker
        ]
        latest = max(
            inputs, key=lambda pair: int(pair[1].chunk.get("fiscal_year") or 0), default=None
        )
        row = _row_with(str(latest[1].chunk.get("text", "")), latest[0].value) if latest else None
        if latest is None or row is None:
            grounded.append(claim)
            continue
        grounded.append(
            claim.model_copy(
                update={
                    "citation_ids": [latest[0].citation_id],
                    "excerpt": row,
                    "fiscal_year": int(latest[1].chunk.get("fiscal_year") or claim.fiscal_year),
                }
            )
        )
    return grounded


def _claim_errors(
    claim: GroundedClaim,
    source_entries: dict[str, EvidenceEntry],
    calculation_results: Sequence[float],
    facts: FactsStore | None = None,
) -> list[str]:
    errors: list[str] = []
    if not claim.citation_ids:
        errors.append(f"Claim has no citation: {claim.text}")
        return errors
    cited = [source_entries.get(source_id) for source_id in claim.citation_ids]
    missing = [source_id for source_id, entry in zip(claim.citation_ids, cited) if entry is None]
    if missing:
        errors.append(f"Claim cites unknown sources: {', '.join(missing)}")
    valid_entries = [entry for entry in cited if entry is not None]
    if not valid_entries:
        return errors
    wrong_scope = [
        entry
        for entry in valid_entries
        if entry.chunk.get("ticker") != claim.ticker
        or entry.chunk.get("fiscal_year") != claim.fiscal_year
    ]
    if wrong_scope:
        errors.append(f"Claim scope does not match its cited filing: {claim.text}")
    excerpt = normalize_excerpt(claim.excerpt)
    containing = [
        entry
        for entry in valid_entries
        if excerpt and excerpt in normalize_excerpt(str(entry.chunk.get("text", "")))
    ]
    if not containing:
        errors.append(f"Claim excerpt is not verbatim in a cited source: {claim.text}")
    excerpt_numbers = _number_keys(claim.excerpt)
    scaled = _scaled_values(claim.excerpt, [entry.chunk.get("units") for entry in containing])
    unsupported = [
        number.raw
        for number in _parse_numbers(claim.text)
        if _key(number.value) not in excerpt_numbers
        and not _scaled_match(number, scaled)
        and not _matches_result(number, calculation_results)
    ]
    if unsupported:
        errors.append(f"Claim has unsupported numbers ({', '.join(unsupported)}): {claim.text}")
    if facts is not None and not wrong_scope:
        errors.extend(_period_errors(claim, valid_entries, facts))
    return errors


def _miscited_prose(
    text: str, claims: Sequence[GroundedClaim], calculation_results: Sequence[float]
) -> list[str]:
    """Prose numbers whose inline citations point at no valid claim stating them.

    Each number binds to the citation run that follows it ("$100 million [S1]"); a claim
    number cited with another claim's source ("$100 million [S2]") is reported.
    """
    claim_numbers = {key for claim in claims for key in _number_keys(claim.text)}
    miscited: list[str] = []
    start = 0
    for run in _CITATION_RUN.finditer(text):
        ids = list(dict.fromkeys(_SOURCE_ID.findall(run.group(0))))
        stated = {
            key
            for claim in claims
            if set(ids) & set(claim.citation_ids)
            for key in _number_keys(claim.text)
        }
        miscited.extend(
            f"{number.raw} [{', '.join(ids)}]"
            for number in _parse_numbers(text[start : run.start()])
            if number.financial
            and _key(number.value) in claim_numbers
            and _key(number.value) not in stated
            and not _matches_result(number, calculation_results)
        )
        start = run.end()
    return miscited


def validate_grounded_answer(
    answer: GroundedAnswer,
    scope: ResolvedScope,
    source_entries: dict[str, EvidenceEntry],
    catalog: FilingCatalog,
    facts: FactsStore | None = None,
) -> ValidationReport:
    """Check citations, numbers, calculations and coverage; with ``facts``, also reject
    numbers taken from another fiscal year's column of a cited table."""
    errors: list[str] = []
    inline_ids = set(_CITATION.findall(answer.answer or ""))
    unknown_inline = sorted(inline_ids - set(source_entries))
    if unknown_inline:
        errors.append(f"Answer cites unknown sources: {', '.join(unknown_inline)}")

    invalid_calculations: list[str] = []
    calculation_results: list[float] = []
    for calculation in answer.calculations:
        calculation_errors = _calculation_errors(calculation, source_entries)
        if calculation_errors:
            errors.extend(calculation_errors)
            invalid_calculations.append(calculation.description)
        else:
            calculation_results.append(calculation.result)

    invalid_claims: list[str] = []
    for claim in answer.claims:
        claim_errors = _claim_errors(claim, source_entries, calculation_results, facts)
        if claim_errors:
            errors.extend(claim_errors)
            invalid_claims.append(claim.text)
    valid_claims = [claim for claim in answer.claims if claim.text not in invalid_claims]

    claim_numbers = {key for claim in valid_claims for key in _number_keys(claim.text)}
    unsupported_prose = [
        number.raw
        for number in _parse_numbers(answer.answer or "")
        if number.financial
        and _key(number.value) not in claim_numbers
        and not _matches_result(number, calculation_results)
    ]
    if unsupported_prose:
        errors.append(
            f"Answer has numbers not backed by a valid claim: {', '.join(unsupported_prose)}"
        )
    miscited = _miscited_prose(answer.answer or "", valid_claims, calculation_results)
    if miscited:
        errors.append(
            f"Answer cites sources that do not state these numbers: {', '.join(miscited)}"
        )

    required = {(cell.ticker, cell.fiscal_year) for cell in scope.coverage}
    supplied = [(cell.ticker, cell.fiscal_year) for cell in answer.coverage]
    duplicates = [cell for cell, count in Counter(supplied).items() if count != 1]
    if duplicates:
        errors.append(f"Coverage contains duplicate cells: {duplicates}")
    missing_cells = sorted(required - set(supplied))
    extra_cells = sorted(set(supplied) - required)
    if missing_cells:
        errors.append(f"Coverage is missing required cells: {missing_cells}")
    if extra_cells:
        errors.append(f"Coverage contains unexpected cells: {extra_cells}")

    for cell in answer.coverage:
        key = (cell.ticker, cell.fiscal_year)
        label = f"{cell.ticker} FY{cell.fiscal_year}"
        if cell.status not in _FINAL_STATUSES:
            errors.append(f"Coverage for {label} has non-final status {cell.status!r}")
        in_catalog = catalog.has(*key)
        if cell.status == "not_in_corpus" and in_catalog:
            errors.append(f"Coverage incorrectly marks {label} not in corpus")
        if cell.status != "not_in_corpus" and not in_catalog:
            errors.append(f"Coverage must mark {label} not in corpus")
        matching_claims = [
            claim
            for claim in valid_claims
            if (claim.ticker, claim.fiscal_year) == key
            and any(source_id in source_entries for source_id in claim.citation_ids)
        ]
        if cell.status == "found" and not matching_claims:
            errors.append(f"Found coverage has no valid claim for {label}")
        if cell.status != "found" and matching_claims:
            errors.append(f"Coverage is not found despite a valid claim for {label}")

    return ValidationReport(
        is_valid=not errors,
        errors=list(dict.fromkeys(errors)),
        dropped_claims=list(dict.fromkeys(invalid_claims)),
        dropped_calculations=list(dict.fromkeys(invalid_calculations)),
    )


def _claims_prose(claims: Sequence[GroundedClaim]) -> str:
    return " ".join(
        f"{_CITATION_GROUP.sub('', claim.text).strip()} "
        + "".join(f"[{source_id}]" for source_id in claim.citation_ids)
        for claim in claims
    )


def _recomputed_cell(cell: CoverageCell, claims: Sequence[GroundedClaim]) -> CoverageCell:
    if cell.status == "not_in_corpus":
        return cell
    found = any(
        (claim.ticker, claim.fiscal_year) == (cell.ticker, cell.fiscal_year) for claim in claims
    )
    return cell.model_copy(update={"status": "found" if found else "not_disclosed"})


def drop_invalid_claims(
    answer: GroundedAnswer,
    report: ValidationReport,
    required: Sequence[CoverageCell] = (),
) -> GroundedAnswer:
    """Remove invalid claims/calculations, recompute coverage and rebuild cited prose.

    ``required`` (the resolved scope's coverage) replaces the model-supplied cells so
    missing or extra cells cannot survive; the prose is rebuilt from the remaining
    claims so no uncited text from the draft is returned.
    """
    dropped_claims = set(report.dropped_claims)
    dropped_calculations = set(report.dropped_calculations)
    claims = [claim for claim in answer.claims if claim.text not in dropped_claims]
    return answer.model_copy(
        update={
            "claims": claims,
            "calculations": [
                calculation
                for calculation in answer.calculations
                if calculation.description not in dropped_calculations
            ],
            "coverage": [_recomputed_cell(cell, claims) for cell in required or answer.coverage],
            "answer": _claims_prose(claims) if claims else None,
            "insufficient_context": not claims,
        }
    )
