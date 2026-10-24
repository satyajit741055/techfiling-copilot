"""
Output Parsing and Validation.

Parses LLM JSON responses and validates:
- Citation validity (source IDs exist in context)
- Citation presence (factual answers have citations)
- Abstention consistency (no specific answers when abstaining)
"""

from __future__ import annotations

import json
import re

from .schemas import GeneratedAnswer


def parse_llm_response(content: str) -> GeneratedAnswer:
    """
    Parse the LLM's JSON response into a GeneratedAnswer.

    Handles common issues:
    - Extra text before/after JSON
    - Markdown code blocks
    - Missing fields (uses defaults)
    - Invalid JSON (returns error state)

    Args:
        content: Raw LLM response content

    Returns:
        GeneratedAnswer with parsed fields or error state
    """
    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        content = "\n".join(json_lines)

    # Try to extract JSON if there's extra text
    if not content.startswith("{"):
        # Look for JSON object in the content
        match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if match:
            content = match.group(0)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return GeneratedAnswer(
            answer=None,
            explanation=f"Failed to parse JSON: {e}",
            citations=[],
            calculation=None,
            insufficient_context=False,
            is_valid=False,
            validation_errors=[f"JSON parse error: {e}"],
        )

    return GeneratedAnswer(
        answer=data.get("answer"),
        explanation=data.get("explanation", ""),
        citations=data.get("citations", []),
        calculation=data.get("calculation"),
        insufficient_context=data.get("insufficient_context", False),
    )


def validate_answer(
    answer: GeneratedAnswer,
    valid_source_ids: list[str],
) -> GeneratedAnswer:
    """
    Validate the generated answer.

    Checks:
    1. All cited sources exist in the context
    2. Non-abstention answers have at least one citation
    3. Abstention answers don't have specific numerical claims

    Args:
        answer: The parsed answer to validate
        valid_source_ids: List of valid source IDs (e.g., ["S1", "S2", "S3"])

    Returns:
        The same answer with is_valid and validation_errors updated
    """
    errors = []

    # Check citation validity
    for citation in answer.citations:
        if citation not in valid_source_ids:
            errors.append(f"Invalid citation: {citation} (valid: {valid_source_ids})")

    # Check citation presence for factual answers
    if answer.answer and not answer.insufficient_context:
        if not answer.citations:
            errors.append("Factual answer without citations")

    # Check abstention consistency
    if answer.insufficient_context:
        if answer.answer and answer.answer.lower() not in ["null", "none", ""]:
            # Allow abstention with explanation but flag if there's a specific answer
            if "$" in str(answer.answer) or any(c.isdigit() for c in str(answer.answer)):
                errors.append("Abstention with specific numerical answer")

    answer.validation_errors = errors
    answer.is_valid = len(errors) == 0

    return answer


def extract_numeric_value(text: str) -> tuple[float | None, str | None]:
    """
    Extract numeric value and unit from a financial answer.

    Examples:
        "$416,161 million" -> (416161.0, "million")
        "$94.83 billion" -> (94.83, "billion")
        "4.2%" -> (4.2, "percent")

    Args:
        text: Answer text to parse

    Returns:
        Tuple of (value, unit) or (None, None) if no number found
    """
    if not text:
        return None, None

    text = str(text).lower().strip()

    # Check for percentage
    pct_match = re.search(r'([\d,]+\.?\d*)\s*%', text)
    if pct_match:
        value = float(pct_match.group(1).replace(",", ""))
        return value, "percent"

    # Check for currency values
    money_match = re.search(r'\$?\s*([\d,]+\.?\d*)\s*(billion|million|thousand)?', text)
    if money_match:
        value = float(money_match.group(1).replace(",", ""))
        unit = money_match.group(2) or "units"
        return value, unit

    # Try to extract any number
    num_match = re.search(r'([\d,]+\.?\d*)', text)
    if num_match:
        value = float(num_match.group(1).replace(",", ""))
        return value, "units"

    return None, None


def normalize_to_millions(value: float, unit: str) -> float:
    """
    Convert value to millions for comparison.

    Args:
        value: Numeric value
        unit: Unit string (billion, million, thousand, percent, units)

    Returns:
        Value normalized to millions (or unchanged for percent)
    """
    if unit == "billion":
        return value * 1000
    elif unit == "thousand":
        return value / 1000
    elif unit == "percent":
        return value  # Keep as-is for percentages
    else:
        return value  # Assume already in millions


def compare_numeric_answers(
    generated: str,
    expected: str,
    tolerance: float = 0.01,
) -> tuple[bool, str]:
    """
    Compare generated and expected numeric answers.

    Handles unit normalization (e.g., "$94.83 billion" == "$94,830 million").

    Args:
        generated: Generated answer text
        expected: Expected answer text
        tolerance: Relative tolerance for comparison (default 1%)

    Returns:
        Tuple of (is_correct, reason)
    """
    gen_val, gen_unit = extract_numeric_value(generated)
    exp_val, exp_unit = extract_numeric_value(expected)

    if gen_val is None:
        return False, f"Could not extract number from generated: '{generated}'"
    if exp_val is None:
        return False, f"Could not extract number from expected: '{expected}'"

    # Handle percentages separately
    if gen_unit == "percent" and exp_unit == "percent":
        diff = abs(gen_val - exp_val)
        if diff <= 0.1:  # Within 0.1 percentage points
            return True, f"Percentage match: {gen_val}% vs {exp_val}%"
        else:
            return False, f"Percentage mismatch: {gen_val}% vs {exp_val}% (diff={diff:.2f}pp)"

    # Normalize to millions for comparison
    gen_normalized = normalize_to_millions(gen_val, gen_unit)
    exp_normalized = normalize_to_millions(exp_val, exp_unit)

    # Check relative error
    if exp_normalized == 0:
        is_correct = gen_normalized == 0
    else:
        relative_error = abs(gen_normalized - exp_normalized) / abs(exp_normalized)
        is_correct = relative_error <= tolerance

    if is_correct:
        return True, f"Match: {gen_val} {gen_unit} ≈ {exp_val} {exp_unit}"
    else:
        return False, f"Mismatch: {gen_val} {gen_unit} vs {exp_val} {exp_unit}"
