"""Offline tests for LLM output parsing, answer validation and numeric comparison."""

import pytest

from generation.schemas import GeneratedAnswer
from generation.validation import (
    compare_numeric_answers,
    extract_numeric_value,
    normalize_to_millions,
    parse_llm_response,
    validate_answer,
)

VALID = ["S1", "S2", "S3"]


def _answer(**overrides) -> GeneratedAnswer:
    fields = {
        "answer": "$416,161 million",
        "explanation": "x",
        "citations": ["S1"],
        "calculation": None,
        "insufficient_context": False,
    }
    fields.update(overrides)
    return GeneratedAnswer(**fields)


class TestParseLlmResponse:
    def test_plain_json(self):
        parsed = parse_llm_response(
            '{"answer": "a", "explanation": "b", "citations": ["S1"], "calculation": null, "insufficient_context": false}'
        )
        assert parsed.answer == "a" and parsed.explanation == "b"
        assert parsed.citations == ["S1"] and parsed.calculation is None
        assert parsed.insufficient_context is False and parsed.is_valid

    def test_markdown_code_block(self):
        parsed = parse_llm_response('```json\n{"answer": "a", "citations": ["S2"]}\n```')
        assert parsed.answer == "a" and parsed.citations == ["S2"]

    def test_prose_around_json(self):
        parsed = parse_llm_response('Here you go: {"answer": "a", "citations": []} Hope that helps!')
        assert parsed.answer == "a"

    def test_missing_fields_use_defaults(self):
        parsed = parse_llm_response("{}")
        assert parsed.answer is None and parsed.explanation == ""
        assert parsed.citations == [] and parsed.calculation is None
        assert parsed.insufficient_context is False

    @pytest.mark.parametrize("raw", ["", "not json at all", "{broken", "[1, 2]", '"just a string"'])
    def test_unparseable_is_error_state(self, raw):
        parsed = parse_llm_response(raw)
        assert parsed.answer is None and parsed.is_valid is False
        assert parsed.validation_errors and parsed.validation_errors[0].startswith("JSON parse error")

    def test_parse_error_survives_validation(self):
        validated = validate_answer(parse_llm_response("garbage"), VALID)
        assert validated.is_valid is False
        assert any("JSON parse error" in e for e in validated.validation_errors)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("true", True), ("True", True), ("yes", True), ("1", True), ("false", False), ("no", False), ("", False)],
    )
    def test_string_booleans_are_coerced(self, value, expected):
        parsed = parse_llm_response(f'{{"insufficient_context": "{value}"}}')
        assert parsed.insufficient_context is expected

    def test_numeric_answer_becomes_text(self):
        assert parse_llm_response('{"answer": 42}').answer == "42"

    def test_container_fields_are_serialized(self):
        parsed = parse_llm_response('{"answer": ["a", "b"], "calculation": {"x": 1}}')
        assert parsed.answer == '["a", "b"]' and parsed.calculation == '{"x": 1}'

    def test_non_string_citations_are_stringified(self):
        assert parse_llm_response('{"citations": [1, "S2"]}').citations == ["1", "S2"]

    def test_invalid_citations_type_becomes_empty(self):
        assert parse_llm_response('{"citations": 5}').citations == []

    def test_first_object_wins(self):
        assert parse_llm_response('{"answer": "first"} {"answer": "second"}').answer == "first"


class TestValidateAnswer:
    def test_valid_answer_passes(self):
        result = validate_answer(_answer(), VALID)
        assert result.is_valid and result.validation_errors == []

    def test_unknown_citation_is_flagged(self):
        result = validate_answer(_answer(citations=["S1", "S9"]), VALID)
        assert not result.is_valid
        assert any("Invalid citation: S9" in e for e in result.validation_errors)

    def test_factual_answer_needs_citations(self):
        result = validate_answer(_answer(citations=[]), VALID)
        assert not result.is_valid and "Factual answer without citations" in result.validation_errors

    def test_abstention_without_answer_is_valid(self):
        result = validate_answer(_answer(answer=None, citations=[], insufficient_context=True), VALID)
        assert result.is_valid

    def test_abstention_text_without_numbers_is_valid(self):
        result = validate_answer(
            _answer(answer="Not stated in the filing", citations=[], insufficient_context=True), VALID
        )
        assert result.is_valid

    @pytest.mark.parametrize("text", ["$5 billion", "about 12", "2025"])
    def test_abstention_with_numeric_answer_is_flagged(self, text):
        result = validate_answer(_answer(answer=text, citations=[], insufficient_context=True), VALID)
        assert not result.is_valid
        assert "Abstention with specific numerical answer" in result.validation_errors

    def test_revalidation_does_not_hide_existing_errors(self):
        parsed = parse_llm_response("garbage")
        parsed.citations = ["S1"]
        assert validate_answer(parsed, VALID).is_valid is False

    def test_empty_source_list_rejects_every_citation(self):
        assert not validate_answer(_answer(citations=["S1"]), []).is_valid


class TestExtractNumericValue:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("$416,161 million", (416161.0, "million")),
            ("$94.83 billion", (94.83, "billion")),
            ("4.2%", (4.2, "percent")),
            ("$1,234", (1234.0, "units")),
            ("12 thousand", (12.0, "thousand")),
        ],
    )
    def test_examples(self, text, expected):
        assert extract_numeric_value(text) == expected

    @pytest.mark.parametrize("text", ["", None, "no digits here"])
    def test_no_number(self, text):
        assert extract_numeric_value(text) == (None, None)


class TestNormalizeToMillions:
    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [(2, "billion", 2000), (5000, "thousand", 5), (7, "million", 7), (3.5, "percent", 3.5), (9, "units", 9)],
    )
    def test_units(self, value, unit, expected):
        assert normalize_to_millions(value, unit) == expected


class TestCompareNumericAnswers:
    def test_billion_equals_million(self):
        assert compare_numeric_answers("$94.83 billion", "$94,830 million")[0]

    def test_within_tolerance(self):
        assert compare_numeric_answers("$100 million", "$100.5 million")[0]

    def test_outside_tolerance(self):
        ok, reason = compare_numeric_answers("$100 million", "$110 million")
        assert not ok and reason.startswith("Mismatch")

    def test_percent_within_a_tenth_of_a_point(self):
        assert compare_numeric_answers("4.2%", "4.25%")[0]
        assert not compare_numeric_answers("4.2%", "4.6%")[0]

    def test_zero_expected(self):
        assert compare_numeric_answers("0", "0")[0]
        assert not compare_numeric_answers("5", "0")[0]

    def test_unparseable_inputs(self):
        ok, reason = compare_numeric_answers("n/a", "$5 million")
        assert not ok and "generated" in reason
        ok, reason = compare_numeric_answers("$5 million", "n/a")
        assert not ok and "expected" in reason
