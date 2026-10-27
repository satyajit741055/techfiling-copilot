"""Regression tests for the audit fixes (validation parsing, chunk attribution, rate limiting)."""

from types import SimpleNamespace

from api.rate_limit import check_rate_limit, in_memory_limiter
from generation.validation import parse_llm_response, validate_answer
from ingestion.chunking.chunker import chunk_document


def _validated(raw: str):
    return validate_answer(parse_llm_response(raw), ["S1"])


def test_numeric_answer_with_abstention_does_not_crash():
    answer = _validated(
        '{"answer": 416161, "explanation": "x", "citations": ["S1"], "insufficient_context": true}'
    )
    assert answer.answer == "416161"
    assert not answer.is_valid


def test_non_object_json_is_a_parse_error():
    answer = _validated("[1, 2]")
    assert not answer.is_valid and answer.explanation.startswith("Failed to parse JSON")


def test_null_explanation_and_string_citation_are_coerced():
    answer = _validated('{"answer": "1", "explanation": null, "citations": "S1"}')
    assert answer.explanation == "" and answer.citations == ["S1"] and answer.is_valid


def test_nested_json_after_prose_is_parsed_whole():
    answer = _validated(
        'Sure! {"answer":"a","explanation":"b","citations":["S1"],"calculation":{"x":1}}'
    )
    assert answer.answer == "a" and answer.citations == ["S1"] and answer.calculation == '{"x": 1}'


def _el(order, type_, text, section):
    return {
        "doc_id": "D",
        "order": order,
        "type": type_,
        "text": text,
        "part": "Part I",
        "section": section,
        "section_title": section,
    }


def test_pseudo_heading_at_item_start_does_not_relabel_previous_item():
    elements = [
        _el(0, "paragraph", "Body text. " * 30, "Item 1"),
        _el(1, "paragraph", "Risk Overview", "Item 1A"),
        _el(2, "paragraph", "More body text here.", "Item 1A"),
    ]
    chunks = chunk_document(elements, min_chars=50)
    assert [(c["section"], c["element_orders"]) for c in chunks] == [
        ("Item 1", [0]),
        ("Item 1A", [1, 2]),
    ]


def _request(user: str):
    return SimpleNamespace(state=SimpleNamespace(user_id=user))


def test_in_memory_limits_are_independent_per_endpoint(monkeypatch):
    monkeypatch.setattr("api.rate_limit.redis_url", None)
    in_memory_limiter.requests.clear()
    request = _request("u1")
    for _ in range(30):
        check_rate_limit(request, 30, 60)
    check_rate_limit(request, 2, 60)
    check_rate_limit(request, 2, 60)
