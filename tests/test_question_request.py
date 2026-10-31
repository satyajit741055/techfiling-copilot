"""Validation of the /questions request scope: doc_id, metadata filters and grouping."""

import pytest
from pydantic import ValidationError

from api.schemas import QuestionRequest

Q = "How did R&D expense change?"


def test_doc_id_only_request_is_unchanged():
    request = QuestionRequest(question=Q, doc_id="AAPL_10-K_2025")
    assert request.retrieval_options() == {}


def test_filters_normalise_tickers_and_dump_ranges():
    request = QuestionRequest(
        question=Q,
        filters={"ticker": ["aapl", "msft"], "fiscal_year": {"gte": 2022, "lte": 2024}},
    )
    assert request.retrieval_options() == {
        "filters": {"ticker": ["AAPL", "MSFT"], "fiscal_year": {"gte": 2022, "lte": 2024}}
    }


def test_grouping_options():
    request = QuestionRequest(
        question=Q, filters={"fiscal_year": 2024}, group_by="ticker", groups=["aapl", "MSFT"]
    )
    assert request.retrieval_options() == {
        "filters": {"fiscal_year": 2024},
        "group_by": "ticker",
        "groups": ["AAPL", "MSFT"],
        "group_limit": 3,
    }
    years = QuestionRequest(
        question=Q, filters={"ticker": "TSLA"}, group_by="fiscal_year", groups=["2023", 2024]
    )
    assert years.groups == [2023, 2024]
    discover = QuestionRequest(question=Q, group_by="doc_id", group_limit=2)
    assert discover.retrieval_options()["groups"] is None
    at_cap = QuestionRequest(question=Q, group_by="ticker", groups=["A", "B", "C"], context_k=10)
    assert at_cap.context_k == 10


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"filters": {}},
        {"filters": {"ticker": []}},
        {"filters": {"fiscal_year": {}}},
        {"filters": {"fiscal_year": {"gte": 2025, "lte": 2021}}},
        {"filters": {"cik": "123"}},
        {"filters": {"type": "image"}},
        {"doc_id": "AAPL_10-K_2025", "groups": ["AAPL"]},
        {"group_by": "ticker", "groups": ["AAPL"], "filters": {"ticker": "AAPL"}},
        {"group_by": "doc_id", "groups": ["AAPL_10-K_2025"], "doc_id": "AAPL_10-K_2025"},
        {"group_by": "fiscal_year", "groups": ["FY24"]},
        {"group_by": "ticker", "groups": ["A", "B", "C", "D", "E", "F", "G"]},
        {"group_by": "section"},
        {"group_by": "ticker", "groups": ["AAPL", "MSFT"], "context_k": 20},
        {"group_by": "doc_id", "group_limit": 6, "context_k": 6},
    ],
)
def test_invalid_scopes_rejected(body):
    with pytest.raises(ValidationError):
        QuestionRequest(question=Q, **body)
