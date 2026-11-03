"""Grading rules of scripts/numeric_eval.py and the shape of the numeric golden set."""

from __future__ import annotations

import json

import pytest

from scripts.numeric_eval import GOLDEN, grade

REVENUE = {
    "ticker": "MSFT",
    "fiscal_year": 2024,
    "metric": "revenue",
    "value": 245122.0,
    "unit": "USD",
    "scale": 6,
    "other_columns": [198270.0, 211915.0],
}
EPS = {**REVENUE, "metric": "eps_diluted", "value": 4.9, "unit": "USD/share", "scale": 0}


@pytest.mark.parametrize(
    ("answer", "exact", "wrong_period"),
    [
        ("Revenue was $245,122 million [S1].", True, False),
        ("Revenue was $245.1 billion.", True, False),
        ("Revenue was about $245 billion.", True, False),
        ("Revenue was $250 billion.", False, False),
        ("Revenue was $211,915 million.", False, True),
        ("Revenue was $211.9 billion.", False, True),
        ("Revenue grew to $245,122 million from $211,915 million.", True, False),
        (None, False, False),
    ],
)
def test_figures_match_as_printed_or_rounded_and_other_columns_are_wrong_period(
    answer, exact, wrong_period
) -> None:
    result = grade({"facts": [REVENUE]}, answer)
    assert result["exact"] is exact
    assert bool(result["wrong_period"]) is wrong_period


def test_per_share_percent_and_forbidden_values() -> None:
    assert grade({"facts": [EPS]}, "Diluted EPS was $4.90.")["exact"]
    assert not grade({"facts": [EPS]}, "Diluted EPS was $1.63.")["exact"]
    assert grade({"percents": [44.64]}, "Operating margin was 44.6%.")["exact"]
    assert not grade({"percents": [44.64]}, "Operating margin was 44.64 million.")["exact"]
    segment = {"forbidden": [245122.0]}
    assert grade(segment, "Intelligent Cloud revenue was $105,362 million.")["exact"]
    assert not grade(segment, "Revenue was $245,122 million.")["exact"]
    assert not grade({}, "Anything")["graded"]


def test_percent_questions_do_not_require_their_input_figures() -> None:
    case = {"facts": [REVENUE], "percents": [15.67]}
    assert grade(case, "Revenue grew 15.67% [S1].")["exact"]
    assert grade(case, "Revenue grew 16%.")["exact"]  # rounded to its printed precision
    assert not grade(case, "Revenue grew 17.1%.")["exact"]
    wrong = grade(case, "Revenue grew 15.67% to $211,915 million.")
    assert not wrong["exact"] and wrong["wrong_period"]


def test_golden_set_is_well_formed() -> None:
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 30 and len({case["id"] for case in cases}) == 30
    for case in cases:
        assert case["question"].endswith("?") or case["question"].endswith(".")
        for item in case["facts"]:
            assert f"{abs(item['value']):,.2f}".rstrip("0").rstrip(".") in item["row"] or (
                f"{abs(item['value']):,.0f}" in item["row"]
            ), item
