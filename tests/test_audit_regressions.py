"""Regression tests for the audit fixes (validation parsing, chunk attribution, rate limiting)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from api.rate_limit import check_rate_limit, in_memory_limiter
from generation.validation import parse_llm_response, validate_answer
from ingestion.chunking.chunker import chunk_document
from ingestion.parser.parser import (
    detect_section,
    doc_metadata,
    iter_blocks,
    parse_all,
    parse_document,
    parse_table,
    score,
)
from scripts.check_coverage import EXPECTED_FILINGS, validate_corpus


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


def _filing_soup(ticker="aapl", form="10-k", fiscal_year="2025", period_end="2025-09-27"):
    html = f"""
    <html><body>
      <ix:nonnumeric name="dei:EntityRegistrantName">Apple Inc.</ix:nonnumeric>
      <ix:nonnumeric name="dei:TradingSymbol">{ticker}</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentType">{form}</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentFiscalYearFocus">{fiscal_year}</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentPeriodEndDate">{period_end}</ix:nonnumeric>
    </body></html>
    """
    return BeautifulSoup(html, "lxml")


def test_doc_metadata_normalizes_and_validates_dei_fields():
    metadata = doc_metadata(
        _filing_soup(period_end="September\ufffd27 , 2025"), Path("AAPL_10-K_2025.html")
    )
    assert metadata["doc_id"] == "AAPL_10-K_2025"
    assert metadata["ticker"] == "AAPL"
    assert metadata["form"] == "10-K"
    assert metadata["fiscal_year"] == 2025
    assert metadata["period_end"] == "2025-09-27"


@pytest.mark.parametrize(
    ("field", "value"),
    [("ticker", ""), ("fiscal_year", "FY2025"), ("period_end", "09/27/2025")],
)
def test_doc_metadata_rejects_invalid_required_dei(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        doc_metadata(_filing_soup(**kwargs), Path("filing.html"))


def _table(rows_html):
    soup = BeautifulSoup(f"<table>{rows_html}</table>", "lxml")
    return parse_table(soup.table, "T_t001", [], {"company": "X"}, {}, {})


def test_footnote_table_becomes_text_block():
    t = _table(
        "<tr><td>(a)</td><td>Includes goodwill of $5.5 billion related to GitHub. "
        "See Note 8 for further information.</td></tr>"
    )
    assert t.kind == "text_block"
    assert t.markdown.startswith("(a) Includes goodwill of $5.5 billion")


def test_numeric_table_is_not_treated_as_footnote():
    t = _table(
        "<tr><td></td><td>2025</td><td>2024</td></tr>"
        "<tr><td>Revenue</td><td>100</td><td>90</td></tr>"
        "<tr><td>Cost</td><td>40</td><td>30</td></tr>"
    )
    assert t.kind == "financial"


def test_unlinked_table_of_contents_is_dropped():
    rows = "".join(
        f"<tr><td>Item {i}.</td><td>Section {i}</td><td>{i * 3}</td></tr>" for i in range(1, 8)
    )
    assert _table(rows).kind == "toc"


def test_header_only_text_fragment_becomes_text_block():
    t = _table(
        "<tr><td>Exhibit</td><td></td><td>Incorporated by Reference</td><td>Filed</td></tr>"
        "<tr><td>Number</td><td>Exhibit Description</td><td>Form</td><td>Herewith</td></tr>"
    )
    assert t.kind == "text_block"
    assert "Exhibit Description" in t.markdown


def _indented(label, pt, values):
    cells = "".join(f"<td>{v}</td>" for v in values)
    return f'<tr><td style="padding-left:{pt}pt">{label}</td>{cells}</tr>'


def test_net_row_closes_net_change_group():
    t = _table(
        "<tr><td></td><td>2025</td><td>2024</td></tr>"
        + _indented("Net change in foreign currency translation adjustments:", 0, ["", ""])
        + _indented("Translation adjustments", 18, ["(538)", "78"])
        + _indented("Reclassification adjustment", 18, ["10", "(108)"])
        + _indented("Net foreign currency translation adjustments", 60, ["(528)", "(30)"])
        + _indented("Net change in unrealized gains on securities:", 18, ["", ""])
        + _indented("Unrealized gains", 36, ["(17)", "83"])
    )
    paths = {r.label: r.path for r in t.rows}
    assert paths["Net change in unrealized gains on securities"] == [
        "Net change in unrealized gains on securities"
    ]


def test_repeated_period_header_row_is_not_data():
    t = _table(
        "<tr><td></td><td>2025</td><td>2024</td></tr>"
        "<tr><td>Net sales</td><td>100</td><td>90</td></tr>"
        "<tr><td></td><td>December 31,</td><td></td></tr>"
        "<tr><td></td><td>2025</td><td>2024</td></tr>"
        "<tr><td>Total assets</td><td>500</td><td>400</td></tr>"
    )
    data = [r.label for r in t.rows if r.row_type == "data"]
    assert data == ["Net sales", "Total assets"]


def test_plain_item_heading_accepted_but_cross_reference_rejected():
    assert detect_section("ITEM 4. MINE SAFETY DISCLOSURES", False, allow_plain=True)[1] == "Item 4"
    assert detect_section("Item 1A of Part I describes risks.", False, allow_plain=True) is None
    assert detect_section("ITEM 4. MINE SAFETY DISCLOSURES", False) is None


def test_loose_inline_text_beside_blocks_is_kept():
    soup = BeautifulSoup(
        "<body><div><div>Table intro</div><span>(1) Includes $548 million</span>"
        "<span> of assets.</span></div></body>",
        "lxml",
    )
    texts = [b.get_text("") for b in iter_blocks(soup.body)]
    assert texts == ["Table intro", "(1) Includes $548 million of assets."]


def test_item_headings_in_tables_and_plain_text_set_sections(tmp_path):
    html = """<html><body>
      <ix:nonnumeric name="dei:TradingSymbol">X</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentType">10-K</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentFiscalYearFocus">2025</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentPeriodEndDate">2025-12-31</ix:nonnumeric>
      <table><tr><td><b>Item 1.</b></td><td><b>Business</b></td></tr></table>
      <p>We make things.</p>
      <p>See Item 7 for more.</p>
      <p>ITEM 7. MANAGEMENT DISCUSSION</p>
      <p>Revenue grew.</p>
    </body></html>"""
    path = tmp_path / "X_10-K_2025.html"
    path.write_text(html, encoding="utf-8")
    doc, *_ = parse_document(path)
    sections = {e.text: e.section for e in doc.elements}
    assert sections["We make things."] == "Item 1"
    assert sections["See Item 7 for more."] == "Item 1"
    assert sections["Revenue grew."] == "Item 7"


def test_score_separates_accuracy_from_completeness():
    q = score("financial", 1, None, None, 0, 10, {})
    assert q["accuracy_score"] == 1.0
    assert q["completeness_score"] == 0.0
    assert q["score"] == 0.8
    assert {"no_title", "no_units"} <= set(q["reasons"])
    non_monetary = score("financial", 1, "Shares", None, 0, 10, {}, units_expected=False)
    assert non_monetary["completeness_score"] == 1.0 and "no_units" not in non_monetary["reasons"]


def test_parse_all_rejects_empty_raw_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .html filings"):
        parse_all(tmp_path, tmp_path / "parsed")


def test_validate_corpus_accepts_expected_matrix():
    raw_paths = []
    metadata = []
    for ticker, year in EXPECTED_FILINGS:
        source_file = f"{ticker}_10-K_{year}.html"
        raw_paths.append(Path(source_file))
        metadata.append(
            {
                "doc_id": f"{ticker}_10-K_{year}",
                "ticker": ticker,
                "form": "10-K",
                "fiscal_year": year,
                "period_end": f"{year}-12-31",
                "source_file": source_file,
            }
        )
    assert validate_corpus(raw_paths, metadata) == []
