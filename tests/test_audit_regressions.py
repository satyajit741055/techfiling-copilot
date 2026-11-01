"""Regression tests for the audit fixes (validation parsing, chunk attribution, rate limiting)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from bs4 import BeautifulSoup

from api.rate_limit import check_rate_limit, in_memory_limiter
from generation.validation import parse_llm_response, validate_answer
from ingestion.chunking.chunker import chunk_doc_id, chunk_document, load_document_metadata
from ingestion.chunking.embedder import build_manifest, embed_and_save
from ingestion.parser.parser import (
    detect_section,
    doc_metadata,
    iter_blocks,
    parse_all,
    parse_document,
    parse_table,
    score,
)
from ingestion.processor import validate_ingestion
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


def test_processor_uses_current_run_doc_ids(tmp_path):
    """When parsing is enabled, only the parsed documents are used, not stale files."""
    from ingestion.processor import process

    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    chunks_dir = tmp_path / "chunks"
    raw_dir.mkdir()
    parsed_dir.mkdir()
    chunks_dir.mkdir()

    # Create a valid filing
    html = """<html><body>
      <ix:nonnumeric name="dei:EntityRegistrantName">Test Inc.</ix:nonnumeric>
      <ix:nonnumeric name="dei:TradingSymbol">TEST</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentType">10-K</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentFiscalYearFocus">2025</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentPeriodEndDate">2025-12-31</ix:nonnumeric>
      <p>Test content.</p>
    </body></html>"""
    (raw_dir / "TEST_10-K_2025.html").write_text(html, encoding="utf-8")

    # Create a stale parsed file from a previous run
    stale_doc = {
        "doc_id": "OLD_10-K_2024",
        "ticker": "OLD",
        "form": "10-K",
        "fiscal_year": 2024,
        "period_end": "2024-12-31",
        "source_file": "OLD_10-K_2024.html",
    }
    (parsed_dir / "OLD_10-K_2024.document.json").write_text(
        json.dumps(stale_doc), encoding="utf-8"
    )
    (parsed_dir / "OLD_10-K_2024.elements.jsonl").write_text(
        '{"doc_id":"OLD_10-K_2024","order":0,"type":"paragraph","text":"Stale content"}',
        encoding="utf-8",
    )

    # Process with parsing enabled
    out_dir = process(
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        chunks_dir=chunks_dir,
        skip_parse=False,
        skip_embed=True,
        pseudo_headings=False,
    )

    # Only TEST_10-K_2025 should be in the output, not OLD_10-K_2024
    chunk_files = list(out_dir.glob("*.chunks.jsonl"))
    assert len(chunk_files) == 1
    assert chunk_files[0].name == "TEST_10-K_2025.chunks.jsonl"


def test_load_document_metadata(tmp_path):
    """Load and validate document metadata from JSON file."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    metadata = {
        "doc_id": "AAPL_10-K_2025",
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "cik": "0000320193",
        "form": "10-K",
        "fiscal_year": 2025,
        "period_end": "2025-09-27",
        "source_file": "AAPL_10-K_2025.html",
    }
    (parsed_dir / "AAPL_10-K_2025.document.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    loaded = load_document_metadata("AAPL_10-K_2025", parsed_dir)
    assert loaded["ticker"] == "AAPL"
    assert loaded["fiscal_year"] == 2025
    assert loaded["form"] == "10-K"
    assert loaded["period_end"] == "2025-09-27"


def test_load_document_metadata_fails_on_missing_file(tmp_path):
    """Missing metadata file raises FileNotFoundError."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Document metadata file not found"):
        load_document_metadata("AAPL_10-K_2025", parsed_dir)


def test_load_document_metadata_fails_on_doc_id_mismatch(tmp_path):
    """Internal doc_id mismatch raises ValueError."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    metadata = {
        "doc_id": "WRONG_10-K_2025",
        "ticker": "AAPL",
        "form": "10-K",
        "fiscal_year": 2025,
        "period_end": "2025-09-27",
    }
    (parsed_dir / "AAPL_10-K_2025.document.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Internal doc_id mismatch"):
        load_document_metadata("AAPL_10-K_2025", parsed_dir)


def test_load_document_metadata_fails_on_missing_fields(tmp_path):
    """Missing required fields raises ValueError."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    metadata = {
        "doc_id": "AAPL_10-K_2025",
        "ticker": "AAPL",
        "form": "10-K",
        # Missing fiscal_year and period_end
    }
    (parsed_dir / "AAPL_10-K_2025.document.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Missing required metadata fields"):
        load_document_metadata("AAPL_10-K_2025", parsed_dir)


def test_chunk_doc_id_stamps_metadata_on_chunks(tmp_path):
    """Every output chunk receives all four metadata fields."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    # Write document metadata
    metadata = {
        "doc_id": "TEST_10-K_2025",
        "ticker": "TEST",
        "form": "10-K",
        "fiscal_year": 2025,
        "period_end": "2025-12-31",
    }
    (parsed_dir / "TEST_10-K_2025.document.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    # Write elements
    elements = [
        {
            "doc_id": "TEST_10-K_2025",
            "order": 0,
            "type": "paragraph",
            "text": "First paragraph. " * 20,
            "part": "Part I",
            "section": "Item 1",
            "section_title": "Business",
        },
        {
            "doc_id": "TEST_10-K_2025",
            "order": 1,
            "type": "paragraph",
            "text": "Second paragraph. " * 20,
            "part": "Part I",
            "section": "Item 1",
            "section_title": "Business",
        },
    ]
    (parsed_dir / "TEST_10-K_2025.elements.jsonl").write_text(
        "\n".join(json.dumps(e) for e in elements), encoding="utf-8"
    )

    chunks = chunk_doc_id("TEST_10-K_2025", parsed_dir, pseudo_headings=False)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["ticker"] == "TEST"
        assert chunk["fiscal_year"] == 2025
        assert chunk["form"] == "10-K"
        assert chunk["period_end"] == "2025-12-31"


def test_chunk_doc_id_preserves_chunk_boundaries_and_embed_text(tmp_path):
    """Metadata addition does not change chunk boundaries or embedding text."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    metadata = {
        "doc_id": "TEST_10-K_2025",
        "ticker": "TEST",
        "form": "10-K",
        "fiscal_year": 2025,
        "period_end": "2025-12-31",
    }
    (parsed_dir / "TEST_10-K_2025.document.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    elements = [
        {
            "doc_id": "TEST_10-K_2025",
            "order": 0,
            "type": "paragraph",
            "text": "Content. " * 50,
            "part": "Part I",
            "section": "Item 1",
            "section_title": "Business",
        }
    ]
    (parsed_dir / "TEST_10-K_2025.elements.jsonl").write_text(
        json.dumps(elements[0]), encoding="utf-8"
    )

    chunks = chunk_doc_id("TEST_10-K_2025", parsed_dir, pseudo_headings=False)

    # Check that embed_text still follows the breadcrumb + text format
    for chunk in chunks:
        assert chunk["embed_text"].startswith(chunk["breadcrumb"])
        assert "\n\n" in chunk["embed_text"]
        assert chunk["text"] in chunk["embed_text"]


def test_validate_ingestion_passes_on_valid_corpus():
    """Valid corpus with unique doc_ids and proper metadata passes validation."""
    doc_ids = ["AAPL_10-K_2025", "MSFT_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-09-27",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
        {
            "chunk_id": "MSFT_10-K_2025_c0000",
            "doc_id": "MSFT_10-K_2025",
            "ticker": "MSFT",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-06-30",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
    ]
    validate_ingestion(doc_ids, chunks)  # Should not raise


def test_validate_ingestion_fails_on_duplicate_doc_ids():
    """Duplicate document IDs in the current run are rejected."""
    doc_ids = ["AAPL_10-K_2025", "AAPL_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-09-27",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        }
    ]
    with pytest.raises(ValueError, match="Duplicate document IDs"):
        validate_ingestion(doc_ids, chunks)


def test_validate_ingestion_fails_on_missing_chunks():
    """Documents with no chunks are rejected."""
    doc_ids = ["AAPL_10-K_2025", "MSFT_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-09-27",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        }
    ]
    with pytest.raises(ValueError, match="Documents with no chunks"):
        validate_ingestion(doc_ids, chunks)


def test_validate_ingestion_fails_on_duplicate_chunk_ids():
    """Duplicate chunk IDs are rejected."""
    doc_ids = ["AAPL_10-K_2025", "MSFT_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-09-27",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
        {
            "chunk_id": "AAPL_10-K_2025_c0000",  # Duplicate
            "doc_id": "MSFT_10-K_2025",
            "ticker": "MSFT",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-06-30",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
    ]
    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        validate_ingestion(doc_ids, chunks)


def test_validate_ingestion_fails_on_missing_metadata():
    """Chunks missing required metadata fields are rejected."""
    doc_ids = ["AAPL_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            # Missing period_end
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        }
    ]
    with pytest.raises(ValueError, match="missing required field"):
        validate_ingestion(doc_ids, chunks)


def test_validate_ingestion_fails_on_non_integer_fiscal_year():
    """Non-integer fiscal_year is rejected."""
    doc_ids = ["AAPL_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": "2025",  # String, not int
            "form": "10-K",
            "period_end": "2025-09-27",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        }
    ]
    with pytest.raises(ValueError, match="non-integer fiscal_year"):
        validate_ingestion(doc_ids, chunks)


def test_validate_ingestion_fails_on_non_sequential_chunk_suffixes():
    """Non-sequential chunk suffixes within a document are rejected."""
    doc_ids = ["AAPL_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-09-27",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
        {
            "chunk_id": "AAPL_10-K_2025_c0002",  # Skipped c0001
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-09-27",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
    ]
    with pytest.raises(ValueError, match="Chunk suffixes not sequential"):
        validate_ingestion(doc_ids, chunks)


def test_validate_ingestion_fails_on_missing_structural_fields():
    """Chunks missing required structural fields are rejected."""
    doc_ids = ["AAPL_10-K_2025"]
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-09-27",
            # Missing text
            "embed_text": "Test",
            "type": "text",
        }
    ]
    with pytest.raises(ValueError, match="missing structural field"):
        validate_ingestion(doc_ids, chunks)


def _valid_chunks(doc_id, ticker, fiscal_year, n):
    return [
        {
            "chunk_id": f"{doc_id}_c{i:04d}",
            "doc_id": doc_id,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "form": "10-K",
            "period_end": f"{fiscal_year}-12-31",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        }
        for i in range(n)
    ]


def test_validate_ingestion_passes_on_multi_chunk_documents():
    """Many chunks per document share one (ticker, form, fiscal_year) key and must pass."""
    chunks = _valid_chunks("AAPL_10-K_2025", "AAPL", 2025, 5) + _valid_chunks(
        "AAPL_10-K_2024", "AAPL", 2024, 3
    )
    validate_ingestion(["AAPL_10-K_2025", "AAPL_10-K_2024"], chunks)


def test_validate_ingestion_passes_with_complete_expected_coverage():
    chunks = _valid_chunks("AAPL_10-K_2024", "AAPL", 2024, 1) + _valid_chunks(
        "MSFT_10-K_2024", "MSFT", 2024, 1
    )
    validate_ingestion(
        ["AAPL_10-K_2024", "MSFT_10-K_2024"],
        chunks,
        expected_tickers=["aapl", "MSFT"],
        expected_years=[2024],
    )


def test_validate_ingestion_fails_on_missing_expected_coverage():
    chunks = _valid_chunks("AAPL_10-K_2024", "AAPL", 2024, 1)
    with pytest.raises(ValueError, match="MSFT FY2024"):
        validate_ingestion(
            ["AAPL_10-K_2024"],
            chunks,
            expected_tickers=["AAPL", "MSFT"],
            expected_years=[2024],
        )


def test_validate_ingestion_requires_both_coverage_expectations():
    chunks = _valid_chunks("AAPL_10-K_2024", "AAPL", 2024, 1)
    with pytest.raises(ValueError, match="must be provided together"):
        validate_ingestion(["AAPL_10-K_2024"], chunks, expected_tickers=["AAPL"])


def test_validate_ingestion_fails_on_duplicate_filing_key_across_documents():
    """Two documents resolving to the same (ticker, form, fiscal_year) are rejected."""
    chunks = _valid_chunks("AAPL_10-K_2025", "AAPL", 2025, 2) + _valid_chunks(
        "AAPL_10-K_2025b", "AAPL", 2025, 2
    )
    with pytest.raises(ValueError, match=r"Duplicate \(ticker, form, fiscal_year\)"):
        validate_ingestion(["AAPL_10-K_2025", "AAPL_10-K_2025b"], chunks)


def test_validate_ingestion_fails_on_inconsistent_metadata_within_document():
    chunks = _valid_chunks("AAPL_10-K_2025", "AAPL", 2025, 2)
    chunks[1]["fiscal_year"] = 2024
    with pytest.raises(ValueError, match="Inconsistent filing metadata"):
        validate_ingestion(["AAPL_10-K_2025"], chunks)


def test_validate_ingestion_fails_on_chunks_for_unknown_document():
    chunks = _valid_chunks("AAPL_10-K_2025", "AAPL", 2025, 1) + _valid_chunks(
        "MSFT_10-K_2025", "MSFT", 2025, 1
    )
    with pytest.raises(ValueError, match="not in current run"):
        validate_ingestion(["AAPL_10-K_2025"], chunks)


def test_processor_skip_embed_generates_chunks_only(tmp_path):
    """--skip-embed generates chunks without embeddings or Qdrant indexing."""
    from ingestion.processor import process

    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    chunks_dir = tmp_path / "chunks"
    raw_dir.mkdir()
    parsed_dir.mkdir()
    chunks_dir.mkdir()

    # Create a valid filing
    html = """<html><body>
      <ix:nonnumeric name="dei:EntityRegistrantName">Test Inc.</ix:nonnumeric>
      <ix:nonnumeric name="dei:TradingSymbol">TEST</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentType">10-K</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentFiscalYearFocus">2025</ix:nonnumeric>
      <ix:nonnumeric name="dei:DocumentPeriodEndDate">2025-12-31</ix:nonnumeric>
      <p>""" + ("Test content. " * 50) + """</p>
    </body></html>"""
    (raw_dir / "TEST_10-K_2025.html").write_text(html, encoding="utf-8")

    # Process with skip-embed
    out_dir = process(
        raw_dir=raw_dir,
        parsed_dir=parsed_dir,
        chunks_dir=chunks_dir,
        skip_parse=False,
        skip_embed=True,
        pseudo_headings=False,
    )

    # Chunks should be written
    chunk_files = list(out_dir.glob("*.chunks.jsonl"))
    assert len(chunk_files) == 1
    assert chunk_files[0].name == "TEST_10-K_2025.chunks.jsonl"

    # No embeddings file should exist
    assert not (out_dir / "embeddings_gpu_base.npy").exists()

    # Verify chunks have metadata
    chunks = [json.loads(line) for line in chunk_files[0].read_text(encoding="utf-8").splitlines()]
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["ticker"] == "TEST"
        assert chunk["fiscal_year"] == 2025
        assert chunk["form"] == "10-K"
        assert chunk["period_end"] == "2025-12-31"


def test_build_manifest_includes_corpus_metadata(tmp_path):
    """Manifest includes corpus and per-document metadata when parsed_dir is provided."""
    from types import SimpleNamespace

    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    # Create document metadata files
    for ticker, year in [("AAPL", 2025), ("MSFT", 2025)]:
        doc_id = f"{ticker}_10-K_{year}"
        metadata = {
            "doc_id": doc_id,
            "ticker": ticker,
            "form": "10-K",
            "fiscal_year": year,
            "period_end": f"{year}-12-31",
            "source_file": f"{doc_id}.html",
        }
        (parsed_dir / f"{doc_id}.document.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    # Create chunks with metadata
    chunks = [
        {
            "chunk_id": "AAPL_10-K_2025_c0000",
            "doc_id": "AAPL_10-K_2025",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-12-31",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
        {
            "chunk_id": "MSFT_10-K_2025_c0000",
            "doc_id": "MSFT_10-K_2025",
            "ticker": "MSFT",
            "fiscal_year": 2025,
            "form": "10-K",
            "period_end": "2025-12-31",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        },
    ]

    # Mock model
    model = SimpleNamespace(
        backend="local",
        dimension=768,
        normalized=True,
        query_prefix="",
        model_name="BAAI/bge-base-en-v1.5",
        tokenizer=SimpleNamespace(model_max_length=512),
        max_seq_length=512,
    )

    token_counts = [10, 10]
    manifest = build_manifest(
        chunks,
        token_counts,
        model,
        "BAAI/bge-base-en-v1.5",
        "2026-10-30_12-00-00",
        "data/chunks/2026-10-30_12-00-00",
        {"max_chars": 1200, "min_chars": 200, "table_max_chars": 900},
        parsed_dir=parsed_dir,
    )

    # Verify corpus metadata
    assert "corpus" in manifest
    assert manifest["corpus"]["document_count"] == 2
    assert manifest["corpus"]["tickers"] == ["AAPL", "MSFT"]
    assert manifest["corpus"]["fiscal_year_start"] == 2025
    assert manifest["corpus"]["fiscal_year_end"] == 2025
    assert manifest["corpus"]["filings_per_ticker"] == {"AAPL": 1, "MSFT": 1}

    # Verify per-document metadata
    assert "ticker" in manifest["documents"]["AAPL_10-K_2025"]
    assert manifest["documents"]["AAPL_10-K_2025"]["ticker"] == "AAPL"
    assert manifest["documents"]["AAPL_10-K_2025"]["form"] == "10-K"
    assert manifest["documents"]["AAPL_10-K_2025"]["fiscal_year"] == 2025
    assert manifest["documents"]["AAPL_10-K_2025"]["period_end"] == "2025-12-31"
    assert manifest["documents"]["AAPL_10-K_2025"]["source_file"] == "AAPL_10-K_2025.html"

    # Verify manifest without parsed_dir has no corpus section
    manifest_no_corpus = build_manifest(
        chunks,
        token_counts,
        model,
        "BAAI/bge-base-en-v1.5",
        "2026-10-30_12-00-00",
        "data/chunks/2026-10-30_12-00-00",
        {"max_chars": 1200, "min_chars": 200, "table_max_chars": 900},
        parsed_dir=None,
    )
    assert "corpus" not in manifest_no_corpus
    assert "ticker" not in manifest_no_corpus["documents"]["AAPL_10-K_2025"]


def test_embed_and_save_skip_npy_does_not_write_npy_file(tmp_path):
    """skip_npy=True generates embeddings in memory without writing .npy file."""
    out_dir = tmp_path / "chunks"
    out_dir.mkdir()

    chunks = [
        {
            "chunk_id": "TEST_c0000",
            "doc_id": "TEST",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        }
    ]

    # Mock model
    model = SimpleNamespace(
        backend="local",
        dimension=768,
        normalized=True,
        query_prefix="",
        model_name="BAAI/bge-base-en-v1.5",
        tokenizer=SimpleNamespace(model_max_length=512, encode=MagicMock(return_value=[1, 2, 3])),
        max_seq_length=512,
    )
    model.encode_documents = MagicMock(return_value=np.array([[0.1] * 768], dtype=np.float32))

    embeddings, manifest = embed_and_save(
        chunks,
        out_dir,
        model=model,
        chunking_config={"max_chars": 1200, "min_chars": 200, "table_max_chars": 900},
        skip_npy=True,
    )

    # Embeddings should be generated
    assert embeddings.shape[0] == 1

    # No .npy file should exist
    assert not (out_dir / "embeddings_gpu_base.npy").exists()

    # Manifest should still be written
    assert (out_dir / "run_manifest.json").exists()


def test_embed_and_save_with_npy_writes_npy_file(tmp_path):
    """skip_npy=False (default) writes the .npy file."""
    out_dir = tmp_path / "chunks"
    out_dir.mkdir()

    chunks = [
        {
            "chunk_id": "TEST_c0000",
            "doc_id": "TEST",
            "text": "Test",
            "embed_text": "Test",
            "type": "text",
        }
    ]

    # Mock model
    model = SimpleNamespace(
        backend="local",
        dimension=768,
        normalized=True,
        query_prefix="",
        model_name="BAAI/bge-base-en-v1.5",
        tokenizer=SimpleNamespace(model_max_length=512, encode=MagicMock(return_value=[1, 2, 3])),
        max_seq_length=512,
    )
    model.encode_documents = MagicMock(return_value=np.array([[0.1] * 768], dtype=np.float32))

    embeddings, manifest = embed_and_save(
        chunks,
        out_dir,
        model=model,
        chunking_config={"max_chars": 1200, "min_chars": 200, "table_max_chars": 900},
        skip_npy=False,
    )

    # .npy file should exist
    assert (out_dir / "embeddings_gpu_base.npy").exists()
