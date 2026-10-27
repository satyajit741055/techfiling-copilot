"""Offline tests for structure-aware chunking of parsed 10-K elements."""

import json

import pytest

from ingestion.chunking.chunker import (
    chunk_doc_id,
    chunk_document,
    concise_table_intro,
    list_doc_ids,
    load_elements,
    looks_like_heading,
    merge_tiny,
    split_long,
    split_table_markdown,
    write_chunks,
)

TABLE_MD = "| Item | 2025 | 2024 |\n|---|---|---|\n" + "\n".join(f"| Row {i} | {i * 10} | {i * 9} |" for i in range(40))


def _el(order, type_="paragraph", text="Body text.", section="Item 7", part="Part II", **extra):
    element = {
        "doc_id": "D_10-K_2025", "order": order, "type": type_, "text": text,
        "part": part, "section": section, "section_title": section,
    }
    element.update(extra)
    return element


def _table(order, markdown=TABLE_MD, intro=None, **extra):
    table = {"table_id": f"T{order}", "markdown": markdown, "intro": intro, "title": "Net sales", "units": "millions"}
    table.update(extra)
    return _el(order, "table", None, table=table)


class TestSplitLong:
    def test_short_text_is_unchanged(self):
        assert split_long("One sentence.", 100) == ["One sentence."]

    def test_splits_at_sentence_boundaries(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        parts = split_long(text, 45)
        assert len(parts) > 1 and all(p.endswith(".") for p in parts)
        assert " ".join(parts) == text

    def test_single_long_sentence_stays_intact(self):
        sentence = "word " * 100
        assert split_long(sentence.strip(), 50) == [sentence.strip()]


class TestConciseTableIntro:
    def test_empty_and_none(self):
        assert concise_table_intro(None) == "" and concise_table_intro("   ") == ""

    def test_keeps_latest_complete_sentences_in_order(self):
        intro = "Alpha is first. Beta is second. Gamma is third."
        assert concise_table_intro(intro, max_chars=32) == "Beta is second. Gamma is third."

    def test_long_last_sentence_is_not_truncated(self):
        intro = "Short one. " + "X" * 400 + " end."
        assert concise_table_intro(intro, max_chars=50) == "X" * 400 + " end."


class TestSplitTableMarkdown:
    def test_small_table_is_single_piece(self):
        md = "| a | b |\n|---|---|\n| 1 | 2 |"
        assert split_table_markdown(md, 500) == [md]

    def test_large_table_repeats_header_and_keeps_all_rows(self):
        pieces = split_table_markdown(TABLE_MD, 300)
        header = "| Item | 2025 | 2024 |\n|---|---|---|"
        assert len(pieces) > 1 and all(p.startswith(header) for p in pieces)
        rows = [line for p in pieces for line in p.split("\n")[2:]]
        assert rows == TABLE_MD.split("\n")[2:]
        assert all(len(p) <= 300 for p in pieces)

    def test_missing_separator_row_raises(self):
        with pytest.raises(StopIteration):
            split_table_markdown("| a |\n" * 50, 20)


class TestLooksLikeHeading:
    def test_short_title_before_body(self):
        assert looks_like_heading(_el(0, text="Intelligent Cloud"), _el(1))

    @pytest.mark.parametrize("text", ["Ends with period.", "Ends with colon:", "(a) enumerated", "1 numbered", "x" * 81, ""])
    def test_rejected_texts(self, text):
        assert not looks_like_heading(_el(0, text=text), _el(1))

    def test_requires_following_element_in_same_section(self):
        assert not looks_like_heading(_el(0, text="Title"), None)
        assert not looks_like_heading(_el(0, text="Title"), _el(1, section="Item 8"))
        assert not looks_like_heading(_el(0, text="Title"), _el(1, type_="heading"))

    def test_only_paragraphs_qualify(self):
        assert not looks_like_heading(_el(0, type_="heading", text="Title"), _el(1))


class TestChunkDocument:
    def test_paragraphs_are_packed_up_to_budget(self):
        els = [_el(i, text="word " * 100) for i in range(4)]
        chunks = chunk_document(els, max_chars=1000, min_chars=50)
        assert [c["element_orders"] for c in chunks] == [[0, 1], [2, 3]]
        assert all(c["type"] == "text" for c in chunks)

    def test_never_crosses_section_boundary(self):
        els = [_el(0, section="Item 1"), _el(1, section="Item 1A"), _el(2, section="Item 1A")]
        chunks = chunk_document(els)
        assert [(c["section"], c["element_orders"]) for c in chunks] == [("Item 1", [0]), ("Item 1A", [1, 2])]

    def test_never_crosses_part_boundary(self):
        els = [_el(0, part="Part I", section="Item 1"), _el(1, part="Part II", section="Item 1")]
        assert [c["part"] for c in chunk_document(els)] == ["Part I", "Part II"]

    def test_heading_after_body_starts_new_chunk_and_is_in_breadcrumb(self):
        els = [_el(0, text="First body."), _el(1, "heading", "Segment Results"), _el(2, text="Second body.")]
        chunks = chunk_document(els)
        assert [c["element_orders"] for c in chunks] == [[0], [2]]
        assert chunks[1]["headings"] == ["Segment Results"]
        assert chunks[1]["breadcrumb"].endswith("Segment Results")

    def test_headings_reset_at_new_section(self):
        els = [_el(0, "heading", "Old heading", section="Item 1"), _el(1, section="Item 1"),
               _el(2, section="Item 2"), ]
        chunks = chunk_document(els)
        assert chunks[0]["headings"] == ["Old heading"] and chunks[1]["headings"] == []

    def test_pseudo_heading_starts_new_chunk_once_min_chars_reached(self):
        els = [_el(0, text="Body sentence here. " * 15), _el(1, text="Greater China"), _el(2, text="China body text.")]
        chunks = chunk_document(els, min_chars=100)
        assert [c["element_orders"] for c in chunks] == [[0], [1, 2]]
        assert chunks[1]["text"].startswith("Greater China")

    def test_pseudo_headings_can_be_disabled(self):
        els = [_el(0, text="Body sentence here. " * 15), _el(1, text="Greater China"), _el(2, text="China body text.")]
        assert len(chunk_document(els, min_chars=100, pseudo_headings=False)) == 1

    def test_oversized_paragraph_is_split(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(60))
        chunks = chunk_document([_el(0, text=text)], max_chars=300)
        assert len(chunks) > 1 and all(c["element_orders"] == [0] for c in chunks)
        assert " ".join(c["text"] for c in chunks) == text

    def test_empty_paragraphs_are_skipped(self):
        assert chunk_document([_el(0, text="  "), _el(1, text=None)]) == []

    def test_empty_input(self):
        assert chunk_document([]) == []

    def test_table_chunk_metadata_and_intro(self):
        md = "| a | b |\n|---|---|\n| 1 | 2 |"
        els = [_el(0, text="Before."), _table(1, md, intro="Net sales were as follows. They rose.")]
        chunks = chunk_document(els)
        table = next(c for c in chunks if c["type"] == "table")
        assert table["table_id"] == "T1" and table["units"] == "millions" and table["element_orders"] == [1]
        assert table["table_context"] == "Net sales were as follows. They rose."
        assert table["text"] == f"{table['table_context']}\n\n{md}"
        assert [c["type"] for c in chunks] == ["text", "table"]

    def test_large_table_is_split_into_multiple_chunks(self):
        chunks = [c for c in chunk_document([_table(0)], table_max_chars=300) if c["type"] == "table"]
        assert len(chunks) > 1 and all(c["table_id"] == "T0" for c in chunks)

    def test_text_before_table_is_flushed_first(self):
        chunks = chunk_document([_el(0), _table(1, "| a |\n|---|\n| 1 |")])
        assert [c["element_orders"] for c in chunks] == [[0], [1]]

    def test_chunk_ids_and_embed_text(self):
        chunks = chunk_document([_el(0, section="Item 1"), _el(1, section="Item 2")])
        assert [c["chunk_id"] for c in chunks] == ["D_10-K_2025_c0000", "D_10-K_2025_c0001"]
        for c in chunks:
            assert c["embed_text"] == f"{c['breadcrumb']}\n\n{c['text']}"
            assert c["breadcrumb"].startswith("D 10-K 2025 > Part II")

    def test_every_element_is_covered_exactly_once(self):
        els = [_el(i, text=f"Paragraph {i}. " * 20, section=f"Item {i // 3}") for i in range(9)]
        orders = [o for c in chunk_document(els, max_chars=500) for o in c["element_orders"]]
        assert sorted(set(orders)) == list(range(9))


class TestMergeTiny:
    def _chunks(self, *texts, section="Item 7"):
        return [
            {"chunk_id": f"x{i}", "doc_id": "D", "section": section, "type": "text", "text": t,
             "element_orders": [i], "breadcrumb": "bc", "embed_text": f"bc\n\n{t}"}
            for i, t in enumerate(texts)
        ]

    def test_tiny_chunk_is_merged_forward_and_ids_renumbered(self):
        merged = merge_tiny(self._chunks("tiny", "x" * 300), min_chars=200, max_chars=1000)
        assert len(merged) == 1
        assert merged[0]["text"] == "tiny\n\n" + "x" * 300 and merged[0]["element_orders"] == [0, 1]
        assert merged[0]["embed_text"] == f"bc\n\n{merged[0]['text']}" and merged[0]["chunk_id"] == "D_c0000"

    def test_does_not_merge_when_over_max(self):
        assert len(merge_tiny(self._chunks("tiny", "x" * 300), min_chars=200, max_chars=100)) == 2

    def test_does_not_merge_across_sections_or_types(self):
        chunks = self._chunks("tiny", "x" * 300)
        chunks[1]["section"] = "Item 8"
        assert len(merge_tiny(chunks, min_chars=200, max_chars=1000)) == 2
        chunks = self._chunks("tiny", "x" * 300)
        chunks[1]["type"] = "table"
        assert len(merge_tiny(chunks, min_chars=200, max_chars=1000)) == 2

    def test_large_chunks_are_untouched(self):
        assert len(merge_tiny(self._chunks("x" * 300, "y" * 300), min_chars=200, max_chars=1000)) == 2


class TestFileHelpers:
    def _write_elements(self, tmp_path, doc_id, elements):
        path = tmp_path / f"{doc_id}.elements.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in elements) + "\n\n", encoding="utf-8")

    def test_load_elements_skips_blank_lines(self, tmp_path):
        self._write_elements(tmp_path, "D_10-K_2025", [_el(0), _el(1)])
        assert [e["order"] for e in load_elements("D_10-K_2025", tmp_path)] == [0, 1]

    def test_list_doc_ids_is_sorted(self, tmp_path):
        for doc_id in ("B_10-K_2025", "A_10-K_2025"):
            self._write_elements(tmp_path, doc_id, [])
        assert list_doc_ids(tmp_path) == ["A_10-K_2025", "B_10-K_2025"]

    def test_chunk_doc_id_and_write_chunks_roundtrip(self, tmp_path):
        self._write_elements(tmp_path, "D_10-K_2025", [_el(0, text="Body. " * 50), _el(1, text="More. " * 50)])
        chunks = chunk_doc_id("D_10-K_2025", tmp_path)
        out = tmp_path / "out.jsonl"
        write_chunks(chunks, out)
        assert [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] == chunks

    def test_write_chunks_preserves_unicode(self, tmp_path):
        out = tmp_path / "out.jsonl"
        write_chunks([{"text": "☐ ☒ café"}], out)
        assert "☐ ☒ café" in out.read_text(encoding="utf-8")
