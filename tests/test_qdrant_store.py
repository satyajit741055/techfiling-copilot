"""Offline tests for the Qdrant vector store (in-memory mode, no server or models)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vector_store import QdrantChunkStore  # noqa: E402
from vector_store.sparse import encode_document, encode_query  # noqa: E402

DIM = 8


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


CHUNKS = [
    {"chunk_id": "A_0", "doc_id": "AAPL", "section": "Item 7", "type": "text",
     "text": "Total net sales were 416161 million", "embed_text": "Total net sales were 416161 million"},
    {"chunk_id": "A_1", "doc_id": "AAPL", "section": "Item 1A", "type": "text",
     "text": "Supply chain risk factors", "embed_text": "Supply chain risk factors"},
    {"chunk_id": "T_0", "doc_id": "TSLA", "section": "Item 7", "type": "table",
     "text": "Automotive revenue grew", "embed_text": "Automotive revenue grew"},
]
VECTORS = np.stack([_unit(i) for i in range(len(CHUNKS))])


@pytest.fixture
def store() -> QdrantChunkStore:
    s = QdrantChunkStore(":memory:", "test")
    s.index_chunks(CHUNKS, VECTORS, {"name": "fake", "dimension": DIM})
    return s


def test_sparse_encoding_is_stable_and_weighted():
    idx, vals = encode_document("sales sales revenue")
    assert len(idx) == len(vals) == 2
    assert encode_document("sales sales revenue") == (idx, vals)
    qi, qv = encode_query("Sales SALES")
    assert len(qi) == 1 and qv == [1.0]


def test_dense_search_returns_nearest_chunk(store):
    hits = store.search("x", VECTORS[1], k=1, mode="dense")
    assert hits[0][1]["chunk_id"] == "A_1"


def test_hybrid_search_uses_lexical_match(store):
    hits = store.search("416161 net sales", VECTORS[2], k=3, mode="hybrid")
    assert hits[0][1]["chunk_id"] in {"A_0", "T_0"}


def test_filters(store):
    hits = store.search("revenue", VECTORS[0], k=5, filters={"doc_id": "TSLA"}, mode="hybrid")
    assert [h[1]["chunk_id"] for h in hits] == ["T_0"]
    assert store.search("revenue", VECTORS[0], k=5, filters={"type": "table", "doc_id": "AAPL"}) == []


def test_reindex_replaces_document_and_delete(store):
    store.index_chunks(CHUNKS[:1], VECTORS[:1], {"name": "fake"})
    assert store.count() == 2
    assert store.doc_ids() == ["AAPL", "TSLA"]
    store.delete_document("TSLA")
    assert store.doc_ids() == ["AAPL"]


def test_dimension_mismatch_rejected(store):
    with pytest.raises(ValueError):
        store.ensure_collection(DIM + 1)
    assert store.index_metadata()["embedding_model"]["name"] == "fake"


FILINGS = {
    "AAPL": {"doc_id": "AAPL", "company": "Apple Inc.", "ticker": "AAPL", "fiscal_year": 2025},
    "TSLA": {"doc_id": "TSLA", "company": "Tesla, Inc.", "ticker": "TSLA", "fiscal_year": 2025},
}


def test_filing_catalog_served_from_collection_and_hidden_from_chunks():
    s = QdrantChunkStore(":memory:", "catalog")
    s.index_chunks(CHUNKS, VECTORS, {"name": "fake"}, filings=FILINGS)
    assert s.list_filings() == [FILINGS["AAPL"], FILINGS["TSLA"]]
    hit = s.search("revenue", VECTORS[2], k=1, mode="hybrid")[0][1]
    assert "filing" not in hit and hit["chunk_id"] == "T_0"
    assert all("filing" not in c for c in s.scroll_chunks())


def test_documents_without_metadata_list_doc_id_only(store):
    assert store.list_filings() == [{"doc_id": "AAPL"}, {"doc_id": "TSLA"}]


def test_load_filing_metadata(tmp_path):
    import json

    from vector_store import load_filing_metadata

    for doc_id in ("AAPL", "MSFT"):
        (tmp_path / f"{doc_id}.document.json").write_text(json.dumps({"doc_id": doc_id, "company": doc_id}))
    assert set(load_filing_metadata(tmp_path)) == {"AAPL", "MSFT"}
    assert set(load_filing_metadata(tmp_path, ["MSFT", "TSLA"])) == {"MSFT"}


def test_model_mismatch_into_existing_collection_rejected(store):
    with pytest.raises(ValueError, match="fake"):
        store.index_chunks(CHUNKS, VECTORS, {"name": "other-model"})


def test_reindex_removes_only_stale_chunks_of_that_document(store):
    changed = [{**CHUNKS[0], "chunk_id": "A_new"}]
    store.index_chunks(changed, VECTORS[:1], {"name": "fake"})
    ids = {c["chunk_id"] for c in store.scroll_chunks()}
    assert ids == {"A_new", "T_0"}
