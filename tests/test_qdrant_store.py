"""Offline tests for the Qdrant vector store (in-memory mode, no server or models)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vector_store import QdrantChunkStore  # noqa: E402
from vector_store.qdrant_store import point_id  # noqa: E402
from vector_store.sparse import encode_document, encode_query  # noqa: E402

DIM = 8


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


CHUNKS = [
    {
        "chunk_id": "A_0",
        "doc_id": "AAPL",
        "section": "Item 7",
        "type": "text",
        "text": "Total net sales were 416161 million",
        "embed_text": "Total net sales were 416161 million",
    },
    {
        "chunk_id": "A_1",
        "doc_id": "AAPL",
        "section": "Item 1A",
        "type": "text",
        "text": "Supply chain risk factors",
        "embed_text": "Supply chain risk factors",
    },
    {
        "chunk_id": "T_0",
        "doc_id": "TSLA",
        "section": "Item 7",
        "type": "table",
        "text": "Automotive revenue grew",
        "embed_text": "Automotive revenue grew",
    },
]
VECTORS = np.stack([_unit(i) for i in range(len(CHUNKS))])


@pytest.fixture
def store() -> QdrantChunkStore:
    s = QdrantChunkStore(":memory:", "test")
    s.index_chunks(CHUNKS, VECTORS, {"name": "fake", "dimension": DIM})
    return s


def test_on_disk_store_persists_across_reopen(tmp_path):
    """path= opens an embedded on-disk store; data survives closing and reopening it."""
    s = QdrantChunkStore("http://unused:6333", "disk", path=tmp_path / "qdrant")
    s.index_chunks(CHUNKS, VECTORS, {"name": "fake", "dimension": DIM})
    s.client.close()

    reopened = QdrantChunkStore("http://unused:6333", "disk", path=tmp_path / "qdrant")
    assert reopened.count() == len(CHUNKS)
    assert reopened.index_metadata()["embedding_model"]["name"] == "fake"
    hits = reopened.search(
        "supply chain", VECTORS[1], k=1, mode="hybrid", filters={"doc_id": "AAPL"}
    )
    assert hits[0][1]["chunk_id"] == "A_1"
    reopened.client.close()


def test_sparse_encoding_is_stable_and_weighted():
    idx, vals = encode_document("sales sales revenue")
    assert len(idx) == len(vals) == 2
    assert encode_document("sales sales revenue") == (idx, vals)
    qi, qv = encode_query("Sales SALES")
    assert len(qi) == 1 and qv == [1.0]


def test_bm25_length_normalisation_uses_average():
    from vector_store.sparse import average_tokens

    assert average_tokens(["a b", "a b c d"]) == 3.0
    short = encode_document("sales " * 4, avg_doc_tokens=4.0)[1][0]
    long_avg = encode_document("sales " * 4, avg_doc_tokens=400.0)[1][0]
    assert short < long_avg  # same text is penalised more when it is long relative to avgdl


def test_new_collection_records_default_avg_doc_tokens(store):
    from vector_store.sparse import AVG_DOC_TOKENS

    assert store.index_metadata()["bm25"] == {
        "avg_doc_tokens": AVG_DOC_TOKENS,
        "k1": 1.5,
        "b": 0.75,
    }


def test_explicit_avg_doc_tokens_is_recorded_and_used_for_encoding():
    s = QdrantChunkStore(":memory:", "tuned")
    s.index_chunks(CHUNKS, VECTORS, {"name": "fake"}, avg_doc_tokens=4.0)
    assert s.avg_doc_tokens() == 4.0
    stored = s.client.retrieve(s.collection, [point_id("A_0")], with_vectors=True)[0]
    expected = encode_document(CHUNKS[0]["embed_text"], 4.0)[1]
    assert stored.vector["bm25"].values == pytest.approx(expected)


def test_existing_collection_keeps_its_avg_doc_tokens(store):
    before = store.avg_doc_tokens()
    store.index_chunks(CHUNKS[:1], VECTORS[:1], {"name": "fake"}, avg_doc_tokens=4.0)
    assert store.avg_doc_tokens() == before


def test_dense_search_returns_nearest_chunk(store):
    hits = store.search("x", VECTORS[1], k=1, mode="dense")
    assert hits[0][1]["chunk_id"] == "A_1"


def test_hybrid_search_uses_lexical_match(store):
    hits = store.search("416161 net sales", VECTORS[2], k=3, mode="hybrid")
    assert hits[0][1]["chunk_id"] in {"A_0", "T_0"}


def test_filters(store):
    hits = store.search("revenue", VECTORS[0], k=5, filters={"doc_id": "TSLA"}, mode="hybrid")
    assert [h[1]["chunk_id"] for h in hits] == ["T_0"]
    assert (
        store.search("revenue", VECTORS[0], k=5, filters={"type": "table", "doc_id": "AAPL"}) == []
    )


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
        (tmp_path / f"{doc_id}.document.json").write_text(
            json.dumps({"doc_id": doc_id, "company": doc_id})
        )
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


class _FakeEmbedder:
    backend = "local"
    model_name = "fake"
    query_prefix = ""
    dimension = DIM

    def encode_query(self, query):
        return VECTORS[0]


FILING_CHUNKS = [
    {
        "chunk_id": f"{ticker}_10-K_{year}_c{i:04d}",
        "doc_id": f"{ticker}_10-K_{year}",
        "ticker": ticker,
        "fiscal_year": year,
        "form": "10-K",
        "section": section,
        "type": "text",
        "text": f"{ticker} {year} research and development {section}",
        "embed_text": f"{ticker} {year} research and development {section}",
    }
    for ticker in ("AAPL", "MSFT", "TSLA")
    for year in (2023, 2024)
    for i, section in enumerate(("Item 7", "Item 8"))
]
FILING_VECTORS = np.stack([_unit(100 + i) for i in range(len(FILING_CHUNKS))])


@pytest.fixture
def filings_store() -> QdrantChunkStore:
    s = QdrantChunkStore(":memory:", "filings")
    s.index_chunks(FILING_CHUNKS, FILING_VECTORS, {"name": "fake", "dimension": DIM})
    return s


def _all(store, filters):
    hits = store.search("research", FILING_VECTORS[0], k=100, filters=filters, mode="dense")
    return {h[1]["chunk_id"] for h in hits}


@pytest.mark.parametrize(
    "filters",
    [
        {"ticker": "MSFT"},
        {"ticker": ["AAPL", "TSLA"]},
        {"fiscal_year": 2024},
        {"fiscal_year": [2023]},
        {"fiscal_year": {"gte": 2024}},
        {"fiscal_year": {"gt": 2023, "lte": 2024}, "ticker": ["MSFT"], "section": "Item 8"},
        {"ticker": "NVDA"},
    ],
)
def test_scalar_list_and_range_filters_match_in_memory_semantics(filings_store, filters):
    from vector_store.filters import matching_indices, merge_filters

    expected = {
        FILING_CHUNKS[i]["chunk_id"]
        for i in matching_indices(FILING_CHUNKS, merge_filters(filters))
    }
    assert _all(filings_store, filters) == expected


def test_range_filter_selects_years(filings_store):
    years = {cid.split("_")[2] for cid in _all(filings_store, {"fiscal_year": {"lte": 2023}})}
    assert years == {"2023"}


@pytest.mark.parametrize(
    "filters, message",
    [
        ({"ticker": []}, "empty"),
        ({"fiscal_year": {}}, "Range"),
        ({"fiscal_year": {"from": 2020}}, "Range"),
        ({"fiscal_year": {"gte": "2020"}}, "numeric"),
    ],
)
def test_invalid_filters_rejected(filings_store, filters, message):
    with pytest.raises(ValueError, match=message):
        filings_store.search("x", FILING_VECTORS[0], filters=filters)


def test_none_filter_values_are_dropped():
    from vector_store.filters import merge_filters

    assert merge_filters({"ticker": None, "form": "10-K"}, doc_id=None, type="text") == {
        "form": "10-K",
        "type": "text",
    }


def test_get_chunks_returns_request_order_and_skips_unknown(filings_store):
    ids = ["MSFT_10-K_2024_c0001", "missing", "AAPL_10-K_2023_c0000", "MSFT_10-K_2024_c0001"]
    chunks = filings_store.get_chunks(ids)
    assert [c["chunk_id"] for c in chunks] == ["MSFT_10-K_2024_c0001", "AAPL_10-K_2023_c0000"]
    assert chunks[0]["ticker"] == "MSFT" and "filing" not in chunks[0]
    assert filings_store.get_chunks([]) == []


def test_grouped_search_with_known_groups_returns_every_group(filings_store):
    grouped = filings_store.search_grouped(
        "research and development",
        FILING_VECTORS[0],
        "ticker",
        groups=["MSFT", "AAPL", "NVDA"],
        group_size=2,
        filters={"fiscal_year": 2024},
    )
    assert list(grouped) == ["MSFT", "AAPL", "NVDA"]
    assert grouped["NVDA"] == []
    for ticker in ("MSFT", "AAPL"):
        assert len(grouped[ticker]) == 2
        assert {(c["ticker"], c["fiscal_year"]) for _, c in grouped[ticker]} == {(ticker, 2024)}


def test_grouped_search_discovers_groups(filings_store):
    grouped = filings_store.search_grouped(
        "research", FILING_VECTORS[0], "doc_id", group_size=1, group_limit=4
    )
    assert len(grouped) == 4
    assert all(len(hits) == 1 and hits[0][1]["doc_id"] == doc for doc, hits in grouped.items())
    by_year = filings_store.search_grouped(
        "research", FILING_VECTORS[0], "fiscal_year", group_size=3, filters={"ticker": "TSLA"}
    )
    assert set(by_year) == {2023, 2024}
    assert all(c["ticker"] == "TSLA" for hits in by_year.values() for _, c in hits)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"group_by": "section"}, "group_by"),
        ({"group_by": "ticker", "groups": []}, "empty"),
        ({"group_by": "ticker", "groups": ["AAPL"], "filters": {"ticker": "AAPL"}}, "either"),
    ],
)
def test_grouped_search_rejects_bad_arguments(filings_store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        filings_store.search_grouped("x", FILING_VECTORS[0], **kwargs)


def test_bare_strings_rejected_for_groups_and_chunk_ids(filings_store):
    with pytest.raises(TypeError, match="groups"):
        filings_store.search_grouped("x", FILING_VECTORS[0], "ticker", groups="AAPL")
    with pytest.raises(TypeError, match="chunk_ids"):
        filings_store.get_chunks("AAPL_10-K_2023_c0000")


def test_payload_indexes_created_once_on_server():
    from unittest.mock import MagicMock

    from qdrant_client import models

    client = MagicMock()
    client.get_collection.return_value.payload_schema = {"doc_id": object()}
    s = QdrantChunkStore("http://unused", "c", client=client)
    created = s.ensure_payload_indexes()
    assert created == ["ticker", "form", "section", "type", "fiscal_year"]
    calls = {c.args[1]: c.args[2] for c in client.create_payload_index.call_args_list}
    assert calls["fiscal_year"] == models.PayloadSchemaType.INTEGER
    assert calls["ticker"] == models.PayloadSchemaType.KEYWORD


def test_payload_indexes_skipped_in_embedded_mode(filings_store):
    assert filings_store.ensure_payload_indexes() == []


class _Item8Reranker:
    backend = "fake"
    model_name = "fake-reranker"

    def __init__(self):
        self.calls = 0

    def score(self, query, documents):
        self.calls += 1
        return np.asarray([float(d.endswith("Item 8")) for d in documents])


def test_qdrant_pipeline_grouped_rerank_and_get_chunks(filings_store, monkeypatch):
    from retriever.qdrant_retriever import QdrantRetrieverWithReranker

    backend = _Item8Reranker()
    monkeypatch.setattr("retriever.hybrid_retriever.create_reranker", lambda *a, **k: backend)
    pipeline = QdrantRetrieverWithReranker(
        filings_store, embedder=_FakeEmbedder(), retrieve_k=4, rerank_k=1, alpha=0.5
    )
    grouped = pipeline.search_grouped(
        "research and development", "ticker", groups=["AAPL", "TSLA"], filters={"fiscal_year": 2023}
    )
    assert backend.calls == 1  # all groups scored in one reranker call
    assert {g: [r.chunk["chunk_id"] for r in hits] for g, hits in grouped.items()} == {
        "AAPL": ["AAPL_10-K_2023_c0001"],
        "TSLA": ["TSLA_10-K_2023_c0001"],
    }
    flat = pipeline.search(
        "research", k=10, filters={"ticker": ["MSFT"], "fiscal_year": {"gte": 2024}}
    )
    assert {r.chunk["doc_id"] for r in flat} == {"MSFT_10-K_2024"}
    assert [c["chunk_id"] for c in pipeline.get_chunks(["TSLA_10-K_2024_c0000"])] == [
        "TSLA_10-K_2024_c0000"
    ]


def test_retriever_rrf_weights_follow_alpha(store):
    from retriever.qdrant_retriever import QdrantRetriever

    default = QdrantRetriever(store, embedder=_FakeEmbedder(), alpha=0.85)
    assert (default.dense_weight, default.sparse_weight) == (0.85, pytest.approx(0.15))
    explicit = QdrantRetriever(store, embedder=_FakeEmbedder(), dense_weight=2.0, sparse_weight=0.5)
    assert (explicit.dense_weight, explicit.sparse_weight) == (2.0, 0.5)
    with pytest.raises(ValueError):
        QdrantRetriever(store, embedder=_FakeEmbedder(), alpha=1.5)
    assert default.search("net sales", k=2)
