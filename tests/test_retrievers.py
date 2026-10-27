"""Offline tests for the numpy retrievers (min-max hybrid, RRF) and the reranker wrapper."""

import json

import numpy as np
import pytest

from retriever.bm25s_rrf_retriever import BM25SRRFRetriever, reciprocal_rank_fusion
from retriever.hybrid_retriever import (
    BM25,
    HybridRetriever,
    HybridRetrieverWithReranker,
    Reranker,
    SearchResult,
)

TEXTS = [
    ("A_10-K_2025", "text", "Item 7", "apple iphone net sales increased"),
    ("A_10-K_2025", "table", "Item 8", "apple total assets balance sheet"),
    ("A_10-K_2025", "text", "Item 1A", "supply chain risk factors"),
    ("B_10-K_2026", "text", "Item 7", "microsoft cloud revenue growth"),
    ("B_10-K_2026", "table", "Item 8", "microsoft total assets balance sheet"),
]
VECTORS = np.asarray(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.7, 0.7, 0, 0]], dtype=np.float32
)
VECTORS = VECTORS / np.linalg.norm(VECTORS, axis=1, keepdims=True)


class FakeEmbedder:
    backend = "local"
    model_name = "fake-embedder"
    query_prefix = ""
    dimension = 4
    tokenizer = None
    max_seq_length = 512

    def __init__(self, vector=(1, 0, 0, 0)):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.vector = self.vector / np.linalg.norm(self.vector)

    def encode_query(self, query):
        return self.vector

    def encode_documents(self, texts, batch_size=64):
        raise NotImplementedError


class FakeRerankBackend:
    backend = "fake"
    model_name = "fake-reranker"

    def score(self, query, documents):
        return np.asarray([float(len(d)) for d in documents])


@pytest.fixture
def run_dir(tmp_path):
    chunks = [
        {
            "chunk_id": f"c{i}",
            "doc_id": doc,
            "type": type_,
            "section": section,
            "text": text,
            "embed_text": text,
        }
        for i, (doc, type_, section, text) in enumerate(TEXTS)
    ]
    for doc in ("A_10-K_2025", "B_10-K_2026"):
        lines = [json.dumps(c) for c in chunks if c["doc_id"] == doc]
        (tmp_path / f"{doc}.chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.save(tmp_path / "embeddings_gpu_base.npy", VECTORS)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"embedding_model": {"name": "fake-embedder", "backend": "local"}}),
        encoding="utf-8",
    )
    return tmp_path


def _ids(results):
    return [r.chunk["chunk_id"] for r in results]


class TestBM25:
    def test_exact_term_ranks_first(self):
        bm25 = BM25().fit(["cat sat", "dog barked loudly", "cat and dog"])
        hits = bm25.search("barked", k=3)
        assert hits[0][0] == 1 and hits[0][1] > hits[1][1]

    def test_doc_indices_restrict_search(self):
        bm25 = BM25().fit(["cat sat", "dog barked loudly", "cat and dog"])
        assert {i for i, _ in bm25.search("dog", k=5, doc_indices=[0, 2])} == {0, 2}

    def test_query_without_tokens_scores_zero(self):
        scores = BM25().fit(["a b", "c d"]).score_batch("!!!")
        assert scores.tolist() == [0.0, 0.0]

    def test_tokenizer_lowercases_and_keeps_numbers(self):
        assert BM25.tokenize("AAPL 416161 Net-Sales")[0] == ["aapl", "416161", "net", "sales"]


class TestNormalizeScores:
    def test_min_max(self):
        assert HybridRetriever._normalize_scores(np.asarray([1.0, 2.0, 3.0])).tolist() == [
            0.0,
            0.5,
            1.0,
        ]

    def test_edge_cases(self):
        assert HybridRetriever._normalize_scores(np.asarray([])).size == 0
        assert HybridRetriever._normalize_scores(np.asarray([7.0])).tolist() == [1.0]
        assert HybridRetriever._normalize_scores(np.asarray([2.0, 2.0])).tolist() == [0.5, 0.5]


class TestHybridRetriever:
    def _retriever(self, run_dir, vector=(1, 0, 0, 0), alpha=0.5):
        return HybridRetriever(run_dir, alpha=alpha, embedder=FakeEmbedder(vector))

    def test_loads_chunks_in_filename_order(self, run_dir):
        assert [c["chunk_id"] for c in self._retriever(run_dir).chunks] == [
            "c0",
            "c1",
            "c2",
            "c3",
            "c4",
        ]

    def test_dense_only_follows_embedding_similarity(self, run_dir):
        results = self._retriever(run_dir, alpha=1.0).search("anything", k=2)
        assert _ids(results)[0] == "c0" and results[0].score == pytest.approx(1.0)

    def test_bm25_only_follows_keywords(self, run_dir):
        results = self._retriever(run_dir, alpha=0.0).search("cloud revenue", k=1)
        assert _ids(results) == ["c3"]

    def test_hybrid_combines_both_signals(self, run_dir):
        retriever = self._retriever(run_dir, vector=(1, 0, 0, 0), alpha=0.5)
        assert set(_ids(retriever.search("microsoft cloud revenue", k=2))) == {"c0", "c3"}

    def test_alpha_override_per_search(self, run_dir):
        retriever = self._retriever(run_dir, alpha=1.0)
        assert _ids(retriever.search("cloud revenue", k=1, alpha=0.0)) == ["c3"]

    def test_doc_filter(self, run_dir):
        results = self._retriever(run_dir).search("assets", k=10, doc_id="B_10-K_2026")
        assert {r.chunk["doc_id"] for r in results} == {"B_10-K_2026"} and len(results) == 2

    def test_type_section_and_extra_filters(self, run_dir):
        retriever = self._retriever(run_dir)
        assert {r.chunk["type"] for r in retriever.search("assets", k=10, type_="table")} == {
            "table"
        }
        assert _ids(retriever.search("risk", k=10, section="Item 1A")) == ["c2"]
        assert _ids(retriever.search("x", k=10, filters={"chunk_id": "c4"})) == ["c4"]

    def test_unmatched_filter_returns_empty(self, run_dir):
        assert self._retriever(run_dir).search("x", doc_id="NOPE") == []

    def test_k_larger_than_corpus(self, run_dir):
        assert len(self._retriever(run_dir).search("x", k=100)) == 5

    def test_results_sorted_descending(self, run_dir):
        scores = [r.score for r in self._retriever(run_dir).search("apple assets", k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_scores_breakdown(self, run_dir):
        rows = self._retriever(run_dir).search_with_scores("apple", k=2)
        assert {"hybrid_score", "dense_score", "bm25_score", "chunk_id"} <= rows[0].keys()
        assert rows[0]["hybrid_score"] >= rows[1]["hybrid_score"]

    @pytest.mark.parametrize("kwargs", [{"query": "  "}, {"query": "x", "k": 0}])
    def test_invalid_arguments(self, run_dir, kwargs):
        with pytest.raises(ValueError):
            self._retriever(run_dir).search(**kwargs)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_invalid_alpha(self, run_dir, alpha):
        with pytest.raises(ValueError, match="alpha"):
            HybridRetriever(run_dir, alpha=alpha, embedder=FakeEmbedder())

    def test_embedding_count_mismatch_is_rejected(self, run_dir):
        np.save(run_dir / "embeddings_gpu_base.npy", VECTORS[:3])
        with pytest.raises(ValueError, match="counts differ"):
            HybridRetriever(run_dir, embedder=FakeEmbedder())

    def test_wrong_embedding_model_is_rejected(self, run_dir):
        embedder = FakeEmbedder()
        embedder.model_name = "other-model"
        with pytest.raises(ValueError, match="re-embed"):
            HybridRetriever(run_dir, embedder=embedder)

    def test_missing_run_directory(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            HybridRetriever(tmp_path / "missing", embedder=FakeEmbedder())


class TestReciprocalRankFusion:
    def test_weighted_scores(self):
        fused = reciprocal_rank_fusion(
            [(np.asarray([5, 7]), 1.0), (np.asarray([7, 9]), 2.0)], rrf_k=60
        )
        assert fused[5] == pytest.approx(1 / 61)
        assert fused[7] == pytest.approx(1 / 62 + 2 / 61)
        assert fused[9] == pytest.approx(2 / 62)

    def test_empty(self):
        assert reciprocal_rank_fusion([]) == {}


class TestBM25SRRFRetriever:
    def _retriever(self, run_dir, **kwargs):
        return BM25SRRFRetriever(run_dir, embedder=FakeEmbedder(), **kwargs)

    def test_agreement_between_signals_wins(self, run_dir):
        results = self._retriever(run_dir).search("apple iphone net sales", k=3)
        assert _ids(results)[0] == "c0"
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_weights_shift_ranking(self, run_dir):
        bm25_heavy = self._retriever(run_dir, dense_weight=0.0, bm25_weight=1.0)
        assert _ids(bm25_heavy.search("cloud revenue", k=1)) == ["c3"]
        dense_heavy = self._retriever(run_dir, dense_weight=1.0, bm25_weight=0.0)
        assert _ids(dense_heavy.search("cloud revenue", k=1)) == ["c0"]

    def test_filters(self, run_dir):
        results = self._retriever(run_dir).search(
            "assets", k=10, doc_id="A_10-K_2025", type_="table"
        )
        assert _ids(results) == ["c1"]

    def test_empty_when_nothing_matches(self, run_dir):
        assert self._retriever(run_dir).search("x", doc_id="NOPE") == []

    def test_extra_kwargs_such_as_alpha_are_ignored(self, run_dir):
        assert self._retriever(run_dir).search("apple", k=1, alpha=0.3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dense_k": 0},
            {"bm25_k": 0},
            {"rrf_k": -1},
            {"dense_weight": -1.0},
            {"dense_weight": 0.0, "bm25_weight": 0.0},
        ],
    )
    def test_invalid_parameters(self, run_dir, kwargs):
        with pytest.raises(ValueError):
            self._retriever(run_dir, **kwargs)


class TestReranker:
    def _reranker(self, monkeypatch):
        monkeypatch.setattr(
            "retriever.hybrid_retriever.create_reranker", lambda *a, **k: FakeRerankBackend()
        )
        return Reranker("fake-reranker")

    def test_reorders_by_cross_encoder_score(self, monkeypatch):
        results = [
            SearchResult(0.9, {"text": "short"}),
            SearchResult(0.1, {"text": "a much longer text"}),
        ]
        ranked = self._reranker(monkeypatch).rerank("q", results)
        assert [r.chunk["text"] for r in ranked] == ["a much longer text", "short"]
        assert ranked[0].score == float(len("a much longer text"))

    def test_k_truncates_and_embed_text_is_preferred(self, monkeypatch):
        chunks = [
            {"text": "t", "embed_text": "e" * 5},
            {"text": "t", "embed_text": "e" * 9},
            {"text": "t"},
        ]
        ranked = self._reranker(monkeypatch).rerank("q", chunks, k=2)
        assert [r.score for r in ranked] == [9.0, 5.0]

    def test_empty_input(self, monkeypatch):
        assert self._reranker(monkeypatch).rerank("q", []) == []


class TestHybridRetrieverWithReranker:
    def _pipeline(self, run_dir, monkeypatch, **kwargs):
        monkeypatch.setattr(
            "retriever.hybrid_retriever.create_embedder", lambda **k: FakeEmbedder()
        )
        monkeypatch.setattr(
            "retriever.hybrid_retriever.create_reranker", lambda *a, **k: FakeRerankBackend()
        )
        return HybridRetrieverWithReranker(run_dir, **kwargs)

    def test_returns_reranked_top_k_with_filter(self, run_dir, monkeypatch):
        pipeline = self._pipeline(run_dir, monkeypatch, retrieve_k=10, rerank_k=2)
        results = pipeline.search("assets", doc_id="A_10-K_2025")
        assert len(results) == 2 and {r.chunk["doc_id"] for r in results} == {"A_10-K_2025"}
        lengths = [len(r.chunk["embed_text"]) for r in results]
        assert lengths == sorted(lengths, reverse=True)

    def test_k_overrides_rerank_k(self, run_dir, monkeypatch):
        pipeline = self._pipeline(run_dir, monkeypatch, retrieve_k=10, rerank_k=2)
        assert len(pipeline.search("assets", k=4)) == 4

    def test_no_candidates_skips_reranking(self, run_dir, monkeypatch):
        assert self._pipeline(run_dir, monkeypatch).search("x", doc_id="NOPE") == []

    def test_exposes_underlying_artifacts(self, run_dir, monkeypatch):
        pipeline = self._pipeline(run_dir, monkeypatch)
        assert len(pipeline.chunks) == 5 and pipeline.run_directory == run_dir.resolve()
        assert pipeline.reranker_model_name == "fake-reranker"
        assert pipeline.manifest["embedding_model"]["name"] == "fake-embedder"
