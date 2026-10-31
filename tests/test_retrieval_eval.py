"""Offline tests for the Qdrant retrieval evals (in-memory Qdrant, fake embedder/reranker)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import dense_retriever_eval as dense_eval  # noqa: E402
import hybrid_rerank_eval as hre  # noqa: E402

from retriever.hybrid_retriever import SearchResult  # noqa: E402
from retriever.qdrant_retriever import (  # noqa: E402
    QdrantRetriever,
    QdrantRetrieverWithReranker,
)
from vector_store import QdrantChunkStore  # noqa: E402

DIM = 8
SECTIONS = {"Item 7": 10, "Item 8": 20}  # section -> element order
CHUNKS = [
    {
        "chunk_id": f"{ticker}_10-K_{year}_c{i:04d}",
        "doc_id": f"{ticker}_10-K_{year}",
        "ticker": ticker,
        "fiscal_year": year,
        "form": "10-K",
        "section": section,
        "type": "text",
        "element_orders": [order],
        "text": f"{ticker} {year} research and development {section}",
        "embed_text": f"{ticker} {year} research and development {section}",
    }
    for ticker in ("AAPL", "MSFT")
    for year in (2023, 2024)
    for i, (section, order) in enumerate(SECTIONS.items())
]


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class _FakeEmbedder:
    backend = "local"
    model_name = "fake"
    query_prefix = ""
    dimension = DIM

    def encode_query(self, query):
        return _unit(0)


class _Item8Reranker:
    """Ranks "Item 8" chunks first, so expected rankings are deterministic."""

    backend = "fake"
    model_name = "fake-reranker"

    def score(self, query, documents):
        return np.asarray([float(d.endswith("Item 8")) for d in documents])


def _item(doc_id: str, section: str = "Item 8", relevance: int = 3) -> dict:
    return {
        "match_by": "element_orders",
        "value": [SECTIONS[section]],
        "relevance": relevance,
        "doc_id": doc_id,
    }


def _golden(qid: str, relevant: list[dict], **scope) -> dict:
    return {
        "id": qid,
        "query": "research and development",
        "doc_id": None,
        "category": "test",
        "difficulty": "easy",
        "expected_answer": "-",
        "relevant": relevant,
        **scope,
    }


GOLDENS = [
    _golden("doc", [_item("AAPL_10-K_2023")], doc_id="AAPL_10-K_2023"),
    _golden(
        "filtered",
        [_item("MSFT_10-K_2024")],
        filters={"ticker": "MSFT", "fiscal_year": {"gte": 2024}},
    ),
    _golden(
        "grouped_known",
        [_item("AAPL_10-K_2023"), _item("MSFT_10-K_2023")],
        filters={"fiscal_year": 2023},
        group_by="ticker",
        groups=["AAPL", "MSFT"],
    ),
    _golden(
        "grouped_discovered",
        [_item("AAPL_10-K_2024", "Item 7")],
        group_by="doc_id",
        group_limit=4,
    ),
    _golden("negative", [], doc_id="MSFT_10-K_2023"),
]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def store() -> QdrantChunkStore:
    s = QdrantChunkStore(":memory:", "evaltest")
    vectors = np.stack([_unit(10 + i) for i in range(len(CHUNKS))])
    s.index_chunks(CHUNKS, vectors, {"name": "fake", "dimension": DIM})
    return s


@pytest.fixture
def pipeline(store, monkeypatch) -> QdrantRetrieverWithReranker:
    monkeypatch.setattr(
        "retriever.hybrid_retriever.create_reranker", lambda *a, **k: _Item8Reranker()
    )
    return QdrantRetrieverWithReranker(
        store, embedder=_FakeEmbedder(), retrieve_k=10, rerank_k=2, alpha=0.5
    )


@pytest.fixture
def golden_path(tmp_path) -> Path:
    return _write(tmp_path / "goldens.jsonl", GOLDENS)


def _by_id(results, k):
    return {qr.query_id: qr for qr in results.query_results[k]}


def test_rerank_eval_scores_flat_filtered_and_grouped_queries(golden_path, pipeline):
    evaluator = hre.HybridEvaluator(golden_path, mode="qdrant_rerank", retriever=pipeline)
    results = evaluator.evaluate(k_values=[1, 2], verbose=False)
    at1, at2 = _by_id(results, 1), _by_id(results, 2)

    assert at1["doc"].hit and at1["doc"].reciprocal_rank == 1.0
    assert at1["filtered"].hit and at1["filtered"].scope == {
        "ticker": "MSFT",
        "fiscal_year": {"gte": 2024},
    }

    known = at1["grouped_known"]
    assert known.scope_kind == "grouped" and known.hit and known.group_recall == 1.0
    assert known.groups_returned == ["AAPL", "MSFT"]
    assert known.retrieved_count == 2  # k is per group
    assert set(known.per_group) == {"AAPL", "MSFT"}

    # Item 7 ranks second within its group: miss at k=1, RR 0.5 at k=2
    discovered = at1["grouped_discovered"]
    assert discovered.groups_expected == ["AAPL_10-K_2024"] and not discovered.hit
    assert at2["grouped_discovered"].reciprocal_rank == 0.5
    assert at2["grouped_discovered"].group_recall == 1.0

    for qr in results.query_results[2]:
        assert qr.filter_violations == 0
        assert qr.latency_ms > 0
    assert results.overall["positive_queries"] == 4
    assert results.overall["negative_queries"] == 1
    assert results.overall["filter_violations"] == 0
    assert results.overall["latency_ms_p50"] is not None
    assert results.checks["get_chunks"]["failed_queries"] == 0
    assert results.checks["get_chunks"]["chunks"] > 0
    assert set(results.metrics_by_scope) == {"doc", "filtered", "grouped"}
    assert results.retrieval_config["collection"] == "evaltest"
    json.dumps(results.to_dict())  # serialisable


def test_grouped_hit_requires_every_expected_group(golden_path, pipeline):
    evaluator = hre.HybridEvaluator(golden_path, mode="qdrant_rerank", retriever=pipeline)
    golden = next(g for g in evaluator.golden_queries if g.id == "grouped_known")
    aapl_only = pipeline.search_grouped(
        golden.query, "ticker", groups=["AAPL"], filters=golden.scope
    )
    qr = evaluator.evaluate_query(golden, 1, hre.QueryRun(grouped=aapl_only))
    assert qr.per_group["AAPL"]["hit"] == 1.0 and qr.filter_violations == 0
    assert not qr.hit
    assert qr.group_recall == 0.5
    assert qr.reciprocal_rank == 0.5  # macro-average of 1.0 (AAPL) and 0.0 (MSFT)


def test_filter_and_group_leaks_are_counted(golden_path, pipeline):
    evaluator = hre.HybridEvaluator(golden_path, mode="qdrant_rerank", retriever=pipeline)
    goldens = {g.id: g for g in evaluator.golden_queries}
    wrong = next(c for c in CHUNKS if c["doc_id"] == "MSFT_10-K_2023")

    flat = evaluator.evaluate_query(
        goldens["doc"], 1, hre.QueryRun(results=[SearchResult(1.0, wrong)])
    )
    assert flat.filter_violations == 1

    grouped = evaluator.evaluate_query(
        goldens["grouped_known"], 1, hre.QueryRun(grouped={"AAPL": [SearchResult(1.0, wrong)]})
    )
    assert grouped.filter_violations == 1  # right year, wrong group


def test_get_chunks_check_detects_mismatch(golden_path, pipeline, monkeypatch):
    evaluator = hre.HybridEvaluator(golden_path, mode="qdrant_rerank", retriever=pipeline)
    run = hre.QueryRun(results=[SearchResult(1.0, CHUNKS[0]), SearchResult(0.5, CHUNKS[1])])
    assert evaluator.check_get_chunks(run)["ok"]
    monkeypatch.setattr(pipeline, "get_chunks", lambda ids: [CHUNKS[1], CHUNKS[0]])
    assert not evaluator.check_get_chunks(run)["ok"]


def test_tune_alpha_changes_rrf_weights_in_place(golden_path, store):
    retriever = QdrantRetriever(store, embedder=_FakeEmbedder(), alpha=0.5)
    evaluator = hre.HybridEvaluator(golden_path, mode="qdrant", retriever=retriever)
    tuned = hre.tune_alpha(evaluator, alpha_values=[0.2, 0.9], k=2)
    assert set(tuned) == {0.2, 0.9}
    assert (retriever.dense_weight, retriever.sparse_weight) == pytest.approx((0.9, 0.1))
    with pytest.raises(ValueError, match="fused mode"):
        hre.tune_alpha(
            hre.HybridEvaluator(golden_path, mode="qdrant_dense", retriever=retriever), k=2
        )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_golden("q", [{**_item("AAPL_10-K_2023"), "doc_id": None}], group_by="ticker"), "doc_id"),
        (_golden("q", [], groups=["AAPL"]), "need group_by"),
        (_golden("q", [], group_by="section"), "group_by must be one of"),
        (
            _golden("q", [], filters={"ticker": "AAPL"}, group_by="ticker", groups=["AAPL"]),
            "not both",
        ),
        (_golden("q", [], filters={"fiscal_year": {"from": 2022}}), "Range filter"),
        (_golden("q", [], filters={"ticker": []}), "empty value list"),
    ],
)
def test_golden_scope_validation(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        hre.load_golden_dataset(_write(tmp_path / "bad.jsonl", [row]))


def test_doc_group_value_parses_doc_ids():
    assert hre.doc_group_value("AAPL_10-K_2024", "doc_id") == "AAPL_10-K_2024"
    assert hre.doc_group_value("AAPL_10-K_2024", "ticker") == "AAPL"
    assert hre.doc_group_value("AAPL_10-K_2024", "fiscal_year") == 2024


def test_find_chunk_run_matches_collection(tmp_path):
    run = tmp_path / "2026-10-31_14-04-40"
    run.mkdir()
    manifest = {"chunking_config": {"max_chars": 1300}, "qdrant_collection": {"name": "c1"}}
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert hre.find_chunk_run("c1", tmp_path) == (run, manifest)
    assert hre.find_chunk_run("other", tmp_path) == (None, {})


def test_save_and_compare_include_collection(golden_path, pipeline, tmp_path, capsys):
    evaluator = hre.HybridEvaluator(golden_path, mode="qdrant_rerank", retriever=pipeline)
    results = evaluator.evaluate(k_values=[1], verbose=False)
    saved = evaluator.save_results(results, tmp_path / "out")
    assert "_qdrant_rerank_evaltest_" in saved.name
    hre.compare_runs(tmp_path / "out")
    assert "evaltest" in capsys.readouterr().out


def test_dense_eval_uses_filters_and_skips_grouped(golden_path, store, tmp_path, capsys):
    retriever = QdrantRetriever(store, mode="dense", embedder=_FakeEmbedder())
    evaluator = dense_eval.RetrieverEvaluator(golden_path, retriever=retriever)
    assert "Skipped 2 grouped queries" in capsys.readouterr().out
    assert [g.id for g in evaluator.golden_queries] == ["doc", "filtered", "negative"]

    results = evaluator.evaluate(k_values=[4], verbose=False)
    by_id = _by_id(results, 4)
    assert {c["chunk_id"].rsplit("_", 1)[0] for c in by_id["filtered"].top_chunks} == {
        "MSFT_10-K_2024"
    }
    assert results.overall["filter_violations"] == 0
    assert results.run_directory == "qdrant:evaltest"

    saved = evaluator.save_results(results, tmp_path / "dense")
    assert ":" not in saved.name and "_dense_evaltest_" in saved.name
