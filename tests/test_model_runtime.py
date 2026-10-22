"""Unit tests for execution-mode settings and the model_runtime backends (no network, no models)."""

import json

import httpx
import numpy as np
import pytest

from config.settings import Settings
from model_runtime import BGE_QUERY_PREFIX, check_index_compatibility, factory
from model_runtime.huggingface_api import HuggingFaceApiEmbedder, HuggingFaceApiReranker
from model_runtime.local import LocalEmbedder, LocalReranker


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "MODEL_EXECUTION_MODE", "EMBEDDING_EXECUTION_MODE", "RERANKER_EXECUTION_MODE",
        "GENERATION_EXECUTION_MODE", "LLM_PROVIDER", "LLM_MODEL", "HUGGINGFACE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# ----------------------------------------------------------------------------- settings


def test_local_mode_defaults_to_ollama_qwen():
    s = make_settings()
    assert (s.effective_embedding_mode, s.effective_reranker_mode, s.effective_generation_mode) == (
        "local", "local", "local",
    )
    assert s.effective_llm_provider == "ollama"
    assert s.effective_model == "qwen2.5-7b-instruct:latest"


def test_huggingface_api_mode_switches_generation_to_hf_qwen():
    s = make_settings(MODEL_EXECUTION_MODE="huggingface_api", HUGGINGFACE_API_KEY="hf_test")
    assert s.effective_embedding_mode == "huggingface_api"
    assert s.effective_llm_provider == "huggingface"
    assert s.effective_model == "Qwen/Qwen2.5-7B-Instruct"


def test_component_override_and_blank_override():
    s = make_settings(
        MODEL_EXECUTION_MODE="huggingface_api",
        RERANKER_EXECUTION_MODE="local",
        GENERATION_EXECUTION_MODE="",
        HUGGINGFACE_API_KEY="hf_test",
    )
    assert s.effective_reranker_mode == "local"
    assert s.GENERATION_EXECUTION_MODE is None
    assert s.effective_generation_mode == "huggingface_api"


def test_hf_api_mode_requires_api_key():
    with pytest.raises(ValueError, match="HUGGINGFACE_API_KEY"):
        make_settings(EMBEDDING_EXECUTION_MODE="huggingface_api")


def test_llm_model_overrides_provider_default():
    s = make_settings(LLM_MODEL="custom-model")
    assert s.model_for_provider("huggingface") == "custom-model"


# ----------------------------------------------------------------------------- factory


def test_factory_selects_backend_by_mode(monkeypatch):
    created = {}

    class FakeLocal:
        backend = "local"

        def __init__(self, model_name, **kwargs):
            created["local"] = (model_name, kwargs["device"])

    monkeypatch.setattr(factory, "LocalEmbedder", FakeLocal)
    s = make_settings(HUGGINGFACE_API_KEY="hf_test")

    factory.create_embedder("BAAI/bge-base-en-v1.5", device="cpu", settings=s)
    assert created["local"] == ("BAAI/bge-base-en-v1.5", "cpu")

    api = factory.create_embedder("BAAI/bge-base-en-v1.5", mode="huggingface_api", settings=s)
    assert isinstance(api, HuggingFaceApiEmbedder)

    reranker = factory.create_reranker("cross-encoder/x", mode="huggingface_api", settings=s)
    assert isinstance(reranker, HuggingFaceApiReranker)
    assert reranker.url == "https://router.huggingface.co/hf-inference/models/cross-encoder/x"


# ----------------------------------------------------------------------------- local backends


class FakeSentenceTransformer:
    max_seq_length = 512
    tokenizer = object()

    def __init__(self):
        self.calls = []

    def get_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append(texts)
        n = 1 if isinstance(texts, str) else len(texts)
        out = np.ones((n, 3), dtype=np.float64)
        return out[0] if isinstance(texts, str) else out


def test_local_embedder_applies_query_prefix():
    model = FakeSentenceTransformer()
    embedder = LocalEmbedder("bge", device="cpu", cache_dir=".", model=model)
    vector = embedder.encode_query("revenue?")
    assert model.calls[0] == BGE_QUERY_PREFIX + "revenue?"
    assert vector.dtype == np.float32 and embedder.dimension == 3


def test_local_reranker_scores_pairs():
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [len(doc) for _, doc in pairs]

    reranker = LocalReranker("ce", device="cpu", cache_dir=".", model=FakeCrossEncoder())
    assert reranker.score("q", ["a", "abc"]).tolist() == [1.0, 3.0]


# ----------------------------------------------------------------------------- HF API backends


class FakeInferenceClient:
    def __init__(self, dim=4):
        self.dim = dim
        self.batches = []

    def feature_extraction(self, texts, model, normalize):
        self.batches.append(list(texts))
        return np.full((len(texts), self.dim), 2.0)


def test_hf_embedder_normalizes_and_batches():
    client = FakeInferenceClient()
    embedder = HuggingFaceApiEmbedder("bge", api_key="k", client=client)
    vectors = embedder.encode_documents([f"t{i}" for i in range(40)], batch_size=64)
    assert vectors.shape == (40, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert [len(b) for b in client.batches] == [32, 8]
    embedder.encode_query("q")
    assert client.batches[-1] == [BGE_QUERY_PREFIX + "q"]


def test_hf_embedder_rejects_token_level_output():
    class TokenLevelClient:
        def feature_extraction(self, texts, model, normalize):
            return np.ones((len(texts), 5, 4))

    embedder = HuggingFaceApiEmbedder("bge", api_key="k", client=TokenLevelClient())
    with pytest.raises(ValueError, match="pooled"):
        embedder.encode_query("q")


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_hf_reranker_pair_format():
    def handler(request):
        inputs = json.loads(request.content)["inputs"]
        return httpx.Response(200, json=[{"label": "LABEL_0", "score": len(i["text_pair"])} for i in inputs])

    reranker = HuggingFaceApiReranker(
        "ce", api_key="k", url="https://x/models/{model}", client=_mock_client(handler)
    )
    assert reranker.score("q", ["aa", "a", "aaa"]).tolist() == [2.0, 1.0, 3.0]


def test_hf_reranker_tei_format_reorders_by_index():
    def handler(request):
        body = json.loads(request.content)
        ranked = sorted(range(len(body["texts"])), key=lambda i: -len(body["texts"][i]))
        return httpx.Response(200, json=[{"index": i, "score": len(body["texts"][i])} for i in ranked])

    reranker = HuggingFaceApiReranker(
        "ce", api_key="k", url="https://endpoint/rerank", client=_mock_client(handler)
    )
    assert reranker.tei_format
    assert reranker.score("q", ["aa", "a", "aaa"]).tolist() == [2.0, 1.0, 3.0]


def test_hf_reranker_raises_on_http_error():
    reranker = HuggingFaceApiReranker(
        "ce", api_key="k", url="https://x/{model}",
        client=_mock_client(lambda request: httpx.Response(503, text="loading")),
    )
    with pytest.raises(RuntimeError, match="503"):
        reranker.score("q", ["a"])


# ----------------------------------------------------------------------------- index compatibility


class StubEmbedder:
    def __init__(self, name="bge", backend="local", dimension=768):
        self.model_name, self.backend, self.dimension = name, backend, dimension


def test_index_compatibility_checks():
    manifest = {"embedding_model": {"name": "bge", "dimension": 768}}
    check_index_compatibility(manifest, StubEmbedder(), 768)
    with pytest.raises(ValueError, match="built with"):
        check_index_compatibility(manifest, StubEmbedder(name="other"), 768)
    with pytest.raises(ValueError, match="dimension"):
        check_index_compatibility(manifest, StubEmbedder(dimension=384), 768)


def test_index_backend_mismatch_only_warns(capsys):
    manifest = {"embedding_model": {"name": "bge"}}
    check_index_compatibility(manifest, StubEmbedder(backend="huggingface_api"), 768)
    assert "re-embedding is recommended" in capsys.readouterr().out
