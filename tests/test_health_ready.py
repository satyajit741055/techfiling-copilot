"""Offline tests for /health/ready in every execution mode, provider and vector store."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_filings, get_rag_service
from api.main import app
from api.routers import health
from config.settings import LLMProvider

LOCAL, HF_API = "local", "huggingface_api"
SECRET = "super-secret-value-123"


@pytest.fixture
def client():
    app.dependency_overrides[get_filings] = lambda: [{"doc_id": "AAPL_10-K_2025"}]
    app.dependency_overrides[get_rag_service] = lambda: type("R", (), {"model_name": "m"})()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def modes(monkeypatch):
    """Set the execution modes, LLM provider and vector store in one call."""

    def apply(*, embedding=LOCAL, reranker=LOCAL, generation=LOCAL, provider=LLMProvider("ollama"), store="numpy"):
        s = health.settings
        monkeypatch.setattr(s, "MODEL_EXECUTION_MODE", LOCAL)
        monkeypatch.setattr(s, "EMBEDDING_EXECUTION_MODE", embedding)
        monkeypatch.setattr(s, "RERANKER_EXECUTION_MODE", reranker)
        monkeypatch.setattr(s, "GENERATION_EXECUTION_MODE", generation)
        monkeypatch.setattr(s, "LLM_PROVIDER", provider)
        monkeypatch.setattr(s, "VECTOR_STORE", store)

    return apply


@pytest.fixture
def probes(monkeypatch):
    """Replace every external probe; returns the list of probes that were called."""
    called: list[str] = []

    def stub(name, result):
        def probe(*args):
            called.append(name)
            return result

        monkeypatch.setattr(health, name, probe)

    stub("_check_filings", health._ok("3 filings loaded"))
    stub("_check_index_data", health._ok("present"))
    stub("_check_huggingface", health._ok("reachable"))
    stub("_check_ollama", health._ok("reachable"))
    stub("_check_provider_key", health._ok("set"))
    return called


def _response(status=200):
    return lambda url, **kwargs: httpx.Response(status, request=httpx.Request("GET", url))


def _raises(exc):
    def fake(url, **kwargs):
        raise exc

    return fake


class TestChecksFollowModes:
    def test_all_local_with_ollama(self, modes, probes):
        modes()
        assert set(health.run_readiness_checks()) == {"filings", "index_data", "ollama"}

    @pytest.mark.parametrize("stage", ["embedding", "reranker", "generation"])
    def test_any_hf_stage_adds_huggingface_check(self, modes, probes, stage):
        modes(**{stage: HF_API})
        assert "huggingface_api" in health.run_readiness_checks()

    def test_generation_in_api_mode_uses_huggingface_provider_not_ollama(self, modes, probes):
        modes(generation=HF_API)
        checks = health.run_readiness_checks()
        assert "huggingface_api" in checks and "ollama" not in checks

    def test_huggingface_provider_alone_adds_huggingface_check(self, modes, probes):
        modes(provider=LLMProvider("huggingface"))
        checks = health.run_readiness_checks()
        assert "huggingface_api" in checks and "ollama" not in checks

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
    def test_cloud_providers_check_their_api_key(self, modes, probes, provider):
        modes(provider=LLMProvider(provider))
        checks = health.run_readiness_checks()
        assert f"{provider}_api_key" in checks and "ollama" not in checks and "huggingface_api" not in checks

    def test_unneeded_probes_are_not_called(self, modes, probes):
        modes()
        health.run_readiness_checks()
        assert "_check_huggingface" not in probes and "_check_ollama" in probes

    def test_full_api_mode_checks_only_huggingface(self, modes, probes):
        modes(embedding=HF_API, reranker=HF_API, generation=HF_API)
        assert set(health.run_readiness_checks()) == {"filings", "index_data", "huggingface_api"}


class TestFilingsCheck:
    def test_ok(self, monkeypatch):
        monkeypatch.setattr(health, "get_filings", lambda: [{}, {}])
        check = health._check_filings()
        assert check.status == "ok" and "2 filings" in check.detail

    def test_empty_catalog_fails(self, monkeypatch):
        monkeypatch.setattr(health, "get_filings", lambda: [])
        assert health._check_filings().status == "fail"

    def test_startup_incomplete_fails(self, monkeypatch):
        def not_loaded():
            raise RuntimeError("not loaded")

        monkeypatch.setattr(health, "get_filings", not_loaded)
        check = health._check_filings()
        assert check.status == "fail" and "startup incomplete" in check.detail


class TestNumpyIndexCheck:
    @pytest.fixture(autouse=True)
    def numpy_store(self, modes):
        modes(store="numpy")

    def _run_dir(self, tmp_path, monkeypatch, files):
        for name in files:
            (tmp_path / name).write_text("x")
        monkeypatch.setattr(health.settings, "RUN_DIRECTORY", str(tmp_path))
        return tmp_path

    def test_complete_run(self, tmp_path, monkeypatch):
        self._run_dir(tmp_path, monkeypatch, ["A.chunks.jsonl", "emb.npy", "run_manifest.json"])
        assert health._check_index_data().status == "ok"

    @pytest.mark.parametrize(
        ("present", "missing"),
        [
            (["emb.npy", "run_manifest.json"], "chunks"),
            (["A.chunks.jsonl", "run_manifest.json"], "embeddings"),
            (["A.chunks.jsonl", "emb.npy"], "manifest"),
            ([], "chunks, embeddings, manifest"),
        ],
    )
    def test_reports_each_missing_artifact(self, tmp_path, monkeypatch, present, missing):
        self._run_dir(tmp_path, monkeypatch, present)
        check = health._check_index_data()
        assert check.status == "fail" and missing in check.detail

    def test_missing_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(health.settings, "RUN_DIRECTORY", str(tmp_path / "nope"))
        assert health._check_index_data().status == "fail"


class _FakeStore:
    def __init__(self, exists=True, count=10, error=None):
        self._exists, self._count, self._error = exists, count, error
        self.closed = False
        self.client = type("C", (), {"close": lambda client: setattr(self, "closed", True)})()

    def exists(self):
        if self._error:
            raise self._error
        return self._exists

    def count(self):
        return self._count


class TestQdrantIndexCheck:
    @pytest.fixture
    def use_store(self, modes, monkeypatch):
        modes(store="qdrant")
        import vector_store

        def install(store):
            monkeypatch.setattr(vector_store.QdrantChunkStore, "from_settings", classmethod(lambda cls, *a: store))
            return store

        return install

    def test_populated_collection(self, use_store):
        store = use_store(_FakeStore(count=1447))
        check = health._check_index_data()
        assert check.status == "ok" and "1447" in check.detail and store.closed

    def test_missing_collection(self, use_store):
        store = use_store(_FakeStore(exists=False))
        check = health._check_index_data()
        assert check.status == "fail" and "not found" in check.detail and store.closed

    def test_empty_collection(self, use_store):
        check = (use_store(_FakeStore(count=0)), health._check_index_data())[1]
        assert check.status == "fail" and "empty" in check.detail

    def test_unreachable_server_does_not_leak_error_text(self, use_store):
        store = use_store(_FakeStore(error=ConnectionError(f"http://user:{SECRET}@qdrant")))
        check = health._check_index_data()
        assert check.status == "fail" and "ConnectionError" in check.detail
        assert SECRET not in check.detail and store.closed

    def test_numpy_run_directory_is_ignored(self, use_store, monkeypatch):
        use_store(_FakeStore(count=1))
        monkeypatch.setattr(health.settings, "RUN_DIRECTORY", "does/not/exist")
        assert health._check_index_data().status == "ok"


class TestHuggingFaceCheck:
    @pytest.fixture(autouse=True)
    def key(self, monkeypatch):
        monkeypatch.setattr(health.settings, "HUGGINGFACE_API_KEY", SECRET)

    @pytest.mark.parametrize(("status", "expected"), [(200, "ok"), (401, "fail"), (403, "fail"), (500, "fail")])
    def test_status_codes(self, monkeypatch, status, expected):
        monkeypatch.setattr(health.httpx, "get", _response(status))
        check = health._check_huggingface()
        assert check.status == expected and SECRET not in check.detail

    def test_rejected_key_message(self, monkeypatch):
        monkeypatch.setattr(health.httpx, "get", _response(403))
        assert "rejected" in health._check_huggingface().detail

    @pytest.mark.parametrize("exc", [httpx.ConnectError("x"), httpx.ReadTimeout("x")])
    def test_network_errors(self, monkeypatch, exc):
        monkeypatch.setattr(health.httpx, "get", _raises(exc))
        check = health._check_huggingface()
        assert check.status == "fail" and type(exc).__name__ in check.detail

    def test_sends_bearer_token_with_timeout(self, monkeypatch):
        seen = {}

        def fake(url, **kwargs):
            seen.update(kwargs, url=url)
            return httpx.Response(200, request=httpx.Request("GET", url))

        monkeypatch.setattr(health.httpx, "get", fake)
        health._check_huggingface()
        assert seen["headers"] == {"Authorization": f"Bearer {SECRET}"}
        assert seen["timeout"] == health.CHECK_TIMEOUT_SECONDS and seen["url"] == health.HF_WHOAMI_URL


class TestOllamaCheck:
    def test_probes_configured_host(self, monkeypatch):
        urls = []
        monkeypatch.setattr(health.settings, "OLLAMA_HOST", "http://ollama:11434/")
        monkeypatch.setattr(health.httpx, "get", lambda url, **kw: (urls.append(url), _response()(url))[1])
        assert health._check_ollama().status == "ok"
        assert urls == ["http://ollama:11434/api/tags"]

    def test_server_error(self, monkeypatch):
        monkeypatch.setattr(health.httpx, "get", _response(500))
        assert health._check_ollama().status == "fail"

    def test_unreachable(self, monkeypatch):
        monkeypatch.setattr(health.httpx, "get", _raises(httpx.ConnectError("down")))
        check = health._check_ollama()
        assert check.status == "fail" and "unreachable" in check.detail


class TestProviderKeyCheck:
    @pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
    def test_key_set_and_missing(self, monkeypatch, provider):
        attr = f"{provider.upper()}_API_KEY"
        monkeypatch.setattr(health.settings, attr, SECRET)
        ok = health._check_provider_key(provider)
        assert ok.status == "ok" and SECRET not in ok.detail
        monkeypatch.setattr(health.settings, attr, None)
        missing = health._check_provider_key(provider)
        assert missing.status == "fail" and attr in missing.detail


class TestReadyEndpoint:
    def test_all_green_in_each_mode(self, client, modes, probes):
        for kwargs in (
            {},
            {"generation": HF_API},
            {"embedding": HF_API, "reranker": HF_API, "generation": HF_API},
            {"provider": LLMProvider("openai")},
        ):
            modes(**kwargs)
            response = client.get("/health/ready")
            assert response.status_code == 200 and response.json()["status"] == "ready", kwargs

    def test_failing_dependency_returns_503_with_details(self, client, modes, probes, monkeypatch):
        modes(generation=HF_API)
        monkeypatch.setattr(health, "_check_huggingface", lambda: health._fail("Hugging Face API key rejected"))
        response = client.get("/health/ready")
        body = response.json()
        assert response.status_code == 503 and body["status"] == "not_ready"
        assert body["checks"]["huggingface_api"] == {"status": "fail", "detail": "Hugging Face API key rejected"}
        assert body["checks"]["filings"]["status"] == "ok"

    def test_one_failure_among_many_is_enough(self, client, modes, probes, monkeypatch):
        modes()
        monkeypatch.setattr(health, "_check_index_data", lambda: health._fail("run directory not found"))
        assert client.get("/health/ready").status_code == 503

    def test_response_never_contains_secrets(self, client, modes, monkeypatch):
        modes(generation=HF_API)
        monkeypatch.setattr(health.settings, "HUGGINGFACE_API_KEY", SECRET)
        monkeypatch.setattr(health.httpx, "get", _raises(httpx.ConnectError(f"token={SECRET}")))
        monkeypatch.setattr(health, "_check_filings", lambda: health._ok("ok"))
        monkeypatch.setattr(health, "_check_index_data", lambda: health._ok("ok"))
        response = client.get("/health/ready")
        assert response.status_code == 503 and SECRET not in json.dumps(response.json())

    def test_stays_open_without_api_key(self, client, modes, probes, monkeypatch):
        from api.config import api_settings

        modes()
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", "k" * 32)
        assert client.get("/health/ready").status_code == 200

    def test_liveness_unaffected_by_failing_dependencies(self, client, modes, monkeypatch):
        modes()
        monkeypatch.setattr(health.httpx, "get", _raises(httpx.ConnectError("down")))
        assert client.get("/health").status_code == 200
