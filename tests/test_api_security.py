"""
Offline tests for API security (config validation, API-key auth, CORS) and
readiness checks. These never load the RAG pipeline: dependencies are overridden.
"""

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.config import DEV_CORS_ORIGINS, APISettings, api_settings
from api.dependencies import get_filings, get_rag_service, get_valid_doc_ids
from api.main import app
from api.routers import health

VALID_KEY = "k" * 32
PROD = {"ENVIRONMENT": "prod", "API_ACCESS_KEY": VALID_KEY, "CORS_ORIGINS": "https://app.example.com"}


def make_settings(**overrides) -> APISettings:
    return APISettings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("ENVIRONMENT", "API_ACCESS_KEY", "CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

class TestConfig:
    def test_dev_defaults(self):
        s = make_settings()
        assert s.ENVIRONMENT == "dev"
        assert s.CORS_ORIGINS == DEV_CORS_ORIGINS
        assert s.API_ACCESS_KEY is None
        assert not s.is_prod

    def test_cors_comma_separated_is_trimmed(self):
        s = make_settings(CORS_ORIGINS=" https://a.example/ , https://b.example ")
        assert s.CORS_ORIGINS == ["https://a.example", "https://b.example"]

    def test_cors_json_array(self):
        s = make_settings(CORS_ORIGINS='["https://a.example"]')
        assert s.CORS_ORIGINS == ["https://a.example"]

    def test_empty_key_is_unset(self):
        assert make_settings(API_ACCESS_KEY="  ").API_ACCESS_KEY is None

    def test_invalid_environment_rejected(self):
        with pytest.raises(ValidationError):
            make_settings(ENVIRONMENT="production")

    def test_prod_valid(self):
        s = make_settings(**PROD)
        assert s.is_prod and s.CORS_ORIGINS == ["https://app.example.com"]

    def test_prod_requires_key(self):
        with pytest.raises(ValidationError, match="API_ACCESS_KEY is required"):
            make_settings(**{**PROD, "API_ACCESS_KEY": ""})

    def test_prod_rejects_short_key(self):
        with pytest.raises(ValidationError, match="at least"):
            make_settings(**{**PROD, "API_ACCESS_KEY": "short"})

    def test_prod_rejects_wildcard_cors(self):
        with pytest.raises(ValidationError, match="CORS_ORIGINS"):
            make_settings(**{**PROD, "CORS_ORIGINS": "*"})

    def test_prod_rejects_default_cors(self):
        with pytest.raises(ValidationError, match="CORS_ORIGINS"):
            make_settings(ENVIRONMENT="prod", API_ACCESS_KEY=VALID_KEY)


# =============================================================================
# AUTH + CORS (real app wiring, pipeline dependencies overridden)
# =============================================================================

class _FakeRag:
    model_name = "fake-model"


@pytest.fixture
def client():
    app.dependency_overrides[get_filings] = lambda: [
        {"doc_id": "AAPL_10-K_2025", "company": "Apple Inc.", "ticker": "AAPL",
         "form": "10-K", "fiscal_year": 2025, "period_end": "September 27, 2025"}
    ]
    app.dependency_overrides[get_valid_doc_ids] = lambda: {"AAPL_10-K_2025"}
    app.dependency_overrides[get_rag_service] = lambda: _FakeRag()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


class TestAuth:
    def test_open_when_key_unset(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
        assert client.get("/filings").status_code == 200

    def test_missing_key_rejected(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", VALID_KEY)
        response = client.get("/filings")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "ApiKey"

    def test_wrong_key_rejected(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", VALID_KEY)
        assert client.get("/filings", headers={"X-API-Key": "x" * 32}).status_code == 401

    def test_correct_key_accepted(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", VALID_KEY)
        assert client.get("/filings", headers={"X-API-Key": VALID_KEY}).status_code == 200

    def test_questions_protected_before_validation(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", VALID_KEY)
        assert client.post("/questions", json={}).status_code == 401
        assert client.post("/questions", json={}, headers={"X-API-Key": VALID_KEY}).status_code == 422

    def test_health_stays_open(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", VALID_KEY)
        assert client.get("/health").status_code == 200


@pytest.mark.skipif("*" in api_settings.CORS_ORIGINS, reason="wildcard CORS configured")
class TestCors:
    def preflight(self, client, origin):
        return client.options(
            "/questions",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-api-key",
            },
        )

    def test_allowed_origin(self, client):
        origin = api_settings.CORS_ORIGINS[0]
        response = self.preflight(client, origin)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin

    def test_disallowed_origin(self, client):
        response = self.preflight(client, "https://evil.example")
        assert "access-control-allow-origin" not in response.headers


# =============================================================================
# READINESS
# =============================================================================

def _fake_get(status=200, exc=None):
    def fake(url, **kwargs):
        if exc:
            raise exc
        return httpx.Response(status, request=httpx.Request("GET", url))
    return fake


class TestReadiness:
    def test_ready_returns_200(self, client, monkeypatch):
        monkeypatch.setattr(health, "run_readiness_checks", lambda: {"filings": health._ok("3 filings loaded")})
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_not_ready_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(
            health,
            "run_readiness_checks",
            lambda: {"filings": health._ok("ok"), "ollama": health._fail("Ollama unreachable")},
        )
        response = client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["ollama"]["status"] == "fail"

    def test_index_data_present(self):
        assert health._check_index_data().status == "ok"

    def test_index_data_missing(self, monkeypatch):
        monkeypatch.setattr(health.settings, "RUN_DIRECTORY", "data/chunks/does-not-exist")
        assert health._check_index_data().status == "fail"

    def test_huggingface_without_key(self, monkeypatch):
        monkeypatch.setattr(health.settings, "HUGGINGFACE_API_KEY", None)
        check = health._check_huggingface()
        assert check.status == "fail" and "not set" in check.detail

    @pytest.mark.parametrize(
        ("status", "exc", "expected"),
        [(200, None, "ok"), (401, None, "fail"), (500, None, "fail"), (None, httpx.ConnectError("boom"), "fail")],
    )
    def test_huggingface_outcomes(self, monkeypatch, status, exc, expected):
        monkeypatch.setattr(health.settings, "HUGGINGFACE_API_KEY", "hf_secret_value")
        monkeypatch.setattr(health.httpx, "get", _fake_get(status or 200, exc))
        check = health._check_huggingface()
        assert check.status == expected
        assert "hf_secret_value" not in check.detail

    def test_ollama_outcomes(self, monkeypatch):
        monkeypatch.setattr(health.httpx, "get", _fake_get(200))
        assert health._check_ollama().status == "ok"
        monkeypatch.setattr(health.httpx, "get", _fake_get(exc=httpx.ConnectError("down")))
        assert health._check_ollama().status == "fail"

    def test_checks_follow_mode_and_provider(self, monkeypatch):
        monkeypatch.setattr(health, "_check_ollama", lambda: health._ok("ok"))
        monkeypatch.setattr(health, "_check_huggingface", lambda: health._ok("ok"))
        cls = type(health.settings)
        monkeypatch.setattr(cls, "effective_llm_provider", property(lambda self: "ollama"))
        monkeypatch.setattr(cls, "effective_embedding_mode", property(lambda self: "local"))
        monkeypatch.setattr(cls, "effective_reranker_mode", property(lambda self: "local"))
        monkeypatch.setattr(cls, "effective_generation_mode", property(lambda self: "local"))
        local = health.run_readiness_checks()
        assert "ollama" in local and "huggingface_api" not in local

        monkeypatch.setattr(cls, "effective_llm_provider", property(lambda self: "huggingface"))
        monkeypatch.setattr(cls, "effective_embedding_mode", property(lambda self: "huggingface_api"))
        remote = health.run_readiness_checks()
        assert "huggingface_api" in remote and "ollama" not in remote
