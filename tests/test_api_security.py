"""
Offline tests for API security (config validation, API-key auth, CORS, rate limiting) and
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
from jose import jwt

VALID_KEY = "k" * 32
PROD = {"ENVIRONMENT": "prod", "API_ACCESS_KEY": VALID_KEY, "CORS_ORIGINS": "https://app.example.com"}


def make_settings(**overrides) -> APISettings:
    return APISettings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "ENVIRONMENT",
        "API_ACCESS_KEY",
        "CORS_ORIGINS",
        "RATE_LIMIT_REDIS_URL",
        "RATE_LIMIT_DEFAULT_PER_MINUTE",
        "RATE_LIMIT_QUESTIONS_PER_MINUTE",
        "RATE_LIMIT_FILINGS_PER_MINUTE",
        "RATE_LIMIT_GLOBAL_PER_MINUTE",
    ):
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
        # Clear rate limiter state between tests
        from api.rate_limit import in_memory_limiter
        in_memory_limiter.requests.clear()


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


# =============================================================================
# RATE LIMITING
# =============================================================================

class TestRateLimiting:
    def test_rate_limit_config_defaults(self):
        """Test that rate limit configuration has sensible defaults."""
        s = make_settings()
        assert s.RATE_LIMIT_DEFAULT_PER_MINUTE > 0
        assert s.RATE_LIMIT_QUESTIONS_PER_MINUTE > 0
        assert s.RATE_LIMIT_FILINGS_PER_MINUTE > 0
        assert s.RATE_LIMIT_GLOBAL_PER_MINUTE > 0
        assert s.RATE_LIMIT_REDIS_URL is None  # Default to in-memory for dev

    def test_rate_limit_config_custom(self):
        """Test that rate limit configuration can be customized."""
        s = make_settings(
            RATE_LIMIT_DEFAULT_PER_MINUTE=100,
            RATE_LIMIT_QUESTIONS_PER_MINUTE=20,
            RATE_LIMIT_FILINGS_PER_MINUTE=50,
            RATE_LIMIT_GLOBAL_PER_MINUTE=200,
            RATE_LIMIT_REDIS_URL="redis://localhost:6379/0"
        )
        assert s.RATE_LIMIT_DEFAULT_PER_MINUTE == 100
        assert s.RATE_LIMIT_QUESTIONS_PER_MINUTE == 20
        assert s.RATE_LIMIT_FILINGS_PER_MINUTE == 50
        assert s.RATE_LIMIT_GLOBAL_PER_MINUTE == 200
        assert s.RATE_LIMIT_REDIS_URL == "redis://localhost:6379/0"

    def test_in_memory_limiter_basic(self):
        """Test that in-memory limiter correctly tracks requests."""
        from api.rate_limit import in_memory_limiter

        # Clear state
        in_memory_limiter.requests.clear()

        # First request should be allowed
        allowed, info = in_memory_limiter.is_allowed("test_ip", limit=3, period=60)
        assert allowed is True
        assert info["remaining"] == 2

        # Second request should be allowed
        allowed, info = in_memory_limiter.is_allowed("test_ip", limit=3, period=60)
        assert allowed is True
        assert info["remaining"] == 1

        # Third request should be allowed
        allowed, info = in_memory_limiter.is_allowed("test_ip", limit=3, period=60)
        assert allowed is True
        assert info["remaining"] == 0

        # Fourth request should be rate limited
        allowed, info = in_memory_limiter.is_allowed("test_ip", limit=3, period=60)
        assert allowed is False
        assert info["remaining"] == 0

    def test_redis_branch_limits_and_sets_retry_after(self, monkeypatch):
        """The Redis code path (limits moving window) must allow up to the limit, then 429."""
        from fastapi import HTTPException
        from limits.storage import storage_from_string
        from limits.strategies import MovingWindowRateLimiter

        from api import rate_limit

        monkeypatch.setattr(rate_limit, "redis_url", "redis://test")
        monkeypatch.setattr(rate_limit, "moving_window", MovingWindowRateLimiter(storage_from_string("memory://")))

        class Req:
            state = type("S", (), {})()
            client = type("C", (), {"host": "1.2.3.4"})()
            headers = {}
            scope = {"client": ("1.2.3.4", 0)}

        for _ in range(2):
            rate_limit.check_rate_limit(Req(), limit=2, period=60)
        with pytest.raises(HTTPException) as exc:
            rate_limit.check_rate_limit(Req(), limit=2, period=60)
        assert exc.value.status_code == 429
        assert int(exc.value.headers["Retry-After"]) >= 1

    def test_rate_limit_key_prefers_user(self):
        from api.rate_limit import get_rate_limit_key

        class Req:
            state = type("S", (), {"user_id": "u1"})()

        assert get_rate_limit_key(Req()) == "user:u1"

    def test_correlation_id_header(self, client, monkeypatch):
        """Test that correlation ID is added to responses."""
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
        response = client.get("/filings")
        assert "X-Correlation-ID" in response.headers
        assert len(response.headers["X-Correlation-ID"]) > 0

    def test_correlation_id_preserved(self, client, monkeypatch):
        """Test that client-provided correlation ID is preserved."""
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
        custom_id = "my-custom-correlation-id"
        response = client.get("/filings", headers={"X-Correlation-ID": custom_id})
        assert response.headers["X-Correlation-ID"] == custom_id


# =============================================================================
# SUPABASE AUTHENTICATION
# =============================================================================

class TestErrorSanitization:
    def test_generation_error_does_not_leak_details(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)

        class _Boom:
            def answer(self, request):
                raise RuntimeError("provider-secret-detail Request ID: Root=1-abc")

        app.dependency_overrides[get_rag_service] = lambda: _Boom()
        response = client.post("/questions", json={"question": "What was net income?", "doc_id": "AAPL_10-K_2025"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "provider-secret-detail" not in detail and "Request ID" not in detail
        assert response.headers["X-Correlation-ID"] in detail


class TestSupabaseAuth:
    def test_jwt_secret_configurable(self):
        """Test that JWT secret can be configured."""
        s = make_settings(SUPABASE_JWT_SECRET="test_secret")
        assert s.SUPABASE_JWT_SECRET == "test_secret"

    def test_jwt_secret_optional(self):
        """Test that JWT secret is optional (auth disabled when not set)."""
        s = make_settings()
        assert s.SUPABASE_JWT_SECRET is None

    def test_verify_valid_token(self, monkeypatch):
        """Test that a valid Supabase JWT can be verified."""
        from api.auth import verify_supabase_token

        # Create a test token
        secret = "test_jwt_secret_for_testing"
        monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", secret)

        payload = {
            "sub": "user_123",
            "email": "test@example.com",
            "role": "authenticated",
            "aud": "authenticated"
        }
        token = jwt.encode(payload, secret, algorithm="HS256")

        user = verify_supabase_token(token)
        assert user.user_id == "user_123"
        assert user.email == "test@example.com"
        assert user.role == "authenticated"

    def test_verify_invalid_token(self, monkeypatch):
        """Test that an invalid token is rejected."""
        from api.auth import verify_supabase_token
        from fastapi import HTTPException

        secret = "test_jwt_secret_for_testing"
        monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", secret)

        with pytest.raises(HTTPException, match="Invalid or expired token"):
            verify_supabase_token("invalid_token")

    def test_verify_token_without_secret(self, monkeypatch):
        """Test that verification fails gracefully when secret is not set."""
        from api.auth import verify_supabase_token
        from fastapi import HTTPException

        monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", None)
        monkeypatch.setattr(api_settings, "SUPABASE_URL", None)

        with pytest.raises(HTTPException, match="not configured"):
            verify_supabase_token("any_token")

    @pytest.fixture
    def es256(self, monkeypatch):
        """ES256 keypair with the public JWK served through a mocked JWKS endpoint."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from jose import jwk

        from api import auth

        private = ec.generate_private_key(ec.SECP256R1())
        private_pem = private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        public_pem = private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        public_jwk = {**jwk.construct(public_pem, "ES256").to_dict(), "kid": "key-1"}

        monkeypatch.setattr(api_settings, "SUPABASE_URL", "https://proj.supabase.co")
        monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", None)
        monkeypatch.setattr(auth, "_fetch_jwks", lambda: [public_jwk])
        monkeypatch.setitem(auth._jwks_cache, "keys", None)
        monkeypatch.setitem(auth._jwks_cache, "fetched_at", 0.0)
        return private_pem

    def test_verify_es256_token_via_jwks(self, es256):
        from api.auth import verify_supabase_token

        token = jwt.encode(
            {"sub": "user_es", "email": "es@example.com", "aud": "authenticated"},
            es256, algorithm="ES256", headers={"kid": "key-1"},
        )
        user = verify_supabase_token(token)
        assert user.user_id == "user_es" and user.email == "es@example.com"

    def test_es256_unknown_kid_rejected(self, es256):
        from fastapi import HTTPException

        from api.auth import verify_supabase_token

        token = jwt.encode(
            {"sub": "x", "aud": "authenticated"}, es256, algorithm="ES256", headers={"kid": "other"},
        )
        with pytest.raises(HTTPException) as exc:
            verify_supabase_token(token)
        assert exc.value.status_code == 401

    def test_hs256_rejected_when_only_url_configured(self, monkeypatch):
        from fastapi import HTTPException

        from api.auth import verify_supabase_token

        monkeypatch.setattr(api_settings, "SUPABASE_URL", "https://proj.supabase.co")
        monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", None)
        token = jwt.encode({"sub": "x", "aud": "authenticated"}, "guess", algorithm="HS256")
        with pytest.raises(HTTPException) as exc:
            verify_supabase_token(token)
        assert exc.value.status_code == 401



# =============================================================================
# LOGIN REQUIRED + PER-USER LIMITS + RATE LIMIT HEADERS (through the real app)
# =============================================================================

SECRET = "test_jwt_secret_for_testing"


def _token(user_id: str, secret: str = SECRET) -> dict:
    token = jwt.encode({"sub": user_id, "aud": "authenticated"}, secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
    monkeypatch.setattr(api_settings, "SUPABASE_JWT_SECRET", SECRET)


class TestLoginRequired:
    def test_missing_token_rejected(self, client, auth_on):
        response = client.get("/filings")
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"

    def test_questions_rejects_guests_before_validation(self, client, auth_on):
        assert client.post("/questions", json={}).status_code == 401
        assert client.post("/questions", json={}, headers=_token("u1")).status_code == 422

    def test_invalid_and_wrong_secret_tokens_rejected(self, client, auth_on):
        assert client.get("/filings", headers={"Authorization": "Bearer garbage"}).status_code == 401
        assert client.get("/filings", headers=_token("u1", secret="other-secret")).status_code == 401

    def test_valid_token_accepted(self, client, auth_on):
        assert client.get("/filings", headers=_token("u1")).status_code == 200

    def test_health_stays_open(self, client, auth_on):
        assert client.get("/health").status_code == 200

    def test_open_when_auth_not_configured(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
        assert client.get("/filings").status_code == 200

    def test_limits_are_per_user(self, client, auth_on):
        limit = api_settings.RATE_LIMIT_FILINGS_PER_MINUTE
        for _ in range(limit):
            assert client.get("/filings", headers=_token("user_a")).status_code == 200
        blocked = client.get("/filings", headers=_token("user_a"))
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1
        assert client.get("/filings", headers=_token("user_b")).status_code == 200

    def test_rate_limit_headers(self, client, monkeypatch):
        monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
        first = client.get("/filings")
        limit = api_settings.RATE_LIMIT_FILINGS_PER_MINUTE
        assert first.headers["X-RateLimit-Limit"] == str(limit)
        assert first.headers["X-RateLimit-Remaining"] == str(limit - 1)
        assert int(first.headers["X-RateLimit-Reset"]) > 0
        second = client.get("/filings")
        assert second.headers["X-RateLimit-Remaining"] == str(limit - 2)
