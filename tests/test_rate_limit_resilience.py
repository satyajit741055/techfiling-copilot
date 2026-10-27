"""Offline tests for the global rate limit and for Redis failing after startup."""

import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from limits.storage import storage_from_string
from limits.strategies import MovingWindowRateLimiter

from api import rate_limit
from api.config import api_settings
from api.dependencies import get_filings, get_rag_service, get_valid_doc_ids
from api.main import app


class _Req:
    def __init__(self, user=None, ip="1.2.3.4"):
        self.state = type("S", (), {"user_id": user} if user else {})()
        self.client = type("C", (), {"host": ip})()
        self.headers = {}
        self.scope = {"client": (ip, 0)}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_settings, "API_ACCESS_KEY", None)
    app.dependency_overrides[get_filings] = lambda: [{
        "doc_id": "AAPL_10-K_2025", "company": "Apple Inc.", "ticker": "AAPL",
        "form": "10-K", "fiscal_year": 2025, "period_end": "September 27, 2025"}]
    app.dependency_overrides[get_valid_doc_ids] = lambda: {"AAPL_10-K_2025"}
    app.dependency_overrides[get_rag_service] = lambda: type("R", (), {"model_name": "m"})()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def in_memory(monkeypatch):
    monkeypatch.setattr(rate_limit, "redis_url", None)
    rate_limit.in_memory_limiter.requests.clear()


class _BrokenWindow:
    """Stands in for a Redis-backed limiter whose server went away."""

    def __init__(self, fail_on="hit"):
        self.fail_on = fail_on
        self.calls = 0

    def hit(self, *args, **kwargs):
        self.calls += 1
        if self.fail_on == "hit":
            raise ConnectionError("redis down")
        return True

    def get_window_stats(self, *args, **kwargs):
        raise ConnectionError("redis down")


class TestGlobalLimit:
    def test_global_scope_ignores_caller_identity(self, in_memory):
        for i in range(3):
            rate_limit.check_rate_limit(_Req(user=f"u{i}", ip=f"10.0.0.{i}"), 3, 60, scope="global")
        with pytest.raises(HTTPException) as exc:
            rate_limit.check_rate_limit(_Req(user="someone-else", ip="9.9.9.9"), 3, 60, scope="global")
        assert exc.value.status_code == 429
        assert exc.value.detail["scope"] == "global" and int(exc.value.headers["Retry-After"]) >= 1

    def test_global_and_per_user_buckets_are_separate(self, in_memory):
        request = _Req(user="u1")
        rate_limit.check_rate_limit(request, 1, 60, scope="global")
        rate_limit.check_rate_limit(request, 1, 60)
        with pytest.raises(HTTPException):
            rate_limit.check_rate_limit(request, 1, 60)

    @staticmethod
    def _exhaust_global_bucket():
        limit = api_settings.RATE_LIMIT_GLOBAL_PER_MINUTE
        for i in range(limit):
            rate_limit.check_rate_limit(_Req(user=f"u{i}", ip=f"10.0.{i}.1"), limit, 60, scope="global")

    def test_filings_endpoint_enforces_global_limit(self, client, in_memory):
        assert client.get("/filings").status_code == 200
        rate_limit.in_memory_limiter.requests.clear()
        self._exhaust_global_bucket()
        blocked = client.get("/filings")
        assert blocked.status_code == 429 and blocked.json()["detail"]["scope"] == "global"
        assert int(blocked.headers["Retry-After"]) >= 1

    def test_questions_endpoint_enforces_global_limit(self, client, in_memory):
        self._exhaust_global_bucket()
        blocked = client.post("/questions", json={"question": "What were net sales?", "doc_id": "AAPL_10-K_2025"})
        assert blocked.status_code == 429 and blocked.json()["detail"]["scope"] == "global"

    def test_questions_and_filings_share_one_global_bucket(self, client, in_memory):
        limit = api_settings.RATE_LIMIT_GLOBAL_PER_MINUTE
        for i in range(limit - 1):
            rate_limit.check_rate_limit(_Req(user=f"u{i}"), limit, 60, scope="global")
        assert client.get("/filings").status_code == 200
        blocked = client.post("/questions", json={"question": "What were net sales?", "doc_id": "AAPL_10-K_2025"})
        assert blocked.status_code == 429 and blocked.json()["detail"]["scope"] == "global"

    def test_health_is_not_globally_limited(self, client, in_memory):
        self._exhaust_global_bucket()
        assert client.get("/health").status_code == 200

    def test_per_user_headers_win_over_global_headers(self, client, in_memory):
        response = client.get("/filings")
        assert response.headers["X-RateLimit-Limit"] == str(api_settings.RATE_LIMIT_FILINGS_PER_MINUTE)


class TestRedisOutageAfterStartup:
    @pytest.fixture
    def redis_mode(self, monkeypatch):
        monkeypatch.setattr(rate_limit, "redis_url", "redis://test")
        rate_limit.in_memory_limiter.requests.clear()

    def test_hit_failure_falls_back_to_in_memory(self, redis_mode, monkeypatch, caplog):
        monkeypatch.setattr(rate_limit, "moving_window", _BrokenWindow("hit"))
        request = _Req(user="u1")
        with caplog.at_level(logging.WARNING, logger="api.rate_limit"):
            rate_limit.check_rate_limit(request, 2, 60)
        assert request.state.rate_limit["remaining"] == 1
        assert "Redis rate limiter unavailable" in caplog.text and "ConnectionError" in caplog.text

    def test_stats_failure_after_successful_hit_also_falls_back(self, redis_mode, monkeypatch):
        monkeypatch.setattr(rate_limit, "moving_window", _BrokenWindow("stats"))
        request = _Req(user="u1")
        rate_limit.check_rate_limit(request, 2, 60)
        assert request.state.rate_limit["limit"] == 2

    def test_limits_are_still_enforced_during_the_outage(self, redis_mode, monkeypatch):
        monkeypatch.setattr(rate_limit, "moving_window", _BrokenWindow("hit"))
        request = _Req(user="u1")
        rate_limit.check_rate_limit(request, 2, 60)
        rate_limit.check_rate_limit(request, 2, 60)
        with pytest.raises(HTTPException) as exc:
            rate_limit.check_rate_limit(request, 2, 60)
        assert exc.value.status_code == 429

    def test_global_limit_also_degrades_gracefully(self, redis_mode, monkeypatch):
        monkeypatch.setattr(rate_limit, "moving_window", _BrokenWindow("hit"))
        rate_limit.check_rate_limit(_Req(), 1, 60, scope="global")
        with pytest.raises(HTTPException) as exc:
            rate_limit.check_rate_limit(_Req(user="other"), 1, 60, scope="global")
        assert exc.value.detail["scope"] == "global"

    def test_endpoint_does_not_500_when_redis_dies(self, client, redis_mode, monkeypatch):
        monkeypatch.setattr(rate_limit, "moving_window", _BrokenWindow("hit"))
        assert client.get("/filings").status_code == 200

    def test_recovery_uses_redis_again(self, redis_mode, monkeypatch):
        broken = _BrokenWindow("hit")
        monkeypatch.setattr(rate_limit, "moving_window", broken)
        rate_limit.check_rate_limit(_Req(user="u1"), 5, 60)
        healthy = MovingWindowRateLimiter(storage_from_string("memory://"))
        monkeypatch.setattr(rate_limit, "moving_window", healthy)
        request = _Req(user="u1")
        rate_limit.check_rate_limit(request, 5, 60)
        assert request.state.rate_limit["remaining"] == 4 and broken.calls == 1

    def test_working_redis_path_does_not_touch_in_memory_store(self, redis_mode, monkeypatch):
        monkeypatch.setattr(rate_limit, "moving_window", MovingWindowRateLimiter(storage_from_string("memory://")))
        rate_limit.check_rate_limit(_Req(user="u1"), 5, 60)
        assert not rate_limit.in_memory_limiter.requests
