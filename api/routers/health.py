"""
Health Check Router.

Provides liveness/readiness endpoints for the API.
"""

import logging
import sys
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, Response

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from api.dependencies import get_filings, get_rag_service
from api.schemas import DependencyCheck, HealthResponse, ReadinessResponse
from api.services import RAGService
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
CHECK_TIMEOUT_SECONDS = 3.0


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is healthy and ready to serve requests",
)
def health_check(
    rag_service: RAGService = Depends(get_rag_service),
    filings: list[dict] = Depends(get_filings),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, environment, loaded model, and filing count.
    Used by load balancers and monitoring systems.
    """
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT.value,
        model=rag_service.model_name,
        filings_loaded=len(filings),
    )


def _ok(detail: str) -> DependencyCheck:
    return DependencyCheck(status="ok", detail=detail)


def _fail(detail: str) -> DependencyCheck:
    return DependencyCheck(status="fail", detail=detail)


def _value(item: object) -> str:
    """Enum or plain string to its string value."""
    return str(getattr(item, "value", item))


def _check_filings() -> DependencyCheck:
    try:
        count = len(get_filings())
    except RuntimeError:
        return _fail("filings not loaded (startup incomplete)")
    return _ok(f"{count} filings loaded") if count else _fail("no filings found")


def _check_qdrant() -> DependencyCheck:
    from vector_store import QdrantChunkStore

    store = None
    try:
        store = QdrantChunkStore.from_settings()
        if not store.exists():
            return _fail(f"Qdrant collection not found: {settings.QDRANT_COLLECTION}")
        count = store.count()
    except Exception as exc:
        return _fail(f"Qdrant unreachable ({type(exc).__name__})")
    finally:
        if store is not None:
            store.client.close()
    return _ok(f"Qdrant collection {settings.QDRANT_COLLECTION}: {count} chunks") if count else _fail("Qdrant collection is empty")


def _check_index_data() -> DependencyCheck:
    if settings.VECTOR_STORE == "qdrant":
        return _check_qdrant()
    run_dir = ROOT / Path(settings.RUN_DIRECTORY)
    if not run_dir.is_dir():
        return _fail(f"run directory not found: {settings.RUN_DIRECTORY}")
    missing = [
        label
        for label, pattern in (("chunks", "*.chunks.jsonl"), ("embeddings", "*.npy"), ("manifest", "run_manifest.json"))
        if not any(run_dir.glob(pattern))
    ]
    return _fail(f"missing in run directory: {', '.join(missing)}") if missing else _ok("chunks, embeddings and manifest present")


def _check_huggingface() -> DependencyCheck:
    if not settings.HUGGINGFACE_API_KEY:
        return _fail("HUGGINGFACE_API_KEY not set")
    try:
        response = httpx.get(
            HF_WHOAMI_URL,
            headers={"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"},
            timeout=CHECK_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        return _fail(f"Hugging Face API unreachable ({type(exc).__name__})")
    if response.status_code in (401, 403):
        return _fail("Hugging Face API key rejected")
    return _ok("Hugging Face API reachable, key accepted") if response.is_success else _fail(f"Hugging Face API returned {response.status_code}")


def _check_ollama() -> DependencyCheck:
    try:
        response = httpx.get(f"{settings.OLLAMA_HOST.rstrip('/')}/api/tags", timeout=CHECK_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return _fail(f"Ollama unreachable at {settings.OLLAMA_HOST} ({type(exc).__name__})")
    return _ok("Ollama reachable") if response.is_success else _fail(f"Ollama returned {response.status_code}")


def _check_provider_key(provider: str) -> DependencyCheck:
    key = getattr(settings, f"{provider.upper()}_API_KEY", None)
    return _ok(f"{provider} API key set") if key else _fail(f"{provider.upper()}_API_KEY not set")


def run_readiness_checks() -> dict[str, DependencyCheck]:
    """Run every check relevant to the current execution modes and LLM provider."""
    checks = {"filings": _check_filings(), "index_data": _check_index_data()}

    provider = _value(settings.effective_llm_provider)
    modes = {
        _value(settings.effective_embedding_mode),
        _value(settings.effective_reranker_mode),
        _value(settings.effective_generation_mode),
    }
    if "huggingface_api" in modes or provider == "huggingface":
        checks["huggingface_api"] = _check_huggingface()
    if provider == "ollama":
        checks["ollama"] = _check_ollama()
    elif provider in ("openai", "anthropic", "google"):
        checks[f"{provider}_api_key"] = _check_provider_key(provider)
    return checks


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Verify data and external dependencies. Returns 503 if any required dependency is unavailable.",
    responses={503: {"description": "A required dependency is unavailable"}},
)
def readiness_check(response: Response) -> ReadinessResponse:
    """
    Readiness endpoint for load balancers and monitoring.

    Unlike /health (liveness, always cheap), this calls external services,
    so poll it sparingly. Details never include secrets.
    """
    checks = run_readiness_checks()
    ready = all(check.status == "ok" for check in checks.values())
    if not ready:
        response.status_code = 503
        logger.warning(f"Readiness check failed: { {name: c.detail for name, c in checks.items() if c.status != 'ok'} }")
    return ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
