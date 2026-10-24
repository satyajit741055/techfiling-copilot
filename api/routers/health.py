"""
Health Check Router.

Provides liveness/readiness endpoints for the API.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from api.dependencies import get_filings, get_rag_service
from api.schemas import HealthResponse
from api.services import RAGService
from config.settings import settings

router = APIRouter()


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
