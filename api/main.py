"""
TechFiling Copilot API — Main Application.

FastAPI backend wrapping the RAG pipeline for SEC 10-K document Q&A.

Usage:
    # From project root
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # Or with explicit PYTHONPATH
    set PYTHONPATH=.
    uvicorn api.main:app --reload

Endpoints:
    GET  /health     - Health check
    GET  /filings    - List available filings
    POST /questions  - Ask a question about a filing

    GET  /docs       - Swagger UI (auto-generated)
    GET  /redoc      - ReDoc (auto-generated)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.config import api_settings
from api.dependencies import init_generator, shutdown_generator
from api.middleware import AuthMiddleware, CorrelationIDMiddleware, RequestLoggingMiddleware
from api.rate_limit import limiter
from api.routers import filings, health, questions
from api.security import require_api_key

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Startup:
        - Load filing metadata
        - Initialize AnswerGenerator (loads models, index)

    Shutdown:
        - Cleanup resources
    """
    # Startup
    logger.info("=" * 60)
    logger.info("TechFiling Copilot API - Starting up...")
    logger.info("=" * 60)
    logger.info(f"Environment: {api_settings.ENVIRONMENT} | CORS origins: {api_settings.CORS_ORIGINS}")
    if api_settings.API_ACCESS_KEY:
        logger.info("API key authentication: ENABLED")
    else:
        logger.warning("API key authentication: DISABLED (API_ACCESS_KEY not set; allowed in dev only)")

    init_generator()

    logger.info("=" * 60)
    logger.info("API ready to serve requests!")
    logger.info(f"  Swagger UI: http://{api_settings.API_HOST}:{api_settings.API_PORT}/docs")
    logger.info(f"  ReDoc:      http://{api_settings.API_HOST}:{api_settings.API_PORT}/redoc")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down...")
    shutdown_generator()
    logger.info("Goodbye!")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="TechFiling Copilot API",
    description="""
SEC 10-K Document Q&A with RAG (Retrieval-Augmented Generation).

## Features

- **Grounded Answers**: Every answer is grounded in the filing with explicit citations
- **Citation Traceability**: Full source metadata for verification
- **Multi-Provider LLM**: Supports Ollama, OpenAI, Anthropic, Google
- **Hybrid Retrieval**: BM25 + Dense embeddings + Cross-encoder reranking
- **Rate Limiting**: Per-IP and global rate limits with Redis backend

## Available Filings

- Apple Inc. (AAPL) - 10-K FY2025
- Microsoft Corporation (MSFT) - 10-K FY2026
- Tesla, Inc. (TSLA) - 10-K FY2025

## Example Question

```json
{
  "question": "What was Apple's total net sales in fiscal 2025?",
  "doc_id": "AAPL_10-K_2025"
}
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if api_settings.is_prod else "/docs",
    redoc_url=None if api_settings.is_prod else "/redoc",
    openapi_url=None if api_settings.is_prod else "/openapi.json",
)

# Add rate limit exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# MIDDLEWARE
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Correlation-ID", "Authorization"],
)

# Add auth middleware to extract user info from JWT (optional, enables per-user rate limiting)
if api_settings.supabase_auth_enabled:
    app.add_middleware(AuthMiddleware)
    logger.info("Supabase JWT authentication: ENABLED (per-user rate limiting)")
else:
    logger.info("Supabase JWT authentication: DISABLED (SUPABASE_URL / SUPABASE_JWT_SECRET not set; rate limiting uses IP)")

# Add correlation ID middleware for request tracking
app.add_middleware(CorrelationIDMiddleware)

# Add request logging middleware (debug level only)
if api_settings.ENVIRONMENT == "dev":
    app.add_middleware(RequestLoggingMiddleware)


# =============================================================================
# ROUTERS
# =============================================================================

app.include_router(health.router, tags=["Health"])
app.include_router(filings.router, tags=["Filings"], dependencies=[Depends(require_api_key)])
app.include_router(questions.router, tags=["Questions"], dependencies=[Depends(require_api_key)])


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/", include_in_schema=False)
def root():
    """Redirect to docs."""
    return {
        "message": "TechFiling Copilot API",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=api_settings.API_HOST,
        port=api_settings.API_PORT,
        reload=True,
    )
