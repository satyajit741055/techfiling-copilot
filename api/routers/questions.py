"""
Questions Router.

Provides the core Q&A endpoint for asking questions about SEC filings.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.config import api_settings
from api.dependencies import get_rag_service, get_valid_doc_ids
from api.rate_limit import rate_limit_dependency
from api.schemas import QuestionRequest, QuestionResponse
from api.services import RAGService
from observability import get_last_record
from observability.records import set_last_record, set_request_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/questions",
    response_model=QuestionResponse,
    summary="Ask a question",
    description=(
        "Ask a question about SEC 10-K filings and get a grounded answer with citations. "
        "Scope it to one filing (doc_id), metadata filters (ticker, fiscal_year, ...), "
        "or compare groups (group_by + groups)."
    ),
    responses={
        200: {"description": "Successful response with answer and citations"},
        404: {"description": "Filing not found"},
        422: {"description": "Validation error in request"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Generation failed"},
    },
    dependencies=[
        Depends(
            rate_limit_dependency(
                limit=api_settings.RATE_LIMIT_GLOBAL_PER_MINUTE, period=60, scope="global"
            )
        ),
        Depends(
            rate_limit_dependency(
                limit=api_settings.RATE_LIMIT_QUESTIONS_PER_MINUTE, period=60, scope="ip"
            )
        ),
    ],
)
def ask_question(
    http_request: Request,
    http_response: Response,
    request: QuestionRequest,
    rag_service: RAGService = Depends(get_rag_service),
    valid_doc_ids: set[str] = Depends(get_valid_doc_ids),
) -> QuestionResponse:
    """
    Ask a question about a SEC 10-K filing.

    The question is processed through the RAG pipeline:
    1. Retrieve relevant chunks using hybrid search
    2. Rerank with cross-encoder
    3. Generate grounded answer with LLM
    4. Validate citations

    Returns the answer with full citation metadata for verification.
    """
    # Validate requested filing IDs exist
    requested_doc_ids = [request.doc_id] if request.doc_id else []
    if request.group_by == "doc_id" and request.groups:
        requested_doc_ids += [str(g) for g in request.groups]
    unknown = [d for d in requested_doc_ids if d not in valid_doc_ids]
    if unknown:
        raise HTTPException(
            status_code=404,
            detail=f"Filing not found: {', '.join(unknown)}. Available filings: {sorted(valid_doc_ids)}",
        )

    try:
        # Call the RAG pipeline
        set_last_record(None)
        set_request_id(getattr(http_request.state, "correlation_id", None))
        response = rag_service.answer(request)
        record = get_last_record()
        if record is not None:
            http_response.headers["Server-Timing"] = (
                f"retrieval;dur={record.retrieval_seconds * 1000:.1f}, "
                f"generation;dur={record.generation_seconds * 1000:.1f}, "
                f"total;dur={record.total_seconds * 1000:.1f}"
            )
            http_response.headers["X-Cold-Start"] = str(record.is_cold).lower()
            http_response.headers["X-LLM-Attempts"] = str(record.llm_attempts)
        return response

    except Exception as e:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        logger.error(
            f"Generation failed [correlation_id={correlation_id}]: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Unable to generate an answer right now. Please try again shortly. (Reference: {correlation_id})",
        )
