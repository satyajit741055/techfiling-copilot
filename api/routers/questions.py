"""
Questions Router.

Provides the core Q&A endpoint for asking questions about SEC filings.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_rag_service, get_valid_doc_ids
from api.schemas import QuestionRequest, QuestionResponse
from api.services import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/questions",
    response_model=QuestionResponse,
    summary="Ask a question",
    description="Ask a question about a specific SEC 10-K filing and get a grounded answer with citations",
    responses={
        200: {"description": "Successful response with answer and citations"},
        404: {"description": "Filing not found"},
        422: {"description": "Validation error in request"},
        503: {"description": "Generation failed"},
    },
)
def ask_question(
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
    # Validate doc_id exists
    if request.doc_id not in valid_doc_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Filing not found: {request.doc_id}. Available filings: {sorted(valid_doc_ids)}"
        )

    try:
        # Call the RAG pipeline
        response = rag_service.answer(request)
        return response

    except Exception as e:
        logger.error(f"Generation failed: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Generation failed: {str(e)}"
        )
