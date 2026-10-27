"""
Filings Router.

Provides endpoints for listing available SEC filings.
"""

from fastapi import APIRouter, Depends

from api.config import api_settings
from api.dependencies import get_filings
from api.rate_limit import rate_limit_dependency
from api.schemas import FilingInfo, FilingsResponse

router = APIRouter()


@router.get(
    "/filings",
    response_model=FilingsResponse,
    summary="List available filings",
    description="Get a list of all SEC 10-K filings available for querying",
    responses={
        200: {"description": "Successful response with filings list"},
        429: {"description": "Rate limit exceeded"},
    },
    dependencies=[
        Depends(rate_limit_dependency(
            limit=api_settings.RATE_LIMIT_GLOBAL_PER_MINUTE,
            period=60,
            scope="global"
        )),
        Depends(rate_limit_dependency(
            limit=api_settings.RATE_LIMIT_FILINGS_PER_MINUTE,
            period=60,
            scope="ip"
        ))
    ],
)
def list_filings(
    filings: list[dict] = Depends(get_filings),
) -> FilingsResponse:
    """
    List all available filings.

    Returns metadata for each filing including company name, ticker,
    fiscal year, and period end date.
    """
    filing_infos = [
        FilingInfo(
            doc_id=f["doc_id"],
            company=f["company"],
            ticker=f["ticker"],
            form=f["form"],
            fiscal_year=f["fiscal_year"],
            period_end=f["period_end"],
        )
        for f in filings
    ]

    return FilingsResponse(filings=filing_infos)
