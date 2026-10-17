"""
Filings Router.

Provides endpoints for listing available SEC filings.
"""

from fastapi import APIRouter, Depends

from api.dependencies import get_filings
from api.schemas import FilingInfo, FilingsResponse


router = APIRouter()


@router.get(
    "/filings",
    response_model=FilingsResponse,
    summary="List available filings",
    description="Get a list of all SEC 10-K filings available for querying",
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
