"""
Pytest configuration and fixtures for TechFiling Copilot tests.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def client():
    """
    Create a test client for the FastAPI application.
    Uses module scope to avoid reinitializing the heavy RAG pipeline
    for each test function.
    """
    from api.config import api_settings
    from api.main import app

    headers = {"X-API-Key": api_settings.API_ACCESS_KEY} if api_settings.API_ACCESS_KEY else {}
    with TestClient(app, headers=headers) as test_client:
        yield test_client


@pytest.fixture
def mock_filings():
    """Sample filing data for testing."""
    return [
        {
            "doc_id": "AAPL_10-K_2025",
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "form": "10-K",
            "fiscal_year": 2025,
            "period_end": "September 27, 2025",
        },
        {
            "doc_id": "MSFT_10-K_2026",
            "company": "Microsoft Corporation",
            "ticker": "MSFT",
            "form": "10-K",
            "fiscal_year": 2026,
            "period_end": "June 30, 2026",
        },
        {
            "doc_id": "TSLA_10-K_2025",
            "company": "Tesla, Inc.",
            "ticker": "TSLA",
            "form": "10-K",
            "fiscal_year": 2025,
            "period_end": "December 31, 2025",
        },
    ]


@pytest.fixture
def sample_question_request():
    """Sample question request for testing."""
    return {
        "question": "What was Apple's total net sales in fiscal 2025?",
        "doc_id": "AAPL_10-K_2025",
    }
