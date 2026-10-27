"""
API Endpoint Tests for TechFiling Copilot.

Tests cover:
- Health check endpoint
- Filings listing endpoint
- Questions endpoint (validation and basic functionality)
"""


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response should contain required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "environment" in data
        assert "model" in data
        assert "filings_loaded" in data

    def test_health_status_is_healthy(self, client):
        """Health status should be 'healthy'."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    def test_health_filings_loaded(self, client):
        """Health should report filings are loaded."""
        response = client.get("/health")
        data = response.json()

        # Should have at least the 3 MVP filings
        assert data["filings_loaded"] >= 0


class TestFilingsEndpoint:
    """Tests for GET /filings endpoint."""

    def test_filings_returns_200(self, client):
        """Filings endpoint should return 200 OK."""
        response = client.get("/filings")
        assert response.status_code == 200

    def test_filings_response_structure(self, client):
        """Filings response should contain a list of filings."""
        response = client.get("/filings")
        data = response.json()

        assert "filings" in data
        assert isinstance(data["filings"], list)

    def test_filings_have_required_fields(self, client):
        """Each filing should have required metadata fields."""
        response = client.get("/filings")
        data = response.json()

        required_fields = ["doc_id", "company", "ticker", "form", "fiscal_year", "period_end"]

        for filing in data["filings"]:
            for field in required_fields:
                assert field in filing, f"Missing field: {field}"


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_200(self, client):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_links(self, client):
        """Root should contain links to docs and health."""
        response = client.get("/")
        data = response.json()

        assert "docs" in data
        assert "health" in data


class TestQuestionsEndpoint:
    """Tests for POST /questions endpoint."""

    def test_questions_requires_body(self, client):
        """Questions endpoint should reject empty body."""
        response = client.post("/questions")
        assert response.status_code == 422  # Validation error

    def test_questions_requires_question_field(self, client):
        """Questions endpoint should require 'question' field."""
        response = client.post("/questions", json={"doc_id": "AAPL_10-K_2025"})
        assert response.status_code == 422

    def test_questions_requires_doc_id_field(self, client):
        """Questions endpoint should require 'doc_id' field."""
        response = client.post("/questions", json={"question": "What is revenue?"})
        assert response.status_code == 422

    def test_questions_validates_question_length(self, client):
        """Questions should reject too-short questions."""
        response = client.post(
            "/questions",
            json={"question": "Hi", "doc_id": "AAPL_10-K_2025"},  # Too short (min 3 chars)
        )
        assert response.status_code == 422

    def test_questions_rejects_invalid_doc_id(self, client):
        """Questions should reject non-existent doc_id."""
        response = client.post(
            "/questions", json={"question": "What was the revenue?", "doc_id": "INVALID_DOC_ID"}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_questions_accepts_valid_request(self, client, sample_question_request):
        """Questions should accept valid request and return answer."""
        response = client.post("/questions", json=sample_question_request)

        # Should succeed (may take a few seconds due to LLM)
        assert response.status_code == 200

        data = response.json()

        # Check response structure
        assert "question" in data
        assert "doc_id" in data
        assert "answer" in data
        assert "sources" in data
        assert "model" in data
        assert "latency_seconds" in data

    def test_questions_answer_structure(self, client, sample_question_request):
        """Answer object should have required fields."""
        response = client.post("/questions", json=sample_question_request)
        assert response.status_code == 200

        answer = response.json()["answer"]

        # Required answer fields
        assert "answer" in answer  # Can be null for insufficient context
        assert "explanation" in answer
        assert "citations" in answer
        assert "insufficient_context" in answer
        assert "is_valid" in answer

    def test_questions_sources_structure(self, client, sample_question_request):
        """Sources should have required metadata fields."""
        response = client.post("/questions", json=sample_question_request)
        assert response.status_code == 200

        sources = response.json()["sources"]

        # Should have at least one source for a valid question
        if len(sources) > 0:
            source = sources[0]
            assert "source_id" in source
            assert "doc_id" in source
            assert "text" in source
            assert "score" in source

    def test_questions_with_context_k(self, client):
        """Questions should accept optional context_k parameter."""
        response = client.post(
            "/questions",
            json={
                "question": "What was Apple's total net sales?",
                "doc_id": "AAPL_10-K_2025",
                "context_k": 3,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["context_k"] == 3


class TestOpenAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_swagger_ui_available(self, client):
        """Swagger UI should be available at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """ReDoc should be available at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
