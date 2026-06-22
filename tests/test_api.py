"""API endpoint tests.

Tests:
  - GET /health returns valid response
  - GET /metrics returns valid response
  - Response schemas match specification
"""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app, set_pipeline
from config.settings import Settings
from streaming.pipeline import StreamingPipeline


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def client(settings: Settings) -> TestClient:
    """Create a test client with initialized pipeline."""
    pipeline = StreamingPipeline(settings)
    set_pipeline(pipeline)
    app = create_app(settings)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client: TestClient) -> None:
        """Health response should have required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "stream_active" in data
        assert "buffer_warm" in data
        assert "events_in_queue" in data

    def test_health_initial_state(self, client: TestClient) -> None:
        """Initially model is not loaded and stream is not active."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is False
        assert data["stream_active"] is False
        assert data["buffer_warm"] is False


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_200(self, client: TestClient) -> None:
        """Metrics endpoint should return 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_schema(self, client: TestClient) -> None:
        """Metrics response should have required fields."""
        response = client.get("/metrics")
        data = response.json()
        assert "load_metrics" in data
        assert "price_metrics" in data
        assert "latency" in data
        assert "events_processed" in data
        assert "uptime_seconds" in data

    def test_metrics_initial_zeros(self, client: TestClient) -> None:
        """Initial metrics should be zero."""
        response = client.get("/metrics")
        data = response.json()
        assert data["events_processed"] == 0


class TestStreamStatusEndpoint:
    """Tests for GET /stream/status."""

    def test_stream_status_returns_200(self, client: TestClient) -> None:
        """Stream status endpoint should return 200."""
        response = client.get("/stream/status")
        assert response.status_code == 200

    def test_stream_status_schema(self, client: TestClient) -> None:
        """Stream status response should have required fields."""
        response = client.get("/stream/status")
        data = response.json()
        assert "events_emitted" in data
        assert "events_remaining" in data
        assert "progress_pct" in data
        assert "rate_limiter" in data
