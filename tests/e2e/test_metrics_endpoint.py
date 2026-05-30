"""E2E: the /metrics endpoint serves the Prometheus registry."""

from __future__ import annotations

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.mark.e2e
class TestMetricsEndpoint:
    async def test_metrics_endpoint_exposes_registry(self) -> None:
        async with (
            LifespanManager(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        ):
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert "text/plain" in response.headers["content-type"]
            assert "mas_tasks_total" in response.text or "mas_agent_handle_seconds" in response.text
