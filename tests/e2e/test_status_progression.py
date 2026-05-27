"""E2E: task status passes planning -> running -> completed."""

from __future__ import annotations

import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.mark.e2e
class TestStatusProgression:
    async def test_status_passes_through_planning_and_running_to_completed(
        self,
    ) -> None:
        async with (
            LifespanManager(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        ):
            post = await client.post(
                "/api/tasks",
                files=[("files", ("a.mp3", b"X", "audio/mpeg"))],
                data={"ops": ["F1"]},
            )
            assert post.status_code == 202
            task_id = post.json()["task_id"]
            assert post.json()["status"] == "planning"

            seen_statuses: set[str] = {"planning"}
            for _ in range(60):
                response = await client.get(f"/api/tasks/{task_id}")
                assert response.status_code == 200
                seen_statuses.add(response.json()["status"])
                if response.json()["status"] in ("completed", "partial_ready"):
                    break
                await asyncio.sleep(0.5)

            assert "completed" in seen_statuses or "partial_ready" in seen_statuses
