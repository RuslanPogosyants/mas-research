"""E2E: an in-flight task resumes and completes after a coordinator restart.

Run 1: F4 (test_generator) is wedged via HANG_AGENT, so the task reaches RUNNING
with F1/F2/F3/F5/F6 persisted and F4 still pending — then the lifespan is torn
down (simulating a crash). Run 2: HANG is cleared and a fresh lifespan starts;
recovery reloads the persisted results and re-drives only F4, completing the task.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.config import get_settings
from src.db.models import SummaryRow
from src.main import app


async def _summary_exists(task_id: str) -> bool:
    engine = create_async_engine(get_settings().database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            row = (await session.execute(select(SummaryRow).where(SummaryRow.task_id == task_id))).scalar_one_or_none()
            return row is not None
    finally:
        await engine.dispose()


@pytest.mark.e2e
class TestRecoveryAfterRestart:
    async def test_in_flight_task_resumes_after_app_restart(self) -> None:
        os.environ["HANG_AGENT"] = "F4"
        try:
            async with (
                LifespanManager(app),
                AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
            ):
                post = await client.post(
                    "/api/tasks",
                    files=[
                        ("files", ("lecture.mp3", b"FAKE_AUDIO", "audio/mpeg")),
                        ("files", ("paper.pdf", b"FAKE_PDF", "application/pdf")),
                    ],
                    data={"ops": ["F1", "F2", "F3", "F4", "F5", "F6"]},
                )
                assert post.status_code == 202
                task_id = post.json()["task_id"]

                # Wait until mid-flight: F3 persisted (so reload has real work) and
                # the task is still RUNNING because F4 is wedged.
                for _ in range(60):
                    status = (await client.get(f"/api/tasks/{task_id}")).json()["status"]
                    if status == "running" and await _summary_exists(task_id):
                        break
                    await asyncio.sleep(0.5)
                else:
                    pytest.fail("task did not reach a persisted RUNNING state in time")
                assert (await client.get(f"/api/tasks/{task_id}")).json()["status"] == "running"
            # First lifespan exited here == simulated crash.

            # Restart with F4 healthy; recovery must resume and complete the task.
            os.environ.pop("HANG_AGENT", None)
            async with (
                LifespanManager(app),
                AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
            ):
                for _ in range(60):
                    status = (await client.get(f"/api/tasks/{task_id}")).json()["status"]
                    if status in ("completed", "partial_ready", "failed"):
                        break
                    await asyncio.sleep(0.5)
                final = (await client.get(f"/api/tasks/{task_id}")).json()
                assert final["status"] == "completed"

                result = (await client.get(f"/api/tasks/{task_id}/result")).json()
                assert result["result"]["summary"] is not None
                assert len(result["result"]["quiz"]) > 0  # F4 was re-driven by recovery
        finally:
            os.environ.pop("HANG_AGENT", None)
