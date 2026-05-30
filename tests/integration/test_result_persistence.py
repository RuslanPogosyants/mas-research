"""Integration: agent results persist into the four output tables."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.agents.store import DbTaskStore
from src.core.schemas import Operation
from src.db.models import Base, CitationRow, QuizRow, SummaryRow, TermRow
from src.db.repos import TaskRepo


@pytest.mark.integration
class TestResultPersistence:
    async def test_results_persist_into_tables(self, postgres_url: str) -> None:
        engine = create_async_engine(postgres_url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                await TaskRepo(session).create(task_id="task-1", requested_outputs=[Operation.F3_SUMMARIZE])
                await session.commit()

            store = DbTaskStore(session_factory)
            await store.save_result(
                "task-1",
                Operation.F3_SUMMARIZE,
                {"summary_id": "sum-1", "sections": [{"type": "thesis", "text": "T"}], "source_chunk_ids": []},
            )
            await store.save_result(
                "task-1", Operation.F4_TEST, {"quiz_id": "quiz-1", "questions": [{"q": 1}], "difficulty": "easy"}
            )
            await store.save_result("task-1", Operation.F5_TERMS, {"terms": [{"term": "граф", "frequency": 1}]})
            await store.save_result(
                "task-1", Operation.F6_RECOMMEND, {"citations": [{"title": "Graphs", "relevance_score": 0.5}]}
            )

            async with session_factory() as session:
                summary = (await session.execute(select(SummaryRow))).scalar_one()
                assert summary.key_points == "T"  # thesis -> key_points
                assert (await session.execute(select(QuizRow))).scalar_one().difficulty == "easy"
                assert (await session.execute(select(TermRow))).scalar_one().term == "граф"
                assert (await session.execute(select(CitationRow))).scalar_one().title == "Graphs"

            await store.save_result(
                "task-1",
                Operation.F3_SUMMARIZE,
                {"summary_id": "sum-1", "sections": [{"type": "thesis", "text": "T2"}], "source_chunk_ids": []},
            )
            async with session_factory() as session:
                summaries = (await session.execute(select(SummaryRow))).scalars().all()
                assert len(summaries) == 1
                assert summaries[0].key_points == "T2"
        finally:
            await engine.dispose()
