"""Integration: recovery-oriented repository reads over a real Postgres."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.core.schemas import DocumentType, Operation, TaskStatus
from src.db.models import Base
from src.db.repos import DocumentRepo, ResultRepo, TaskRepo


@pytest_asyncio.fixture
async def session_factory(postgres_url: str):
    engine = create_async_engine(postgres_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.mark.integration
async def test_find_by_status_returns_only_matching(session_factory) -> None:
    async with session_factory() as session:
        repo = TaskRepo(session)
        await repo.create(task_id="run-1", requested_outputs=[Operation.F1_TRANSCRIBE])
        await repo.create(task_id="done-1", requested_outputs=[Operation.F1_TRANSCRIBE])
        await repo.update_status("run-1", TaskStatus.RUNNING)
        await repo.update_status("done-1", TaskStatus.COMPLETED)
        await session.commit()
    async with session_factory() as session:
        rows = await TaskRepo(session).find_by_status({TaskStatus.RUNNING, TaskStatus.PLANNING})
        assert {r.id for r in rows} == {"run-1"}


@pytest.mark.integration
async def test_result_reads_round_trip(session_factory) -> None:
    async with session_factory() as session:
        await TaskRepo(session).create(task_id="t", requested_outputs=[Operation.F1_TRANSCRIBE, Operation.F5_TERMS])
        await DocumentRepo(session).create(
            document_id="doc-t-0", task_id="t", document_type=DocumentType.AUDIO, file_path="/a.mp3"
        )
        await session.commit()
    async with session_factory() as session:
        repo = ResultRepo(session)
        await repo.save_chunks(
            "t",
            {
                "chunks": [
                    {
                        "id": "chunk-doc-t-0-0",
                        "task_id": "t",
                        "document_id": "doc-t-0",
                        "source_type": "audio",
                        "content": "c",
                        "chunk_index": 0,
                        "confidence": None,
                        "meta": {},
                    },
                ]
            },
        )
        await repo.save_summary(
            "t", {"summary_id": "sum-t", "sections": [{"type": "thesis", "text": "x"}], "source_chunk_ids": []}
        )
        await repo.save_terms(
            "t",
            {
                "terms": [
                    {
                        "term": "A",
                        "lemma": "a",
                        "frequency": 1,
                        "category": "general",
                        "source_chunk_id": "chunk-doc-t-0-0",
                    }
                ]
            },
        )
        await session.commit()
    async with session_factory() as session:
        repo = ResultRepo(session)
        assert [r.id for r in await repo.list_chunk_rows("t")] == ["chunk-doc-t-0-0"]
        assert (await repo.get_summary_row("t")).key_points == "x"
        assert [r.term for r in await repo.list_term_rows("t")] == ["A"]
        assert await repo.get_quiz_row("t") is None
        assert await repo.list_citation_rows("t") == []
