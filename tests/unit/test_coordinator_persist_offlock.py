"""Persistence must run outside the coordinator's dispatch lock (no head-of-line block)."""

from __future__ import annotations

from typing import Any

from src.agents.coordinator import Coordinator
from src.core.bus import channel_for_agent
from src.core.messages import Message, Performative, make_message
from src.core.schemas import Document, DocumentType, Operation, Task, TaskStatus

from tests.support.fake_bus import FakeBus, FakeTaskStore


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


_FAST_TIMEOUTS = {
    "transcriber": 5.0,
    "ocr": 5.0,
    "summarizer": 5.0,
    "test_generator": 5.0,
    "terminology": 5.0,
    "recommender": 2.0,
}


def _inform(request: Message, content: dict[str, object]) -> Message:
    return make_message(
        performative=Performative.INFORM,
        sender=request.receiver,
        receiver="CoordinatorAgent",
        task_id=request.task_id,
        conversation_id=request.conversation_id,
        content=content,
        in_reply_to=request.message_id,
        subtask_id=request.subtask_id,
    )


async def test_persist_runs_outside_dispatch_lock() -> None:
    lock_states: list[bool] = []

    class _RecordingStore(FakeTaskStore):
        """Behaves like FakeTaskStore but records the dispatch-lock state at save_result time."""

        async def save_result(self, task_id: str, operation: Any, content: dict[str, Any]) -> None:
            lock_states.append(coordinator._lock.locked())
            await super().save_result(task_id, operation, content)

    bus, store, clock = FakeBus(), _RecordingStore(), Clock()
    coordinator = Coordinator(bus=bus, store=store, agent_timeouts=_FAST_TIMEOUTS, clock=clock)

    task = Task(
        id="task-1",
        status=TaskStatus.PLANNING,
        requested_outputs=[Operation.F1_TRANSCRIBE],
        conversation_id="conv-task-1",
        documents=[Document(id="task-1-a", task_id="task-1", document_type=DocumentType.AUDIO, file_path="/x.mp3")],
    )
    await coordinator.submit(task)
    f1_request = bus.requests_for(channel_for_agent("transcriber"))[0]
    bus.feed_inbox(_inform(f1_request, {"chunks": [{"content": "lecture"}]}))
    await coordinator._tick()

    assert lock_states == [False], "save_result must not run while the dispatch lock is held"
