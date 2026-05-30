"""TranscriberAgent (F1): audio -> TextChunks via TranscriberAdapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents._chunk_stamp import stamp_chunk
from src.agents.base import AgentBase
from src.core.schemas import Operation

if TYPE_CHECKING:
    from src.adapters.transcriber import TranscriberAdapter
    from src.core.bus import RedisStreamBus
    from src.core.messages import Message


class TranscriberAgent(AgentBase):
    name = "TranscriberAgent"

    def __init__(self, *, bus: RedisStreamBus, transcriber: TranscriberAdapter) -> None:
        super().__init__(
            bus=bus,
            channel="agent.transcriber",
            group="worker-transcriber",
            operation=Operation.F1_TRANSCRIBE,
        )
        self._transcriber = transcriber

    async def handle(self, message: Message) -> Message | None:
        file_path = message.content.get("file_path")
        if not isinstance(file_path, str):
            return self._refuse(message, reason="missing or invalid file_path")
        document_id = message.content.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            return self._refuse(message, reason="missing or invalid document_id")
        language = message.content.get("language", "ru")
        if not isinstance(language, str):
            language = "ru"
        chunks = await self._transcriber.transcribe(file_path=file_path, language=language)
        stamped = [
            stamp_chunk(chunk, task_id=message.task_id, document_id=document_id, index=index)
            for index, chunk in enumerate(chunks)
        ]
        return self._inform(
            message,
            content={"chunks": [chunk.model_dump() for chunk in stamped], "language": language},
        )
