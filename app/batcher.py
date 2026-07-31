import asyncio
from dataclasses import dataclass, field

from app.config import settings
from app.model import ProteinEmbeddingModel


@dataclass
class _PendingRequest:
    sequence: str
    future: asyncio.Future = field(default=None)


class DynamicBatcher:
    """Collects concurrent single-sequence requests into batches and runs
    one model forward pass per batch, so throughput under concurrent load
    doesn't scale linearly with per-request inference cost.

    A background worker pulls from the queue, waiting up to
    `max_batch_wait_ms` (or until `max_batch_size` requests have queued)
    before running a batched forward pass.
    """

    def __init__(self, model: ProteinEmbeddingModel):
        self.model = model
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def predict(self, sequence: str) -> list[float]:
        req = _PendingRequest(sequence=sequence, future=asyncio.get_event_loop().create_future())
        await self._queue.put(req)
        return await req.future

    async def _run(self) -> None:
        while True:
            batch: list[_PendingRequest] = [await self._queue.get()]

            wait_seconds = settings.max_batch_wait_ms / 1000
            deadline = asyncio.get_event_loop().time() + wait_seconds
            while len(batch) < settings.max_batch_size:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            self._process_batch(batch)

    def _process_batch(self, batch: list[_PendingRequest]) -> None:
        try:
            embeddings = self.model.embed_batch([r.sequence for r in batch])
            for req, embedding in zip(batch, embeddings):
                if not req.future.done():
                    req.future.set_result(embedding)
        except Exception as exc:  # noqa: BLE001 - propagate to all callers in the batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
