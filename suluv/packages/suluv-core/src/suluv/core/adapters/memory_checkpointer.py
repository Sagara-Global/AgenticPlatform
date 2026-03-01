"""InMemoryCheckpointer — dict-backed thread persistence for dev/test."""

from __future__ import annotations

from typing import Any

from suluv.core.agent.thread import Thread
from suluv.core.ports.checkpointer import Checkpointer


class InMemoryCheckpointer(Checkpointer):
    """In-memory checkpointer backed by a dict.

    Suitable for development, testing, and single-process use.
    For production, swap for a database-backed checkpointer.
    """

    def __init__(self) -> None:
        self._threads: dict[str, Thread] = {}

    async def get(self, thread_id: str) -> Thread | None:
        return self._threads.get(thread_id)

    async def put(self, thread: Thread) -> None:
        self._threads[thread.thread_id] = thread

    async def delete(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)

    async def list(
        self,
        *,
        limit: int = 50,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Thread]:
        threads = list(self._threads.values())

        # Filter by metadata
        if metadata_filter:
            filtered = []
            for t in threads:
                if all(t.metadata.get(k) == v for k, v in metadata_filter.items()):
                    filtered.append(t)
            threads = filtered

        # Sort by updated_at descending
        threads.sort(key=lambda t: t.updated_at, reverse=True)
        return threads[:limit]

    @property
    def thread_count(self) -> int:
        """Number of threads stored (useful for tests)."""
        return len(self._threads)

    def clear(self) -> None:
        """Remove all threads."""
        self._threads.clear()
