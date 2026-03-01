"""Checkpointer port — ABC for persisting thread state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from suluv.core.agent.thread import Thread


class Checkpointer(ABC):
    """Persist and retrieve conversation threads.

    Implementations can store threads in memory, SQLite, PostgreSQL,
    Redis, etc.  The agent calls these methods automatically when a
    ``thread_id`` is provided.

    Lifecycle::

        thread = await checkpointer.get("t1")  # load or None
        if thread is None:
            thread = Thread(thread_id="t1")
        # … agent appends messages …
        await checkpointer.put(thread)          # persist
    """

    @abstractmethod
    async def get(self, thread_id: str) -> Thread | None:
        """Load a thread by ID.  Return None if not found."""
        ...

    @abstractmethod
    async def put(self, thread: Thread) -> None:
        """Save (create or update) a thread."""
        ...

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        """Delete a thread by ID."""
        ...

    @abstractmethod
    async def list(
        self,
        *,
        limit: int = 50,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Thread]:
        """List threads, optionally filtered by metadata.

        Returns the most recently updated threads first.
        """
        ...
