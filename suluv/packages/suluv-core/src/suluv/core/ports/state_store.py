"""StateStore — persistence for execution state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StateStore(ABC):
    """Port for persisting execution state (graph runs, process instances)."""

    @abstractmethod
    async def save(self, key: str, state: dict[str, Any]) -> None:
        """Save state by key."""
        ...

    @abstractmethod
    async def load(self, key: str) -> dict[str, Any] | None:
        """Load state by key. Returns None if not found."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete state by key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return (await self.load(key)) is not None
