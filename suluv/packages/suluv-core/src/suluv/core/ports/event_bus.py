"""EventBus — publish/subscribe event backbone."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable


class EventBus(ABC):
    """Port for event-driven communication between components."""

    @abstractmethod
    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event to a topic."""
        ...

    @abstractmethod
    async def subscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Subscribe a handler to a topic."""
        ...

    async def request(
        self, topic: str, event: dict[str, Any], timeout: float = 30.0
    ) -> dict[str, Any]:
        """Publish an event and wait for a response (request/reply pattern)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support request/reply"
        )

    async def unsubscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Remove a handler from a topic. Optional."""
        raise NotImplementedError
