"""InMemoryEventBus — simple in-process event bus."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Awaitable

from suluv.core.ports.event_bus import EventBus


class InMemoryEventBus(EventBus):
    """In-memory event bus using async callbacks."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict[str, Any]], Awaitable[None]]]] = defaultdict(list)
        self._history: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        self._history.append((topic, event))
        handlers = list(self._handlers.get(topic, []))
        for handler in handlers:
            await handler(event)

    async def subscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._handlers[topic].append(handler)

    async def unsubscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        if topic in self._handlers:
            self._handlers[topic] = [h for h in self._handlers[topic] if h is not handler]

    async def request(
        self, topic: str, event: dict[str, Any], timeout: float = 30.0
    ) -> dict[str, Any]:
        result_future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        reply_topic = f"{topic}.reply.{id(result_future)}"

        async def reply_handler(reply: dict[str, Any]) -> None:
            if not result_future.done():
                result_future.set_result(reply)

        await self.subscribe(reply_topic, reply_handler)
        event["_reply_topic"] = reply_topic
        await self.publish(topic, event)

        try:
            return await asyncio.wait_for(result_future, timeout=timeout)
        finally:
            await self.unsubscribe(reply_topic, reply_handler)

    @property
    def history(self) -> list[tuple[str, dict[str, Any]]]:
        """Access event history for testing."""
        return list(self._history)

    def clear(self) -> None:
        """Clear all handlers and history."""
        self._handlers.clear()
        self._history.clear()
