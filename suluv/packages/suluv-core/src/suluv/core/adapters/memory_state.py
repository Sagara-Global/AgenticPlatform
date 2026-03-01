"""InMemoryStateStore — dict-backed state persistence."""

from __future__ import annotations

from typing import Any

from suluv.core.ports.state_store import StateStore


class InMemoryStateStore(StateStore):
    """In-memory state store backed by a dict."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def save(self, key: str, state: dict[str, Any]) -> None:
        self._store[key] = state

    async def load(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()
