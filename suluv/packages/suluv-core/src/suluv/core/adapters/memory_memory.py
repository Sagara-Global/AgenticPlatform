"""In-memory memory backends — all 4 tiers."""

from __future__ import annotations

from typing import Any

from suluv.core.ports.memory_backend import (
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
    SemanticMemory,
    MemoryEntry,
)


class InMemoryShortTermMemory(ShortTermMemory):
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    async def get(self, key: str) -> Any | None:
        return self._store.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    async def clear(self) -> None:
        self._store.clear()

    async def all(self) -> dict[str, Any]:
        return dict(self._store)


class InMemoryLongTermMemory(LongTermMemory):
    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def get(self, key: str, scope: str = "default") -> Any | None:
        return self._store.get(scope, {}).get(key)

    async def set(self, key: str, value: Any, scope: str = "default") -> None:
        if scope not in self._store:
            self._store[scope] = {}
        self._store[scope][key] = value

    async def delete(self, key: str, scope: str = "default") -> None:
        if scope in self._store:
            self._store[scope].pop(key, None)

    async def list_keys(self, scope: str = "default") -> list[str]:
        return list(self._store.get(scope, {}).keys())


class InMemoryEpisodicMemory(EpisodicMemory):
    def __init__(self) -> None:
        self._episodes: list[dict[str, Any]] = []

    async def store(self, episode: dict[str, Any]) -> None:
        self._episodes.append(episode)

    async def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        # Simple: return most recent episodes (no semantic matching)
        return list(reversed(self._episodes[-limit:]))

    async def clear(self) -> None:
        self._episodes.clear()


class InMemorySemanticMemory(SemanticMemory):
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    async def store(self, key: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._entries[key] = MemoryEntry(
            key=key, value=text, metadata=metadata or {}
        )

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        # Simple keyword match (production would use embeddings)
        query_lower = query.lower()
        scored = []
        for entry in self._entries.values():
            text = str(entry.value).lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                entry.score = score
                scored.append(entry)
        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:limit]

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)
