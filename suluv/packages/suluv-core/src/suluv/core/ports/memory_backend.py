"""Memory backends — 4 tiers of agent memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry."""

    key: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # relevance score for semantic search


class ShortTermMemory(ABC):
    """In-session memory — auto-cleared on session end."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        ...

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...

    @abstractmethod
    async def all(self) -> dict[str, Any]:
        ...


class LongTermMemory(ABC):
    """Cross-session persistent memory (user/org scoped)."""

    @abstractmethod
    async def get(self, key: str, scope: str = "default") -> Any | None:
        ...

    @abstractmethod
    async def set(self, key: str, value: Any, scope: str = "default") -> None:
        ...

    @abstractmethod
    async def delete(self, key: str, scope: str = "default") -> None:
        ...

    @abstractmethod
    async def list_keys(self, scope: str = "default") -> list[str]:
        ...


class EpisodicMemory(ABC):
    """Past interaction recall — stores interaction summaries."""

    @abstractmethod
    async def store(self, episode: dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def recall(
        self, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...


class SemanticMemory(ABC):
    """Vector similarity search over stored knowledge."""

    @abstractmethod
    async def store(self, key: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        ...
