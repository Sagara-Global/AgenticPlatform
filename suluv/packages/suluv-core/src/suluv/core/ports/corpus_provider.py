"""CorpusProvider — document/knowledge search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A chunk of text from a corpus search."""

    text: str
    source: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CorpusProvider(ABC):
    """Port for searching a knowledge corpus (RAG)."""

    @abstractmethod
    async def search(
        self, query: str, context: dict[str, Any] | None = None, limit: int = 5
    ) -> list[Chunk]:
        """Search the corpus and return relevant chunks."""
        ...
