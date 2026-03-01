"""LLMBackend — abstract interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from suluv.core.messages.content import ContentBlock
from suluv.core.messages.prompt import SuluvPrompt


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""

    content: list[ContentBlock] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        from suluv.core.messages.content import ContentType

        return "".join(
            b.text for b in self.content if b.type == ContentType.TEXT and b.text
        )


class LLMBackend(ABC):
    """Port for LLM providers (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def complete(self, prompt: SuluvPrompt) -> LLMResponse:
        """Send a prompt and get a complete response."""
        ...

    @abstractmethod
    async def stream(self, prompt: SuluvPrompt) -> AsyncIterator[str]:
        """Stream text chunks from the LLM."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector. Optional — not all backends support this."""
        raise NotImplementedError(f"{type(self).__name__} does not support embeddings")
