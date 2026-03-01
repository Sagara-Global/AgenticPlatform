"""MockLLM — test double for LLM backends."""

from __future__ import annotations

from typing import Any, AsyncIterator

from suluv.core.messages.content import ContentBlock
from suluv.core.messages.prompt import SuluvPrompt
from suluv.core.ports.llm_backend import LLMBackend, LLMResponse


class MockLLM(LLMBackend):
    """Mock LLM backend for testing.

    Configure with a list of responses that are returned in order,
    or a callable for dynamic responses.
    """

    def __init__(
        self,
        responses: list[str | LLMResponse] | None = None,
        default_response: str = "Mock LLM response",
    ) -> None:
        self._responses = list(responses or [])
        self._default = default_response
        self._call_count = 0
        self._history: list[SuluvPrompt] = []

    async def complete(self, prompt: SuluvPrompt) -> LLMResponse:
        self._history.append(prompt)
        self._call_count += 1

        if self._responses:
            resp = self._responses.pop(0)
            if isinstance(resp, LLMResponse):
                return resp
            return LLMResponse(
                content=[ContentBlock.text_block(resp)],
                input_tokens=10,
                output_tokens=len(resp.split()),
                total_tokens=10 + len(resp.split()),
                model="mock",
            )

        return LLMResponse(
            content=[ContentBlock.text_block(self._default)],
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            model="mock",
        )

    async def stream(self, prompt: SuluvPrompt) -> AsyncIterator[str]:
        response = await self.complete(prompt)
        text = response.text
        for word in text.split():
            yield word + " "

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def history(self) -> list[SuluvPrompt]:
        return list(self._history)

    def reset(self) -> None:
        self._call_count = 0
        self._history.clear()
