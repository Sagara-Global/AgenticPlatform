"""AnthropicBackend — adapter for Anthropic Messages API."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from suluv.core.messages.content import ContentBlock, ContentType
from suluv.core.messages.message import MessageRole, SuluvMessage
from suluv.core.messages.prompt import SuluvPrompt
from suluv.core.ports.llm_backend import LLMBackend, LLMResponse

logger = logging.getLogger("suluv.adapters.anthropic")


def _message_to_anthropic(msg: SuluvMessage) -> dict[str, Any]:
    """Convert a SuluvMessage to Anthropic message format."""
    role_map = {
        MessageRole.USER: "user",
        MessageRole.ASSISTANT: "assistant",
        MessageRole.TOOL: "user",  # Anthropic uses user role for tool results
    }
    role = role_map.get(msg.role, "user")

    # Tool result messages → tool_result content block
    tool_results = [
        b for b in msg.content if b.type == ContentType.TOOL_RESULT
    ]
    if tool_results:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.text or "",
                }
                for tr in tool_results
            ],
        }

    # Tool call messages (assistant) → tool_use content block
    tool_calls = [
        b for b in msg.content if b.type == ContentType.TOOL_CALL
    ]
    if tool_calls and msg.role == MessageRole.ASSISTANT:
        blocks: list[dict[str, Any]] = []
        if msg.text:
            blocks.append({"type": "text", "text": msg.text})
        for tc in tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc.tool_call_id,
                "name": tc.name,
                "input": tc.arguments or {},
            })
        return {"role": "assistant", "content": blocks}

    # Build content blocks
    parts: list[dict[str, Any]] = []
    for b in msg.content:
        if b.type == ContentType.TEXT and b.text:
            parts.append({"type": "text", "text": b.text})
        elif b.type == ContentType.IMAGE_URL and b.url:
            # Anthropic requires base64 for images; URL must be fetched externally
            parts.append({"type": "text", "text": f"[Image: {b.url}]"})
        elif b.type == ContentType.IMAGE_BASE64 and b.base64_data:
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": b.media_type or "image/png",
                    "data": b.base64_data,
                },
            })

    if not parts:
        parts = [{"type": "text", "text": ""}]

    return {"role": role, "content": parts}


def _tools_to_anthropic(prompt: SuluvPrompt) -> list[dict[str, Any]] | None:
    """Convert SuluvPrompt tools to Anthropic tool format."""
    if not prompt.tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in prompt.tools
    ]


class AnthropicBackend(LLMBackend):
    """Anthropic Claude adapter.

    Usage::

        llm = AnthropicBackend(model="claude-sonnet-4-20250514")
        llm = AnthropicBackend(model="claude-sonnet-4-20250514", api_key="sk-ant-...")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        default_temperature: float = 0.1,
        default_max_tokens: int = 4096,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = AsyncAnthropic(**kwargs)

    async def complete(self, prompt: SuluvPrompt) -> LLMResponse:
        """Send prompt to Anthropic and return structured response."""
        # Extract system message (Anthropic handles it separately)
        system_text = ""
        messages: list[dict[str, Any]] = []

        for msg in prompt.messages:
            if msg.role == MessageRole.SYSTEM:
                system_text = msg.text
            else:
                messages.append(_message_to_anthropic(msg))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": prompt.max_tokens or self._default_max_tokens,
            "temperature": prompt.temperature or self._default_temperature,
        }

        if system_text:
            kwargs["system"] = system_text

        tools = _tools_to_anthropic(prompt)
        if tools:
            kwargs["tools"] = tools

        if prompt.stop:
            kwargs["stop_sequences"] = prompt.stop

        response = await self._client.messages.create(**kwargs)

        # Build content blocks from response
        content_blocks: list[ContentBlock] = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(ContentBlock.text_block(block.text))
            elif block.type == "tool_use":
                content_blocks.append(ContentBlock.tool_call_block(
                    tool_call_id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(
            content=content_blocks,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            model=response.model,
            finish_reason=response.stop_reason or "",
            raw=response.model_dump(),
        )

    async def stream(self, prompt: SuluvPrompt) -> AsyncIterator[str]:
        """Stream text chunks from Anthropic."""
        system_text = ""
        messages: list[dict[str, Any]] = []

        for msg in prompt.messages:
            if msg.role == MessageRole.SYSTEM:
                system_text = msg.text
            else:
                messages.append(_message_to_anthropic(msg))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": prompt.max_tokens or self._default_max_tokens,
            "temperature": prompt.temperature or self._default_temperature,
        }

        if system_text:
            kwargs["system"] = system_text

        if prompt.stop:
            kwargs["stop_sequences"] = prompt.stop

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
