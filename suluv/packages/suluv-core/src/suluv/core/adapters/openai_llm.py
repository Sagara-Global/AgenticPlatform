"""OpenAIBackend — adapter for OpenAI chat completions API."""

from __future__ import annotations

import os
import logging
from typing import Any, AsyncIterator

from suluv.core.messages.content import ContentBlock, ContentType
from suluv.core.messages.message import MessageRole, SuluvMessage
from suluv.core.messages.prompt import SuluvPrompt
from suluv.core.ports.llm_backend import LLMBackend, LLMResponse

logger = logging.getLogger("suluv.adapters.openai")


def _message_to_openai(msg: SuluvMessage) -> dict[str, Any]:
    """Convert a SuluvMessage to OpenAI chat format."""
    role_map = {
        MessageRole.SYSTEM: "system",
        MessageRole.USER: "user",
        MessageRole.ASSISTANT: "assistant",
        MessageRole.TOOL: "tool",
    }
    role = role_map.get(msg.role, "user")

    # Check for tool calls in assistant messages
    tool_calls = [
        b for b in msg.content if b.type == ContentType.TOOL_CALL
    ]
    if tool_calls and role == "assistant":
        result: dict[str, Any] = {
            "role": "assistant",
            "content": msg.text or None,
            "tool_calls": [
                {
                    "id": tc.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": __import__("json").dumps(tc.arguments or {}),
                    },
                }
                for tc in tool_calls
            ],
        }
        return result

    # Tool result messages
    tool_results = [
        b for b in msg.content if b.type == ContentType.TOOL_RESULT
    ]
    if tool_results and role == "tool":
        return {
            "role": "tool",
            "tool_call_id": tool_results[0].tool_call_id,
            "content": tool_results[0].text or "",
        }

    # Build content — simple string or multimodal array
    has_images = any(
        b.type in (ContentType.IMAGE_URL, ContentType.IMAGE_BASE64)
        for b in msg.content
    )

    if has_images:
        parts: list[dict[str, Any]] = []
        for b in msg.content:
            if b.type == ContentType.TEXT and b.text:
                parts.append({"type": "text", "text": b.text})
            elif b.type == ContentType.IMAGE_URL and b.url:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": b.url},
                })
            elif b.type == ContentType.IMAGE_BASE64 and b.base64_data:
                mt = b.media_type or "image/png"
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mt};base64,{b.base64_data}",
                    },
                })
        return {"role": role, "content": parts}

    # Plain text
    return {"role": role, "content": msg.text or ""}


def _tools_to_openai(prompt: SuluvPrompt) -> list[dict[str, Any]] | None:
    """Convert SuluvPrompt tools to OpenAI function-calling format."""
    if not prompt.tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in prompt.tools
    ]


class OpenAIBackend(LLMBackend):
    """OpenAI chat completions adapter.

    Usage::

        llm = OpenAIBackend(model="gpt-4o")
        llm = OpenAIBackend(model="gpt-4o-mini", api_key="sk-...")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        default_temperature: float = 0.1,
        default_max_tokens: int = 4096,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self._model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization

        self._client = AsyncOpenAI(**kwargs)

    async def complete(self, prompt: SuluvPrompt) -> LLMResponse:
        """Send prompt to OpenAI and return structured response."""
        messages = [_message_to_openai(m) for m in prompt.messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": prompt.temperature or self._default_temperature,
            "max_tokens": prompt.max_tokens or self._default_max_tokens,
        }

        tools = _tools_to_openai(prompt)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if prompt.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        if prompt.stop:
            kwargs["stop"] = prompt.stop

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        # Build content blocks
        content_blocks: list[ContentBlock] = []

        if message.content:
            content_blocks.append(ContentBlock.text_block(message.content))

        if message.tool_calls:
            import json
            for tc in message.tool_calls:
                content_blocks.append(ContentBlock.tool_call_block(
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        usage = response.usage

        return LLMResponse(
            content=content_blocks,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "",
            raw=response.model_dump(),
        )

    async def stream(self, prompt: SuluvPrompt) -> AsyncIterator[str]:
        """Stream text chunks from OpenAI."""
        messages = [_message_to_openai(m) for m in prompt.messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": prompt.temperature or self._default_temperature,
            "max_tokens": prompt.max_tokens or self._default_max_tokens,
            "stream": True,
        }

        if prompt.stop:
            kwargs["stop"] = prompt.stop

        response = await self._client.chat.completions.create(**kwargs)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using OpenAI embeddings API."""
        response = await self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
