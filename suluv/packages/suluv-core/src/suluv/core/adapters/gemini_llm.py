"""GeminiBackend — adapter for Google Gemini via ``google-genai`` SDK.

Uses the new ``google.genai`` client with ``types.Content`` / ``types.Part``
objects — same pattern as LangChain's ``ChatGoogleGenerativeAI``.

Install::

    pip install google-genai
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from suluv.core.messages.content import ContentBlock, ContentType
from suluv.core.messages.message import MessageRole, SuluvMessage
from suluv.core.messages.prompt import SuluvPrompt
from suluv.core.ports.llm_backend import LLMBackend, LLMResponse

logger = logging.getLogger("suluv.adapters.gemini")


# ---------------------------------------------------------------------------
# Message conversion — SuluvMessage ➜ google.genai.types.Content
# ---------------------------------------------------------------------------

def _to_gemini_contents(
    messages: list[SuluvMessage],
) -> tuple[str | None, list[Any]]:
    """Convert ``SuluvMessage`` list into Gemini format.

    Returns ``(system_instruction_text, list[types.Content])``.

    * ``SYSTEM`` messages are extracted and concatenated into a single
      ``system_instruction`` string (Gemini's approach).
    * Everything else becomes ``types.Content(role=..., parts=[...])``.
    * Consecutive messages with the same role are merged (Gemini
      requires strict user/model alternation).
    """
    from google.genai import types

    system_parts: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        # ── system ──────────────────────────────────────────
        if msg.role == MessageRole.SYSTEM:
            system_parts.append(msg.text or "")
            continue

        # ── role mapping ────────────────────────────────────
        role = "model" if msg.role == MessageRole.ASSISTANT else "user"

        # ── parts ───────────────────────────────────────────
        parts: list[types.Part] = []
        for block in msg.content:
            if block.type == ContentType.TEXT and block.text:
                parts.append(types.Part(text=block.text))
            elif block.type == ContentType.IMAGE_URL and block.url:
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type=block.media_type or "image/png",
                        data=block.url.encode(),
                    )
                ))
            elif block.type == ContentType.IMAGE_BASE64 and block.base64_data:
                import base64
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type=block.media_type or "image/png",
                        data=base64.b64decode(block.base64_data),
                    )
                ))

        if not parts:
            parts.append(types.Part(text=msg.text or ""))

        contents.append(types.Content(role=role, parts=parts))

    # ── merge consecutive same-role (Gemini requirement) ───
    merged = _merge_consecutive(contents)

    system_text = "\n".join(system_parts) if system_parts else None
    return system_text, merged


def _merge_consecutive(contents: list[Any]) -> list[Any]:
    """Merge consecutive ``types.Content`` with the same role."""
    if not contents:
        return []
    merged = [contents[0]]
    for item in contents[1:]:
        if item.role == merged[-1].role:
            merged[-1].parts.extend(item.parts)
        else:
            merged.append(item)
    return merged


# ---------------------------------------------------------------------------
# GeminiBackend
# ---------------------------------------------------------------------------

class GeminiBackend(LLMBackend):
    """Google Gemini adapter using ``google.genai`` client.

    Models messages the same way LangChain does — ``types.Content`` with
    ``types.Part`` objects, system instruction via config.

    Usage::

        llm = GeminiBackend(model="gemini-2.0-flash", api_key="...")
        llm = GeminiBackend(model="gemini-2.5-flash-preview-05-20")
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        default_temperature: float = 0.1,
        default_max_tokens: int = 4096,
    ) -> None:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package required. "
                "Install with: pip install google-genai"
            )

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set "
                "GEMINI_API_KEY / GOOGLE_API_KEY env var."
            )

        self._client = genai.Client(api_key=key)
        self._model_name = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    # ────────────────────────────────────────────────────────
    # complete
    # ────────────────────────────────────────────────────────
    async def complete(self, prompt: SuluvPrompt) -> LLMResponse:
        """Send prompt to Gemini and return structured response."""
        from google.genai import types

        system_text, contents = _to_gemini_contents(prompt.messages)

        config_kwargs: dict[str, Any] = {
            "temperature": prompt.temperature or self._default_temperature,
            "max_output_tokens": prompt.max_tokens or self._default_max_tokens,
        }
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if prompt.response_format == "json_object":
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )

        # Extract text
        text = response.text or ""

        # Token usage
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        content_blocks = [ContentBlock.text_block(text)] if text else []

        return LLMResponse(
            content=content_blocks,
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=(input_tokens or 0) + (output_tokens or 0),
            model=self._model_name,
            finish_reason=_get_finish_reason(response),
            raw={},
        )

    # ────────────────────────────────────────────────────────
    # stream
    # ────────────────────────────────────────────────────────
    async def stream(self, prompt: SuluvPrompt) -> AsyncIterator[str]:
        """Stream text chunks from Gemini."""
        from google.genai import types

        system_text, contents = _to_gemini_contents(prompt.messages)

        config_kwargs: dict[str, Any] = {
            "temperature": prompt.temperature or self._default_temperature,
            "max_output_tokens": prompt.max_tokens or self._default_max_tokens,
        }
        if system_text:
            config_kwargs["system_instruction"] = system_text

        config = types.GenerateContentConfig(**config_kwargs)

        async for chunk in self._client.aio.models.generate_content_stream(
            model=self._model_name,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    # ────────────────────────────────────────────────────────
    # embed
    # ────────────────────────────────────────────────────────
    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using Gemini embedding model."""
        result = await self._client.aio.models.embed_content(
            model="text-embedding-004",
            contents=text,
        )
        return list(result.embeddings[0].values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_finish_reason(response: Any) -> str:
    """Extract finish reason string from Gemini response."""
    try:
        if response.candidates:
            reason = response.candidates[0].finish_reason
            return str(reason.name) if reason else ""
    except Exception:
        pass
    return ""
