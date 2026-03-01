"""SuluvPrompt — the full payload sent to an LLM backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from suluv.core.messages.message import SuluvMessage


@dataclass
class ToolSchema:
    """Schema describing a tool the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuluvPrompt:
    """Complete prompt payload for an LLM request.

    Includes messages, available tools, optional structured output schema,
    and generation parameters.
    """

    messages: list[SuluvMessage] = field(default_factory=list)
    tools: list[ToolSchema] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | None = None  # e.g. "json_object"
    stop: list[str] | None = None
