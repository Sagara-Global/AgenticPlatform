"""ContentBlock — the atomic unit of multimodal content."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"
    AUDIO_URL = "audio_url"
    AUDIO_BASE64 = "audio_base64"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"


@dataclass
class ContentBlock:
    """A single block of content within a message.

    Examples::

        ContentBlock(type=ContentType.TEXT, text="Hello")
        ContentBlock(type=ContentType.IMAGE_URL, url="https://...")
        ContentBlock(type=ContentType.TOOL_CALL, tool_call_id="tc1",
                     name="search", arguments={"q": "test"})
        ContentBlock(type=ContentType.TOOL_RESULT, tool_call_id="tc1",
                     text='{"results": [...]}')
    """

    type: ContentType
    text: str | None = None
    url: str | None = None
    base64_data: str | None = None
    media_type: str | None = None  # e.g. "image/png", "audio/mp3"

    # Tool-specific
    tool_call_id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None

    # Document
    document_name: str | None = None
    document_bytes: bytes | None = None

    @staticmethod
    def text_block(text: str) -> ContentBlock:
        return ContentBlock(type=ContentType.TEXT, text=text)

    @staticmethod
    def image_url_block(url: str) -> ContentBlock:
        return ContentBlock(type=ContentType.IMAGE_URL, url=url)

    @staticmethod
    def tool_call_block(
        tool_call_id: str, name: str, arguments: dict[str, Any]
    ) -> ContentBlock:
        return ContentBlock(
            type=ContentType.TOOL_CALL,
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )

    @staticmethod
    def tool_result_block(tool_call_id: str, text: str) -> ContentBlock:
        return ContentBlock(
            type=ContentType.TOOL_RESULT,
            tool_call_id=tool_call_id,
            text=text,
        )
