"""SuluvMessage — a single message in a conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from suluv.core.messages.content import ContentBlock, ContentType


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class SuluvMessage:
    """A single message with multimodal content blocks.

    Example::

        SuluvMessage(role=MessageRole.USER, content=[
            ContentBlock.text_block("What is in this image?"),
            ContentBlock.image_url_block("https://example.com/img.png"),
        ])
    """

    role: MessageRole
    content: list[ContentBlock] = field(default_factory=list)

    @staticmethod
    def system(text: str) -> SuluvMessage:
        return SuluvMessage(
            role=MessageRole.SYSTEM,
            content=[ContentBlock.text_block(text)],
        )

    @staticmethod
    def user(text: str) -> SuluvMessage:
        return SuluvMessage(
            role=MessageRole.USER,
            content=[ContentBlock.text_block(text)],
        )

    @staticmethod
    def assistant(text: str) -> SuluvMessage:
        return SuluvMessage(
            role=MessageRole.ASSISTANT,
            content=[ContentBlock.text_block(text)],
        )

    @property
    def text(self) -> str:
        """Convenience: concatenate all TEXT blocks."""
        return "".join(
            b.text for b in self.content if b.type == ContentType.TEXT and b.text
        )
