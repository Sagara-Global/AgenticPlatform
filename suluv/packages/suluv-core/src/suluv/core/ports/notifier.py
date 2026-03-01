"""NotifierPort — send notifications to channels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Notification:
    channel: str  # e.g. "email", "slack", "sms"
    recipient: str
    subject: str = ""
    body: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class NotifierPort(ABC):
    """Port for sending notifications."""

    @abstractmethod
    async def notify(self, notification: Notification) -> None:
        """Send a notification."""
        ...
