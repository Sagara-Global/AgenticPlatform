"""ConsentProvider — data consent checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ConsentResult:
    granted: bool
    purpose: str
    reason: str = ""


class ConsentProvider(ABC):
    """Port for checking user/data consent."""

    @abstractmethod
    async def check(self, context: dict[str, Any], purpose: str) -> ConsentResult:
        """Check if consent is granted for a given purpose."""
        ...
