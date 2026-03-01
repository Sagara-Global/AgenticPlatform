"""ConnectorPort — external API gateway."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConnectorRequest:
    method: str = "GET"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    params: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass
class ConnectorResponse:
    status_code: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)


class ConnectorPort(ABC):
    """Port for external API calls."""

    @abstractmethod
    async def send(self, request: ConnectorRequest) -> ConnectorResponse:
        """Send an external API request."""
        ...
