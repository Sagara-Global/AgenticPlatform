"""AuditBackend — write and query audit events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from suluv.core.types import AuditEvent


class AuditBackend(ABC):
    """Port for audit logging."""

    @abstractmethod
    async def write(self, event: AuditEvent) -> None:
        """Write an audit event."""
        ...

    @abstractmethod
    async def query(self, filters: dict[str, Any]) -> list[AuditEvent]:
        """Query audit events by filters."""
        ...
