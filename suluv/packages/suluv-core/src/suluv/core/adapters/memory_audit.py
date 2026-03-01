"""InMemoryAuditBackend — list-backed audit logging."""

from __future__ import annotations

from typing import Any

from suluv.core.types import AuditEvent
from suluv.core.ports.audit_backend import AuditBackend


class InMemoryAuditBackend(AuditBackend):
    """In-memory audit backend backed by a list."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    async def write(self, event: AuditEvent) -> None:
        self._events.append(event)

    async def query(self, filters: dict[str, Any] | None = None, **kwargs: Any) -> list[AuditEvent]:
        all_filters = dict(filters or {})
        all_filters.update(kwargs)
        results = list(self._events)
        for key, value in all_filters.items():
            results = [
                e for e in results if getattr(e, key, None) == value
            ]
        return results

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
