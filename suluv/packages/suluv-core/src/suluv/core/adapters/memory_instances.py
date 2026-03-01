"""InMemoryProcessInstanceStore — dict-backed process instance persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from suluv.core.types import InstanceID, InstanceStatus, Priority
from suluv.core.ports.process_instance_store import ProcessInstanceStore, ProcessInstance


class InMemoryProcessInstanceStore(ProcessInstanceStore):
    """In-memory process instance store."""

    def __init__(self) -> None:
        self._instances: dict[InstanceID, ProcessInstance] = {}

    async def save(self, instance: ProcessInstance) -> None:
        self._instances[instance.instance_id] = instance

    async def load(self, instance_id: InstanceID) -> ProcessInstance | None:
        return self._instances.get(instance_id)

    async def query(
        self,
        process_name: str | None = None,
        status: list[InstanceStatus] | None = None,
        priority: Priority | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProcessInstance]:
        results = list(self._instances.values())

        if process_name:
            results = [i for i in results if i.process_name == process_name]
        if status:
            results = [i for i in results if i.status in status]
        if priority:
            results = [i for i in results if i.priority == priority]
        if created_after:
            results = [i for i in results if i.created_at >= created_after]
        if created_before:
            results = [i for i in results if i.created_at <= created_before]

        results.sort(key=lambda i: i.created_at, reverse=True)
        return results[offset : offset + limit]

    async def delete(self, instance_id: InstanceID) -> None:
        self._instances.pop(instance_id, None)

    async def count(
        self,
        process_name: str | None = None,
        status: list[InstanceStatus] | None = None,
    ) -> int:
        results = list(self._instances.values())
        if process_name:
            results = [i for i in results if i.process_name == process_name]
        if status:
            results = [i for i in results if i.status in status]
        return len(results)

    def clear(self) -> None:
        self._instances.clear()
