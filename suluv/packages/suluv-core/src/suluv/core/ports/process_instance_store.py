"""ProcessInstanceStore — persistence for process instances."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from suluv.core.types import InstanceID, InstanceStatus, Priority, new_instance_id


@dataclass
class ProcessInstance:
    """A single running instance of a process definition."""

    instance_id: InstanceID = field(default_factory=new_instance_id)
    process_name: str = ""
    version: str = ""
    status: InstanceStatus = InstanceStatus.CREATED
    priority: Priority = Priority.MEDIUM
    variables: dict[str, Any] = field(default_factory=dict)
    current_stage: str | None = None
    current_step: str | None = None
    execution_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None


class ProcessInstanceStore(ABC):
    """Port for persisting and querying process instances."""

    @abstractmethod
    async def save(self, instance: ProcessInstance) -> None:
        """Save or update a process instance."""
        ...

    @abstractmethod
    async def load(self, instance_id: InstanceID) -> ProcessInstance | None:
        """Load a process instance by ID."""
        ...

    @abstractmethod
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
        """Query instances with filters."""
        ...

    @abstractmethod
    async def delete(self, instance_id: InstanceID) -> None:
        """Delete a process instance."""
        ...

    @abstractmethod
    async def count(
        self,
        process_name: str | None = None,
        status: list[InstanceStatus] | None = None,
    ) -> int:
        """Count instances matching filters."""
        ...
