"""HumanTaskQueue — human-in-the-loop task management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from suluv.core.types import SuluvID, UserID, Priority, new_id


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class HumanTask:
    """A task for a human to complete."""

    task_id: SuluvID = field(default_factory=new_id)
    title: str = ""
    description: str = ""
    task_type: str = "generic"  # "form", "approval", "review", etc.
    data: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    role: str = ""  # required role
    assigned_to: UserID | None = None
    claimed_by: UserID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    claimed_at: datetime | None = None
    completed_at: datetime | None = None
    execution_id: str | None = None
    node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HumanTaskQueue(ABC):
    """Port for human task management with claim/release semantics."""

    @abstractmethod
    async def emit(self, task: HumanTask) -> SuluvID:
        """Add a task to the queue. Returns the task ID."""
        ...

    @abstractmethod
    async def poll(
        self,
        role: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 10,
    ) -> list[HumanTask]:
        """List tasks matching filters."""
        ...

    @abstractmethod
    async def claim(self, task_id: SuluvID, user_id: UserID) -> HumanTask:
        """Claim a task for a user (locks it)."""
        ...

    @abstractmethod
    async def release(self, task_id: SuluvID) -> HumanTask:
        """Release a claimed task back to the pool."""
        ...

    @abstractmethod
    async def delegate(self, task_id: SuluvID, to_user: UserID) -> HumanTask:
        """Delegate a task to another user."""
        ...

    @abstractmethod
    async def complete(self, task_id: SuluvID, result: dict[str, Any]) -> HumanTask:
        """Complete a task with a result."""
        ...

    async def get(self, task_id: SuluvID) -> HumanTask | None:
        """Get a task by ID. Optional."""
        raise NotImplementedError
