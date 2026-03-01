"""InMemoryHumanTaskQueue — dict-backed task queue with claim/release."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from suluv.core.types import SuluvID, UserID
from suluv.core.ports.human_task_queue import HumanTaskQueue, HumanTask, TaskStatus


class InMemoryHumanTaskQueue(HumanTaskQueue):
    """In-memory task queue with claim/release semantics."""

    def __init__(self) -> None:
        self._tasks: dict[SuluvID, HumanTask] = {}

    async def emit(self, task: HumanTask) -> SuluvID:
        self._tasks[task.task_id] = task
        return task.task_id

    async def poll(
        self,
        role: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 10,
    ) -> list[HumanTask]:
        results = list(self._tasks.values())
        if role:
            results = [t for t in results if t.role == role]
        if status:
            results = [t for t in results if t.status == status]
        # Sort by priority (critical first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        results.sort(key=lambda t: priority_order.get(t.priority.value, 2))
        return results[:limit]

    async def claim(self, task_id: SuluvID, user_id: UserID) -> HumanTask:
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        if task.status == TaskStatus.CLAIMED:
            raise ValueError(f"Task {task_id} already claimed by {task.claimed_by}")
        task.status = TaskStatus.CLAIMED
        task.claimed_by = user_id
        task.claimed_at = datetime.now(timezone.utc)
        return task

    async def release(self, task_id: SuluvID) -> HumanTask:
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        task.status = TaskStatus.PENDING
        task.claimed_by = None
        task.claimed_at = None
        return task

    async def delegate(self, task_id: SuluvID, to_user: UserID) -> HumanTask:
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        task.assigned_to = to_user
        task.claimed_by = to_user
        task.status = TaskStatus.CLAIMED
        task.claimed_at = datetime.now(timezone.utc)
        return task

    async def complete(self, task_id: SuluvID, result: dict[str, Any]) -> HumanTask:
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = datetime.now(timezone.utc)
        return task

    async def get(self, task_id: SuluvID) -> HumanTask | None:
        return self._tasks.get(task_id)

    def clear(self) -> None:
        self._tasks.clear()
