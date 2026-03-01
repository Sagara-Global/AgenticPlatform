"""HumanNode — pauses execution for human input."""

from __future__ import annotations

from typing import Any

from suluv.core.types import NodeType, Priority
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput
from suluv.core.ports.human_task_queue import HumanTaskQueue, HumanTask


class HumanNode(GraphNode):
    """Emits a task to HumanTaskQueue and waits for completion."""

    def __init__(
        self,
        task_queue: HumanTaskQueue,
        title: str = "",
        role: str = "",
        task_type: str = "generic",
        priority: Priority = Priority.MEDIUM,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.HUMAN, name=name, **kwargs
        )
        self._queue = task_queue
        self._title = title
        self._role = role
        self._task_type = task_type
        self._priority = priority

    async def execute(self, input: NodeInput) -> NodeOutput:
        task = HumanTask(
            title=self._title or self.name,
            description=str(input.data) if input.data else "",
            task_type=self._task_type,
            data=input.data if isinstance(input.data, dict) else {"input": input.data},
            role=self._role,
            priority=self._priority,
            execution_id=input.context.get("execution_id"),
            node_id=self.node_id,
        )
        task_id = await self._queue.emit(task)

        # In a real runtime, this would wait/poll. Here we check immediately.
        completed = await self._queue.get(task_id)
        if completed and completed.result is not None:
            return NodeOutput(data=completed.result, success=True)

        # Task is pending — return WAITING status for runtime to handle
        return NodeOutput(
            data={"task_id": task_id, "status": "waiting"},
            success=True,
            metadata={"waiting": True, "task_id": task_id},
        )
