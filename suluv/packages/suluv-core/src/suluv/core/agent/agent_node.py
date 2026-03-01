"""AgentNode — wraps a SuluvAgent as a GraphNode for use in graphs."""

from __future__ import annotations

from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput
from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.context import AgentContext


class AgentNode(GraphNode):
    """Thin wrapper that executes a SuluvAgent within a graph.

    The input data is passed as the task string (or extracted from
    a dict via ``task_key``).
    """

    def __init__(
        self,
        agent: SuluvAgent,
        task_key: str = "task",
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.AGENT,
            name=name or agent.role.name,
            **kwargs,
        )
        self._agent = agent
        self._task_key = task_key

    @property
    def agent(self) -> SuluvAgent:
        return self._agent

    async def execute(self, input: NodeInput) -> NodeOutput:
        # Extract task string from input
        if isinstance(input.data, str):
            task = input.data
        elif isinstance(input.data, dict):
            task = input.data.get(self._task_key, str(input.data))
        else:
            task = str(input.data) if input.data is not None else ""

        # Build context from graph context
        ctx = AgentContext(
            org_id=input.context.get("org_id"),
            user_id=input.context.get("user_id"),
            session_id=input.context.get("session_id"),
            thread_id=input.context.get("thread_id"),
            execution_id=input.context.get("execution_id"),
        )

        result = await self._agent.run(task, context=ctx)

        return NodeOutput(
            data={
                "answer": result.answer,
                "structured": result.structured,
                "steps": result.step_count,
            },
            success=result.success,
            error=result.error,
            metadata={
                "tokens": result.total_tokens,
                "cost_usd": result.cost_usd,
                "step_count": result.step_count,
            },
        )
