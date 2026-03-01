"""ToolNode — runs a function directly, no LLM."""

from __future__ import annotations

from typing import Any, Callable, Awaitable

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class ToolNode(GraphNode):
    """Executes a tool/function directly without an LLM.

    Like an n8n-style function node.
    """

    def __init__(
        self,
        func: Callable[..., Any | Awaitable[Any]],
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.TOOL, name=name, **kwargs
        )
        self._func = func

    async def execute(self, input: NodeInput) -> NodeOutput:
        try:
            import asyncio

            result = self._func(input.data, input.context)
            if asyncio.iscoroutine(result):
                result = await result
            return NodeOutput(data=result, success=True)
        except Exception as e:
            return NodeOutput(data=None, success=False, error=str(e))
