"""MapNode — parallel for-each over a list."""

from __future__ import annotations

import asyncio
from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class MapNode(GraphNode):
    """Execute a body node for each item in a list, in parallel."""

    def __init__(
        self,
        body: GraphNode,
        max_concurrency: int = 10,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.MAP, name=name, **kwargs
        )
        self._body = body
        self._max_concurrency = max_concurrency

    async def execute(self, input: NodeInput) -> NodeOutput:
        items = input.data
        if not isinstance(items, list):
            return NodeOutput(
                data=None, success=False, error="MapNode input must be a list"
            )

        semaphore = asyncio.Semaphore(self._max_concurrency)
        results: list[Any] = [None] * len(items)
        errors: list[str] = []

        async def process_item(index: int, item: Any) -> None:
            async with semaphore:
                item_input = NodeInput(data=item, context=input.context)
                output = await self._body.execute(item_input)
                results[index] = output.data
                if not output.success:
                    errors.append(f"Item {index}: {output.error}")

        tasks = [process_item(i, item) for i, item in enumerate(items)]
        await asyncio.gather(*tasks)

        return NodeOutput(
            data=results,
            success=len(errors) == 0,
            error="; ".join(errors) if errors else None,
            metadata={"total": len(items), "errors": len(errors)},
        )
