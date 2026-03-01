"""DelayNode — pauses execution for a specified duration."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class DelayNode(GraphNode):
    """Wait for a fixed or dynamic duration before continuing.

    Supports both static timedelta and dynamic delays derived from input.
    """

    def __init__(
        self,
        delay: timedelta | None = None,
        delay_key: str | None = None,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.DELAY, name=name, **kwargs
        )
        self._delay = delay or timedelta(seconds=0)
        self._delay_key = delay_key  # read seconds from input[key]

    async def execute(self, input: NodeInput) -> NodeOutput:
        seconds = self._delay.total_seconds()
        if self._delay_key and isinstance(input.data, dict):
            seconds = float(input.data.get(self._delay_key, seconds))

        if seconds > 0:
            await asyncio.sleep(seconds)

        return NodeOutput(
            data=input.data,
            success=True,
            metadata={"delayed_seconds": seconds},
        )
