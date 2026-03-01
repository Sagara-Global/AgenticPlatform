"""GatewayNode — N-of-M join for parallel approval/convergence."""

from __future__ import annotations

from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class GatewayNode(GraphNode):
    """Wait for N-of-M incoming nodes to complete before proceeding.

    Used for parallel approval (e.g., 2-of-3 approvers must agree).
    The runtime collects outputs from source nodes and passes them here.
    """

    def __init__(
        self,
        required: int = 1,
        total: int = 1,
        merge_strategy: str = "all",  # "all" | "first" | "majority"
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.GATEWAY, name=name, **kwargs
        )
        self._required = required
        self._total = total
        self._merge_strategy = merge_strategy

    async def execute(self, input: NodeInput) -> NodeOutput:
        # Input data should be a dict of {source_node_id: output}
        incoming = input.data if isinstance(input.data, dict) else {}
        completed = {k: v for k, v in incoming.items() if v is not None}

        if len(completed) >= self._required:
            if self._merge_strategy == "first":
                first = next(iter(completed.values()))
                return NodeOutput(data=first, success=True)
            else:
                return NodeOutput(
                    data=completed,
                    success=True,
                    metadata={
                        "completed": len(completed),
                        "required": self._required,
                    },
                )
        else:
            return NodeOutput(
                data=None,
                success=True,
                metadata={
                    "waiting": True,
                    "completed": len(completed),
                    "required": self._required,
                },
            )
