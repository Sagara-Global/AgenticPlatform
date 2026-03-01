"""GraphEdge — connections between nodes with conditions and error policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from suluv.core.types import EdgeID, NodeID, ErrorPolicy, new_id
from suluv.core.engine.node import NodeOutput


@dataclass
class GraphEdge:
    """An edge connecting two nodes in a graph.

    Edges carry:
    - condition: a function that decides if this edge should fire
    - task_transform: optional transform of the output before passing to target
    - error_policy: what to do if the target node fails
    """

    source_id: NodeID
    target_id: NodeID
    edge_id: EdgeID = field(default_factory=lambda: EdgeID(new_id()))
    condition: Callable[[NodeOutput], bool] | None = None
    task_transform: Callable[[NodeOutput], Any] | None = None
    error_policy: ErrorPolicy = ErrorPolicy.FAIL_FAST
    fallback_node_id: NodeID | None = None
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def should_fire(self, output: NodeOutput) -> bool:
        """Check if this edge should activate given the source output."""
        if self.condition is None:
            return True
        return self.condition(output)

    def transform(self, output: NodeOutput) -> Any:
        """Transform the source output into input for the target."""
        if self.task_transform is None:
            return output.data
        return self.task_transform(output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "error_policy": self.error_policy.value,
            "max_retries": self.max_retries,
        }
