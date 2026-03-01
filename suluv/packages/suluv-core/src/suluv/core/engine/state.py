"""ExecutionState — tracks per-node state during a graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from suluv.core.types import (
    ExecutionID,
    NodeID,
    NodeState,
    new_execution_id,
    NodeExecution,
)


@dataclass
class ExecutionState:
    """Tracks the state of every node in a graph execution.

    Persisted via StateStore after every node completes.
    """

    execution_id: ExecutionID = field(default_factory=new_execution_id)
    node_states: dict[NodeID, NodeState] = field(default_factory=dict)
    node_outputs: dict[NodeID, Any] = field(default_factory=dict)
    node_errors: dict[NodeID, str] = field(default_factory=dict)
    node_retries: dict[NodeID, int] = field(default_factory=dict)
    trace: list[NodeExecution] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def get_state(self, node_id: NodeID) -> NodeState:
        return self.node_states.get(node_id, NodeState.PENDING)

    def set_state(self, node_id: NodeID, state: NodeState) -> None:
        self.node_states[node_id] = state

    def set_output(self, node_id: NodeID, output: Any) -> None:
        self.node_outputs[node_id] = output

    def set_error(self, node_id: NodeID, error: str) -> None:
        self.node_errors[node_id] = error

    def increment_retries(self, node_id: NodeID) -> int:
        count = self.node_retries.get(node_id, 0) + 1
        self.node_retries[node_id] = count
        return count

    def is_terminal(self) -> bool:
        """Check if execution has reached a terminal state."""
        states = set(self.node_states.values())
        # Terminal if no PENDING or RUNNING nodes
        active = {NodeState.PENDING, NodeState.RUNNING, NodeState.RETRYING, NodeState.WAITING}
        return not states.intersection(active) if self.node_states else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "node_states": {k: v.value for k, v in self.node_states.items()},
            "node_outputs": self.node_outputs,
            "node_errors": self.node_errors,
            "node_retries": self.node_retries,
            "variables": self.variables,
            "started_at": self.started_at.isoformat(),
        }
