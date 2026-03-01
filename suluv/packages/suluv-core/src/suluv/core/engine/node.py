"""GraphNode ABC — the base class all node types implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from suluv.core.types import NodeID, NodeType, new_id


@dataclass
class NodeInput:
    """Input payload for a node execution."""

    data: Any = None
    context: dict[str, Any] = field(default_factory=dict)
    source_node_id: NodeID | None = None


@dataclass
class NodeOutput:
    """Output from a node execution."""

    data: Any = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GraphNode(ABC):
    """Abstract base class for all graph nodes.

    All 16 node types implement this ABC.
    Framework users can create custom node types by subclassing.
    """

    def __init__(
        self,
        node_id: NodeID | None = None,
        node_type: NodeType = NodeType.TOOL,
        name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.node_id: NodeID = node_id or NodeID(new_id())
        self.node_type = node_type
        self.name = name or self.node_id
        self.metadata: dict[str, Any] = metadata or {}

    @abstractmethod
    async def execute(self, input: NodeInput) -> NodeOutput:
        """Execute this node with the given input."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dict."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.node_id!r}, type={self.node_type.value})"
