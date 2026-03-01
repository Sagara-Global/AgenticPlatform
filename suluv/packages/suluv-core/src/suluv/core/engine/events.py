"""GraphEvent types for streaming execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from suluv.core.types import NodeID, ExecutionID, ExecutionResult


@dataclass
class GraphEvent:
    """Base class for all graph execution events."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NodeStarted(GraphEvent):
    node_id: NodeID = NodeID("")
    node_type: str = ""


@dataclass
class NodeOutput(GraphEvent):
    """Partial output chunk from a node (e.g., streaming agent)."""

    node_id: NodeID = NodeID("")
    chunk: Any = None


@dataclass
class NodeCompleted(GraphEvent):
    node_id: NodeID = NodeID("")
    result: Any = None


@dataclass
class NodeFailed(GraphEvent):
    node_id: NodeID = NodeID("")
    error: str = ""


@dataclass
class NodeRetrying(GraphEvent):
    node_id: NodeID = NodeID("")
    attempt: int = 0
    max_retries: int = 0
    error: str = ""


@dataclass
class GraphCompleted(GraphEvent):
    execution_id: ExecutionID = ExecutionID("")
    result: ExecutionResult | None = None
