"""Core types — IDs, enums, and result dataclasses used throughout Suluv."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, NewType

# ── ID types ──────────────────────────────────────────────────────────────────

SuluvID = NewType("SuluvID", str)
NodeID = NewType("NodeID", str)
EdgeID = NewType("EdgeID", str)
ExecutionID = NewType("ExecutionID", str)
SessionID = NewType("SessionID", str)
OrgID = NewType("OrgID", str)
UserID = NewType("UserID", str)
ProcessID = NewType("ProcessID", str)
InstanceID = NewType("InstanceID", str)


def new_id() -> SuluvID:
    """Generate a new unique Suluv ID."""
    return SuluvID(uuid.uuid4().hex[:16])


def new_execution_id() -> ExecutionID:
    return ExecutionID(f"exec-{uuid.uuid4().hex[:12]}")


def new_instance_id() -> InstanceID:
    return InstanceID(f"inst-{uuid.uuid4().hex[:12]}")


# ── Enums ─────────────────────────────────────────────────────────────────────


class NodeState(str, Enum):
    """State of a node within an execution."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    WAITING = "waiting"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ErrorPolicy(str, Enum):
    """What to do when a node fails."""
    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"


class NodeType(str, Enum):
    """All supported node types in the graph engine."""
    AGENT = "agent"
    TOOL = "tool"
    HUMAN = "human"
    ROUTER = "router"
    LOOP = "loop"
    MAP = "map"
    GATEWAY = "gateway"
    DELAY = "delay"
    SUBGRAPH = "subgraph"
    PROCESS = "process"
    TRIGGER = "trigger"
    # Process engine nodes
    DECISION = "decision"
    FORM = "form"
    SIGNAL = "signal"
    COMPENSATION = "compensation"
    TIMER = "timer"


class Priority(str, Enum):
    """Priority levels for process instances and tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InstanceStatus(str, Enum):
    """Lifecycle state of a process instance."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    WAITING = "waiting"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class NodeExecution:
    """Record of a single node's execution within a graph run."""
    node_id: NodeID
    node_type: NodeType
    state: NodeState
    input: Any = None
    output: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cost_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0


@dataclass
class ExecutionResult:
    """Final result of a graph execution."""
    execution_id: ExecutionID
    output: Any = None
    success: bool = True
    error: str | None = None
    cost_tokens: int = 0
    cost_usd: float = 0.0
    trace: list[NodeExecution] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


@dataclass
class CostRecord:
    """Token and cost tracking for a single operation."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    thread_id: str | None = None


@dataclass
class AuditEvent:
    """A single auditable event in the system."""
    event_id: SuluvID = field(default_factory=new_id)
    event_type: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    org_id: OrgID | None = None
    user_id: UserID | None = None
    session_id: SessionID | None = None
    thread_id: str | None = None
    execution_id: ExecutionID | None = None
    node_id: NodeID | None = None
    data: dict[str, Any] = field(default_factory=dict)
