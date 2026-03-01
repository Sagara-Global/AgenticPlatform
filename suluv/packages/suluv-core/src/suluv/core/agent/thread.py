"""Thread — a persistent conversation between a user and an agent.

A Thread is the unit of conversation continuity.  Every call to
``agent.run(task, thread_id="t1")`` appends to the same thread, and the
full message history is replayed as context to the LLM.

Threads are persisted by a :class:`Checkpointer`.  If no checkpointer is
configured the agent still works — it just can't resume across restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from suluv.core.messages.message import SuluvMessage


@dataclass
class Checkpoint:
    """A snapshot of a thread's state at a specific point.

    Each successful ``agent.run()`` produces one checkpoint.
    """

    checkpoint_id: str = ""
    step: int = 0
    messages: list[SuluvMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class Thread:
    """A stateful conversation thread.

    Attributes:
        thread_id:  unique identifier (user-supplied or auto-generated)
        messages:   the full ordered message history
        metadata:   arbitrary kv (user_id, org_id, agent_name, …)
        created_at: when the thread was first created
        updated_at: when the thread was last modified
        checkpoints: ordered list of checkpoints (one per run)
    """

    thread_id: str = ""
    messages: list[SuluvMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    checkpoints: list[Checkpoint] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)

    @property
    def last_checkpoint(self) -> Checkpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None

    def append_message(self, msg: SuluvMessage) -> None:
        """Append a message and update ``updated_at``."""
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)

    def add_checkpoint(
        self,
        messages: list[SuluvMessage],
        *,
        step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint from the current messages."""
        import uuid

        cp = Checkpoint(
            checkpoint_id=uuid.uuid4().hex[:12],
            step=step,
            messages=list(messages),
            metadata=metadata or {},
        )
        self.checkpoints.append(cp)
        self.updated_at = datetime.now(timezone.utc)
        return cp

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "message_count": self.message_count,
            "checkpoint_count": self.checkpoint_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }
