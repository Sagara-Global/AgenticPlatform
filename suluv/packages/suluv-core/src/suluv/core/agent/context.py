"""AgentContext — identity, session, and scoping for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from suluv.core.types import OrgID, UserID, SessionID, ExecutionID


@dataclass
class AgentContext:
    """Execution context passed to every agent run.

    Carries identity (who is asking), session tracking, thread
    assignment, and arbitrary key-value data the agent or graph
    may need.

    When ``thread_id`` is set the agent will:
    1. Load the existing thread (conversation history) from the
       checkpointer before building the prompt.
    2. Append new messages to the thread.
    3. Save a checkpoint after the run completes.

    This gives every agent automatic short-term memory scoped to
    the thread — no manual ``MemoryManager`` wiring needed.
    """

    org_id: OrgID | None = None
    user_id: UserID | None = None
    session_id: SessionID | None = None
    execution_id: ExecutionID | None = None
    thread_id: str | None = None
    locale: str = "en"
    timezone: str = "UTC"
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "execution_id": self.execution_id,
            "thread_id": self.thread_id,
            "locale": self.locale,
            "timezone": self.timezone,
            "variables": self.variables,
            "metadata": self.metadata,
        }
