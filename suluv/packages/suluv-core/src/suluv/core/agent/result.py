"""AgentResult — output of an agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StepRecord:
    """One step in the agent's ReAct loop."""

    step: int
    thought: str = ""
    action: str = ""
    action_input: Any = None
    observation: Any = None
    error: str | None = None
    tokens_used: int = 0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class AgentResult:
    """Final result of a SuluvAgent run.

    Includes the answer, structured data (if requested), full step
    trace, and cost information.
    """

    answer: str = ""
    structured: Any = None  # populated if output_schema was provided
    success: bool = True
    error: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    total_tokens: int = 0
    cost_usd: float = 0.0
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "structured": self.structured,
            "success": self.success,
            "error": self.error,
            "step_count": self.step_count,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }
