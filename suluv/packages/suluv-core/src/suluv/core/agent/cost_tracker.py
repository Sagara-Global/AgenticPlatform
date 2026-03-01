"""CostTracker — tracks token usage and cost across agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from suluv.core.types import CostRecord


@dataclass
class CostBudget:
    """Budget limits for cost tracking."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_steps: int | None = None


class CostTracker:
    """Accumulates token costs across the lifetime of an agent.

    Can enforce budgets — raising BudgetExceeded if limits are hit.
    Tracks both *global* totals and *per-thread* breakdowns so that
    callers can inspect cost for a specific conversation thread.
    """

    def __init__(self, budget: CostBudget | None = None) -> None:
        self._budget = budget
        self._records: list[CostRecord] = []
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._step_count: int = 0
        # Per-thread breakdown: thread_id -> {tokens, cost, steps}
        self._thread_totals: dict[str, dict[str, Any]] = {}

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def step_count(self) -> int:
        return self._step_count

    def record(self, cost: CostRecord) -> None:
        """Record a cost entry. Raises if budget exceeded."""
        self._records.append(cost)
        self._total_tokens += cost.total_tokens
        self._total_cost += cost.cost_usd
        self._step_count += 1

        # Per-thread accumulation
        tid = cost.thread_id
        if tid:
            entry = self._thread_totals.setdefault(
                tid, {"tokens": 0, "cost_usd": 0.0, "steps": 0}
            )
            entry["tokens"] += cost.total_tokens
            entry["cost_usd"] += cost.cost_usd
            entry["steps"] += 1

        self._check_budget()

    def thread_cost(self, thread_id: str) -> dict[str, Any]:
        """Return cost breakdown for a specific thread."""
        return dict(
            self._thread_totals.get(
                thread_id, {"tokens": 0, "cost_usd": 0.0, "steps": 0}
            )
        )

    def _check_budget(self) -> None:
        if self._budget is None:
            return
        if (
            self._budget.max_tokens is not None
            and self._total_tokens > self._budget.max_tokens
        ):
            raise BudgetExceeded(
                f"Token budget exceeded: {self._total_tokens} > {self._budget.max_tokens}"
            )
        if (
            self._budget.max_cost_usd is not None
            and self._total_cost > self._budget.max_cost_usd
        ):
            raise BudgetExceeded(
                f"Cost budget exceeded: ${self._total_cost:.4f} > ${self._budget.max_cost_usd:.4f}"
            )
        if (
            self._budget.max_steps is not None
            and self._step_count > self._budget.max_steps
        ):
            raise BudgetExceeded(
                f"Step budget exceeded: {self._step_count} > {self._budget.max_steps}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "step_count": self._step_count,
            "records": len(self._records),
            "per_thread": dict(self._thread_totals),
        }


class BudgetExceeded(Exception):
    """Raised when a cost budget is exceeded."""
    pass
