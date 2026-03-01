"""RulesEngine — evaluate decision tables and scoring matrices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Decision:
    """Result of a rules engine evaluation."""

    outcome: Any  # the matched result value
    matched_rules: list[dict[str, Any]] = field(default_factory=list)
    score: float | None = None  # for scoring matrices
    metadata: dict[str, Any] = field(default_factory=dict)


class RulesEngine(ABC):
    """Port for evaluating business rules (decision tables, scoring matrices)."""

    @abstractmethod
    async def evaluate(
        self, table_name: str, inputs: dict[str, Any]
    ) -> Decision:
        """Evaluate a named decision table with given inputs."""
        ...

    @abstractmethod
    async def register_table(self, table_name: str, table: Any) -> None:
        """Register a decision table or scoring matrix."""
        ...

    @abstractmethod
    async def get_table(self, table_name: str) -> Any | None:
        """Retrieve a registered table by name."""
        ...
