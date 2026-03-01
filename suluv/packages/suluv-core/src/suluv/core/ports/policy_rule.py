"""PolicyRule — business rule evaluation for agent actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass
class PolicyResult:
    """Result of a policy evaluation."""

    decision: PolicyDecision
    reason: str = ""
    metadata: dict[str, Any] | None = None


class PolicyRule(ABC):
    """Port for business policy evaluation."""

    @abstractmethod
    async def evaluate(self, context: dict[str, Any], action: dict[str, Any]) -> PolicyResult:
        """Evaluate whether an action is allowed by policy."""
        ...
