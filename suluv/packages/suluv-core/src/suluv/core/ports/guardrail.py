"""Guardrail — safety filters for agent input/output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    action: GuardrailAction
    message: str = ""
    redacted_text: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.action in (GuardrailAction.ALLOW, GuardrailAction.WARN)


class Guardrail(ABC):
    """Port for input/output safety filtering."""

    @abstractmethod
    async def check_input(self, context: dict[str, Any], text: str) -> GuardrailResult:
        """Check input text before it reaches the LLM."""
        ...

    @abstractmethod
    async def check_output(self, context: dict[str, Any], text: str) -> GuardrailResult:
        """Check output text before it reaches the user."""
        ...
