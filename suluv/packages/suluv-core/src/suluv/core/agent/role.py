"""AgentRole — defines an agent's identity and capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRole:
    """Describes what an agent is, what it can do, and its constraints.

    Used to generate the system prompt and to enforce operational limits.
    """

    name: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    instructions: str = ""
    max_steps: int = 25
    max_tokens_per_step: int = 4096
    temperature: float = 0.1
    output_format: str | None = None  # "json", "markdown", etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_system_prompt(self) -> str:
        """Generate a system prompt from the role definition."""
        parts = [f"You are {self.name}."]
        if self.description:
            parts.append(self.description)
        if self.capabilities:
            parts.append(
                "Your capabilities: "
                + ", ".join(self.capabilities)
                + "."
            )
        if self.instructions:
            parts.append(self.instructions)
        if self.output_format:
            parts.append(
                f"Always respond in {self.output_format} format."
            )
        return "\n\n".join(parts)
