"""VerificationPort — identity verification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationResult:
    valid: bool
    identity_type: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class VerificationPort(ABC):
    """Port for verifying identities (PAN, Aadhaar, etc.)."""

    @abstractmethod
    async def verify(self, identity: str, identity_type: str, context: dict[str, Any] | None = None) -> VerificationResult:
        """Verify an identity value."""
        ...
