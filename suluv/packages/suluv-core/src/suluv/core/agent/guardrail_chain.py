"""GuardrailChain — runs input/output through a chain of guardrails."""

from __future__ import annotations

from suluv.core.ports.guardrail import Guardrail, GuardrailAction, GuardrailResult


class GuardrailChain:
    """Evaluates a chain of guardrails in order.

    Stops on first BLOCK or REDACT. ALLOW continues to next guardrail.
    If all pass, the overall result is ALLOW.
    """

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self._guardrails = guardrails or []

    def add(self, guardrail: Guardrail) -> None:
        self._guardrails.append(guardrail)

    async def check_input(self, content: str, context: dict | None = None) -> GuardrailResult:
        """Run all guardrails on input content."""
        ctx = context or {}
        for g in self._guardrails:
            result = await g.check_input(ctx, content)
            if result.action in (GuardrailAction.BLOCK, GuardrailAction.REDACT):
                return result
        return GuardrailResult(
            action=GuardrailAction.ALLOW, message="All guardrails passed"
        )

    async def check_output(self, content: str, context: dict | None = None) -> GuardrailResult:
        """Run all guardrails on output content."""
        ctx = context or {}
        for g in self._guardrails:
            result = await g.check_output(ctx, content)
            if result.action in (GuardrailAction.BLOCK, GuardrailAction.REDACT):
                return result
        return GuardrailResult(
            action=GuardrailAction.ALLOW, message="All guardrails passed"
        )
