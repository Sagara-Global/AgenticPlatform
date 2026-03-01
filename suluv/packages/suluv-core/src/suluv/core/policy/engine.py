"""PolicyEngine — evaluates a chain of policy rules."""

from __future__ import annotations

from typing import Any

from suluv.core.ports.policy_rule import PolicyDecision, PolicyResult, PolicyRule


class PolicyEngine:
    """Evaluates a list of policy rules against an action.

    Combines results: any DENY → overall DENY.
    Any ESCALATE → overall ESCALATE (if no DENY).
    Otherwise ALLOW.
    """

    def __init__(self, rules: list[PolicyRule] | None = None) -> None:
        self._rules = rules or []

    def add_rule(self, rule: PolicyRule) -> None:
        self._rules.append(rule)

    async def evaluate(
        self,
        action: str,
        context: dict[str, Any],
    ) -> PolicyResult:
        """Evaluate all rules and return the combined decision."""
        results: list[PolicyResult] = []

        for rule in self._rules:
            result = await rule.evaluate(action, context)
            results.append(result)

            if result.decision == PolicyDecision.DENY:
                return result  # fail fast

        # Check for escalation
        escalations = [r for r in results if r.decision == PolicyDecision.ESCALATE]
        if escalations:
            return PolicyResult(
                decision=PolicyDecision.ESCALATE,
                reason="; ".join(r.reason for r in escalations),
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="All policies passed",
        )
