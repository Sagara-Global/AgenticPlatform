"""InMemoryRulesEngine — evaluates decision tables and scoring matrices in-memory."""

from __future__ import annotations

import operator
from typing import Any

from suluv.core.ports.rules_engine import RulesEngine, Decision


# Operator mapping for rule conditions
_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _parse_condition(condition: str, value: Any) -> bool:
    """Parse a condition string like '>500000' and test against a value."""
    for op_str, op_fn in _OPS.items():
        if condition.startswith(op_str):
            threshold = type(value)(condition[len(op_str):])
            return op_fn(value, threshold)
    # Exact match
    return str(value) == str(condition)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from dict or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class InMemoryRulesEngine(RulesEngine):
    """In-memory rules engine supporting decision tables and scoring matrices."""

    def __init__(self) -> None:
        self._tables: dict[str, Any] = {}

    async def register_table(self, table_name: str, table: Any) -> None:
        self._tables[table_name] = table

    async def get_table(self, table_name: str) -> Any | None:
        return self._tables.get(table_name)

    async def evaluate(self, table_name: str, inputs: dict[str, Any]) -> Decision:
        table = self._tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not registered")

        table_type = _get(table, "table_type", "decision")

        if table_type == "scoring":
            return self._evaluate_scoring(table, inputs)
        else:
            return self._evaluate_decision(table, inputs)

    def _evaluate_decision(self, table: Any, inputs: dict[str, Any]) -> Decision:
        """Evaluate a decision table."""
        hit_policy = _get(table, "hit_policy", "FIRST")
        rules = _get(table, "rules", [])
        matched: list[dict[str, Any]] = []

        for rule in rules:
            is_default = _get(rule, "default", False)
            if is_default:
                matched.append({"rule": rule, "outcome": _get(rule, "then")})
                continue

            conditions = _get(rule, "when", {})
            all_match = True
            for field_name, condition in conditions.items():
                if field_name not in inputs:
                    all_match = False
                    break
                if not _parse_condition(condition, inputs[field_name]):
                    all_match = False
                    break

            if all_match:
                matched.append({"rule": rule, "outcome": _get(rule, "then")})

                if hit_policy == "FIRST":
                    return Decision(
                        outcome=_get(rule, "then"),
                        matched_rules=matched,
                    )

        if hit_policy == "PRIORITY" and matched:
            # Sort by priority, lowest number = highest priority
            matched.sort(key=lambda m: _get(m["rule"], "priority", 999))
            return Decision(
                outcome=matched[0]["outcome"],
                matched_rules=matched,
            )

        if matched:
            outcomes = [m["outcome"] for m in matched]
            return Decision(
                outcome=outcomes if hit_policy in ("ALL", "COLLECT") else outcomes[0],
                matched_rules=matched,
            )

        return Decision(outcome=None, matched_rules=[])

    def _evaluate_scoring(self, table: Any, inputs: dict[str, Any]) -> Decision:
        """Evaluate a scoring matrix."""
        factors = _get(table, "factors", [])
        total_score = 0.0
        details: list[dict[str, Any]] = []

        for factor in factors:
            name = _get(factor, "name")
            weight = _get(factor, "weight", 1.0)
            value = inputs.get(name)
            if value is None:
                continue

            factor_score = 0
            for range_cond, points in _get(factor, "ranges", []):
                if _parse_condition(range_cond, value):
                    factor_score = points
                    break

            weighted = factor_score * weight
            total_score += weighted
            details.append({
                "factor": name,
                "value": value,
                "raw_score": factor_score,
                "weight": weight,
                "weighted_score": weighted,
            })

        return Decision(
            outcome=total_score,
            score=total_score,
            matched_rules=details,
        )

    def clear(self) -> None:
        self._tables.clear()
