"""CompensationNode — saga-style rollback for failed process steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


@dataclass
class CompensationAction:
    """A single compensating action to undo a previous step."""

    step_name: str
    action: Callable[..., Awaitable[Any]]
    description: str = ""


class CompensationNode(GraphNode):
    """Execute compensating actions in reverse order (saga rollback).

    Records which compensation actions were executed and any failures.
    Compensation continues even if individual actions fail (best-effort).
    """

    def __init__(
        self,
        actions: list[CompensationAction] | None = None,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.COMPENSATION,
            name=name,
            **kwargs,
        )
        self._actions = list(reversed(actions or []))

    def add_action(self, action: CompensationAction) -> None:
        """Add a compensation action (will be prepended — LIFO)."""
        self._actions.insert(0, action)

    async def execute(self, input: NodeInput) -> NodeOutput:
        results: list[dict[str, Any]] = []
        all_ok = True

        # If actions are stored in the execution context stack, use those
        context_actions = (input.context or {}).get("compensation_stack")
        actions = context_actions if context_actions else self._actions

        for action in actions:
            try:
                await action.action(input.data)
                results.append(
                    {
                        "step": action.step_name,
                        "status": "compensated",
                        "description": action.description,
                    }
                )
            except Exception as exc:
                all_ok = False
                results.append(
                    {
                        "step": action.step_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        return NodeOutput(
            data={"compensations": results},
            success=all_ok,
            metadata={
                "total": len(actions),
                "succeeded": sum(
                    1 for r in results if r["status"] == "compensated"
                ),
            },
        )
