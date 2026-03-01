"""DecisionNode — evaluates business rules via RulesEngine port."""

from __future__ import annotations

from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class DecisionNode(GraphNode):
    """Evaluate a decision table and route based on the result.

    Wraps the RulesEngine port for inline use within a graph.
    The decision table name must be registered with the rules engine
    before the graph is executed.
    """

    def __init__(
        self,
        table_name: str,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.DECISION, name=name, **kwargs
        )
        self._table_name = table_name

    async def execute(self, input: NodeInput) -> NodeOutput:
        rules_engine = (input.context or {}).get("rules_engine")
        if rules_engine is None:
            return NodeOutput(
                data=None,
                success=False,
                error="No rules_engine in context",
            )

        facts = input.data if isinstance(input.data, dict) else {}
        decision = await rules_engine.evaluate(self._table_name, facts)

        return NodeOutput(
            data={
                "decisions": [
                    {
                        "action": decision.outcome,
                        "score": decision.score,
                        "matched_rules": decision.matched_rules,
                    }
                ],
                "table": self._table_name,
            },
            success=True,
        )
