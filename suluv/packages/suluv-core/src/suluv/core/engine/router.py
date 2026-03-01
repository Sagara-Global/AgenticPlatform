"""RouterNode — conditional branching, no execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


@dataclass
class Route:
    """A named route with a condition."""

    name: str
    condition: Callable[[Any], bool]


class RouterNode(GraphNode):
    """Evaluates conditions to determine which path to take.

    Routes are checked in order; the first matching route name
    is returned as the output. Edge conditions can match on this.
    """

    def __init__(
        self,
        routes: list[Route] | None = None,
        default_route: str = "default",
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.ROUTER, name=name, **kwargs
        )
        self._routes = routes or []
        self._default = default_route

    async def execute(self, input: NodeInput) -> NodeOutput:
        for route in self._routes:
            try:
                if route.condition(input.data):
                    return NodeOutput(data=route.name, success=True)
            except Exception:
                continue
        return NodeOutput(data=self._default, success=True)
