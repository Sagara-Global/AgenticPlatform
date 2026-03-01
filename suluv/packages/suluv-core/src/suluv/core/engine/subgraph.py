"""SubgraphNode — nests a child GraphDefinition as a single node."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput

if TYPE_CHECKING:
    from suluv.core.engine.graph import GraphDefinition


class SubgraphNode(GraphNode):
    """Execute a nested graph definition as a single node step.

    The child graph receives the parent node's input and its final
    output becomes this node's output.  Requires a GraphRuntime
    reference at execute time (injected via input.context).
    """

    def __init__(
        self,
        subgraph: GraphDefinition,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.SUBGRAPH, name=name, **kwargs
        )
        self._subgraph = subgraph

    @property
    def subgraph(self) -> GraphDefinition:
        return self._subgraph

    async def execute(self, input: NodeInput) -> NodeOutput:
        # The runtime should inject itself in input.context["runtime"]
        runtime_factory = (input.context or {}).get("runtime_factory")
        if runtime_factory is None:
            return NodeOutput(
                data=None,
                success=False,
                error="No runtime_factory in context — cannot run subgraph",
            )

        child_runtime = runtime_factory(self._subgraph)
        result = await child_runtime.run(input.data)

        return NodeOutput(
            data=result,
            success=True,
            metadata={"subgraph_id": self._subgraph.graph_id},
        )
