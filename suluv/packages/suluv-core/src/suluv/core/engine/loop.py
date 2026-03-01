"""LoopNode — repeats a body node until a condition is met."""

from __future__ import annotations

from typing import Any, Callable

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class LoopNode(GraphNode):
    """Iterative refinement — repeat a body node until condition met.

    The body node is executed repeatedly. After each iteration,
    the exit condition is checked. If True, the loop exits.
    """

    def __init__(
        self,
        body: GraphNode,
        exit_condition: Callable[[NodeOutput, int], bool],
        max_iterations: int = 10,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.LOOP, name=name, **kwargs
        )
        self._body = body
        self._exit_condition = exit_condition
        self._max_iterations = max_iterations

    async def execute(self, input: NodeInput) -> NodeOutput:
        current_input = input
        last_output = NodeOutput(data=input.data, success=True)

        for i in range(self._max_iterations):
            last_output = await self._body.execute(current_input)

            if not last_output.success:
                return last_output

            if self._exit_condition(last_output, i + 1):
                last_output.metadata["iterations"] = i + 1
                return last_output

            # Feed output back as input for next iteration
            current_input = NodeInput(
                data=last_output.data,
                context=input.context,
            )

        last_output.metadata["iterations"] = self._max_iterations
        last_output.metadata["max_reached"] = True
        return last_output
