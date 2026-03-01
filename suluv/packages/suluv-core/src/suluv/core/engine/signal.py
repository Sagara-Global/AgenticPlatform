"""SignalNode — emits or waits for inter-process signals."""

from __future__ import annotations

from enum import Enum
from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class SignalMode(str, Enum):
    THROW = "throw"  # emit signal
    CATCH = "catch"  # wait for signal


class SignalNode(GraphNode):
    """Throw or catch a named signal on the EventBus.

    THROW mode publishes a signal event.
    CATCH mode subscribes and waits for a matching signal.
    """

    def __init__(
        self,
        signal_name: str,
        mode: SignalMode = SignalMode.THROW,
        timeout_seconds: float | None = None,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.SIGNAL, name=name, **kwargs
        )
        self._signal_name = signal_name
        self._mode = mode
        self._timeout = timeout_seconds

    async def execute(self, input: NodeInput) -> NodeOutput:
        event_bus = (input.context or {}).get("event_bus")
        if event_bus is None:
            return NodeOutput(
                data=None, success=False, error="No event_bus in context"
            )

        if self._mode == SignalMode.THROW:
            await event_bus.publish(
                f"signal.{self._signal_name}",
                {"data": input.data, "signal": self._signal_name},
            )
            return NodeOutput(
                data={"signal_sent": self._signal_name}, success=True
            )

        else:
            # CATCH: use request/reply pattern to wait
            import asyncio

            fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()

            async def _handler(payload: Any) -> None:
                if not fut.done():
                    fut.set_result(payload)

            await event_bus.subscribe(f"signal.{self._signal_name}", _handler)

            try:
                result = await asyncio.wait_for(fut, timeout=self._timeout)
                return NodeOutput(data=result, success=True)
            except asyncio.TimeoutError:
                return NodeOutput(
                    data=None,
                    success=False,
                    error=f"Signal '{self._signal_name}' timed out after {self._timeout}s",
                )
            finally:
                await event_bus.unsubscribe(
                    f"signal.{self._signal_name}", _handler
                )
