"""TriggerNode — initiates graph execution (webhook/cron/event)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class TriggerType(str, Enum):
    WEBHOOK = "webhook"
    CRON = "cron"
    EVENT = "event"
    MANUAL = "manual"


class TriggerNode(GraphNode):
    """Entry point that initiates a graph execution.

    By the time execute() runs, the trigger has already fired.
    This node simply passes through the trigger payload.
    """

    def __init__(
        self,
        trigger_type: TriggerType = TriggerType.MANUAL,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.TRIGGER, name=name, **kwargs
        )
        self.trigger_type = trigger_type

    async def execute(self, input: NodeInput) -> NodeOutput:
        return NodeOutput(data=input.data, success=True)
