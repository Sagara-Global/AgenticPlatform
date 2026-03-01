"""TimerNode — waits until a specific time or for a duration (process-aware)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from suluv.core.types import NodeType
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class TimerNode(GraphNode):
    """Process-aware timer that respects business calendars.

    Can wait until:
    - A fixed datetime (``until``)
    - A duration (``duration``)
    - A business-hours duration via BusinessCalendar port

    When a business_calendar is present in context, durations are
    computed in business hours rather than wall-clock time.
    """

    def __init__(
        self,
        until: datetime | None = None,
        duration: timedelta | None = None,
        use_business_hours: bool = False,
        node_id: str | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=node_id, node_type=NodeType.TIMER, name=name, **kwargs
        )
        self._until = until
        self._duration = duration
        self._use_business_hours = use_business_hours

    async def execute(self, input: NodeInput) -> NodeOutput:
        now = datetime.now(timezone.utc)

        if self._until is not None:
            target = self._until
        elif self._duration is not None:
            if self._use_business_hours:
                calendar = (input.context or {}).get("business_calendar")
                if calendar:
                    target = await calendar.add_business_hours(
                        now, self._duration
                    )
                else:
                    target = now + self._duration
            else:
                target = now + self._duration
        else:
            # No wait configured — pass through
            return NodeOutput(data=input.data, success=True)

        wait_seconds = max(0, (target - now).total_seconds())

        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        return NodeOutput(
            data=input.data,
            success=True,
            metadata={
                "waited_seconds": wait_seconds,
                "target": target.isoformat(),
                "business_hours": self._use_business_hours,
            },
        )
