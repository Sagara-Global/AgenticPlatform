"""NodeExecutor — dispatches execution to the correct node type."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from suluv.core.types import (
    ErrorPolicy,
    NodeExecution,
    NodeState,
    NodeType,
)
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput
from suluv.core.engine.edge import GraphEdge
from suluv.core.engine.middleware import Middleware

logger = logging.getLogger("suluv.executor")


class NodeExecutor:
    """Runs a single node, handling retries, error policy, and middleware.

    The GraphRuntime delegates per-node execution to this class.
    """

    def __init__(
        self,
        middlewares: list[Middleware] | None = None,
    ) -> None:
        self._middlewares = middlewares or []

    async def execute(
        self,
        node: GraphNode,
        input: NodeInput,
        edge: GraphEdge | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[NodeOutput, NodeExecution]:
        """Execute a node with middleware hooks and error handling.

        Returns (output, execution_record).
        """
        ctx = context or {}
        started = datetime.now(timezone.utc)
        error_policy = edge.error_policy if edge else ErrorPolicy.FAIL_FAST
        max_retries = edge.max_retries if edge else 3
        retry_delay = edge.retry_delay_seconds if edge else 1.0

        record = NodeExecution(
            node_id=node.node_id,
            node_type=node.node_type,
            state=NodeState.RUNNING,
            input=input.data,
            started_at=started,
        )

        # Run middleware before hooks
        for mw in self._middlewares:
            try:
                await mw.before_node(node, input, ctx)
            except Exception as exc:
                logger.warning("Middleware before_node error: %s", exc)

        attempt = 0
        output: NodeOutput | None = None

        while True:
            attempt += 1
            try:
                output = await node.execute(input)

                if output.success:
                    record.state = NodeState.DONE
                    record.output = output.data
                else:
                    # Node returned failure — treat like an error
                    if error_policy == ErrorPolicy.RETRY and attempt <= max_retries:
                        record.state = NodeState.RETRYING
                        record.retries = attempt
                        await asyncio.sleep(retry_delay)
                        continue
                    elif error_policy == ErrorPolicy.SKIP:
                        record.state = NodeState.SKIPPED
                        output = NodeOutput(data=None, success=True, metadata={"skipped": True})
                    elif error_policy == ErrorPolicy.FALLBACK:
                        record.state = NodeState.DONE
                        # Fallback handled by runtime
                    else:
                        record.state = NodeState.FAILED
                        record.error = output.error

                break

            except Exception as exc:
                logger.error("Node %s failed (attempt %d): %s", node.node_id, attempt, exc)

                if error_policy == ErrorPolicy.RETRY and attempt <= max_retries:
                    record.state = NodeState.RETRYING
                    record.retries = attempt
                    for mw in self._middlewares:
                        try:
                            await mw.on_error(node, exc, ctx)
                        except Exception:
                            pass
                    await asyncio.sleep(retry_delay)
                    continue
                elif error_policy == ErrorPolicy.SKIP:
                    record.state = NodeState.SKIPPED
                    output = NodeOutput(data=None, success=True, metadata={"skipped": True})
                else:
                    record.state = NodeState.FAILED
                    record.error = str(exc)
                    output = NodeOutput(data=None, success=False, error=str(exc))
                    for mw in self._middlewares:
                        try:
                            await mw.on_error(node, exc, ctx)
                        except Exception:
                            pass
                break

        record.completed_at = datetime.now(timezone.utc)

        # Run middleware after hooks
        if output and output.success:
            for mw in self._middlewares:
                try:
                    await mw.after_node(node, output, ctx)
                except Exception as exc:
                    logger.warning("Middleware after_node error: %s", exc)

        return output or NodeOutput(data=None, success=False), record
