"""Middleware — before/after node hooks for cross-cutting concerns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from suluv.core.types import AuditEvent, new_id
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput


class Middleware(ABC):
    """Abstract middleware for graph execution hooks."""

    async def before_node(
        self, node: GraphNode, input: NodeInput, context: dict[str, Any]
    ) -> None:
        """Called before a node executes."""
        pass

    async def after_node(
        self, node: GraphNode, output: NodeOutput, context: dict[str, Any]
    ) -> None:
        """Called after a node completes successfully."""
        pass

    async def on_error(
        self, node: GraphNode, error: Exception, context: dict[str, Any]
    ) -> None:
        """Called when a node fails."""
        pass


class AuditMiddleware(Middleware):
    """Logs node execution to an AuditBackend."""

    def __init__(self, audit_backend: Any) -> None:
        self._audit = audit_backend

    async def before_node(
        self, node: GraphNode, input: NodeInput, context: dict[str, Any]
    ) -> None:
        await self._audit.write(AuditEvent(
            event_type="node_started",
            node_id=node.node_id,
            org_id=context.get("org_id"),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            thread_id=context.get("thread_id"),
            execution_id=context.get("execution_id"),
            data={"node_type": node.node_type.value, "name": node.name},
        ))

    async def after_node(
        self, node: GraphNode, output: NodeOutput, context: dict[str, Any]
    ) -> None:
        await self._audit.write(AuditEvent(
            event_type="node_completed",
            node_id=node.node_id,
            org_id=context.get("org_id"),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            thread_id=context.get("thread_id"),
            execution_id=context.get("execution_id"),
            data={"success": output.success},
        ))

    async def on_error(
        self, node: GraphNode, error: Exception, context: dict[str, Any]
    ) -> None:
        await self._audit.write(AuditEvent(
            event_type="node_failed",
            node_id=node.node_id,
            org_id=context.get("org_id"),
            user_id=context.get("user_id"),
            session_id=context.get("session_id"),
            thread_id=context.get("thread_id"),
            execution_id=context.get("execution_id"),
            data={"error": str(error)},
        ))


class CostMiddleware(Middleware):
    """Tracks token costs across node executions."""

    def __init__(self) -> None:
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._per_node: dict[str, dict[str, Any]] = {}

    async def after_node(
        self, node: GraphNode, output: NodeOutput, context: dict[str, Any]
    ) -> None:
        tokens = output.metadata.get("tokens", 0)
        cost = output.metadata.get("cost_usd", 0.0)
        self.total_tokens += tokens
        self.total_cost_usd += cost
        self._per_node[node.node_id] = {"tokens": tokens, "cost_usd": cost}


class LogMiddleware(Middleware):
    """Simple logging middleware."""

    def __init__(self) -> None:
        self.logs: list[str] = []

    async def before_node(
        self, node: GraphNode, input: NodeInput, context: dict[str, Any]
    ) -> None:
        self.logs.append(f"START {node.name} ({node.node_type.value})")

    async def after_node(
        self, node: GraphNode, output: NodeOutput, context: dict[str, Any]
    ) -> None:
        status = "OK" if output.success else "FAIL"
        self.logs.append(f"END {node.name} [{status}]")

    async def on_error(
        self, node: GraphNode, error: Exception, context: dict[str, Any]
    ) -> None:
        self.logs.append(f"ERROR {node.name}: {error}")
