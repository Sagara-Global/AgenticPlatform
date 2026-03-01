"""GraphRuntime — the main execution loop for graph definitions.

Implements frontier-based execution with fan-out/fan-in, error
handling, middleware, cancellation, event streaming, and state
persistence.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Callable

from suluv.core.types import (
    ErrorPolicy,
    ExecutionID,
    ExecutionResult,
    NodeID,
    NodeState,
    new_execution_id,
)
from suluv.core.engine.cancel import CancellationToken, CancellationError
from suluv.core.engine.edge import GraphEdge
from suluv.core.engine.events import (
    GraphCompleted,
    GraphEvent,
    NodeCompleted,
    NodeFailed,
    NodeRetrying,
    NodeStarted,
)
from suluv.core.engine.executor import NodeExecutor
from suluv.core.engine.graph import GraphDefinition
from suluv.core.engine.middleware import Middleware
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput
from suluv.core.engine.state import ExecutionState

logger = logging.getLogger("suluv.runtime")


class GraphRuntime:
    """Execute a GraphDefinition.

    Usage::

        runtime = GraphRuntime(graph)
        result = await runtime.run({"key": "value"})

    Or stream events::

        async for event in runtime.stream({"key": "value"}):
            print(event)
    """

    def __init__(
        self,
        graph: GraphDefinition,
        middlewares: list[Middleware] | None = None,
        state_store: Any | None = None,
        event_bus: Any | None = None,
        context: dict[str, Any] | None = None,
        max_concurrency: int = 10,
    ) -> None:
        errors = graph.validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        self._graph = graph
        self._executor = NodeExecutor(middlewares=middlewares)
        self._state_store = state_store
        self._event_bus = event_bus
        self._context = context or {}
        self._max_concurrency = max_concurrency
        self._cancel_token = CancellationToken()
        self._event_queue: asyncio.Queue[GraphEvent] = asyncio.Queue()

        # Inject runtime factory for subgraph nodes
        self._context.setdefault(
            "runtime_factory",
            lambda g: GraphRuntime(
                g,
                middlewares=middlewares,
                state_store=state_store,
                event_bus=event_bus,
                context=self._context,
            ),
        )

    @property
    def cancel_token(self) -> CancellationToken:
        return self._cancel_token

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        input_data: Any = None,
        execution_id: ExecutionID | None = None,
    ) -> ExecutionResult:
        """Execute the graph to completion and return the final result."""
        state = ExecutionState(
            execution_id=execution_id or new_execution_id()
        )

        try:
            await self._execute_graph(state, input_data)
        except CancellationError:
            for nid in state.node_states:
                if state.get_state(nid) in (NodeState.RUNNING, NodeState.PENDING):
                    state.set_state(nid, NodeState.CANCELLED)
        except Exception as exc:
            logger.exception("Graph execution failed: %s", exc)
            return ExecutionResult(
                execution_id=state.execution_id,
                success=False,
                error=str(exc),
                trace=state.trace,
                completed_at=datetime.now(timezone.utc),
            )

        # Collect output from exit nodes
        output = self._collect_output(state)
        success = all(
            state.get_state(nid) in (NodeState.DONE, NodeState.SKIPPED)
            for nid in self._graph.exit_nodes
        ) if self._graph.exit_nodes else not state.node_errors

        result = ExecutionResult(
            execution_id=state.execution_id,
            output=output,
            success=success,
            error=None if success else "; ".join(state.node_errors.values()),
            trace=state.trace,
            completed_at=datetime.now(timezone.utc),
        )

        await self._emit_event(GraphCompleted(
            execution_id=state.execution_id, result=result
        ))

        return result

    async def stream(
        self,
        input_data: Any = None,
        execution_id: ExecutionID | None = None,
    ) -> AsyncIterator[GraphEvent]:
        """Execute the graph, yielding events as they happen."""
        task = asyncio.create_task(self.run(input_data, execution_id))

        while not task.done():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue

        # Drain remaining events
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

        # Re-raise if task failed
        if task.exception():
            raise task.exception()  # type: ignore[misc]

    async def resume(
        self,
        state: ExecutionState,
        node_id: NodeID,
        data: Any = None,
    ) -> ExecutionResult:
        """Resume a paused execution (e.g., after human task completion).

        Sets the specified node to DONE with the given data and
        continues execution from there.
        """
        state.set_state(node_id, NodeState.DONE)
        state.set_output(node_id, data)

        try:
            await self._execute_graph(state, data, resume_from=node_id)
        except CancellationError:
            pass

        output = self._collect_output(state)
        success = not state.node_errors

        return ExecutionResult(
            execution_id=state.execution_id,
            output=output,
            success=success,
            trace=state.trace,
            completed_at=datetime.now(timezone.utc),
        )

    # ── Core execution loop ───────────────────────────────────────────────

    async def _execute_graph(
        self,
        state: ExecutionState,
        input_data: Any,
        resume_from: NodeID | None = None,
    ) -> None:
        """Frontier-based execution loop."""
        # Initialize node states
        for nid in self._graph.nodes:
            if nid not in state.node_states:
                state.set_state(nid, NodeState.PENDING)

        # Determine initial frontier
        if resume_from:
            frontier = self._compute_frontier(state, after_node=resume_from)
        else:
            frontier = list(self._graph.entry_nodes)
            # Store initial input for entry nodes
            for nid in frontier:
                state.set_output("__input__", input_data)

        sem = asyncio.Semaphore(self._max_concurrency)
        iteration = 0
        max_iterations = len(self._graph.nodes) * 3  # safety limit

        while frontier and iteration < max_iterations:
            iteration += 1
            self._cancel_token.check()

            # Execute frontier nodes concurrently
            tasks = []
            for nid in frontier:
                tasks.append(
                    self._execute_node_with_sem(
                        sem, state, nid, input_data
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle errors
            for nid, result in zip(frontier, results):
                if isinstance(result, Exception):
                    state.set_state(nid, NodeState.FAILED)
                    state.set_error(nid, str(result))
                    await self._emit_event(
                        NodeFailed(node_id=nid, error=str(result))
                    )

            # Compute next frontier
            frontier = self._compute_frontier(state)

            # Persist state after each wave
            if self._state_store:
                await self._state_store.save(
                    f"exec:{state.execution_id}", state.to_dict()
                )

    async def _execute_node_with_sem(
        self,
        sem: asyncio.Semaphore,
        state: ExecutionState,
        node_id: NodeID,
        graph_input: Any,
    ) -> None:
        """Execute a single node with semaphore-controlled concurrency."""
        async with sem:
            await self._execute_single_node(state, node_id, graph_input)

    async def _execute_single_node(
        self,
        state: ExecutionState,
        node_id: NodeID,
        graph_input: Any,
    ) -> None:
        """Execute one node: prepare input, run, update state."""
        node = self._graph.nodes[node_id]
        state.set_state(node_id, NodeState.RUNNING)

        await self._emit_event(
            NodeStarted(node_id=node_id, node_type=node.node_type.value)
        )

        # Build input from predecessor outputs
        input_data = self._build_node_input(state, node_id, graph_input)
        node_input = NodeInput(
            data=input_data,
            context={**self._context, "state": state},
        )

        # Find the edge that led here (for error policy)
        incoming = self._graph.get_incoming_edges(node_id)
        edge = incoming[0] if incoming else None

        # Execute
        output, record = await self._executor.execute(
            node, node_input, edge=edge, context=self._context
        )

        # Update state
        state.set_state(node_id, record.state)
        state.trace.append(record)

        if record.state == NodeState.DONE:
            state.set_output(node_id, output.data)
            await self._emit_event(
                NodeCompleted(node_id=node_id, result=output.data)
            )
        elif record.state == NodeState.WAITING:
            state.set_state(node_id, NodeState.WAITING)
        elif record.state == NodeState.FAILED:
            state.set_error(node_id, record.error or "Unknown error")
            await self._emit_event(
                NodeFailed(node_id=node_id, error=record.error or "")
            )

            # Handle fallback
            if edge and edge.error_policy == ErrorPolicy.FALLBACK and edge.fallback_node_id:
                fallback_id = edge.fallback_node_id
                if fallback_id in self._graph.nodes:
                    state.set_state(node_id, NodeState.SKIPPED)
                    await self._execute_single_node(
                        state, fallback_id, graph_input
                    )
        elif record.state == NodeState.SKIPPED:
            await self._emit_event(
                NodeCompleted(node_id=node_id, result=None)
            )

    # ── Frontier computation ──────────────────────────────────────────────

    def _compute_frontier(
        self,
        state: ExecutionState,
        after_node: NodeID | None = None,
    ) -> list[NodeID]:
        """Compute the next set of nodes ready to execute.

        A node is ready when all its predecessors have completed
        (DONE or SKIPPED) and the edge conditions are met.
        """
        ready: list[NodeID] = []

        if after_node:
            # Only consider successors of the specified node
            candidates = {
                e.target_id
                for e in self._graph.get_outgoing_edges(after_node)
            }
        else:
            candidates = set(self._graph.nodes.keys())

        for nid in candidates:
            current = state.get_state(nid)
            if current != NodeState.PENDING:
                continue

            incoming = self._graph.get_incoming_edges(nid)
            if not incoming:
                # Entry nodes (no predecessors)
                if nid in self._graph.entry_nodes:
                    ready.append(nid)
                continue

            # Check if all predecessors are done
            all_done = True
            any_fires = False

            for edge in incoming:
                src_state = state.get_state(edge.source_id)
                if src_state not in (NodeState.DONE, NodeState.SKIPPED):
                    all_done = False
                    break

                # Check edge condition
                src_output = state.node_outputs.get(edge.source_id)
                dummy_output = NodeOutput(data=src_output, success=True)
                if edge.should_fire(dummy_output):
                    any_fires = True

            if all_done and any_fires:
                ready.append(nid)

        return ready

    def _build_node_input(
        self,
        state: ExecutionState,
        node_id: NodeID,
        graph_input: Any,
    ) -> Any:
        """Build the input for a node from its predecessors' outputs."""
        incoming = self._graph.get_incoming_edges(node_id)

        if not incoming:
            # Entry node — use graph input
            return graph_input

        if len(incoming) == 1:
            edge = incoming[0]
            src_output = state.node_outputs.get(edge.source_id)
            dummy = NodeOutput(data=src_output, success=True)
            return edge.transform(dummy)

        # Multiple incoming edges — merge into dict
        merged: dict[str, Any] = {}
        for edge in incoming:
            src_output = state.node_outputs.get(edge.source_id)
            dummy = NodeOutput(data=src_output, success=True)
            merged[edge.source_id] = edge.transform(dummy)

        return merged

    def _collect_output(self, state: ExecutionState) -> Any:
        """Collect the final output from exit nodes."""
        if not self._graph.exit_nodes:
            return state.node_outputs

        if len(self._graph.exit_nodes) == 1:
            return state.node_outputs.get(self._graph.exit_nodes[0])

        return {
            nid: state.node_outputs.get(nid)
            for nid in self._graph.exit_nodes
        }

    # ── Event emission ────────────────────────────────────────────────────

    async def _emit_event(self, event: GraphEvent) -> None:
        """Emit an event to the queue and event bus."""
        self._event_queue.put_nowait(event)
        if self._event_bus:
            topic = type(event).__name__
            try:
                await self._event_bus.publish(topic, event)
            except Exception:
                logger.warning("Failed to publish event %s", topic)
