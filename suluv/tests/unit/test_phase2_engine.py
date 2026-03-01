"""Phase 2 tests — Graph Engine: nodes, edges, graph, runtime, middleware."""

import asyncio
import pytest
from typing import Any

from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput
from suluv.core.engine.edge import GraphEdge
from suluv.core.engine.graph import GraphDefinition
from suluv.core.engine.state import ExecutionState
from suluv.core.engine.cancel import CancellationToken, CancellationError
from suluv.core.engine.events import NodeStarted, NodeCompleted, NodeFailed, GraphCompleted
from suluv.core.engine.middleware import Middleware, LogMiddleware, CostMiddleware
from suluv.core.engine.runtime import GraphRuntime
from suluv.core.engine.executor import NodeExecutor
from suluv.core.types import NodeType, NodeState, ErrorPolicy, NodeID


# ── Test Helpers ──────────────────────────────────────────────────────────────


class EchoNode(GraphNode):
    """Simple test node that returns its input."""

    def __init__(self, node_id: str = "", name: str = "echo", **kwargs):
        super().__init__(node_id=NodeID(node_id or "echo"), node_type=NodeType.TOOL, name=name, **kwargs)

    async def execute(self, input: NodeInput) -> NodeOutput:
        return NodeOutput(data=input.data, success=True)


class UpperNode(GraphNode):
    """Uppercases the input string."""

    def __init__(self, node_id: str = "", name: str = "upper", **kwargs):
        super().__init__(node_id=NodeID(node_id or "upper"), node_type=NodeType.TOOL, name=name, **kwargs)

    async def execute(self, input: NodeInput) -> NodeOutput:
        text = str(input.data) if input.data else ""
        return NodeOutput(data=text.upper(), success=True)


class FailNode(GraphNode):
    """Always fails."""

    def __init__(self, node_id: str = "", error_msg: str = "intentional failure", **kwargs):
        super().__init__(node_id=NodeID(node_id or "fail"), node_type=NodeType.TOOL, name="fail", **kwargs)
        self._error = error_msg

    async def execute(self, input: NodeInput) -> NodeOutput:
        raise RuntimeError(self._error)


class AddNode(GraphNode):
    """Adds a value to input."""

    def __init__(self, node_id: str, add_value: int = 1, **kwargs):
        super().__init__(node_id=NodeID(node_id), node_type=NodeType.TOOL, name=f"add-{add_value}", **kwargs)
        self._add = add_value

    async def execute(self, input: NodeInput) -> NodeOutput:
        val = (input.data or 0) + self._add
        return NodeOutput(data=val, success=True)


class SlowNode(GraphNode):
    """Sleeps briefly to test concurrency."""

    def __init__(self, node_id: str, delay: float = 0.05, **kwargs):
        super().__init__(node_id=NodeID(node_id), node_type=NodeType.TOOL, name="slow", **kwargs)
        self._delay = delay

    async def execute(self, input: NodeInput) -> NodeOutput:
        await asyncio.sleep(self._delay)
        return NodeOutput(data=input.data, success=True)


def make_linear_graph(*nodes: GraphNode) -> GraphDefinition:
    """Helper: wire nodes in a linear chain, first is entry, last is exit."""
    g = GraphDefinition(name="linear")
    for n in nodes:
        g.add_node(n)
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    g.set_entry(nodes[0])
    g.set_exit(nodes[-1])
    return g


# ── Node tests ────────────────────────────────────────────────────────────────


class TestGraphNode:
    def test_node_has_id(self):
        node = EchoNode("n1")
        assert node.node_id == "n1"
        assert node.node_type == NodeType.TOOL

    def test_node_to_dict(self):
        node = EchoNode("n1", name="echo-1")
        d = node.to_dict()
        assert d["node_id"] == "n1"
        assert d["node_type"] == "tool"
        assert d["name"] == "echo-1"

    @pytest.mark.asyncio
    async def test_echo_node(self):
        node = EchoNode("n1")
        out = await node.execute(NodeInput(data="hello"))
        assert out.data == "hello"
        assert out.success


# ── Edge tests ────────────────────────────────────────────────────────────────


class TestGraphEdge:
    def test_edge_creation(self):
        edge = GraphEdge(source_id=NodeID("a"), target_id=NodeID("b"))
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.error_policy == ErrorPolicy.FAIL_FAST

    def test_edge_unconditional_fires(self):
        edge = GraphEdge(source_id=NodeID("a"), target_id=NodeID("b"))
        out = NodeOutput(data="x", success=True)
        assert edge.should_fire(out) is True

    def test_edge_condition(self):
        edge = GraphEdge(
            source_id=NodeID("a"), target_id=NodeID("b"),
            condition=lambda o: o.data == "yes",
        )
        assert edge.should_fire(NodeOutput(data="yes")) is True
        assert edge.should_fire(NodeOutput(data="no")) is False

    def test_edge_transform(self):
        edge = GraphEdge(
            source_id=NodeID("a"), target_id=NodeID("b"),
            task_transform=lambda o: o.data * 2,
        )
        result = edge.transform(NodeOutput(data=5))
        assert result == 10

    def test_edge_no_transform(self):
        edge = GraphEdge(source_id=NodeID("a"), target_id=NodeID("b"))
        result = edge.transform(NodeOutput(data=5))
        assert result == 5

    def test_edge_to_dict(self):
        edge = GraphEdge(source_id=NodeID("a"), target_id=NodeID("b"))
        d = edge.to_dict()
        assert d["source_id"] == "a"
        assert d["target_id"] == "b"


# ── Graph tests ───────────────────────────────────────────────────────────────


class TestGraphDefinition:
    def test_add_node(self):
        g = GraphDefinition()
        n = EchoNode("n1")
        g.add_node(n)
        assert "n1" in g.nodes

    def test_add_edge(self):
        g = GraphDefinition()
        n1, n2 = EchoNode("n1"), EchoNode("n2")
        g.add_node(n1)
        g.add_node(n2)
        edge = g.add_edge(n1, n2)
        assert len(g.edges) == 1
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"

    def test_entry_exit(self):
        g = GraphDefinition()
        n1, n2 = EchoNode("n1"), EchoNode("n2")
        g.add_node(n1)
        g.add_node(n2)
        g.set_entry(n1)
        g.set_exit(n2)
        assert g.entry_nodes == ["n1"]
        assert g.exit_nodes == ["n2"]

    def test_get_outgoing_edges(self):
        g = GraphDefinition()
        n1, n2, n3 = EchoNode("n1"), EchoNode("n2"), EchoNode("n3")
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)
        g.add_edge(n1, n2)
        g.add_edge(n1, n3)
        assert len(g.get_outgoing_edges(NodeID("n1"))) == 2

    def test_get_incoming_edges(self):
        g = GraphDefinition()
        n1, n2, n3 = EchoNode("n1"), EchoNode("n2"), EchoNode("n3")
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)
        g.add_edge(n1, n3)
        g.add_edge(n2, n3)
        assert len(g.get_incoming_edges(NodeID("n3"))) == 2

    def test_validate_empty_graph(self):
        g = GraphDefinition()
        errors = g.validate()
        assert "Graph has no nodes" in errors

    def test_validate_no_entry(self):
        g = GraphDefinition()
        g.add_node(EchoNode("n1"))
        errors = g.validate()
        assert "Graph has no entry nodes" in errors

    def test_validate_missing_entry_node(self):
        g = GraphDefinition()
        g.add_node(EchoNode("n1"))
        g.entry_nodes = [NodeID("missing")]
        errors = g.validate()
        assert any("missing" in e for e in errors)

    def test_validate_success(self):
        g = make_linear_graph(EchoNode("n1"), EchoNode("n2"))
        errors = g.validate()
        assert errors == []

    def test_to_dict(self):
        g = make_linear_graph(EchoNode("n1"), EchoNode("n2"))
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 2


# ── ExecutionState tests ──────────────────────────────────────────────────────


class TestExecutionState:
    def test_initial_state(self):
        state = ExecutionState()
        assert state.execution_id.startswith("exec-")
        assert not state.node_states

    def test_get_set_state(self):
        state = ExecutionState()
        state.set_state(NodeID("n1"), NodeState.RUNNING)
        assert state.get_state(NodeID("n1")) == NodeState.RUNNING

    def test_default_state_is_pending(self):
        state = ExecutionState()
        assert state.get_state(NodeID("unknown")) == NodeState.PENDING

    def test_set_output(self):
        state = ExecutionState()
        state.set_output(NodeID("n1"), {"result": 42})
        assert state.node_outputs["n1"] == {"result": 42}

    def test_increment_retries(self):
        state = ExecutionState()
        assert state.increment_retries(NodeID("n1")) == 1
        assert state.increment_retries(NodeID("n1")) == 2

    def test_is_terminal(self):
        state = ExecutionState()
        state.set_state(NodeID("n1"), NodeState.DONE)
        state.set_state(NodeID("n2"), NodeState.DONE)
        assert state.is_terminal()

    def test_not_terminal_when_running(self):
        state = ExecutionState()
        state.set_state(NodeID("n1"), NodeState.DONE)
        state.set_state(NodeID("n2"), NodeState.RUNNING)
        assert not state.is_terminal()


# ── CancellationToken tests ──────────────────────────────────────────────────


class TestCancellationToken:
    def test_initial_not_cancelled(self):
        token = CancellationToken()
        assert not token.is_cancelled

    def test_cancel(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled

    def test_check_raises(self):
        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancellationError):
            token.check()

    @pytest.mark.asyncio
    async def test_wait_cancelled(self):
        token = CancellationToken()
        token.cancel()
        result = await token.wait(timeout=0.1)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        token = CancellationToken()
        result = await token.wait(timeout=0.05)
        assert result is False


# ── Middleware tests ──────────────────────────────────────────────────────────


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_log_middleware(self):
        log_mw = LogMiddleware()
        node = EchoNode("n1")
        inp = NodeInput(data="hi")
        out = NodeOutput(data="hi", success=True)
        await log_mw.before_node(node, inp, {})
        await log_mw.after_node(node, out, {})
        assert len(log_mw.logs) == 2
        assert "START" in log_mw.logs[0]
        assert "END" in log_mw.logs[1]

    @pytest.mark.asyncio
    async def test_cost_middleware(self):
        cost_mw = CostMiddleware()
        node = EchoNode("n1")
        out = NodeOutput(data="hi", success=True, metadata={"tokens": 100, "cost_usd": 0.01})
        await cost_mw.after_node(node, out, {})
        assert cost_mw.total_tokens == 100
        assert cost_mw.total_cost_usd == 0.01


# ── NodeExecutor tests ────────────────────────────────────────────────────────


class TestNodeExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        executor = NodeExecutor()
        node = EchoNode("n1")
        output, record = await executor.execute(node, NodeInput(data="hello"))
        assert output.success
        assert output.data == "hello"
        assert record.state == NodeState.DONE

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        executor = NodeExecutor()
        node = FailNode("f1")
        output, record = await executor.execute(node, NodeInput())
        assert not output.success
        assert record.state == NodeState.FAILED
        assert "intentional failure" in record.error

    @pytest.mark.asyncio
    async def test_execute_skip_policy(self):
        executor = NodeExecutor()
        node = FailNode("f1")
        edge = GraphEdge(
            source_id=NodeID("x"), target_id=NodeID("f1"),
            error_policy=ErrorPolicy.SKIP,
        )
        output, record = await executor.execute(node, NodeInput(), edge=edge)
        assert record.state == NodeState.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_retry_policy(self):
        executor = NodeExecutor()
        call_count = 0

        class CountFailNode(GraphNode):
            def __init__(self):
                super().__init__(node_id=NodeID("cf"), node_type=NodeType.TOOL, name="cf")

            async def execute(self, input: NodeInput) -> NodeOutput:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("fail")

        node = CountFailNode()
        edge = GraphEdge(
            source_id=NodeID("x"), target_id=NodeID("cf"),
            error_policy=ErrorPolicy.RETRY,
            max_retries=2,
            retry_delay_seconds=0.01,
        )
        output, record = await executor.execute(node, NodeInput(), edge=edge)
        assert call_count == 3  # initial + 2 retries
        assert record.state == NodeState.FAILED

    @pytest.mark.asyncio
    async def test_middleware_hooks(self):
        log_mw = LogMiddleware()
        executor = NodeExecutor(middlewares=[log_mw])
        node = EchoNode("n1")
        await executor.execute(node, NodeInput(data="hi"))
        assert len(log_mw.logs) == 2  # before + after


# ── GraphRuntime tests ────────────────────────────────────────────────────────


class TestGraphRuntime:
    @pytest.mark.asyncio
    async def test_single_node(self):
        g = GraphDefinition(name="single")
        n = EchoNode("n1")
        g.add_node(n)
        g.set_entry(n)
        g.set_exit(n)

        rt = GraphRuntime(g)
        result = await rt.run("hello")
        assert result.success
        assert result.output == "hello"

    @pytest.mark.asyncio
    async def test_linear_chain(self):
        n1 = EchoNode("n1")
        n2 = UpperNode("n2")
        g = make_linear_graph(n1, n2)

        rt = GraphRuntime(g)
        result = await rt.run("hello")
        assert result.success
        assert result.output == "HELLO"

    @pytest.mark.asyncio
    async def test_three_node_chain(self):
        n1 = AddNode("n1", 10)
        n2 = AddNode("n2", 20)
        n3 = AddNode("n3", 5)
        g = make_linear_graph(n1, n2, n3)

        rt = GraphRuntime(g)
        result = await rt.run(0)
        assert result.success
        assert result.output == 35  # 0+10+20+5

    @pytest.mark.asyncio
    async def test_fan_out(self):
        """A → B, A → C (parallel), D joins."""
        a = EchoNode("a")
        b = AddNode("b", 1)
        c = AddNode("c", 2)
        d = EchoNode("d")

        g = GraphDefinition(name="fan-out")
        g.add_node(a)
        g.add_node(b)
        g.add_node(c)
        g.add_node(d)
        g.add_edge(a, b)
        g.add_edge(a, c)
        g.add_edge(b, d)
        g.add_edge(c, d)
        g.set_entry(a)
        g.set_exit(d)

        rt = GraphRuntime(g)
        result = await rt.run(10)
        assert result.success
        # d gets merged input from b and c
        assert isinstance(result.output, dict)

    @pytest.mark.asyncio
    async def test_conditional_edge(self):
        """A → B (if data > 5), A → C (if data <= 5)."""
        a = EchoNode("a")
        b = EchoNode("b")
        c = EchoNode("c")

        g = GraphDefinition(name="conditional")
        g.add_node(a)
        g.add_node(b)
        g.add_node(c)
        g.add_edge(a, b, condition=lambda o: (o.data or 0) > 5)
        g.add_edge(a, c, condition=lambda o: (o.data or 0) <= 5)
        g.set_entry(a)
        # Only set the node that should actually fire as exit
        g.set_exit(b)

        rt = GraphRuntime(g)
        result = await rt.run(10)
        assert result.success
        assert result.output == 10

    @pytest.mark.asyncio
    async def test_error_fail_fast(self):
        n1 = EchoNode("n1")
        n2 = FailNode("n2")
        g = make_linear_graph(n1, n2)

        rt = GraphRuntime(g)
        result = await rt.run("hello")
        assert not result.success

    @pytest.mark.asyncio
    async def test_error_skip(self):
        n1 = EchoNode("n1")
        n2 = FailNode("n2")
        n3 = EchoNode("n3")

        g = GraphDefinition(name="skip-test")
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)
        g.add_edge(n1, n2, error_policy=ErrorPolicy.SKIP)
        g.add_edge(n2, n3)
        g.set_entry(n1)
        g.set_exit(n3)

        rt = GraphRuntime(g)
        result = await rt.run("hello")
        # n2 is skipped, n3 should still run
        assert result.success

    @pytest.mark.asyncio
    async def test_cancellation(self):
        n1 = SlowNode("n1", delay=0.5)
        g = GraphDefinition(name="cancel-test")
        g.add_node(n1)
        g.set_entry(n1)
        g.set_exit(n1)

        rt = GraphRuntime(g)

        async def cancel_soon():
            await asyncio.sleep(0.05)
            rt.cancel_token.cancel()

        asyncio.create_task(cancel_soon())
        result = await rt.run("hello")
        # May or may not succeed depending on timing, but shouldn't hang

    @pytest.mark.asyncio
    async def test_execution_trace(self):
        n1 = EchoNode("n1")
        n2 = EchoNode("n2")
        g = make_linear_graph(n1, n2)

        rt = GraphRuntime(g)
        result = await rt.run("hello")
        assert len(result.trace) >= 2

    @pytest.mark.asyncio
    async def test_middleware_integration(self):
        log_mw = LogMiddleware()
        n1 = EchoNode("n1")
        n2 = EchoNode("n2")
        g = make_linear_graph(n1, n2)

        rt = GraphRuntime(g, middlewares=[log_mw])
        result = await rt.run("hello")
        assert result.success
        assert len(log_mw.logs) >= 4  # before+after for each node

    @pytest.mark.asyncio
    async def test_state_persistence(self):
        from suluv.core.adapters import InMemoryStateStore
        store = InMemoryStateStore()

        n1 = EchoNode("n1")
        n2 = EchoNode("n2")
        g = make_linear_graph(n1, n2)

        rt = GraphRuntime(g, state_store=store)
        result = await rt.run("hello")
        assert result.success
        # State should have been persisted
        state = await store.load(f"exec:{result.execution_id}")
        assert state is not None

    @pytest.mark.asyncio
    async def test_invalid_graph_raises(self):
        g = GraphDefinition()  # empty
        with pytest.raises(ValueError, match="Invalid graph"):
            GraphRuntime(g)

    @pytest.mark.asyncio
    async def test_stream(self):
        n1 = EchoNode("n1")
        n2 = EchoNode("n2")
        g = make_linear_graph(n1, n2)

        rt = GraphRuntime(g)
        events = []
        async for event in rt.stream("hello"):
            events.append(event)
        assert len(events) >= 1
        # Should have a GraphCompleted event
        assert any(isinstance(e, GraphCompleted) for e in events)


# ── Specific Node Type tests ─────────────────────────────────────────────────


class TestToolNode:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        from suluv.core.engine.tool_node import ToolNode
        node = ToolNode(func=lambda data, ctx=None: data * 2, node_id="t1")
        out = await node.execute(NodeInput(data=5))
        assert out.data == 10
        assert out.success

    @pytest.mark.asyncio
    async def test_async_function(self):
        from suluv.core.engine.tool_node import ToolNode

        async def double(data, ctx=None):
            return data * 2

        node = ToolNode(func=double, node_id="t2")
        out = await node.execute(NodeInput(data=5))
        assert out.data == 10


class TestRouterNode:
    @pytest.mark.asyncio
    async def test_routes(self):
        from suluv.core.engine.router import RouterNode, Route
        node = RouterNode(
            routes=[
                Route(name="big", condition=lambda data: data > 100),
                Route(name="small", condition=lambda data: data <= 100),
            ],
            node_id="r1",
        )
        out = await node.execute(NodeInput(data=150))
        assert out.success
        assert out.data == "big"

    @pytest.mark.asyncio
    async def test_default_route(self):
        from suluv.core.engine.router import RouterNode, Route
        node = RouterNode(
            routes=[
                Route(name="special", condition=lambda data: data == "magic"),
            ],
            default_route="fallback",
            node_id="r2",
        )
        out = await node.execute(NodeInput(data="normal"))
        assert out.data == "fallback"


class TestLoopNode:
    @pytest.mark.asyncio
    async def test_basic_loop(self):
        from suluv.core.engine.loop import LoopNode

        body = AddNode("body", 1)
        loop = LoopNode(
            body=body,
            exit_condition=lambda output, iteration: output.data >= 5,
            max_iterations=100,
            node_id="loop1",
        )
        out = await loop.execute(NodeInput(data=0))
        assert out.success
        assert out.data == 5

    @pytest.mark.asyncio
    async def test_loop_max_iterations(self):
        from suluv.core.engine.loop import LoopNode

        body = AddNode("body", 1)
        loop = LoopNode(
            body=body,
            exit_condition=lambda output, iteration: False,  # never exits
            max_iterations=3,
            node_id="loop2",
        )
        out = await loop.execute(NodeInput(data=0))
        assert out.data == 3  # ran 3 times


class TestMapNode:
    @pytest.mark.asyncio
    async def test_parallel_map(self):
        from suluv.core.engine.map_node import MapNode

        class SquareNode(GraphNode):
            def __init__(self):
                super().__init__(node_id=NodeID("sq"), node_type=NodeType.TOOL)

            async def execute(self, input: NodeInput) -> NodeOutput:
                return NodeOutput(data=input.data ** 2, success=True)

        node = MapNode(body=SquareNode(), node_id="map1")
        out = await node.execute(NodeInput(data=[1, 2, 3, 4]))
        assert out.success
        assert sorted(out.data) == [1, 4, 9, 16]


class TestGatewayNode:
    @pytest.mark.asyncio
    async def test_n_of_m_met(self):
        from suluv.core.engine.gateway import GatewayNode
        node = GatewayNode(required=2, total=3, node_id="gw1")
        out = await node.execute(NodeInput(data={"a": "ok", "b": "ok", "c": None}))
        assert out.success
        assert out.metadata.get("completed") == 2

    @pytest.mark.asyncio
    async def test_n_of_m_not_met(self):
        from suluv.core.engine.gateway import GatewayNode
        node = GatewayNode(required=3, total=3, node_id="gw2")
        out = await node.execute(NodeInput(data={"a": "ok", "b": None, "c": None}))
        assert out.metadata.get("waiting") is True


class TestDelayNode:
    @pytest.mark.asyncio
    async def test_short_delay(self):
        from suluv.core.engine.delay import DelayNode
        from datetime import timedelta
        node = DelayNode(delay=timedelta(seconds=0.01), node_id="d1")
        out = await node.execute(NodeInput(data="hello"))
        assert out.success
        assert out.data == "hello"


class TestTriggerNode:
    @pytest.mark.asyncio
    async def test_manual_trigger(self):
        from suluv.core.engine.trigger import TriggerNode, TriggerType
        node = TriggerNode(trigger_type=TriggerType.MANUAL, node_id="tr1")
        out = await node.execute(NodeInput(data={"event": "start"}))
        assert out.success


class TestDecisionNode:
    @pytest.mark.asyncio
    async def test_with_rules_engine(self):
        from suluv.core.engine.decision import DecisionNode
        from suluv.core.adapters import InMemoryRulesEngine

        engine = InMemoryRulesEngine()
        table = {
            "name": "test",
            "inputs": ["score"],
            "output": "result",
            "hit_policy": "FIRST",
            "rules": [
                {"when": {"score": ">80"}, "then": "pass"},
                {"default": True, "then": "fail"},
            ],
        }
        await engine.register_table("test", table)

        node = DecisionNode(table_name="test", node_id="dec1")
        out = await node.execute(NodeInput(
            data={"score": 90},
            context={"rules_engine": engine},
        ))
        assert out.success
        assert out.data["decisions"][0]["action"] == "pass"


class TestFormNode:
    @pytest.mark.asyncio
    async def test_form_response_validation(self):
        from suluv.core.engine.form import FormNode, FormSchema, FormField
        schema = FormSchema(
            title="Test Form",
            fields=[
                FormField(name="name", required=True),
                FormField(name="email", required=True),
            ],
        )
        node = FormNode(schema=schema, node_id="form1")

        # Valid response (form responses don't need task_queue)
        out = await node.execute(NodeInput(
            data={"_form_response": {"name": "Ravi", "email": "r@x.com"}},
        ))
        assert out.success
        assert out.data["name"] == "Ravi"

    @pytest.mark.asyncio
    async def test_form_missing_required(self):
        from suluv.core.engine.form import FormNode, FormSchema, FormField
        schema = FormSchema(
            title="Test",
            fields=[FormField(name="name", required=True)],
        )
        node = FormNode(schema=schema, node_id="form2")
        out = await node.execute(NodeInput(
            data={"_form_response": {"name": ""}},
        ))
        assert not out.success
        assert "required" in str(out.data)


class TestSignalNode:
    @pytest.mark.asyncio
    async def test_throw_signal(self):
        from suluv.core.engine.signal import SignalNode, SignalMode
        from suluv.core.adapters import InMemoryEventBus
        bus = InMemoryEventBus()
        node = SignalNode(signal_name="alert", mode=SignalMode.THROW, node_id="sig1")
        out = await node.execute(NodeInput(
            data={"details": "fraud"},
            context={"event_bus": bus},
        ))
        assert out.success
        assert "alert" in out.data["signal_sent"]
        # Event should be in the bus history
        assert any("signal.alert" in topic for topic, _ in bus.history)


class TestCompensationNode:
    @pytest.mark.asyncio
    async def test_compensation_runs_reverse(self):
        from suluv.core.engine.compensation import CompensationNode, CompensationAction
        log = []

        async def comp_a(data):
            log.append("A")

        async def comp_b(data):
            log.append("B")

        node = CompensationNode(
            actions=[
                CompensationAction(step_name="step_a", action=comp_a),
                CompensationAction(step_name="step_b", action=comp_b),
            ],
            node_id="comp1",
        )
        out = await node.execute(NodeInput())
        assert out.success
        # Actions are reversed in __init__, so B runs first
        assert log == ["B", "A"]

    @pytest.mark.asyncio
    async def test_compensation_failure_handling(self):
        from suluv.core.engine.compensation import CompensationNode, CompensationAction

        async def failing_comp(data):
            raise RuntimeError("comp failed")

        node = CompensationNode(
            actions=[CompensationAction(step_name="fail_step", action=failing_comp)],
            node_id="comp2",
        )
        out = await node.execute(NodeInput())
        assert not out.success
        assert out.data["compensations"][0]["status"] == "failed"


class TestTimerNode:
    @pytest.mark.asyncio
    async def test_short_timer(self):
        from suluv.core.engine.timer import TimerNode
        from datetime import timedelta
        node = TimerNode(duration=timedelta(seconds=0.01), node_id="timer1")
        out = await node.execute(NodeInput(data="payload"))
        assert out.success
        assert out.data == "payload"

    @pytest.mark.asyncio
    async def test_no_wait_passthrough(self):
        from suluv.core.engine.timer import TimerNode
        node = TimerNode(node_id="timer2")
        out = await node.execute(NodeInput(data="payload"))
        assert out.success
