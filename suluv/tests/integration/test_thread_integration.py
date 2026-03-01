"""Tests — Thread integration across ALL agent subsystems.

Verifies that thread_id flows correctly through:
- MemoryManager (load_context, save_interaction, scoped short-term)
- ToolRunner (audit events, call history)
- CostTracker (per-thread token/cost breakdown)
- GuardrailChain (context dict includes thread_id)
- Full agent pipeline (end-to-end thread propagation)
"""

from __future__ import annotations

import json
import pytest

from suluv.core.adapters.mock_llm import MockLLM
from suluv.core.adapters.memory_audit import InMemoryAuditBackend
from suluv.core.adapters.memory_checkpointer import InMemoryCheckpointer
from suluv.core.adapters.memory_memory import (
    InMemoryEpisodicMemory,
    InMemoryShortTermMemory,
)
from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.context import AgentContext
from suluv.core.agent.cost_tracker import CostBudget, CostTracker
from suluv.core.agent.guardrail_chain import GuardrailChain
from suluv.core.agent.memory_manager import MemoryManager
from suluv.core.agent.role import AgentRole
from suluv.core.ports.guardrail import Guardrail, GuardrailAction, GuardrailResult
from suluv.core.tools.builtins import calculator
from suluv.core.tools.runner import ToolRunner
from suluv.core.types import CostRecord


# ── Helpers ──────────────────────────────────────────────────


def _react_final(thought: str, answer: str) -> str:
    return json.dumps({"thought": thought, "final_answer": answer})


def _react_tool_call(thought: str, tool: str, args: dict) -> str:
    return json.dumps({"thought": thought, "action": tool, "action_input": args})


class _SpyGuardrail(Guardrail):
    """Records the context dict passed to check_input / check_output."""

    def __init__(self) -> None:
        self.input_contexts: list[dict] = []
        self.output_contexts: list[dict] = []

    async def check_input(self, context: dict, content: str) -> GuardrailResult:
        self.input_contexts.append(dict(context))
        return GuardrailResult(action=GuardrailAction.ALLOW, message="ok")

    async def check_output(self, context: dict, content: str) -> GuardrailResult:
        self.output_contexts.append(dict(context))
        return GuardrailResult(action=GuardrailAction.ALLOW, message="ok")


# ──────────────────────────────────────────────────────────────
#  1. MemoryManager — thread_id scoping
# ──────────────────────────────────────────────────────────────


class TestMemoryManagerThreads:
    """MemoryManager load_context / save_interaction respect thread_id."""

    async def test_save_interaction_scopes_by_thread(self):
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        await mm.save_interaction(
            session_id="s1", user_id="u1", content="hello",
            thread_id="t1",
        )
        all_items = await stm.all()
        # Short-term key should be prefixed with t:t1:
        keys = list(all_items.keys())
        assert len(keys) == 1
        assert keys[0].startswith("t:t1:")

    async def test_save_interaction_no_thread_uses_session(self):
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        await mm.save_interaction(
            session_id="s1", user_id="u1", content="hello",
        )
        all_items = await stm.all()
        keys = list(all_items.keys())
        assert len(keys) == 1
        assert keys[0].startswith("t:s1:")

    async def test_load_context_filters_by_thread(self):
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        # Write entries under two different threads
        await mm.save_interaction(
            session_id="s1", content="thread1 data", thread_id="t1",
        )
        await mm.save_interaction(
            session_id="s1", content="thread2 data", thread_id="t2",
        )
        # Load only thread1 scoped entries
        ctx = await mm.load_context(session_id="s1", thread_id="t1")
        stm_data = ctx.get("short_term", {})
        assert len(stm_data) == 1
        assert all(k.startswith("t:t1:") for k in stm_data)
        assert list(stm_data.values())[0] == "thread1 data"

    async def test_episodic_stores_thread_id_in_metadata(self):
        ep = InMemoryEpisodicMemory()
        mm = MemoryManager(episodic=ep)
        await mm.save_interaction(
            session_id="s1", content="important", thread_id="t42",
        )
        assert len(ep._episodes) == 1
        assert ep._episodes[0]["metadata"]["thread_id"] == "t42"

    async def test_episodic_no_thread_no_metadata_key(self):
        ep = InMemoryEpisodicMemory()
        mm = MemoryManager(episodic=ep)
        await mm.save_interaction(session_id="s1", content="plain")
        assert len(ep._episodes) == 1
        assert "thread_id" not in ep._episodes[0]["metadata"]

    async def test_backward_compat_no_thread_id(self):
        """Calling without thread_id still works (backward compat)."""
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        await mm.save_interaction(session_id="s1", content="compat")
        ctx = await mm.load_context(session_id="s1")
        # Should get all items (no thread filter applies)
        stm_data = ctx.get("short_term", {})
        assert len(stm_data) == 1


# ──────────────────────────────────────────────────────────────
#  2. ToolRunner — thread context in audit & history
# ──────────────────────────────────────────────────────────────


class TestToolRunnerThreadContext:
    """ToolRunner.run() propagates context to audit and call_history."""

    async def test_call_history_includes_thread_id(self):
        runner = ToolRunner()
        ctx = {"thread_id": "t1", "user_id": "u1", "org_id": "o1"}
        await runner.run(calculator, {"expression": "1+1"}, context=ctx)
        assert len(runner.call_history) == 1
        rec = runner.call_history[0]
        assert rec["thread_id"] == "t1"
        assert rec["user_id"] == "u1"
        assert rec["org_id"] == "o1"

    async def test_call_history_no_context(self):
        """Without context all identity fields are None — backward compat."""
        runner = ToolRunner()
        await runner.run(calculator, {"expression": "2+3"})
        rec = runner.call_history[0]
        assert rec["thread_id"] is None
        assert rec["user_id"] is None

    async def test_audit_event_includes_thread_id(self):
        audit = InMemoryAuditBackend()
        runner = ToolRunner(audit_backend=audit)
        ctx = {"thread_id": "t1", "user_id": "u1", "session_id": "s1"}
        await runner.run(calculator, {"expression": "3*4"}, context=ctx)
        events = audit.events
        assert len(events) == 1
        assert events[0].event_type == "tool_call"
        assert events[0].user_id == "u1"
        assert events[0].session_id == "s1"
        assert events[0].thread_id == "t1"

    async def test_audit_event_on_error_has_thread(self):
        from suluv.core.tools.decorator import suluv_tool

        @suluv_tool(name="fail_tool", description="always fails")
        async def fail_tool() -> str:
            raise ValueError("boom")

        audit = InMemoryAuditBackend()
        runner = ToolRunner(audit_backend=audit)
        ctx = {"thread_id": "t-err", "user_id": "u2"}
        await runner.run(fail_tool, {}, context=ctx)
        error_events = [e for e in audit.events if e.event_type == "tool_error"]
        assert len(error_events) == 1
        assert error_events[0].thread_id == "t-err"
        assert error_events[0].user_id == "u2"

    async def test_session_id_in_call_history(self):
        runner = ToolRunner()
        ctx = {"thread_id": "t1", "session_id": "sess-abc"}
        await runner.run(calculator, {"expression": "5+5"}, context=ctx)
        assert runner.call_history[0]["session_id"] == "sess-abc"


# ──────────────────────────────────────────────────────────────
#  3. CostTracker — per-thread breakdown
# ──────────────────────────────────────────────────────────────


class TestCostTrackerPerThread:
    """CostTracker tracks costs per thread_id alongside global totals."""

    def test_record_with_thread_id(self):
        ct = CostTracker()
        ct.record(CostRecord(total_tokens=100, cost_usd=0.01, thread_id="t1"))
        ct.record(CostRecord(total_tokens=200, cost_usd=0.02, thread_id="t1"))
        ct.record(CostRecord(total_tokens=50, cost_usd=0.005, thread_id="t2"))
        # Global
        assert ct.total_tokens == 350
        assert ct.step_count == 3
        # Per-thread
        t1 = ct.thread_cost("t1")
        assert t1["tokens"] == 300
        assert t1["steps"] == 2
        t2 = ct.thread_cost("t2")
        assert t2["tokens"] == 50
        assert t2["steps"] == 1

    def test_thread_cost_unknown_thread(self):
        ct = CostTracker()
        t = ct.thread_cost("nonexistent")
        assert t["tokens"] == 0
        assert t["steps"] == 0

    def test_to_dict_includes_per_thread(self):
        ct = CostTracker()
        ct.record(CostRecord(total_tokens=10, thread_id="t1"))
        d = ct.to_dict()
        assert "per_thread" in d
        assert "t1" in d["per_thread"]
        assert d["per_thread"]["t1"]["tokens"] == 10

    def test_no_thread_id_still_works(self):
        """Records without thread_id go to global only."""
        ct = CostTracker()
        ct.record(CostRecord(total_tokens=100))
        assert ct.total_tokens == 100
        d = ct.to_dict()
        assert d["per_thread"] == {}

    def test_budget_enforcement_with_threads(self):
        from suluv.core.agent.cost_tracker import BudgetExceeded
        ct = CostTracker(budget=CostBudget(max_tokens=150))
        ct.record(CostRecord(total_tokens=100, thread_id="t1"))
        with pytest.raises(BudgetExceeded):
            ct.record(CostRecord(total_tokens=100, thread_id="t2"))
        # Both threads contributed to global limit
        assert ct.total_tokens == 200


# ──────────────────────────────────────────────────────────────
#  4. GuardrailChain — receives thread context from agent
# ──────────────────────────────────────────────────────────────


class TestGuardrailChainThreadContext:
    """GuardrailChain receives context dict with thread_id when called from agent."""

    async def test_spy_guardrail_receives_thread_context(self):
        spy = _SpyGuardrail()
        chain = GuardrailChain(guardrails=[spy])
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=3),
            llm=MockLLM(responses=[_react_final("ok", "done")]),
            tools=[],
            guardrails=chain,
        )
        ctx = AgentContext(
            thread_id="t-guard", user_id="u1", org_id="o1", session_id="s1",
        )
        await agent.run("test input", context=ctx)

        # Input guardrail should have received our context
        assert len(spy.input_contexts) == 1
        assert spy.input_contexts[0]["thread_id"] == "t-guard"
        assert spy.input_contexts[0]["user_id"] == "u1"
        assert spy.input_contexts[0]["org_id"] == "o1"

        # Output guardrail too
        assert len(spy.output_contexts) == 1
        assert spy.output_contexts[0]["thread_id"] == "t-guard"

    async def test_guardrail_context_without_thread(self):
        """When no thread_id, context values are None — still works."""
        spy = _SpyGuardrail()
        chain = GuardrailChain(guardrails=[spy])
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=3),
            llm=MockLLM(responses=[_react_final("ok", "done")]),
            tools=[],
            guardrails=chain,
        )
        await agent.run("test", context=AgentContext())
        assert spy.input_contexts[0]["thread_id"] is None

    async def test_guardrail_chain_standalone_with_context(self):
        """GuardrailChain.check_input/check_output accept context param."""
        spy = _SpyGuardrail()
        chain = GuardrailChain(guardrails=[spy])
        ctx = {"thread_id": "t1", "extra": "data"}
        result = await chain.check_input("hello", context=ctx)
        assert result.action == GuardrailAction.ALLOW
        assert spy.input_contexts[0]["thread_id"] == "t1"
        assert spy.input_contexts[0]["extra"] == "data"


# ──────────────────────────────────────────────────────────────
#  5. Full Agent Pipeline — end-to-end thread propagation
# ──────────────────────────────────────────────────────────────


class TestAgentFullThreadPipeline:
    """End-to-end: thread_id reaches memory, tools, cost, guardrails, audit."""

    async def test_thread_flows_to_all_subsystems(self):
        """The big integration test: one agent run with thread_id propagates
        to memory, tool_runner audit, cost_tracker, guardrails, and audit."""

        # Setup all subsystems
        stm = InMemoryShortTermMemory()
        ep = InMemoryEpisodicMemory()
        memory = MemoryManager(short_term=stm, episodic=ep)
        audit = InMemoryAuditBackend()
        spy_guard = _SpyGuardrail()
        guardrails = GuardrailChain(guardrails=[spy_guard])
        cost_tracker = CostTracker()
        cp = InMemoryCheckpointer()

        llm = MockLLM(responses=[
            _react_tool_call("calc", "calculator", {"expression": "2+2"}),
            _react_final("got result", "The answer is 4"),
        ])

        agent = SuluvAgent(
            role=AgentRole(name="FullBot", max_steps=5),
            llm=llm,
            tools=[calculator],
            memory=memory,
            guardrails=guardrails,
            audit_backend=audit,
            cost_tracker=cost_tracker,
            checkpointer=cp,
        )

        ctx = AgentContext(
            thread_id="t-full",
            user_id="u1",
            org_id="o1",
            session_id="s1",
        )
        result = await agent.run("What is 2+2?", context=ctx)
        assert result.success

        # ── Memory: short-term scoped by thread ──
        all_stm = await stm.all()
        assert any(k.startswith("t:t-full:") for k in all_stm)

        # ── Memory: episodic has thread_id in metadata ──
        assert len(ep._episodes) == 1
        assert ep._episodes[0]["metadata"]["thread_id"] == "t-full"

        # ── Tool audit: includes thread context ──
        tool_events = [e for e in audit.events if e.event_type == "tool_call"]
        assert len(tool_events) >= 1
        assert tool_events[0].thread_id == "t-full"
        assert tool_events[0].user_id == "u1"
        assert tool_events[0].session_id == "s1"

        # ── Cost tracker: per-thread breakdown ──
        t_cost = cost_tracker.thread_cost("t-full")
        assert t_cost["tokens"] > 0
        assert t_cost["steps"] >= 2  # 2 LLM calls

        # ── Guardrails: received thread context ──
        assert spy_guard.input_contexts[0]["thread_id"] == "t-full"
        assert spy_guard.output_contexts[0]["thread_id"] == "t-full"

        # ── Main audit event: has thread_id ──
        agent_events = [e for e in audit.events if e.event_type == "agent_run"]
        assert len(agent_events) == 1
        assert agent_events[0].thread_id == "t-full"
        assert agent_events[0].user_id == "u1"

        # ── Checkpoint: thread saved ──
        thread = await cp.get("t-full")
        assert thread is not None
        assert thread.checkpoint_count == 1

    async def test_no_thread_id_backward_compat(self):
        """Agent works fine without thread_id — no regressions."""
        stm = InMemoryShortTermMemory()
        memory = MemoryManager(short_term=stm)
        audit = InMemoryAuditBackend()
        cost_tracker = CostTracker()

        agent = SuluvAgent(
            role=AgentRole(name="NoThread", max_steps=3),
            llm=MockLLM(responses=[_react_final("ok", "done")]),
            tools=[],
            memory=memory,
            audit_backend=audit,
            cost_tracker=cost_tracker,
        )
        # Run with session_id but no thread_id
        ctx = AgentContext(session_id="s1", user_id="u1")
        result = await agent.run("hello", context=ctx)
        assert result.success

        # Memory saved under session scope
        all_stm = await stm.all()
        assert len(all_stm) == 1
        assert any(k.startswith("t:s1:") for k in all_stm)

        # Cost tracked globally (no per-thread entry)
        assert cost_tracker.total_tokens > 0
        d = cost_tracker.to_dict()
        assert d["per_thread"] == {}

    async def test_multi_thread_isolation_in_memory(self):
        """Two threads sharing the same agent get isolated memory."""
        stm = InMemoryShortTermMemory()
        memory = MemoryManager(short_term=stm)
        cost_tracker = CostTracker()

        llm = MockLLM(responses=[
            _react_final("t1", "Thread 1"),
            _react_final("t2", "Thread 2"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="MultiBot", max_steps=3),
            llm=llm,
            tools=[],
            memory=memory,
            cost_tracker=cost_tracker,
        )

        await agent.run("q1", context=AgentContext(thread_id="t1", session_id="s1"))
        await agent.run("q2", context=AgentContext(thread_id="t2", session_id="s1"))

        # Short-term should have entries scoped by each thread
        all_stm = await stm.all()
        t1_keys = [k for k in all_stm if k.startswith("t:t1:")]
        t2_keys = [k for k in all_stm if k.startswith("t:t2:")]
        assert len(t1_keys) == 1
        assert len(t2_keys) == 1

        # Cost tracker shows both threads
        assert cost_tracker.thread_cost("t1")["tokens"] > 0
        assert cost_tracker.thread_cost("t2")["tokens"] > 0

    async def test_tool_runner_context_from_agent(self):
        """Tool calls made by the agent include thread context in runner history."""
        audit = InMemoryAuditBackend()
        llm = MockLLM(responses=[
            _react_tool_call("calc", "calculator", {"expression": "7*8"}),
            _react_final("done", "56"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="ToolBot", max_steps=5),
            llm=llm,
            tools=[calculator],
            audit_backend=audit,
        )
        ctx = AgentContext(thread_id="t-tool", user_id="u2", org_id="o2")
        await agent.run("calc 7*8", context=ctx)

        # Check the internal tool_runner's history
        history = agent._tool_runner.call_history
        assert len(history) == 1
        assert history[0]["thread_id"] == "t-tool"
        assert history[0]["user_id"] == "u2"
        assert history[0]["org_id"] == "o2"

    async def test_cost_record_has_thread_id(self):
        """CostRecord objects created during agent run carry thread_id."""
        cost_tracker = CostTracker()
        agent = SuluvAgent(
            role=AgentRole(name="CostBot", max_steps=3),
            llm=MockLLM(responses=[_react_final("ok", "done")]),
            tools=[],
            cost_tracker=cost_tracker,
        )
        await agent.run("q", context=AgentContext(thread_id="t-cost"))
        # Inspect the internal records
        assert len(cost_tracker._records) == 1
        assert cost_tracker._records[0].thread_id == "t-cost"
