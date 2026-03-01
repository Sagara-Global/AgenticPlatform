"""Phase 3 tests — Agent System: agents, tools, policy, compliance."""

import json
import pytest
from typing import Any
from datetime import datetime, timezone

from suluv.core.agent.role import AgentRole
from suluv.core.agent.context import AgentContext
from suluv.core.agent.result import AgentResult, StepRecord
from suluv.core.agent.memory_manager import MemoryManager
from suluv.core.agent.guardrail_chain import GuardrailChain
from suluv.core.agent.cost_tracker import CostTracker, CostBudget, BudgetExceeded
from suluv.core.agent.corpus_registry import CorpusRegistry
from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.agent_node import AgentNode
from suluv.core.tools.decorator import SuluvTool, suluv_tool
from suluv.core.tools.runner import ToolRunner
from suluv.core.policy.engine import PolicyEngine
from suluv.core.compliance.audit_hooks import AuditHooks
from suluv.core.compliance.consent_enforcer import ConsentEnforcer, ConsentRequired
from suluv.core.types import CostRecord, AuditEvent
from suluv.core.ports.guardrail import Guardrail, GuardrailAction, GuardrailResult
from suluv.core.ports.policy_rule import PolicyRule, PolicyDecision, PolicyResult
from suluv.core.ports.consent_provider import ConsentProvider, ConsentResult
from suluv.core.ports.corpus_provider import CorpusProvider, Chunk
from suluv.core.adapters import (
    MockLLM,
    InMemoryAuditBackend,
    InMemoryShortTermMemory,
    InMemoryLongTermMemory,
    InMemoryEpisodicMemory,
    InMemorySemanticMemory,
)
from suluv.core.engine.node import NodeInput, NodeOutput


# ── Test Helpers ──────────────────────────────────────────────────────────────


class BlockingGuardrail(Guardrail):
    """Blocks any input containing the word 'banned'."""

    async def check_input(self, context: dict, text: str) -> GuardrailResult:
        if "banned" in text.lower():
            return GuardrailResult(action=GuardrailAction.BLOCK, message="Contains banned word")
        return GuardrailResult(action=GuardrailAction.ALLOW, message="OK")

    async def check_output(self, context: dict, text: str) -> GuardrailResult:
        if "secret" in text.lower():
            return GuardrailResult(action=GuardrailAction.BLOCK, message="Output contains secret")
        return GuardrailResult(action=GuardrailAction.ALLOW, message="OK")


class DenyRule(PolicyRule):
    """Always denies."""

    async def evaluate(self, action: str, context: dict[str, Any]) -> PolicyResult:
        return PolicyResult(decision=PolicyDecision.DENY, reason="Always denied")


class AllowRule(PolicyRule):
    """Always allows."""

    async def evaluate(self, action: str, context: dict[str, Any]) -> PolicyResult:
        return PolicyResult(decision=PolicyDecision.ALLOW, reason="Allowed")


class EscalateRule(PolicyRule):
    """Always escalates."""

    async def evaluate(self, action: str, context: dict[str, Any]) -> PolicyResult:
        return PolicyResult(decision=PolicyDecision.ESCALATE, reason="Needs review")


class AlwaysGrantConsent(ConsentProvider):
    async def check(self, context: dict, purpose: str) -> ConsentResult:
        return ConsentResult(granted=True, purpose=purpose, reason="Auto-granted")


class AlwaysDenyConsent(ConsentProvider):
    async def check(self, context: dict, purpose: str) -> ConsentResult:
        return ConsentResult(granted=False, purpose=purpose, reason="User declined")


class DummyCorpus(CorpusProvider):
    def __init__(self, docs: list[str] | None = None):
        self._docs = docs or ["sample document"]

    async def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        return [
            Chunk(text=doc, score=1.0 - i * 0.1, metadata={"source": f"doc-{i}"})
            for i, doc in enumerate(self._docs[:top_k])
        ]


# ── AgentRole tests ───────────────────────────────────────────────────────────


class TestAgentRole:
    def test_default_values(self):
        role = AgentRole(name="Test")
        assert role.name == "Test"
        assert role.max_steps == 25
        assert role.temperature == 0.1

    def test_system_prompt_basic(self):
        role = AgentRole(name="Assistant")
        prompt = role.to_system_prompt()
        assert "You are Assistant" in prompt

    def test_system_prompt_with_capabilities(self):
        role = AgentRole(
            name="Helper",
            capabilities=["search", "calculate"],
            description="A helpful bot",
        )
        prompt = role.to_system_prompt()
        assert "search" in prompt
        assert "calculate" in prompt
        assert "A helpful bot" in prompt

    def test_system_prompt_with_format(self):
        role = AgentRole(name="Bot", output_format="json")
        prompt = role.to_system_prompt()
        assert "json" in prompt

    def test_system_prompt_with_instructions(self):
        role = AgentRole(name="Bot", instructions="Be concise.")
        prompt = role.to_system_prompt()
        assert "Be concise" in prompt


# ── AgentContext tests ────────────────────────────────────────────────────────


class TestAgentContext:
    def test_defaults(self):
        ctx = AgentContext()
        assert ctx.org_id is None
        assert ctx.locale == "en"
        assert ctx.timezone == "UTC"
        assert ctx.variables == {}

    def test_get_set(self):
        ctx = AgentContext()
        ctx.set("key1", "val1")
        assert ctx.get("key1") == "val1"
        assert ctx.get("missing", "default") == "default"

    def test_to_dict(self):
        ctx = AgentContext(org_id="org1", user_id="u1")
        d = ctx.to_dict()
        assert d["org_id"] == "org1"
        assert d["user_id"] == "u1"
        assert "variables" in d


# ── AgentResult tests ─────────────────────────────────────────────────────────


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult()
        assert result.success is True
        assert result.answer == ""
        assert result.step_count == 0

    def test_with_steps(self):
        result = AgentResult()
        result.steps.append(StepRecord(step=1, thought="thinking"))
        result.steps.append(StepRecord(step=2, action="search", observation="found it"))
        assert result.step_count == 2

    def test_to_dict(self):
        result = AgentResult(answer="42", total_tokens=100, cost_usd=0.01)
        d = result.to_dict()
        assert d["answer"] == "42"
        assert d["total_tokens"] == 100
        assert d["success"] is True


# ── MemoryManager tests ──────────────────────────────────────────────────────


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_empty_manager(self):
        mm = MemoryManager()
        ctx = await mm.load_context(session_id="s1", user_id="u1")
        assert ctx == {}

    @pytest.mark.asyncio
    async def test_short_term_roundtrip(self):
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        await mm.save_interaction(session_id="s1", content="hello world")
        ctx = await mm.load_context(session_id="s1")
        assert "short_term" in ctx
        assert len(ctx["short_term"]) >= 1

    @pytest.mark.asyncio
    async def test_long_term_roundtrip(self):
        ltm = InMemoryLongTermMemory()
        mm = MemoryManager(long_term=ltm)
        await mm.save_interaction(user_id="u1", content="important fact")
        ctx = await mm.load_context(user_id="u1")
        assert "long_term" in ctx
        assert len(ctx["long_term"]) >= 1

    @pytest.mark.asyncio
    async def test_episodic_roundtrip(self):
        ep = InMemoryEpisodicMemory()
        mm = MemoryManager(episodic=ep)
        await mm.save_interaction(content="debug session info")
        ctx = await mm.load_context(query="debug")
        assert "episodic" in ctx

    @pytest.mark.asyncio
    async def test_clear_session(self):
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        await mm.save_interaction(session_id="s1", content="temp data")
        await mm.clear_session()
        ctx = await mm.load_context(session_id="s1")
        # After clear, should be empty
        assert ctx.get("short_term", {}) == {}

    @pytest.mark.asyncio
    async def test_all_tiers(self):
        mm = MemoryManager(
            short_term=InMemoryShortTermMemory(),
            long_term=InMemoryLongTermMemory(),
            episodic=InMemoryEpisodicMemory(),
            semantic=InMemorySemanticMemory(),
        )
        await mm.save_interaction(
            session_id="s1", user_id="u1", content="multi-tier test"
        )
        ctx = await mm.load_context(
            session_id="s1", user_id="u1", query="multi"
        )
        assert "short_term" in ctx


# ── GuardrailChain tests ─────────────────────────────────────────────────────


class TestGuardrailChain:
    @pytest.mark.asyncio
    async def test_empty_chain_allows(self):
        chain = GuardrailChain()
        result = await chain.check_input("hello")
        assert result.action == GuardrailAction.ALLOW

    @pytest.mark.asyncio
    async def test_blocking_guardrail(self):
        chain = GuardrailChain(guardrails=[BlockingGuardrail()])
        result = await chain.check_input("this has banned content")
        assert result.action == GuardrailAction.BLOCK
        assert "banned" in result.message.lower()

    @pytest.mark.asyncio
    async def test_allow_passes_through(self):
        chain = GuardrailChain(guardrails=[BlockingGuardrail()])
        result = await chain.check_input("this is fine")
        assert result.action == GuardrailAction.ALLOW

    @pytest.mark.asyncio
    async def test_output_check(self):
        chain = GuardrailChain(guardrails=[BlockingGuardrail()])
        result = await chain.check_output("the secret is 42")
        assert result.action == GuardrailAction.BLOCK

    @pytest.mark.asyncio
    async def test_add_guardrail(self):
        chain = GuardrailChain()
        chain.add(BlockingGuardrail())
        result = await chain.check_input("banned word here")
        assert result.action == GuardrailAction.BLOCK


# ── CostTracker tests ────────────────────────────────────────────────────────


class TestCostTracker:
    def test_initial_values(self):
        tracker = CostTracker()
        assert tracker.total_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.step_count == 0

    def test_record_accumulates(self):
        tracker = CostTracker()
        tracker.record(CostRecord(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01, model="test"))
        tracker.record(CostRecord(input_tokens=20, output_tokens=10, total_tokens=30, cost_usd=0.02, model="test"))
        assert tracker.total_tokens == 45
        assert tracker.total_cost_usd == pytest.approx(0.03)
        assert tracker.step_count == 2

    def test_budget_token_limit(self):
        budget = CostBudget(max_tokens=100)
        tracker = CostTracker(budget=budget)
        tracker.record(CostRecord(input_tokens=50, output_tokens=30, total_tokens=80, cost_usd=0.01, model="test"))
        with pytest.raises(BudgetExceeded, match="Token budget"):
            tracker.record(CostRecord(input_tokens=15, output_tokens=10, total_tokens=25, cost_usd=0.01, model="test"))

    def test_budget_cost_limit(self):
        budget = CostBudget(max_cost_usd=0.05)
        tracker = CostTracker(budget=budget)
        tracker.record(CostRecord(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.04, model="test"))
        with pytest.raises(BudgetExceeded, match="Cost budget"):
            tracker.record(CostRecord(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.02, model="test"))

    def test_budget_step_limit(self):
        budget = CostBudget(max_steps=2)
        tracker = CostTracker(budget=budget)
        tracker.record(CostRecord(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01, model="test"))
        tracker.record(CostRecord(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01, model="test"))
        with pytest.raises(BudgetExceeded, match="Step budget"):
            tracker.record(CostRecord(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01, model="test"))

    def test_to_dict(self):
        tracker = CostTracker()
        tracker.record(CostRecord(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01, model="test"))
        d = tracker.to_dict()
        assert d["total_tokens"] == 15
        assert d["records"] == 1


# ── CorpusRegistry tests ─────────────────────────────────────────────────────


class TestCorpusRegistry:
    @pytest.mark.asyncio
    async def test_empty_registry(self):
        reg = CorpusRegistry()
        results = await reg.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_single_provider(self):
        reg = CorpusRegistry()
        reg.register("docs", DummyCorpus(["doc1", "doc2"]))
        results = await reg.search("test")
        assert len(results) == 2
        assert results[0].text == "doc1"

    @pytest.mark.asyncio
    async def test_multiple_providers_merged(self):
        reg = CorpusRegistry()
        reg.register("docs", DummyCorpus(["doc1"]))
        reg.register("wiki", DummyCorpus(["wiki1"]))
        results = await reg.search("test", top_k=5)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_unregister(self):
        reg = CorpusRegistry()
        reg.register("docs", DummyCorpus(["doc1"]))
        reg.unregister("docs")
        results = await reg.search("test")
        assert results == []

    def test_provider_names(self):
        reg = CorpusRegistry()
        reg.register("a", DummyCorpus())
        reg.register("b", DummyCorpus())
        assert sorted(reg.provider_names) == ["a", "b"]


# ── SuluvTool and @suluv_tool decorator tests ────────────────────────────────


class TestSuluvTool:
    def test_sync_tool(self):
        def add(a: int, b: int) -> int:
            return a + b
        tool = SuluvTool(fn=add, name="add", description="Add two numbers")
        assert tool.name == "add"
        assert tool.description == "Add two numbers"
        assert "a" in tool.parameters["properties"]
        assert "b" in tool.parameters["properties"]
        assert "a" in tool.parameters["required"]

    @pytest.mark.asyncio
    async def test_sync_execution(self):
        def multiply(x: int, y: int) -> int:
            return x * y
        tool = SuluvTool(fn=multiply)
        result = await tool.execute(x=3, y=4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_async_execution(self):
        async def greet(name: str) -> str:
            return f"Hello {name}"
        tool = SuluvTool(fn=greet)
        result = await tool.execute(name="World")
        assert result == "Hello World"

    def test_infer_parameters(self):
        def fn(name: str, count: int = 5, active: bool = True) -> str:
            return ""
        tool = SuluvTool(fn=fn)
        params = tool.parameters
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["active"]["type"] == "boolean"
        assert "name" in params["required"]
        assert "count" not in params["required"]

    def test_to_dict(self):
        tool = SuluvTool(fn=lambda x: x, name="echo")
        d = tool.to_dict()
        assert d["name"] == "echo"

    @pytest.mark.asyncio
    async def test_timeout(self):
        import asyncio

        async def slow():
            await asyncio.sleep(10)

        tool = SuluvTool(fn=slow, timeout=0.01)
        with pytest.raises(asyncio.TimeoutError):
            await tool.execute()


class TestSuluvToolDecorator:
    def test_decorator_creates_tool(self):
        @suluv_tool(name="search", description="Search the web")
        async def search(query: str) -> str:
            return f"results for {query}"

        assert isinstance(search, SuluvTool)
        assert search.name == "search"
        assert search.description == "Search the web"

    @pytest.mark.asyncio
    async def test_decorated_tool_executes(self):
        @suluv_tool(name="calc")
        def calc(expression: str) -> str:
            return str(eval(expression))

        result = await calc.execute(expression="2+3")
        assert result == "5"

    def test_decorator_with_timeout(self):
        @suluv_tool(timeout=5.0)
        def fast_fn() -> str:
            return "done"
        assert fast_fn.timeout == 5.0


# ── ToolRunner tests ──────────────────────────────────────────────────────────


class TestToolRunner:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        def add(a: int, b: int) -> int:
            return a + b
        tool = SuluvTool(fn=add, name="add")
        runner = ToolRunner()
        result = await runner.run(tool, {"a": 1, "b": 2})
        assert result["result"] == 3
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_failing_tool(self):
        def fail():
            raise ValueError("oops")
        tool = SuluvTool(fn=fail, name="fail")
        runner = ToolRunner()
        result = await runner.run(tool, {})
        assert result["result"] is None
        assert "oops" in result["error"]

    @pytest.mark.asyncio
    async def test_timeout_tool(self):
        import asyncio

        async def slow():
            await asyncio.sleep(10)

        tool = SuluvTool(fn=slow, name="slow", timeout=0.01)
        runner = ToolRunner(default_timeout=0.01)
        result = await runner.run(tool, {})
        assert result["result"] is None
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_call_history(self):
        tool = SuluvTool(fn=lambda: 42, name="answer")
        runner = ToolRunner()
        await runner.run(tool, {})
        await runner.run(tool, {})
        assert len(runner.call_history) == 2

    @pytest.mark.asyncio
    async def test_audit_integration(self):
        audit = InMemoryAuditBackend()
        tool = SuluvTool(fn=lambda: "ok", name="test-tool")
        runner = ToolRunner(audit_backend=audit)
        await runner.run(tool, {})
        events = await audit.query(event_type="tool_call")
        assert len(events) >= 1


# ── PolicyEngine tests ────────────────────────────────────────────────────────


class TestPolicyEngine:
    @pytest.mark.asyncio
    async def test_empty_engine_allows(self):
        engine = PolicyEngine()
        result = await engine.evaluate("action", {})
        assert result.decision == PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_deny_fast_fails(self):
        engine = PolicyEngine(rules=[AllowRule(), DenyRule(), AllowRule()])
        result = await engine.evaluate("action", {})
        assert result.decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_all_allow(self):
        engine = PolicyEngine(rules=[AllowRule(), AllowRule()])
        result = await engine.evaluate("action", {})
        assert result.decision == PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_escalate_without_deny(self):
        engine = PolicyEngine(rules=[AllowRule(), EscalateRule()])
        result = await engine.evaluate("action", {})
        assert result.decision == PolicyDecision.ESCALATE

    @pytest.mark.asyncio
    async def test_deny_overrides_escalate(self):
        engine = PolicyEngine(rules=[DenyRule(), EscalateRule()])
        result = await engine.evaluate("action", {})
        assert result.decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_add_rule(self):
        engine = PolicyEngine()
        engine.add_rule(DenyRule())
        result = await engine.evaluate("action", {})
        assert result.decision == PolicyDecision.DENY


# ── AuditHooks tests ─────────────────────────────────────────────────────────


class TestAuditHooks:
    @pytest.mark.asyncio
    async def test_log_agent_start(self):
        backend = InMemoryAuditBackend()
        hooks = AuditHooks(backend)
        await hooks.log_agent_start("Bot", "analyze data", org_id="org1", user_id="u1")
        events = await backend.query(event_type="agent_start")
        assert len(events) == 1
        assert events[0].data["agent"] == "Bot"

    @pytest.mark.asyncio
    async def test_log_agent_end(self):
        backend = InMemoryAuditBackend()
        hooks = AuditHooks(backend)
        await hooks.log_agent_end("Bot", success=True, steps=3, tokens=100)
        events = await backend.query(event_type="agent_end")
        assert len(events) == 1
        assert events[0].data["success"] is True

    @pytest.mark.asyncio
    async def test_log_tool_call(self):
        backend = InMemoryAuditBackend()
        hooks = AuditHooks(backend)
        await hooks.log_tool_call("search", {"query": "test"}, result="found")
        events = await backend.query(event_type="tool_call")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_log_guardrail_block(self):
        backend = InMemoryAuditBackend()
        hooks = AuditHooks(backend)
        await hooks.log_guardrail_block("bad content", "input", "Profanity detected")
        events = await backend.query(event_type="guardrail_block")
        assert len(events) == 1
        assert events[0].data["direction"] == "input"

    @pytest.mark.asyncio
    async def test_log_consent_check(self):
        backend = InMemoryAuditBackend()
        hooks = AuditHooks(backend)
        await hooks.log_consent_check("analytics", granted=True, user_id="u1")
        events = await backend.query(event_type="consent_check")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_log_custom(self):
        backend = InMemoryAuditBackend()
        hooks = AuditHooks(backend)
        await hooks.log_custom("my_event", {"key": "value"}, org_id="org1")
        events = await backend.query(event_type="my_event")
        assert len(events) == 1


# ── ConsentEnforcer tests ────────────────────────────────────────────────────


class TestConsentEnforcer:
    @pytest.mark.asyncio
    async def test_consent_granted(self):
        enforcer = ConsentEnforcer(AlwaysGrantConsent())
        result = await enforcer.require_consent(user_id="u1", purpose="analytics")
        assert result.granted is True

    @pytest.mark.asyncio
    async def test_consent_denied_raises(self):
        enforcer = ConsentEnforcer(AlwaysDenyConsent())
        with pytest.raises(ConsentRequired, match="Consent required"):
            await enforcer.require_consent(user_id="u1", purpose="marketing")

    @pytest.mark.asyncio
    async def test_soft_check_granted(self):
        enforcer = ConsentEnforcer(AlwaysGrantConsent())
        assert await enforcer.check_consent(user_id="u1", purpose="analytics") is True

    @pytest.mark.asyncio
    async def test_soft_check_denied(self):
        enforcer = ConsentEnforcer(AlwaysDenyConsent())
        assert await enforcer.check_consent(user_id="u1", purpose="marketing") is False

    @pytest.mark.asyncio
    async def test_data_categories_passed(self):
        class InspectingProvider(ConsentProvider):
            last_context = None
            async def check(self, context: dict, purpose: str) -> ConsentResult:
                InspectingProvider.last_context = context
                return ConsentResult(granted=True, purpose=purpose)

        provider = InspectingProvider()
        enforcer = ConsentEnforcer(provider)
        await enforcer.require_consent(
            user_id="u1", purpose="processing",
            data_categories=["email", "location"],
        )
        assert provider.last_context["data_categories"] == ["email", "location"]


# ── SuluvAgent integration tests ─────────────────────────────────────────────


class TestSuluvAgent:
    @pytest.mark.asyncio
    async def test_simple_run(self):
        """Agent with no tools should get a direct answer from the LLM."""
        llm = MockLLM(responses=[
            json.dumps({"thought": "Simple question", "final_answer": "42"})
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Test"),
            llm=llm,
        )
        result = await agent.run("What is the answer?")
        assert result.success
        assert result.answer == "42"
        assert result.step_count == 1

    @pytest.mark.asyncio
    async def test_tool_call_flow(self):
        """Agent calls a tool then returns final answer."""
        @suluv_tool(name="calc")
        def calc(expression: str) -> str:
            """Evaluate math."""
            return str(eval(expression))

        llm = MockLLM(responses=[
            json.dumps({
                "thought": "I need to calculate",
                "action": "calc",
                "action_input": {"expression": "2+2"},
            }),
            json.dumps({
                "thought": "Got the result",
                "final_answer": "4",
            }),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Calc", max_steps=5),
            llm=llm,
            tools=[calc],
        )
        result = await agent.run("What is 2+2?")
        assert result.success
        assert result.answer == "4"
        assert result.step_count == 2
        assert result.steps[0].action == "calc"
        assert result.steps[0].observation is not None

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Agent tries unknown tool, gets error, then answers."""
        llm = MockLLM(responses=[
            json.dumps({
                "thought": "Let me use a tool",
                "action": "nonexistent",
                "action_input": {},
            }),
            json.dumps({
                "thought": "Tool not found, I'll answer directly",
                "final_answer": "I tried but failed",
            }),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Test", max_steps=5),
            llm=llm,
        )
        result = await agent.run("Do something")
        assert result.success
        assert "nonexistent" in result.steps[0].error

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self):
        """Agent reaches max steps without final_answer."""
        llm = MockLLM(responses=[
            json.dumps({"thought": "thinking", "action": "search", "action_input": {"q": "x"}})
        ] * 10)

        @suluv_tool(name="search")
        def search(q: str) -> str:
            return "no result"

        agent = SuluvAgent(
            role=AgentRole(name="Stuck", max_steps=3),
            llm=llm,
            tools=[search],
        )
        result = await agent.run("Find something")
        assert not result.success
        assert "Max steps" in result.error

    @pytest.mark.asyncio
    async def test_input_guardrail_block(self):
        """Input is blocked by guardrail."""
        llm = MockLLM()
        agent = SuluvAgent(
            role=AgentRole(name="Test"),
            llm=llm,
            guardrails=GuardrailChain(guardrails=[BlockingGuardrail()]),
        )
        result = await agent.run("This has banned content")
        assert not result.success
        assert "blocked" in result.error.lower()
        # LLM should not have been called
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_output_guardrail_block(self):
        """Output is blocked by guardrail."""
        llm = MockLLM(responses=[
            json.dumps({"thought": "Done", "final_answer": "The secret is 42"})
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Test"),
            llm=llm,
            guardrails=GuardrailChain(guardrails=[BlockingGuardrail()]),
        )
        result = await agent.run("Tell me something")
        assert not result.success
        assert "[output blocked" in result.answer

    @pytest.mark.asyncio
    async def test_non_json_response(self):
        """LLM returns plain text (not JSON) — treated as final answer."""
        llm = MockLLM(responses=["Just a plain text response"])
        agent = SuluvAgent(
            role=AgentRole(name="Test"),
            llm=llm,
        )
        result = await agent.run("Say hello")
        assert result.success
        assert result.answer == "Just a plain text response"

    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Agent tracks token costs."""
        llm = MockLLM(responses=[
            json.dumps({"final_answer": "done"})
        ])
        tracker = CostTracker()
        agent = SuluvAgent(
            role=AgentRole(name="Test"),
            llm=llm,
            cost_tracker=tracker,
        )
        result = await agent.run("Quick task")
        assert tracker.total_tokens > 0
        assert tracker.step_count == 1

    @pytest.mark.asyncio
    async def test_audit_logging(self):
        """Agent logs run to audit backend."""
        audit = InMemoryAuditBackend()
        llm = MockLLM(responses=[
            json.dumps({"final_answer": "done"})
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Audited"),
            llm=llm,
            audit_backend=audit,
        )
        ctx = AgentContext(org_id="org1", user_id="u1", session_id="s1")
        result = await agent.run("Task", context=ctx)
        events = await audit.query(event_type="agent_run")
        assert len(events) == 1
        assert events[0].data["role"] == "Audited"

    @pytest.mark.asyncio
    async def test_memory_integration(self):
        """Agent loads and saves memory."""
        mm = MemoryManager(
            short_term=InMemoryShortTermMemory(),
            episodic=InMemoryEpisodicMemory(),
        )
        llm = MockLLM(responses=[
            json.dumps({"final_answer": "I remember"})
        ])
        agent = SuluvAgent(
            role=AgentRole(name="MemBot"),
            llm=llm,
            memory=mm,
        )
        ctx = AgentContext(session_id="s1", user_id="u1")
        result = await agent.run("Remember this", context=ctx)
        assert result.success

        # Memory should have been saved
        loaded = await mm.load_context(session_id="s1")
        assert "short_term" in loaded

    @pytest.mark.asyncio
    async def test_streaming(self):
        """Agent streaming yields events."""
        llm = MockLLM(responses=[
            json.dumps({"thought": "thinking", "final_answer": "42"})
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Test"),
            llm=llm,
        )
        events = []
        async for event in agent.run_stream("question"):
            events.append(event)
        assert any(e["type"] == "answer" for e in events)


# ── AgentNode tests ───────────────────────────────────────────────────────────


class TestAgentNode:
    @pytest.mark.asyncio
    async def test_agent_node_execution(self):
        llm = MockLLM(responses=[
            json.dumps({"final_answer": "computed"})
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Worker"),
            llm=llm,
        )
        node = AgentNode(agent=agent, node_id="agent1")
        out = await node.execute(NodeInput(
            data="Process this",
            context={"org_id": "org1", "user_id": "u1"},
        ))
        assert out.success
        assert out.data["answer"] == "computed"
        assert "tokens" in out.metadata

    @pytest.mark.asyncio
    async def test_agent_node_with_dict_input(self):
        llm = MockLLM(responses=[
            json.dumps({"final_answer": "processed"})
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Worker"),
            llm=llm,
        )
        node = AgentNode(agent=agent, task_key="prompt", node_id="agent2")
        out = await node.execute(NodeInput(
            data={"prompt": "Analyze data", "extra": 42},
            context={},
        ))
        assert out.success
        assert out.data["answer"] == "processed"

    @pytest.mark.asyncio
    async def test_agent_node_type(self):
        from suluv.core.types import NodeType
        llm = MockLLM()
        agent = SuluvAgent(role=AgentRole(name="Test"), llm=llm)
        node = AgentNode(agent=agent, node_id="a1")
        assert node.node_type == NodeType.AGENT
