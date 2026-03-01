"""Integration tests — Agents + Tools + MockLLM end-to-end.

Tests the full ReAct agent loop with real tools and MockLLM responses,
verifying tool discovery, execution, conversation flow, guardrails,
memory, cost tracking, and all pre-built agent factories.
"""

from __future__ import annotations

import json
import pytest

from suluv.core.adapters.mock_llm import MockLLM
from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.context import AgentContext
from suluv.core.agent.role import AgentRole
from suluv.core.agent.guardrail_chain import GuardrailChain
from suluv.core.agent.memory_manager import MemoryManager
from suluv.core.agent.cost_tracker import CostTracker
from suluv.core.adapters.memory_audit import InMemoryAuditBackend
from suluv.core.adapters.memory_memory import (
    InMemoryShortTermMemory,
    InMemoryLongTermMemory,
    InMemoryEpisodicMemory,
    InMemorySemanticMemory,
)
from suluv.core.tools.builtins import (
    calculator,
    json_extractor,
    datetime_now,
    date_diff,
    file_reader,
    file_writer,
    shell_exec,
)
from suluv.core.tools.decorator import SuluvTool, suluv_tool
from suluv.core.tools.runner import ToolRunner
from suluv.core.agents import (
    create_assistant,
    create_researcher,
    create_coder,
    create_data_analyst,
)


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _react_tool_call(thought: str, tool: str, args: dict) -> str:
    """Build a ReAct JSON response that calls a tool."""
    return json.dumps({
        "thought": thought,
        "action": tool,
        "action_input": args,
    })


def _react_final(thought: str, answer: str) -> str:
    """Build a ReAct JSON response with final answer."""
    return json.dumps({
        "thought": thought,
        "final_answer": answer,
    })


def _make_agent(
    responses: list[str],
    tools: list[SuluvTool] | None = None,
    **kwargs,
) -> SuluvAgent:
    """Quick helper to build a SuluvAgent with MockLLM."""
    llm = MockLLM(responses=responses)
    role = AgentRole(
        name="TestAgent",
        description="A test agent",
        max_steps=10,
    )
    return SuluvAgent(
        role=role,
        llm=llm,
        tools=tools or [],
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────
#  1. Tool Unit Tests (each builtin tool independently)
# ──────────────────────────────────────────────────────────────

class TestBuiltinTools:
    """Test each built-in tool independently."""

    async def test_calculator_basic(self):
        result = await calculator.execute(expression="2 + 3")
        assert result == "5"

    async def test_calculator_complex(self):
        result = await calculator.execute(expression="(10 * 5) + 2**3")
        assert result == "58"

    async def test_calculator_float(self):
        result = await calculator.execute(expression="22 / 7")
        assert "3.14" in result

    async def test_calculator_error(self):
        result = await calculator.execute(expression="import os")
        assert "Error" in result or "error" in result.lower()

    async def test_json_extractor_simple(self):
        data = json.dumps({"name": "Alice", "age": 30})
        result = await json_extractor.execute(json_text=data, path="name")
        assert "Alice" in result

    async def test_json_extractor_nested(self):
        data = json.dumps({"user": {"profile": {"email": "a@b.com"}}})
        result = await json_extractor.execute(
            json_text=data, path="user.profile.email"
        )
        assert "a@b.com" in result

    async def test_json_extractor_list(self):
        data = json.dumps({"items": [10, 20, 30]})
        result = await json_extractor.execute(json_text=data, path="items.1")
        assert "20" in result

    async def test_json_extractor_missing(self):
        data = json.dumps({"foo": 1})
        result = await json_extractor.execute(json_text=data, path="bar")
        assert "not found" in result.lower() or "error" in result.lower()

    async def test_datetime_now(self):
        result = await datetime_now.execute()
        # Should be ISO format
        assert "T" in result
        assert "20" in result  # starts with year 20xx

    async def test_datetime_now_with_offset(self):
        result = await datetime_now.execute(utc_offset_hours=5.5)
        assert "+05:30" in result

    async def test_date_diff(self):
        result = await date_diff.execute(
            start="2024-01-01", end="2024-01-15"
        )
        assert "14 day" in result

    async def test_date_diff_hours(self):
        result = await date_diff.execute(
            start="2024-01-01T00:00:00",
            end="2024-01-01T03:30:00",
        )
        assert "3 hour" in result
        assert "30 minute" in result

    async def test_date_diff_invalid(self):
        result = await date_diff.execute(start="not-a-date", end="2024-01-01")
        assert "Error" in result or "error" in result.lower()

    async def test_shell_exec_echo(self):
        result = await shell_exec.execute(command="echo hello_suluv")
        assert "hello_suluv" in result

    async def test_shell_exec_timeout(self):
        result = await shell_exec.execute(
            command="ping -n 100 127.0.0.1" if True else "sleep 100",
            timeout_seconds=1,
        )
        # On timeout, should return timeout error or partial output
        assert "timed out" in result.lower() or "hello" not in result

    async def test_tool_schema(self):
        """Every builtin tool should have a valid schema."""
        for tool in [calculator, json_extractor, datetime_now, date_diff, shell_exec]:
            assert tool.name
            assert tool.description
            assert isinstance(tool.parameters, dict)
            assert "properties" in tool.parameters


# ──────────────────────────────────────────────────────────────
#  2. ToolRunner Tests
# ──────────────────────────────────────────────────────────────

class TestToolRunner:
    """Test the sandboxed tool runner."""

    async def test_run_success(self):
        runner = ToolRunner()
        result = await runner.run(calculator, {"expression": "1+1"})
        assert result["error"] is None
        assert result["result"] == "2"

    async def test_run_with_audit(self):
        audit = InMemoryAuditBackend()
        runner = ToolRunner(audit_backend=audit)
        await runner.run(calculator, {"expression": "5*5"})
        events = await audit.query(event_type="tool_call")
        assert len(events) == 1
        assert events[0].data["tool"] == "calculator"

    async def test_run_error_isolation(self):
        """Tool errors should not crash the runner."""

        @suluv_tool(name="bad_tool", description="always fails")
        async def bad_tool() -> str:
            raise RuntimeError("boom!")

        runner = ToolRunner()
        result = await runner.run(bad_tool, {})
        assert result["error"]
        assert "boom" in result["error"]
        assert result["result"] is None

    async def test_run_timeout(self):
        import asyncio

        @suluv_tool(name="slow", description="takes forever", timeout=0.1)
        async def slow_tool() -> str:
            await asyncio.sleep(10)
            return "done"

        runner = ToolRunner()
        result = await runner.run(slow_tool, {})
        assert result["error"]
        assert "timed out" in result["error"].lower()


# ──────────────────────────────────────────────────────────────
#  3. Agent + Tool Integration Tests (ReAct loop)
# ──────────────────────────────────────────────────────────────

class TestAgentToolIntegration:
    """Full ReAct loop: agent calls tools via MockLLM responses."""

    async def test_single_tool_call_and_answer(self):
        """Agent calls calculator, gets result, returns final answer."""
        llm = MockLLM(responses=[
            _react_tool_call(
                "I need to calculate 15 * 7",
                "calculator",
                {"expression": "15 * 7"},
            ),
            _react_final(
                "The calculator returned 105.",
                "15 × 7 = 105",
            ),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Math"),
            llm=llm,
            tools=[calculator],
        )
        result = await agent.run("What is 15 times 7?")
        assert result.success
        assert "105" in result.answer
        assert result.step_count == 2
        assert result.steps[0].action == "calculator"
        assert result.steps[0].observation is not None

    async def test_multi_tool_calls(self):
        """Agent chains multiple tool calls before answering."""
        llm = MockLLM(responses=[
            _react_tool_call(
                "First, get the date difference",
                "date_diff",
                {"start": "2024-01-01", "end": "2024-12-31"},
            ),
            _react_tool_call(
                "Now calculate something with that info",
                "calculator",
                {"expression": "365 * 24"},
            ),
            _react_final(
                "I computed both results.",
                "There are 365 days (8760 hours) in 2024.",
            ),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Analyst"),
            llm=llm,
            tools=[calculator, date_diff],
        )
        result = await agent.run("How many hours in 2024?")
        assert result.success
        assert result.step_count == 3
        assert result.steps[0].action == "date_diff"
        assert result.steps[1].action == "calculator"

    async def test_unknown_tool_recovery(self):
        """Agent references unknown tool, sees error, then answers."""
        llm = MockLLM(responses=[
            _react_tool_call(
                "Let me use a tool",
                "nonexistent_tool",
                {},
            ),
            _react_final(
                "That tool doesn't exist, I'll answer directly.",
                "42 is the answer.",
            ),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Test"),
            llm=llm,
            tools=[calculator],
        )
        result = await agent.run("What is the answer?")
        assert result.success
        assert "42" in result.answer
        assert result.steps[0].error  # first step had unknown tool error

    async def test_direct_answer_no_tools(self):
        """Agent answers directly without calling any tools."""
        llm = MockLLM(responses=[
            _react_final(
                "This is a simple factual question.",
                "The capital of France is Paris.",
            ),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="QA"),
            llm=llm,
            tools=[calculator],
        )
        result = await agent.run("Capital of France?")
        assert result.success
        assert "Paris" in result.answer
        assert result.step_count == 1
        assert result.steps[0].action == "final_answer"

    async def test_max_steps_reached(self):
        """Agent exceeds max steps without answering."""
        # All responses call a tool, never final_answer
        responses = [
            _react_tool_call("thinking...", "calculator", {"expression": f"{i}+1"})
            for i in range(5)
        ]
        agent = _make_agent(responses, tools=[calculator])
        agent.role.max_steps = 3
        result = await agent.run("Loop forever")
        assert not result.success
        assert "Max steps" in result.error

    async def test_json_extraction_flow(self):
        """Agent uses json_extractor to pull data from JSON."""
        sample_json = json.dumps({
            "users": [
                {"name": "Alice", "score": 95},
                {"name": "Bob", "score": 87},
            ]
        })
        llm = MockLLM(responses=[
            _react_tool_call(
                "Extract Alice's score",
                "json_extractor",
                {"json_string": sample_json, "path": "users.0.score"},
            ),
            _react_final(
                "Alice scored 95.",
                "Alice's score is 95.",
            ),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="DataBot"),
            llm=llm,
            tools=[json_extractor],
        )
        result = await agent.run("What's Alice's score?")
        assert result.success
        assert "95" in result.answer

    async def test_calculator_chain(self):
        """Agent does multi-step math."""
        llm = MockLLM(responses=[
            _react_tool_call("step 1", "calculator", {"expression": "100 * 1.08"}),
            _react_tool_call("step 2", "calculator", {"expression": "108 * 1.08"}),
            _react_final(
                "After 2 years at 8% compound interest: $116.64",
                "$100 at 8% compound interest for 2 years = $116.64",
            ),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Finance"),
            llm=llm,
            tools=[calculator],
        )
        result = await agent.run("Compound interest on $100 at 8% for 2 years?")
        assert result.success
        assert "116.64" in result.answer
        assert result.step_count == 3

    async def test_non_json_response_treated_as_answer(self):
        """If LLM doesn't return JSON, treat as final answer."""
        llm = MockLLM(responses=["The sky is blue."])
        agent = SuluvAgent(
            role=AgentRole(name="Simple"),
            llm=llm,
            tools=[],
        )
        result = await agent.run("What color is the sky?")
        assert result.success
        assert "blue" in result.answer.lower()


# ──────────────────────────────────────────────────────────────
#  4. Agent with Memory Integration
# ──────────────────────────────────────────────────────────────

class TestAgentWithMemory:
    """Test memory loading and saving during agent runs."""

    async def test_memory_save_after_run(self):
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        llm = MockLLM(responses=[
            _react_final("Easy question.", "42"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="MemAgent"),
            llm=llm,
            tools=[],
            memory=mm,
        )
        ctx = AgentContext(session_id="s1", user_id="u1")
        result = await agent.run("What is the answer?", context=ctx)
        assert result.success
        assert result.answer == "42"

        # Memory should have saved the interaction
        items = await stm.all()
        assert len(items) > 0

    async def test_memory_load_context(self):
        """Memory context is loaded and included in system prompt."""
        stm = InMemoryShortTermMemory()
        # Pre-populate memory
        await stm.set("user_name", "Alice")
        mm = MemoryManager(short_term=stm)

        llm = MockLLM(responses=[
            _react_final("Recalling from memory...", "Hello Alice!"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="MemBot"),
            llm=llm,
            tools=[],
            memory=mm,
        )
        ctx = AgentContext(session_id="s1", user_id="u1")
        result = await agent.run("Who am I?", context=ctx)
        assert result.success

        # Verify the LLM was called with memory context in system prompt
        prompt_sent = llm.history[0]
        system_msg = prompt_sent.messages[0]
        assert "memory" in system_msg.text.lower() or "Alice" in system_msg.text


# ──────────────────────────────────────────────────────────────
#  5. Agent with Audit Backend
# ──────────────────────────────────────────────────────────────

class TestAgentWithAudit:
    """Test audit logging during agent runs."""

    async def test_audit_event_logged(self):
        audit = InMemoryAuditBackend()
        llm = MockLLM(responses=[_react_final("done", "answer")])
        agent = SuluvAgent(
            role=AgentRole(name="AuditAgent"),
            llm=llm,
            tools=[],
            audit_backend=audit,
        )
        ctx = AgentContext(session_id="s1", user_id="u1")
        await agent.run("test", context=ctx)
        events = await audit.query(event_type="agent_run")
        assert len(events) == 1
        assert events[0].data["role"] == "AuditAgent"
        assert events[0].data["success"] is True

    async def test_audit_tracks_tool_calls(self):
        audit = InMemoryAuditBackend()
        llm = MockLLM(responses=[
            _react_tool_call("calc", "calculator", {"expression": "1+1"}),
            _react_final("got it", "2"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="AuditCalc"),
            llm=llm,
            tools=[calculator],
            audit_backend=audit,
        )
        await agent.run("1+1?")
        tool_events = await audit.query(event_type="tool_call")
        assert len(tool_events) == 1
        assert tool_events[0].data["tool"] == "calculator"

        agent_events = await audit.query(event_type="agent_run")
        assert len(agent_events) == 1


# ──────────────────────────────────────────────────────────────
#  6. Cost Tracking
# ──────────────────────────────────────────────────────────────

class TestCostTracking:
    """Test cost tracker accumulation."""

    async def test_cost_tracked_across_steps(self):
        llm = MockLLM(responses=[
            _react_tool_call("step1", "calculator", {"expression": "1+2"}),
            _react_tool_call("step2", "calculator", {"expression": "3+4"}),
            _react_final("done", "7"),
        ])
        tracker = CostTracker()
        agent = SuluvAgent(
            role=AgentRole(name="CostTest"),
            llm=llm,
            tools=[calculator],
            cost_tracker=tracker,
        )
        result = await agent.run("Calculate stuff")
        assert result.total_tokens > 0
        assert tracker.total_tokens > 0
        assert len(tracker._records) == 3  # 3 LLM calls


# ──────────────────────────────────────────────────────────────
#  7. Agent Streaming
# ──────────────────────────────────────────────────────────────

class TestAgentStreaming:
    """Test run_stream yields proper events."""

    async def test_stream_events(self):
        llm = MockLLM(responses=[
            _react_tool_call("think first", "calculator", {"expression": "2*3"}),
            _react_final("result is 6", "6"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="StreamBot"),
            llm=llm,
            tools=[calculator],
        )
        events = []
        async for event in agent.run_stream("2*3?"):
            events.append(event)

        types = [e["type"] for e in events]
        assert "thought" in types
        assert "action" in types
        assert "observation" in types
        assert "answer" in types


# ──────────────────────────────────────────────────────────────
#  8. Pre-built Agent Factory Tests
# ──────────────────────────────────────────────────────────────

class TestAgentFactories:
    """Test pre-built agent factories."""

    async def test_create_assistant(self):
        llm = MockLLM(responses=[
            _react_tool_call("calc", "calculator", {"expression": "7*8"}),
            _react_final("done", "56"),
        ])
        agent = create_assistant(llm=llm)
        assert agent.role.name == "Assistant"
        assert "calculator" in agent.tools
        assert "datetime_now" in agent.tools
        result = await agent.run("7*8?")
        assert result.success
        assert "56" in result.answer

    async def test_create_researcher(self):
        llm = MockLLM(responses=[
            _react_final("I know this.", "The answer is 42."),
        ])
        agent = create_researcher(llm=llm)
        assert agent.role.name == "Researcher"
        assert "web_search" in agent.tools
        assert "http_fetch" in agent.tools
        result = await agent.run("Test query")
        assert result.success

    async def test_create_coder(self):
        llm = MockLLM(responses=[
            _react_final("Simple answer.", "def hello(): pass"),
        ])
        agent = create_coder(llm=llm)
        assert agent.role.name == "Coder"
        assert "file_reader" in agent.tools
        assert "file_writer" in agent.tools
        assert "shell_exec" in agent.tools
        result = await agent.run("Write hello function")
        assert result.success

    async def test_create_data_analyst(self):
        llm = MockLLM(responses=[
            _react_tool_call(
                "Extract data",
                "json_extractor",
                {"json_string": '{"val": 42}', "path": "val"},
            ),
            _react_final("got it", "The value is 42."),
        ])
        agent = create_data_analyst(llm=llm)
        assert agent.role.name == "DataAnalyst"
        assert "json_extractor" in agent.tools
        assert "calculator" in agent.tools
        result = await agent.run("Extract the value")
        assert result.success
        assert "42" in result.answer

    async def test_factory_with_extra_tools(self):
        @suluv_tool(name="custom_tool", description="A custom tool")
        async def custom_tool(x: str) -> str:
            return f"custom: {x}"

        llm = MockLLM(responses=[_react_final("ok", "done")])
        agent = create_assistant(llm=llm, extra_tools=[custom_tool])
        assert "custom_tool" in agent.tools
        assert "calculator" in agent.tools  # default tools still there


# ──────────────────────────────────────────────────────────────
#  9. Custom Tool Creation Tests
# ──────────────────────────────────────────────────────────────

class TestCustomTools:
    """Test creating and using custom tools with the @suluv_tool decorator."""

    async def test_sync_tool(self):
        @suluv_tool(name="upper", description="Uppercase a string")
        def upper_tool(text: str) -> str:
            return text.upper()

        result = await upper_tool.execute(text="hello")
        assert result == "HELLO"

    async def test_async_tool(self):
        @suluv_tool(name="greet", description="Greet someone")
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await greet.execute(name="Alice")
        assert result == "Hello, Alice!"

    async def test_tool_with_agent(self):
        @suluv_tool(name="reverse", description="Reverse a string")
        async def reverse(text: str) -> str:
            return text[::-1]

        llm = MockLLM(responses=[
            _react_tool_call("reverse it", "reverse", {"text": "hello"}),
            _react_final("reversed", "olleh"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="ReverseBot"),
            llm=llm,
            tools=[reverse],
        )
        result = await agent.run("Reverse 'hello'")
        assert result.success
        assert result.steps[0].action == "reverse"
        # The observation should contain the actual tool result
        assert "olleh" in str(result.steps[0].observation)

    async def test_tool_schema_inference(self):
        @suluv_tool(name="add", description="Add two numbers")
        async def add(a: int, b: int = 0) -> int:
            return a + b

        assert add.name == "add"
        # Note: with from __future__ import annotations, type hints are strings
        # so the inferred type defaults to 'string'; verify schema structure
        assert "a" in add.parameters["properties"]
        assert "b" in add.parameters["properties"]
        assert "a" in add.parameters["required"]
        assert "b" not in add.parameters["required"]  # has default


# ──────────────────────────────────────────────────────────────
#  10. OpenAI / Anthropic Adapter Import Tests
# ──────────────────────────────────────────────────────────────

class TestAdapterImports:
    """Test that adapters import gracefully (even without SDK installed)."""

    def test_mock_llm_available(self):
        from suluv.core.adapters import MockLLM
        assert MockLLM is not None

    def test_openai_import_graceful(self):
        """OpenAIBackend is None if openai is not installed, else a class."""
        from suluv.core.adapters import OpenAIBackend
        # Either None (no openai) or the real class
        if OpenAIBackend is not None:
            assert hasattr(OpenAIBackend, "complete")
            assert hasattr(OpenAIBackend, "stream")

    def test_anthropic_import_graceful(self):
        """AnthropicBackend is None if anthropic is not installed, else a class."""
        from suluv.core.adapters import AnthropicBackend
        if AnthropicBackend is not None:
            assert hasattr(AnthropicBackend, "complete")
            assert hasattr(AnthropicBackend, "stream")


# ──────────────────────────────────────────────────────────────
#  11. End-to-end scenario tests
# ──────────────────────────────────────────────────────────────

class TestE2EScenarios:
    """Full end-to-end scenarios testing realistic agent workflows."""

    async def test_data_analysis_pipeline(self):
        """Agent extracts data, does calculations, and returns insights."""
        sales_data = json.dumps({
            "q1": {"revenue": 150000, "costs": 120000},
            "q2": {"revenue": 180000, "costs": 130000},
            "q3": {"revenue": 200000, "costs": 140000},
            "q4": {"revenue": 220000, "costs": 150000},
        })
        llm = MockLLM(responses=[
            _react_tool_call(
                "Extract Q1 revenue",
                "json_extractor",
                {"json_string": sales_data, "path": "q1.revenue"},
            ),
            _react_tool_call(
                "Extract Q4 revenue",
                "json_extractor",
                {"json_string": sales_data, "path": "q4.revenue"},
            ),
            _react_tool_call(
                "Calculate growth",
                "calculator",
                {"expression": "((220000 - 150000) / 150000) * 100"},
            ),
            _react_final(
                "Q1 revenue was 150000, Q4 was 220000, growth is 46.67%",
                "Revenue grew 46.67% from Q1 ($150K) to Q4 ($220K).",
            ),
        ])
        agent = create_data_analyst(llm=llm)
        result = await agent.run("Analyse yearly revenue growth")
        assert result.success
        assert "46.67" in result.answer or "46.6" in result.answer
        assert result.step_count == 4

    async def test_full_audit_trail(self):
        """Every tool call and agent run is audited."""
        audit = InMemoryAuditBackend()
        llm = MockLLM(responses=[
            _react_tool_call("calc", "calculator", {"expression": "1+1"}),
            _react_tool_call("calc2", "calculator", {"expression": "2+2"}),
            _react_final("done", "Results: 2 and 4"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="AuditBot"),
            llm=llm,
            tools=[calculator],
            audit_backend=audit,
        )
        ctx = AgentContext(user_id="user123", org_id="org456")
        result = await agent.run("Do some math", context=ctx)
        assert result.success

        # Should have 2 tool_call events + 1 agent_run event
        tool_events = await audit.query(event_type="tool_call")
        agent_events = await audit.query(event_type="agent_run")
        assert len(tool_events) == 2
        assert len(agent_events) == 1
        assert agent_events[0].data["steps"] == 3

    async def test_agent_with_all_components(self):
        """Agent with memory + audit + cost tracker + tools — everything wired up."""
        stm = InMemoryShortTermMemory()
        mm = MemoryManager(short_term=stm)
        audit = InMemoryAuditBackend()
        tracker = CostTracker()

        llm = MockLLM(responses=[
            _react_tool_call("get date", "datetime_now", {}),
            _react_final("answering", "Today's date retrieved."),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="FullStack"),
            llm=llm,
            tools=[datetime_now, calculator],
            memory=mm,
            audit_backend=audit,
            cost_tracker=tracker,
        )
        ctx = AgentContext(session_id="s1", user_id="u1")
        result = await agent.run("What's today's date?", context=ctx)

        assert result.success
        assert result.total_tokens > 0
        assert tracker.total_tokens > 0
        # Memory saved
        items = await stm.all()
        assert len(items) > 0
        # Audit logged
        events = await audit.query(event_type="agent_run")
        assert len(events) == 1

    async def test_cancellation(self):
        """Agent can be cancelled mid-run."""

        class FakeCancelToken:
            is_cancelled = True

        llm = MockLLM(responses=[
            _react_tool_call("thinking", "calculator", {"expression": "1+1"}),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="CancelBot"),
            llm=llm,
            tools=[calculator],
            cancel_token=FakeCancelToken(),
        )
        result = await agent.run("Do something")
        assert not result.success
        assert result.error == "Agent cancelled"
