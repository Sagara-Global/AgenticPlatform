"""Tests — Threads + Checkpointing.

Tests conversation continuity via threads, checkpoint persistence,
multi-turn memory, thread management APIs, and integration with
pre-built agent factories.
"""

from __future__ import annotations

import json
import pytest

from suluv.core.adapters.mock_llm import MockLLM
from suluv.core.adapters.memory_checkpointer import InMemoryCheckpointer
from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.context import AgentContext
from suluv.core.agent.role import AgentRole
from suluv.core.agent.thread import Thread, Checkpoint
from suluv.core.agents import create_assistant, create_data_analyst
from suluv.core.tools.builtins import calculator


# ── Helpers ──────────────────────────────────────────────────

def _react_final(thought: str, answer: str) -> str:
    return json.dumps({"thought": thought, "final_answer": answer})


def _react_tool_call(thought: str, tool: str, args: dict) -> str:
    return json.dumps({"thought": thought, "action": tool, "action_input": args})


def _make_agent(
    responses: list[str],
    checkpointer: InMemoryCheckpointer | None = None,
    **kwargs,
) -> SuluvAgent:
    return SuluvAgent(
        role=AgentRole(name="TestBot", max_steps=10),
        llm=MockLLM(responses=responses),
        tools=[calculator],
        checkpointer=checkpointer,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────
#  1. Thread Model
# ──────────────────────────────────────────────────────────────

class TestThreadModel:
    def test_default_thread(self):
        t = Thread(thread_id="t1")
        assert t.thread_id == "t1"
        assert t.message_count == 0
        assert t.checkpoint_count == 0
        assert t.last_checkpoint is None

    def test_append_message(self):
        from suluv.core.messages.message import SuluvMessage
        t = Thread(thread_id="t1")
        t.append_message(SuluvMessage.user("hello"))
        t.append_message(SuluvMessage.assistant("hi!"))
        assert t.message_count == 2

    def test_add_checkpoint(self):
        from suluv.core.messages.message import SuluvMessage
        t = Thread(thread_id="t1")
        msgs = [SuluvMessage.user("q"), SuluvMessage.assistant("a")]
        cp = t.add_checkpoint(msgs, step=1, metadata={"task": "test"})
        assert isinstance(cp, Checkpoint)
        assert cp.step == 1
        assert len(cp.messages) == 2
        assert t.checkpoint_count == 1
        assert t.last_checkpoint is cp

    def test_to_dict(self):
        t = Thread(thread_id="t1", metadata={"user_id": "u1"})
        d = t.to_dict()
        assert d["thread_id"] == "t1"
        assert d["metadata"]["user_id"] == "u1"
        assert d["message_count"] == 0


# ──────────────────────────────────────────────────────────────
#  2. InMemoryCheckpointer
# ──────────────────────────────────────────────────────────────

class TestInMemoryCheckpointer:
    async def test_put_get(self):
        cp = InMemoryCheckpointer()
        t = Thread(thread_id="t1")
        await cp.put(t)
        loaded = await cp.get("t1")
        assert loaded is not None
        assert loaded.thread_id == "t1"

    async def test_get_missing(self):
        cp = InMemoryCheckpointer()
        assert await cp.get("nope") is None

    async def test_delete(self):
        cp = InMemoryCheckpointer()
        await cp.put(Thread(thread_id="t1"))
        await cp.delete("t1")
        assert await cp.get("t1") is None

    async def test_list_sorted(self):
        import asyncio
        cp = InMemoryCheckpointer()
        t1 = Thread(thread_id="t1")
        await cp.put(t1)
        await asyncio.sleep(0.01)
        t2 = Thread(thread_id="t2")
        await cp.put(t2)
        threads = await cp.list()
        assert len(threads) == 2
        assert threads[0].thread_id == "t2"  # most recent first

    async def test_list_with_metadata_filter(self):
        cp = InMemoryCheckpointer()
        await cp.put(Thread(thread_id="t1", metadata={"user": "alice"}))
        await cp.put(Thread(thread_id="t2", metadata={"user": "bob"}))
        await cp.put(Thread(thread_id="t3", metadata={"user": "alice"}))
        result = await cp.list(metadata_filter={"user": "alice"})
        assert len(result) == 2
        assert all(t.metadata["user"] == "alice" for t in result)

    async def test_list_limit(self):
        cp = InMemoryCheckpointer()
        for i in range(10):
            await cp.put(Thread(thread_id=f"t{i}"))
        result = await cp.list(limit=3)
        assert len(result) == 3

    async def test_thread_count(self):
        cp = InMemoryCheckpointer()
        assert cp.thread_count == 0
        await cp.put(Thread(thread_id="t1"))
        assert cp.thread_count == 1

    async def test_clear(self):
        cp = InMemoryCheckpointer()
        await cp.put(Thread(thread_id="t1"))
        cp.clear()
        assert cp.thread_count == 0


# ──────────────────────────────────────────────────────────────
#  3. Agent with Threads (no checkpointer — local cache)
# ──────────────────────────────────────────────────────────────

class TestAgentLocalThreads:
    """Threads work even without a checkpointer (in-process cache)."""

    async def test_first_turn_creates_thread(self):
        agent = _make_agent([_react_final("hi", "Hello!")])
        ctx = AgentContext(thread_id="t1")
        result = await agent.run("hi", context=ctx)
        assert result.success
        assert result.metadata["thread_id"] == "t1"
        assert result.metadata["checkpoint_count"] == 1

        thread = await agent.get_thread("t1")
        assert thread is not None
        assert thread.message_count > 0

    async def test_second_turn_sees_history(self):
        """The LLM prompt on turn 2 includes turn 1's messages."""
        llm = MockLLM(responses=[
            _react_final("greeting", "Hello Alice!"),
            _react_final("recall", "Your name is Alice."),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="MemBot", max_steps=5),
            llm=llm,
            tools=[],
        )
        ctx = AgentContext(thread_id="t1")
        await agent.run("My name is Alice", context=ctx)
        await agent.run("What's my name?", context=ctx)

        # The second prompt should have 5+ messages:
        # system + user1 + assistant1 (from turn 1) + user2 (turn 2)
        second_prompt = llm.history[1]
        # system(1) + history(user+assistant from first turn) + user2 = 4
        assert len(second_prompt.messages) >= 4

        # Verify the first turn's user message is in the history
        all_texts = [m.text for m in second_prompt.messages]
        assert any("Alice" in t for t in all_texts)

    async def test_separate_threads_isolated(self):
        """Different thread_ids have independent histories."""
        llm = MockLLM(responses=[
            _react_final("t1", "Thread 1 answer"),
            _react_final("t2", "Thread 2 answer"),
            _react_final("t1 again", "Back to thread 1"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=llm,
            tools=[],
        )
        await agent.run("Question A", context=AgentContext(thread_id="thread-1"))
        await agent.run("Question B", context=AgentContext(thread_id="thread-2"))
        await agent.run("Question C", context=AgentContext(thread_id="thread-1"))

        t1 = await agent.get_thread("thread-1")
        t2 = await agent.get_thread("thread-2")
        assert t1.checkpoint_count == 2  # two runs on thread-1
        assert t2.checkpoint_count == 1  # one run on thread-2

    async def test_no_thread_id_no_thread(self):
        """Without thread_id, no thread is created."""
        agent = _make_agent([_react_final("ok", "done")])
        result = await agent.run("just a question")
        assert "thread_id" not in result.metadata

    async def test_list_threads(self):
        llm = MockLLM(responses=[
            _react_final("a", "1"),
            _react_final("b", "2"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=llm,
            tools=[],
        )
        await agent.run("q1", context=AgentContext(thread_id="t1"))
        await agent.run("q2", context=AgentContext(thread_id="t2"))
        threads = await agent.list_threads()
        assert len(threads) == 2

    async def test_delete_thread(self):
        agent = _make_agent([_react_final("a", "1")])
        await agent.run("q", context=AgentContext(thread_id="t1"))
        await agent.delete_thread("t1")
        assert await agent.get_thread("t1") is None


# ──────────────────────────────────────────────────────────────
#  4. Agent with Checkpointer (persistent threads)
# ──────────────────────────────────────────────────────────────

class TestAgentWithCheckpointer:
    """Threads persist via the checkpointer."""

    async def test_checkpoint_saves_thread(self):
        cp = InMemoryCheckpointer()
        agent = _make_agent(
            [_react_final("ok", "Hello!")],
            checkpointer=cp,
        )
        await agent.run("hi", context=AgentContext(thread_id="t1"))
        assert cp.thread_count == 1
        thread = await cp.get("t1")
        assert thread is not None
        assert thread.checkpoint_count == 1

    async def test_multi_turn_with_checkpointer(self):
        cp = InMemoryCheckpointer()
        llm = MockLLM(responses=[
            _react_final("greeting", "Hi Alice!"),
            _react_final("recall", "You are Alice."),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=llm,
            tools=[],
            checkpointer=cp,
        )
        ctx = AgentContext(thread_id="t1")
        r1 = await agent.run("I'm Alice", context=ctx)
        r2 = await agent.run("Who am I?", context=ctx)

        thread = await cp.get("t1")
        assert thread.checkpoint_count == 2
        # Thread should have messages from both turns
        assert thread.message_count >= 4  # user1+asst1+user2+asst2

    async def test_resume_across_agent_instances(self):
        """Thread survives agent recreation — checkpointer persists."""
        cp = InMemoryCheckpointer()

        # Agent 1: first turn
        llm1 = MockLLM(responses=[_react_final("ok", "Noted: key=42")])
        agent1 = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=llm1,
            tools=[],
            checkpointer=cp,
        )
        await agent1.run("Remember: key=42",
                         context=AgentContext(thread_id="t1"))

        # Agent 2: new instance, same checkpointer
        llm2 = MockLLM(responses=[_react_final("recall", "key=42")])
        agent2 = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=llm2,
            tools=[],
            checkpointer=cp,
        )
        r = await agent2.run("What was the key?",
                             context=AgentContext(thread_id="t1"))
        assert r.success

        # Second agent's prompt should include the first turn's messages
        prompt = llm2.history[0]
        all_text = " ".join(m.text for m in prompt.messages)
        assert "42" in all_text

    async def test_tool_calls_preserved_in_thread(self):
        """Tool call/result messages are saved in the thread."""
        cp = InMemoryCheckpointer()
        llm = MockLLM(responses=[
            _react_tool_call("calc", "calculator", {"expression": "7*6"}),
            _react_final("done", "42"),
        ])
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=llm,
            tools=[calculator],
            checkpointer=cp,
        )
        await agent.run("7*6?", context=AgentContext(thread_id="t1"))

        thread = await cp.get("t1")
        # Messages: user + assistant(tool_call) + tool(result) + assistant(final)
        assert thread.message_count >= 4

    async def test_checkpoint_metadata(self):
        """Checkpoints store task/answer/success metadata."""
        cp = InMemoryCheckpointer()
        agent = _make_agent(
            [_react_final("ok", "The answer is 42")],
            checkpointer=cp,
        )
        await agent.run("What is the answer?",
                        context=AgentContext(thread_id="t1"))
        thread = await cp.get("t1")
        last_cp = thread.last_checkpoint
        assert last_cp.metadata["success"] is True
        assert "answer" in last_cp.metadata["task"].lower() or "42" in last_cp.metadata["answer"]

    async def test_thread_metadata_tracks_user(self):
        """Thread metadata includes user_id and agent name."""
        cp = InMemoryCheckpointer()
        agent = _make_agent(
            [_react_final("ok", "hi")],
            checkpointer=cp,
        )
        await agent.run("hi",
                        context=AgentContext(thread_id="t1", user_id="u1"))
        thread = await cp.get("t1")
        assert thread.metadata["user_id"] == "u1"
        assert thread.metadata["agent"] == "TestBot"


# ──────────────────────────────────────────────────────────────
#  5. Factory Agents with Checkpointer
# ──────────────────────────────────────────────────────────────

class TestFactoryWithCheckpointer:
    async def test_assistant_threaded(self):
        cp = InMemoryCheckpointer()
        llm = MockLLM(responses=[
            _react_final("greeting", "Hello!"),
            _react_final("recall", "I remember you."),
        ])
        agent = create_assistant(llm=llm, checkpointer=cp)
        ctx = AgentContext(thread_id="conv-1")
        await agent.run("Hi there", context=ctx)
        await agent.run("Do you remember me?", context=ctx)

        thread = await cp.get("conv-1")
        assert thread.checkpoint_count == 2

    async def test_data_analyst_threaded(self):
        cp = InMemoryCheckpointer()
        llm = MockLLM(responses=[
            _react_final("ok", "Revenue is 100K"),
        ])
        agent = create_data_analyst(llm=llm, checkpointer=cp)
        await agent.run("Analyse Q1",
                        context=AgentContext(thread_id="analysis-1"))
        assert cp.thread_count == 1


# ──────────────────────────────────────────────────────────────
#  6. Edge Cases
# ──────────────────────────────────────────────────────────────

class TestThreadEdgeCases:
    async def test_empty_thread_first_run(self):
        """First run on a thread_id that doesn't exist yet."""
        cp = InMemoryCheckpointer()
        agent = _make_agent(
            [_react_final("first", "Hello!")],
            checkpointer=cp,
        )
        result = await agent.run("hello",
                                 context=AgentContext(thread_id="new-thread"))
        assert result.success
        thread = await cp.get("new-thread")
        assert thread is not None
        assert thread.checkpoint_count == 1

    async def test_guardrail_block_no_thread_corruption(self):
        """Guardrail-blocked input should NOT add to thread."""
        from suluv.core.agent.guardrail_chain import GuardrailChain
        from suluv.core.ports.guardrail import Guardrail, GuardrailResult, GuardrailAction

        class BlockAll(Guardrail):
            async def check_input(self, context: dict, text: str) -> GuardrailResult:
                return GuardrailResult(action=GuardrailAction.BLOCK, message="blocked")

            async def check_output(self, context: dict, text: str) -> GuardrailResult:
                return GuardrailResult(action=GuardrailAction.ALLOW)

        cp = InMemoryCheckpointer()
        chain = GuardrailChain()
        chain.add(BlockAll())

        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=MockLLM(responses=[_react_final("ok", "hi")]),
            tools=[],
            checkpointer=cp,
            guardrails=chain,
        )
        result = await agent.run("bad input",
                                 context=AgentContext(thread_id="t1"))
        assert not result.success
        # Thread should not exist (input was blocked before thread save)
        thread = await cp.get("t1")
        assert thread is None

    async def test_max_steps_still_checkpoints(self):
        """Even if agent hits max steps, thread is still saved."""
        cp = InMemoryCheckpointer()
        responses = [
            _react_tool_call("loop", "calculator", {"expression": f"{i}+1"})
            for i in range(5)
        ]
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=2),
            llm=MockLLM(responses=responses),
            tools=[calculator],
            checkpointer=cp,
        )
        result = await agent.run("loop",
                                 context=AgentContext(thread_id="t1"))
        assert not result.success  # max steps
        thread = await cp.get("t1")
        assert thread is not None
        assert thread.checkpoint_count == 1

    async def test_context_thread_id_field(self):
        """AgentContext exposes thread_id properly."""
        ctx = AgentContext(thread_id="my-thread", user_id="u1")
        assert ctx.thread_id == "my-thread"
        d = ctx.to_dict()
        assert d["thread_id"] == "my-thread"

    async def test_multiple_checkpoints_on_same_thread(self):
        """Each run adds exactly one checkpoint."""
        cp = InMemoryCheckpointer()
        responses = [_react_final(f"turn{i}", f"Answer {i}") for i in range(5)]
        llm = MockLLM(responses=responses)
        agent = SuluvAgent(
            role=AgentRole(name="Bot", max_steps=5),
            llm=llm,
            tools=[],
            checkpointer=cp,
        )
        ctx = AgentContext(thread_id="t1")
        for _ in range(5):
            await agent.run("question", context=ctx)

        thread = await cp.get("t1")
        assert thread.checkpoint_count == 5
        # Steps should be sequential
        for i, ckpt in enumerate(thread.checkpoints):
            assert ckpt.step == i + 1
