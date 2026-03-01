"""Phase 1 tests — Types, Messages, Ports, and Adapters."""

import pytest
from datetime import datetime, timedelta, timezone, time


# ── Types ─────────────────────────────────────────────────────────────────────


class TestTypes:
    def test_new_id_unique(self):
        from suluv.core.types import new_id
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_new_execution_id_prefix(self):
        from suluv.core.types import new_execution_id
        eid = new_execution_id()
        assert eid.startswith("exec-")

    def test_new_instance_id_prefix(self):
        from suluv.core.types import new_instance_id
        iid = new_instance_id()
        assert iid.startswith("inst-")

    def test_node_state_enum(self):
        from suluv.core.types import NodeState
        assert NodeState.PENDING.value == "pending"
        assert NodeState.DONE.value == "done"

    def test_node_type_enum_has_16_values(self):
        from suluv.core.types import NodeType
        assert len(NodeType) == 16

    def test_error_policy_enum(self):
        from suluv.core.types import ErrorPolicy
        assert ErrorPolicy.FAIL_FAST.value == "fail_fast"
        assert ErrorPolicy.RETRY.value == "retry"
        assert ErrorPolicy.SKIP.value == "skip"
        assert ErrorPolicy.FALLBACK.value == "fallback"

    def test_priority_enum(self):
        from suluv.core.types import Priority
        assert len(Priority) == 4

    def test_instance_status_enum(self):
        from suluv.core.types import InstanceStatus
        assert InstanceStatus.CREATED.value == "created"
        assert InstanceStatus.COMPENSATING.value == "compensating"

    def test_node_execution_dataclass(self):
        from suluv.core.types import NodeExecution, NodeID, NodeType, NodeState
        ne = NodeExecution(
            node_id=NodeID("n1"),
            node_type=NodeType.TOOL,
            state=NodeState.DONE,
            output="hello",
        )
        assert ne.node_id == "n1"
        assert ne.cost_tokens == 0

    def test_execution_result_dataclass(self):
        from suluv.core.types import ExecutionResult, ExecutionID
        r = ExecutionResult(execution_id=ExecutionID("e1"), output="result")
        assert r.success is True
        assert r.output == "result"

    def test_audit_event_has_defaults(self):
        from suluv.core.types import AuditEvent
        ae = AuditEvent(event_type="test")
        assert ae.event_type == "test"
        assert ae.event_id  # auto-generated
        assert ae.timestamp  # auto-generated

    def test_cost_record(self):
        from suluv.core.types import CostRecord
        cr = CostRecord(input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.01)
        assert cr.total_tokens == 30


# ── Messages ──────────────────────────────────────────────────────────────────


class TestMessages:
    def test_content_block_text(self):
        from suluv.core.messages.content import ContentBlock, ContentType
        cb = ContentBlock.text_block("hello")
        assert cb.type == ContentType.TEXT
        assert cb.text == "hello"

    def test_content_block_image_url(self):
        from suluv.core.messages.content import ContentBlock, ContentType
        cb = ContentBlock.image_url_block("https://img.png")
        assert cb.type == ContentType.IMAGE_URL
        assert cb.url == "https://img.png"

    def test_content_block_tool_call(self):
        from suluv.core.messages.content import ContentBlock, ContentType
        cb = ContentBlock.tool_call_block("tc1", "search", {"q": "test"})
        assert cb.type == ContentType.TOOL_CALL
        assert cb.name == "search"
        assert cb.arguments == {"q": "test"}

    def test_content_block_tool_result(self):
        from suluv.core.messages.content import ContentBlock, ContentType
        cb = ContentBlock.tool_result_block("tc1", "found it")
        assert cb.type == ContentType.TOOL_RESULT
        assert cb.tool_call_id == "tc1"

    def test_message_system(self):
        from suluv.core.messages.message import SuluvMessage, MessageRole
        m = SuluvMessage.system("You are helpful")
        assert m.role == MessageRole.SYSTEM
        assert m.text == "You are helpful"

    def test_message_user(self):
        from suluv.core.messages.message import SuluvMessage, MessageRole
        m = SuluvMessage.user("Hello")
        assert m.role == MessageRole.USER
        assert m.text == "Hello"

    def test_message_assistant(self):
        from suluv.core.messages.message import SuluvMessage, MessageRole
        m = SuluvMessage.assistant("Hi there")
        assert m.role == MessageRole.ASSISTANT
        assert m.text == "Hi there"

    def test_message_text_concat(self):
        from suluv.core.messages.content import ContentBlock
        from suluv.core.messages.message import SuluvMessage, MessageRole
        m = SuluvMessage(
            role=MessageRole.USER,
            content=[ContentBlock.text_block("a "), ContentBlock.text_block("b")],
        )
        assert m.text == "a b"

    def test_prompt_construction(self):
        from suluv.core.messages.message import SuluvMessage
        from suluv.core.messages.prompt import SuluvPrompt, ToolSchema
        p = SuluvPrompt(
            messages=[SuluvMessage.user("hi")],
            tools=[ToolSchema(name="search", description="search the web")],
            temperature=0.5,
        )
        assert len(p.messages) == 1
        assert len(p.tools) == 1
        assert p.temperature == 0.5


# ── Adapters ──────────────────────────────────────────────────────────────────


class TestInMemoryEventBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        from suluv.core.adapters import InMemoryEventBus
        bus = InMemoryEventBus()
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("test", handler)
        await bus.publish("test", {"data": "hello"})
        assert len(received) == 1
        assert received[0]["data"] == "hello"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        from suluv.core.adapters import InMemoryEventBus
        bus = InMemoryEventBus()
        received = []

        async def handler(event):
            received.append(event)

        await bus.subscribe("test", handler)
        await bus.unsubscribe("test", handler)
        await bus.publish("test", {"data": "hello"})
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_history(self):
        from suluv.core.adapters import InMemoryEventBus
        bus = InMemoryEventBus()
        await bus.publish("t1", {"a": 1})
        await bus.publish("t2", {"b": 2})
        assert len(bus.history) == 2

    @pytest.mark.asyncio
    async def test_clear(self):
        from suluv.core.adapters import InMemoryEventBus
        bus = InMemoryEventBus()
        await bus.publish("t1", {"a": 1})
        bus.clear()
        assert len(bus.history) == 0


class TestInMemoryStateStore:
    @pytest.mark.asyncio
    async def test_save_load(self):
        from suluv.core.adapters import InMemoryStateStore
        store = InMemoryStateStore()
        await store.save("k1", {"x": 1})
        val = await store.load("k1")
        assert val == {"x": 1}

    @pytest.mark.asyncio
    async def test_load_missing(self):
        from suluv.core.adapters import InMemoryStateStore
        store = InMemoryStateStore()
        val = await store.load("missing")
        assert val is None

    @pytest.mark.asyncio
    async def test_delete(self):
        from suluv.core.adapters import InMemoryStateStore
        store = InMemoryStateStore()
        await store.save("k1", {"x": 1})
        await store.delete("k1")
        assert not await store.exists("k1")

    @pytest.mark.asyncio
    async def test_exists(self):
        from suluv.core.adapters import InMemoryStateStore
        store = InMemoryStateStore()
        assert not await store.exists("k1")
        await store.save("k1", {"x": 1})
        assert await store.exists("k1")


class TestInMemoryAuditBackend:
    @pytest.mark.asyncio
    async def test_write_query(self):
        from suluv.core.adapters import InMemoryAuditBackend
        from suluv.core.types import AuditEvent
        audit = InMemoryAuditBackend()
        await audit.write(AuditEvent(event_type="login", data={"user": "a"}))
        await audit.write(AuditEvent(event_type="logout", data={"user": "a"}))
        results = await audit.query(event_type="login")
        assert len(results) == 1
        assert results[0].event_type == "login"

    @pytest.mark.asyncio
    async def test_query_all(self):
        from suluv.core.adapters import InMemoryAuditBackend
        from suluv.core.types import AuditEvent
        audit = InMemoryAuditBackend()
        await audit.write(AuditEvent(event_type="a"))
        await audit.write(AuditEvent(event_type="b"))
        results = await audit.query()
        assert len(results) == 2


class TestInMemoryMemory:
    @pytest.mark.asyncio
    async def test_short_term_set_get(self):
        from suluv.core.adapters import InMemoryShortTermMemory
        mem = InMemoryShortTermMemory()
        await mem.set("key1", "value1")
        val = await mem.get("key1")
        assert val == "value1"

    @pytest.mark.asyncio
    async def test_short_term_clear(self):
        from suluv.core.adapters import InMemoryShortTermMemory
        mem = InMemoryShortTermMemory()
        await mem.set("key1", "value1")
        await mem.clear()
        val = await mem.get("key1")
        assert val is None

    @pytest.mark.asyncio
    async def test_short_term_all(self):
        from suluv.core.adapters import InMemoryShortTermMemory
        mem = InMemoryShortTermMemory()
        await mem.set("a", 1)
        await mem.set("b", 2)
        all_items = await mem.all()
        assert all_items == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_long_term_scoped(self):
        from suluv.core.adapters import InMemoryLongTermMemory
        mem = InMemoryLongTermMemory()
        await mem.set("k1", "v1", scope="user-a")
        await mem.set("k1", "v2", scope="user-b")
        assert await mem.get("k1", scope="user-a") == "v1"
        assert await mem.get("k1", scope="user-b") == "v2"

    @pytest.mark.asyncio
    async def test_long_term_list_keys(self):
        from suluv.core.adapters import InMemoryLongTermMemory
        mem = InMemoryLongTermMemory()
        await mem.set("k1", "v1", scope="s")
        await mem.set("k2", "v2", scope="s")
        keys = await mem.list_keys(scope="s")
        assert sorted(keys) == ["k1", "k2"]

    @pytest.mark.asyncio
    async def test_episodic_store_recall(self):
        from suluv.core.adapters import InMemoryEpisodicMemory
        mem = InMemoryEpisodicMemory()
        await mem.store({"summary": "talked about weather"})
        await mem.store({"summary": "discussed project"})
        results = await mem.recall("project", limit=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_semantic_store_search(self):
        from suluv.core.adapters import InMemorySemanticMemory
        mem = InMemorySemanticMemory()
        await mem.store("doc1", "Python is a programming language")
        await mem.store("doc2", "Java is also a programming language")
        results = await mem.search("Python programming", limit=5)
        assert len(results) >= 1
        # The one mentioning Python should rank higher
        assert any("Python" in r.value for r in results)


class TestInMemoryHumanTaskQueue:
    @pytest.mark.asyncio
    async def test_emit_poll(self):
        from suluv.core.adapters import InMemoryHumanTaskQueue
        from suluv.core.ports.human_task_queue import HumanTask
        queue = InMemoryHumanTaskQueue()
        task = HumanTask(title="Review doc", assigned_to="user-1", role="reviewer")
        task_id = await queue.emit(task)
        assert task_id  # auto-assigned
        tasks = await queue.poll(role="reviewer")
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_claim_release(self):
        from suluv.core.adapters import InMemoryHumanTaskQueue
        from suluv.core.ports.human_task_queue import HumanTask, TaskStatus
        queue = InMemoryHumanTaskQueue()
        task = HumanTask(title="Review")
        task_id = await queue.emit(task)
        claimed = await queue.claim(task_id, "user-1")
        assert claimed.status == TaskStatus.CLAIMED
        released = await queue.release(task_id)
        assert released.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_complete(self):
        from suluv.core.adapters import InMemoryHumanTaskQueue
        from suluv.core.ports.human_task_queue import HumanTask, TaskStatus
        queue = InMemoryHumanTaskQueue()
        task = HumanTask(title="Review")
        task_id = await queue.emit(task)
        await queue.claim(task_id, "user-1")
        completed = await queue.complete(task_id, {"approved": True})
        assert completed.status == TaskStatus.COMPLETED
        assert completed.result == {"approved": True}


class TestInMemoryRulesEngine:
    @pytest.mark.asyncio
    async def test_register_evaluate(self):
        from suluv.core.adapters import InMemoryRulesEngine
        engine = InMemoryRulesEngine()
        table = {
            "name": "eligibility",
            "inputs": ["income", "score"],
            "output": "decision",
            "hit_policy": "FIRST",
            "rules": [
                {"when": {"income": ">500000", "score": ">700"}, "then": "approve"},
                {"when": {"score": "<500"}, "then": "reject"},
                {"default": True, "then": "review"},
            ],
        }
        await engine.register_table("eligibility", table)
        decision = await engine.evaluate("eligibility", {"income": 600000, "score": 750})
        assert decision.outcome == "approve"

    @pytest.mark.asyncio
    async def test_default_rule(self):
        from suluv.core.adapters import InMemoryRulesEngine
        engine = InMemoryRulesEngine()
        table = {
            "name": "simple",
            "inputs": ["x"],
            "output": "result",
            "hit_policy": "FIRST",
            "rules": [
                {"when": {"x": ">100"}, "then": "big"},
                {"default": True, "then": "small"},
            ],
        }
        await engine.register_table("simple", table)
        decision = await engine.evaluate("simple", {"x": 50})
        assert decision.outcome == "small"


class TestInMemoryBusinessCalendar:
    @pytest.mark.asyncio
    async def test_is_working_time(self):
        from suluv.core.adapters import InMemoryBusinessCalendar
        cal = InMemoryBusinessCalendar(
            start_hour=9, end_hour=18,
            working_days=[0, 1, 2, 3, 4],  # Mon-Fri
        )
        # A Wednesday at 10am
        dt = datetime(2026, 2, 25, 10, 0, tzinfo=timezone.utc)  # Wed
        assert await cal.is_working_time(dt)

        # Same Wednesday at 20:00
        dt_late = datetime(2026, 2, 25, 20, 0, tzinfo=timezone.utc)
        assert not await cal.is_working_time(dt_late)

    @pytest.mark.asyncio
    async def test_add_business_hours(self):
        from suluv.core.adapters import InMemoryBusinessCalendar
        cal = InMemoryBusinessCalendar(
            start_hour=9, end_hour=17,
            working_days=[0, 1, 2, 3, 4],
        )
        # Start at Wed 15:00, add 4 business hours → should land on Thu 11:00
        start = datetime(2026, 2, 25, 15, 0, tzinfo=timezone.utc)
        result = await cal.add_business_hours(start, timedelta(hours=4))
        assert result.hour == 11
        assert result.day == 26  # Thursday

    @pytest.mark.asyncio
    async def test_next_working_time(self):
        from suluv.core.adapters import InMemoryBusinessCalendar
        cal = InMemoryBusinessCalendar(
            start_hour=9, end_hour=17,
            working_days=[0, 1, 2, 3, 4],
        )
        # Saturday at noon → should return Monday 9am
        sat = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
        nxt = await cal.next_working_time(sat)
        assert nxt.weekday() == 0  # Monday
        assert nxt.hour == 9


class TestInMemoryProcessInstanceStore:
    @pytest.mark.asyncio
    async def test_save_load(self):
        from suluv.core.adapters import InMemoryProcessInstanceStore
        from suluv.core.ports.process_instance_store import ProcessInstance
        store = InMemoryProcessInstanceStore()
        inst = ProcessInstance(instance_id="i1", process_name="test", version="1.0")
        await store.save(inst)
        loaded = await store.load("i1")
        assert loaded is not None
        assert loaded.process_name == "test"

    @pytest.mark.asyncio
    async def test_query(self):
        from suluv.core.adapters import InMemoryProcessInstanceStore
        from suluv.core.ports.process_instance_store import ProcessInstance
        store = InMemoryProcessInstanceStore()
        await store.save(ProcessInstance(instance_id="i1", process_name="loan", version="1.0"))
        await store.save(ProcessInstance(instance_id="i2", process_name="kyc", version="1.0"))
        results = await store.query(process_name="loan")
        assert len(results) == 1
        assert results[0].instance_id == "i1"

    @pytest.mark.asyncio
    async def test_delete_count(self):
        from suluv.core.adapters import InMemoryProcessInstanceStore
        from suluv.core.ports.process_instance_store import ProcessInstance
        store = InMemoryProcessInstanceStore()
        await store.save(ProcessInstance(instance_id="i1", process_name="test", version="1.0"))
        assert await store.count() == 1
        await store.delete("i1")
        assert await store.count() == 0


class TestInMemoryTemplateEngine:
    @pytest.mark.asyncio
    async def test_render(self):
        from suluv.core.adapters import InMemoryTemplateEngine
        from suluv.core.ports.template_engine import DocumentTemplate
        engine = InMemoryTemplateEngine()
        template = DocumentTemplate(
            name="letter",
            template_content="Dear $name, your loan of $amount is approved.",
            output_format="text",
        )
        doc = await engine.render(template, {"name": "Ravi", "amount": "500000"})
        content = doc.content.decode("utf-8") if isinstance(doc.content, bytes) else doc.content
        assert "Ravi" in content
        assert "500000" in content

    @pytest.mark.asyncio
    async def test_validate_template(self):
        from suluv.core.adapters import InMemoryTemplateEngine
        from suluv.core.ports.template_engine import DocumentTemplate
        engine = InMemoryTemplateEngine()
        template = DocumentTemplate(
            name="test",
            template_content="Hello $name",
            output_format="text",
        )
        errors = await engine.validate_template(template)
        assert errors == []


class TestMockLLM:
    @pytest.mark.asyncio
    async def test_default_response(self):
        from suluv.core.adapters import MockLLM
        from suluv.core.messages.prompt import SuluvPrompt
        from suluv.core.messages.message import SuluvMessage
        llm = MockLLM()
        resp = await llm.complete(SuluvPrompt(messages=[SuluvMessage.user("hi")]))
        assert resp.text == "Mock LLM response"
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_queued_responses(self):
        from suluv.core.adapters import MockLLM
        from suluv.core.messages.prompt import SuluvPrompt
        from suluv.core.messages.message import SuluvMessage
        llm = MockLLM(responses=["first", "second"])
        p = SuluvPrompt(messages=[SuluvMessage.user("hi")])
        r1 = await llm.complete(p)
        r2 = await llm.complete(p)
        assert r1.text == "first"
        assert r2.text == "second"
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_stream(self):
        from suluv.core.adapters import MockLLM
        from suluv.core.messages.prompt import SuluvPrompt
        from suluv.core.messages.message import SuluvMessage
        llm = MockLLM(responses=["hello world"])
        p = SuluvPrompt(messages=[SuluvMessage.user("hi")])
        chunks = []
        async for chunk in llm.stream(p):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert "hello" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_reset(self):
        from suluv.core.adapters import MockLLM
        from suluv.core.messages.prompt import SuluvPrompt
        from suluv.core.messages.message import SuluvMessage
        llm = MockLLM()
        await llm.complete(SuluvPrompt(messages=[SuluvMessage.user("hi")]))
        assert llm.call_count == 1
        llm.reset()
        assert llm.call_count == 0
