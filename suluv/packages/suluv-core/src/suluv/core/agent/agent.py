"""SuluvAgent — the core ReAct agent. Works standalone (Level 1), no graph required."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from suluv.core.agent.context import AgentContext
from suluv.core.agent.cost_tracker import CostTracker
from suluv.core.agent.guardrail_chain import GuardrailChain
from suluv.core.agent.memory_manager import MemoryManager
from suluv.core.agent.result import AgentResult, StepRecord
from suluv.core.agent.role import AgentRole
from suluv.core.agent.thread import Thread
from suluv.core.messages.content import ContentBlock, ContentType
from suluv.core.messages.message import MessageRole, SuluvMessage
from suluv.core.messages.prompt import SuluvPrompt, ToolSchema
from suluv.core.ports.audit_backend import AuditBackend
from suluv.core.ports.checkpointer import Checkpointer
from suluv.core.ports.guardrail import GuardrailAction
from suluv.core.ports.llm_backend import LLMBackend
from suluv.core.tools.decorator import SuluvTool
from suluv.core.tools.runner import ToolRunner
from suluv.core.types import AuditEvent, CostRecord

logger = logging.getLogger("suluv.agent")

_FENCE_RE = None  # lazy-compiled


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON.

    Many LLMs (Gemini, Llama, …) return JSON in ` ```json\n{...}\n``` `.
    This helper extracts the inner content so ``json.loads`` works.
    """
    import re
    global _FENCE_RE
    if _FENCE_RE is None:
        _FENCE_RE = re.compile(
            r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$",
            re.DOTALL,
        )
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text.strip()

# ReAct prompt template
REACT_SYSTEM = """You are a ReAct agent. You reason step by step.

On each step, output EXACTLY one of:

1. A thought and action (to call a tool):
```json
{{"thought": "...", "action": "tool_name", "action_input": {{...}}}}
```

2. A final answer (when you are done):
```json
{{"thought": "...", "final_answer": "..."}}
```

Available tools:
{tool_descriptions}

Rules:
- Always include "thought" explaining your reasoning.
- Tool action_input must match the tool's parameter schema.
- When you have enough information, provide a final_answer.
- Do NOT call tools unnecessarily.
"""


class SuluvAgent:
    """Standalone ReAct agent with tools, memory, guardrails, cost
    tracking, **threads**, and **checkpointing**.

    Threads
    -------
    Pass ``thread_id`` in ``AgentContext`` to enable conversation
    continuity.  The agent will:

    1. Load the existing thread from the *checkpointer* (if any).
    2. Prepend the full conversation history to the prompt so the LLM
       sees previous turns.
    3. Append the new user message + assistant reply to the thread.
    4. Save a checkpoint after the run completes.

    If no ``checkpointer`` is provided but ``thread_id`` is set, the
    agent keeps threads in a local dict (in-process memory) so threaded
    conversations still work within a single session.

    Usage::

        from suluv.core.adapters.memory_checkpointer import InMemoryCheckpointer

        cp = InMemoryCheckpointer()
        agent = SuluvAgent(
            role=AgentRole(name="Assistant"),
            llm=my_llm,
            tools=[calc],
            checkpointer=cp,
        )
        # First turn
        r1 = await agent.run("My name is Alice",
                             context=AgentContext(thread_id="t1"))
        # Second turn — Alice is remembered
        r2 = await agent.run("What's my name?",
                             context=AgentContext(thread_id="t1"))
    """

    def __init__(
        self,
        role: AgentRole,
        llm: LLMBackend,
        tools: list[SuluvTool] | None = None,
        memory: MemoryManager | None = None,
        guardrails: GuardrailChain | None = None,
        audit_backend: AuditBackend | None = None,
        cost_tracker: CostTracker | None = None,
        cancel_token: Any | None = None,
        checkpointer: Checkpointer | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.role = role
        self.llm = llm
        self.tools: dict[str, SuluvTool] = {t.name: t for t in (tools or [])}
        self.memory = memory
        self.guardrails = guardrails or GuardrailChain()
        self.audit = audit_backend
        self.cost_tracker = cost_tracker or CostTracker()
        self._cancel_token = cancel_token
        self._tool_runner = ToolRunner(audit_backend=audit_backend)
        self.checkpointer = checkpointer
        self.event_bus = event_bus
        # Fallback in-process thread cache (no persistence across restarts)
        self._local_threads: dict[str, Thread] = {}

    # ── Thread helpers ────────────────────────────────────────

    async def _publish_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an agent event to the event bus (if configured)."""
        if self.event_bus:
            payload = {"agent": self.role.name, "event_type": event_type, **data}
            try:
                await self.event_bus.publish(f"agent.{event_type}", payload)
            except Exception:
                logger.warning("Failed to publish agent event: %s", event_type)

    async def _load_thread(self, thread_id: str) -> Thread:
        """Load or create a thread."""
        thread: Thread | None = None
        if self.checkpointer:
            thread = await self.checkpointer.get(thread_id)
        else:
            thread = self._local_threads.get(thread_id)
        if thread is None:
            thread = Thread(thread_id=thread_id)
        return thread

    async def _save_thread(self, thread: Thread) -> None:
        """Persist a thread."""
        if self.checkpointer:
            await self.checkpointer.put(thread)
        else:
            self._local_threads[thread.thread_id] = thread

    async def get_thread(self, thread_id: str) -> Thread | None:
        """Public API: retrieve a thread by ID (or None)."""
        if self.checkpointer:
            return await self.checkpointer.get(thread_id)
        return self._local_threads.get(thread_id)

    async def list_threads(self, limit: int = 50) -> list[Thread]:
        """Public API: list recent threads."""
        if self.checkpointer:
            return await self.checkpointer.list(limit=limit)
        threads = sorted(
            self._local_threads.values(),
            key=lambda t: t.updated_at,
            reverse=True,
        )
        return threads[:limit]

    async def delete_thread(self, thread_id: str) -> None:
        """Public API: delete a thread."""
        if self.checkpointer:
            await self.checkpointer.delete(thread_id)
        else:
            self._local_threads.pop(thread_id, None)

    # ── Main run ──────────────────────────────────────────────

    async def run(
        self,
        task: str,
        context: AgentContext | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Execute the ReAct loop on a task and return the result."""
        ctx = context or AgentContext()
        started = datetime.now(timezone.utc)
        result = AgentResult(started_at=started)

        # ── Build identity context for subsystems ──────────────
        _ctx_dict = {
            "thread_id": ctx.thread_id,
            "user_id": ctx.user_id,
            "org_id": ctx.org_id,
            "session_id": ctx.session_id,
            "execution_id": ctx.execution_id,
        }

        # ── Guardrail input check ─────────────────────────────
        gr = await self.guardrails.check_input(task, context=_ctx_dict)
        if gr.action == GuardrailAction.BLOCK:
            result.success = False
            result.error = f"Input blocked: {gr.message}"
            result.completed_at = datetime.now(timezone.utc)
            return result

        # ── Thread: load conversation history ──────────────────
        thread: Thread | None = None
        if ctx.thread_id:
            thread = await self._load_thread(ctx.thread_id)
            thread.metadata.update({
                k: v for k, v in {
                    "user_id": ctx.user_id,
                    "org_id": ctx.org_id,
                    "agent": self.role.name,
                }.items() if v is not None
            })

        # ── Load memory context ────────────────────────────────
        memory_context = ""
        if self.memory:
            mem = await self.memory.load_context(
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                query=task,
                thread_id=ctx.thread_id,
            )
            if mem:
                parts = []
                for tier, entries in mem.items():
                    if not entries:
                        continue
                    if isinstance(entries, dict):
                        texts = [str(v) for v in list(entries.values())[:5]]
                    elif isinstance(entries, list):
                        texts = []
                        for e in entries[:5]:
                            if hasattr(e, "value"):
                                texts.append(str(e.value))
                            elif isinstance(e, dict):
                                texts.append(str(e))
                            else:
                                texts.append(str(e))
                    else:
                        continue
                    if texts:
                        parts.append(f"[{tier}]: " + "; ".join(texts))
                if parts:
                    memory_context = "\n\nRelevant memory:\n" + "\n".join(parts)

        # ── Build system prompt ────────────────────────────────
        tool_descs = "\n".join(
            f"- {t.name}: {t.description} | params: {json.dumps(t.parameters)}"
            for t in self.tools.values()
        )
        system_text = (
            self.role.to_system_prompt()
            + "\n\n"
            + REACT_SYSTEM.format(tool_descriptions=tool_descs)
        )
        if memory_context:
            system_text += memory_context

        # ── Assemble messages ──────────────────────────────────
        messages: list[SuluvMessage] = [SuluvMessage.system(system_text)]

        # Replay thread history (previous turns)
        if thread and thread.messages:
            messages.extend(thread.messages)

        # Append current user message
        user_msg = SuluvMessage.user(task)
        messages.append(user_msg)

        # Track new messages added during this run (for thread)
        new_messages: list[SuluvMessage] = [user_msg]

        # ── ReAct loop ────────────────────────────────────────
        for step_num in range(1, self.role.max_steps + 1):
            # Check cancellation
            if self._cancel_token and self._cancel_token.is_cancelled:
                result.answer = "Cancelled"
                result.success = False
                result.error = "Agent cancelled"
                break

            prompt = SuluvPrompt(
                messages=messages,
                temperature=self.role.temperature,
                max_tokens=self.role.max_tokens_per_step,
                response_format="json_object" if output_schema else None,
            )

            if output_schema:
                prompt.output_schema = output_schema

            llm_response = await self.llm.complete(prompt)
            response_text = llm_response.text

            # Track cost
            cost = CostRecord(
                input_tokens=llm_response.input_tokens,
                output_tokens=llm_response.output_tokens,
                total_tokens=llm_response.total_tokens,
                cost_usd=0.0,
                model=llm_response.model,
                thread_id=ctx.thread_id,
            )
            self.cost_tracker.record(cost)
            result.total_tokens += llm_response.total_tokens

            # Parse response
            step = StepRecord(step=step_num, tokens_used=llm_response.total_tokens)

            # Strip markdown code fences (```json ... ```) — some LLMs
            # (e.g. Gemini) wrap JSON output in fences.
            cleaned = _strip_code_fences(response_text)

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                step.thought = response_text
                result.steps.append(step)
                result.answer = response_text
                # Record assistant reply for thread
                assistant_msg = SuluvMessage.assistant(response_text)
                messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                break

            step.thought = parsed.get("thought", "")

            # Publish thought event
            if step.thought:
                await self._publish_event("thought", {
                    "step": step_num, "content": step.thought,
                })

            # Check for final answer
            if "final_answer" in parsed:
                step.action = "final_answer"
                result.steps.append(step)
                result.answer = parsed["final_answer"]
                await self._publish_event("answer", {
                    "step": step_num, "content": result.answer,
                })
                if output_schema:
                    result.structured = parsed.get("structured", parsed.get("final_answer"))
                # Record assistant reply for thread
                assistant_msg = SuluvMessage.assistant(response_text)
                messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                break

            # Execute tool action
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", {})
            step.action = action
            step.action_input = action_input

            # Publish action event
            if action:
                await self._publish_event("action", {
                    "step": step_num, "tool": action,
                    "input": action_input,
                })

            if action and action in self.tools:
                tool = self.tools[action]
                tool_result = await self._tool_runner.run(
                    tool,
                    action_input if isinstance(action_input, dict) else {},
                    context=_ctx_dict,
                )
                observation = tool_result.get("result") or tool_result.get("error", "No result")
                step.observation = observation

                # Publish observation event
                await self._publish_event("observation", {
                    "step": step_num, "tool": action,
                    "content": str(observation)[:500],
                })

                assistant_msg = SuluvMessage.assistant(response_text)
                tool_msg = SuluvMessage(
                    role=MessageRole.TOOL,
                    content=[
                        ContentBlock.text_block(
                            json.dumps(observation, default=str)
                        )
                    ],
                )
                messages.append(assistant_msg)
                messages.append(tool_msg)
                new_messages.extend([assistant_msg, tool_msg])
            elif action:
                step.error = f"Unknown tool: {action}"
                step.observation = f"Error: tool '{action}' not found"
                assistant_msg = SuluvMessage.assistant(response_text)
                tool_msg = SuluvMessage(
                    role=MessageRole.TOOL,
                    content=[ContentBlock.text_block(step.observation)],
                )
                messages.append(assistant_msg)
                messages.append(tool_msg)
                new_messages.extend([assistant_msg, tool_msg])
            else:
                result.answer = step.thought
                result.steps.append(step)
                assistant_msg = SuluvMessage.assistant(response_text)
                messages.append(assistant_msg)
                new_messages.append(assistant_msg)
                break

            result.steps.append(step)
        else:
            result.error = f"Max steps ({self.role.max_steps}) reached without final answer"
            result.success = False

        # ── Guardrail output check ─────────────────────────────
        if result.answer:
            gr = await self.guardrails.check_output(result.answer, context=_ctx_dict)
            if gr.action == GuardrailAction.BLOCK:
                result.answer = "[output blocked by guardrail]"
                result.success = False
                result.error = f"Output blocked: {gr.message}"

        # ── Thread: save conversation + checkpoint ─────────────
        if thread is not None:
            for msg in new_messages:
                thread.append_message(msg)
            thread.add_checkpoint(
                messages=thread.messages,
                step=len(thread.checkpoints) + 1,
                metadata={
                    "task": task[:200],
                    "answer": (result.answer or "")[:200],
                    "success": result.success,
                    "tokens": result.total_tokens,
                },
            )
            await self._save_thread(thread)
            result.metadata["thread_id"] = thread.thread_id
            result.metadata["checkpoint_count"] = thread.checkpoint_count

        # ── Save to memory ────────────────────────────────────
        if self.memory:
            await self.memory.save_interaction(
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                content=f"Q: {task}\nA: {result.answer}",
                thread_id=ctx.thread_id,
            )

        # ── Audit ─────────────────────────────────────────────
        if self.audit:
            await self.audit.write(AuditEvent(
                event_type="agent_run",
                org_id=ctx.org_id,
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                thread_id=ctx.thread_id,
                data={
                    "role": self.role.name,
                    "task": task[:200],
                    "steps": result.step_count,
                    "tokens": result.total_tokens,
                    "success": result.success,
                },
            ))

        result.cost_usd = self.cost_tracker.total_cost_usd
        result.completed_at = datetime.now(timezone.utc)
        return result

    async def run_stream(
        self,
        task: str,
        context: AgentContext | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent execution events.

        Yields dicts with type: "thought", "action", "observation", "answer", "error".
        """
        result = await self.run(task, context)
        for step in result.steps:
            if step.thought:
                yield {"type": "thought", "step": step.step, "content": step.thought}
            if step.action and step.action != "final_answer":
                yield {"type": "action", "step": step.step, "tool": step.action, "input": step.action_input}
            if step.observation is not None:
                yield {"type": "observation", "step": step.step, "content": step.observation}
        if result.answer:
            yield {"type": "answer", "content": result.answer}
        if result.error:
            yield {"type": "error", "content": result.error}
