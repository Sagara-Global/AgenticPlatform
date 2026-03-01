"""Example — Graph Engine with real-time AI streaming.

Demonstrates a multi-agent pipeline with FULL observability:

  - **EventBus streaming**: Real-time graph events (NodeStarted, NodeCompleted, …)
  - **AI streaming**: Agent thoughts, tool calls, and observations as they happen
  - Middleware (audit + logging + cost), event tracing

Pipeline::

    Researcher (Gemini + web search)
        → Extractor (ToolNode)
        → Analyst (Gemini + calculator)
        → Formatter (ToolNode)

Run::

    python examples/graph_engine.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

# ── Engine core ────────────────────────────────────────────────────────────
from suluv.core.engine.graph import GraphDefinition
from suluv.core.engine.node import GraphNode, NodeInput, NodeOutput
from suluv.core.engine.runtime import GraphRuntime
from suluv.core.engine.tool_node import ToolNode
from suluv.core.engine.events import (
    NodeStarted,
    NodeCompleted,
    NodeFailed,
    GraphCompleted,
)
from suluv.core.engine.middleware import (
    AuditMiddleware,
    LogMiddleware,
    CostMiddleware,
)

# ── Agent / LLM ───────────────────────────────────────────────────────────
from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.agent_node import AgentNode
from suluv.core.agent.role import AgentRole
from suluv.core.agent.cost_tracker import CostTracker
from suluv.core.adapters.gemini_llm import GeminiBackend
from suluv.core.adapters.memory_checkpointer import InMemoryCheckpointer
from suluv.core.tools.builtins import web_search, http_fetch, calculator, datetime_now

# ── Adapters ───────────────────────────────────────────────────────────────
from suluv.core.adapters.memory_bus import InMemoryEventBus
from suluv.core.adapters.memory_audit import InMemoryAuditBackend
from suluv.core.types import NodeType, NodeID


# ═══════════════════════════════════════════════════════════════════════════
#  ANSI helpers for coloured terminal output
# ═══════════════════════════════════════════════════════════════════════════

class C:
    """ANSI colour constants."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;236m"


def _ts() -> str:
    """Short timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ═══════════════════════════════════════════════════════════════════════════
#  EventBus streaming handlers — called in real-time during execution
# ═══════════════════════════════════════════════════════════════════════════

async def on_node_started(event: Any) -> None:
    nid = getattr(event, "node_id", "?")
    ntype = getattr(event, "node_type", "")
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.CYAN}{C.BOLD}▶ NODE STARTED{C.RESET}  "
        f"{C.WHITE}{nid}{C.RESET}  "
        f"{C.DIM}({ntype}){C.RESET}"
    )


async def on_node_completed(event: Any) -> None:
    nid = getattr(event, "node_id", "?")
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.GREEN}{C.BOLD}✔ NODE COMPLETED{C.RESET}  "
        f"{C.WHITE}{nid}{C.RESET}"
    )


async def on_node_failed(event: Any) -> None:
    nid = getattr(event, "node_id", "?")
    err = getattr(event, "error", "")
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.RED}{C.BOLD}✘ NODE FAILED{C.RESET}  "
        f"{C.WHITE}{nid}{C.RESET}  "
        f"{C.RED}{err[:80]}{C.RESET}"
    )


async def on_graph_completed(event: Any) -> None:
    r = getattr(event, "result", None)
    ok = getattr(r, "success", "?") if r else "?"
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.GREEN}{C.BOLD}◼ GRAPH COMPLETED{C.RESET}  "
        f"success={ok}"
    )


# ── Agent-level streaming handlers ────────────────────────────────────────

async def on_agent_thought(event: dict) -> None:
    agent = event.get("agent", "?")
    step = event.get("step", "?")
    content = event.get("content", "")
    display = content[:200] + "…" if len(content) > 200 else content
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.YELLOW}  💭 [{agent} step {step}]{C.RESET}  "
        f"{C.DIM}{display}{C.RESET}"
    )


async def on_agent_action(event: dict) -> None:
    agent = event.get("agent", "?")
    step = event.get("step", "?")
    tool = event.get("tool", "?")
    inp = event.get("input", {})
    inp_str = json.dumps(inp, default=str)[:120]
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.MAGENTA}  🔧 [{agent} step {step}]{C.RESET}  "
        f"{C.BOLD}{tool}{C.RESET}({C.DIM}{inp_str}{C.RESET})"
    )


async def on_agent_observation(event: dict) -> None:
    agent = event.get("agent", "?")
    step = event.get("step", "?")
    tool = event.get("tool", "?")
    content = event.get("content", "")
    display = content[:150] + "…" if len(content) > 150 else content
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.BLUE}  👁 [{agent} step {step}]{C.RESET}  "
        f"{C.DIM}{tool} → {display}{C.RESET}"
    )


async def on_agent_answer(event: dict) -> None:
    agent = event.get("agent", "?")
    content = event.get("content", "")
    display = content[:200] + "…" if len(content) > 200 else content
    print(
        f"  {C.DIM}{_ts()}{C.RESET}  "
        f"{C.GREEN}  ✅ [{agent}]{C.RESET}  "
        f"{C.WHITE}{display}{C.RESET}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Build real agents (with event_bus for real-time streaming)
# ═══════════════════════════════════════════════════════════════════════════

def make_researcher_agent(
    llm: GeminiBackend,
    cost_tracker: CostTracker,
    event_bus: InMemoryEventBus,
) -> SuluvAgent:
    """Agent that researches a company using web search."""
    role = AgentRole(
        name="Researcher",
        description="An expert web researcher who finds company information.",
        capabilities=["search the web", "fetch web pages", "find company details"],
        instructions=(
            "Search the web to gather information about the company. "
            "Find: what the company does, key products/services, "
            "leadership, and recent news. "
            "Return a concise summary of your findings."
        ),
        max_steps=10,
        temperature=0.2,
    )
    return SuluvAgent(
        role=role,
        llm=llm,
        tools=[web_search, http_fetch, datetime_now],
        checkpointer=InMemoryCheckpointer(),
        cost_tracker=cost_tracker,
        event_bus=event_bus,
    )


def make_analyst_agent(
    llm: GeminiBackend,
    cost_tracker: CostTracker,
    event_bus: InMemoryEventBus,
) -> SuluvAgent:
    """Agent that analyses research and produces risk assessment."""
    role = AgentRole(
        name="Analyst",
        description="A business analyst who identifies risks and opportunities.",
        capabilities=["analyse data", "assess risk", "calculate metrics"],
        instructions=(
            "Given company research, produce a concise risk analysis. "
            "Identify the top 3-5 risks and rate each as HIGH/MEDIUM/LOW. "
            "Be specific and cite data from the research. "
            "Format as a numbered list."
        ),
        max_steps=5,
        temperature=0.1,
    )
    return SuluvAgent(
        role=role,
        llm=llm,
        tools=[calculator],
        checkpointer=InMemoryCheckpointer(),
        cost_tracker=cost_tracker,
        event_bus=event_bus,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Build the graph
# ═══════════════════════════════════════════════════════════════════════════

def build_research_pipeline(
    llm: GeminiBackend,
    cost_tracker: CostTracker,
    event_bus: InMemoryEventBus,
) -> GraphDefinition:
    """Build a multi-agent research pipeline.

    Graph::

        [company name]
             │
             ▼
        ┌──────────┐
        │ Researcher│  (Gemini agent — web search)
        └────┬─────┘
             │ research findings
             ▼
        ┌──────────┐
        │ Extractor │  (ToolNode — pull key facts)
        └────┬─────┘
             │ structured facts
             ▼
        ┌──────────┐
        │  Analyst  │  (Gemini agent — risk assessment)
        └────┬─────┘
             │ risk analysis
             ▼
        ┌──────────┐
        │ Formatter │  (ToolNode — final report)
        └──────────┘
    """
    graph = GraphDefinition(name="company-research-pipeline")

    # ── Node 1: Researcher agent ──────────────────────────────────────
    researcher = make_researcher_agent(llm, cost_tracker, event_bus)
    n_research = AgentNode(
        agent=researcher,
        node_id="researcher",
        name="Researcher",
    )

    # ── Node 2: Extract key facts (pure function) ─────────────────────
    def extract_facts(data: Any, ctx: dict) -> dict:
        """Pull the answer from the agent output and package it."""
        answer = data.get("answer", "") if isinstance(data, dict) else str(data)
        return {
            "research": answer,
            "task": f"Analyse the following company research and identify the top risks:\n\n{answer}",
        }

    n_extract = ToolNode(
        extract_facts,
        node_id="extractor",
        name="Extractor",
    )

    # ── Node 3: Analyst agent ─────────────────────────────────────────
    analyst = make_analyst_agent(llm, cost_tracker, event_bus)
    n_analyse = AgentNode(
        agent=analyst,
        node_id="analyst",
        name="Analyst",
    )

    # ── Node 4: Format final report (pure function) ───────────────────
    def format_final(data: Any, ctx: dict) -> str:
        """Combine research + analysis into a final report."""
        analysis = data.get("answer", "") if isinstance(data, dict) else str(data)
        return (
            f"\n{'='*60}\n"
            f"  FINAL COMPANY RESEARCH REPORT\n"
            f"{'='*60}\n\n"
            f"RISK ANALYSIS:\n{analysis}\n"
            f"{'='*60}"
        )

    n_format = ToolNode(
        format_final,
        node_id="formatter",
        name="Formatter",
    )

    # ── Wire the graph ────────────────────────────────────────────────
    for node in [n_research, n_extract, n_analyse, n_format]:
        graph.add_node(node)

    graph.add_edge(n_research, n_extract)
    graph.add_edge(n_extract, n_analyse)
    graph.add_edge(n_analyse, n_format)

    graph.set_entry(n_research)
    graph.set_exit(n_format)

    return graph


# ═══════════════════════════════════════════════════════════════════════════
#  Main — run with real-time streaming
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY environment variable (or add it to .env)")

    # ── Setup ─────────────────────────────────────────────────────────
    llm = GeminiBackend(
        model="gemini-2.0-flash",
        api_key=api_key,
        default_temperature=0.2,
    )
    cost_tracker = CostTracker()
    audit = InMemoryAuditBackend()
    event_bus = InMemoryEventBus()
    log_mw = LogMiddleware()
    cost_mw = CostMiddleware()
    audit_mw = AuditMiddleware(audit)

    # ── Subscribe EventBus handlers for real-time streaming ───────────
    # Graph-level events
    await event_bus.subscribe("NodeStarted", on_node_started)
    await event_bus.subscribe("NodeCompleted", on_node_completed)
    await event_bus.subscribe("NodeFailed", on_node_failed)
    await event_bus.subscribe("GraphCompleted", on_graph_completed)

    # Agent-level events (thoughts, tool calls, observations)
    await event_bus.subscribe("agent.thought", on_agent_thought)
    await event_bus.subscribe("agent.action", on_agent_action)
    await event_bus.subscribe("agent.observation", on_agent_observation)
    await event_bus.subscribe("agent.answer", on_agent_answer)

    # ── Build graph ───────────────────────────────────────────────────
    graph = build_research_pipeline(llm, cost_tracker, event_bus)

    runtime = GraphRuntime(
        graph,
        middlewares=[log_mw, cost_mw, audit_mw],
        event_bus=event_bus,
        context={
            "org_id": "suluv",
            "user_id": "analyst",
            "thread_id": "pipeline-stream-1",
        },
    )

    # ── Execute with streaming ────────────────────────────────────────
    company = "SagaraGlobal"
    print(f"\n{C.BOLD}{'═'*60}{C.RESET}")
    print(f"  {C.BOLD}Multi-Agent Research Pipeline — STREAMING MODE{C.RESET}")
    print(f"  Company: {C.CYAN}{company}{C.RESET}")
    print(f"{C.BOLD}{'═'*60}{C.RESET}")
    print(f"  Pipeline: Researcher → Extractor → Analyst → Formatter")
    print(f"  Streaming: EventBus (graph) + AI (thoughts/tools/observations)")
    print(f"{C.DIM}{'─'*60}{C.RESET}\n")

    # Use runtime.stream() for graph-level event iteration
    final_result = None
    event_count = 0

    async for event in runtime.stream(
        input_data=f"Research the company '{company}'"
    ):
        event_count += 1
        # Graph events are already handled by EventBus subscribers above.
        # Capture the final result from GraphCompleted.
        if isinstance(event, GraphCompleted):
            final_result = event.result

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{C.BOLD}{'═'*60}{C.RESET}")
    print(f"  {C.BOLD}EXECUTION SUMMARY{C.RESET}")
    print(f"{C.BOLD}{'═'*60}{C.RESET}")

    if final_result:
        print(f"  Success:        {C.GREEN if final_result.success else C.RED}{final_result.success}{C.RESET}")
        print(f"  Nodes executed: {len(final_result.trace)}")
    print(f"  Graph events:   {event_count}")
    print(f"  Cost tracker:   {cost_tracker.to_dict()}")

    # ── Execution trace ───────────────────────────────────────────────
    print(f"\n{C.DIM}{'─'*60}{C.RESET}")
    print(f"  {C.BOLD}EXECUTION TRACE:{C.RESET}")
    if final_result:
        for rec in final_result.trace:
            tokens = rec.output.get("tokens", 0) if isinstance(rec.output, dict) else "-"
            state_colour = C.GREEN if rec.state.value == "done" else C.RED
            print(
                f"    {rec.node_id:15s} │ "
                f"{state_colour}{rec.state.value:8s}{C.RESET} │ "
                f"tokens={tokens}"
            )

    # ── Final output ──────────────────────────────────────────────────
    print(f"\n{C.DIM}{'─'*60}{C.RESET}")
    print(f"  {C.BOLD}FINAL REPORT:{C.RESET}")
    if final_result:
        output = final_result.output
        if isinstance(output, str):
            print(output)
        elif isinstance(output, dict):
            print(output.get("answer", output))
        else:
            print(output)

    # ── Middleware logs ────────────────────────────────────────────────
    print(f"\n{C.DIM}{'─'*60}{C.RESET}")
    print(f"  Middleware logs: {len(log_mw.logs)}")
    for log in log_mw.logs:
        print(f"    {C.DIM}{log}{C.RESET}")

    # ── Audit events ──────────────────────────────────────────────────
    print(f"\n{C.DIM}{'─'*60}{C.RESET}")
    print(f"  Audit events: {len(audit.events)}")
    for e in audit.events:
        print(f"    {C.DIM}[{e.event_type}] node={e.node_id} thread={e.thread_id}{C.RESET}")

    # ── EventBus full history ─────────────────────────────────────────
    print(f"\n{C.DIM}{'─'*60}{C.RESET}")
    print(f"  EventBus total publishes: {C.BOLD}{len(event_bus.history)}{C.RESET}")
    # Group by topic
    topics: dict[str, int] = {}
    for topic, _ in event_bus.history:
        topics[topic] = topics.get(topic, 0) + 1
    for topic, count in sorted(topics.items()):
        print(f"    {topic:25s}  ×{count}")

    print(f"\n{C.BOLD}{'═'*60}{C.RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
