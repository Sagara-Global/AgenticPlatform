# SuluvAI

**Agentic Business Process Framework** — pure Python, zero required dependencies, hexagonal architecture.

Use only the level of complexity you need:

```
Level 1: SuluvAgent          ← standalone agent + LLM + tools. No graph needed.
Level 2: GraphRuntime        ← multi-agent orchestration with nodes + edges
Level 3: ProcessDefinition   ← full business process management (BPM)
```

## Installation

```bash
pip install suluvai
```

With LLM backends:

```bash
pip install suluvai[openai]      # OpenAI
pip install suluvai[anthropic]   # Anthropic
pip install suluvai[gemini]      # Google Gemini
pip install suluvai[llm]         # All LLM backends
```

## Quick Start

### Level 1 — Standalone Agent

```python
from suluv.core import SuluvAgent, AgentRole, suluv_tool

@suluv_tool
async def check_pan(pan: str) -> dict:
    """Verify a PAN number."""
    return {"valid": True, "name": "Ramesh Kumar"}

agent = SuluvAgent(
    role=AgentRole(name="kyc-officer"),
    llm=OpenAIBackend(model="gpt-4o"),
    tools=[check_pan],
)
result = await agent.run("Check PAN ABCDE1234F")
```

### Level 2 — Multi-Agent Graph

```python
kyc_node = AgentNode(agent=kyc_agent)
credit_node = AgentNode(agent=credit_agent)

graph = GraphDefinition()
graph.add_node(kyc_node)
graph.add_node(credit_node)
graph.add_edge(kyc_node, credit_node, condition=lambda r: r.success)

runtime = GraphRuntime(event_bus=InMemoryEventBus())
result = await runtime.execute(graph, input="Process loan", context=ctx)
```

### Level 3 — Business Process

```python
process = ProcessDefinition(name="nbfc-loan", version="1.0")
process.add_variable("customer_id", type=str, required=True, immutable=True)
process.add_variable("loan_amount", type=float, required=True)

process.add_decision_table("eligibility", DecisionTable(
    inputs=["income", "cibil_score", "age"],
    rules=[
        Rule(when={"income": ">500000", "cibil_score": ">700"}, then="AUTO_APPROVE"),
        Rule(when={"income": ">300000", "cibil_score": ">650"}, then="MANUAL_REVIEW"),
        Rule(default=True, then="REJECT"),
    ],
))

process.add_stage(ProcessStage(
    name="kyc", agent=kyc_agent,
    sla=SLA(duration=4, unit="business_hours", calendar=india_calendar),
    compensation=reverse_kyc_hold,
))

result = await process.run(
    input={"customer_id": "C123", "loan_amount": 500000},
    context=ctx,
)
```

## The 3 Pillars

| Pillar | What It Does | Key Features |
|---|---|---|
| **Agent System** | AI agents that reason, use tools, and follow policies | ReAct loop, tool ownership, guardrails, 4-tier memory, cost tracking, structured output |
| **Graph Engine** | Orchestrates agents and nodes in directed graphs | 16 node types, fan-out/join, middleware, retry/skip/fallback, cancellation, streaming |
| **Process Engine** | Full business process management | Decision tables, forms, SLAs with business calendars, saga compensation, signals, correlation, work assignment, analytics |

**Process defines WHAT to do → Graph Engine decides HOW to run it → Agents do the WORK.**

## Architecture

### Hexagonal / Ports & Adapters

18 port ABCs with pluggable adapters — swap implementations without changing business logic:

- **LLMBackend** — OpenAI, Anthropic, Gemini (or bring your own)
- **EventBus** — publish/subscribe for node coordination
- **StateStore** — persist execution state for resume
- **AuditBackend** — compliance and audit trail
- **MemoryBackend** — short-term, long-term, episodic, semantic
- **HumanTaskQueue** — claim/release/delegate for human-in-the-loop
- **RulesEngine** — decision tables and scoring matrices
- **BusinessCalendar** — working hours, holidays, timezone-aware SLAs
- And 10 more...

All ports ship with **in-memory adapters** — no external infrastructure needed to get started.

### 16 Graph Node Types

| Node | Purpose |
|---|---|
| `AgentNode` | AI reasoning with tools |
| `ToolNode` | Direct function execution |
| `HumanNode` | Human decision / approval |
| `RouterNode` | Conditional branching |
| `LoopNode` | Iterative refinement |
| `MapNode` | Parallel for-each |
| `GatewayNode` | N-of-M join (e.g., 2-of-3 approval) |
| `DelayNode` | Timed wait |
| `SubgraphNode` | Nested graph composition |
| `TriggerNode` | Webhook / cron / event initiation |
| `DecisionNode` | Business rules evaluation |
| `FormNode` | Structured data collection |
| `SignalNode` | Business signal catch/throw |
| `CompensationNode` | Saga rollback |
| `TimerNode` | Calendar-aware scheduled wait |
| `ProcessNode` | Embedded business workflow |

### Process Engine Features

- **Decision Tables** — DMN-style rules, updatable without redeployment
- **Forms** — field types, conditional visibility, validation, sections
- **SLAs** — measured in business hours with escalation chains
- **Compensation** — saga pattern with reverse-order rollback
- **Signals** — react to external events (fraud alerts, document uploads)
- **Correlation** — route events to the correct process instance
- **Work Assignment** — round-robin, least-loaded, manual claim, skill-based
- **Analytics** — cycle time, SLA compliance, bottleneck detection

## Requirements

- Python 3.11+
- No required dependencies (LLM backends are optional extras)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2026 SagaraGlobal.

If you fork or build on this project, you must include the following attribution in your source code, documentation, or user-facing materials:

> "Built on SuluvAI Framework — https://github.com/sagaraglobal/suluvai"

This is required by the [NOTICE](NOTICE) file under Section 4(d) of the Apache 2.0 License.

Developed by [SagaraGlobal](https://sagaraglobal.com)
