# SuluvAI 3.0

**Complete rewrite** — Agentic Business Process Framework built from scratch.

> **Breaking change from v2.x**: SuluvAI 3.0 is a ground-up rewrite. No LangGraph dependency. Pure Python, hexagonal architecture, zero required dependencies.

## What's New in 3.0

- **No LangGraph/LangChain dependency** — pure Python framework with pluggable LLM backends
- **3 levels of complexity** — use only what you need:
  - `SuluvAgent` — standalone agent + LLM + tools (no graph needed)
  - `GraphRuntime` — multi-agent orchestration with 16 node types
  - `ProcessDefinition` — full business process management (BPM)
- **Hexagonal architecture** — 18 port ABCs with pluggable adapters
- **Process Engine** — decision tables, forms, SLAs, saga compensation, signals, correlation
- **Graph Engine** — fan-out/join, middleware, cancellation, streaming, retry policies

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
process.add_variable("customer_id", type=str, required=True)
process.add_stage(ProcessStage(name="kyc", agent=kyc_agent))
result = await process.run(input={"customer_id": "C123"}, context=ctx)
```

## Installation

```bash
pip install suluvai
```

With LLM backends:

```bash
pip install suluvai[openai]      # OpenAI
pip install suluvai[anthropic]   # Anthropic
pip install suluvai[llm]         # All LLM backends
```

## Architecture

- **Graph Engine** — 16 node types, edges with conditions, fan-out/join, middleware, cancellation, streaming
- **Agent System** — ReAct loop, tool ownership, guardrails, policy rules, memory (4 tiers), cost tracking
- **Process Engine** — Decision tables, forms, SLAs with business calendars, saga compensation, signals, correlation, work assignment, analytics
- **Hexagonal/Ports & Adapters** — 18 port ABCs with pluggable in-memory adapters shipped in core

## Migration from v2.x

SuluvAI 3.0 is a complete rewrite and is **not backward compatible** with v2.x. The API, architecture, and dependencies are entirely different. See the [architecture docs](https://github.com/sagaraglobal/suluvai) for the full design.

## License

MIT

Developed by [SagaraGlobal](https://sagaraglobal.com)
