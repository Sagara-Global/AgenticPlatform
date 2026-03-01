# Suluv — Complete Architecture & Phased Build Plan

## Core Principle: 3 Levels of Complexity (use only what you need)

```
Level 1: SuluvAgent          ← just an agent + LLM + tools. No graph, no engine.
Level 2: GraphRuntime        ← multi-agent orchestration with nodes + edges
Level 3: ProcessDefinition   ← business workflows that compile to graphs
```

### Level 1 — Standalone Agent (simplest)
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

### Level 2 — Multi-Agent Graph (orchestration)
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

### Level 3 — Business Process (workflows)
```python
process = ProcessDefinition(name="nbfc-loan", version="1.0")

# Process-level variables — visible across all stages
process.add_variable("customer_id", type=str, required=True, immutable=True)
process.add_variable("loan_amount", type=float, required=True)

# Business rules — change without redeploying code
process.add_decision_table("eligibility", DecisionTable(
    inputs=["income", "cibil_score", "age"],
    rules=[
        Rule(when={"income": ">500000", "cibil_score": ">700"}, then="AUTO_APPROVE"),
        Rule(when={"income": ">300000", "cibil_score": ">650"}, then="MANUAL_REVIEW"),
        Rule(default=True, then="REJECT"),
    ],
))

# Stages with rich SLA, escalation, compensation, and forms
process.add_stage(ProcessStage(
    name="kyc", agent=kyc_agent,
    sla=SLA(duration=4, unit="business_hours", calendar=india_calendar,
            escalation=EscalationChain([
                Escalation(at="80%", action=notify("assignee")),
                Escalation(at="100%", action=escalate_to("manager")),
            ])),
    compensation=reverse_kyc_hold,  # saga rollback if later stages fail
))
process.add_stage(ProcessStage(
    name="approval",
    form=FormDefinition(fields=[
        Field("decision", type="select", options=["approve", "reject"]),
        Field("comments", type="text", required=True),
        Field("evidence", type="file_upload"),
    ]),
    assignment=WorkAssignment(role="credit-manager", strategy="round-robin"),
))

# Signals — react to external business events
process.on_signal("fraud_alert", action=suspend_instance)
process.correlate_on("customer_id")  # route events to correct instance

result = await process.run(
    input={"customer_id": "C123", "loan_amount": 500000},
    context=ctx,
)
```

---

## The 3 Pillars

```
┌─────────────────────────────────────────────────────────────────┐
│                     SULUV FRAMEWORK                             │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  GRAPH       │    │  AGENT       │    │  PROCESS     │      │
│  │  ENGINE      │    │  SYSTEM      │    │  ENGINE      │      │
│  │              │    │              │    │              │      │
│  │  Nodes       │    │  SuluvAgent  │    │  Stages      │      │
│  │  Edges       │    │  Tools ←own  │    │  Decisions   │      │
│  │  Runtime     │    │  ReAct loop  │    │  Forms/HITL  │      │
│  │  EventBus    │    │  Guardrails  │    │  SLA+Escal.  │      │
│  │  State       │    │  Policy      │    │  Saga/Comp.  │      │
│  │              │    │              │    │  Signals     │      │
│  │              │    │              │    │  Variables   │      │
│  │              │    │              │    │  Versioning  │      │
│  │              │    │              │    │  Assignment  │      │
│  │              │    │              │    │  Correlation │      │
│  │              │    │              │    │  Calendars   │      │
│  │              │    │              │    │  Analytics   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │            WORKS STANDALONE            │              │
│         │            (no graph needed)           │              │
│         │                   │                   │              │
│         └───────────────────┼───────────────────┘              │
│                             │                                  │
│                    ┌────────▼────────┐                         │
│                    │     PORTS       │                         │
│                    │  (ABCs / SPI)   │                         │
│                    │                 │                         │
│                    │  LLMBackend     │                         │
│                    │  EventBus       │                         │
│                    │  StateStore     │                         │
│                    │  AuditBackend   │                         │
│                    │  MemoryBackend  │                         │
│                    │  ...            │                         │
│                    └────────┬────────┘                         │
│                             │                                  │
│                    ┌────────▼────────┐                         │
│                    │    ADAPTERS     │                         │
│                    │  (Plug & Play)  │                         │
│                    │                 │                         │
│                    │  InMemoryBus    │                         │
│                    │  OpenAIBackend  │                         │
│                    │  RedisState     │                         │
│                    │  ...            │                         │
│                    └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### How the 3 pillars connect:

1. **Agent** is a self-contained unit — it has its own LLM, tools, guardrails, and policy. Works standalone (`agent.run(task)`).
2. **Graph Engine** is the orchestration backbone — `AgentNode` wraps an agent so the graph engine can dispatch it alongside ToolNodes, HumanNodes, etc.
3. **Process Engine** is a full business process management system — it models real-world workflows with decisions, forms, SLAs with escalation chains, compensation (saga pattern), versioning, work assignment, signals, and correlation. A `ProcessDefinition` compiles to a `GraphDefinition` and runs via `GraphRuntime`, but the process engine adds rich business semantics on top.

So: **Process defines WHAT to do → Graph Engine decides HOW to run it → Agents do the WORK**
And if you just need one agent? Skip the graph entirely.

---

## System Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SULUV FRAMEWORK                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        GRAPH ENGINE (core)                          │    │
│  │                                                                     │    │
│  │  GraphNode ◄──── AgentNode | ToolNode | HumanNode | RouterNode     │    │
│  │      │            LoopNode | MapNode | GatewayNode | DelayNode      │    │
│  │      │            SubgraphNode | ProcessNode | TriggerNode          │    │
│  │      │            DecisionNode | FormNode | SignalNode              │    │
│  │      │            CompensationNode | TimerNode                      │    │
│  │      │                                                              │    │
│  │  GraphEdge ────── condition(NodeOutput) → bool                      │    │
│  │      │            task_transform(NodeOutput) → NodeInput            │    │
│  │      │            error_policy: FAIL_FAST | RETRY | SKIP | FALLBACK│    │
│  │      │                                                              │    │
│  │  GraphDefinition ── nodes + edges + entry + exit                    │    │
│  │      │               to_dict() / from_dict() ← SERIALIZATION       │    │
│  │      │                                                              │    │
│  │  ExecutionState ─── per-node: PENDING|RUNNING|DONE|FAILED|WAITING  │    │
│  │      │              persisted via StateStore port after every node   │    │
│  │      │              cancel_token: CancellationToken ← CANCELLATION  │    │
│  │      │                                                              │    │
│  │  Middleware ──────── before_node / after_node hooks                  │    │
│  │      │               [CostMiddleware, AuditMiddleware, LogMiddleware]│    │
│  │      │                                                              │    │
│  │  GraphRuntime ────── compute frontier → dispatch via EventBus       │    │
│  │                      execute() → ExecutionResult                     │    │
│  │                      execute_stream() → AsyncIterator[GraphEvent]   │    │
│  │                      resume(execution_id) → ExecutionResult          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────┐  ┌─────────────────┐                              │
│  │    AGENT SYSTEM       │  │  TEST HARNESS   │                              │
│  │   (works standalone)  │  │                 │                              │
│  │                       │  │  MockLLM        │                              │
│  │  AgentRole            │  │  MockTools      │                              │
│  │  AgentContext         │  │  MockEventBus   │                              │
│  │  SuluvAgent (ReAct)   │  │  AgentHarness   │                              │
│  │    ├── llm            │  │  GraphHarness   │                              │
│  │    ├── tools ← OWNED  │  │  EvalSuite      │                              │
│  │    ├── memory         │  │  expect_call()  │                              │
│  │    ├── guardrails     │  │  assert_audit() │                              │
│  │    ├── policy_rules   │  └─────────────────┘                              │
│  │    └── cost_tracker   │                                                    │
│  │  AgentNode (wrapper)  │                                                    │
│  │  Structured Output    │                                                    │
│  └──────────────────────┘                                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      PROCESS ENGINE                                  │    │
│  │                                                                       │    │
│  │  Core Primitives:                                                     │    │
│  │  ProcessDefinition   ProcessVersion      ProcessVariables             │    │
│  │  ProcessStage        ProcessStep (ABC)    DecisionTable                │    │
│  │  FormDefinition      ScoringMatrix        Rule / HitPolicy             │    │
│  │                                                                       │    │
│  │  Workflow Management:                                                  │    │
│  │  SLAManager          EscalationChain      BusinessCalendar             │    │
│  │  WorkAssignment      DelegationRules      CompensationHandler          │    │
│  │  SagaConfig          CorrelationEngine    SignalHandler                │    │
│  │  BoundaryEvent       ProcessInstanceMgr   PolicyCheckpoint             │    │
│  │                                                                       │    │
│  │  Human Interaction:                                                    │    │
│  │  FormNode            FormSection          FieldType / FieldSchema       │    │
│  │  HumanTask           WorkQueue            Claim / Release / Delegate   │    │
│  │  ProcessComment      NoteAttachment       CommentVisibility            │    │
│  │                                                                       │    │
│  │  Observability:                                                        │    │
│  │  ProcessAnalytics    CycleTimeReport      SLAComplianceReport          │    │
│  │  BottleneckDetector  DropoffAnalysis       PerformanceComparison        │    │
│  │                                                                       │    │
│  │  ProcessDefinition → compile() → GraphDefinition → GraphRuntime        │    │
│  │  ProcessNode ← wraps a process as a GraphNode for embedding             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     PORTS (Abstract Base Classes)                    │    │
│  │                                                                     │    │
│  │  LLMBackend ─── complete(SuluvPrompt) → LLMResponse                │    │
│  │                 stream(SuluvPrompt) → AsyncIterator[str]            │    │
│  │                 embed(text) → list[float]                           │    │
│  │                                                                     │    │
│  │  EventBus ──── publish(topic, event) / subscribe(topic, handler)   │    │
│  │                request(topic, event, timeout) → response            │    │
│  │                                                                     │    │
│  │  StateStore ── save(id, state) / load(id) / delete(id)             │    │
│  │                                                                     │    │
│  │  AuditBackend ── write(AuditEvent) / query(filters) → list         │    │
│  │                                                                     │    │
│  │  MemoryBackend ── ShortTerm | LongTerm | Episodic | Semantic       │    │
│  │                                                                     │    │
│  │  GuardrailPort ── check_input(ctx, text) / check_output(ctx, text) │    │
│  │  PolicyRule ───── evaluate(ctx, action) → ALLOW/DENY/ESCALATE      │    │
│  │  ConsentProvider ─ check(ctx, purpose) → granted/denied             │    │
│  │  CorpusProvider ── search(query, ctx) → list[Chunk]                 │    │
│  │  ConnectorPort ─── send(request) → response (ext API gateway)      │    │
│  │  HumanTaskQueue ── emit / poll / claim / release / delegate / done  │    │
│  │  ArtifactStore ─── put(id, bytes) / get(id) → bytes                │    │
│  │  NotifierPort ──── notify(channel, message)                         │    │
│  │  VerificationPort ─ verify(identity) → valid/invalid                │    │
│  │                                                                     │    │
│  │  ─── Process Engine Ports (new) ───────────────────────────────  │    │
│  │  RulesEngine ───── evaluate(table, inputs) → Decision               │    │
│  │                    supports DecisionTable + ScoringMatrix           │    │
│  │  BusinessCalendar ─ is_working(dt) / add_biz_hours(dt, h) → dt      │    │
│  │                    working_hours, holidays, timezone               │    │
│  │  ProcessInstStore ─ save / load / query / list / delete instances   │    │
│  │  TemplateEngine ─── render(template, vars) → Document               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    MESSAGES (multimodal protocol)                    │    │
│  │                                                                     │    │
│  │  ContentBlock ── TEXT | IMAGE_URL | IMAGE_BASE64 | AUDIO_URL        │    │
│  │                  AUDIO_BASE64 | TOOL_CALL | TOOL_RESULT | DOCUMENT  │    │
│  │                                                                     │    │
│  │  SuluvMessage ── role: SYSTEM|USER|ASSISTANT|TOOL                   │    │
│  │                  content: list[ContentBlock]                         │    │
│  │                                                                     │    │
│  │  SuluvPrompt ─── messages: list[SuluvMessage]                       │    │
│  │                  tools: list[ToolSchema]                             │    │
│  │                  output_schema: dict | None  ← STRUCTURED OUTPUT    │    │
│  │                  temperature, max_tokens, response_format            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    IN-MEMORY ADAPTERS (ship with core)              │    │
│  │                                                                     │    │
│  │  InMemoryEventBus | InMemoryStateStore | InMemoryAuditBackend       │    │
│  │  InMemoryShortTerm | InMemoryLongTerm | InMemoryEpisodic            │    │
│  │  InMemoryHumanTaskQueue | InMemoryArtifactStore                     │    │
│  │  InMemoryRulesEngine | InMemoryBusinessCalendar                     │    │
│  │  InMemoryProcessInstanceStore | InMemoryTemplateEngine              │    │
│  │  MockLLM | MockConsentProvider                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    EXTERNAL ADAPTERS (separate packages)            │    │
│  │                                                                     │    │
│  │  suluv-lang: OpenAIBackend, AnthropicBackend, GeminiBackend         │    │
│  │  suluv-connectors: HttpConnector, ConnectorPipeline                 │    │
│  │  suluv-india: PAN, Aadhaar, GSTIN types + PIIGuardrail             │    │
│  │  suluv-cli: init, create, verify commands                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Node Types (16)

### Graph Engine Nodes (11)

| Type | LLM? | I/O? | Blocks? | Pattern |
|---|---|---|---|---|
| `AgentNode` | Yes | Tools | No | AI reasoning |
| `ToolNode` | No | Function | No | Direct execution (n8n-style) |
| `HumanNode` | No | TaskQueue | Yes (pause) | Human decision |
| `RouterNode` | No | No | No | Conditional branch (if/else) |
| `LoopNode` | Depends | Depends | No | Iterative refinement |
| `MapNode` | Depends | Depends | No | Parallel for-each over list |
| `GatewayNode` | No | No | Partial | N-of-M join (2-of-3 approval) |
| `DelayNode` | No | No | Yes (time) | Timed wait / cooldown |
| `SubgraphNode` | Depends | Depends | Depends | Nested graph composition |
| `ProcessNode` | Depends | Depends | Depends | Business workflow as node |
| `TriggerNode` | No | External | No | Graph initiation (webhook/cron/event) |

### Process Engine Nodes (5)

| Type | LLM? | I/O? | Blocks? | Pattern |
|---|---|---|---|---|
| `DecisionNode` | No | Rules | No | Business rules / decision table evaluation |
| `FormNode` | No | Human | Yes (pause) | Structured data collection with validation |
| `SignalNode` | No | Events | Yes (wait) | Catch/throw business signals (boundary events) |
| `CompensationNode` | No | Function | No | Saga rollback / undo step |
| `TimerNode` | No | Calendar | Yes (time) | Calendar-aware scheduled wait (business hours) |

### Node Type Details

**DecisionNode** — Evaluates a `DecisionTable` or `ScoringMatrix` against process variables. No LLM needed — pure business rules. Returns the matched decision (e.g., `AUTO_APPROVE`, `REJECT`, `MANUAL_REVIEW`). Decision tables are data, not code, so business users can update rules without redeployment.

**FormNode** — Presents a `FormDefinition` to a human via `HumanTaskQueue`. Includes field types, validations, conditional visibility, sections, and file uploads. Blocks until the form is submitted. Applies field-level and form-level validation before accepting. Integrates with `WorkAssignment` for role-based routing.

**SignalNode** — Listens for named business signals (e.g., `fraud_alert`, `document_received`). Can be **interrupting** (aborts current work and reroutes) or **non-interrupting** (injects data alongside running work). Used for boundary events on stages: "if X happens while this stage is running, do Y."

**CompensationNode** — Executes a compensation handler to reverse the effects of a previously completed stage. Used in the saga pattern: when stage N fails, compensation nodes for stages 1..N-1 fire in reverse order. Handlers are idempotent and retriable.

**TimerNode** — Like `DelayNode` but calculates wait times against a `BusinessCalendar`. "Wait 4 business hours" respects working hours (9am-6pm), holidays, and timezone. Powers SLA enforcement — the process engine injects timer nodes to track SLA deadlines and trigger escalation chains.

All 16 node types implement the `GraphNode` ABC. New types can be added without modifying core — framework users can create custom node types by implementing the ABC.

---

## Agent Component Spec

```
SuluvAgent — Component Ownership (standalone, no graph required)
├── role: AgentRole                     # name, capabilities, max_steps
├── llm: LLMBackend                     # any LLM provider
├── tools: list[SuluvTool]              # tools OWNED by this agent
├── memory: MemoryManager = None        # optional — wires ShortTerm/LongTerm/Episodic/Semantic
├── guardrails: list[Guardrail] = []    # optional safety filters
├── policy_rules: list[PolicyRule] = [] # optional business rules
├── audit_backend: AuditBackend = None  # optional audit logging
├── cost_tracker: CostTracker = None    # optional cost tracking
│
├── run(task) → AgentResult                        # simple — no context needed
├── run(task, context) → AgentResult               # with identity/session context
├── run(task, output_schema={...}) → AgentResult   # structured output → result.structured
└── run_stream(task) → AsyncIterator[Event]        # streaming output

MemoryManager — Wires 4 memory tiers to the agent
├── short_term: ShortTermMemory = None  # in-session (auto-cleared on session end)
├── long_term: LongTermMemory = None    # cross-session (user/org scoped)
├── episodic: EpisodicMemory = None     # past interaction recall
├── semantic: SemanticMemory = None     # vector similarity search
│
├── Agent reads memory at start of run()
├── Agent writes memory at end of run()
└── Memory survives across multiple agent.run() calls

AgentNode — Thin Graph Wrapper
├── agent: SuluvAgent                   # wraps an existing agent
├── node_id: str                        # unique in graph
├── node_type: NodeType.AGENT
└── execute(input, ctx) → NodeOutput    # called by GraphRuntime
```

---

## Data Flow — Single Request

```
User request
    │
    ▼
GraphRuntime.execute(graph_def, input, context)
    │
    ├── 1. Load/create ExecutionState (persisted)
    ├── 2. Check CancellationToken
    ├── 3. Compute frontier (nodes whose deps are satisfied)
    │
    ├── For each frontier node:
    │   ├── Middleware.before_node(node, input, ctx)
    │   │       ├── CostMiddleware: check budget
    │   │       ├── AuditMiddleware: log node_start
    │   │       └── custom middleware...
    │   │
    │   ├── EventBus.publish("node.execute", {node_id, input})
    │   │       │
    │   │       ▼
    │   │   NodeExecutor picks up event
    │   │       │
    │   │       ├── if AgentNode:
    │   │       │       SuluvAgent.run(task, context)
    │   │       │           ├── GuardrailChain.check_input()
    │   │       │           ├── ReAct loop:
    │   │       │           │   ├── LLM.complete(SuluvPrompt)    ← multimodal
    │   │       │           │   ├── parse action
    │   │       │           │   ├── PolicyEngine.evaluate()
    │   │       │           │   ├── agent.tools[name].execute()  ← tools owned by agent
    │   │       │           │   └── repeat until final_answer
    │   │       │           ├── GuardrailChain.check_output()
    │   │       │           ├── CostTracker.record(tokens)
    │   │       │           └── return AgentResult
    │   │       │
    │   │       ├── if ToolNode: run tool directly (no LLM)
    │   │       ├── if HumanNode: emit to HumanTaskQueue, WAIT
    │   │       ├── if RouterNode: evaluate conditions, no execution
    │   │       ├── if LoopNode: repeat body node until condition met (max N)
    │   │       ├── if MapNode: parallel for-each over list items
    │   │       ├── if GatewayNode: wait for N-of-M incoming nodes
    │   │       ├── if DelayNode: sleep for duration or until timestamp
    │   │       ├── if SubgraphNode: recursive GraphRuntime.execute()
    │   │       ├── if TriggerNode: already fired (webhook/cron/event started the graph)
    │   │       ├── if DecisionNode: evaluate DecisionTable via RulesEngine
    │   │       ├── if FormNode: emit FormDefinition to HumanTaskQueue, WAIT
    │   │       ├── if SignalNode: wait for named signal via CorrelationEngine
    │   │       ├── if CompensationNode: run compensation handler (saga rollback)
    │   │       └── if TimerNode: calendar-aware wait via BusinessCalendar
    │   │
    │   ├── EventBus.publish("node.complete", {node_id, output})
    │   ├── Middleware.after_node(node, output, ctx)
    │   │       ├── CostMiddleware: accumulate cost
    │   │       └── AuditMiddleware: log node_complete
    │   │
    │   └── Persist ExecutionState (node → DONE)
    │
    ├── Evaluate outgoing edges (conditions)
    │       ├── edge.condition(output) → True? add target to frontier
    │       ├── Multiple edges True? → parallel fan-out
    │       └── No edges True? → node is terminal
    │
    ├── Repeat until no more frontier nodes
    │
    └── Return ExecutionResult
            ├── output: final node outputs
            ├── execution_id: for resume
            ├── cost: total tokens + estimated USD
            └── trace: list[NodeExecution] for debugging
```

---

## Error Flow

```
Node execution fails
    │
    ├── Check edge.error_policy:
    │   ├── FAIL_FAST → stop graph, return error
    │   ├── RETRY → re-queue with backoff (max N times)
    │   ├── SKIP → mark node SKIPPED, continue to next edges
    │   └── FALLBACK → route to fallback_node specified on edge
    │
    ├── Middleware.on_error(node, error, ctx)
    │       └── AuditMiddleware: log node_failed
    │
    └── Persist ExecutionState (node → FAILED or RETRYING)
```

---

## Cancellation Flow

```
User calls handle.cancel()
    │
    ├── CancellationToken.cancel()
    ├── GraphRuntime checks token before every frontier dispatch
    ├── Running nodes check token between steps (cooperative)
    │       └── SuluvAgent checks between ReAct steps
    ├── All PENDING nodes → CANCELLED
    └── Return partial ExecutionResult
```

---

## Streaming Flow

```
async for event in runtime.execute_stream(graph, input, ctx):
    match event:
        case NodeStarted(node_id, timestamp):  ...
        case NodeOutput(node_id, chunk):        ...   # partial output from agent
        case NodeCompleted(node_id, result):    ...
        case NodeFailed(node_id, error):        ...
        case GraphCompleted(result):            ...
```

---

## Process Engine — Full Business Process Management

### Design Philosophy

The Process Engine is not just a "graph template factory." It is a **first-class business process management system** that models real-world workflows with their full complexity: decisions, forms, SLAs with escalation chains, compensation (saga pattern), versioning, work assignment, signals, correlation, business calendars, and analytics.

A `ProcessDefinition` compiles down to a `GraphDefinition` for execution, but it carries rich semantic metadata that the graph engine alone does not understand. The process engine adds:

- **Business semantics** — stages, steps, variables with scoping
- **Human workflow** — forms, work assignment, claim/release, delegation
- **Time awareness** — SLAs calculated against business calendars, escalation chains
- **Reliability** — compensation handlers (saga), checkpoints, correlation
- **Governance** — versioning, migration, instance lifecycle management
- **Observability** — process analytics, bottleneck detection, SLA compliance

```
ProcessDefinition  →  compile()  →  GraphDefinition  →  GraphRuntime
     ↑                                                        │
     │              Process Engine adds:                      │
     │              • Variable scoping & mutation tracking     │
     │              • SLA timers & escalation                  │
     │              • Compensation orchestration               │
     │              • Signal routing & correlation             │
     │              • Work assignment & RBAC                   │
     │              • Instance lifecycle management            │
     │              • Version control                          │
     └────────────── Process analytics ◄──────────────────────┘
```

---

### Process Primitives

#### 1. ProcessDefinition & Versioning

```python
process = ProcessDefinition(
    name="nbfc-loan",
    version="2.1",
    description="NBFC loan origination workflow",
    owner="credit-ops-team",
)

# Version management
registry = ProcessVersionRegistry()
registry.register(process)                         # register v2.1
registry.set_active("nbfc-loan", "2.1")            # new instances use v2.1
registry.deprecate("nbfc-loan", "1.0")             # no new instances on v1.0
# In-flight v1.0 instances continue on v1.0 until completion

# Migration (optional)
registry.add_migration("nbfc-loan", from_="1.0", to="2.1",
    strategy=MigrationStrategy.CHECKPOINT_BOUNDARY,  # migrate at next checkpoint
    transformer=v1_to_v2_transformer,                # transform state shape
)
```

**Key concepts:**
- Every `ProcessDefinition` has a `name` + `version` pair (immutable once registered)
- In-flight instances always run on the version they started with
- Optional migration strategies: `AT_CHECKPOINT`, `IMMEDIATE`, `MANUAL`
- Side-by-side execution of multiple versions

---

#### 2. Process Variables & Data Scoping

```python
# Process-level variables — visible to all stages
process.add_variable("customer_id", type=str, required=True, immutable=True)
process.add_variable("loan_amount", type=float, required=True)
process.add_variable("status", type=str, default="initiated")

# Stage-scoped variables — only visible within that stage
kyc_stage.add_variable("pan_verified", type=bool, scope=VariableScope.STAGE)
kyc_stage.add_variable("aadhaar_verified", type=bool, scope=VariableScope.STAGE)

# Access pattern inside agents/steps
ctx.variables["customer_id"]          # read process variable
ctx.variables["loan_amount"] = 600000 # write (mutation tracked)
ctx.stage_variables["pan_verified"]   # read stage variable
```

**Key concepts:**
- `VariableScope.PROCESS` — visible everywhere, persisted with instance
- `VariableScope.STAGE` — visible only within the stage
- `VariableScope.STEP` — visible only within the step
- All mutations are tracked: who, when, old value, new value
- `immutable=True` — set once, never changed (e.g., customer_id)

---

#### 3. Stages & Steps

```python
kyc_stage = ProcessStage(
    name="kyc",
    description="Know Your Customer verification",
    entry_criteria=lambda ctx: ctx.variables["loan_amount"] > 0,
    exit_criteria=lambda ctx: ctx.stage_variables.get("pan_verified") and
                               ctx.stage_variables.get("aadhaar_verified"),
    checkpoint=True,  # persist state after this stage completes
)

# Steps within a stage — executed sequentially or conditionally
kyc_stage.add_step(AgentStep(name="pan-check", agent=pan_agent))
kyc_stage.add_step(AgentStep(name="aadhaar-check", agent=aadhaar_agent))
kyc_stage.add_step(DecisionStep(name="kyc-decision", table="kyc_rules"))
```

**Step types (all implement `ProcessStep` ABC):**

| Step Type | Description |
|---|---|
| `AgentStep` | Executes a SuluvAgent |
| `ToolStep` | Runs a tool directly (no LLM) |
| `DecisionStep` | Evaluates a DecisionTable |
| `FormStep` | Collects structured data from a human |
| `HumanStep` | Generic human task (approve/reject/comment) |
| `NotifyStep` | Sends a notification (non-blocking) |
| `DelayStep` | Waits for duration or timestamp |
| `SubprocessStep` | Invokes another ProcessDefinition |
| `CompensationStep` | Runs compensation logic (saga rollback) |

---

#### 4. Business Rules & Decision Tables

```python
# Define a decision table (DMN-style)
eligibility_table = DecisionTable(
    name="loan_eligibility",
    inputs=["income", "cibil_score", "age", "existing_loans"],
    output="decision",
    rules=[
        Rule(when={"income": ">500000", "cibil_score": ">750", "age": "<55"},
             then="AUTO_APPROVE", priority=1),
        Rule(when={"income": ">300000", "cibil_score": ">650"},
             then="MANUAL_REVIEW", priority=2),
        Rule(when={"cibil_score": "<500"},
             then="REJECT", priority=3),
        Rule(default=True, then="MANUAL_REVIEW"),
    ],
    hit_policy=HitPolicy.FIRST,  # FIRST | ALL | COLLECT | PRIORITY
)

process.add_decision_table("eligibility", eligibility_table)

# Scoring matrix
risk_scoring = ScoringMatrix(
    name="risk_score",
    factors=[
        Factor("cibil_score", weight=0.4, ranges=[
            (">750", 100), (">650", 70), (">500", 40), ("<=500", 10)
        ]),
        Factor("income", weight=0.3, ranges=[
            (">1000000", 100), (">500000", 70), (">300000", 40), ("<=300000", 20)
        ]),
        Factor("employment_years", weight=0.3, ranges=[
            (">5", 100), (">2", 60), ("<=2", 30)
        ]),
    ],
)
```

**Key concepts:**
- Decision tables are **data, not code** — can be updated by business users without deployment
- `HitPolicy`: `FIRST` (first matching rule), `ALL` (all matching), `COLLECT` (aggregate), `PRIORITY` (highest priority)
- Scoring matrices produce weighted numeric scores
- `DecisionNode` in the graph evaluates tables; `DecisionStep` in processes
- `RulesEngine` port allows pluggable rule evaluation backends

---

#### 5. Forms & Structured Data Collection

```python
approval_form = FormDefinition(
    name="loan_approval",
    title="Loan Approval Decision",
    fields=[
        Field("decision", type=FieldType.SELECT,
              options=["approve", "reject", "refer_back"],
              required=True),
        Field("approved_amount", type=FieldType.NUMBER,
              visible_when=lambda f: f["decision"] == "approve",
              validation=lambda v, ctx: v <= ctx.variables["loan_amount"]),
        Field("rejection_reason", type=FieldType.TEXT,
              visible_when=lambda f: f["decision"] == "reject",
              required_when=lambda f: f["decision"] == "reject"),
        Field("comments", type=FieldType.TEXTAREA),
        Field("evidence", type=FieldType.FILE_UPLOAD, max_files=5),
    ],
    sections=[
        FormSection("Customer Info", fields=["customer_summary"], read_only=True),
        FormSection("Decision", fields=["decision", "approved_amount", "rejection_reason"]),
        FormSection("Supporting", fields=["comments", "evidence"]),
    ],
)
```

**FieldType enum:**
`TEXT` | `TEXTAREA` | `NUMBER` | `SELECT` | `MULTI_SELECT` | `DATE` | `DATETIME` | `CHECKBOX` | `FILE_UPLOAD` | `SIGNATURE` | `READONLY` | `RICH_TEXT`

**Key concepts:**
- Forms are structured definitions, not UI components (rendering is adapter concern)
- Conditional visibility (`visible_when`) and conditional required (`required_when`)
- Field-level validation with access to process context
- Sections for logical grouping
- Read-only sections for displaying context to the reviewer
- `FormNode` renders in graph; `FormStep` renders in process

---

#### 6. SLAs, Business Calendars & Escalation

```python
# Business calendar
india_calendar = BusinessCalendar(
    name="india-ops",
    working_hours=WorkingHours(start="09:00", end="18:00", timezone="Asia/Kolkata"),
    working_days=[Mon, Tue, Wed, Thu, Fri],
    holidays=[
        Holiday("2026-01-26", "Republic Day"),
        Holiday("2026-08-15", "Independence Day"),
        Holiday("2026-10-02", "Gandhi Jayanti"),
        # ... importable from standard holiday packs
    ],
)

# SLA with escalation chain
kyc_sla = SLA(
    duration=4,
    unit=SLAUnit.BUSINESS_HOURS,  # not wall-clock hours
    calendar=india_calendar,
    escalation=EscalationChain([
        Escalation(at_percent=50,
            action=notify_assignee("SLA 50% elapsed")),
        Escalation(at_percent=80,
            action=notify_assignee("SLA critical — 20% remaining")),
        Escalation(at_percent=100, actions=[
            escalate_to_role("team-lead"),
            notify_channel("ops-alerts", "SLA breached for {instance_id}"),
        ]),
        Escalation(at_percent=150, actions=[
            escalate_to_role("manager"),
            auto_reassign(strategy="round-robin"),
        ]),
        Escalation(at_percent=200, actions=[
            escalate_to_role("vp-operations"),
            suspend_downstream_stages(),
            notify_compliance("Critical SLA breach: {instance_id}"),
        ]),
    ]),
    breach_policy=SLABreachPolicy.ESCALATE_AND_CONTINUE,
    # ESCALATE_AND_CONTINUE | BLOCK_UNTIL_RESOLVED | AUTO_SKIP | FAIL_PROCESS
)
```

**Key concepts:**
- SLAs measured in **business hours**, not wall-clock time
- `BusinessCalendar` port — working hours, holidays, timezone-aware
- `EscalationChain` — configurable ladder of actions at percentage thresholds
- Escalation actions: notify, reassign, escalate to role, suspend, block
- `SLABreachPolicy` controls what happens after breach

---

#### 7. Work Assignment (RBAC)

```python
approval_stage = ProcessStage(
    name="approval",
    assignment=WorkAssignment(
        role="credit-officer",
        strategy=AssignmentStrategy.ROUND_ROBIN,
        # ROUND_ROBIN | LEAST_LOADED | MANUAL_CLAIM | SPECIFIC_USER | RULE_BASED
        filters=[
            BranchFilter(field="branch_code", match_variable="customer_branch"),
            SkillFilter(required=["high_value_loans"]),
        ],
        delegation=DelegationRules(
            allow_delegation=True,
            auto_delegate_on_absence=True,
            delegate_to="deputy",
        ),
        claim_timeout=timedelta(minutes=15),  # auto-reassign if not claimed
    ),
)

# Work queue operations (HumanTaskQueue port)
tasks = await task_queue.list(role="credit-officer", status="pending")
await task_queue.claim(task_id, user_id="emp-456")     # I'll take this one
await task_queue.release(task_id)                       # put it back
await task_queue.delegate(task_id, to_user="emp-789")   # give to someone else
await task_queue.complete(task_id, result={...})
```

**Key concepts:**
- Role-based assignment with configurable strategies
- Filters: branch, skill, workload, availability
- Delegation rules: manual, auto-on-absence, hierarchical
- Claim/release semantics — prevent double-work
- Priority-aware queuing (high-value loans first)

---

#### 8. Compensation & Saga Pattern

```python
# Each stage can define a compensation handler
process.add_stage(ProcessStage(
    name="kyc",
    agent=kyc_agent,
    compensation=CompensationHandler(
        handler=reverse_kyc_hold,  # async function(ctx) → None
        description="Release KYC verification hold",
        timeout=timedelta(minutes=5),
        retry_on_failure=True,
    ),
))

process.add_stage(ProcessStage(
    name="credit_lock",
    agent=credit_agent,
    compensation=CompensationHandler(handler=release_credit_lock),
))

process.add_stage(ProcessStage(
    name="disbursement",
    agent=disburse_agent,
    # No compensation — this is the final step
))

# If disbursement fails after KYC + credit_lock succeeded:
# 1. CompensationEngine runs release_credit_lock (reverse stage 2)
# 2. CompensationEngine runs reverse_kyc_hold (reverse stage 1)
# 3. Order: reverse chronological (last completed → first completed)

process.saga_config = SagaConfig(
    compensation_order=CompensationOrder.REVERSE_CHRONOLOGICAL,
    on_compensation_failure=CompensationFailurePolicy.LOG_AND_ALERT,
    # LOG_AND_ALERT | RETRY_FOREVER | MANUAL_INTERVENTION
)
```

**Key concepts:**
- Each stage/step can register a compensation handler
- On failure, compensations run in reverse order (saga pattern)
- Compensation handlers are idempotent and retriable
- `CompensationNode` in graph; `CompensationStep` in process
- `SagaConfig` controls behavior when compensation itself fails

---

#### 9. Signals & Boundary Events

```python
# Signals — external business events that affect running instances
process.on_signal("fraud_alert", handler=FraudAlertHandler(
    action=SignalAction.SUSPEND,  # SUSPEND | CANCEL | REROUTE | ESCALATE | NOTIFY
    message="Fraud detected — instance suspended pending investigation",
    resume_requires_role="fraud-investigator",
))

process.on_signal("customer_document_uploaded",
    unblocks="document_collection",  # resumes a waiting step
    validator=lambda event: event["doc_type"] in ["pan_card", "aadhaar"],
)

process.on_signal("regulatory_hold",
    action=SignalAction.SUSPEND,
    scope=SignalScope.ALL_INSTANCES,  # affects ALL instances, not just one
    filter=lambda instance: instance.variables["product_type"] == "personal_loan",
)

# Boundary events — attached to a specific node/stage
kyc_stage.add_boundary_event(BoundaryEvent(
    signal="customer_cancelled",
    interrupting=True,   # True = abort the stage; False = parallel handler
    handler=handle_cancellation,
))

kyc_stage.add_boundary_event(BoundaryEvent(
    signal="additional_info_received",
    interrupting=False,  # non-interrupting: inject data without stopping
    handler=lambda ctx, event: ctx.variables.update(event["data"]),
))
```

**Key concepts:**
- **Signals** are named external events (fraud alert, document received, regulatory hold)
- Signals can suspend, cancel, reroute, or unblock instances
- **Boundary events** attach to stages — react to events *while the stage is running*
- Interrupting boundary events abort the current work
- Non-interrupting boundary events inject data alongside running work
- Signal scope: single instance, filtered instances, or all instances

---

#### 10. Correlation & Event Routing

```python
# Correlation — route external events to the correct process instance
process.correlate_on("customer_id")         # primary correlation key
process.correlate_on("loan_application_id") # secondary

# When an external event arrives:
# 1. CorrelationEngine extracts correlation keys from the event
# 2. Looks up which process instance(s) match
# 3. Routes the event to matched instance(s)

@app.post("/webhook/payment-received")
async def payment_received(request):
    event = ProcessEvent(
        signal="payment_received",
        correlation={"customer_id": request.customer_id},
        data={"amount": request.amount, "reference": request.ref},
    )
    await correlation_engine.route(event)
    # → finds instance with customer_id=request.customer_id
    # → delivers signal to that instance
```

**Key concepts:**
- Each process defines which fields are correlation keys
- External events carry correlation data
- `CorrelationEngine` matches events to instances
- Multiple correlation keys supported (any match routes the event)
- Used for: API callbacks, webhook handlers, inter-process communication

---

#### 11. Process Instance Lifecycle

```python
instance_mgr = ProcessInstanceManager(store=process_instance_store)

# Start a new instance
instance = await instance_mgr.start("nbfc-loan",
    input={"customer_id": "C123", "loan_amount": 500000},
    priority=Priority.HIGH,
    metadata={"source": "web-portal", "branch": "mumbai-01"},
)

# Query instances
active = await instance_mgr.list(
    process_name="nbfc-loan",
    status=[InstanceStatus.RUNNING, InstanceStatus.WAITING],
    created_after=datetime(2026, 1, 1),
    assigned_to="emp-456",
    sort_by="priority",
)

# Lifecycle operations
await instance_mgr.suspend(instance_id, reason="Pending investigation")
await instance_mgr.resume(instance_id)
await instance_mgr.cancel(instance_id, reason="Customer withdrew")
await instance_mgr.reassign(instance_id, stage="approval", to_user="emp-789")

# Bulk operations
await instance_mgr.bulk_suspend(
    filter={"process_name": "nbfc-loan", "version": "1.0"},
    reason="Migrating to v2.0",
)
```

**Instance states:**
```
CREATED → RUNNING → COMPLETED
           ↓
           → FAILED → COMPENSATING → COMPENSATED
           ↕ SUSPENDED (manual or signal-triggered)
           → CANCELLED
           → WAITING (human task, delay, or signal)
```

---

#### 12. Comments, Notes & Collaboration

```python
await instance.add_comment(
    user="emp-456",
    text="Spoke with customer. Will upload PAN by tomorrow.",
    visibility=CommentVisibility.INTERNAL,  # INTERNAL | CUSTOMER_VISIBLE
    tagged_users=["emp-789"],
)

await instance.add_comment(
    user="emp-789",
    text="Overriding credit score per manager approval. See attached email.",
    attachments=[Attachment(name="approval_email.pdf", artifact_id="art-123")],
    stage="credit",  # attached to a specific stage
)

comments = await instance.get_comments(stage="credit", after=datetime(2026, 1, 1))
```

---

#### 13. Document Generation

```python
# Template-based document generation
sanction_letter = DocumentTemplate(
    name="sanction_letter_v2",
    template_path="templates/sanction_letter.html",
    output_format="pdf",
    merge_fields=["customer_name", "loan_amount", "interest_rate", "tenure"],
)

# TemplateEngine port renders templates
engine: TemplateEngine = JinjaTemplateEngine()
document = await engine.render(sanction_letter, variables={
    "customer_name": ctx.variables["customer_name"],
    "loan_amount": ctx.variables["loan_amount"],
    "interest_rate": 12.5,
    "tenure": "36 months",
})

# Store generated document
await artifact_store.put(f"docs/{instance_id}/sanction_letter.pdf", document)
```

---

#### 14. Process Analytics & Reporting

```python
analytics = ProcessAnalytics(store=process_instance_store)

# Cycle time analysis
cycle_times = await analytics.cycle_times("nbfc-loan", period="last_30_days")
# → {stage: "kyc", avg: 25min, p50: 20min, p95: 55min, p99: 120min}
# → {stage: "credit", avg: 3.2hrs, p50: 2hrs, p95: 8hrs, p99: 24hrs}

# SLA compliance
sla_report = await analytics.sla_compliance("nbfc-loan", period="last_30_days")
# → {stage: "kyc", compliance: 94.2%, breached: 58, total: 1000}

# Bottleneck detection
bottlenecks = await analytics.bottlenecks("nbfc-loan")
# → [{stage: "credit", avg_wait: 4.5hrs, recommendation: "Add more credit officers"}]

# Drop-off analysis
dropoffs = await analytics.dropoff_analysis("nbfc-loan")
# → [{stage: "document_upload", drop_rate: 38%, avg_abandon_time: 12hrs}]

# Agent vs Human performance
perf = await analytics.performance_comparison("nbfc-loan")
# → {agent_stages: {avg: 2min, error_rate: 1.2%},
#    human_stages: {avg: 3.5hrs, error_rate: 4.8%}}
```

---

### Process → Graph Compilation

```
ProcessDefinition                          GraphDefinition
┌──────────────────────┐                   ┌──────────────────────────┐
│ Variables:            │                   │                          │
│   customer_id (str)   │                   │ TriggerNode: start       │
│   loan_amount (float) │                   │     │                    │
│                       │                   │     ▼                    │
│ DecisionTable:        │   compile()       │ DecisionNode: eligibility│
│   eligibility         │ ────────────►     │     │          │         │
│                       │                   │     ▼          ▼         │
│ Stage: KYC            │                   │ AgentNode:   RouterNode  │
│   Step: pan-check     │                   │   kyc          (reject)  │
│   Step: aadhaar-check │                   │     │                    │
│   SLA: 4 biz hours    │                   │     ▼                    │
│   Escalation: 3-tier  │                   │ TimerNode: sla-check     │
│   Compensation: ✓     │                   │     │                    │
│                       │                   │     ▼                    │
│ Stage: Credit         │                   │ AgentNode: credit        │
│   Step: score         │                   │     │                    │
│   Step: approve       │                   │     ▼                    │
│   Form: approval_form │                   │ FormNode: approval       │
│   Assignment: round-  │                   │     │                    │
│     robin credit-mgr  │                   │     ▼                    │
│                       │                   │ CompensationNode (on err)│
│ Stage: Disburse       │                   │     │                    │
│   Step: transfer      │                   │     ▼                    │
│                       │                   │ AgentNode: disburse      │
│ Signals:              │                   │     │                    │
│   fraud_alert →       │                   │ SignalNode: fraud_catch  │
│     suspend           │                   │     │                    │
│                       │                   │     ▼                    │
│ Correlation:          │                   │ GatewayNode: complete    │
│   customer_id         │                   │                          │
└──────────────────────┘                   └──────────────────────────┘
```

The compiler generates:
- `DecisionNode` for each `DecisionTable`
- `AgentNode` for each `AgentStep`
- `FormNode` for each `FormStep`
- `TimerNode` for SLA enforcement (calendar-aware)
- `SignalNode` for boundary events and process signals
- `CompensationNode` wired as error-path alternatives
- `GatewayNode` for join points and parallel approval gates
- Middleware injection for variable tracking, escalation, and analytics

---

## Package Layout

```
suluv/
├── pyproject.toml                    # uv workspace root
├── packages/
│   ├── suluv-core/                   # THE framework
│   │   └── src/suluv/core/
│   │       ├── types.py              # IDs, enums, result dataclasses
│   │       ├── messages/             # Multimodal message protocol
│   │       │   ├── content.py        # ContentBlock, ContentType
│   │       │   ├── message.py        # SuluvMessage, MessageRole
│   │       │   └── prompt.py         # SuluvPrompt (with output_schema)
│   │       ├── ports/                # All ABCs (18 ports)
│   │       │   ├── llm_backend.py
│   │       │   ├── event_bus.py
│   │       │   ├── state_store.py
│   │       │   ├── audit_backend.py
│   │       │   ├── memory_backend.py
│   │       │   ├── guardrail.py
│   │       │   ├── policy_rule.py
│   │       │   ├── consent_provider.py
│   │       │   ├── corpus_provider.py
│   │       │   ├── connector.py
│   │       │   ├── human_task_queue.py  # emit/poll/claim/release/delegate/complete
│   │       │   ├── artifact_store.py
│   │       │   ├── notifier.py
│   │       │   ├── verification.py
│   │       │   ├── rules_engine.py     # evaluate(table, inputs) → Decision
│   │       │   ├── business_calendar.py # is_working(dt) / add_biz_hours(dt, h)
│   │       │   ├── process_instance_store.py  # save/load/query process instances
│   │       │   └── template_engine.py  # render(template, vars) → Document
│   │       ├── engine/               # Graph execution engine
│   │       │   ├── node.py           # GraphNode ABC, NodeType, NodeInput/Output
│   │       │   ├── edge.py           # GraphEdge, ErrorPolicy
│   │       │   ├── trigger.py        # TriggerNode, TriggerType (webhook/cron/event)
│   │       │   ├── loop.py           # LoopNode — repeat until condition
│   │       │   ├── map_node.py       # MapNode — parallel for-each
│   │       │   ├── gateway.py        # GatewayNode — N-of-M join
│   │       │   ├── delay.py          # DelayNode — time-based wait
│   │       │   ├── decision.py       # DecisionNode — rules/decision table eval
│   │       │   ├── form.py           # FormNode — structured data collection
│   │       │   ├── signal.py         # SignalNode — catch/throw business signals
│   │       │   ├── compensation.py   # CompensationNode — saga rollback
│   │       │   ├── timer.py          # TimerNode — calendar-aware wait
│   │       │   ├── graph.py          # GraphDefinition (serializable)
│   │       │   ├── state.py          # ExecutionState, NodeState
│   │       │   ├── runtime.py        # GraphRuntime (the main loop)
│   │       │   ├── middleware.py     # Middleware ABC, CostMiddleware, AuditMiddleware
│   │       │   ├── cancel.py        # CancellationToken
│   │       │   ├── events.py        # GraphEvent types for streaming
│   │       │   └── executor.py      # NodeExecutor (dispatches by NodeType)
│   │       ├── agent/                # Agent system (works standalone, no graph needed)
│   │       │   ├── role.py           # AgentRole
│   │       │   ├── context.py        # AgentContext
│   │       │   ├── agent.py          # SuluvAgent (ReAct loop, owns tools)
│   │       │   ├── result.py         # AgentResult, StepRecord
│   │       │   ├── agent_node.py     # AgentNode — wraps agent as GraphNode
│   │       │   ├── memory_manager.py # MemoryManager — wires 4 memory tiers
│   │       │   ├── guardrail_chain.py
│   │       │   ├── corpus_registry.py
│   │       │   └── cost_tracker.py
│   │       ├── tools/                # Tool system
│   │       │   ├── decorator.py      # @suluv_tool — defines a tool
│   │       │   └── runner.py         # SandboxedToolRunner (timeout, audit)
│   │       ├── policy/               # Policy engine
│   │       │   └── engine.py         # PolicyEngine
│   │       ├── compliance/           # Guardrails + audit hooks
│   │       │   ├── audit_hooks.py
│   │       │   └── consent_enforcer.py
│   │       ├── process/              # Process engine (full BPM)
│   │       │   ├── definition.py     # ProcessDefinition
│   │       │   ├── version.py        # ProcessVersion, ProcessVersionRegistry, migration
│   │       │   ├── variables.py      # ProcessVariables, VariableScope, mutation tracking
│   │       │   ├── stage.py          # ProcessStage, entry/exit criteria
│   │       │   ├── step.py           # ProcessStep ABC + all step types
│   │       │   ├── decision.py       # DecisionTable, ScoringMatrix, Rule, HitPolicy
│   │       │   ├── form.py           # FormDefinition, Field, FieldType, FormSection
│   │       │   ├── sla.py            # SLA, SLAUnit, SLABreachPolicy
│   │       │   ├── escalation.py     # EscalationChain, Escalation, escalation actions
│   │       │   ├── calendar.py       # BusinessCalendar, WorkingHours, Holiday
│   │       │   ├── compensation.py   # CompensationHandler, SagaConfig, CompensationEngine
│   │       │   ├── assignment.py     # WorkAssignment, AssignmentStrategy, DelegationRules
│   │       │   ├── correlation.py    # CorrelationEngine, CorrelationKey, ProcessEvent
│   │       │   ├── signal.py         # SignalHandler, BoundaryEvent, SignalAction, SignalScope
│   │       │   ├── instance.py       # ProcessInstanceManager, InstanceStatus, lifecycle ops
│   │       │   ├── comments.py       # ProcessComment, Attachment, CommentVisibility
│   │       │   ├── analytics.py      # ProcessAnalytics, cycle times, SLA compliance
│   │       │   ├── compiler.py       # process → graph compiler (enhanced)
│   │       │   └── process_node.py   # ProcessNode — wraps process as GraphNode
│   │       ├── adapters/             # In-memory adapters (ship with core)
│   │       │   ├── memory_bus.py     # InMemoryEventBus
│   │       │   ├── memory_state.py   # InMemoryStateStore
│   │       │   ├── memory_audit.py   # InMemoryAuditBackend
│   │       │   ├── memory_memory.py  # InMemory ShortTerm/LongTerm/etc
│   │       │   ├── memory_tasks.py   # InMemoryHumanTaskQueue
│   │       │   ├── memory_rules.py   # InMemoryRulesEngine
│   │       │   ├── memory_calendar.py # InMemoryBusinessCalendar
│   │       │   ├── memory_instances.py # InMemoryProcessInstanceStore
│   │       │   ├── memory_templates.py # InMemoryTemplateEngine
│   │       │   └── mock_llm.py       # MockLLM for testing
│   │       └── testing/              # Test harness for framework users
│   │           ├── harness.py        # AgentTestHarness, GraphTestHarness
│   │           ├── eval_suite.py     # EvalSuite — batch evaluation + scoring
│   │           ├── mocks.py          # MockLLM, MockTools, etc (re-exports)
│   │           └── assertions.py     # assert_audit_contains, assert_no_pii, etc
│   │
│   ├── suluv-lang/                   # LLM adapters (separate install)
│   ├── suluv-connectors/             # External API connectors
│   ├── suluv-india/                  # India identity + PII guardrail
│   └── suluv-cli/                    # Developer CLI
│
└── tests/
    ├── unit/                         # Per-module unit tests
    ├── contracts/                    # Port contract tests
    ├── integration/                  # Multi-component integration
    └── e2e/                          # Full pipeline tests
```

---

## Phased Build Plan

### Phase 1 — Foundation (types + ports + hexagonal skeleton)

**Build:**
- Project setup (monorepo, pyproject.toml)
- Core types (IDs, enums, result dataclasses)
- Message protocol (multimodal: text/image/audio/tool_call)
- All 18 port ABCs (LLMBackend, EventBus, StateStore, AuditBackend, RulesEngine, BusinessCalendar, etc.)
- In-memory adapters for every port

**Test:**
- 18 port contract tests (each adapter passes its ABC contract)

**Ship:**
- Foundation compiles, every port has a working adapter

---

### Phase 2 — Graph Engine (the execution backbone)

**Build:**
- GraphNode ABC + NodeType enum (16 types including process engine nodes)
- GraphEdge (conditions, transforms)
- GraphDefinition (nodes + edges + entry/exit points)
- ExecutionState (per-node state tracking, persisted)
- EventBus integration (publish node events)
- GraphRuntime (frontier computation, node dispatch, fan-out/join)
- Middleware (before_node / after_node hooks)
- ErrorPolicy + Retry (FAIL_FAST, RETRY, SKIP, FALLBACK)
- CancellationToken (cooperative cancellation)
- Streaming (execute_stream → AsyncIterator[GraphEvent])
- Serialization (to_dict / from_dict for graph definitions)

**Test:**
- 18+ tests: linear graph, branch, fan-out/join, retry on fail, cancel mid-run, stream events, save/load JSON, middleware fires, trigger node starts graph, loop until condition, map over list, gateway N-of-M, delay node

**Ship:**
- Graph engine runs standalone with dummy nodes

---

### Phase 3 — Agent System (agents as graph nodes)

**Build:**
- AgentRole + AgentContext
- Tool system (@suluv_tool decorator, SandboxedRunner — no global registry, tools owned by agent)
- GuardrailChain + PolicyEngine
- SuluvAgent (ReAct loop, works standalone without graph)
- MemoryManager (wires ShortTerm/LongTerm/Episodic/Semantic to agent)
- Structured output (output_schema enforcement, result.structured)
- AgentNode (thin wrapper — plugs SuluvAgent into GraphRuntime)
- Orchestrator (capability routing → builds graph → runs)
- CostTracker (per-node, per-graph, per-session token + USD tracking)
- Test Harness (MockLLM, expect_call, assert_audit for framework users)
- EvalSuite (batch evaluation: accuracy, latency, cost, failure reporting)

**Test:**
- 15+ tests: standalone agent (no graph), tool call, guardrail block, policy deny, agent-in-graph, parallel agents, cost tracking, harness expect_call, memory read/write across runs, structured output parsing, eval suite scoring

**Ship:**
- Agents run inside graphs

---

### Phase 4 — Process Engine (full business process management)

**Build (4a — Core Process Primitives):**
- ProcessDefinition + ProcessVersion + ProcessVersionRegistry
- ProcessVariables with scoping (PROCESS / STAGE / STEP) and mutation tracking
- ProcessStage / ProcessStep ABCs + all step types (Agent, Tool, Decision, Form, Human, Notify, Delay, Subprocess, Compensation)
- ProcessDefinition → GraphDefinition compiler (enhanced)
- ProcessNode (wraps process as a GraphNode for subgraph embedding)

**Build (4b — Business Rules & Forms):**
- DecisionTable, ScoringMatrix, Rule, HitPolicy
- RulesEngine port + InMemoryRulesEngine adapter
- DecisionNode (graph) + DecisionStep (process)
- FormDefinition, Field, FieldType, FormSection (conditional visibility, validation)
- FormNode (graph) + FormStep (process)

**Build (4c — SLAs, Calendars & Escalation):**
- BusinessCalendar port + InMemoryBusinessCalendar adapter
- SLA with SLAUnit.BUSINESS_HOURS, breach policies
- EscalationChain — configurable percentage-based escalation ladder
- TimerNode (calendar-aware, replaces raw DelayNode for SLA enforcement)

**Build (4d — Compensation, Signals & Correlation):**
- CompensationHandler + SagaConfig + CompensationEngine
- CompensationNode (saga rollback in graph)
- SignalHandler + BoundaryEvent (interrupting / non-interrupting)
- SignalNode (catches business signals in graph)
- CorrelationEngine + CorrelationKey + ProcessEvent routing

**Build (4e — Work Assignment & Human Workflow):**
- WorkAssignment strategies (ROUND_ROBIN, LEAST_LOADED, MANUAL_CLAIM, RULE_BASED)
- Filters (branch, skill, workload)
- DelegationRules (manual, auto-on-absence)
- Claim/release/delegate semantics on HumanTaskQueue
- Priority-aware queuing

**Build (4f — Instance Lifecycle & Observability):**
- ProcessInstanceManager — start, suspend, resume, cancel, reassign, bulk ops
- ProcessInstanceStore port + InMemoryProcessInstanceStore adapter
- Instance state machine (CREATED → RUNNING → COMPLETED/FAILED/CANCELLED/SUSPENDED/WAITING)
- ProcessComment + Attachment (collaboration on instances)
- ProcessAnalytics — cycle time, SLA compliance, bottleneck detection, drop-off analysis
- TemplateEngine port + InMemoryTemplateEngine adapter (document generation)

**Test:**
- 25+ tests: linear process, SLA breach with escalation, HITL pause/resume, policy checkpoint, decision table evaluation (FIRST/ALL/PRIORITY hit policies), form validation (conditional visibility, required_when), compensation rollback (saga), signal suspend/resume, boundary event interrupting/non-interrupting, correlation routing (event → correct instance), work assignment (round-robin, claim/release), business calendar (holidays, working hours), process versioning (in-flight instances on old version), instance lifecycle (suspend/resume/cancel), process compiles to graph and runs, analytics queries

**Ship:**
- Full business process management — processes run as graphs with rich semantics

---

### Phase 5 — India Pack + CLI (domain + developer tools)

**Build:**
- Identity types (PAN, Aadhaar, GSTIN, CIN, DIN, UDYAM, IFSC, UPI...)
- PII guardrail (regex masking for Indian identifiers)
- suluv-lang (OpenAI, Anthropic backends with multimodal message support)
- suluv-connectors (HttpConnector, ConnectorPipeline)
- suluv-cli (init, create agent/tool/process, verify, info)
- End-to-end example: NBFC loan pipeline (KYC → Credit → Disbursement)

**Test:**
- E2E: full NBFC loan pipeline with PII masking, SLA, HITL approval

**Ship:**
- Framework is usable end-to-end

---

## Intentionally Excluded (can add later)

| Feature | Reason |
|---|---|
| Redis/Kafka EventBus | InMemory is fine for P1-P5. Broker adapters are just adapter swaps. |
| OpenTelemetry tracing | Audit backend covers 90%. OTel is an adapter concern, not core. |
| Visual graph editor | Serialization enables it. UI is a separate project. |
| Visual form builder | FormDefinition is data — UI builder is a separate tool. |
| Rate limiter | Goes in suluv-lang adapters, not core engine. |
| Semantic cache | Can be a middleware later — not core. |
| Multi-region / sharding | OrgID scoping + StateStore port already enables this. Swap adapter when needed. |
| BPMN import/export | ProcessDefinition is code-first. BPMN bridge is a future adapter. |
| Real-time dashboards | ProcessAnalytics provides data. Dashboard UI is a separate project. |
| Email/SMS adapters | NotifierPort defines the interface. Channel adapters are external packages. |
