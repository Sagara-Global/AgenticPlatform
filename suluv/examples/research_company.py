"""Example — Company Research Agent powered by Gemini.

Run::

    python examples/research_company.py
"""

import asyncio
import os

from suluv.core.adapters.gemini_llm import GeminiBackend
from suluv.core.adapters.memory_checkpointer import InMemoryCheckpointer
from suluv.core.adapters.memory_audit import InMemoryAuditBackend
from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.context import AgentContext
from suluv.core.agent.cost_tracker import CostTracker
from suluv.core.agent.role import AgentRole
from suluv.core.tools.builtins import web_search, http_fetch, calculator, datetime_now


def create_company_researcher(
    llm,
    *,
    checkpointer=None,
    audit_backend=None,
    cost_tracker=None,
) -> SuluvAgent:
    """Build a company research agent — example-local factory."""
    role = AgentRole(
        name="CompanyResearcher",
        description=(
            "An expert business analyst who researches companies thoroughly. "
            "Gathers information about company overview, leadership, products, "
            "financials, recent news, and competitive landscape."
        ),
        capabilities=[
            "search the web for company information",
            "fetch and read company web pages",
            "analyse financial data",
            "identify competitors",
            "summarise findings into a structured report",
        ],
        instructions=(
            "When asked to research a company:\n"
            "1. Search for the company overview and basic info\n"
            "2. Search for recent news about the company\n"
            "3. Search for the company's key products/services\n"
            "4. Search for leadership team information\n"
            "5. Compile everything into a structured report with sections:\n"
            "   - Company Overview\n"
            "   - Key Products/Services\n"
            "   - Leadership\n"
            "   - Recent News\n"
            "   - Competitors\n"
            "Always cite your sources. Use the calculator for financial figures. "
            "Be thorough but concise."
        ),
        max_steps=20,
        temperature=0.2,
    )
    tools = [web_search, http_fetch, calculator, datetime_now]
    return SuluvAgent(
        role=role,
        llm=llm,
        tools=tools,
        checkpointer=checkpointer,
        audit_backend=audit_backend,
        cost_tracker=cost_tracker,
    )


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY environment variable (or add it to .env)")

    # 1. Create the Gemini LLM backend
    llm = GeminiBackend(
        model="gemini-2.0-flash",
        api_key=api_key,
        default_temperature=0.2,
    )

    # 2. Setup subsystems
    checkpointer = InMemoryCheckpointer()
    audit = InMemoryAuditBackend()
    cost_tracker = CostTracker()

    # 3. Create the company research agent
    agent = create_company_researcher(
        llm=llm,
        checkpointer=checkpointer,
        audit_backend=audit,
        cost_tracker=cost_tracker,
    )

    # 4. Run the research — threaded conversation
    company = "SagaraGlobal"
    print(f"\n{'='*60}")
    print(f"  Researching: {company}")
    print(f"{'='*60}\n")

    ctx = AgentContext(
        thread_id="research-1",
        user_id="analyst",
        org_id="suluv",
        session_id="session-1",
    )

    result = await agent.run(
        f"Research the company '{company}'. Give me a comprehensive overview "
        f"including what they do, key products, leadership, recent news, "
        f"and main competitors.",
        context=ctx,
    )

    print(f"Success: {result.success}")
    print(f"Steps: {result.step_count}")
    print(f"Tokens: {result.total_tokens}")
    print(f"Cost tracker: {cost_tracker.to_dict()}")
    print(f"\n{'─'*60}")
    print(f"REPORT:\n")
    print(result.answer)

    # 5. Follow-up on same thread
    print(f"\n{'='*60}")
    print(f"  Follow-up question (same thread)")
    print(f"{'='*60}\n")

    result2 = await agent.run(
        "Based on your research, what are the biggest risks facing this company right now?",
        context=ctx,
    )

    print(f"Steps: {result2.step_count} | Tokens: {result2.total_tokens}")
    print(f"\n{'─'*60}")
    print(f"ANSWER:\n")
    print(result2.answer)

    # 6. Show thread info
    thread = await agent.get_thread("research-1")
    if thread:
        print(f"\n{'─'*60}")
        print(f"Thread: {thread.thread_id}")
        print(f"  Messages: {thread.message_count}")
        print(f"  Checkpoints: {thread.checkpoint_count}")
        print(f"  Per-thread cost: {cost_tracker.thread_cost('research-1')}")

    # 7. Show audit trail
    print(f"\n{'─'*60}")
    print(f"Audit events: {len(audit.events)}")
    for e in audit.events:
        print(f"  [{e.event_type}] thread={e.thread_id} user={e.user_id}")


if __name__ == "__main__":
    asyncio.run(main())
