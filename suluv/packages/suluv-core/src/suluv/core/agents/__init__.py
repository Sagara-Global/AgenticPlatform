"""Pre-built agent factories — ready-to-use agents for common patterns.

Usage::

    from suluv.core.agents import create_assistant, create_researcher, create_coder
    from suluv.core.adapters import InMemoryCheckpointer

    cp = InMemoryCheckpointer()
    agent = create_assistant(llm=my_llm, checkpointer=cp)

    # Threaded conversation — agent remembers across runs
    r1 = await agent.run("My name is Alice",
                         context=AgentContext(thread_id="t1"))
    r2 = await agent.run("What's my name?",
                         context=AgentContext(thread_id="t1"))
"""

from __future__ import annotations

from typing import Any

from suluv.core.agent.agent import SuluvAgent
from suluv.core.agent.context import AgentContext
from suluv.core.agent.role import AgentRole
from suluv.core.ports.checkpointer import Checkpointer
from suluv.core.ports.llm_backend import LLMBackend
from suluv.core.tools.decorator import SuluvTool
from suluv.core.tools.builtins import (
    calculator,
    json_extractor,
    datetime_now,
    date_diff,
)


def create_assistant(
    llm: LLMBackend,
    *,
    checkpointer: Checkpointer | None = None,
    extra_tools: list[SuluvTool] | None = None,
    name: str = "Assistant",
    max_steps: int = 15,
    temperature: float = 0.1,
    **kwargs: Any,
) -> SuluvAgent:
    """General-purpose assistant agent with calculator + datetime tools.

    Good for Q&A, summarisation, simple data lookups.
    Pass a ``checkpointer`` for conversation threading.
    """
    role = AgentRole(
        name=name,
        description="A helpful, accurate, and concise assistant.",
        capabilities=[
            "answer questions",
            "perform calculations",
            "work with dates and times",
            "summarise text",
            "explain concepts",
        ],
        instructions=(
            "Be concise. When a calculation is needed, use the calculator tool. "
            "When dates are involved, use the datetime tools. "
            "Always think step by step before answering."
        ),
        max_steps=max_steps,
        temperature=temperature,
    )
    tools = [calculator, json_extractor, datetime_now, date_diff]
    if extra_tools:
        tools.extend(extra_tools)
    return SuluvAgent(role=role, llm=llm, tools=tools, checkpointer=checkpointer, **kwargs)


def create_researcher(
    llm: LLMBackend,
    *,
    checkpointer: Checkpointer | None = None,
    extra_tools: list[SuluvTool] | None = None,
    name: str = "Researcher",
    max_steps: int = 20,
    temperature: float = 0.2,
    **kwargs: Any,
) -> SuluvAgent:
    """Research agent with web search + HTTP fetch + calculator.

    Good for fact-finding, data gathering, web lookups.
    Pass a ``checkpointer`` for conversation threading.
    """
    from suluv.core.tools.builtins import web_search, http_fetch

    role = AgentRole(
        name=name,
        description=(
            "A thorough researcher who gathers information from multiple "
            "sources and synthesises findings."
        ),
        capabilities=[
            "search the web",
            "fetch and read web pages",
            "perform calculations",
            "compare data",
            "summarise findings",
        ],
        instructions=(
            "Always search for information before answering factual questions. "
            "Cross-reference multiple sources when possible. "
            "Cite your sources in the final answer. "
            "Use the calculator for numerical analysis."
        ),
        max_steps=max_steps,
        temperature=temperature,
    )
    tools: list[SuluvTool] = [
        web_search,
        http_fetch,
        calculator,
        datetime_now,
    ]
    if extra_tools:
        tools.extend(extra_tools)
    return SuluvAgent(role=role, llm=llm, tools=tools, checkpointer=checkpointer, **kwargs)


def create_coder(
    llm: LLMBackend,
    *,
    checkpointer: Checkpointer | None = None,
    allowed_dir: str | None = None,
    extra_tools: list[SuluvTool] | None = None,
    name: str = "Coder",
    max_steps: int = 25,
    temperature: float = 0.0,
    **kwargs: Any,
) -> SuluvAgent:
    """Code-focused agent with file read/write + shell exec.

    Good for code analysis, generation, and execution.
    Pass a ``checkpointer`` for conversation threading.
    """
    from suluv.core.tools.builtins import file_reader, file_writer, shell_exec

    role = AgentRole(
        name=name,
        description=(
            "An expert software engineer who can read, write, and execute code. "
            "Writes clean, well-documented, production-quality code."
        ),
        capabilities=[
            "read source files",
            "write and modify files",
            "execute shell commands",
            "analyse code",
            "debug errors",
            "refactor code",
        ],
        instructions=(
            "Read existing code before making changes. "
            "Write clean, typed code with docstrings. "
            "Test your changes by running the code. "
            "Explain what you changed and why."
        ),
        max_steps=max_steps,
        temperature=temperature,
    )
    tools: list[SuluvTool] = [file_reader, file_writer, shell_exec, calculator]
    if extra_tools:
        tools.extend(extra_tools)
    return SuluvAgent(role=role, llm=llm, tools=tools, checkpointer=checkpointer, **kwargs)


def create_data_analyst(
    llm: LLMBackend,
    *,
    checkpointer: Checkpointer | None = None,
    extra_tools: list[SuluvTool] | None = None,
    name: str = "DataAnalyst",
    max_steps: int = 20,
    temperature: float = 0.0,
    **kwargs: Any,
) -> SuluvAgent:
    """Data analysis agent with calculator + JSON extraction.

    Good for analysing structured data, computing statistics,
    and extracting insights from JSON/CSV.
    Pass a ``checkpointer`` for conversation threading.
    """
    role = AgentRole(
        name=name,
        description=(
            "A precise data analyst who works with structured data, "
            "performs calculations, and presents clear insights."
        ),
        capabilities=[
            "parse and extract data from JSON",
            "perform mathematical calculations",
            "compute statistics",
            "identify trends and patterns",
            "format results clearly",
        ],
        instructions=(
            "Use json_extractor to pull data from JSON payloads. "
            "Use calculator for all numerical computations — never guess. "
            "Present findings with exact numbers. "
            "When analysing data, first extract it, then compute, then summarise."
        ),
        max_steps=max_steps,
        temperature=temperature,
    )
    tools: list[SuluvTool] = [calculator, json_extractor, datetime_now, date_diff]
    if extra_tools:
        tools.extend(extra_tools)
    return SuluvAgent(role=role, llm=llm, tools=tools, checkpointer=checkpointer, **kwargs)



