"""Adapters — in-memory (dev/test) plus optional real LLM backends."""

from suluv.core.adapters.memory_bus import InMemoryEventBus
from suluv.core.adapters.memory_state import InMemoryStateStore
from suluv.core.adapters.memory_audit import InMemoryAuditBackend
from suluv.core.adapters.memory_memory import (
    InMemoryShortTermMemory,
    InMemoryLongTermMemory,
    InMemoryEpisodicMemory,
    InMemorySemanticMemory,
)
from suluv.core.adapters.memory_tasks import InMemoryHumanTaskQueue
from suluv.core.adapters.memory_rules import InMemoryRulesEngine
from suluv.core.adapters.memory_calendar import InMemoryBusinessCalendar
from suluv.core.adapters.memory_instances import InMemoryProcessInstanceStore
from suluv.core.adapters.memory_templates import InMemoryTemplateEngine
from suluv.core.adapters.memory_checkpointer import InMemoryCheckpointer
from suluv.core.adapters.mock_llm import MockLLM

# Optional real LLM backends (require extra deps)
try:
    from suluv.core.adapters.openai_llm import OpenAIBackend
except ImportError:
    OpenAIBackend = None  # type: ignore[assignment,misc]

try:
    from suluv.core.adapters.anthropic_llm import AnthropicBackend
except ImportError:
    AnthropicBackend = None  # type: ignore[assignment,misc]

try:
    from suluv.core.adapters.gemini_llm import GeminiBackend
except ImportError:
    GeminiBackend = None  # type: ignore[assignment,misc]

__all__ = [
    "InMemoryEventBus",
    "InMemoryStateStore",
    "InMemoryAuditBackend",
    "InMemoryShortTermMemory",
    "InMemoryLongTermMemory",
    "InMemoryEpisodicMemory",
    "InMemorySemanticMemory",
    "InMemoryHumanTaskQueue",
    "InMemoryRulesEngine",
    "InMemoryBusinessCalendar",
    "InMemoryProcessInstanceStore",
    "InMemoryTemplateEngine",
    "InMemoryCheckpointer",
    "MockLLM",
    "OpenAIBackend",
    "AnthropicBackend",
    "GeminiBackend",
]
