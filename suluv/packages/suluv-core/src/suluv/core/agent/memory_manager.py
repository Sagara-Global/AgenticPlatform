"""MemoryManager — wires the 4 memory tiers to an agent."""

from __future__ import annotations

import uuid
from typing import Any

from suluv.core.ports.memory_backend import (
    EpisodicMemory,
    LongTermMemory,
    MemoryEntry,
    SemanticMemory,
    ShortTermMemory,
)


class MemoryManager:
    """Manages agent memory across four tiers.

    - **short_term**: current session context (auto-cleared)
    - **long_term**: cross-session, org/user-scoped
    - **episodic**: past interaction recall (recency-weighted)
    - **semantic**: vector similarity search over knowledge

    Used by SuluvAgent at the start and end of each run().
    """

    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self.episodic = episodic
        self.semantic = semantic

    async def load_context(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        query: str | None = None,
        max_entries: int = 10,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Load relevant memory at the start of an agent run.

        Returns a dict with available memory tier data.
        When *thread_id* is provided the short-term scope is narrowed
        to entries saved under that thread.
        """
        context: dict[str, Any] = {}
        # Use thread_id as scope prefix when available
        scope_key = thread_id or session_id

        if self.short_term and scope_key:
            # Short-term is a KV store; filter by scope prefix
            all_items = await self.short_term.all()
            prefix = f"t:{scope_key}:"
            scoped = {k: v for k, v in all_items.items() if k.startswith(prefix)}
            context["short_term"] = scoped if scoped else all_items

        if self.long_term and user_id:
            # Long-term is user-scoped KV; get keys for this user scope
            keys = await self.long_term.list_keys(scope=user_id)
            entries = []
            for key in keys[:max_entries]:
                val = await self.long_term.get(key, scope=user_id)
                if val is not None:
                    entries.append(MemoryEntry(key=key, value=val))
            context["long_term"] = entries

        if self.episodic and query:
            # Episodic returns list[dict] based on query
            episodes = await self.episodic.recall(query, limit=max_entries)
            entries = [
                MemoryEntry(key=str(i), value=ep)
                for i, ep in enumerate(episodes)
            ]
            context["episodic"] = entries

        if self.semantic and query:
            # Semantic returns list[MemoryEntry]
            results = await self.semantic.search(query, limit=max_entries)
            context["semantic"] = results

        return context

    async def save_interaction(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        content: str = "",
        metadata: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Save interaction to relevant memory tiers."""
        raw_key = uuid.uuid4().hex[:12]
        # Scope short-term key by thread when available
        scope_key = thread_id or session_id
        interaction_key = f"t:{scope_key}:{raw_key}" if scope_key else raw_key

        if self.short_term and scope_key:
            await self.short_term.set(interaction_key, content)

        if self.long_term and user_id:
            await self.long_term.set(
                interaction_key, content, scope=user_id
            )

        if self.episodic:
            episode_meta = dict(metadata or {})
            if thread_id:
                episode_meta["thread_id"] = thread_id
            await self.episodic.store({
                "key": interaction_key,
                "content": content,
                "metadata": episode_meta,
            })

    async def clear_session(self) -> None:
        """Clear short-term memory for current session."""
        if self.short_term:
            await self.short_term.clear()
