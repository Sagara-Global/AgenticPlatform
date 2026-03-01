"""CorpusRegistry — registers and searches across multiple corpus providers."""

from __future__ import annotations

from suluv.core.ports.corpus_provider import Chunk, CorpusProvider


class CorpusRegistry:
    """Aggregates multiple CorpusProviders for RAG.

    The agent queries all registered providers and merges results
    by relevance score.
    """

    def __init__(self) -> None:
        self._providers: dict[str, CorpusProvider] = {}

    def register(self, name: str, provider: CorpusProvider) -> None:
        self._providers[name] = provider

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)

    async def search(
        self, query: str, top_k: int = 5
    ) -> list[Chunk]:
        """Search all providers and return top-k merged results."""
        all_chunks: list[Chunk] = []
        for provider in self._providers.values():
            chunks = await provider.search(query, top_k=top_k)
            all_chunks.extend(chunks)

        # Sort by score descending and return top-k
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return all_chunks[:top_k]

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())
