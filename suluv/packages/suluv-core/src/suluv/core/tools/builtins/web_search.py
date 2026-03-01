"""Web search tool — real web search via ``duckduckgo-search`` library.

Falls back to DuckDuckGo instant-answer API (stdlib only) when the
``duckduckgo-search`` package is not installed.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from suluv.core.tools.decorator import suluv_tool


# ---------------------------------------------------------------------------
# Primary: duckduckgo-search library (returns real search results)
# ---------------------------------------------------------------------------

async def _ddgs_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo using the ``duckduckgo-search`` library."""
    import asyncio

    def _fetch() -> list[dict[str, str]]:
        try:
            from ddgs import DDGS

            results: list[dict[str, str]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })
            if not results:
                results.append({
                    "title": "No results",
                    "snippet": f"No results found for: {query}",
                    "url": "",
                })
            return results
        except Exception as e:
            return [{"title": "Error", "snippet": str(e), "url": ""}]

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)


# ---------------------------------------------------------------------------
# Fallback: DuckDuckGo instant-answer API (no pip dep required)
# ---------------------------------------------------------------------------

async def _instant_answer_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo instant-answer API (no API key needed)."""
    import asyncio

    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"

    def _fetch() -> list[dict[str, str]]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SuluvAgent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results: list[dict[str, str]] = []

            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", "Answer"),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                })

            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                    })

            if not results:
                results.append({
                    "title": "No results",
                    "snippet": f"No instant answers found for: {query}",
                    "url": "",
                })

            return results[:max_results]
        except Exception as e:
            return [{"title": "Error", "snippet": str(e), "url": ""}]

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)


# ---------------------------------------------------------------------------
# Pick the best backend at import time
# ---------------------------------------------------------------------------

try:
    from ddgs import DDGS as _check  # noqa: F401
    _search_fn = _ddgs_search
except ImportError:
    _search_fn = _instant_answer_search


@suluv_tool(
    name="web_search",
    description=(
        "Search the web for information. Returns top results with titles, "
        "snippets, and URLs. Use for factual lookups, current events, or "
        "finding documentation."
    ),
    timeout=15.0,
)
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return results as JSON."""
    # Coerce max_results — some LLMs send it as a string ("5")
    max_results = int(max_results)
    results = await _search_fn(query, max_results)
    return json.dumps(results, indent=2)
