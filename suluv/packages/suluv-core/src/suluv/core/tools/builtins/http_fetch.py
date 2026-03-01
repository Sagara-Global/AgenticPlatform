"""HTTP fetch tool — fetch content from a URL."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from suluv.core.tools.decorator import suluv_tool


@suluv_tool(
    name="http_fetch",
    description=(
        "Fetch content from a URL via HTTP GET. Returns the response body "
        "as text. Useful for fetching APIs, web pages, or raw data. "
        "Set max_chars to limit response size."
    ),
    timeout=15.0,
)
async def http_fetch(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and return the response text."""
    max_chars = int(max_chars)  # Gemini may send string "5000"

    def _fetch() -> str:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "SuluvAgent/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8", errors="replace")

                if len(body) > max_chars:
                    body = body[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"

                return body
        except Exception as e:
            return f"Error fetching {url}: {e}"

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)
