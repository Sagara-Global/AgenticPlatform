"""File reader tool — read file contents safely."""

from __future__ import annotations

import os
from pathlib import Path

from suluv.core.tools.decorator import suluv_tool


@suluv_tool(
    name="file_reader",
    description=(
        "Read the contents of a file from the local filesystem. "
        "Provide the file path and optionally limit the number of lines. "
        "Supports text files only."
    ),
)
def file_reader(path: str, max_lines: int = 500) -> str:
    """Read a file and return its text contents."""
    try:
        file_path = Path(path).resolve()

        if not file_path.exists():
            return f"Error: File not found — {path}"
        if not file_path.is_file():
            return f"Error: Not a file — {path}"

        size = file_path.stat().st_size
        if size > 10 * 1024 * 1024:  # 10 MB limit
            return f"Error: File too large ({size:,} bytes). Max 10 MB."

        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        if len(lines) > max_lines:
            result = "".join(lines[:max_lines])
            result += f"\n\n[Truncated: showing {max_lines} of {len(lines)} lines]"
            return result

        return text
    except PermissionError:
        return f"Error: Permission denied — {path}"
    except Exception as e:
        return f"Error reading file: {e}"
