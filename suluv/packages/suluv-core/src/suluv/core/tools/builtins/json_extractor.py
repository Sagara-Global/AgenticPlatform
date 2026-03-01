"""JSON extractor tool — extract fields from JSON text using JSONPath-like expressions."""

from __future__ import annotations

import json
from typing import Any

from suluv.core.tools.decorator import suluv_tool


def _extract_path(data: Any, path: str) -> Any:
    """Extract a value from nested data using dot-notation path.

    Supports: 'key.subkey', 'key[0]', 'key[*].field' (map over arrays).
    """
    parts = path.replace("[", ".[").split(".")
    current = data

    for part in parts:
        if not part:
            continue
        if part == "[*]":
            if isinstance(current, list):
                continue  # handled in next iteration
            raise KeyError(f"Cannot iterate over non-list with [*]")
        elif part.startswith("[") and part.endswith("]"):
            idx_str = part[1:-1]
            if idx_str == "*":
                # handled above
                continue
            idx = int(idx_str)
            if isinstance(current, list):
                current = current[idx]
            else:
                raise KeyError(f"Cannot index non-list with [{idx}]")
        else:
            # Check if previous step was [*] and current is a list
            if isinstance(current, list):
                current = [
                    item.get(part) if isinstance(item, dict) else item
                    for item in current
                ]
            elif isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"Key '{part}' not found")
                current = current[part]
            else:
                raise KeyError(f"Cannot access '{part}' on {type(current).__name__}")

    return current


@suluv_tool(
    name="json_extractor",
    description=(
        "Parse a JSON string and extract values using dot-notation paths. "
        "Supports nested access (key.subkey), array indexing (key[0]), "
        "and array mapping (items[*].name). "
        "Input: json_text — the JSON string, path — the extraction path. "
        "Returns the extracted value as a JSON string."
    ),
)
def json_extractor(json_text: str, path: str = "") -> str:
    """Parse JSON and extract a value at the given path."""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {e}"

    if not path:
        return json.dumps(data, indent=2, default=str)

    try:
        result = _extract_path(data, path)
        return json.dumps(result, indent=2, default=str)
    except (KeyError, IndexError, TypeError) as e:
        return f"Error: {e}"
