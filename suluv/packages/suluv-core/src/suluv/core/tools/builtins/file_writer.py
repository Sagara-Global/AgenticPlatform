"""File writer tool — write content to files safely."""

from __future__ import annotations

from pathlib import Path

from suluv.core.tools.decorator import suluv_tool


@suluv_tool(
    name="file_writer",
    description=(
        "Write content to a file. Creates the file if it doesn't exist, "
        "or overwrites it if it does. Set append=true to append instead. "
        "Parent directories are created automatically."
    ),
)
def file_writer(path: str, content: str, append: bool = False) -> str:
    """Write content to a file."""
    try:
        file_path = Path(path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        file_path.write_text(content, encoding="utf-8") if not append else \
            file_path.open(mode, encoding="utf-8").write(content)

        return f"Successfully {'appended to' if append else 'wrote'} {file_path} ({len(content)} chars)"
    except PermissionError:
        return f"Error: Permission denied — {path}"
    except Exception as e:
        return f"Error writing file: {e}"
