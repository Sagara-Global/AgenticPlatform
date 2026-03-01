"""Shell exec tool — run shell commands safely."""

from __future__ import annotations

import asyncio
import subprocess

from suluv.core.tools.decorator import suluv_tool


@suluv_tool(
    name="shell_exec",
    description=(
        "Execute a shell command and return its output. "
        "Use for running scripts, checking system info, or invoking CLIs. "
        "Commands are run with a 30-second timeout by default."
    ),
    timeout=35.0,
)
async def shell_exec(command: str, timeout_seconds: int = 30) -> str:
    """Execute a shell command and return stdout + stderr."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )

        output = ""
        if stdout:
            output += stdout.decode("utf-8", errors="replace")
        if stderr:
            output += "\n[STDERR]\n" + stderr.decode("utf-8", errors="replace")

        exit_code = proc.returncode
        if exit_code != 0:
            output += f"\n[Exit code: {exit_code}]"

        # Truncate if too large
        if len(output) > 10000:
            output = output[:10000] + "\n\n[Output truncated at 10000 chars]"

        return output.strip() or "(no output)"

    except asyncio.TimeoutError:
        return f"Error: Command timed out after {timeout_seconds}s"
    except Exception as e:
        return f"Error executing command: {e}"
