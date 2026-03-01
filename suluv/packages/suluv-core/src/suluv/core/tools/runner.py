"""SandboxedToolRunner — executes tools with timeout, audit, and error isolation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from suluv.core.tools.decorator import SuluvTool
from suluv.core.types import AuditEvent

logger = logging.getLogger("suluv.tools")


class ToolRunner:
    """Execute tools with safety guardrails.

    - Timeout enforcement
    - Error isolation (tool failure doesn't crash agent)
    - Audit logging of every tool call
    """

    def __init__(
        self,
        audit_backend: Any | None = None,
        default_timeout: float = 30.0,
    ) -> None:
        self._audit = audit_backend
        self._default_timeout = default_timeout
        self._call_history: list[dict[str, Any]] = []

    async def run(
        self,
        tool: SuluvTool,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a tool and return {"result": ..., "error": ...}.

        *context* carries identity/thread info (thread_id, user_id,
        org_id, session_id) so that audit events and call history
        are properly scoped.
        """
        ctx = context or {}
        timeout = tool.timeout or self._default_timeout
        started = datetime.now(timezone.utc)

        record: dict[str, Any] = {
            "tool": tool.name,
            "arguments": arguments,
            "started_at": started.isoformat(),
            "thread_id": ctx.get("thread_id"),
            "user_id": ctx.get("user_id"),
            "org_id": ctx.get("org_id"),
            "session_id": ctx.get("session_id"),
        }

        try:
            result = await asyncio.wait_for(
                tool.execute(**arguments), timeout=timeout
            )
            record["result"] = result
            record["success"] = True
            self._call_history.append(record)

            if self._audit:
                await self._audit.write(AuditEvent(
                    event_type="tool_call",
                    org_id=ctx.get("org_id"),
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    thread_id=ctx.get("thread_id"),
                    data={
                        "tool": tool.name,
                        "success": True,
                    },
                ))

            return {"result": result, "error": None}

        except asyncio.TimeoutError:
            error = f"Tool '{tool.name}' timed out after {timeout}s"
            logger.warning(error)
            record["error"] = error
            record["success"] = False
            self._call_history.append(record)
            return {"result": None, "error": error}

        except Exception as exc:
            error = f"Tool '{tool.name}' failed: {exc}"
            logger.error(error)
            record["error"] = error
            record["success"] = False
            self._call_history.append(record)

            if self._audit:
                await self._audit.write(AuditEvent(
                    event_type="tool_error",
                    org_id=ctx.get("org_id"),
                    user_id=ctx.get("user_id"),
                    session_id=ctx.get("session_id"),
                    thread_id=ctx.get("thread_id"),
                    data={
                        "tool": tool.name,
                        "error": str(exc),
                    },
                ))

            return {"result": None, "error": error}

    @property
    def call_history(self) -> list[dict[str, Any]]:
        return list(self._call_history)
