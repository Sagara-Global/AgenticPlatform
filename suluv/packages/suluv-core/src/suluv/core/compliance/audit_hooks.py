"""AuditHooks — automatic audit event generation for key system events."""

from __future__ import annotations

from typing import Any

from suluv.core.ports.audit_backend import AuditBackend
from suluv.core.types import AuditEvent, OrgID, UserID, SessionID


class AuditHooks:
    """Convenience class for emitting structured audit events.

    Used by framework internals and graph middleware to log
    security-relevant and compliance-relevant events.
    """

    def __init__(self, backend: AuditBackend) -> None:
        self._backend = backend

    async def log_agent_start(
        self,
        agent_name: str,
        task: str,
        org_id: OrgID | None = None,
        user_id: UserID | None = None,
        session_id: SessionID | None = None,
        thread_id: str | None = None,
    ) -> None:
        await self._backend.write(AuditEvent(
            event_type="agent_start",
            org_id=org_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            data={"agent": agent_name, "task": task[:500]},
        ))

    async def log_agent_end(
        self,
        agent_name: str,
        success: bool,
        steps: int = 0,
        tokens: int = 0,
        org_id: OrgID | None = None,
        user_id: UserID | None = None,
        thread_id: str | None = None,
    ) -> None:
        await self._backend.write(AuditEvent(
            event_type="agent_end",
            org_id=org_id,
            user_id=user_id,
            thread_id=thread_id,
            data={
                "agent": agent_name,
                "success": success,
                "steps": steps,
                "tokens": tokens,
            },
        ))

    async def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        user_id: UserID | None = None,
        thread_id: str | None = None,
    ) -> None:
        await self._backend.write(AuditEvent(
            event_type="tool_call",
            user_id=user_id,
            thread_id=thread_id,
            data={
                "tool": tool_name,
                "arguments": arguments,
                "error": error,
                "success": error is None,
            },
        ))

    async def log_guardrail_block(
        self,
        content_preview: str,
        direction: str,  # "input" or "output"
        reason: str,
        user_id: UserID | None = None,
        thread_id: str | None = None,
    ) -> None:
        await self._backend.write(AuditEvent(
            event_type="guardrail_block",
            user_id=user_id,
            thread_id=thread_id,
            data={
                "direction": direction,
                "reason": reason,
                "preview": content_preview[:200],
            },
        ))

    async def log_consent_check(
        self,
        purpose: str,
        granted: bool,
        user_id: UserID | None = None,
        thread_id: str | None = None,
    ) -> None:
        await self._backend.write(AuditEvent(
            event_type="consent_check",
            user_id=user_id,
            thread_id=thread_id,
            data={"purpose": purpose, "granted": granted},
        ))

    async def log_custom(
        self,
        event_type: str,
        data: dict[str, Any],
        org_id: OrgID | None = None,
        user_id: UserID | None = None,
        thread_id: str | None = None,
    ) -> None:
        await self._backend.write(AuditEvent(
            event_type=event_type,
            org_id=org_id,
            user_id=user_id,
            thread_id=thread_id,
            data=data,
        ))
