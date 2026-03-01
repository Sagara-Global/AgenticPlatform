"""CancellationToken — cooperative cancellation for graph execution."""

from __future__ import annotations

import asyncio


class CancellationToken:
    """Cooperative cancellation token.

    Checked by GraphRuntime before dispatching each frontier node,
    and by agents between ReAct steps.
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation. Returns True if cancelled."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def check(self) -> None:
        """Raise if cancelled. Use between steps."""
        if self._cancelled:
            raise CancellationError("Operation cancelled")


class CancellationError(Exception):
    """Raised when a cancellation token is checked after cancellation."""
    pass
