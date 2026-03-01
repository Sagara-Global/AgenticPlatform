"""ConsentEnforcer — checks consent before processing data."""

from __future__ import annotations

from suluv.core.ports.consent_provider import ConsentProvider, ConsentResult
from suluv.core.types import UserID


class ConsentEnforcer:
    """Ensures data processing only happens with valid consent.

    Wraps one or more ConsentProviders and checks before
    any data use that requires user consent.
    """

    def __init__(self, provider: ConsentProvider) -> None:
        self._provider = provider

    async def require_consent(
        self,
        user_id: UserID,
        purpose: str,
        data_categories: list[str] | None = None,
        thread_id: str | None = None,
    ) -> ConsentResult:
        """Check consent. Raises ConsentRequired if not granted."""
        context = {
            "user_id": user_id,
            "thread_id": thread_id,
            "data_categories": data_categories or [],
        }
        result = await self._provider.check(context, purpose)
        if not result.granted:
            raise ConsentRequired(
                f"Consent required for '{purpose}': {result.reason}"
            )
        return result

    async def check_consent(
        self,
        user_id: UserID,
        purpose: str,
        data_categories: list[str] | None = None,
        thread_id: str | None = None,
    ) -> bool:
        """Soft check — returns True/False without raising."""
        context = {
            "user_id": user_id,
            "thread_id": thread_id,
            "data_categories": data_categories or [],
        }
        result = await self._provider.check(context, purpose)
        return result.granted


class ConsentRequired(Exception):
    """Raised when required consent has not been granted."""
    pass
