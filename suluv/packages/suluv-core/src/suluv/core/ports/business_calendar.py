"""BusinessCalendarPort — working hours, holidays, and business-time calculations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class BusinessCalendarPort(ABC):
    """Port for business calendar operations."""

    @abstractmethod
    async def is_working_time(self, dt: datetime) -> bool:
        """Check if a datetime falls within working hours on a working day."""
        ...

    @abstractmethod
    async def add_business_hours(self, dt: datetime, hours: float) -> datetime:
        """Add business hours to a datetime, skipping non-working time."""
        ...

    @abstractmethod
    async def business_hours_between(self, start: datetime, end: datetime) -> float:
        """Calculate business hours between two datetimes."""
        ...

    @abstractmethod
    async def next_working_time(self, dt: datetime) -> datetime:
        """Find the next working time at or after the given datetime."""
        ...
