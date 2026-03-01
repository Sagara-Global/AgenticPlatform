"""InMemoryBusinessCalendar — working hours and holiday calculation."""

from __future__ import annotations

from datetime import datetime, timedelta, time, date
from typing import Any

from suluv.core.ports.business_calendar import BusinessCalendarPort


class InMemoryBusinessCalendar(BusinessCalendarPort):
    """In-memory business calendar with configurable working hours and holidays."""

    def __init__(
        self,
        start_hour: int = 9,
        end_hour: int = 18,
        working_days: list[int] | None = None,  # 0=Mon, 6=Sun
        holidays: list[date] | None = None,
        timezone_name: str = "UTC",
    ) -> None:
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.working_days = working_days or [0, 1, 2, 3, 4]  # Mon-Fri
        self.holidays = set(holidays or [])
        self.timezone_name = timezone_name
        self._hours_per_day = end_hour - start_hour

    def _is_working_day(self, d: date) -> bool:
        return d.weekday() in self.working_days and d not in self.holidays

    async def is_working_time(self, dt: datetime) -> bool:
        if not self._is_working_day(dt.date()):
            return False
        return time(self.start_hour) <= dt.time() < time(self.end_hour)

    async def add_business_hours(self, dt: datetime, hours: float | timedelta) -> datetime:
        if isinstance(hours, timedelta):
            hours = hours.total_seconds() / 3600.0
        if hours <= 0:
            return dt

        remaining = hours
        current = dt

        # If before working hours, move to start
        if current.time() < time(self.start_hour):
            current = current.replace(
                hour=self.start_hour, minute=0, second=0, microsecond=0
            )
        # If after working hours, move to next working day
        elif current.time() >= time(self.end_hour):
            current = current + timedelta(days=1)
            current = current.replace(
                hour=self.start_hour, minute=0, second=0, microsecond=0
            )

        # Skip to next working day if needed
        while not self._is_working_day(current.date()):
            current = current + timedelta(days=1)
            current = current.replace(
                hour=self.start_hour, minute=0, second=0, microsecond=0
            )

        while remaining > 0:
            if not self._is_working_day(current.date()):
                current = current + timedelta(days=1)
                current = current.replace(
                    hour=self.start_hour, minute=0, second=0, microsecond=0
                )
                continue

            end_of_day = current.replace(
                hour=self.end_hour, minute=0, second=0, microsecond=0
            )
            available = (end_of_day - current).total_seconds() / 3600.0

            if remaining <= available:
                current = current + timedelta(hours=remaining)
                remaining = 0
            else:
                remaining -= available
                current = current + timedelta(days=1)
                current = current.replace(
                    hour=self.start_hour, minute=0, second=0, microsecond=0
                )

        return current

    async def business_hours_between(self, start: datetime, end: datetime) -> float:
        if start >= end:
            return 0.0

        total = 0.0
        current = start

        while current.date() <= end.date():
            if self._is_working_day(current.date()):
                day_start = max(
                    current,
                    current.replace(hour=self.start_hour, minute=0, second=0, microsecond=0),
                )
                day_end = min(
                    end,
                    current.replace(hour=self.end_hour, minute=0, second=0, microsecond=0),
                )
                if day_start < day_end:
                    total += (day_end - day_start).total_seconds() / 3600.0

            current = (current + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        return total

    async def next_working_time(self, dt: datetime) -> datetime:
        if await self.is_working_time(dt):
            return dt

        current = dt
        if current.time() >= time(self.end_hour):
            current = current + timedelta(days=1)

        current = current.replace(
            hour=self.start_hour, minute=0, second=0, microsecond=0
        )

        while not self._is_working_day(current.date()):
            current = current + timedelta(days=1)

        return current
