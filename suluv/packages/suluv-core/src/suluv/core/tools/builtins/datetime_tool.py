"""Datetime tools — current time and date arithmetic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from suluv.core.tools.decorator import suluv_tool


@suluv_tool(
    name="datetime_now",
    description=(
        "Get the current date and time. "
        "Optionally specify a UTC offset in hours (e.g. 5.5 for IST, -5 for EST)."
    ),
)
async def datetime_now(utc_offset_hours: float = 0.0) -> str:
    """Return current datetime in ISO format with optional timezone offset."""
    tz = timezone(timedelta(hours=utc_offset_hours))
    now = datetime.now(tz)
    return now.isoformat()


@suluv_tool(
    name="date_diff",
    description=(
        "Calculate the difference between two dates/datetimes. "
        "Accepts ISO 8601 strings (e.g. '2024-01-15' or '2024-01-15T10:30:00'). "
        "Returns a human-readable duration."
    ),
)
async def date_diff(start: str, end: str) -> str:
    """Calculate difference between two ISO datetime strings."""
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    def _parse(s: str) -> datetime:
        for fmt in formats:
            try:
                return datetime.strptime(s.strip(), fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse datetime: {s!r}")

    try:
        dt_start = _parse(start)
        dt_end = _parse(end)
    except ValueError as e:
        return f"Error: {e}"

    delta = dt_end - dt_start
    total_seconds = int(delta.total_seconds())
    negative = total_seconds < 0
    total_seconds = abs(total_seconds)

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts: list[str] = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    result = ", ".join(parts)
    if negative:
        result += " (end is before start)"

    return result
