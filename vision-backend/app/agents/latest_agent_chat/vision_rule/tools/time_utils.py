"""
Natural language time parsing for vision rules.
Handles ISO, HH:MM, and expressions like "now", "in 10 min", "tomorrow 3pm", etc.
"""

import re
from datetime import datetime, timezone

import dateparser


def _preprocess_relative(text: str) -> str:
    """Convert 'after X min/hours' to 'in X minutes/hours' for dateparser."""
    if not text:
        return text
    lower = text.strip().lower()
    # "after 10 min", "after 30 minutes", "after 2 hours", etc.
    m = re.match(
        r"^\s*after\s+(\d+)\s*(min(?:ute)?s?|hrs?|hours?)\s*$",
        lower,
        re.IGNORECASE,
    )
    if m:
        num, unit = m.group(1), m.group(2)
        if "min" in unit:
            return f"in {num} minutes"
        if "hr" in unit or "hour" in unit:
            return f"in {num} hours"
    return text


def parse_natural_time(
    time_input: str,
    reference_datetime: datetime | None = None,
) -> datetime:
    """
    Parse user input into a timezone-aware datetime (UTC).

    Supports:
    - ISO: 2025-02-20T14:30:00+00:00, 2025-02-20 14:30:00
    - Time-only: 08:00, 14:30:00
    - Natural language: now, in 10 min, after 10 min, tomorrow 3pm, tomorrow 7, next monday 9am

    Args:
        time_input: Raw string from user (e.g. "now", "tomorrow 3", "after 10 min")
        reference_datetime: Reference for relative times (e.g. start_time for "end after 10 min").
            Defaults to now (UTC).

    Returns:
        timezone-aware datetime in UTC.

    Raises:
        ValueError: If parsing fails.
    """
    if not time_input or not isinstance(time_input, str):
        raise ValueError("Empty or invalid time")

    raw = time_input.strip()
    ref = reference_datetime or datetime.now(timezone.utc)
    normalized = raw.replace("Z", "+00:00")

    # 1. Try ISO format
    try:
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # 2. Try space-separated date-time (2025-02-20 14:30:00)
    if " " in normalized and "T" not in normalized:
        try:
            dt = datetime.fromisoformat(normalized.replace(" ", "T", 1))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

    # 3. Try time-only HH:MM or HH:MM:SS (today in UTC)
    if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", normalized):
        parts = normalized.split(":")
        hour, minute = int(parts[0]), int(parts[1])
        second = int(parts[2]) if len(parts) > 2 else 0
        today = ref.date()
        return datetime(
            today.year, today.month, today.day,
            hour, minute, second,
            tzinfo=timezone.utc,
        )

    # 4. Natural language via dateparser
    preprocessed = _preprocess_relative(raw)
    settings: dict = {
        "RELATIVE_BASE": ref,
        "TIMEZONE": "UTC",
        "RETURN_AS_TIMEZONE_AWARE": True,
    }
    parsed = dateparser.parse(preprocessed, settings=settings)
    if parsed is None:
        raise ValueError(
            f"Cannot parse time '{raw}'. Try ISO (e.g. 2025-02-20T14:30:00+00:00), "
            "HH:MM, or phrases like 'now', 'in 10 min', 'tomorrow 3pm'."
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_to_iso(time_input: str, reference_datetime: datetime | None = None) -> str:
    """Parse time input and return ISO string (UTC)."""
    dt = parse_natural_time(time_input, reference_datetime)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
