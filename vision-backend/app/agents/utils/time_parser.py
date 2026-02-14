"""
Parse natural language time expressions into ISO 8601.

Handles: "tomorrow", "next Monday", "09:00", and full ISO passthrough.
Converts to UTC and returns strings with 'Z' suffix.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import re
from datetime import datetime, timedelta
from typing import Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
import parsedatetime
from pytz import UTC, timezone

# -----------------------------------------------------------------------------
# Application (relative from agents.utils)
# -----------------------------------------------------------------------------
from ...utils.datetime_utils import _get_app_timezone, now

# -----------------------------------------------------------------------------
# Parsedatetime calendar (shared instance)
# -----------------------------------------------------------------------------

parsedatetime_calendar = parsedatetime.Calendar()


# -----------------------------------------------------------------------------
# Timezone helpers
# -----------------------------------------------------------------------------


def get_app_tz_pytz():
    """Return application timezone as pytz timezone for parsedatetime."""
    app_tz = _get_app_timezone()
    if hasattr(app_tz, "key"):
        return timezone(app_tz.key)
    if app_tz == timezone("UTC"):
        return UTC
    try:
        return timezone(str(app_tz))
    except Exception:
        return UTC


def get_current_datetime_context() -> datetime:
    """Return current datetime in application timezone for parsing context."""
    return now()


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------


def parse_time_input(
    time_expression: str,
    reference_datetime: Optional[datetime] = None,
    default_time: Optional[str] = None,
) -> str:
    """
    Parse a natural language time expression into ISO 8601 (UTC, Z suffix).

    Handles: "tomorrow", "next Monday", "09:00", "today 18:00", and full ISO passthrough.
    """
    if not time_expression or not time_expression.strip():
        raise ValueError("Time expression cannot be empty")

    time_expression = time_expression.strip()

    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?$"
    if re.match(iso_pattern, time_expression):
        if not time_expression.endswith("Z"):
            time_expression = time_expression + "Z"
        return time_expression

    app_tz = get_app_tz_pytz()

    if reference_datetime is None:
        reference_datetime = get_current_datetime_context()
    else:
        if reference_datetime.tzinfo is None:
            reference_datetime = app_tz.localize(reference_datetime)
        elif reference_datetime.tzinfo != app_tz:
            reference_datetime = reference_datetime.astimezone(app_tz)

    local_ref = (
        reference_datetime.astimezone(app_tz)
        if reference_datetime.tzinfo
        else app_tz.localize(reference_datetime)
    )

    try:
        parsed_time, status = parsedatetime_calendar.parseDT(
            datetimeString=time_expression,
            sourceTime=local_ref,
        )

        if status == 0:
            for expr in [f"at {time_expression}", f"today {time_expression}", time_expression]:
                parsed_time, status = parsedatetime_calendar.parseDT(
                    datetimeString=expr,
                    sourceTime=local_ref,
                )
                if status > 0:
                    break
            if status == 0:
                raise ValueError(f"Could not parse time expression: '{time_expression}'")

        if parsed_time.tzinfo is None:
            parsed_time = app_tz.localize(parsed_time)
        if parsed_time.tzinfo != UTC:
            parsed_time = parsed_time.astimezone(UTC)

        return parsed_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error parsing time expression '{time_expression}': {e}") from e


def format_time_for_display(iso_time: str) -> str:
    """Convert ISO 8601 string to human-readable format in application timezone."""
    if not iso_time:
        return ""

    try:
        app_tz = get_app_tz_pytz()
        app_tz_name = str(app_tz)

        if iso_time.endswith("Z"):
            dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(iso_time)

        if dt.tzinfo is None:
            dt = UTC.localize(dt)
        elif dt.tzinfo != UTC:
            dt = dt.astimezone(UTC)

        dt_app = dt.astimezone(app_tz)
        now_dt = get_current_datetime_context()
        today_start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_start = today_start + timedelta(days=1)

        if dt_app.date() == today_start.date():
            return f"Today at {dt_app.strftime('%H:%M')} {app_tz_name}"
        if dt_app.date() == tomorrow_start.date():
            return f"Tomorrow at {dt_app.strftime('%H:%M')} {app_tz_name}"
        return dt_app.strftime(f"%B %d, %Y at %H:%M {app_tz_name}")

    except Exception:
        return iso_time
