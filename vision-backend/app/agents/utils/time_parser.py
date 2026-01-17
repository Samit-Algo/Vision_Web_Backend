"""
Time parsing utility for converting natural language time expressions to ISO 8601 format.

This module handles:
- Natural language parsing ("tomorrow", "next Monday", "coming Sunday")
- Time-only parsing ("09:00" → today at 09:00)
- Already-ISO format passthrough
- Timezone handling (converts to UTC, stores with 'Z' suffix)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import parsedatetime
from pytz import UTC, timezone
import re

from app.utils.datetime_utils import _get_app_timezone, now


# Create parsedatetime calendar instance
_cal = parsedatetime.Calendar()


def _get_app_tz_pytz():
    """
    Get application timezone as pytz timezone object.
    This is needed for parsedatetime compatibility.
    """
    app_tz = _get_app_timezone()
    # Convert zoneinfo ZoneInfo to pytz timezone
    if hasattr(app_tz, 'key'):  # zoneinfo.ZoneInfo
        return timezone(app_tz.key)
    elif app_tz == timezone('UTC'):
        return UTC
    else:
        # Try to get timezone name from the timezone object
        try:
            return timezone(str(app_tz))
        except Exception:
            return UTC


def get_current_datetime_context() -> datetime:
    """
    Get current datetime in application timezone for use as parsing context.
    Times are interpreted in app timezone and then converted to UTC for storage.
    
    Returns:
        Current datetime in application-configured timezone
    """
    return now()


def parse_time_input(
    time_expression: str,
    reference_datetime: Optional[datetime] = None,
    default_time: Optional[str] = None
) -> str:
    """
    Parse a natural language time expression into ISO 8601 format.
    
    This function handles:
    - Natural language: "tomorrow", "next Monday", "coming Sunday", "next week"
    - Time-only: "09:00", "9 AM", "5 PM" (uses reference date)
    - Date + time: "Monday 9", "next Monday 9", "tomorrow 18:00"
    - Already ISO: "2025-11-16T09:00:00Z" (passthrough)
    
    Args:
        time_expression: Natural language time expression from user
        reference_datetime: Reference datetime for relative parsing (defaults to now UTC)
        default_time: Default time if only date is provided (e.g., "00:00")
    
    Returns:
        ISO 8601 formatted datetime string (e.g., "2025-11-16T09:00:00Z")
    
    Examples:
        >>> parse_time_input("tomorrow 09:00")
        "2025-11-17T09:00:00Z"
        
        >>> parse_time_input("next Monday 9")
        "2025-11-24T09:00:00Z"
        
        >>> parse_time_input("09:00")  # Uses today's date
        "2025-11-16T09:00:00Z"
    """
    if not time_expression or not time_expression.strip():
        raise ValueError("Time expression cannot be empty")
    
    time_expression = time_expression.strip()
    
    # Check if already in ISO 8601 format
    iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?$'
    if re.match(iso_pattern, time_expression):
        # Ensure it has 'Z' suffix for UTC
        if not time_expression.endswith('Z'):
            time_expression = time_expression + 'Z'
        return time_expression
    
    # Get application timezone
    app_tz = _get_app_tz_pytz()
    
    # Use current app timezone time as reference if not provided
    if reference_datetime is None:
        reference_datetime = get_current_datetime_context()
    else:
        # Ensure reference is timezone-aware (app timezone)
        if reference_datetime.tzinfo is None:
            reference_datetime = app_tz.localize(reference_datetime)
        elif reference_datetime.tzinfo != app_tz:
            reference_datetime = reference_datetime.astimezone(app_tz)
    
    # Use app timezone as local timezone for parsedatetime (works better with local context)
    # We'll convert to UTC after parsing for storage
    local_ref = reference_datetime.astimezone(app_tz) if reference_datetime.tzinfo else app_tz.localize(reference_datetime)
    
    # Try to parse the time expression
    try:
        # parsedatetime returns (struct_time, parse_status)
        # parse_status: 0 = failed, 1 = parsed as date, 2 = parsed as time, 3 = parsed as datetime
        parsed_time, status = _cal.parseDT(
            datetimeString=time_expression,
            sourceTime=local_ref
        )
        
        if status == 0:
            # Parsing failed, try with some common patterns
            # Try adding "at" or handling time-only expressions
            enhanced_expressions = [
                f"at {time_expression}",
                f"today {time_expression}",
                time_expression
            ]
            
            for expr in enhanced_expressions:
                parsed_time, status = _cal.parseDT(
                    datetimeString=expr,
                    sourceTime=local_ref
                )
                if status > 0:
                    break
            
            if status == 0:
                raise ValueError(f"Could not parse time expression: '{time_expression}'")
        
        # Ensure parsed_time is timezone-aware (assume app timezone if not specified)
        if parsed_time.tzinfo is None:
            parsed_time = app_tz.localize(parsed_time)
        
        # Convert from IST to UTC for storage
        if parsed_time.tzinfo != UTC:
            parsed_time = parsed_time.astimezone(UTC)
        
        # Format as ISO 8601 with 'Z' suffix
        iso_string = parsed_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return iso_string
        
    except Exception as e:
        raise ValueError(f"Error parsing time expression '{time_expression}': {str(e)}")


def format_time_for_display(iso_time: str) -> str:
    """
    Convert ISO 8601 time string to human-readable format for display in application timezone.
    
    Args:
        iso_time: ISO 8601 formatted time string (e.g., "2025-11-16T09:00:00Z")
    
    Returns:
        Human-readable time string in application timezone (e.g., "November 16, 2025 at 14:30")
    """
    if not iso_time:
        return ""
    
    try:
        # Get application timezone
        app_tz = _get_app_tz_pytz()
        app_tz_name = str(app_tz)
        
        # Parse ISO string (assumed to be in UTC)
        if iso_time.endswith('Z'):
            dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(iso_time)
        
        # Convert to UTC if not already
        if dt.tzinfo is None:
            dt = UTC.localize(dt)
        elif dt.tzinfo != UTC:
            dt = dt.astimezone(UTC)
        
        # Convert to app timezone for display
        dt_app = dt.astimezone(app_tz)
        
        # Format for display
        # Check if it's today, tomorrow, or a specific date (in app timezone)
        now_dt = get_current_datetime_context()
        today_start = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_start = today_start + timedelta(days=1)
        
        if dt_app.date() == today_start.date():
            return f"Today at {dt_app.strftime('%H:%M')} {app_tz_name}"
        elif dt_app.date() == tomorrow_start.date():
            return f"Tomorrow at {dt_app.strftime('%H:%M')} {app_tz_name}"
        else:
            return dt_app.strftime(f'%B %d, %Y at %H:%M {app_tz_name}')
            
    except Exception as e:
        # If parsing fails, return the original string
        return iso_time

