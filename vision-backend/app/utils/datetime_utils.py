"""
Centralized DateTime Utilities
==============================

Provides consistent datetime handling across the entire application.
All datetime operations use the timezone configured in app.core.config.

Functions:
- now(): Returns timezone-aware datetime object
- now_iso(): Returns ISO 8601 string (replaces iso_now)
- parse_iso(): Safely parse ISO 8601 string to datetime
- to_iso(): Convert datetime object to ISO 8601 string

This ensures all datetime operations (create, update, start, stop, etc.)
use the same timezone consistently.

Adapted for vision-backend: Uses local_timezone instead of timezone.
"""
from datetime import datetime, timezone as dt_timezone
from typing import Optional

try:
    import zoneinfo
except ImportError:
    # Python < 3.9 fallback
    try:
        from backports import zoneinfo  # type: ignore
    except ImportError:
        zoneinfo = None

from ..core.config import get_settings


def _get_app_timezone() -> dt_timezone:
    """
    Get the application timezone from config.
    Returns timezone object (defaults to UTC if invalid).
    """
    settings = get_settings()
    # Vision-backend uses local_timezone instead of timezone
    tz_str = settings.local_timezone
    
    # Handle UTC explicitly
    if tz_str.upper() == "UTC":
        return dt_timezone.utc
    
    # Try to get timezone from zoneinfo (Python 3.9+)
    if zoneinfo is not None:
        try:
            return zoneinfo.ZoneInfo(tz_str)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            # Fallback to UTC if timezone is invalid
            print(f"[datetime_utils] ⚠️  Invalid timezone '{tz_str}', falling back to UTC")
            return dt_timezone.utc
    
    # Fallback to UTC if zoneinfo is not available
    print(f"[datetime_utils] ⚠️  zoneinfo not available, using UTC")
    return dt_timezone.utc


def utc_now() -> datetime:
    """
    Get current UTC time as a timezone-aware datetime.

    Use this for all timestamps that will be persisted to MongoDB (BSON Date).
    """
    return datetime.now(dt_timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalize a datetime into a timezone-aware UTC datetime.

    - If dt is None -> None
    - If dt is naive -> assume it represents UTC (this matches MongoDB/PyMongo behavior)
    - If dt is aware -> convert to UTC
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc)


def now() -> datetime:
    """
    Get current datetime with application-configured timezone.
    
    Returns:
        timezone-aware datetime object
    """
    app_tz = _get_app_timezone()
    return datetime.now(app_tz)


def now_iso() -> str:
    """
    Get current datetime as ISO 8601 string with application-configured timezone.
    
    This replaces the old iso_now() function and ensures consistent timezone usage.
    
    Returns:
        ISO 8601 formatted string (e.g., "2025-12-24T10:30:00+05:30" or "2025-12-24T10:30:00Z")
    """
    dt = now()
    # Format with timezone offset, or 'Z' if UTC
    if dt.tzinfo == dt_timezone.utc:
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return dt.replace(microsecond=0).isoformat()


def parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO 8601 string to datetime object.
    Handles both timezone-aware and naive strings.
    If string is naive, assumes application timezone.
    
    Args:
        dt_str: ISO 8601 string (e.g., "2025-12-24T10:30:00Z" or "2025-12-24T10:30:00+05:30")
    
    Returns:
        timezone-aware datetime object, or None if parsing fails
    """
    if not dt_str:
        return None
    
    try:
        # Replace 'Z' with '+00:00' for parsing
        normalized = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        
        # If timezone-naive, assume application timezone
        if dt.tzinfo is None:
            app_tz = _get_app_timezone()
            dt = dt.replace(tzinfo=app_tz)
        
        return dt
    except Exception:
        return None


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Convert datetime object to ISO 8601 string.
    If datetime is naive, assumes application timezone.
    
    Args:
        dt: datetime object (timezone-aware or naive)
    
    Returns:
        ISO 8601 formatted string, or None if dt is None
    """
    if dt is None:
        return None
    
    # If naive, assume application timezone
    if dt.tzinfo is None:
        app_tz = _get_app_timezone()
        dt = dt.replace(tzinfo=app_tz)
    
    # Format with timezone offset, or 'Z' if UTC
    if dt.tzinfo == dt_timezone.utc:
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return dt.replace(microsecond=0).isoformat()


def mongo_datetime_to_app_timezone(mongo_dt: Optional[datetime]) -> Optional[datetime]:
    """
    Convert MongoDB datetime object to application timezone.
    
    MongoDB stores all datetimes as UTC, but PyMongo returns them as naive
    datetime objects (representing UTC time). This function properly converts
    them to timezone-aware datetimes in the application's configured timezone.
    
    Args:
        mongo_dt: datetime object from MongoDB (naive, representing UTC)
                 or None
    
    Returns:
        timezone-aware datetime object in application timezone, or None
    
    Example:
        >>> mongo_dt = datetime(2026, 1, 11, 9, 46, 0)  # Naive, represents UTC
        >>> app_dt = mongo_datetime_to_app_timezone(mongo_dt)
        >>> # If app timezone is Asia/Kolkata (UTC+5:30):
        >>> # app_dt = datetime(2026, 1, 11, 15, 16, 0, tzinfo=ZoneInfo('Asia/Kolkata'))
    """
    if mongo_dt is None:
        return None
    
    # If already timezone-aware, convert to app timezone
    if mongo_dt.tzinfo is not None:
        app_tz = _get_app_timezone()
        return mongo_dt.astimezone(app_tz)
    
    # MongoDB naive datetime = UTC time, so mark it as UTC first
    mongo_dt_utc = mongo_dt.replace(tzinfo=dt_timezone.utc)
    
    # Convert to application timezone
    app_tz = _get_app_timezone()
    return mongo_dt_utc.astimezone(app_tz)


def mongo_datetime_to_utc(mongo_dt: Optional[datetime]) -> Optional[datetime]:
    """
    Convert MongoDB datetime object to timezone-aware UTC.

    MongoDB stores all datetimes as UTC, but PyMongo/Motor often return them as naive
    datetime objects representing UTC. This helper makes them explicitly UTC-aware.
    """
    return ensure_utc(mongo_dt)


# Backward compatibility: alias for iso_now
iso_now = now_iso
