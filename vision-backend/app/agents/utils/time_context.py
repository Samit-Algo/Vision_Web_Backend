"""
Time context helpers for agent prompts.

Provides current time in UTC and application timezone for use in LLM instructions.
Timezone is validated on first use; invalid names are logged and UTC is used.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import logging
from datetime import datetime
from typing import Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from pytz import UTC, timezone
from pytz.exceptions import UnknownTimeZoneError

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...core.config import get_settings

logger = logging.getLogger(__name__)

_cached_local_tz: Optional[object] = None


def get_local_tz():
    """Return application timezone from config (e.g. Asia/Kolkata). Validates once and caches."""
    global _cached_local_tz
    if _cached_local_tz is not None:
        return _cached_local_tz
    tz_str = (get_settings().local_timezone or "UTC").strip()
    if not tz_str:
        tz_str = "UTC"
    try:
        _cached_local_tz = timezone(tz_str)
        return _cached_local_tz
    except UnknownTimeZoneError:
        logger.warning("Invalid LOCAL_TIMEZONE %r; using UTC", tz_str)
        _cached_local_tz = UTC
        return _cached_local_tz


def get_current_time_context() -> str:
    """
    Return current time context for the LLM prompt.

    Includes machine-readable and human-readable UTC and local time.
    Uses LOCAL_TIMEZONE from config.
    """
    local_tz = get_local_tz()
    tz_name = get_settings().local_timezone or "UTC"

    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(local_tz)

    now_utc_iso_z = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_local_iso = now_local.isoformat()
    now_epoch_sec = int(now_utc.timestamp())
    now_local_human = now_local.strftime("%A, %B %d, %Y at %H:%M:%S %Z")
    now_utc_human = now_utc.strftime("%A, %B %d, %Y at %H:%M:%S UTC")

    return (
        "TIME_CONTEXT:\n"
        f"- NOW_UTC_ISO_Z: {now_utc_iso_z}\n"
        f"- NOW_UTC_EPOCH_SEC: {now_epoch_sec}\n"
        f"- NOW_LOCAL_ISO: {now_local_iso}\n"
        f"- NOW_LOCAL_HUMAN: {now_local_human}\n"
        f"- NOW_UTC_HUMAN: {now_utc_human}\n"
        f"GUIDELINE: Interpret times in {tz_name}. Use these values for 'today', 'yesterday', or specific times."
    )


def get_short_time_context() -> str:
    """Return a minimal one-line time context for per-turn instructions."""
    local_tz = get_local_tz()
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(local_tz)
    tz_name = get_settings().local_timezone or "UTC"
    return now_local.strftime(f"%A, %B %d, %Y at %H:%M:%S {tz_name}")


def get_utc_iso_z() -> str:
    """Return current UTC as ISO 8601 string (YYYY-MM-DDTHH:MM:SSZ)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
