"""
Time context helpers for agent prompts.

Provides current time in UTC and application timezone for use in LLM instructions.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from datetime import datetime

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from pytz import UTC, timezone

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...core.config import get_settings


def get_local_tz():
    """Return application timezone from config (e.g. Asia/Kolkata)."""
    tz_str = get_settings().local_timezone or "UTC"
    try:
        return timezone(tz_str)
    except Exception:
        return UTC


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
