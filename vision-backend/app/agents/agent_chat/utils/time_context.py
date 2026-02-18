from __future__ import annotations

from datetime import datetime
from typing import Optional

from pytz import UTC, timezone
from pytz.exceptions import UnknownTimeZoneError

from ....core.config import get_settings

CACHED_TZ: Optional[object] = None


def get_local_timezone():
    global CACHED_TZ
    if CACHED_TZ is not None:
        return CACHED_TZ
    tz_str = (get_settings().local_timezone or "UTC").strip() or "UTC"
    try:
        CACHED_TZ = timezone(tz_str)
    except UnknownTimeZoneError:
        CACHED_TZ = UTC
    return CACHED_TZ


def get_current_time_context() -> str:
    local_tz = get_local_timezone()
    tz_name = get_settings().local_timezone or "UTC"
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(local_tz)
    return (
        "TIME_CONTEXT:\n"
        f"- NOW_UTC_ISO_Z: {now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"- NOW_UTC_EPOCH_SEC: {int(now_utc.timestamp())}\n"
        f"- NOW_LOCAL_ISO: {now_local.isoformat()}\n"
        f"- NOW_LOCAL_HUMAN: {now_local.strftime('%A, %B %d, %Y at %H:%M:%S %Z')}\n"
        f"- NOW_UTC_HUMAN: {now_utc.strftime('%A, %B %d, %Y at %H:%M:%S UTC')}\n"
        f"GUIDELINE: Interpret times in {tz_name}."
    )


def get_utc_iso_z() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
