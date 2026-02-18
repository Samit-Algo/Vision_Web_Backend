from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import dateparser
import parsedatetime
from pytz import UTC, timezone
from pytz.exceptions import UnknownTimeZoneError

from ....utils.datetime_utils import _get_app_timezone, now

CALENDAR = parsedatetime.Calendar()
DURATION_PATTERN = re.compile(
    r"^\s*(\d+)\s*(min(?:ute)?s?|hrs?|hours?|sec(?:ond)?s?)\s*$",
    re.IGNORECASE,
)

# Keywords that mark range/duration (used with dateparser-style detection; single list to maintain).
DURATION_OR_RANGE_KEYWORDS = ("to", "until", "for", "after", "before", "by")
RANGE_SEPARATORS = (" - ", "–")
PHRASE_END_ONLY_KEYWORDS = ("until", "by", "before")  # wording that implies single end boundary


def get_app_timezone_pytz():
    app_tz = _get_app_timezone()
    if hasattr(app_tz, "key"):
        try:
            return timezone(app_tz.key)
        except UnknownTimeZoneError:
            return UTC
    try:
        return timezone(str(app_tz)) if str(app_tz) != "UTC" else UTC
    except UnknownTimeZoneError:
        return UTC


def get_current_datetime_context() -> datetime:
    return now()


def parse_time_input(
    time_expression: str,
    reference_datetime: Optional[datetime] = None,
) -> str:
    if not time_expression or not time_expression.strip():
        raise ValueError("Time expression cannot be empty")
    expr = time_expression.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?$", expr):
        return expr if expr.endswith("Z") else expr + "Z"
    app_tz = get_app_timezone_pytz()
    ref = reference_datetime or get_current_datetime_context()
    if ref.tzinfo is None:
        ref = app_tz.localize(ref)
    # Prefer dateparser (package) for natural-language time; fallback to parsedatetime.
    tz_name = getattr(app_tz, "zone", None) or str(app_tz)
    dateparser_result = dateparser.parse(
        expr,
        settings={
            "RELATIVE_BASE": ref,
            "TIMEZONE": tz_name,
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )
    if dateparser_result is not None:
        if dateparser_result.tzinfo is None:
            dateparser_result = app_tz.localize(dateparser_result)
        return dateparser_result.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    parsed_time, status = CALENDAR.parseDT(datetimeString=expr, sourceTime=ref)
    if status == 0:
        for fallback in [f"at {expr}", f"today {expr}", expr]:
            parsed_time, status = CALENDAR.parseDT(datetimeString=fallback, sourceTime=ref)
            if status > 0:
                break
        if status == 0:
            raise ValueError(f"Could not parse: '{expr}'")
    if parsed_time.tzinfo is None:
        parsed_time = app_tz.localize(parsed_time)
    return parsed_time.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_time_for_display(iso_time: str) -> str:
    if not iso_time:
        return ""
    try:
        app_tz = get_app_timezone_pytz()
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = UTC.localize(dt)
        dt_local = dt.astimezone(app_tz)
        return dt_local.strftime(f"%B %d, %Y at %H:%M {app_tz}")
    except (ValueError, TypeError):
        return iso_time


def parse_duration_minutes(expression: str) -> int:
    if not expression or not expression.strip():
        raise ValueError("Duration expression cannot be empty")
    s = expression.strip()
    m = DURATION_PATTERN.match(s)
    if m:
        num, unit = int(m.group(1)), m.group(2).lower()
        if "min" in unit:
            return num
        if "hr" in unit or "hour" in unit:
            return num * 60
        if "sec" in unit:
            return max(1, (num + 29) // 60)
        return num
    try:
        ref = get_current_datetime_context()
        end_dt, status = CALENDAR.parseDT(datetimeString=s, sourceTime=ref)
        if status > 0:
            if end_dt.tzinfo is None:
                end_dt = get_app_timezone_pytz().localize(end_dt)
            ref_utc = ref.astimezone(UTC) if ref.tzinfo else get_app_timezone_pytz().localize(ref).astimezone(UTC)
            delta = end_dt.astimezone(UTC) - ref_utc
            return max(0, (int(delta.total_seconds()) + 29) // 60)
    except Exception:
        pass
    raise ValueError("Could not parse duration. Try '10 min' or '2 hours'.")


def _has_duration_or_range_keyword(text: str) -> bool:
    """True if input contains explicit range/duration markers (uses DURATION_OR_RANGE_KEYWORDS)."""
    t = text.strip().lower()
    if not t:
        return False
    if any(sep in t for sep in RANGE_SEPARATORS):
        return True
    if re.search(r"end\s+after|run\s+for", t):
        return True
    return any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in DURATION_OR_RANGE_KEYWORDS)


def parse_time_range(expression: str) -> Tuple[str, str]:
    """Parse explicit range only (e.g. '9 to 5', 'Sunday 1 - 5 PM'). No silent day adjustment."""
    if not expression or not expression.strip():
        raise ValueError("Time range cannot be empty")
    s = expression.strip()
    app_tz = get_app_timezone_pytz()
    ref = get_current_datetime_context()
    if ref.tzinfo is None:
        ref = app_tz.localize(ref)
    for sep in [" to ", " - ", "–", " until "]:
        if sep in s:
            parts = s.split(sep, 1)
            if len(parts) == 2:
                start_str, end_str = parts[0].strip(), parts[1].strip()
                try:
                    start_iso = parse_time_input(start_str, reference_datetime=ref)
                    end_iso = parse_time_input(end_str, reference_datetime=ref)
                    return start_iso, end_iso
                except ValueError:
                    continue
    raise ValueError("Could not parse range. Try 'Sunday 1 to 5 PM'.")


def parse_time_window(
    user_input: str,
    reference_start_iso: Optional[str] = None,
) -> Dict[str, Any]:
    if not user_input or not user_input.strip():
        return {
            "success": False,
            "user_message": "Please specify when to start and when to end (e.g. 'now', then 'in 10 minutes').",
        }
    raw = user_input.strip().lower()
    app_tz = get_app_timezone_pytz()
    ref_dt = get_current_datetime_context()
    if ref_dt.tzinfo is None:
        ref_dt = app_tz.localize(ref_dt)
    now_utc = ref_dt.astimezone(UTC)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    if raw in ("now", "start now", "start now."):
        return {"success": True, "start_time": now_iso, "end_time": None, "duration_minutes": None, "user_message": None}

    duration_match = re.search(
        r"(?:end\s+after|run\s+for|for)\s+(\d+)\s*(min(?:ute)?s?|hrs?|hours?)",
        user_input.strip(),
        re.IGNORECASE,
    )
    if duration_match and reference_start_iso:
        try:
            mins = parse_duration_minutes(f"{duration_match.group(1)} {duration_match.group(2)}")
            start_dt = datetime.fromisoformat(reference_start_iso.replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = UTC.localize(start_dt)
            end_dt = start_dt + timedelta(minutes=mins)
            return {
                "success": True,
                "start_time": reference_start_iso,
                "end_time": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_minutes": mins,
                "user_message": None,
            }
        except ValueError:
            pass
    if duration_match and not reference_start_iso:
        try:
            mins = parse_duration_minutes(f"{duration_match.group(1)} {duration_match.group(2)}")
            end_dt = now_utc + timedelta(minutes=mins)
            return {
                "success": True,
                "start_time": None,
                "end_time": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_minutes": mins,
                "user_message": None,
            }
        except ValueError:
            pass

    # Pure duration phrase (e.g. "10 minutes") with no "from"/start context: treat as end only from now, do not set start.
    try:
        mins = parse_duration_minutes(user_input)
        end_dt = now_utc + timedelta(minutes=mins)
        return {
            "success": True,
            "start_time": None,
            "end_time": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_minutes": mins,
            "user_message": None,
        }
    except ValueError:
        pass

    try:
        start_iso, end_iso = parse_time_range(user_input)
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = UTC.localize(start_dt)
        if end_dt.tzinfo is None:
            end_dt = UTC.localize(end_dt)
        if end_dt <= start_dt:
            return {
                "success": False,
                "user_message": "The end time must be after the start time.",
            }
        return {"success": True, "start_time": start_iso, "end_time": end_iso, "duration_minutes": None, "user_message": None}
    except ValueError:
        pass

    # Single time expression, no duration/range keywords → one boundary only (start or end from wording).
    if not _has_duration_or_range_keyword(user_input):
        try:
            single_iso = parse_time_input(user_input, reference_datetime=ref_dt)
            raw_lower = user_input.strip().lower()
            is_end_only = any(
                raw_lower.startswith(f"{kw} ") or re.search(rf"\b{re.escape(kw)}\s+", raw_lower)
                for kw in PHRASE_END_ONLY_KEYWORDS
            )
            if is_end_only:
                return {"success": True, "start_time": None, "end_time": single_iso, "duration_minutes": None, "user_message": None}
            return {"success": True, "start_time": single_iso, "end_time": None, "duration_minutes": None, "user_message": None}
        except ValueError:
            pass

    return {
        "success": False,
        "user_message": "I couldn't understand that time. Try 'now' then 'in 10 minutes', or 'Sunday 1 to 5 PM'.",
    }
