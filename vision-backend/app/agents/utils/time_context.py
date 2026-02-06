from datetime import datetime
from pytz import UTC, timezone

def get_current_time_context() -> str:
    """
    Get current time context for the LLM prompt.
    Machine-readable and human-readable formats for UTC and IST.
    """
    IST = timezone("Asia/Kolkata")

    now_utc = datetime.now(UTC)
    now_ist = now_utc.astimezone(IST)

    now_utc_iso_z = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_ist_iso = now_ist.isoformat()
    now_epoch_sec = int(now_utc.timestamp())
    now_ist_human = now_ist.strftime("%A, %B %d, %Y at %H:%M:%S IST")
    now_utc_human = now_utc.strftime("%A, %B %d, %Y at %H:%M:%S UTC")

    return (
        "TIME_CONTEXT:\n"
        f"- NOW_UTC_ISO_Z: {now_utc_iso_z}\n"
        f"- NOW_UTC_EPOCH_SEC: {now_epoch_sec}\n"
        f"- NOW_IST_ISO: {now_ist_iso}\n"
        f"- NOW_IST_HUMAN: {now_ist_human}\n"
        f"- NOW_UTC_HUMAN: {now_utc_human}\n"
        "GUIDELINE: Use these values to interpret 'today', 'yesterday', or specific times. "
        "When calling tools, use the corresponding machine-readable values if required."
    )


def get_short_time_context() -> str:
    """
    Minimal one-line time context for dynamic (per-turn) instructions.
    Use for low-token dynamic instruction updates.
    """
    IST = timezone("Asia/Kolkata")
    now_utc = datetime.now(UTC)
    now_ist = now_utc.astimezone(IST)
    return now_ist.strftime("%A, %B %d, %Y at %H:%M:%S IST")
