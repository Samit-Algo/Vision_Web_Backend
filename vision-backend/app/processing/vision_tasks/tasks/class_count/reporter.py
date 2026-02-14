"""
Report generator
-----------------

Builds count report (current count, min/max/avg, time range). Used by class_count and box_count.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from datetime import datetime, timezone
from typing import Any, Dict


def safe_timestamp(dt: datetime) -> float:
    """Return Unix timestamp for dt; on Windows .timestamp() can raise OSError [Errno 22] for some dates."""
    try:
        return dt.timestamp()
    except OSError:
        return datetime.now(timezone.utc).timestamp()


def generate_report(
    state: Dict[str, Any],
    current_count: int,
    now: datetime,
    zone_applied: bool = False
) -> Dict[str, Any]:
    """
    Generate a detailed report with statistics from collected count data.
    
    Args:
        state: Scenario state dictionary
        current_count: Current count value
        now: Current timestamp
        zone_applied: Whether zone filtering was applied
    
    Returns:
        Report dictionary with statistics
    """
    if "count_history" not in state:
        state["count_history"] = []
    if "first_sample_time" not in state:
        state["first_sample_time"] = now.isoformat()
    
    count_history = state["count_history"]
    count_history.append({
        "count": current_count,
        "timestamp": now.isoformat(),
        "timestamp_epoch": safe_timestamp(now)
    })
    
    # Limit history size
    if len(count_history) > 1000:
        state["count_history"] = count_history[-1000:]
        count_history = state["count_history"]
    
    counts = [entry["count"] for entry in count_history]
    
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        total_samples = len(count_history)
        
        first_time = state.get("first_sample_time")
        duration_seconds = 0
        try:
            if first_time:
                first_dt = datetime.fromisoformat(first_time.replace('Z', '+00:00'))
                duration_seconds = int((safe_timestamp(now) - safe_timestamp(first_dt)))
        except (ValueError, TypeError, OSError):
            pass
        
        report = {
            "current_count": current_count,
            "statistics": {
                "minimum": int(min_count),
                "maximum": int(max_count),
                "average": round(avg_count, 2),
                "total_samples": total_samples
            },
            "time_range": {
                "start": first_time,
                "end": now.isoformat(),
                "duration_seconds": duration_seconds
            },
            "zone_applied": zone_applied
        }
    else:
        report = {
            "current_count": current_count,
            "statistics": {
                "minimum": current_count,
                "maximum": current_count,
                "average": float(current_count),
                "total_samples": 1
            },
            "time_range": {
                "start": now.isoformat(),
                "end": now.isoformat(),
                "duration_seconds": 0
            },
            "zone_applied": zone_applied
        }
    
    return report
