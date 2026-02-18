"""
Restricted zone report storage
------------------------------

Single-record report per session in MongoDB (restricted_zone_reports).
One document per active session: violations array, total_unique_persons, max_concurrent_in_zone.
"""

from datetime import datetime
from typing import Optional

from app.utils.db import get_collection

COLLECTION = "restricted_zone_reports"


def add_violation(
    agent_id: str,
    agent_name: str,
    camera_id: str,
    track_id: int,
    entry_time: datetime,
    exit_time: datetime,
    duration_seconds: float,
    total_unique_persons: int,
    max_concurrent_in_zone: int,
) -> bool:
    """
    Append one violation to the active report (one record per session).
    Creates the report document on first violation.
    """
    try:
        coll = get_collection(COLLECTION)
        now = datetime.utcnow()
        violation = {
            "track_id": track_id,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "duration_seconds": round(duration_seconds, 2),
        }

        # Find active report for this agent/camera
        report = coll.find_one(
            {"agent_id": agent_id, "camera_id": camera_id, "status": "active"}
        )

        if not report:
            doc = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "camera_id": camera_id,
                "session_start": now,
                "session_end": None,
                "total_unique_persons": total_unique_persons,
                "max_concurrent_in_zone": max_concurrent_in_zone,
                "violations": [violation],
                "status": "active",
                "created_at": now,
            }
            coll.insert_one(doc)
            return True

        # Update existing report: push violation, set counts
        coll.update_one(
            {"agent_id": agent_id, "camera_id": camera_id, "status": "active"},
            {
                "$push": {"violations": violation},
                "$set": {
                    "total_unique_persons": total_unique_persons,
                    "max_concurrent_in_zone": max_concurrent_in_zone,
                    "updated_at": now,
                },
            },
        )
        return True
    except Exception as e:
        print(f"[RESTRICTED_ZONE_REPORT] ❌ Error saving violation: {e}")
        return False


def finalize_report(agent_id: str, camera_id: str) -> bool:
    """Set session_end and status=completed for the active report."""
    try:
        coll = get_collection(COLLECTION)
        now = datetime.utcnow()
        r = coll.update_one(
            {"agent_id": agent_id, "camera_id": camera_id, "status": "active"},
            {"$set": {"session_end": now, "status": "completed", "updated_at": now}},
        )
        if r.modified_count:
            print(f"[RESTRICTED_ZONE_REPORT] ✅ Finalized report | agent_id={agent_id}")
        return r.modified_count > 0
    except Exception as e:
        print(f"[RESTRICTED_ZONE_REPORT] ❌ Error finalizing report: {e}")
        return False
