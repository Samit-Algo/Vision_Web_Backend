"""
Pipeline Context
---------------

Holds task configuration and per-agent state for the pipeline.
Provides structured access to task fields and manages rule state.
Used by PipelineRunner to know what to run (camera, rules, FPS, mode).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from datetime import datetime
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.utils.datetime_utils import now, parse_iso, mongo_datetime_to_app_timezone


# -----------------------------------------------------------------------------
# Pipeline context (task config + state)
# -----------------------------------------------------------------------------


class PipelineContext:
    """
    Context for pipeline execution.

    Holds task configuration (from MongoDB), agent metadata, and rule state.
    PipelineRunner uses this to know: which camera/source, which rules, FPS,
    run mode (continuous vs patrol), and when to stop.
    """

    def __init__(self, task: Dict[str, Any], task_id: str):
        """
        Initialize pipeline context from a task document.

        Args:
            task: Task document from MongoDB (name, camera_id, rules, fps, etc.)
            task_id: Task identifier (used for DB updates and logging)
        """
        self.task_id = task_id
        self.task = task

        # ----- Agent and source -----
        self.agent_name = task.get("name") or task.get("task_name") or f"agent-{task_id}"
        self.agent_id = task.get("id") or task.get("agent_id") or task_id
        self.camera_id = (task.get("camera_id") or "").strip()
        self.video_path = (task.get("video_path") or "").strip()
        source_type = (task.get("source_type") or "").strip().lower()
        self.is_video_file = bool(self.video_path) or source_type == "video_file"
        self.fps = int(task.get("fps", 5))

        # ----- Run mode (continuous = run forever; patrol = sleep then process window) -----
        self.run_mode = (task.get("run_mode") or "continuous").strip().lower()
        self.interval_minutes = int(task.get("interval_minutes") or 5)
        self.check_duration_seconds = int(task.get("check_duration_seconds") or 10)

        # ----- Rules (list of rule configs; each has "type" and scenario-specific fields) -----
        self.rules: List[Dict[str, Any]] = task.get("rules") or []

        # ----- Rule state (per-rule index; used by some scenarios) -----
        self.rule_state: Dict[int, Dict[str, Any]] = {}

        # ----- Frame tracking (for FPS and skip reporting) -----
        self.frame_index = 0
        self.last_seen_hub_index: Optional[int] = None

        # ----- Diagnostics -----
        self.processed_in_window = 0
        self.skipped_in_window = 0
        self.last_status_time = 0.0

    # -------------------------------------------------------------------------
    # Public: get stop condition (user cancelled, task deleted, or end_time reached)
    # -------------------------------------------------------------------------

    def get_stop_condition(self, tasks_collection) -> Optional[str]:
        """
        Check if the pipeline should stop (user cancelled, task deleted, or end_time reached).

        Args:
            tasks_collection: MongoDB collection for tasks

        Returns:
            Reason string if should stop ("stop_requested", "task_deleted", "end_time_reached"),
            or None if pipeline should keep running.
        """
        from bson import ObjectId

        task_document = tasks_collection.find_one(
            {"_id": ObjectId(self.task_id)},
            projection={"stop_requested": 1, "end_time": 1, "end_at": 1},
        )

        if not task_document:
            return "task_deleted"

        if bool(task_document.get("stop_requested")):
            return "stop_requested"

        # Video file tasks: no end_time; stop only on EOF
        if self.is_video_file:
            return None

        end_at_value = task_document.get("end_time") or task_document.get("end_at")
        if end_at_value:
            if isinstance(end_at_value, datetime):
                end_at_dt = mongo_datetime_to_app_timezone(end_at_value)
            else:
                end_at_dt = parse_iso(end_at_value)
            if end_at_dt and now() >= end_at_dt:
                return "end_time_reached"
        return None

    # -------------------------------------------------------------------------
    # Public: update task status (completed, cancelled, etc.)
    # -------------------------------------------------------------------------

    def update_status(self, tasks_collection, status: str) -> None:
        """
        Update task status in the database (e.g. "completed", "cancelled").

        Args:
            tasks_collection: MongoDB collection for tasks
            status: New status string to set
        """
        from bson import ObjectId
        from app.utils.datetime_utils import utc_now

        try:
            tasks_collection.update_one(
                {"_id": ObjectId(self.task_id)},
                {"$set": {"status": status, "stopped_at": utc_now(), "updated_at": utc_now()}},
            )
        except Exception:
            pass
