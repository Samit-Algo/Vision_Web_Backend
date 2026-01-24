"""
Pipeline Context
----------------

Holds task configuration and per-agent state for the pipeline.
Provides structured access to task fields and manages rule state.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from app.utils.datetime_utils import now, parse_iso, mongo_datetime_to_app_timezone


class PipelineContext:
    """
    Context for pipeline execution.
    
    Holds task configuration, agent metadata, and rule state.
    Provides structured access to task fields.
    """
    
    def __init__(self, task: Dict[str, Any], task_id: str):
        """
        Initialize pipeline context from task document.
        
        Args:
            task: Task document from MongoDB
            task_id: Task identifier
        """
        self.task_id = task_id
        self.task = task
        
        # Extract common fields
        self.agent_name = task.get("name") or task.get("task_name") or f"agent-{task_id}"
        self.agent_id = task.get("id") or task.get("agent_id") or task_id
        self.camera_id = (task.get("camera_id") or "").strip()
        self.fps = int(task.get("fps", 5))
        
        # Run mode
        self.run_mode = (task.get("run_mode") or "continuous").strip().lower()
        self.interval_minutes = int(task.get("interval_minutes") or 5)
        self.check_duration_seconds = int(task.get("check_duration_seconds") or 10)
        
        # Rules
        self.rules: List[Dict[str, Any]] = task.get("rules") or []
        
        # Rule state (indexed by rule index)
        self.rule_state: Dict[int, Dict[str, Any]] = {}
        
        # Frame tracking
        self.frame_index = 0
        self.last_seen_hub_index: Optional[int] = None
        
        # Diagnostics
        self.processed_in_window = 0
        self.skipped_in_window = 0
        self.last_status_time = 0.0
    
    def get_stop_condition(self, tasks_collection) -> Optional[str]:
        """
        Check if pipeline should stop.
        
        Args:
            tasks_collection: MongoDB tasks collection
        
        Returns:
            Stop reason string if should stop, None otherwise
        """
        from bson import ObjectId
        
        task_document = tasks_collection.find_one(
            {"_id": ObjectId(self.task_id)},
            projection={"stop_requested": 1, "end_time": 1, "end_at": 1}
        )
        
        if not task_document:
            return "task_deleted"
        
        stop_requested = bool(task_document.get("stop_requested"))
        if stop_requested:
            return "stop_requested"
        
        end_at_value = task_document.get("end_time") or task_document.get("end_at")
        if end_at_value:
            if isinstance(end_at_value, datetime):
                end_at_dt = mongo_datetime_to_app_timezone(end_at_value)
            else:
                end_at_dt = parse_iso(end_at_value)
            
            if end_at_dt and now() >= end_at_dt:
                return "end_time_reached"
        
        return None
    
    def update_status(self, tasks_collection, status: str) -> None:
        """
        Update task status in database.
        
        Args:
            tasks_collection: MongoDB tasks collection
            status: Status to set
        """
        from bson import ObjectId
        from app.utils.datetime_utils import utc_now
        
        try:
            tasks_collection.update_one(
                {"_id": ObjectId(self.task_id)},
                {"$set": {"status": status, "stopped_at": utc_now(), "updated_at": utc_now()}}
            )
        except Exception:
            pass  # Ignore errors
