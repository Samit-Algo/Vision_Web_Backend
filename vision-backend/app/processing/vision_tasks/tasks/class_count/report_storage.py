"""
Report Storage Utility
----------------------

This module handles storing counting reports to MongoDB.
Used by both class_count and box_count scenarios.

What it does:
- Stores entry/exit events to MongoDB in real-time
- Tracks agent details, counts, and track IDs
- Simple and beginner-friendly implementation
"""

from datetime import datetime
from typing import Dict, Any, Optional
from app.utils.db import get_collection


# Collection name for reports in MongoDB
REPORTS_COLLECTION = "counting_reports"


def save_counting_event(
    agent_id: str,
    agent_name: str,
    camera_id: str,
    report_type: str,  # "class_count" or "box_count"
    track_id: int,
    event_type: str,  # "entry" or "exit"
    timestamp: datetime,
    entry_count: int,
    exit_count: int,
    active_tracks: int,  # Standby count (objects currently in area)
    target_class: str
) -> bool:
    """
    Save a single entry/exit event to MongoDB.

    This function is called every time an object enters or exits the counting line.
    It stores the event immediately to MongoDB.

    Args:
        agent_id: Unique ID of the agent
        agent_name: Name of the agent
        camera_id: Camera ID (can be empty string)
        report_type: "class_count" or "box_count"
        track_id: Unique track ID of the object
        event_type: "entry" or "exit"
        timestamp: When the event happened
        entry_count: Total entry count so far
        exit_count: Total exit count so far
        active_tracks: Number of objects currently in the area (standby)
        target_class: What we're counting (e.g., "person", "box")

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Get MongoDB collection
        collection = get_collection(REPORTS_COLLECTION)

        # Create document to store
        report_doc = {
            # Agent information
            "agent_id": agent_id,
            "agent_name": agent_name,
            "camera_id": camera_id,

            # Report type
            "report_type": report_type,  # "class_count" or "box_count"
            "target_class": target_class,

            # Event details
            "track_id": track_id,
            "event_type": event_type,  # "entry" or "exit"
            "timestamp": timestamp,

            # Current counts
            "entry_count": entry_count,
            "exit_count": exit_count,
            "standby_count": active_tracks,  # Objects currently in area

            # Metadata
            "created_at": datetime.utcnow()
        }

        # Insert into MongoDB
        result = collection.insert_one(report_doc)

        # Print confirmation (like the existing print statements)
        print(f"[REPORT] ✅ Saved {event_type.upper()} event: Track ID {track_id} | "
              f"IN: {entry_count}, OUT: {exit_count}, STANDBY: {active_tracks}")

        return True

    except Exception as e:
        # Print error but don't crash the pipeline
        print(f"[REPORT] ❌ Error saving report to MongoDB: {e}")
        return False


def initialize_report_session(
    agent_id: str,
    agent_name: str,
    camera_id: str,
    report_type: str,
    target_class: str
) -> bool:
    """
    Initialize a new report session when agent starts.

    This creates a summary document that tracks the entire session.
    Called once when the agent starts.

    Args:
        agent_id: Unique ID of the agent
        agent_name: Name of the agent
        camera_id: Camera ID
        report_type: "class_count" or "box_count"
        target_class: What we're counting

    Returns:
        True if initialized successfully
    """
    try:
        collection = get_collection(REPORTS_COLLECTION)

        # Create session summary document
        session_doc = {
            # Agent information
            "agent_id": agent_id,
            "agent_name": agent_name,
            "camera_id": camera_id,

            # Report type
            "report_type": report_type,
            "target_class": target_class,

            # Session tracking
            "session_type": "summary",  # Mark this as a summary document
            "start_time": datetime.utcnow(),
            "end_time": None,  # Will be updated when agent stops

            # Counts (will be updated as events happen)
            "total_entry_count": 0,
            "total_exit_count": 0,
            "final_standby_count": 0,

            # Status
            "status": "active",  # "active" or "completed"

            "created_at": datetime.utcnow()
        }

        # Insert session document
        result = collection.insert_one(session_doc)
        session_id = str(result.inserted_id)

        print(f"[REPORT] 🆕 Started report session: {session_id} | "
              f"Agent: {agent_name} | Type: {report_type}")

        return True

    except Exception as e:
        print(f"[REPORT] ❌ Error initializing report session: {e}")
        return False


def finalize_report_session(
    agent_id: str,
    report_type: str,
    final_entry_count: int,
    final_exit_count: int,
    final_standby_count: int
) -> bool:
    """
    Finalize the report session when agent stops.

    This updates the session summary with final counts and end time.
    Called when the agent ends.

    Args:
        agent_id: Unique ID of the agent
        report_type: "class_count" or "box_count"
        final_entry_count: Final total entry count
        final_exit_count: Final total exit count
        final_standby_count: Final standby count

    Returns:
        True if finalized successfully
    """
    try:
        collection = get_collection(REPORTS_COLLECTION)

        # Find the active session for this agent and report type
        query = {
            "agent_id": agent_id,
            "report_type": report_type,
            "session_type": "summary",
            "status": "active"
        }

        # Update with final counts and end time
        update = {
            "$set": {
                "end_time": datetime.utcnow(),
                "total_entry_count": final_entry_count,
                "total_exit_count": final_exit_count,
                "final_standby_count": final_standby_count,
                "status": "completed",
                "updated_at": datetime.utcnow()
            }
        }

        result = collection.update_one(query, update)

        if result.modified_count > 0:
            print(f"[REPORT] ✅ Finalized report session | "
                  f"Agent: {agent_id} | IN: {final_entry_count}, OUT: {final_exit_count}, STANDBY: {final_standby_count}")
            return True
        else:
            print(f"[REPORT] ⚠️ No active session found to finalize for agent: {agent_id}")
            return False

    except Exception as e:
        print(f"[REPORT] ❌ Error finalizing report session: {e}")
        return False
