"""
Vision runner (orchestrator)
----------------------------

Polls MongoDB for cameras and agent tasks, starts CameraPublishers and workers.
- Cameras with active agents → one CameraPublisher per camera (RTSP → shared_store)
- Active tasks → one worker process per task (reads from shared_store or video file)
Cleans up stopped cameras and finished workers each poll.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import os
import sys
import time
from datetime import datetime
from multiprocessing import Manager, Process, Queue
from typing import Any, Dict, List, Optional, Tuple

_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
_project_root = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.worker.agent_main import run_task_worker
from app.processing.data_input.camera_publisher import CameraCommand, CameraPublisher
from app.utils.db import get_collection
from app.utils.datetime_utils import mongo_datetime_to_app_timezone, now, parse_iso, utc_now

# -----------------------------------------------------------------------------
# Helpers (parse task start/end time once, reuse)
# -----------------------------------------------------------------------------


def parse_task_start_end(task: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse start_time/start_at and end_time/end_at from task. Returns (start_at, end_at) in app timezone."""
    start_val = task.get("start_time") or task.get("start_at")
    end_val = task.get("end_time") or task.get("end_at")
    start_at = None
    if start_val is not None:
        if isinstance(start_val, datetime):
            start_at = mongo_datetime_to_app_timezone(start_val)
        else:
            start_at = parse_iso(start_val)
    end_at = None
    if end_val is not None:
        if isinstance(end_val, datetime):
            end_at = mongo_datetime_to_app_timezone(end_val)
        else:
            end_at = parse_iso(end_val)
    return (start_at, end_at)


# -----------------------------------------------------------------------------
# Main runner loop
# -----------------------------------------------------------------------------


def main(shared_store=None) -> None:
    """
    Main loop: poll MongoDB, start/stop CameraPublishers and workers.

    1. Query active tasks → collect camera_ids that have at least one active agent.
    2. Start CameraPublisher only for those cameras (RTSP → shared_store).
    3. Stop publishers for cameras with no active agents.
    4. Start a worker for each active task; workers read from shared_store or video file.
    5. Clean up finished workers and stopped publishers.

    shared_store: multiprocessing Manager().dict() or None (creates one).
    """
    print("[runner] 🚀 Starting Agent Runner")
    tasks_collection = get_collection()  # Default: 'agents' collection
    cameras_collection = get_collection("cameras")  # 'cameras' collection
    poll_interval = int(os.getenv("AGENT_POLL_INTERVAL_SEC", "5"))

    # Use provided shared_store or create new one
    if shared_store is None:
        manager = Manager()
        shared_store = manager.dict()
        print("[runner] 📦 Created new shared_store")

    task_processes: Dict[str, Process] = {}
    # camera_id -> (process, command_queue, source_uri)
    camera_publishers: Dict[str, Tuple[Process, Queue, str]] = {}

    try:
        while True:
            current_time = now()

            # Step 0: Active tasks and camera_ids that need a publisher
            active_task_cursor = tasks_collection.find({
                "status": {"$in": ["PENDING", "ACTIVE", "RUNNING", "pending", "scheduled", "running", None]}
            }).sort("created_at", 1)
            tasks_list = list(active_task_cursor)
            
            # Collect camera_ids that have at least one RTSP agent runnable NOW.
            # Video-file tasks (video_path / source_type video_file) do not need a camera or publisher.
            camera_ids_with_agents = set()
            for task in tasks_list:
                video_path = (task.get("video_path") or "").strip()
                source_type = (task.get("source_type") or "").strip().lower()
                if video_path or source_type == "video_file":
                    continue
                camera_id = task.get("camera_id")
                if not camera_id:
                    continue

                start_at, end_at = parse_task_start_end(task)
                # Runnable now if:
                # - no start time OR start time already reached
                # - no end time OR we are before end time
                if start_at and current_time < start_at:
                    continue
                if end_at and current_time >= end_at:
                    continue

                camera_ids_with_agents.add(camera_id)
            
            # Step 1: Start CameraPublisher for each camera that has active agents
            # Query cameras: include all cameras that either don't have a status field,
            # or have status != "inactive". This supports both old (no status) and new (with status) schemas.
            active_cameras = cameras_collection.find({})
            active_camera_ids = set()
            
            for camera_doc in active_cameras:
                # Support both old and new field names for backward compatibility
                camera_id = camera_doc.get("id") or camera_doc.get("camera_id")
                source_uri = camera_doc.get("stream_url") or camera_doc.get("rtsp_url") or camera_doc.get("source_uri")
                
                if not camera_id or not source_uri:
                    continue
                
                # Only process cameras that have at least one active agent
                if camera_id not in camera_ids_with_agents:
                    continue
                    
                active_camera_ids.add(camera_id)
                
                # Check if CameraPublisher is already running
                if camera_id in camera_publishers:
                    publisher_process, command_queue, _source_uri = camera_publishers[camera_id]
                    # Check if process is still alive
                    if publisher_process.is_alive():
                        # Already running, skip
                        continue
                    else:
                        # Process died, remove from dict and restart
                        print(f"[runner] ⚠️  CameraPublisher for {camera_id} died, restarting...")
                        del camera_publishers[camera_id]
                
                # Start CameraPublisher (either new or restarted)
                command_queue: Queue = Queue()
                publisher_process = CameraPublisher(
                    camera_id=camera_id,
                    source_uri=source_uri,
                    shared_store=shared_store,
                    command_queue=command_queue,
                )
                publisher_process.start()
                camera_publishers[camera_id] = (publisher_process, command_queue, source_uri)
                print(f"[runner] 🎥 Started publisher for camera {camera_id} (has active agents)")

            # Step 2: Stop publishers for cameras with no active agents
            for camera_id in list(camera_publishers.keys()):
                if camera_id not in active_camera_ids:
                    publisher_process, command_queue, _source_uri = camera_publishers[camera_id]
                    try:
                        command_queue.put(CameraCommand(kind="stop"))
                    except Exception:
                        pass
                    if publisher_process.is_alive():
                        publisher_process.join(timeout=1.0)
                    del camera_publishers[camera_id]
                    print(f"[runner] 🎥 Stopped publisher for camera {camera_id} (no active agents)")

            # Step 3: Terminate workers for tasks that are no longer active
            
            # Terminate workers for tasks that are no longer active
            # Support both old and new field names for task ID
            active_task_ids = set()
            for task in tasks_list:
                task_id_from_doc = str(task.get("id") or task.get("agent_id") or task["_id"])
                active_task_ids.add(task_id_from_doc)
            
            for task_id_for_cleanup, worker_process in list(task_processes.items()):
                if task_id_for_cleanup not in active_task_ids:
                    if worker_process.is_alive():
                        worker_process.terminate()
                        worker_process.join(timeout=0.5)
                    del task_processes[task_id_for_cleanup]
                    print(f"[runner] 🛑 Terminated worker for inactive/missing task {task_id_for_cleanup}")

            # Step 4: Launch new workers for active tasks
            for task in tasks_list:
                # Support both old and new field names
                task_id = str(task.get("id") or task.get("agent_id") or task["_id"])

                # Skip if already running
                if task_id in task_processes and task_processes[task_id].is_alive():
                    continue

                video_path = (task.get("video_path") or "").strip()
                source_type = (task.get("source_type") or "").strip().lower()
                is_video_file = bool(video_path) or source_type == "video_file"
                start_at, end_at = (None, None) if is_video_file else parse_task_start_end(task)

                # Handle "scheduled" tasks (future start time)
                if start_at and current_time < start_at:
                    current_status = task.get("status")
                    # Support both old and new status values
                    if current_status not in ["scheduled", "SCHEDULED", "PENDING"]:
                        tasks_collection.update_one(
                            {"_id": task["_id"]},
                            {"$set": {"status": "scheduled", "updated_at": utc_now()}},
                        )
                    continue

                # Handle "expired" tasks (past end time)
                if end_at and current_time >= end_at:
                    current_status = task.get("status")
                    if current_status not in {"completed", "COMPLETED", "cancelled", "CANCELLED"}:
                        tasks_collection.update_one(
                            {"_id": task["_id"]},
                            {"$set": {
                                "status": "COMPLETED",
                                "stopped_at": utc_now(),
                                "updated_at": utc_now(),
                            }},
                        )
                    continue

                # Launch new worker
                tasks_collection.update_one(
                    {"_id": task["_id"]},
                    {"$set": {
                        "status": "RUNNING",
                        "started_at": utc_now(),
                        "updated_at": utc_now(),
                    }},
                )

                # Pass shared_store proxy to worker so it can subscribe to frames
                worker_process = Process(target=run_task_worker, args=(task_id, shared_store))
                worker_process.daemon = True
                worker_process.start()
                task_processes[task_id] = worker_process

                print(f"[runner] 🏃 Launched worker for task {task_id} (pid={worker_process.pid}) | overlay store_key={task_id!r}")

            # Clean up finished processes
            for finished_task_id, worker_process in list(task_processes.items()):
                if not worker_process.is_alive():
                    worker_process.join(timeout=0.1)
                    del task_processes[finished_task_id]
                    print(f"[runner] 🧹 Cleaned up worker for task {finished_task_id}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[runner] 🛑 Stopping Agent Runner (Ctrl+C)")
        # Stop publishers
        for camera_id, (publisher_process, command_queue, _source_uri) in list(camera_publishers.items()):
            try:
                command_queue.put(CameraCommand(kind="stop"))
            except Exception:
                pass
            if publisher_process.is_alive():
                publisher_process.terminate()
        camera_publishers.clear()
        for worker_process in task_processes.values():
            if worker_process.is_alive():
                worker_process.terminate()
        print("[runner] ✅ All workers terminated.")


if __name__ == "__main__":
    main()

