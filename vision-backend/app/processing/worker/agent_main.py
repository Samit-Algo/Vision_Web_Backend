"""
Agent worker (per-task process)
-------------------------------

Runs the pipeline for one agent task: load models, create source, run continuous or patrol.
Started by the vision runner (main_process). One process per task.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import os
import sys
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId

# Ensure project root on path when run as script or from runner
if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.utils.db import get_collection
from app.processing.models.model_loader import ModelLoader
from app.processing.data_input.source_factory import create_source
from app.processing.pipeline.context import PipelineContext
from app.processing.pipeline.pipeline import PipelineRunner


# -----------------------------------------------------------------------------
# Worker entry point
# -----------------------------------------------------------------------------


def run_task_worker(task_id: str, shared_store: Optional[Dict[str, Any]] = None) -> None:
    """
    Run the pipeline for one task (called in a separate process by the vision runner).

    1. Load task from DB; exit if not found.
    2. Resolve model IDs (single "model" or "model_ids" list).
    3. Load models via ModelLoader.
    4. Create source (HubSource or VideoFileSource) from task + shared_store.
    5. Build PipelineContext and PipelineRunner; run continuous or patrol.
    6. On exit, close source if it has close().
    """
    tasks_collection = get_collection()
    # Query by _id (MongoDB's native field) with ObjectId conversion
    try:
        task = tasks_collection.find_one({"_id": ObjectId(task_id)})
    except (InvalidId, ValueError, TypeError):
        # Fallback: try querying by "id" field if ObjectId conversion fails
        task = tasks_collection.find_one({"id": task_id})

    if not task:
        print(f"[worker {task_id}] ❌ Task not found. Exiting.")
        return

    # Task fields (support both old and new names)
    agent_name = task.get("name") or task.get("task_name") or f"agent-{task_id}"
    fps = int(task.get("fps", 5))
    model_value = task.get("model")
    if model_value is not None:
        model_ids = [model_value] if isinstance(model_value, str) else (model_value if isinstance(model_value, list) else [])
    else:
        model_ids = task.get("model_ids", []) or []
    run_mode = (task.get("run_mode") or "continuous").strip().lower()
    interval_minutes = int(task.get("interval_minutes") or 5)
    check_duration_seconds = int(task.get("check_duration_seconds") or 10)

    print(f"[worker {task_id}] ▶️ Starting '{agent_name}' | mode={run_mode} fps={fps} models={model_ids}")

    # Create context first to check if YOLO is required
    context = PipelineContext(task, task_id)
    
    # Check if any scenario requires YOLO detections
    from app.processing.pipeline.pipeline import any_scenario_requires_yolo
    requires_yolo = any_scenario_requires_yolo(context)
    
    # Load models only if YOLO is required
    models = []
    if requires_yolo:
        model_loader = ModelLoader()
        # Filter out "none" or empty model IDs
        valid_model_ids = [mid for mid in model_ids if mid and mid.strip() and mid.lower() != "none"]
        if valid_model_ids:
            models = model_loader.load_models(valid_model_ids)
            if not models:
                print(f"[worker {task_id}] ❌ No models loaded (YOLO required but models failed to load). Exiting.")
                return
        else:
            print(f"[worker {task_id}] ❌ YOLO required but no valid models provided. Exiting.")
            return
    else:
        # No YOLO required - models can be empty
        print(f"[worker {task_id}] ℹ️ No YOLO models required for this scenario (motion-based detection only)")

    # Create source (RTSP via shared_store or video file)
    source = create_source(task, shared_store)
    if source is None:
        print(f"[worker {task_id}] ⚠️ Could not create source (need camera_id or video_path). Exiting.")
        return

    # Debug: key used for agent overlay stream (must match API agent_id in /agents/{agent_id}/overlay/frames/ws)
    print(f"[worker {task_id}] 📌 shared_store key for overlay stream: {context.agent_id!r}")

    runner = PipelineRunner(context, source, models, shared_store)

    try:
        if run_mode == "continuous":
            runner.run_continuous()
        else:
            runner.run_patrol()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if source is not None and hasattr(source, "close"):
                source.close()
        except Exception:
            pass
        print(f"[worker {task_id}] ⏹️ Exiting")
