"""
Agent main loop
---------------

Implements the per-agent processing with hybrid behavior:
- Load models
- Open source
- Modes:
  - continuous: read and process frames at agent FPS indefinitely
  - patrol: sleep for interval, then process frames for a short window at agent FPS, repeat
- Merge detections and apply rule engine
- Print alerts, send heartbeat
- Stop on end_at/stop_requested/file end
"""
import time
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from bson import ObjectId
from bson.errors import InvalidId

# Allow running when imported without installed package by ensuring project root is on sys.path
if __package__ is None or __package__ == "":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

from app.utils.db import get_collection
from app.processing.models.model_loader import ModelLoader
from app.processing.data_input.source_factory import create_source
from app.processing.pipeline.context import PipelineContext
from app.processing.pipeline.pipeline import PipelineRunner
import numpy as np


def run_task_worker(task_id: str, shared_store: Optional["Dict[str, Any]"] = None) -> None:
    """
    Standalone agent main loop.
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

    # Support both old and new field names
    agent_name = task.get("name") or task.get("task_name") or f"agent-{task_id}"
    fps = int(task.get("fps", 5))
    
    # Handle model: new format uses single "model" string, old format uses "model_ids" array
    model_value = task.get("model")
    if model_value:
        # New format: single model string
        if isinstance(model_value, str):
            model_ids = [model_value]
        elif isinstance(model_value, list):
            model_ids = model_value
        else:
            model_ids = []
    else:
        # Old format: model_ids array
        model_ids = task.get("model_ids", []) or []
    
    run_mode = (task.get("run_mode") or "continuous").strip().lower()  # "continuous" | "patrol"
    # Handle None values explicitly (MongoDB null values)
    interval_minutes = int(task.get("interval_minutes") or 5)  # patrol sleep interval
    check_duration_seconds = int(task.get("check_duration_seconds") or 10)  # patrol detection window

    print(f"[worker {task_id}] ▶️ Starting '{agent_name}' | mode={run_mode} fps={fps} models={model_ids}")

    # Load models using ModelManager
    model_manager = ModelLoader()
    models = model_manager.load_models(model_ids)
    if not models:
        print(f"[worker {task_id}] ❌ No models loaded. Exiting.")
        return

    source = create_source(task, shared_store)
    if source is None:
        print(f"[worker {task_id}] ⚠️ Could not create source (need camera_id for RTSP or video_path for file). Exiting.")
        return

    # Create pipeline context
    context = PipelineContext(task, task_id)

    # Create pipeline runner
    runner = PipelineRunner(context, source, models, shared_store)

    try:
        # Run pipeline based on mode
        if run_mode == "continuous":
            # Run pipeline in continuous mode
            runner.run_continuous()
        else:
            # Run pipeline in patrol mode
            runner.run_patrol()

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup: close source if needed
        try:
            if source is not None and hasattr(source, 'close'):
                source.close()
        except Exception:
            pass
        print(f"[worker {task_id}] ⏹️ Exiting")


