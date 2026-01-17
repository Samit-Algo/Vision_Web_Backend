"""
YOLO utilities and simple rule engine
-------------------------------------

Used by workers to:
- load YOLO model by name/path (init_yolo_model)
- run a simple rule engine on detected class names (check_event_match)
- infer object targets from task_name (infer_object_targets)
- provide a draw_boxes placeholder for future overlays
"""
import os
import re
from typing import Optional

# Optional YOLO (used if installed)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]


def init_yolo_model(task: dict):
    """
    Initialize a YOLO model if the dependency is installed.
    
    Automatically downloads standard YOLO models if they don't exist locally.
    Falls back to yolov8n.pt which is small and suitable for quick demos.
    Returns the model instance or None if unavailable.
    """
    if YOLO is None:
        print("⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
        return None
    
    model_name = task.get("yolo_model_path") or "yolov8n.pt"
    
    # Clean model name: remove trailing slashes, whitespace, and normalize
    model_name = model_name.strip().rstrip('/').rstrip('\\')
    
    # Check if it's a standard YOLO model name (will be auto-downloaded if not found)
    is_standard_model = (
        model_name.startswith("yolov8") or 
        model_name.startswith("yolov5") or 
        model_name.startswith("yolo11") or
        model_name.startswith("yolo10") or
        model_name.startswith("yolo9")
    ) and model_name.endswith(".pt")
    
    # If it's a file path (contains path separators), check if file exists
    is_file_path = os.sep in model_name or '/' in model_name or '\\' in model_name
    
    if is_file_path and not os.path.exists(model_name):
        print(f"❌ Model file not found: {model_name}")
        return None
    
    try:
        # YOLO will automatically download standard models if they don't exist
        # For standard models, YOLO handles the download automatically
        # For custom paths, we've already checked existence above
        print(f"📥 Loading YOLO model: {model_name}...")
        model = YOLO(model_name)
        print(f"✅ YOLO model loaded: {model_name}")
        return model
    except FileNotFoundError as exc:
        # If it's a standard model and download failed, provide helpful message
        if is_standard_model:
            print(f"⚠️ Model file not found locally: {model_name}")
            print(f"💡 Attempting to download '{model_name}' automatically...")
            try:
                # YOLO automatically downloads on first use, but we can retry
                # The download happens automatically when YOLO() is called with a standard model name
                model = YOLO(model_name)
                print(f"✅ YOLO model downloaded and loaded: {model_name}")
                return model
            except Exception as retry_exc:  # noqa: BLE001
                print(f"❌ Failed to download YOLO model '{model_name}': {retry_exc}")
                print(f"💡 Please check your internet connection or download manually")
                return None
        else:
            print(f"❌ Model file not found: {model_name}")
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to load YOLO model '{model_name}': {exc}")
        # If it's a standard model, YOLO should auto-download, so this might be a different error
        if is_standard_model:
            print(f"💡 Note: Standard YOLO models should download automatically. Check internet connection.")
        return None


def infer_object_targets(task: dict) -> set[str]:
    """
    Infer intended target classes from the task name using simple keyword heuristics.
    Returns a set of COCO-ish class names to match against detections.
    """
    task_name = (task.get("task_name") or "").lower()
    if not task_name:
        return set()

    tokens = re.findall(r"[a-zA-Z]+", task_name)
    mapping: dict[str, set[str]] = {
        "person": {"person"},
        "intruder": {"person"},
        "car": {"car"},
        "truck": {"truck"},
        "bus": {"bus"},
        "bicycle": {"bicycle"},
        "bike": {"bicycle", "motorcycle"},
        "motorcycle": {"motorcycle"},
        "knife": {"knife"},
        "scissor": {"scissors"},
        "scissors": {"scissors"},
        "phone": {"cell phone"},
        "mobile": {"cell phone"},
        "cell": {"cell phone"},
        "dog": {"dog"},
        "cat": {"cat"},
    }

    targets: set[str] = set()
    for token in tokens:
        if token in mapping:
            targets |= mapping[token]
    return {t.lower() for t in targets}


def check_event_match(task: dict, class_names: list[str]) -> Optional[str]:
    """
    Map task_type to a simplistic event rule using detected class names.

    Returns a human-readable event label if matched, else None.
    """
    task_type = (task.get("task_type") or "").lower()
    if not task_type:
        return None
    lower_classes = [c.lower() for c in class_names]

    if task_type == "person_activity_detection":
        if "person" in lower_classes:
            return "Person activity detected"

    if task_type == "fight_detection":
        # Naive heuristic: multiple persons in frame might indicate potential altercation
        if lower_classes.count("person") >= 2:
            return "Potential fight activity (multiple persons)"

    if task_type == "object_detection":
        # Only trigger if the detected classes include a target inferred from task_name
        targets = infer_object_targets(task)
        if not targets:
            return None
        matched = sorted(t for t in targets if t in lower_classes)
        if matched:
            return f"Object detected: {', '.join(matched)}"

    return None


def draw_boxes(frame, detections) -> "any":
    """
    Placeholder for drawing overlays on frames.
    detections is expected to carry boxes/classes/scores in the future.
    Currently returns the frame unchanged.
    """
    return frame


