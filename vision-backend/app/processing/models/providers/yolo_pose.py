"""
YOLO Pose Provider
-----------------

Provider for YOLO pose detection models (models with pose keypoint detection).
Handles loading and initialization of YOLO pose models.
"""

import os
from typing import Optional

from app.processing.models.contracts import Model, Provider

try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]
    YOLO_AVAILABLE = False


# ============================================================================
# YOLO POSE PROVIDER
# ============================================================================

class YOLOPoseProvider(Provider):
    """
    Provider for YOLO pose detection models.

    Handles loading of YOLO pose models (e.g., yolov8n-pose.pt).
    Uses the same YOLO class as detector, but registered separately for clarity.
    """

    def load(self, model_id: str) -> Optional[Model]:
        """Load a YOLO pose model."""
        if not YOLO_AVAILABLE:
            print("[YOLOPoseProvider] ⚠️ YOLO not available (ultralytics not installed). Skipping pose detection.")
            return None

        model_name = model_id.strip().rstrip('/').rstrip('\\')

        is_standard_model = (
            model_name.startswith("yolov8") or
            model_name.startswith("yolov5") or
            model_name.startswith("yolo11") or
            model_name.startswith("yolo10") or
            model_name.startswith("yolo9")
        ) and model_name.endswith(".pt")

        is_file_path = os.sep in model_name or '/' in model_name or '\\' in model_name

        if is_file_path and not os.path.exists(model_name):
            print(f"[YOLOPoseProvider] ❌ Model file not found: {model_name}")
            return None

        try:
            print(f"[YOLOPoseProvider] 📥 Loading YOLO pose model: {model_name}...")
            model = YOLO(model_name)
            print(f"[YOLOPoseProvider] ✅ YOLO pose model loaded: {model_name}")
            return model
        except FileNotFoundError:
            if is_standard_model:
                print(f"[YOLOPoseProvider] ⚠️ Model file not found locally: {model_name}")
                print(f"[YOLOPoseProvider] 💡 Attempting to download '{model_name}' automatically...")
                try:
                    model = YOLO(model_name)
                    print(f"[YOLOPoseProvider] ✅ YOLO pose model downloaded and loaded: {model_name}")
                    return model
                except Exception as retry_exc:  # noqa: BLE001
                    print(f"[YOLOPoseProvider] ❌ Failed to download YOLO pose model '{model_name}': {retry_exc}")
                    print(f"[YOLOPoseProvider] 💡 Please check your internet connection or download manually")
                    return None
            else:
                print(f"[YOLOPoseProvider] ❌ Model file not found: {model_name}")
                return None
        except Exception as exc:  # noqa: BLE001
            print(f"[YOLOPoseProvider] ❌ Failed to load YOLO pose model '{model_name}': {exc}")
            if is_standard_model:
                print(f"[YOLOPoseProvider] 💡 Note: Standard YOLO models should download automatically. Check internet connection.")
            return None
