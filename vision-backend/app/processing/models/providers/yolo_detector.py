"""
YOLO Detector Provider
----------------------

Provider for YOLO object detection models (standard YOLO models without pose).
Handles loading and initialization of YOLO detection models.
"""

import os
from typing import Any, Optional

from app.processing.models.data_models import Model, Provider
from app.core.config import get_settings

try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]
    YOLO_AVAILABLE = False


# ============================================================================
# YOLO DETECTOR PROVIDER
# ============================================================================

class YOLODetectorProvider(Provider):
    """
    Provider for YOLO object detection models.

    Handles loading of standard YOLO models (yolov5, yolov8, yolov9, yolov10, yolov11).
    Automatically downloads standard models if not found locally.
    """

    def load(self, model_id: str) -> Optional[Model]:
        """Load a YOLO detection model."""
        if not YOLO_AVAILABLE:
            print("[YOLODetectorProvider] ⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
            return None

        model_name = model_id.strip().rstrip('/').rstrip('\\')

        # Resolve bare filenames against MODEL_DIR (Docker: /app/models, local: ./models)
        is_file_path = os.sep in model_name or '/' in model_name or '\\' in model_name
        if not is_file_path:
            model_dir_path = os.path.join(get_settings().model_dir, model_name)
            if os.path.isfile(model_dir_path):
                model_name = model_dir_path

        is_standard_model = (
            model_name.startswith("yolov8") or
            model_name.startswith("yolov5") or
            model_name.startswith("yolo11") or
            model_name.startswith("yolo10") or
            model_name.startswith("yolo9")
        ) and model_name.endswith(".pt")

        is_file_path = os.sep in model_name or '/' in model_name or '\\' in model_name

        if is_file_path and not os.path.exists(model_name):
            print(f"[YOLODetectorProvider] ❌ Model file not found: {model_name}")
            return None

        try:
            print(f"[YOLODetectorProvider] 📥 Loading YOLO model: {model_name}...")
            model = YOLO(model_name)
            print(f"[YOLODetectorProvider] ✅ YOLO model loaded: {model_name}")
            return model
        except FileNotFoundError:
            if is_standard_model:
                print(f"[YOLODetectorProvider] ⚠️ Model file not found locally: {model_name}")
                print(f"[YOLODetectorProvider] 💡 Attempting to download '{model_name}' automatically...")
                try:
                    model = YOLO(model_name)
                    print(f"[YOLODetectorProvider] ✅ YOLO model downloaded and loaded: {model_name}")
                    return model
                except Exception as retry_exc:  # noqa: BLE001
                    print(f"[YOLODetectorProvider] ❌ Failed to download YOLO model '{model_name}': {retry_exc}")
                    print(f"[YOLODetectorProvider] 💡 Please check your internet connection or download manually")
                    return None
            else:
                print(f"[YOLODetectorProvider] ❌ Model file not found: {model_name}")
                return None
        except Exception as exc:  # noqa: BLE001
            print(f"[YOLODetectorProvider] ❌ Failed to load YOLO model '{model_name}': {exc}")
            if is_standard_model:
                print(f"[YOLODetectorProvider] 💡 Note: Standard YOLO models should download automatically. Check internet connection.")
            return None
