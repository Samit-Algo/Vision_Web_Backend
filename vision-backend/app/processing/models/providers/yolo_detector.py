"""
YOLO detector provider
----------------------

Loads standard YOLO object detection models (no pose). Resolves paths against MODEL_DIR;
downloads standard names (e.g. yolov8m.pt) if not found locally.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import os
from typing import Optional

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from app.processing.models.data_models import Model, Provider
from app.core.config import get_settings

try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]
    YOLO_AVAILABLE = False

# -----------------------------------------------------------------------------
# Provider
# -----------------------------------------------------------------------------


class YOLODetectorProvider(Provider):
    """Loads YOLO detection models (yolov5/v8/v9/v10/v11, box_detection, fire_detection)."""

    def load(self, model_id: str) -> Optional[Model]:
        """Load a YOLO detection model. Resolves path; downloads standard names if missing."""
        if not YOLO_AVAILABLE:
            print("[YOLODetectorProvider] ⚠️ ultralytics not installed. Skipping.")
            return None

        model_name = model_id.strip().rstrip("/").rstrip("\\")
        is_file_path = os.sep in model_name or "/" in model_name or "\\" in model_name
        if not is_file_path:
            model_dir_path = os.path.join(get_settings().model_dir, model_name)
            if os.path.isfile(model_dir_path):
                model_name = model_dir_path

        is_standard_model = (
            (model_name.startswith("yolov8") or model_name.startswith("yolov5")
             or model_name.startswith("yolo11") or model_name.startswith("yolo10")
             or model_name.startswith("yolo9") or model_name.startswith("box_detection")
             or model_name.startswith("fire_detection"))
            and model_name.endswith(".pt")
        )

        if is_file_path and not os.path.exists(model_name):
            print(f"[YOLODetectorProvider] ❌ Model file not found: {model_name}")
            return None

        try:
            print(f"[YOLODetectorProvider] 📥 Loading: {model_name}...")
            model = YOLO(model_name)
            print(f"[YOLODetectorProvider] ✅ Loaded: {model_name}")
            return model
        except FileNotFoundError:
            if is_standard_model:
                print(f"[YOLODetectorProvider] 💡 Downloading '{model_name}'...")
                try:
                    model = YOLO(model_name)
                    print(f"[YOLODetectorProvider] ✅ Downloaded and loaded: {model_name}")
                    return model
                except Exception as e:  # noqa: BLE001
                    print(f"[YOLODetectorProvider] ❌ Download failed: {e}")
                    return None
            print(f"[YOLODetectorProvider] ❌ Model file not found: {model_name}")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"[YOLODetectorProvider] ❌ Load failed: {exc}")
            return None
