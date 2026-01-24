"""
YOLO Detector Provider
----------------------

Provider for YOLO object detection models (standard YOLO models without pose).
Handles loading and initialization of YOLO detection models.
"""

import os
from typing import Optional, Any

# Optional YOLO (used if installed)
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:  # noqa: BLE001
    YOLO = None  # type: ignore[assignment]
    YOLO_AVAILABLE = False

from app.processing.models.contracts import Provider, Model


class YOLODetectorProvider(Provider):
    """
    Provider for YOLO object detection models.
    
    Handles loading of standard YOLO models (yolov5, yolov8, yolov9, yolov10, yolov11).
    Automatically downloads standard models if not found locally.
    """
    
    def load(self, model_id: str) -> Optional[Model]:
        """
        Load a YOLO detection model.
        
        Args:
            model_id: Model identifier (path or standard model name like "yolov8n.pt")
        
        Returns:
            Loaded YOLO model instance or None if loading failed
        """
        if not YOLO_AVAILABLE:
            print("[YOLODetectorProvider] ⚠️ YOLO not available (ultralytics not installed). Skipping detection.")
            return None
        
        # Clean model name: remove trailing slashes, whitespace, and normalize
        model_name = model_id.strip().rstrip('/').rstrip('\\')
        
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
            print(f"[YOLODetectorProvider] ❌ Model file not found: {model_name}")
            return None
        
        try:
            # YOLO will automatically download standard models if they don't exist
            # For standard models, YOLO handles the download automatically
            # For custom paths, we've already checked existence above
            print(f"[YOLODetectorProvider] 📥 Loading YOLO model: {model_name}...")
            model = YOLO(model_name)
            print(f"[YOLODetectorProvider] ✅ YOLO model loaded: {model_name}")
            return model
        except FileNotFoundError as exc:
            # If it's a standard model and download failed, provide helpful message
            if is_standard_model:
                print(f"[YOLODetectorProvider] ⚠️ Model file not found locally: {model_name}")
                print(f"[YOLODetectorProvider] 💡 Attempting to download '{model_name}' automatically...")
                try:
                    # YOLO automatically downloads on first use, but we can retry
                    # The download happens automatically when YOLO() is called with a standard model name
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
            # If it's a standard model, YOLO should auto-download, so this might be a different error
            if is_standard_model:
                print(f"[YOLODetectorProvider] 💡 Note: Standard YOLO models should download automatically. Check internet connection.")
            return None
