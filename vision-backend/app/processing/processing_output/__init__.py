"""
Processing Output Module
------------------------

Provides detection extraction, normalization, and merging:
- detection_converter: Converts model results to DetectionPacket
- DetectionMerger: Merges multiple DetectionPackets into one
"""

from .detection_converter import convert_from_yolo_result
from .data_models import DetectionPacket
from .merger import DetectionMerger

__all__ = ["convert_from_yolo_result", "DetectionMerger", "DetectionPacket"]
