"""
Detections Module
-----------------

Provides detection extraction, normalization, and merging:
- DetectionBuilder: Converts model results to DetectionPacket
- DetectionMerger: Merges multiple DetectionPackets into one
"""

from app.processing.detections.builder import DetectionBuilder
from app.processing.detections.merger import DetectionMerger
from app.processing.detections.contracts import DetectionPacket

__all__ = ["DetectionBuilder", "DetectionMerger", "DetectionPacket"]
