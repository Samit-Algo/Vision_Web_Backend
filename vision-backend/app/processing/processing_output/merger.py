"""
Detection Merger
----------------

Merges multiple DetectionPacket objects into a single DetectionPacket.
Used when multiple models are run on the same frame.
"""

from datetime import datetime
from typing import List

from app.processing.processing_output.data_models import DetectionPacket


# ============================================================================
# DETECTION MERGER
# ============================================================================

class DetectionMerger:
    """
    Merges detection packets from multiple models.

    Simply concatenates all detections from all models into a single packet.
    No deduplication or filtering is performed (preserves current behavior).
    """

    @staticmethod
    def merge(packets: List[DetectionPacket], timestamp: datetime) -> DetectionPacket:
        """Merge multiple DetectionPackets into one."""
        if not packets:
            return DetectionPacket(
                classes=[],
                scores=[],
                boxes=[],
                keypoints=[],
                ts=timestamp
            )

        merged_classes: List[str] = []
        merged_scores: List[float] = []
        merged_boxes: List[List[float]] = []
        merged_keypoints: List[List[List[float]]] = []

        for packet in packets:
            merged_classes.extend(packet.classes)
            merged_scores.extend(packet.scores)
            merged_boxes.extend(packet.boxes)
            merged_keypoints.extend(packet.keypoints)

        return DetectionPacket(
            classes=merged_classes,
            scores=merged_scores,
            boxes=merged_boxes,
            keypoints=merged_keypoints,
            ts=timestamp
        )
