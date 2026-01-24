"""
Detection Merger
----------------

Merges multiple DetectionPacket objects into a single DetectionPacket.
Used when multiple models are run on the same frame.
"""

from typing import List
from datetime import datetime

from app.processing.detections.contracts import DetectionPacket


class DetectionMerger:
    """
    Merges detection packets from multiple models.
    
    Simply concatenates all detections from all models into a single packet.
    No deduplication or filtering is performed (preserves current behavior).
    """
    
    @staticmethod
    def merge(packets: List[DetectionPacket], timestamp: datetime) -> DetectionPacket:
        """
        Merge multiple DetectionPackets into one.
        
        Args:
            packets: List of DetectionPackets to merge
            timestamp: Timestamp for the merged packet (typically current time)
        
        Returns:
            Single merged DetectionPacket
        """
        if not packets:
            # Return empty packet if no packets provided
            return DetectionPacket(
                classes=[],
                scores=[],
                boxes=[],
                keypoints=[],
                ts=timestamp
            )
        
        # Merge all detections by concatenating lists
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
