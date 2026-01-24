"""
Pipeline Data Contracts
-----------------------

Defines the data structures passed between pipeline stages.
These contracts are frozen - they define the stable API between stages.
"""

from typing import Optional, Dict, Any
from datetime import datetime

# Re-export contracts from other modules for convenience
from app.processing.sources.contracts import FramePacket
from app.processing.detections.contracts import DetectionPacket


class RuleMatch:
    """
    Result of rule evaluation.
    
    This is what rules return when they match.
    """
    
    def __init__(
        self,
        label: str,
        rule_index: int,
        matched_detection_indices: list[int],
        report: Optional[Dict[str, Any]] = None
    ):
        self.label = label
        self.rule_index = rule_index
        self.matched_detection_indices = matched_detection_indices
        self.report = report
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backward compatibility)."""
        result = {
            "label": self.label,
            "rule_index": self.rule_index,
            "matched_detection_indices": self.matched_detection_indices,
        }
        if self.report is not None:
            result["report"] = self.report
        return result
