"""
Pipeline Data Contracts
-----------------------

Defines the data structures passed between pipeline stages.
Stage 4 (evaluate_rules_stage) produces RuleMatch; other stages use FramePacket and DetectionPacket.
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------
# Re-exports (so callers can import pipeline data models from one place)
# -----------------------------------------------------------------------------
from app.processing.data_input.data_models import FramePacket
from app.processing.processing_output.data_models import DetectionPacket

# -----------------------------------------------------------------------------
# Rule evaluation result
# -----------------------------------------------------------------------------


class RuleMatch:
    """
    Result of rule evaluation (Stage 4).

    When a scenario matches (e.g. fall detected, person in zone), the pipeline
    creates a RuleMatch with label, rule index, matched detection indices, and
    optional report (e.g. VLM description, counts).
    """

    def __init__(
        self,
        label: str,
        rule_index: int,
        matched_detection_indices: list[int],
        report: Optional[Dict[str, Any]] = None,
    ):
        self.label = label
        self.rule_index = rule_index
        self.matched_detection_indices = matched_detection_indices
        self.report = report

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for event payloads and backward compatibility)."""
        result = {
            "label": self.label,
            "rule_index": self.rule_index,
            "matched_detection_indices": self.matched_detection_indices,
        }
        if self.report is not None:
            result["report"] = self.report
        return result
