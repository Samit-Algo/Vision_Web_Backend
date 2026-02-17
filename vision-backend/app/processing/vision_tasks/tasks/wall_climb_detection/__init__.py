"""Wall climb detection: keypoint-based crossing (below → above) with VLM confirmation; red keypoints only."""
from .scenario import WallClimbScenario
from .config import WallClimbConfig
from .wall_zone_utils import (
    wall_top_and_x_range,
    is_person_above_wall_by_keypoints,
    is_box_climbing,
    check_climbing_keypoints_above_zone,
)
from .state import WallClimbDetectionState
from .types import PoseFrame, WallClimbAnalysis, WallClimbVLMConfirmation
from .vlm_handler import (
    build_wall_climb_detection_prompt,
    should_call_vlm,
    call_vlm,
)

__all__ = [
    "WallClimbScenario",
    "WallClimbConfig",
    "WallClimbDetectionState",
    "PoseFrame",
    "WallClimbAnalysis",
    "WallClimbVLMConfirmation",
    "wall_top_and_x_range",
    "is_person_above_wall_by_keypoints",
    "is_box_climbing",
    "check_climbing_keypoints_above_zone",
    "build_wall_climb_detection_prompt",
    "should_call_vlm",
    "call_vlm",
]
