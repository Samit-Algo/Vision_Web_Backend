"""
Pipeline Stages
---------------

Individual processing stages that make up the pipeline.
Each stage is a pure function that transforms input to output.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.processing.detections.builder import DetectionBuilder
from app.processing.detections.contracts import DetectionPacket
from app.processing.detections.merger import DetectionMerger
from app.processing.pipeline.context import PipelineContext
from app.processing.pipeline.contracts import RuleMatch
from app.processing.scenarios.contracts import ScenarioFrameContext
from app.processing.scenarios.registry import get_scenario_class
from app.processing.sources.contracts import FramePacket
from app.utils.datetime_utils import now

# Import scenarios module to trigger registration decorators
from app.processing.scenarios import scenario_registry  # noqa: F401


# ============================================================================
# STAGE 1: ACQUIRE FRAME
# ============================================================================

def acquire_frame_stage(source) -> Optional[FramePacket]:
    """Stage 1: Acquire frame from source."""
    return source.read_frame()


# ============================================================================
# INFERENCE HELPERS
# ============================================================================

def _any_scenario_requires_yolo(context: PipelineContext) -> bool:
    """Check if any scenario in the rules requires YOLO detections."""
    if not context.rules:
        return False

    if hasattr(context, '_scenario_instances'):
        for scenario_instance in context._scenario_instances.values():
            if scenario_instance.requires_yolo_detections():
                return True

    for rule_idx, rule in enumerate(context.rules):
        rule_type = str(rule.get("type", "")).strip().lower()
        scenario_class = get_scenario_class(rule_type)

        if scenario_class is None:
            continue

        if hasattr(context, '_scenario_instances') and rule_idx in context._scenario_instances:
            if context._scenario_instances[rule_idx].requires_yolo_detections():
                return True
        else:
            try:
                scenario_config = {k: v for k, v in rule.items() if k != "type"}
                temp_instance = scenario_class(scenario_config, context)
                if temp_instance.requires_yolo_detections():
                    return True
            except Exception:
                return True

    return False


# ============================================================================
# STAGE 2: INFERENCE
# ============================================================================

def inference_stage(
    context: PipelineContext,
    frame_packet: FramePacket,
    models: List[Any]
) -> List[DetectionPacket]:
    """Stage 2: Run model inference on frame."""
    if not _any_scenario_requires_yolo(context):
        return []

    frame = frame_packet.frame
    detection_packets: List[DetectionPacket] = []

    for model in models:
        try:
            results = model(frame, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {context.task_id}] ⚠️ YOLO error: {exc}")
            continue
        if results:
            first_result = results[0]
            packet = DetectionBuilder.from_yolo_result(first_result, now())
            detection_packets.append(packet)

    return detection_packets


# ============================================================================
# STAGE 3: MERGE DETECTIONS
# ============================================================================

def merge_detections_stage(detection_packets: List[DetectionPacket]) -> DetectionPacket:
    """Stage 3: Merge detections from multiple models."""
    return DetectionMerger.merge(detection_packets, now())


# ============================================================================
# STAGE 4: EVALUATE RULES
# ============================================================================

def evaluate_rules_stage(
    context: PipelineContext,
    merged_packet: DetectionPacket,
    frame_packet: Optional[FramePacket] = None
) -> tuple[Optional[RuleMatch], List[int]]:
    """
    Stage 4: Evaluate rules against detections.

    All rules now use scenario implementations.
    Each rule type must have a corresponding scenario registered.
    """
    if not context.rules:
        return None, []

    detections = merged_packet.to_dict()

    if not hasattr(context, '_scenario_instances'):
        context._scenario_instances: Dict[int, Any] = {}

    all_matched_indices: List[int] = []
    event = None

    for rule_idx, rule in enumerate(context.rules):
        rule_type = str(rule.get("type", "")).strip().lower()

        scenario_class = get_scenario_class(rule_type)
        if scenario_class is None:
            print(f"[evaluate_rules_stage] ⚠️  No scenario found for rule type '{rule_type}'. Skipping.")
            continue

        if frame_packet is None:
            print(f"[evaluate_rules_stage] ⚠️  Frame packet required for scenario '{rule_type}'. Skipping.")
            continue

        try:
            if rule_idx not in context._scenario_instances:
                scenario_config = {k: v for k, v in rule.items() if k != "type"}
                context._scenario_instances[rule_idx] = scenario_class(scenario_config, context)

            scenario_instance = context._scenario_instances[rule_idx]

            frame_timestamp = datetime.fromtimestamp(frame_packet.timestamp)
            frame_context = ScenarioFrameContext(
                frame=frame_packet.frame,
                frame_index=context.frame_index,
                timestamp=frame_timestamp,
                detections=merged_packet,
                rule_matches=[],
                pipeline_context=context
            )

            scenario_events = scenario_instance.process(frame_context)

            if scenario_events:
                scenario_event = scenario_events[0]

                report = None
                if scenario_event.metadata:
                    report = scenario_event.metadata.get("report")
                    if report is None:
                        report = {k: v for k, v in scenario_event.metadata.items() if k != "report"}

                rule_result = {
                    "label": scenario_event.label,
                    "matched_detection_indices": scenario_event.detection_indices,
                    "report": report
                }

                if rule_result:
                    idx_list = rule_result.get("matched_detection_indices")
                    if isinstance(idx_list, list):
                        all_matched_indices.extend(idx_list)
                    if not event and rule_result.get("label"):
                        event = rule_result
                        event.setdefault("rule_index", rule_idx)
        except Exception as exc:
            print(f"[evaluate_rules_stage] ⚠️  Error processing scenario '{rule_type}': {exc}")
            continue

    rule_match = None
    if event and event.get("label"):
        rule_match = RuleMatch(
            label=str(event["label"]).strip(),
            rule_index=event.get("rule_index", 0),
            matched_detection_indices=event.get("matched_detection_indices", []),
            report=event.get("report")
        )

    return rule_match, all_matched_indices
