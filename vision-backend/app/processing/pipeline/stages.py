"""
Pipeline Stages
---------------

Individual processing stages that make up the pipeline.
Each stage is a pure function that transforms input to output.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from app.processing.sources.contracts import FramePacket
from app.processing.detections.contracts import DetectionPacket
from app.processing.detections.builder import DetectionBuilder
from app.processing.detections.merger import DetectionMerger
from app.processing.pipeline.contracts import RuleMatch
from app.processing.pipeline.context import PipelineContext
# Old rule engine removed - all rules now use scenarios
from app.processing.scenarios.contracts import ScenarioFrameContext, ScenarioEvent
from app.processing.scenarios.registry import get_scenario_class
# Import scenarios module to trigger registration decorators
# This ensures all @register_scenario decorators execute
from app.processing.scenarios import scenario_registry  # noqa: F401
from app.utils.datetime_utils import now


def acquire_frame_stage(source) -> Optional[FramePacket]:
    """
    Stage 1: Acquire frame from source.
    
    Args:
        source: Source instance (HubSource)
    
    Returns:
        FramePacket or None if no frame available
    """
    return source.read_frame()


def inference_stage(
    context: PipelineContext,
    frame_packet: FramePacket,
    models: List[Any]
) -> List[DetectionPacket]:
    """
    Stage 2: Run model inference on frame.
    
    Args:
        context: Pipeline context
        frame_packet: Frame to process
        models: List of loaded models
    
    Returns:
        List of DetectionPackets (one per model)
    """
    frame = frame_packet.frame
    detection_packets: List[DetectionPacket] = []
    
    if not models:
        print(f"[worker {context.task_id}] ⚠️ No models loaded!")
        return detection_packets
    
    for model in models:
        try:
            results = model(frame, verbose=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {context.task_id}] ⚠️ YOLO error: {exc}")
            import traceback
            print(f"[worker {context.task_id}] Traceback: {traceback.format_exc()}")
            continue
        if results:
            first_result = results[0]
            packet = DetectionBuilder.from_yolo_result(first_result, now())
            detection_packets.append(packet)
            # Debug: Log detection counts
            print(f"[worker {context.task_id}] 🎯 YOLO inference: {len(packet.boxes)} objects detected")
    
    return detection_packets


def merge_detections_stage(
    detection_packets: List[DetectionPacket]
) -> DetectionPacket:
    """
    Stage 3: Merge detections from multiple models.
    
    Args:
        detection_packets: List of DetectionPackets to merge
    
    Returns:
        Single merged DetectionPacket
    """
    return DetectionMerger.merge(detection_packets, now())


def evaluate_rules_stage(
    context: PipelineContext,
    merged_packet: DetectionPacket,
    frame_packet: Optional[FramePacket] = None
) -> tuple[Optional[RuleMatch], List[int]]:
    """
    Stage 4: Evaluate rules against detections.
    
    All rules now use scenario implementations.
    Each rule type must have a corresponding scenario registered.
    
    Args:
        context: Pipeline context (contains rules and rule state)
        merged_packet: Merged detection packet
        frame_packet: Frame packet (required for scenarios)
    
    Returns:
        Tuple of (RuleMatch or None, list of all matched detection indices)
    """
    if not context.rules:
        return None, []
    
    # Convert packet to dict (for compatibility with scenario processing)
    detections = merged_packet.to_dict()
    
    # Cache scenario instances per rule (avoid recreating each frame)
    if not hasattr(context, '_scenario_instances'):
        context._scenario_instances: Dict[int, Any] = {}
    
    # Evaluate all rules to collect matched_detection_indices for overlay
    all_matched_indices: List[int] = []
    event = None
    
    # Evaluate each rule - all rules now use scenario implementations
    for rule_idx, rule in enumerate(context.rules):
        rule_type = str(rule.get("type", "")).strip().lower()
        
        # Get scenario implementation for this rule type
        scenario_class = get_scenario_class(rule_type)
        if scenario_class is None:
            print(f"[evaluate_rules_stage] ⚠️  No scenario found for rule type '{rule_type}'. Skipping.")
            continue
        
        if frame_packet is None:
            print(f"[evaluate_rules_stage] ⚠️  Frame packet required for scenario '{rule_type}'. Skipping.")
            continue
        
        # Use scenario implementation
        try:
            # Get or create scenario instance (cached per rule index)
            if rule_idx not in context._scenario_instances:
                # Convert rule to scenario config format
                # Rule fields (except "type") become scenario config
                scenario_config = {k: v for k, v in rule.items() if k != "type"}
                context._scenario_instances[rule_idx] = scenario_class(scenario_config, context)
            
            scenario_instance = context._scenario_instances[rule_idx]
            
            # Build ScenarioFrameContext
            from datetime import datetime
            frame_timestamp = datetime.fromtimestamp(frame_packet.timestamp)
            frame_context = ScenarioFrameContext(
                frame=frame_packet.frame,
                frame_index=context.frame_index,
                timestamp=frame_timestamp,
                detections=merged_packet,
                rule_matches=[],
                pipeline_context=context
            )
            
            # Process through scenario
            scenario_events = scenario_instance.process(frame_context)
            
            # Convert scenario events to rule results
            if scenario_events:
                # Use first scenario event as rule match
                scenario_event = scenario_events[0]
                rule_result = {
                    "label": scenario_event.label,
                    "matched_detection_indices": scenario_event.detection_indices,
                    "report": scenario_event.metadata.get("report") if scenario_event.metadata else None
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
    
    # Convert event dict to RuleMatch if present
    rule_match = None
    if event and event.get("label"):
        rule_match = RuleMatch(
            label=str(event["label"]).strip(),
            rule_index=event.get("rule_index", 0),
            matched_detection_indices=event.get("matched_detection_indices", []),
            report=event.get("report")
        )
    
    return rule_match, all_matched_indices


