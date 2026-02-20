from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from bson.errors import InvalidId

from ...domain.models.agent import Agent
from ...domain.constants import AgentFields
from ...utils.datetime_utils import utc_now
from ...utils.db import get_collection
from ..session_state.agent_state import get_agent_state
from ..exceptions import DatabaseError, RuleNotFoundError
from .flow_diagram_utils import generate_agent_flow_diagram

logger = logging.getLogger(__name__)



# ============================================================================
# REPOSITORY MANAGEMENT
# ============================================================================

agent_repository: Optional[Any] = None
camera_repository: Optional[Any] = None


def set_agent_repository(repository):
    """Set the agent repository for saving agents."""
    global agent_repository
    agent_repository = repository
    logger.debug("Agent repository set: %s", type(repository))


def set_camera_repository(repository):
    """Set the camera repository for looking up cameras."""
    global camera_repository
    camera_repository = repository


def get_camera_repository():
    """Return the camera repository (for use by other tools)."""
    return camera_repository


# ============================================================================
# SYNC AGENT SAVE (no asyncio – works in Docker and from running event loop)
# ============================================================================

def agent_to_dict_sync(agent: Agent) -> dict:
    """Convert Agent to MongoDB document (sync, same shape as MongoAgentRepository)._agent_to_dict)."""
    d = {
        AgentFields.NAME: agent.name,
        AgentFields.CAMERA_ID: agent.camera_id or "",
        AgentFields.MODEL: agent.model,
        AgentFields.FPS: agent.fps,
        AgentFields.RULES: agent.rules,
        AgentFields.RUN_MODE: agent.run_mode,
        AgentFields.INTERVAL_MINUTES: agent.interval_minutes,
        AgentFields.CHECK_DURATION_SECONDS: agent.check_duration_seconds,
        AgentFields.START_TIME: agent.start_time,
        AgentFields.END_TIME: agent.end_time,
        AgentFields.ZONE: agent.zone,
        AgentFields.REQUIRES_ZONE: agent.requires_zone,
        AgentFields.STATUS: agent.status,
        AgentFields.CREATED_AT: agent.created_at,
        AgentFields.OWNER_USER_ID: agent.owner_user_id,
        AgentFields.STREAM_CONFIG: agent.stream_config,
        AgentFields.VIDEO_PATH: getattr(agent, "video_path", "") or "",
        AgentFields.SOURCE_TYPE: getattr(agent, "source_type", "rtsp") or "rtsp",
    }
    if agent.id:
        try:
            d[AgentFields.MONGO_ID] = ObjectId(agent.id)
        except (InvalidId, ValueError, TypeError):
            pass
    return d


def document_to_agent_sync(document: dict) -> Agent:
    """Convert MongoDB document to Agent (sync, same shape as MongoAgentRepository._document_to_agent)."""
    if not document or AgentFields.MONGO_ID not in document:
        raise ValueError("Invalid document: missing _id field")
    start_time = document.get(AgentFields.START_TIME)
    end_time = document.get(AgentFields.END_TIME)
    if start_time and isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if end_time and isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    return Agent(
        id=str(document[AgentFields.MONGO_ID]),
        name=document.get(AgentFields.NAME, ""),
        camera_id=document.get(AgentFields.CAMERA_ID, ""),
        model=document.get(AgentFields.MODEL, ""),
        fps=document.get(AgentFields.FPS),
        rules=document.get(AgentFields.RULES, []),
        run_mode=document.get(AgentFields.RUN_MODE),
        interval_minutes=document.get(AgentFields.INTERVAL_MINUTES),
        check_duration_seconds=document.get(AgentFields.CHECK_DURATION_SECONDS),
        start_time=start_time,
        end_time=end_time,
        zone=document.get(AgentFields.ZONE),
        requires_zone=document.get(AgentFields.REQUIRES_ZONE, False),
        status=document.get(AgentFields.STATUS, "ACTIVE"),
        created_at=document.get(AgentFields.CREATED_AT),
        owner_user_id=document.get(AgentFields.OWNER_USER_ID),
        stream_config=document.get(AgentFields.STREAM_CONFIG),
        video_path=document.get(AgentFields.VIDEO_PATH, "") or "",
        source_type=document.get(AgentFields.SOURCE_TYPE, "rtsp") or "rtsp",
    )


def save_agent_sync(agent: Agent) -> Agent:
    """
    Save agent using sync PyMongo with retry logic.
    
    Args:
        agent: Agent instance to save
        
    Returns:
        Agent: Saved agent with ID
        
    Raises:
        DatabaseError: If save operation fails
    """
    source_desc = (
        f"video_path={getattr(agent, 'video_path', '')}"
        if (getattr(agent, "source_type", "") or "").strip().lower() == "video_file"
        else f"camera_id={agent.camera_id}"
    )
    logger.info("Saving agent: %s (%s)", agent.name, source_desc)
    coll = get_collection("agents")
    agent_dict = agent_to_dict_sync(agent)

    # Update existing agent when a valid ObjectId is present.
    if agent.id:
        try:
            oid = ObjectId(agent.id)
            logger.debug("Updating existing agent with ID: %s", agent.id)

            result = coll.update_one(
                {AgentFields.MONGO_ID: oid},
                {"$set": {k: v for k, v in agent_dict.items() if k != AgentFields.MONGO_ID}},
            )

            if result.matched_count > 0:
                doc = coll.find_one({AgentFields.MONGO_ID: oid})
                if doc is None:
                    raise DatabaseError(
                        f"Agent {agent.id} was updated but could not be retrieved",
                        operation="update",
                        retryable=False,
                    )
                logger.info("Agent updated successfully: %s", agent.id)
                return document_to_agent_sync(doc)

            logger.warning("Agent %s not found for update, inserting instead", agent.id)
        except (InvalidId, ValueError, TypeError) as exc:
            logger.warning("Invalid agent ID %s, inserting instead: %s", agent.id, exc)

    # Insert new agent document.
    if AgentFields.MONGO_ID in agent_dict:
        del agent_dict[AgentFields.MONGO_ID]

    logger.debug("Inserting new agent")
    result = coll.insert_one(agent_dict)
    doc = coll.find_one({AgentFields.MONGO_ID: result.inserted_id})
    if doc is None:
        raise DatabaseError(
            "Agent was created but could not be retrieved",
            operation="insert",
            retryable=False,
        )
    return document_to_agent_sync(doc)


# ============================================================================
# AGENT SAVING
# ============================================================================

def save_to_db(session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Finalize the agent configuration and save to database.
    Returns deterministic payloads that the LLM can reason about.
    """

    def failure(
        *,
        code: str,
        user_message: str,
        status_value: str,
        retryable: bool = False,
    ) -> Dict[str, Any]:
        return {
            "error": code,
            "code": code,
            "user_message": user_message,
            "status": status_value,
            "saved": False,
            "message": user_message,
            "retryable": retryable,
        }

    try:
        if agent_repository is None:
            return failure(
                code="repository_not_initialized",
                user_message="System error: Agent repository not available.",
                status_value="COLLECTING",
                retryable=False,
            )

        agent_state = get_agent_state(session_id)

        # Recompute once if status/fields are out of sync.
        if agent_state.status == "COLLECTING" and not agent_state.missing_fields:
            from .kb_utils import compute_missing_fields, get_rule

            try:
                rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
                if rule:
                    compute_missing_fields(agent_state, rule)
                    if not agent_state.missing_fields:
                        agent_state.status = "CONFIRMATION"
            except Exception:
                logger.exception("Failed to recompute missing_fields for session %s", session_id)

        if agent_state.status != "CONFIRMATION":
            return failure(
                code="invalid_state_transition",
                user_message="Agent is not ready to save. Please complete all required fields first.",
                status_value=agent_state.status,
                retryable=False,
            )

        if agent_state.missing_fields:
            return failure(
                code="missing_fields",
                user_message="Please provide all required fields before saving.",
                status_value=agent_state.status,
                retryable=False,
            )

        if not user_id:
            return failure(
                code="missing_user_id",
                user_message="Authentication error: User ID is required.",
                status_value=agent_state.status,
                retryable=False,
            )

        run_mode = agent_state.fields.get("run_mode")
        if run_mode and run_mode not in ["continuous", "patrol"]:
            return failure(
                code="invalid_run_mode",
                user_message="Invalid run mode. Please use continuous or patrol.",
                status_value=agent_state.status,
                retryable=False,
            )

        from .kb_utils import compute_requires_zone, get_rule

        try:
            rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
        except RuleNotFoundError:
            return failure(
                code="invalid_rule_id",
                user_message="The requested rule is not available.",
                status_value=agent_state.status,
                retryable=False,
            )

        requires_zone = compute_requires_zone(rule, run_mode) if rule else False

        start_time_str = agent_state.fields.get("start_time")
        end_time_str = agent_state.fields.get("end_time")
        start_time = None
        end_time = None
        try:
            if start_time_str:
                start_time = (
                    datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    if isinstance(start_time_str, str)
                    else start_time_str
                )
            if end_time_str:
                end_time = (
                    datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                    if isinstance(end_time_str, str)
                    else end_time_str
                )
        except ValueError:
            return failure(
                code="invalid_time_format",
                user_message="Invalid time format. Please use a valid date and time.",
                status_value=agent_state.status,
                retryable=False,
            )

        source_type = (agent_state.fields.get("source_type") or "").strip().lower()
        video_path = (agent_state.fields.get("video_path") or "").strip()
        is_video_file = source_type == "video_file" and bool(video_path)
        if is_video_file:
            start_time = None
            end_time = None

        payload: Dict[str, Any] = {
            "name": agent_state.fields.get("name", ""),
            "camera_id": agent_state.fields.get("camera_id") or "",
            "model": agent_state.fields.get("model"),
            "fps": agent_state.fields.get("fps"),
            "rules": agent_state.fields.get("rules", []),
            "run_mode": run_mode,
            "start_time": start_time,
            "end_time": end_time,
            "requires_zone": requires_zone,
            "status": "PENDING",
            "created_at": utc_now(),
            "owner_user_id": user_id,
            "video_path": video_path if is_video_file else "",
            "source_type": "video_file" if is_video_file else "rtsp",
        }

        if run_mode == "patrol":
            interval_minutes = agent_state.fields.get("interval_minutes")
            check_duration = agent_state.fields.get("check_duration_seconds")
            if interval_minutes is not None:
                payload["interval_minutes"] = interval_minutes
            if check_duration is not None:
                payload["check_duration_seconds"] = check_duration

        if requires_zone:
            zone = agent_state.fields.get("zone")
            if zone is not None:
                payload["zone"] = zone

        # Rules that use zone type motion_rois: build rules from zone (looms) + config
        zone = agent_state.fields.get("zone")
        zone_has_looms = (
            isinstance(zone, dict)
            and zone.get("type") == "motion_rois"
            and isinstance(zone.get("looms"), list)
            and len(zone.get("looms", [])) >= 1
        )

        if agent_state.rule_id == "loom_machine_state" and zone_has_looms:
            idle_threshold = agent_state.fields.get("idle_threshold_minutes")
            if idle_threshold is None:
                return failure(
                    code="missing_idle_threshold",
                    user_message="Idle threshold (in minutes) is required. Please provide how many minutes the machine should be idle before alerting.",
                    status_value=agent_state.status,
                    retryable=False,
                )
            def _get(key, default):
                v = agent_state.fields.get(key)
                return v if v is not None else default
            payload["rules"] = [
                {
                    "type": "loom_machine_state",
                    "looms": zone["looms"],
                    "config": {
                        "idle_threshold_minutes": idle_threshold,
                        "analysis_window_minutes": _get("analysis_window_minutes", 15),
                        "frames_per_minute": _get("frames_per_minute", 5),
                        "notify_on_idle": _get("notify_on_idle", True),
                        "alert_cooldown_minutes": _get("alert_cooldown_minutes", 30),
                        "rolling_window_seconds": _get("rolling_window_seconds", 15),
                        "motion_ratio_running": _get("motion_ratio_running", 0.02),
                        "motion_ratio_stopped": _get("motion_ratio_stopped", 0.005),
                        "debounce_seconds": _get("debounce_seconds", 5),
                        "mog2_history": _get("mog2_history", 500),
                        "mog2_var_threshold": _get("mog2_var_threshold", 16),
                        "mog2_detect_shadows": _get("mog2_detect_shadows", False),
                        "preprocess_blur_ksize": _get("preprocess_blur_ksize", 5),
                        "morph_ksize": _get("morph_ksize", 3),
                    },
                }
            ]
        elif agent_state.rule_id == "person_near_machine":
            zone_polygon = (
                isinstance(zone, dict)
                and zone.get("type") == "polygon"
                and isinstance(zone.get("coordinates"), list)
                and len(zone.get("coordinates", [])) >= 3
            )
            if zone_polygon:
                absence_threshold = agent_state.fields.get("absence_threshold_minutes")
                if absence_threshold is None:
                    return failure(
                        code="missing_absence_threshold",
                        user_message="Unattended threshold (absence_threshold_minutes) is required. Please provide how many minutes the machine can be unattended before alerting.",
                        status_value=agent_state.status,
                        retryable=False,
                    )
                def _get_pnm(key, default):
                    v = agent_state.fields.get(key)
                    return v if v is not None else default
                payload["rules"] = [
                    {
                        "type": "person_near_machine",
                        "zone": {"type": "polygon", "coordinates": zone["coordinates"]},
                        "config": {
                            "absence_threshold_minutes": absence_threshold,
                            "grace_time_seconds": _get_pnm("grace_time_seconds", 10.0),
                            "min_presence_seconds": _get_pnm("min_presence_seconds", 5.0),
                            "confidence_threshold": _get_pnm("confidence_threshold", 0.5),
                            "tracker_max_age": _get_pnm("tracker_max_age", 30),
                            "tracker_min_hits": _get_pnm("tracker_min_hits", 3),
                            "tracker_iou_threshold": _get_pnm("tracker_iou_threshold", 0.3),
                            "notify_on_absence": _get_pnm("notify_on_absence", True),
                            "alert_cooldown_minutes": _get_pnm("alert_cooldown_minutes", 30.0),
                            "emit_state_transitions": _get_pnm("emit_state_transitions", False),
                            "emit_periodic_updates": _get_pnm("emit_periodic_updates", False),
                            "fps": _get_pnm("fps", 5),
                        },
                    }
                ]

        try:
            agent = Agent(**payload)
        except Exception as exc:
            logger.exception("Invalid agent payload for session %s: %s", session_id, exc)
            return failure(
                code="invalid_agent_payload",
                user_message="Invalid configuration. Please check the provided values.",
                status_value=agent_state.status,
                retryable=False,
            )

        try:
            saved_agent = save_agent_sync(agent)
        except Exception as exc:
            logger.exception("Failed saving agent for session %s", session_id)
            raise DatabaseError(
                f"Failed to save agent to database: {str(exc)}",
                operation="save",
                retryable=True,
                user_message="Failed to save agent. Please try again.",
            )

        if not saved_agent or not getattr(saved_agent, "id", None):
            return failure(
                code="save_result_invalid",
                user_message="Database error: Agent was not properly saved. Please try again.",
                status_value=agent_state.status,
                retryable=True,
            )

        agent_state.status = "SAVED"
        agent_state.saved_agent_id = saved_agent.id
        agent_state.saved_agent_name = saved_agent.name

        flow_diagram = None
        try:
            flow_diagram = generate_agent_flow_diagram(saved_agent)
        except Exception:
            logger.exception("Failed to generate flow diagram for agent %s", saved_agent.id)

        return {
            "status": "SAVED",
            "saved": True,
            "message": "Agent configuration saved. Tell user agent is successfully created",
            "agent_id": saved_agent.id,
            "agent_name": saved_agent.name,
            "flow_diagram": flow_diagram,
            "retryable": False,
            "code": "saved",
        }
    except Exception as exc:
        logger.exception("Unexpected error saving agent for session %s: %s", session_id, exc)
        return failure(
            code="unexpected_save_error",
            user_message="An unexpected error occurred while saving. Please try again.",
            status_value="COLLECTING",
            retryable=True,
        )

