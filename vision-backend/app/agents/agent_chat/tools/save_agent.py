from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from bson.errors import InvalidId

from ....domain.models.agent import Agent
from ....domain.constants import AgentFields
from ....utils.datetime_utils import utc_now
from ....utils.db import get_collection
from ...exceptions import DatabaseError, RuleNotFoundError
from ..state.agent_state import STEP_COMPLETED, get_agent_state
from .flow_diagram import generate_agent_flow_diagram
from .knowledge_base import can_enter_confirmation, compute_missing_fields, compute_requires_zone, get_rule

AGENT_REPOSITORY: Optional[Any] = None
CAMERA_REPOSITORY: Optional[Any] = None


def set_agent_repository(repository: Any) -> None:
    global AGENT_REPOSITORY
    AGENT_REPOSITORY = repository


def set_camera_repository(repository: Any) -> None:
    global CAMERA_REPOSITORY
    CAMERA_REPOSITORY = repository


def get_camera_repository() -> Any:
    return CAMERA_REPOSITORY


def agent_to_document(agent: Agent) -> dict:
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


def document_to_agent(doc: dict) -> Agent:
    if not doc or AgentFields.MONGO_ID not in doc:
        raise ValueError("Invalid document: missing _id")
    start_time = doc.get(AgentFields.START_TIME)
    end_time = doc.get(AgentFields.END_TIME)
    if start_time and isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if end_time and isinstance(end_time, str):
        end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    return Agent(
        id=str(doc[AgentFields.MONGO_ID]),
        name=doc.get(AgentFields.NAME, ""),
        camera_id=doc.get(AgentFields.CAMERA_ID, ""),
        model=doc.get(AgentFields.MODEL, ""),
        fps=doc.get(AgentFields.FPS),
        rules=doc.get(AgentFields.RULES, []),
        run_mode=doc.get(AgentFields.RUN_MODE),
        interval_minutes=doc.get(AgentFields.INTERVAL_MINUTES),
        check_duration_seconds=doc.get(AgentFields.CHECK_DURATION_SECONDS),
        start_time=start_time,
        end_time=end_time,
        zone=doc.get(AgentFields.ZONE),
        requires_zone=doc.get(AgentFields.REQUIRES_ZONE, False),
        status=doc.get(AgentFields.STATUS, "ACTIVE"),
        created_at=doc.get(AgentFields.CREATED_AT),
        owner_user_id=doc.get(AgentFields.OWNER_USER_ID),
        stream_config=doc.get(AgentFields.STREAM_CONFIG),
        video_path=doc.get(AgentFields.VIDEO_PATH, "") or "",
        source_type=doc.get(AgentFields.SOURCE_TYPE, "rtsp") or "rtsp",
    )


def persist_agent(agent: Agent) -> Agent:
    coll = get_collection("agents")
    doc = agent_to_document(agent)
    if agent.id:
        try:
            oid = ObjectId(agent.id)
            result = coll.update_one(
                {AgentFields.MONGO_ID: oid},
                {"$set": {k: v for k, v in doc.items() if k != AgentFields.MONGO_ID}},
            )
            if result.matched_count > 0:
                out = coll.find_one({AgentFields.MONGO_ID: oid})
                if out:
                    return document_to_agent(out)
        except (InvalidId, ValueError, TypeError):
            pass
    if AgentFields.MONGO_ID in doc:
        del doc[AgentFields.MONGO_ID]
    result = coll.insert_one(doc)
    out = coll.find_one({AgentFields.MONGO_ID: result.inserted_id})
    if not out:
        raise DatabaseError("Agent created but could not be retrieved", operation="insert", retryable=False)
    return document_to_agent(out)


def failure(code: str, user_message: str, status_value: str, retryable: bool = False) -> Dict[str, Any]:
    return {"error": code, "code": code, "user_message": user_message, "status": status_value, "saved": False, "message": user_message, "retryable": retryable}


def save_agent_to_db(session_id: str = "default", user_id: Optional[str] = None) -> Dict[str, Any]:
    if AGENT_REPOSITORY is None:
        return failure("repository_not_initialized", "System error: Agent repository not available.", "COLLECTING", False)

    agent_state = get_agent_state(session_id)
    if agent_state.status == "COLLECTING" and not agent_state.missing_fields:
        try:
            rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
            if rule:
                compute_missing_fields(agent_state, rule)
                if not agent_state.missing_fields and can_enter_confirmation(agent_state, rule):
                    agent_state.status = "CONFIRMATION"
        except Exception:
            pass

    if agent_state.status != "CONFIRMATION":
        return failure("invalid_state_transition", "Please complete all required fields first.", agent_state.status, False)
    if agent_state.missing_fields:
        return failure("missing_fields", "Please provide all required fields before saving.", agent_state.status, False)
    if not user_id:
        return failure("missing_user_id", "Authentication error: User ID is required.", agent_state.status, False)

    run_mode = agent_state.fields.get("run_mode")
    if run_mode and run_mode not in ("continuous", "patrol"):
        return failure("invalid_run_mode", "Invalid run mode. Use continuous or patrol.", agent_state.status, False)

    try:
        rule = get_rule(agent_state.rule_id) if agent_state.rule_id else None
    except RuleNotFoundError:
        return failure("invalid_rule_id", "The requested rule is not available.", agent_state.status, False)

    requires_zone = compute_requires_zone(rule, run_mode) if rule else False
    start_time_str = agent_state.fields.get("start_time")
    end_time_str = agent_state.fields.get("end_time")
    start_time = None
    end_time = None
    try:
        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00")) if isinstance(start_time_str, str) else start_time_str
        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00")) if isinstance(end_time_str, str) else end_time_str
    except ValueError:
        return failure("invalid_time_format", "Invalid time format. Please use a valid date and time.", agent_state.status, False)

    source_type = (agent_state.fields.get("source_type") or "").strip().lower()
    video_path = (agent_state.fields.get("video_path") or "").strip()
    is_video = source_type == "video_file" and bool(video_path)
    if is_video:
        start_time = end_time = None

    name = (agent_state.fields.get("name") or "").strip()
    if not name and rule:
        rule_name = rule.get("rule_name", "Agent")
        class_name = agent_state.fields.get("class") or agent_state.fields.get("gesture") or ""
        name = f"{class_name.replace('_', ' ').title()} {rule_name} Agent" if class_name else f"{rule_name} Agent"

    payload: Dict[str, Any] = {
        "name": name,
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
        "video_path": video_path if is_video else "",
        "source_type": "video_file" if is_video else "rtsp",
    }
    if run_mode == "patrol":
        if agent_state.fields.get("interval_minutes") is not None:
            payload["interval_minutes"] = agent_state.fields["interval_minutes"]
        if agent_state.fields.get("check_duration_seconds") is not None:
            payload["check_duration_seconds"] = agent_state.fields["check_duration_seconds"]
    if requires_zone and agent_state.fields.get("zone") is not None:
        payload["zone"] = agent_state.fields["zone"]

    try:
        agent = Agent(**payload)
    except Exception:
        return failure("invalid_agent_payload", "Invalid configuration. Please check the provided values.", agent_state.status, False)

    try:
        saved = persist_agent(agent)
    except Exception as e:
        raise DatabaseError(str(e), operation="save", retryable=True, user_message="Failed to save agent. Please try again.")

    if not saved or not getattr(saved, "id", None):
        return failure("save_result_invalid", "Database error: Agent was not properly saved. Please try again.", agent_state.status, True)

    agent_state.status = "SAVED"
    agent_state.current_step = STEP_COMPLETED
    agent_state.saved_agent_id = saved.id
    agent_state.saved_agent_name = saved.name
    flow_diagram = None
    try:
        flow_diagram = generate_agent_flow_diagram(saved)
    except Exception:
        pass

    return {
        "status": "SAVED",
        "saved": True,
        "message": "Agent configuration saved. Tell user agent is successfully created",
        "agent_id": saved.id,
        "agent_name": saved.name,
        "flow_diagram": flow_diagram,
        "retryable": False,
        "code": "saved",
    }
