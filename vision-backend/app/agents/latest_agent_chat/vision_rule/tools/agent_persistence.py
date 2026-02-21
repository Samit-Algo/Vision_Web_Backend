"""
Persist agent to MongoDB. Same document shape and logic as ADK save_to_db_tool.
Uses only app.domain and app.utils (no app.agents). Used by save_to_db tool.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# Optional: only when running with main app (app.domain / app.utils on path)
try:
    from bson import ObjectId
    from bson.errors import InvalidId
    from app.domain.constants import AgentFields
    from app.domain.models.agent import Agent
    from app.utils.db import get_collection
    from app.utils.datetime_utils import utc_now
    _PERSISTENCE_AVAILABLE = True
except ImportError:
    _PERSISTENCE_AVAILABLE = False
    Agent = None  # type: ignore
    AgentFields = None  # type: ignore
    get_collection = None  # type: ignore
    utc_now = None  # type: ignore
    ObjectId = None  # type: ignore
    InvalidId = Exception  # type: ignore


def is_persistence_available() -> bool:
    return _PERSISTENCE_AVAILABLE


def agent_to_dict(agent: Any) -> dict[str, Any]:
    """Agent -> MongoDB document (same shape as ADK agent_to_dict_sync)."""
    if not _PERSISTENCE_AVAILABLE or AgentFields is None:
        raise RuntimeError("Persistence not available")
    d: dict[str, Any] = {
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


def document_to_agent(document: dict[str, Any]) -> Any:
    """MongoDB document -> Agent (same shape as ADK document_to_agent_sync)."""
    if not _PERSISTENCE_AVAILABLE or AgentFields is None or Agent is None:
        raise RuntimeError("Persistence not available")
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


def build_and_save_agent(payload: dict[str, Any]) -> Any:
    """
    Build Agent from payload (add created_at if missing) and save to DB.
    Returns saved Agent. Raises on validation or DB error.
    """
    if not _PERSISTENCE_AVAILABLE or Agent is None or utc_now is None:
        raise RuntimeError("Persistence not available")
    if payload.get("created_at") is None:
        payload = {**payload, "created_at": utc_now()}
    agent = Agent(**payload)
    return save_agent_sync(agent)


def save_agent_sync(agent: Any) -> Any:
    """
    Insert or update agent in MongoDB. Same behavior as ADK save_agent_sync.
    Returns saved Agent with id set.
    """
    if not _PERSISTENCE_AVAILABLE or get_collection is None:
        raise RuntimeError("Persistence not available: app.utils.db not on path")
    coll = get_collection("agents")
    agent_dict = agent_to_dict(agent)

    if agent.id:
        try:
            oid = ObjectId(agent.id)
            result = coll.update_one(
                {AgentFields.MONGO_ID: oid},
                {"$set": {k: v for k, v in agent_dict.items() if k != AgentFields.MONGO_ID}},
            )
            if result.matched_count > 0:
                doc = coll.find_one({AgentFields.MONGO_ID: oid})
                if doc is None:
                    raise RuntimeError("Agent was updated but could not be retrieved")
                return document_to_agent(doc)
        except (InvalidId, ValueError, TypeError):
            pass

    if AgentFields.MONGO_ID in agent_dict:
        del agent_dict[AgentFields.MONGO_ID]
    result = coll.insert_one(agent_dict)
    doc = coll.find_one({AgentFields.MONGO_ID: result.inserted_id})
    if doc is None:
        raise RuntimeError("Agent was created but could not be retrieved")
    return document_to_agent(doc)
