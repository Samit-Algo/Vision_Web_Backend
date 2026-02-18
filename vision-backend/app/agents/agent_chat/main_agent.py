from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.planners import BuiltInPlanner
from google.adk.tools import FunctionTool
from google.genai import types

from ...core.config import get_settings
from ..exceptions import ValidationError, VisionAgentError
from .instructions import STATIC_INSTRUCTION, get_step_hint
from .state.agent_state import get_agent_state
from .tools import (
    get_current_step,
    get_rule,
    initialize_state,
    list_cameras,
    parse_time_window_tool,
    save_agent_to_db,
    set_field_value,
    resolve_camera,
)
from .tools.knowledge_base import RULES
from .utils.time_context import get_current_time_context, get_utc_iso_z


@lru_cache(maxsize=1)
def load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    try:
        load_dotenv(env_path)
    except Exception:
        pass


def detect_model_provider(model_name: str) -> str:
    return "gemini" if "gemini" in (model_name or "").lower() else "groq"


def ensure_model_api_key(model_name: str) -> tuple[str, str]:
    load_env()
    provider = detect_model_provider(model_name)
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY") or ""
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env")
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GEMINI_API_KEY"] = api_key
        return ("gemini", api_key)
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in .env")
    os.environ["GROQ_API_KEY"] = api_key
    return ("groq", api_key)


def compact_rule(rule: Dict) -> Dict:
    if not rule:
        return {}
    modes = rule.get("execution_modes") or {}
    compact_modes = {
        m: {"required_fields": (c or {}).get("required_fields", []), "zone_required": bool((c or {}).get("zone_required", False))}
        for m, c in modes.items()
    }
    zone_support = rule.get("zone_support") or {}
    out = {
        "rule_id": rule.get("rule_id"),
        "rule_name": rule.get("rule_name"),
        "description": rule.get("description"),
        "required_fields_from_user": rule.get("required_fields_from_user", []),
        "time_window_required": bool((rule.get("time_window") or {}).get("required", False)),
        "zone_required": bool(zone_support.get("required", False)),
        "zone_type": zone_support.get("zone_type", "polygon"),
        "zone_support": zone_support,
        "defaults": rule.get("defaults") or {},
        "execution_modes": compact_modes,
    }
    if rule.get("detectable_classes"):
        out["detectable_classes"] = rule["detectable_classes"]
    if rule.get("detectable_gestures"):
        out["detectable_gestures"] = rule["detectable_gestures"]
    return out


def rules_catalog_json() -> str:
    return json.dumps([compact_rule(r) for r in RULES], separators=(",", ":"), ensure_ascii=False)


def build_dynamic_instruction(context: ReadonlyContext, session_id: str) -> str:
    current_time = get_current_time_context()
    state = get_agent_state(session_id)

    if state.status == "SAVED":
        state_summary = {"status": "SAVED", "saved_agent_id": state.saved_agent_id, "saved_agent_name": state.saved_agent_name}
        kb_context = "{}"
        step_hint = "Do: Confirm the agent is active. Do not ask for more fields. If the user wants another agent or changes, respond to that."
    elif state.rule_id:
        state_summary = {
            "rule_id": state.rule_id,
            "status": state.status,
            "collected_fields": {k: v for k, v in state.fields.items() if v is not None},
            "missing_fields": state.missing_fields,
        }
        active_rule = get_rule(state.rule_id)
        current_step = get_current_step(state, active_rule)
        state_summary["current_step"] = current_step
        kb_context = json.dumps(compact_rule(active_rule), separators=(",", ":"), ensure_ascii=False)
        step_hint = get_step_hint(current_step, active_rule)
    else:
        state_summary = {"status": "UNINITIALIZED"}
        kb_context = rules_catalog_json()
        step_hint = "Do: Infer the user's intent and call initialize_state(rule_id) with the best-matching rule_id from ACTIVE_RULE_CONTEXT_JSON. Do not list all rules."

    state_json = json.dumps(state_summary, ensure_ascii=False)
    now_utc = get_utc_iso_z()
    return (
        f"CURRENT_STATE_JSON:\n{state_json}\n\n"
        f"ACTIVE_RULE_CONTEXT_JSON:\n{kb_context}\n\n"
        f"WHAT_TO_DO_NOW: {step_hint}\n\n"
        f"{current_time}\n"
        f"NOW_UTC_ISO_Z: {now_utc}\n"
    )


def create_tool_wrappers(session_id: str, user_id: Optional[str]):
    def init_wrapper(rule_id: str) -> Dict:
        return initialize_state(rule_id=rule_id, session_id=session_id, user_id=user_id)

    def set_field_wrapper(field_values_json: Any) -> Dict:
        if isinstance(field_values_json, dict):
            payload = json.dumps(field_values_json)
        elif isinstance(field_values_json, str):
            payload = field_values_json
        else:
            raise ValidationError("Expected JSON object or string", user_message="Invalid input format.")
        return set_field_value(field_values_json=payload, session_id=session_id)

    def save_wrapper() -> Dict:
        return save_agent_to_db(session_id=session_id, user_id=user_id)

    def list_cameras_wrapper() -> Dict:
        state = get_agent_state(session_id)
        uid = state.user_id or user_id
        return list_cameras(user_id=uid, session_id=session_id)

    def resolve_camera_wrapper(name_or_id: str) -> Dict:
        state = get_agent_state(session_id)
        uid = state.user_id or user_id
        return resolve_camera(name_or_id=name_or_id, user_id=uid, session_id=session_id)

    def parse_time_window_wrapper(user_time_phrase: str, reference_start_iso: Optional[str] = None) -> Dict:
        return parse_time_window_tool(user_time_phrase=user_time_phrase, reference_start_iso=reference_start_iso, session_id=session_id)

    return [
        FunctionTool(init_wrapper),
        FunctionTool(set_field_wrapper),
        FunctionTool(save_wrapper),
        FunctionTool(list_cameras_wrapper),
        FunctionTool(resolve_camera_wrapper),
        FunctionTool(parse_time_window_wrapper),
    ]


def create_agent_for_session(session_id: str = "default", user_id: Optional[str] = None) -> LlmAgent:
    try:
        settings = get_settings()
        ensure_model_api_key(settings.agent_creation_model)
        state = get_agent_state(session_id)
        if user_id:
            state.user_id = user_id

        tools = create_tool_wrappers(session_id, user_id)

        def instruction_provider(context: ReadonlyContext) -> str:
            try:
                return build_dynamic_instruction(context, session_id)
            except VisionAgentError:
                raise
            except Exception as e:
                raise VisionAgentError(str(e), user_message="Failed to load session state. Please try again.") from e

        planner = BuiltInPlanner(thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=-1))

        return LlmAgent(
            name="main_agent",
            description="Guides users through creating vision analytics agents.",
            static_instruction=STATIC_INSTRUCTION,
            instruction=instruction_provider,
            tools=tools,
            planner=planner,
            model=settings.agent_creation_model,
            generate_content_config=types.GenerateContentConfig(temperature=0.0),
        )
    except VisionAgentError:
        raise
    except Exception as e:
        raise VisionAgentError(str(e), user_message="Failed to initialize agent. Please try again.") from e
