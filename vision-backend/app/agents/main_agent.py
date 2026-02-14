"""
Main agent for Vision Agent Creation flow.

Builds an LLM agent with dynamic instructions, session state, and tools
(initialize_state, set_field_value, save_to_db, list_cameras, resolve_camera).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.adk.tools import FunctionTool
from google.genai import types

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ..core.config import get_settings
from .exceptions import VisionAgentError
from .tools.camera_selection_tool import (
    list_cameras as list_cameras_impl,
    resolve_camera as resolve_camera_impl,
)
from .tools.initialize_state_tool import initialize_state as initialize_state_impl
from .tools.save_to_db_tool import save_to_db as save_to_db_impl
from .tools.set_field_value_tool import set_field_value as set_field_value_impl
from .utils.time_context import get_current_time_context, get_utc_iso_z

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Knowledge base and rules
# -----------------------------------------------------------------------------

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    kb_data = json.load(f)

rules = kb_data.get("rules", [])
rules_by_id: Dict[str, dict] = {r.get("rule_id"): r for r in rules if r.get("rule_id")}

static_instruction = """
You are the assistant for a Vision Agent Creation system.

NON-NEGOTIABLE RULES:
- Never reveal internal tooling, internal state, schemas, or implementation details.
- Never expose rule IDs or internal field names to the user.
- Never guess values. If something is missing, ask.
- Ask only for items listed in missing_fields (one at a time).
- If missing_fields is empty, summarize and ask for confirmation.
- If status is SAVED, do not restart; confirm success and wait for a new explicit request.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    GLOBAL FLOW (RULE → CAMERA → REST)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    THE FLOW IS STRICT AND MUST NEVER CHANGE:

    1. Infer user intent
    2. Initialize the correct rule
    3. THEN resolve camera
    4. THEN collect remaining required fields
    5. Confirm
    6. Save

    Camera selection MUST NEVER happen before rule initialization.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CRITICAL: missing_fields IS THE ONLY SOURCE OF TRUTH
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ABSOLUTE RULES FOR FIELD COLLECTION:
    - ONLY ask for fields that are EXPLICITLY listed in missing_fields
    - If missing_fields is empty → ask for confirmation, DO NOT ask for any field
    - If a field is NOT in missing_fields → it is either collected OR not required
    - NEVER infer, guess, or assume what fields are needed beyond missing_fields
    - NEVER ask "would you like to add [field]?" if that field is not in missing_fields
    
    ZONE-SPECIFIC RULES (CRITICAL):
    - zone IN missing_fields → Ask for zone
    - zone NOT IN missing_fields → NEVER mention zone, NEVER ask about zone, NEVER suggest zone
    - If user mentions zone but zone not in missing_fields → politely ignore and proceed with actual missing_fields
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CONVERSATION BEHAVIOR (VERY STRICT)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Ask ONLY ONE question per turn.
    - Ask ONLY for the FIRST field in missing_fields (nothing else).
    - NEVER repeat a question once answered.
    - NEVER ask about fields NOT in missing_fields.
    - NEVER ask about fields already in collected_fields.
    - NEVER restart or reset unless explicitly requested.

    If status is SAVED:
    - Confirm success ONCE
    - Wait for a new explicit request

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RULE SELECTION (INTENT-FIRST, HARD RULE)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - If state is UNINITIALIZED:
    - Infer the best rule silently from user intent
    - DO NOT list rules
    - DO NOT ask about camera yet
    - If intent is ambiguous:
    - Ask ONE clarification question
    - Offer MAXIMUM 2-3 short options
    - Once a rule is inferred:
    → initialize state IMMEDIATELY
    → re-process the SAME user message to extract values

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SOURCE: CAMERA (RTSP) OR VIDEO FILE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    After rule init the first required "source" is either camera_id (live RTSP) OR video_path (uploaded file).

    VIDEO FILE: If source_type is video_file or video_path is in collected_fields:
    - Do NOT ask for camera_id. Do NOT list cameras.
    - Do NOT ask for start_time or end_time.
    - Use the provided video and proceed to the next missing field.

    CAMERA / RTSP (no video_path): If camera is missing:
    RULES:
    - If camera is missing:
    1. ALWAYS call list_cameras_wrapper FIRST
    2. ALWAYS show available cameras as suggestions
    3. NEVER ask the user to type camera id or exact name

    - If the user says:
    - a camera name
    - “this one”
    - “that one”
    - “yes”
    → IMMEDIATELY resolve and set the camera in the SAME turn

    - If multiple cameras match:
    → ask the user to choose ONE (only from the shown list)

    - If no match:
    → re-list cameras and ask again

    ONCE source is set (camera_id OR video_path):
    - NEVER ask about camera again (for RTSP flow)
    - NEVER re-list cameras again
    - MOVE TO NEXT MISSING FIELD IMMEDIATELY

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    GENERIC CLASS / OBJECT HANDLING (NO CONFIRMATION EVER)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Users may refer to objects using natural language, vague terms, or synonyms.

    RULES:
    - Map user terms to the closest canonical class supported by the active rule.
    - Examples (NOT LIMITED TO):
    - someone / somebody / anyone / human → person
    - vehicle / car / bike / truck → vehicle or car (based on rule support)
    - animal / dog / cat → animal or specific class if supported

    ABSOLUTE RULES:
    - NEVER ask for confirmation of inferred class
    - NEVER ask “Do you mean X?”
    - Treat inferred class as FINAL and CONFIRMED
    - Proceed immediately to the next missing requirement

    If multiple canonical classes are possible:
    - Choose the MOST GENERIC class supported by the rule
    - Do NOT ask the user to choose

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ZONE HANDLING (STRICT)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Ask about zone ONLY AFTER source is set (camera_id or video_path)
    - If zone is REQUIRED:
    → ask once and wait
    - If zone is OPTIONAL:
    - Ask once
    - If user says “no”, “skip”, or ignores → set zone = null
    - NEVER re-ask zone once resolved

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TIME WINDOW HANDLING
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - If source_type is video_file OR video_path is present:
    - NEVER ask start_time or end_time
    - Ignore time window collection entirely
    - Interpret all times in the configured LOCAL_TIMEZONE (see NOW_UTC_ISO_Z and time context each turn)
    - start_time and end_time MUST be full ISO 8601 UTC format: YYYY-MM-DDTHH:MM:SSZ
    - FORBIDDEN: time-only strings like "05:00", "05:00+05:30" - these will cause save failure
    - Build full ISO from NOW_UTC_ISO_Z and user intent (convert local time to UTC for storage)
    - If time window is required:
    - BOTH start time AND end time are mandatory
    - NEVER assume missing values
    - Ask for missing time fields ONE AT A TIME

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TOOL USAGE (INTERNAL ONLY)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - When user provides information:
    → call the tool in the SAME turn
    - If multiple values are provided:
    → set all in ONE call
    - NEVER delay tool execution
    - NEVER ask a question if a tool can be called

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CONFIRMATION & SAVE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    When all required information is collected:
    - Summarize clearly in human language
    - Ask ONE confirmation question

    If user confirms:
    - Save immediately
    - Confirm success
    - Stop

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    STYLE RULES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Short
    - Direct
    - Professional

    CRITICAL UX RULES:
    - Do NOT ask the user to choose from a long menu of rules.
    - Infer the best rule from the user's intent.
    - Only ask a clarification question if intent is ambiguous; if needed, offer at most 2–3 short options (no numbering requirement).
    - Do NOT say "current state", "missing fields", "start_time/end_time", or similar internal terms. Ask in plain language like "start time" / "end time" / "camera to monitor".
    """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def debug_adk_enabled() -> bool:
    """Return True if ADK debug mode is enabled (DEBUG_ADK=true)."""
    return os.getenv("DEBUG_ADK") == "true"


@lru_cache(maxsize=1)
def load_env() -> None:
    """Load environment variables from .env once."""
    try:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load .env: %s", exc)


def ensure_groq_api_key() -> str:
    """Ensure GROQ_API_KEY is set; return it. Raises ValueError if missing."""
    load_env()
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it in your .env file or environment variables. "
            "Provider-style models require the API key to be set."
        )
    os.environ["GROQ_API_KEY"] = groq_api_key
    return groq_api_key


def compact_rule(rule: dict) -> dict:
    """
    Create a compact rule representation for prompt context (token-efficient).
    Do NOT include long examples or huge class lists.
    """
    if not rule:
        return {}

    execution_modes = rule.get("execution_modes") or {}
    compact_modes = {
        mode: {
            "required_fields": (cfg or {}).get("required_fields", []),
            "zone_required": bool((cfg or {}).get("zone_required", False)),
        }
        for mode, cfg in execution_modes.items()
    }

    return {
        "rule_id": rule.get("rule_id"),
        "rule_name": rule.get("rule_name"),
        "description": rule.get("description"),
        "required_fields_from_user": rule.get("required_fields_from_user", []),
        "time_window_required": bool((rule.get("time_window") or {}).get("required", False)),
        "zone_required": bool((rule.get("zone_support") or {}).get("required", False)),
        "defaults": (rule.get("defaults") or {}),
        "execution_modes": compact_modes,
    }


def rules_catalog_json() -> str:
    """Return compact JSON catalog of all rules for initial selection."""
    catalog = [compact_rule(r) for r in rules]
    return json.dumps(catalog, separators=(",", ":"), ensure_ascii=False)


# ============================================================================
# INSTRUCTION BUILDERS
# ============================================================================

def build_instruction_dynamic_with_session(context: ReadonlyContext, session_id: str) -> str:
    """
    Dynamic instruction provider that reads state each time it's called.
    This ensures the instruction always has the latest state, even after tool executions.
    Uses session_id from closure to read from our internal state store.
    """
    from .session_state.agent_state import get_agent_state

    current_time_context = get_current_time_context()
    agent_state = get_agent_state(session_id)

    if agent_state.status == "SAVED":
        state_summary = {
            "status": "SAVED",
            "saved_agent_id": agent_state.saved_agent_id,
            "saved_agent_name": agent_state.saved_agent_name,
        }
    elif agent_state.rule_id:
        collected_fields = {k: v for k, v in agent_state.fields.items() if v is not None}
        state_summary = {
            "rule_id": agent_state.rule_id,
            "status": agent_state.status,
            "collected_fields": collected_fields,
            "missing_fields": agent_state.missing_fields,
        }
    else:
        state_summary = {"status": "UNINITIALIZED"}

    if agent_state.rule_id:
        active_rule = rules_by_id.get(agent_state.rule_id)
        kb_context = json.dumps(compact_rule(active_rule), separators=(",", ":"), ensure_ascii=False)
    else:
        kb_context = rules_catalog_json()

    state_json = json.dumps(state_summary, ensure_ascii=False)
    active_rule_block = f"ACTIVE_RULE_CONTEXT_JSON:\n{kb_context}\n"
    now_utc_iso = get_utc_iso_z()
    return (
        f"CURRENT_STATE_JSON:\n{state_json}\n"
        f"{active_rule_block}"
        f"{current_time_context}\n"
        f"NOW_UTC_ISO_Z: {now_utc_iso}\n"
    )


# ============================================================================
# TOOL WRAPPERS
# ============================================================================

def create_tool_wrappers(current_session_id: str, current_user_id: Optional[str]):
    """Create tool wrappers with session context injected."""

    def initialize_state_wrapper(rule_id: str) -> Dict:
        """Initialize agent state for the selected rule."""
        return initialize_state_impl(rule_id=rule_id, session_id=current_session_id, user_id=current_user_id)

    def set_field_value_wrapper(field_values_json: Any) -> Dict:
        """Update one or more fields in the agent state."""
        # Groq tool calls frequently send structured JSON objects for this tool.
        # Normalize to the JSON string expected by set_field_value_impl.
        if isinstance(field_values_json, dict):
            payload = json.dumps(field_values_json)
        elif isinstance(field_values_json, str):
            # Defensive fallback if provider sends a raw JSON string.
            payload = field_values_json
        else:
            payload = json.dumps({})
        return set_field_value_impl(field_values_json=payload, session_id=current_session_id)

    def save_to_db_wrapper() -> Dict:
        """Save the confirmed agent configuration to the database."""
        if debug_adk_enabled():
            logger.debug(
                "[save_to_db_wrapper] Called (session_id=%s user_id=%s)",
                current_session_id,
                current_user_id,
            )
        result = save_to_db_impl(session_id=current_session_id, user_id=current_user_id)
        if debug_adk_enabled():
            logger.debug("[save_to_db_wrapper] Result: %s", result)
        return result

    def list_cameras_wrapper() -> Dict:
        """List all cameras owned by the current user."""
        from .session_state.agent_state import get_agent_state

        agent_state = get_agent_state(current_session_id)
        user_id_for_camera = agent_state.user_id or current_user_id

        if not user_id_for_camera:
            return {
                "error": "user_id is required. Agent state must have user_id set.",
                "cameras": []
            }

        return list_cameras_impl(user_id=user_id_for_camera, session_id=current_session_id)

    def resolve_camera_wrapper(name_or_id: str) -> Dict:
        """Resolve a camera by name (partial match) or ID."""
        from .session_state.agent_state import get_agent_state

        agent_state = get_agent_state(current_session_id)
        user_id_for_camera = agent_state.user_id or current_user_id

        if not user_id_for_camera:
            return {
                "status": "not_found",
                "error": "user_id is required. Agent state must have user_id set."
            }

        return resolve_camera_impl(name_or_id=name_or_id, user_id=user_id_for_camera, session_id=current_session_id)

    initialize_state_wrapper.__name__ = "initialize_state_wrapper"
    set_field_value_wrapper.__name__ = "set_field_value_wrapper"
    save_to_db_wrapper.__name__ = "save_to_db_wrapper"
    list_cameras_wrapper.__name__ = "list_cameras_wrapper"
    resolve_camera_wrapper.__name__ = "resolve_camera_wrapper"

    tools = [
        FunctionTool(initialize_state_wrapper),
        FunctionTool(set_field_value_wrapper),
        FunctionTool(save_to_db_wrapper),
        FunctionTool(list_cameras_wrapper),
        FunctionTool(resolve_camera_wrapper),
    ]
    return tools


# ============================================================================
# MAIN AGENT CREATION
# ============================================================================

def create_agent_for_session(session_id: str = "default", user_id: Optional[str] = None) -> LlmAgent:
    """Create and configure the main agent for a session."""
    try:
        ensure_groq_api_key()

        from .session_state.agent_state import get_agent_state

        agent_state = get_agent_state(session_id)
        if user_id:
            agent_state.user_id = user_id

        wrapped_tools = create_tool_wrappers(session_id, user_id)

        def create_dynamic_instruction_provider(current_session_id: str):
            """Create instruction provider that captures session_id in closure."""

            def instruction_provider(context: ReadonlyContext) -> str:
                """Per-turn dynamic instruction: minimal state + time."""
                try:
                    return build_instruction_dynamic_with_session(context, current_session_id)
                except Exception as e:
                    logger.error(
                        "Failed to build dynamic instruction for session %s: %s",
                        current_session_id,
                        e,
                    )
                    # Return minimal valid JSON instruction to allow agent to continue.
                    return 'CURRENT_STATE_JSON:\n{"status":"COLLECTING"}\n'

            return instruction_provider

        dynamic_instruction = create_dynamic_instruction_provider(session_id)
        planner = BuiltInPlanner(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            )
        )

        settings = get_settings()
        agent = LlmAgent(
            name="main_agent",
            description="A main agent that guides users through creating vision analytics agents.",
            static_instruction=static_instruction,
            instruction=dynamic_instruction,
            tools=wrapped_tools,
            planner=planner,
            model=settings.agent_creation_model,
        )

        logger.info("Agent created successfully for session %s", session_id)
        return agent

    except VisionAgentError:
        raise
    except Exception as e:
        logger.exception("Failed to create agent for session %s: %s", session_id, e)
        raise VisionAgentError(
            f"Failed to create agent: {str(e)}",
            user_message="Failed to initialize agent. Please try again.",
        )

