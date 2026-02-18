import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from pytz import UTC, timezone

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.adk.tools import FunctionTool
from google.genai import types

from .tools.camera_selection_tool import (
    list_cameras as list_cameras_impl,
    resolve_camera as resolve_camera_impl,
)
from .tools.initialize_state_tool import initialize_state as initialize_state_impl
from .tools.save_to_db_tool import save_to_db as save_to_db_impl
from .tools.set_field_value_tool import set_field_value as set_field_value_impl


# ============================================================================
# LOGGING & CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

with open(KB_PATH, "r") as f:
    _kb_data = json.load(f)

rules = _kb_data.get("rules", [])
_rules_by_id: Dict[str, dict] = {r.get("rule_id"): r for r in rules if r.get("rule_id")}

static_instruction = """
You are the assistant for a Vision Agent Creation system.

NON-NEGOTIABLE RULES:
- Never reveal internal tooling, internal state, schemas, or implementation details.
- Never expose rule IDs or internal field names to the user.
- Never guess values. If something is missing, ask.
- Ask only for items listed in missing_fields (one at a time).
- If missing_fields is empty, summarize and ask for confirmation.
- If status is SAVED, do not restart; confirm success and wait for a new explicit request.

STYLE:
- Be direct, short, and professional.
- Use Markdown formatting for readability (### sections, lists, **bold**, `code`, fenced code blocks).

CRITICAL UX RULES:
- Do NOT ask the user to choose from a long menu of rules.
- Infer the best rule from the user's intent.
- Only ask a clarification question if intent is ambiguous; if needed, offer at most 2–3 short options (no numbering requirement).
- Do NOT say "current state", "missing fields", "start_time/end_time", or similar internal terms. Ask in plain language like "start time" / "end time" / "camera to monitor".
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _debug_adk_enabled() -> bool:
    """Check if ADK debug mode is enabled."""
    return os.getenv("DEBUG_ADK") == "true"


@lru_cache(maxsize=1)
def _load_env() -> None:
    """Load environment variables once (no stdout prints)."""
    try:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load .env: %s", exc)


def _ensure_groq_api_key() -> str:
    """Ensure GROQ_API_KEY is present; return it."""
    _load_env()
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it in your .env file or environment variables. "
            "Provider-style models require the API key to be set."
        )
    os.environ["GROQ_API_KEY"] = groq_api_key
    return groq_api_key


from .utils.time_context import get_current_time_context



def _compact_rule(rule: dict) -> dict:
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


def _rules_catalog_json() -> str:
    """
    Compact catalog of all rules for initial selection.
    This replaces the full KB dump to reduce tokens drastically.
    """
    catalog = [_compact_rule(r) for r in rules]
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
        active_rule = _rules_by_id.get(agent_state.rule_id)
        kb_context = json.dumps(_compact_rule(active_rule), separators=(",", ":"), ensure_ascii=False)
    else:
        kb_context = _rules_catalog_json()

    # Inject list of cameras for this user when rule is initialized and camera_id is still missing.
    cameras_inject = ""
    if agent_state.rule_id and "camera_id" in (agent_state.missing_fields or []) and not agent_state.fields.get("camera_id"):
        user_id_for_cameras = agent_state.user_id
        if user_id_for_cameras:
            try:
                result = list_cameras_impl(user_id=user_id_for_cameras, session_id=session_id)
                cameras = result.get("cameras") or []
                if cameras:
                    lines = [f"- {c.get('name', '')} (ID: {c.get('id', '')})" for c in cameras[:20]]
                    cameras_inject = (
                        "\n\nCAMERAS_AVAILABLE_FOR_USER (show this list when asking user to choose a camera):\n"
                        + "\n".join(lines)
                        + "\n\nIn your reply, list these cameras and ask which one to use. Use resolve_camera_wrapper when the user picks one.\n"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not inject cameras into prompt: %s", exc)

    return (
        "ROLE: Help the user create a vision analytics agent via a strict state machine.\n"
        "NEVER SAY:\n"
        "- 'current state', 'missing fields', tool names, or internal JSON keys.\n"
        "- Do not echo internal IDs back unless the user explicitly asks.\n"
        "YOU MUST:\n"
        "- Use CURRENT_STATE as the only source of truth.\n"
        "- Ask only for the first item in missing_fields (one question at a time).\n"
        "- Never guess; if missing, ask.\n"
        "- If status=CONFIRMATION and missing_fields is empty: summarize and ask to confirm.\n"
        "- If status=SAVED: confirm success and wait.\n"
        "\n"
        "RULE SELECTION:\n"
        "- If state is UNINITIALIZED, infer the rule from the user's message.\n"
        "- Do NOT list all rules by default. If ambiguous, ask one clarifying question and present max 2–3 options.\n"
        "\n"
        "TOOL CALLING (internal; never mention tools to user):\n"
        "- Allowed tools: initialize_state_wrapper(rule_id), set_field_value_wrapper(field_values_json), save_to_db_wrapper(), list_cameras_wrapper(), resolve_camera_wrapper(name_or_id).\n"
        "- field_values_json MUST be a JSON STRING.\n"
        "- If user provides multiple missing values in one message, set them all in one call.\n"
        "\n"
        "CAMERA SELECTION (CRITICAL):\n"
        "- When camera_id is missing, ALWAYS resolve it using camera selection tools.\n"
        "- NEVER assume or guess a camera_id - always use resolve_camera_wrapper() or list_cameras_wrapper().\n"
        "- When camera_id is in missing_fields and the user did NOT give a camera name/ID in their message:\n"
        "  FIRST call list_cameras_wrapper() in the same turn, then in your reply SHOW the list of cameras (name and ID for each) and ask which one to use. Do NOT reply with only 'Which camera?' without listing them.\n"
        "- Flow:\n"
        "  1. If user mentions a camera name/ID: call resolve_camera_wrapper(name_or_id)\n"
        "  2. If resolve_camera_wrapper returns 'exact_match': use the camera_id from response and set it via set_field_value_wrapper\n"
        "  3. If resolve_camera_wrapper returns 'multiple_matches': ask user which camera they want from the list (NEVER auto-select)\n"
        "  4. If resolve_camera_wrapper returns 'not_found' OR user didn't specify a camera: call list_cameras_wrapper(), then show the list in your reply and ask user to choose\n"
        "  5. After user clarifies, call resolve_camera_wrapper again with their choice\n"
        "- Users can provide camera name (partial match supported), camera ID, or just describe it - handle all cases.\n"
        "- Example user inputs: 'Front Gate', 'loading area', 'CAM-001', 'the camera at the warehouse'\n"
        "\n"
        "INITIALIZATION FLOW (CRITICAL):\n"
        "- If state is UNINITIALIZED and you infer a rule:\n"
        "- Call initialize_state_wrapper(rule_id)\n"
        "- Then IMMEDIATELY re-evaluate the same user message\n"
        "- Extract any values that match required or missing fields\n"
        "- Call set_field_value_wrapper with those values\n"
        "- Only then ask for remaining missing information\n"
        "\n"
        "TIME WINDOW RULES:\n"
        "- Interpret user times in IST; convert to UTC ISO8601 with 'Z'.\n"
        "- If time_window_required is true for the selected rule, BOTH start_time and end_time are required.\n"
        "- NEVER assume end_time.\n"
        "- Accept natural phrases like 'now', 'after 10 min', 'tomorrow 9am' (convert internally).\n"
        "\n"
        "RULES_CONTEXT_JSON:\n"
        f"{kb_context}\n"
        "\n"
        "CURRENT_STATE_JSON:\n"
        f"{json.dumps(state_summary, separators=(',', ':'), ensure_ascii=False)}\n"
        f"{cameras_inject}"
        "\n"
        f"CURRENT_TIME_IST: {current_time_context}\n"
    )


# ============================================================================
# TOOL WRAPPERS
# ============================================================================

def _create_tool_wrappers(current_session_id: str, current_user_id: Optional[str]):
    """Create tool wrappers with session context injected."""

    def initialize_state_wrapper(rule_id: str) -> Dict:
        """Initialize agent state for the selected rule."""
        return initialize_state_impl(rule_id=rule_id, session_id=current_session_id, user_id=current_user_id)

    def set_field_value_wrapper(field_values_json: str) -> Dict:
        """Update one or more fields in the agent state."""
        return set_field_value_impl(field_values_json=field_values_json, session_id=current_session_id)

    def save_to_db_wrapper() -> Dict:
        """Save the confirmed agent configuration to the database."""
        if _debug_adk_enabled():
            logger.debug(
                "[save_to_db_wrapper] Called (session_id=%s user_id=%s)",
                current_session_id,
                current_user_id,
            )
        result = save_to_db_impl(session_id=current_session_id, user_id=current_user_id)
        if _debug_adk_enabled():
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

    if _debug_adk_enabled():
        logger.debug(
            "[create_tool_wrappers] Created tools: %s",
            [getattr(tool, "name", "<unknown>") for tool in tools],
        )
        for tool in tools:
            try:
                decl = tool._get_declaration()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[create_tool_wrappers] Error getting declaration: %s", exc)
                continue
            if decl and getattr(tool, "name", None) != getattr(decl, "name", None):
                logger.warning(
                    "[create_tool_wrappers] Tool name mismatch: tool.name=%s decl.name=%s",
                    getattr(tool, "name", None),
                    getattr(decl, "name", None),
                )

    if _debug_adk_enabled():
        logger.debug("[tools] Created %d wrapped tools", len(tools))
        for tool in tools:
            try:
                decl = tool._get_declaration()
                desc = getattr(decl, "description", None) if decl else None
                logger.debug(
                    "[tools] %s (%s) decl.name=%s desc=%s",
                    getattr(tool, "name", "<unknown>"),
                    type(tool).__name__,
                    getattr(decl, "name", None) if decl else None,
                    (desc[:50] + "...") if isinstance(desc, str) and len(desc) > 50 else desc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[tools] Error introspecting tool: %s", exc)

    return tools


# ============================================================================
# CALLBACKS
# ============================================================================

async def _log_thinking_request(
    *, callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Log thinking config before model request (debug only)."""
    if not _debug_adk_enabled():
        return None

    tc = None
    try:
        tc = llm_request.config.thinking_config if llm_request.config else None
    except Exception:  # noqa: BLE001
        tc = None

    logger.debug(
        "[adk] planning request agent=%s include_thoughts=%s budget=%s level=%s tools=%s",
        getattr(callback_context, "agent_name", None),
        getattr(tc, "include_thoughts", None) if tc else None,
        getattr(tc, "thinking_budget", None) if tc else None,
        getattr(tc, "thinking_level", None) if tc else None,
        list(getattr(llm_request, "tools_dict", {}) or {}),
    )
    return None


async def _log_thinking_response(
    *, callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Log response metadata (debug only; do not print full content)."""
    if not _debug_adk_enabled():
        return None

    parts = []
    try:
        parts = list(getattr(getattr(llm_response, "content", None), "parts", []) or [])
    except Exception:  # noqa: BLE001
        parts = []

    thought_count = 0
    tool_call_count = 0
    text_part_count = 0
    for part in parts:
        if getattr(part, "thought", False):
            thought_count += 1
        if getattr(part, "function_call", None):
            tool_call_count += 1
        if getattr(part, "text", None):
            text_part_count += 1

    usage = getattr(llm_response, "usage_metadata", None)
    logger.debug(
        "[adk] planning response agent=%s thoughts=%d text_parts=%d tool_calls=%d err=%s in_tok=%s out_tok=%s",
        getattr(callback_context, "agent_name", None),
        thought_count,
        text_part_count,
        tool_call_count,
        getattr(llm_response, "error_code", None),
        getattr(usage, "prompt_token_count", None) if usage else None,
        getattr(usage, "candidates_token_count", None) if usage else None,
    )
    return None


# ============================================================================
# MAIN AGENT CREATION
# ============================================================================

def create_agent_for_session(session_id: str = "default", user_id: Optional[str] = None) -> LlmAgent:
    """Create and configure the main agent for a session."""
    _ensure_groq_api_key()

    from .session_state.agent_state import get_agent_state

    agent_state = get_agent_state(session_id)
    if user_id:
        agent_state.user_id = user_id

    wrapped_tools = _create_tool_wrappers(session_id, user_id)

    def create_dynamic_instruction_provider(current_session_id: str):
        """Create instruction provider that captures session_id in closure."""
        def instruction_provider(context: ReadonlyContext) -> str:
            return build_instruction_dynamic_with_session(context, current_session_id)
        return instruction_provider

    dynamic_instruction = create_dynamic_instruction_provider(session_id)

    planner = BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,
        )
    )

    if _debug_adk_enabled():
        logger.debug(
            "[planner] include_thoughts=%s budget=%s level=%s",
            getattr(planner.thinking_config, "include_thoughts", None),
            getattr(planner.thinking_config, "thinking_budget", None),
            getattr(planner.thinking_config, "thinking_level", None),
        )

    main_agent = LlmAgent(
        name="main_agent",
        description="A main agent that guides users through creating vision analytics agents.",
        static_instruction=static_instruction,
        instruction=dynamic_instruction,
        tools=wrapped_tools,
        planner=planner,
        model="groq/qwen/qwen3-32b",
        before_model_callback=[_log_thinking_request],
        after_model_callback=[_log_thinking_response],
    )

    if _debug_adk_enabled():
        logger.debug("[agent] created tools=%d model=%s", len(wrapped_tools), main_agent.model)

    return main_agent


# ============================================================================
# MODULE-LEVEL VARIABLES
# ============================================================================

default_agent: Optional[LlmAgent] = None
