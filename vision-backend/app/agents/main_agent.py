from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
import logging
import json
import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from typing import Optional, Dict
from datetime import datetime
from pytz import UTC, timezone
from google.adk.planners import BuiltInPlanner
from google.genai import types


from .tools.initialize_state_tool import initialize_state as initialize_state_impl
from .tools.set_field_value_tool import set_field_value as set_field_value_impl
from .tools.save_to_db_tool import save_to_db as save_to_db_impl




logger = logging.getLogger(__name__)


def _debug_adk_enabled() -> bool:
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


# Load knowledge base
KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

with open(KB_PATH, "r") as f:
    _kb_data = json.load(f)

rules = _kb_data.get("rules", [])


def get_current_time_context() -> str:
    """
    Get current time context for the LLM prompt.

    IMPORTANT:
    - Provide both UTC + IST in machine-readable formats so the model can copy exact values.
    - This reduces "rounded" times like 15:30 when the user says "start now".
    """
    IST = timezone("Asia/Kolkata")  # UTC+5:30

    # Use an explicit UTC anchor first, then convert to IST.
    now_utc = datetime.now(UTC)
    now_ist = now_utc.astimezone(IST)

    # Machine-readable anchors
    now_utc_iso_z = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_ist_iso = now_ist.isoformat()
    now_epoch_sec = int(now_utc.timestamp())

    # Human-friendly
    now_ist_human = now_ist.strftime("%A, %B %d, %Y at %H:%M:%S IST")
    now_utc_human = now_utc.strftime("%A, %B %d, %Y at %H:%M:%S UTC")

    return (
        "TIME_CONTEXT (copy these values exactly; do not round):\n"
        f"- NOW_UTC_ISO_Z: {now_utc_iso_z}\n"
        f"- NOW_UTC_EPOCH_SEC: {now_epoch_sec}\n"
        f"- NOW_IST_ISO: {now_ist_iso}\n"
        f"- NOW_IST_HUMAN: {now_ist_human}\n"
        f"- NOW_UTC_HUMAN: {now_utc_human}\n"
        "RULE: If user says 'start now', use NOW_UTC_ISO_Z exactly. If user says 'end after X', compute end_time from NOW_UTC_EPOCH_SEC (+X).\n"
    )


# Static instruction - stable system behavior (kept short to reduce tokens)
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


# Pre-index rules for smaller dynamic prompts (active rule only when possible)
_rules_by_id: Dict[str, dict] = {r.get("rule_id"): r for r in rules if r.get("rule_id")}


def _rules_catalog_json() -> str:
    """
    Compact catalog of all rules for initial selection.
    This replaces the full KB dump to reduce tokens drastically.
    """
    catalog = [_compact_rule(r) for r in rules]
    # Minified JSON reduces tokens substantially vs pretty-print.
    return json.dumps(catalog, separators=(",", ":"), ensure_ascii=False)


def build_instruction_dynamic_with_session(context: ReadonlyContext, session_id: str) -> str:
    """
    Dynamic instruction provider that reads state each time it's called.
    This ensures the instruction always has the latest state, even after tool executions.
    Uses session_id from closure to read from our internal state store.
    """
    from .session_state.agent_state import get_agent_state
    
    current_time_context = get_current_time_context()
    
    # Read current agent state to inject into instruction
    # This is called EVERY time the instruction is needed, so state is always fresh
    agent_state = get_agent_state(session_id)
    
    # Build state summary (keep it compact)
    # CRITICAL: Check SAVED status FIRST - it's a terminal state
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

    # Knowledge base context: send only what is needed to reduce tokens.
    if agent_state.rule_id:
        active_rule = _rules_by_id.get(agent_state.rule_id)
        kb_context = json.dumps(_compact_rule(active_rule), separators=(",", ":"), ensure_ascii=False)
    else:
        kb_context = _rules_catalog_json()

    # Compact, strict dynamic instruction (token-efficient).
    # Keep tool calling rules and time-window rules, but remove long examples.
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
        "- Allowed tools: initialize_state_wrapper(rule_id), set_field_value_wrapper(field_values_json), save_to_db_wrapper().\n"
        "- field_values_json MUST be a JSON STRING.\n"
        "- If user provides multiple missing values in one message, set them all in one call.\n"
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
        "\n"
        f"CURRENT_TIME_IST: {current_time_context}\n"
    )


def create_agent_for_session(session_id: str = "default", user_id: Optional[str] = None) -> LlmAgent:
    # Ensure GROQ_API_KEY is set before creating the agent
    _ensure_groq_api_key()

    # Wrap all tools with FunctionTool for proper ADK registration
    # Create wrapper functions that inject session_id from a closure
    # This allows us to hide session_id from the LLM while still using the correct session
    
    def create_tool_wrappers(current_session_id: str, current_user_id: Optional[str]):
        """Create tool wrappers with session context injected"""
        
        def initialize_state_wrapper(rule_id: str) -> Dict:
            """
            Initialize agent state for the selected rule.
            
            This function MUST be called first before setting any field values.
            It sets up the agent state based on the rule type from the knowledge base.
            
            Args:
                rule_id (str): The rule ID to initialize. Must be a valid rule ID from the knowledge base.
                    Examples: "object_enter_zone", "class_presence", "gesture_detected", "loitering_detected"
            
            Returns:
                Dict: A dictionary with status and message indicating if initialization was successful.
            
            Example:
                initialize_state_wrapper(rule_id="object_enter_zone")
            """
            return initialize_state_impl(rule_id=rule_id, session_id=current_session_id)
        
        def set_field_value_wrapper(field_values_json: str) -> Dict:
            """
            Update one or more fields in the agent state.
            
            CRITICAL: The parameter MUST be a JSON STRING, not a dictionary object.
            You MUST convert any dictionary/object to a JSON string before passing it.
            
            Args:
                field_values_json (str): A JSON STRING containing field name -> value mappings.
                    This parameter is a STRING, not a dictionary. You must serialize dictionaries to JSON strings.
                    The JSON string should contain key-value pairs where keys are field names and values are the field values.
            
            Returns:
                Dict: A dictionary with updated_fields, status, and message indicating the result.
            
            Examples:
                # Single field:
                set_field_value_wrapper(field_values_json='{"camera_id": "CAM-001"}')
                
                # Multiple fields:
                set_field_value_wrapper(field_values_json='{"camera_id": "CAM-001", "class": "person", "start_time": "2025-12-27T14:00:00Z"}')
                
                # Zone field (polygon coordinates):
                set_field_value_wrapper(field_values_json='{"zone": {"type": "polygon", "coordinates": [[100, 200], [300, 400], [500, 600]]}}')
            
            IMPORTANT:
                - Always pass a JSON STRING, never a raw dictionary/object
                - Use json.dumps() or equivalent to convert dictionaries to JSON strings
                - Field names must match exactly: camera_id, class, gesture, start_time, end_time, zone, etc.
            """
            return set_field_value_impl(field_values_json=field_values_json, session_id=current_session_id)
        
        def save_to_db_wrapper() -> Dict:
            """
            Save the confirmed agent configuration to the database.
            
            This function persists the agent configuration after all required fields have been collected
            and the user has confirmed the configuration. It should ONLY be called when:
            - Agent state status is CONFIRMATION
            - All required fields are collected (missing_fields is empty)
            - User has explicitly confirmed (e.g., "yes", "proceed", "save", "confirm")
            
            Args:
                None: This function takes no parameters. Do not pass any arguments.
            
            Returns:
                Dict: A dictionary with status ("SAVED" or error), saved (bool), agent_id, and agent_name.
                    On success: {"status": "SAVED", "saved": True, "agent_id": "...", "agent_name": "..."}
                    On error: {"error": "...", "status": "...", "saved": False, "message": "..."}
            
            Example:
                save_to_db_wrapper()
                # Do NOT call with arguments: save_to_db_wrapper() is correct
                # Do NOT call like: save_to_db_wrapper({}) or save_to_db_wrapper(None)
            
            IMPORTANT:
                - Call this function with NO parameters: save_to_db_wrapper()
                - Only call after user confirms the configuration
                - Do not call if missing_fields is not empty
                - Do not call if status is not CONFIRMATION
            """
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
        
        # Explicitly set function names to ensure they match what Groq expects
        # This prevents name mangling issues with LiteLLM/Groq compatibility
        initialize_state_wrapper.__name__ = "initialize_state_wrapper"
        set_field_value_wrapper.__name__ = "set_field_value_wrapper"
        save_to_db_wrapper.__name__ = "save_to_db_wrapper"
        
        # Create FunctionTool instances with explicitly named functions
        tools = [
            FunctionTool(initialize_state_wrapper),
            FunctionTool(set_field_value_wrapper),
            FunctionTool(save_to_db_wrapper),
        ]
        
        if _debug_adk_enabled():
            # Verify tool names match expected names (debug only)
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
    
    # For now, use "default" session - this will be overridden per-request
    # The actual session_id will be injected when the agent is created per request
    wrapped_tools = create_tool_wrappers(session_id, user_id)
    
    # Use dynamic instruction provider that reads state each time
    # This ensures state is always fresh, even after tool executions within the same run
    def create_dynamic_instruction_provider(current_session_id: str):
        """Create instruction provider that captures session_id in closure"""
        def instruction_provider(context: ReadonlyContext) -> str:
            # Store our internal session_id in ADK session state for future lookups
            if "internal_session_id" not in context.state:
                # This is a one-time setup - ADK will persist it in session state
                pass  # We'll use the closure value instead
            
            # Use the session_id from closure (our internal session management)
            # This ensures we're reading from our own state store
            return build_instruction_dynamic_with_session(context, current_session_id)
        return instruction_provider
    
    # Create the dynamic instruction provider
    dynamic_instruction = create_dynamic_instruction_provider(session_id)

    # Create callbacks to log thinking/planning
    async def log_thinking_request(
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
    
    async def log_thinking_response(
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
        return None  # Return None to keep original response unchanged

    # Create planner with thinking enabled
    planner = BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,  # Enable thinking output
            thinking_budget=-1,  # AUTOMATIC budget (or set specific token count)
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
        static_instruction=static_instruction,  # Identity + hard UX rules (cached)
        instruction=dynamic_instruction,  # Dynamic workflow logic - rebuilt each time
        tools=wrapped_tools,
        planner=planner,
        model="groq/qwen/qwen3-32b",
        before_model_callback=[log_thinking_request],  # Add callback to log planning request
        after_model_callback=[log_thinking_response],  # Add callback to log planning response
    )
    
    if _debug_adk_enabled():
        logger.debug("[agent] created tools=%d model=%s", len(wrapped_tools), main_agent.model)

    return main_agent

# Backward-compat symbol (avoid import-time side effects)
default_agent: Optional[LlmAgent] = None


