from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Union, Awaitable
from datetime import datetime
from pytz import UTC, timezone
from google.adk.planners import BuiltInPlanner
from google.genai import types


from .tools.initialize_state_tool import initialize_state as initialize_state_impl
from .tools.set_field_value_tool import set_field_value as set_field_value_impl
from .tools.save_to_db_tool import save_to_db as save_to_db_impl




# Load knowledge base
KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "vision_rule_knowledge_base.json"

# Load environment variables and ensure GROQ_API_KEY is set
# Use the same path pattern as main.py to find .env file
try:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Set environment variable for LiteLlm to use Groq
    if GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        print(f"GROQ_API_KEY loaded: {'*' * 10}{GROQ_API_KEY[-4:] if len(GROQ_API_KEY) > 4 else '****'}")
    else:
        print("Warning: GROQ_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error loading environment variables: {e}")

with open(KB_PATH, "r") as f:
    _kb_data = json.load(f)

rules = _kb_data.get("rules", [])


def get_current_time_context() -> str:
    """Get current time context in human-readable format for LLM prompt"""
    # Use IST (Indian Standard Time) timezone
    IST = timezone('Asia/Kolkata')  # UTC+5:30
    now = datetime.now(IST)
    # Format: "Monday, November 18, 2025 at 14:30 IST"
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%H:%M")
    return f"{day_name}, {date_str} at {time_str} IST"


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
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        # Try loading from .env again with explicit path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
    
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set it in your .env file or environment variables. "
            "Provider-style models (groq/llama-3.3-70b-versatile) require the API key to be set."
        )

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
            print(f"[save_to_db_wrapper] Called with session_id={current_session_id}, user_id={current_user_id}")
            result = save_to_db_impl(session_id=current_session_id, user_id=current_user_id)
            print(f"[save_to_db_wrapper] Result: {result}")
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
        
        # Verify tool names match expected names
        print(f"[create_tool_wrappers] Created tools with names: {[tool.name for tool in tools]}")
        for tool in tools:
            decl = tool._get_declaration()
            if decl:
                print(f"[create_tool_wrappers] Tool '{tool.name}' declaration name: '{decl.name}'")
                if tool.name != decl.name:
                    print(f"[create_tool_wrappers] WARNING: Tool name mismatch! Tool.name='{tool.name}' but decl.name='{decl.name}'")
        
        # Debug: Verify tools are properly wrapped (only if DEBUG_ADK env var is set)
        if os.getenv("DEBUG_ADK") == "true":
            print(f"[DEBUG] Created {len(tools)} wrapped tools")
            for tool in tools:
                print(f"[DEBUG] Tool: {tool.name}, Type: {type(tool).__name__}")
                try:
                    decl = tool._get_declaration()
                    if decl:
                        print(f"[DEBUG]   Declaration: name={decl.name}, description={decl.description[:50] if decl.description else 'None'}...")
                except Exception as e:
                    print(f"[DEBUG]   Error getting declaration: {e}")
        
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
        """Log thinking config before model request."""
        print("\n" + "="*80)
        print(f"🧠 PLANNING REQUEST - Agent: {callback_context.agent_name}")
        print("="*80)
        
        # Log thinking config
        if llm_request.config and llm_request.config.thinking_config:
            tc = llm_request.config.thinking_config
            print(f"Thinking Config:")
            print(f"  - include_thoughts: {tc.include_thoughts}")
            print(f"  - thinking_budget: {tc.thinking_budget}")
            print(f"  - thinking_level: {tc.thinking_level}")
        else:
            print("Thinking Config: Not set")
        
        # Log available tools
        if llm_request.tools_dict:
            print(f"Available Tools: {list(llm_request.tools_dict.keys())}")
        
        # Log instruction preview (first 500 chars)
        if llm_request.contents:
            instruction_str = str(llm_request.contents)
            instruction_preview = instruction_str
            print(f"Instruction Preview: {instruction_preview}")
        
        print("="*80 + "\n")
        return None
    
    async def log_thinking_response(
        *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        """Log thinking content from model response."""
        print("\n" + "="*80)
        print(f"💭 PLANNING RESPONSE - Agent: {callback_context.agent_name}")
        print("="*80)
        
        if llm_response.content and llm_response.content.parts:
            thinking_parts = []
            regular_parts = []
            
            for part in llm_response.content.parts:
                # Check if this is a thinking part
                if hasattr(part, 'thought') and part.thought:
                    thinking_parts.append(part)
                else:
                    regular_parts.append(part)
            
            # Print thinking parts
            if thinking_parts:
                print("🧠 THINKING/PLANNING CONTENT:")
                for i, part in enumerate(thinking_parts, 1):
                    if hasattr(part, 'text') and part.text:
                        print(f"\n[Thought {i}]")
                        print("-" * 60)
                        print(part.text)
                        print("-" * 60)
            else:
                print("🧠 THINKING: No thinking content found in response")
            
            # Print regular response preview
            if regular_parts:
                print("\n📝 REGULAR RESPONSE:")
                for i, part in enumerate(regular_parts, 1):
                    if hasattr(part, 'text') and part.text:
                        preview = part.text
            
                        print(f"[Response {i}] {preview}")
                    elif hasattr(part, 'function_call') and part.function_call:
                        func_call = part.function_call
                        func_name = getattr(func_call, 'name', 'unknown')
                        args_preview = str(getattr(func_call, 'args', {}))[:200]
                        print(f"[Function Call {i}] {func_name}({args_preview}...)")
        else:
            print("No content in response")
        
        # Log error if present
        if llm_response.error_code:
            print(f"\n❌ ERROR:")
            print(f"  - Error Code: {llm_response.error_code}")
            print(f"  - Error Message: {llm_response.error_message}")
        
        # Log usage metadata if available
        if llm_response.usage_metadata:
            print(f"\n📊 Token Usage:")
            print(f"  - Input: {llm_response.usage_metadata.prompt_token_count}")
            print(f"  - Output: {llm_response.usage_metadata.candidates_token_count}")
        
        print("="*80 + "\n")
        return None  # Return None to keep original response unchanged

    # Create planner with thinking enabled
    planner = BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,  # Enable thinking output
            thinking_budget=-1,  # AUTOMATIC budget (or set specific token count)
        )
    )
    
    # Print planner config (only if DEBUG_ADK is enabled)
    if os.getenv("DEBUG_ADK") == "true":
        print(f"[DEBUG] Planner Thinking Config:")
        print(f"  - include_thoughts: {planner.thinking_config.include_thoughts}")
        print(f"  - thinking_budget: {planner.thinking_config.thinking_budget}")
        print(f"  - thinking_level: {planner.thinking_config.thinking_level}")

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
    
    # Debug: Verify agent has tools (only if DEBUG_ADK env var is set)
    if os.getenv("DEBUG_ADK") == "true":
        print(f"[DEBUG] Agent created with {len(wrapped_tools)} tools")
        print(f"[DEBUG] Agent model: {main_agent.model}")

    return main_agent

# Default agent instance (for backward compatibility)
default_agent = create_agent_for_session()


