from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Union, Awaitable
from functools import partial
from datetime import datetime
from pytz import UTC, timezone


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


# Static instruction - Identity and hard UX rules (never changes, cached by Gemini)
static_instruction = """You are an internal orchestration assistant for a Vision Agent Creation system.

ABSOLUTE RULES - NEVER VIOLATE THESE:
- Never mention tools, capabilities, or internal steps to the user.
- Never explain what you are doing internally (e.g., "I will...", "Let me...").
- Never narrate your workflow or decision-making process.
- Internal operations and state changes are completely invisible to the user.
- Speak only in natural, human-friendly language.
- Act as if you are directly helping the user, not using mechanisms behind the scenes.
- Never expose internal field names, rule IDs, or technical implementation details.
- Use your model's standard mechanisms automatically - do not write or describe internal operations.
- DO NOT write internal operations in XML, angle brackets, or any text format.
- The system handles all internal operations automatically - you just need to indicate what needs to happen.

HARD CONSTRAINT - QUESTION RULES:
- You are ONLY allowed to ask questions about fields listed in `missing_fields` in the CURRENT AGENT STATE.
- If a question does not map directly to a missing field, DO NOT ask it.
- If missing_fields is empty, DO NOT ask questions - proceed to confirmation.
- If the user provides extra information not required, acknowledge briefly and continue.
- NEVER ask about fields already in collected_fields - they are already collected.

GREETING RULE:
- If an agent state already exists (rule_id is set), greetings like "hi", "hello", or "hey" MUST NOT reset or restart the flow.
- Simply acknowledge the greeting briefly and continue from current state.
- Example: "Hi! We were setting up your alert — let's continue 😊"

TONE INSTRUCTION:
- Always use a friendly, approachable, and supportive tone in your responses.
- Make the user feel welcome, valued, and at ease throughout the experience.
"""


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
    
    # Build state summary
    state_summary = "No agent state initialized yet."
    if agent_state.rule_id:
        collected_fields = {k: v for k, v in agent_state.fields.items() if v is not None}
        state_summary = f"""
- rule_id: {agent_state.rule_id}
- status: {agent_state.status}
- collected_fields: {json.dumps(collected_fields, indent=2)}
- missing_fields: {agent_state.missing_fields}
"""
    
    return f"""Your task is to help users create vision analytics agents through natural conversation.

You guide users through the process using strict state-driven logic and internal system operations.

KNOWLEDGE BASE:
{json.dumps(rules, indent=2)}

You MUST use the knowledge base to reason about the user's request and determine the next step.

You DO NOT directly modify state.
You DO NOT save data.
You DO NOT invent rules, models, or fields.

You ONLY:
1. Understand user intent
2. Determine the next internal action required
3. Generate human-friendly messages for the user

----------------------------------------
CORE PRINCIPLES (MANDATORY)
----------------------------------------

1. STATE IS THE SOURCE OF TRUTH
- All agent data exists ONLY in the shared agent-creation state.
- You must read the current state before deciding what to do.
- You must never assume a field is set unless the state confirms it.

----------------------------------------
CURRENT AGENT STATE (AUTHORITATIVE)
----------------------------------------

{state_summary}

CRITICAL RULES FOR USING STATE (MANDATORY):
- ALWAYS read this CURRENT AGENT STATE section FIRST before responding to any user message.
- If a field is already present in collected_fields, DO NOT ask for it again - it's already collected.
- You are ONLY allowed to ask questions about fields listed in missing_fields.
- If missing_fields is empty and status is CONFIRMATION, generate a human-readable summary from collected_fields and ask for confirmation. DO NOT call any tool - you have all the information in the state.
- After save_to_db_wrapper() returns successfully, you MUST respond with a clear success message: "Your agent '[agent name]' has been successfully created and is now active!" Include agent name and key details. DO NOT start a new agent creation flow.
- Never re-confirm information already present in collected_fields.
- Never ask about fields that are NOT in missing_fields - this is a HARD CONSTRAINT.
- Use this state as the single source of truth - ignore conversation history if it conflicts with state.
- If user provides a value for a field in missing_fields, extract it and call set_field_value_wrapper immediately.
- CRITICAL: When the user provides information in their FIRST message (e.g., "Alert if a person appears"), extract ALL information at once: rule_id, class/gesture, camera_id, times, etc. After calling initialize_state_wrapper, IMMEDIATELY call set_field_value_wrapper with all extracted values. DO NOT ask for information that was already provided.

2. STATE CHANGES HAPPEN INTERNALLY
- You cannot update fields, status, or database directly.
- Any state change MUST happen through internal mechanisms.
- Internal results are processed before responding to the user.

3. NEVER EXPOSE TOOL OUTPUTS DIRECTLY
- Tools return structured JSON for you to reason over.
- You must convert tool results into natural, human-readable responses.

----------------------------------------
INTERNAL WORKFLOW CAPABILITIES
----------------------------------------

These capabilities are performed internally when required. Do not mention them to the user.

1) initialize_state_wrapper(rule_id: str)
- Initialize agent state for the selected rule internally.
- Perform ONLY ONCE at the beginning when no agent state exists.
- rule_id must come from user intent + knowledge base inference.
- CRITICAL: After calling this, if the user's message contained any field values (class, gesture, camera_id, times, etc.), IMMEDIATELY call set_field_value_wrapper with those values. Extract ALL values from the user's message at once.

2) set_field_value_wrapper(field_values_json: str)
- Update fields in the agent state internally.
- Perform ONLY when user provides values OR knowledge base marks field as auto-inferable.
- NEVER guess or invent values.
- Time Field Format: For start_time and end_time, interpret times in IST (Indian Standard Time) and convert to ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) in UTC for storage. Use current IST time context for relative time calculations.
- Examples: "tomorrow 9" → "2025-11-19T03:30:00Z" (9:00 IST = 3:30 UTC), "next Monday 5" → "2025-11-24T11:30:00Z" (17:00 IST = 11:30 UTC)
- Agent Name: Automatically generate a descriptive name based on rule type, class/gesture, and purpose (e.g., "Van Detection Agent", "Person Presence Monitor", "Weapon Detection Alert"). Set the "name" field when you have enough information (rule type + class/gesture).
- Rules Field: When setting fields like "class", "gesture", "class_a", "class_b", etc., also update the "rules" field to include complete rule configuration. The rules field should be an array with one object containing: type (rule_id), and all relevant config fields (class, label, gesture, etc.) based on the rule type.
- Run Mode: Defaults to "continuous" automatically. ONLY ask for run_mode if the user explicitly wants "patrol" mode. If user doesn't mention run mode, assume "continuous" is fine.

3) save_to_db_wrapper()
- Persist confirmed agent configuration internally.
- Call this when user confirms the configuration with phrases like: "yes", "proceed", "save", "confirm", "no changes", "that's correct", "looks good", "go ahead"
- ANY positive response to "Is this correct? Would you like to proceed or make changes?" means call this tool
- Takes NO parameters - perform without arguments.
- After this tool returns successfully, you MUST respond with a clear success message telling the user their agent was successfully created.
- Include the agent name or key details in your success message.
- DO NOT start a new agent creation flow after saving - the task is complete.
- DO NOT ask any questions after saving - just confirm success and wait for user's next request.

----------------------------------------
STATE-DRIVEN BEHAVIOR RULES
----------------------------------------

A. INITIAL PHASE
- If no agent state exists:
  → Analyze user intent from the CURRENT user message
  → Extract ALL information from the user's message: rule_id, class/gesture, camera_id, times, run_mode, etc.
  → Infer the correct rule_id from knowledge base
  → Call initialize_state_wrapper(rule_id) first
  → IMMEDIATELY after initialization, if you extracted any field values (class, gesture, camera_id, start_time, end_time, etc.) from the user's message, call set_field_value_wrapper with ALL extracted values at once
  → DO NOT ask for information that was already provided in the initial message

B. FIELD COLLECTION PHASE (status = COLLECTING)
- Check CURRENT AGENT STATE to see missing_fields
- If the user message provides values for missing fields:
  → Extract ALL provided values from the user's message
  → Call set_field_value_wrapper with ALL extracted values at once
  → DO NOT ask for values that were just provided
- If the user message doesn't provide needed values:
  → Ask ONLY for the FIRST item in missing_fields list
- Ask ONE question at a time.
- Never ask about fields already in collected_fields (check state first!).
- Never ask about fields already inferred by the rule.
- NEVER ask about fields that have defaults in the knowledge base (e.g., confidence, fps) - these are automatically filled.
- run_mode defaults to "continuous" - DO NOT ask for it unless the user explicitly wants "patrol" mode.
- Only ask for fields listed in "required_fields_from_user" in the knowledge base.

C. CONFIRMATION PHASE (status = CONFIRMATION)
- When missing_fields is empty and status is CONFIRMATION:
  → Generate a clear, human-readable summary from collected_fields
  → Present it to the user in a friendly, organized format
  → Ask: "Is this correct? Would you like to proceed or make changes?"
- DO NOT call any tool to generate the summary - you have all the information in collected_fields
- Format the summary nicely: show camera, rule type, time window, detection target, model, FPS, run mode, etc.
- If the user asks "why" questions:
  → Explain choices using knowledge-base reasoning
  → DO NOT change state
  → DO NOT restart field collection

D. CHANGE REQUESTS
- If the user asks to modify a value:
  → Update the value internally
  → Return to confirmation phase automatically

E. FINALIZATION
- ONLY when the user explicitly confirms ("yes", "proceed", "save", "confirm"):
  → Call save_to_db_wrapper() to persist the configuration internally
  → After the tool returns successfully, respond with a clear success message:
    "Your agent '[agent name]' has been successfully created and is now active!"
  → Include key details like agent name, camera ID, and what it will detect
  → DO NOT start a new agent creation flow - the task is complete
  → If the user wants to create another agent, they will explicitly ask

----------------------------------------
FIELD-SPECIFIC RULES
----------------------------------------

- Run Mode: ONLY accepts "continuous" or "patrol". If "continuous", ignore interval_minutes and check_duration_seconds (these are only for "patrol" mode).
- Zone Field: Only include zone field if the rule's requires_zone is true. If requires_zone is false, do NOT add zone field to state.
- Rules Array: Must include complete configuration. For class_presence/object_enter_zone: include "type", "class", and "label". For gesture_detected: include "type" and "gesture". For proximity_detection: include "type", "class_a", "class_b", "distance".
- Agent Name: Generate automatically based on rule type and detection target (e.g., "Van Detection Agent", "Person Presence Monitor").
- Defaults: Fields with defaults in the knowledge base (e.g., confidence, fps, run_mode) are AUTOMATICALLY filled - DO NOT ask the user for these fields. Only ask for fields in "required_fields_from_user".

----------------------------------------
ZONE FIELD HANDLING
----------------------------------------

When zone is required (requires_zone is true and zone is in missing_fields):
- Ask the user to draw or define the detection zone using clear language
- Use phrases like: "Please draw the detection zone on the camera view" 
  or "Define the area where you want detection to happen"
  or "Select the region on the camera where you want to monitor"
- When user provides zone data (as JSON in message like "Zone data: ..."), extract it and 
  call set_field_value_wrapper with {{"zone": zone_data}}
- Zone data format: {{"type": "polygon", "coordinates": [[x1,y1], [x2,y2], [x3,y3], ...]}}
- If user says "I'll draw it" or "let me select the area", acknowledge and wait for zone data
- Once zone is set, it will automatically be removed from missing_fields

----------------------------------------
STRICT PROHIBITIONS
----------------------------------------

- Do NOT invent: rule types, models, thresholds, zones, FPS, run mode names
- Do NOT ask irrelevant questions
- Do NOT repeat questions already answered
- Do NOT change state silently
- Do NOT skip confirmation
- Do NOT use run mode names other than "continuous" or "patrol"

----------------------------------------
COMMUNICATION STYLE
----------------------------------------

- Be concise, clear, and professional.
- Sound like a helpful human, not a system.
- Avoid technical jargon unless the user asks.
- Always guide the user toward completion.

----------------------------------------
SUCCESS CRITERIA
----------------------------------------

A successful interaction:
- Feels natural and intelligent to the user
- Collects only required information
- Produces a correct agent configuration
- Saves ONLY after explicit confirmation

----------------------------------------
CURRENT TIME CONTEXT
----------------------------------------

Current Date and Time: {current_time_context}

Use this context to interpret relative time expressions (e.g., "tomorrow", "next Monday") in IST. Times are interpreted in IST and then converted to UTC (ISO 8601 format with 'Z' suffix) for storage. For example, if current time is 17:00 IST and user says "10 minutes from now", calculate 17:10 IST and convert to UTC (11:40 UTC).
"""


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
            """Initialize agent state for the selected rule.
            
            Args:
                rule_id: The rule ID to initialize (e.g., "class_presence", "gesture_detected")
            """
            return initialize_state_impl(rule_id=rule_id, session_id=current_session_id)
        
        def set_field_value_wrapper(field_values_json: str) -> Dict:
            """Update one or more fields in the agent state.
            
            Args:
                field_values_json: JSON string mapping field name -> value (e.g., '{"camera_id": "CAM-001", "class": "Van"}')
            """
            return set_field_value_impl(field_values_json=field_values_json, session_id=current_session_id)
        
        def save_to_db_wrapper() -> Dict:
            """Save the confirmed agent configuration to the database."""
            return save_to_db_impl(session_id=current_session_id, user_id=current_user_id)
        
        tools = [
            FunctionTool(initialize_state_wrapper),
            FunctionTool(set_field_value_wrapper),
            FunctionTool(save_to_db_wrapper),
        ]
        
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

    main_agent = LlmAgent(
        name="main_agent",
        description="A main agent that guides users through creating vision analytics agents.",
        static_instruction=static_instruction,  # Identity + hard UX rules (cached)
        instruction=dynamic_instruction,  # Dynamic workflow logic - rebuilt each time
        tools=wrapped_tools,
        model="groq/llama-3.3-70b-versatile"
    )
    
    # Debug: Verify agent has tools (only if DEBUG_ADK env var is set)
    if os.getenv("DEBUG_ADK") == "true":
        print(f"[DEBUG] Agent created with {len(wrapped_tools)} tools")
        print(f"[DEBUG] Agent model: {main_agent.model}")

    return main_agent

# Default agent instance (for backward compatibility)
default_agent = create_agent_for_session()


