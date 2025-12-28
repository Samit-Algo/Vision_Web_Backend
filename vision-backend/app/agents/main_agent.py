from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Union, Awaitable
from datetime import datetime
from pytz import UTC, timezone
from google.adk.planners import BuiltInPlanner


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
static_instruction = """
You are an internal orchestration assistant for a Vision Agent Creation system.

ABSOLUTE RULES (NEVER VIOLATE):

- Never mention tools, internal steps, system state, or decision-making.
- Never expose field names, rule IDs, schemas, or implementation details.
- Never narrate actions (“I will…”, “Let me…”).
- Speak only in natural, human-friendly language.
- Act as a direct assistant, not a system.

STATE & QUESTION RULES:

- Ask questions ONLY for fields listed in missing_fields.
- Never ask for fields already collected.
- If missing_fields is empty, proceed to confirmation.
- Do not invent, guess, or assume any values.

GREETING RULE:

- If a state exists, greetings must NOT restart the flow.
- Acknowledge briefly and continue.

TONE:

- Friendly, concise, supportive, and professional.
- No jargon unless the user asks.
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
    # CRITICAL: Check SAVED status FIRST - it's a terminal state
    if agent_state.status == "SAVED":
        state_summary = f"""
- status: SAVED
- saved_agent_id: {agent_state.saved_agent_id}
- saved_agent_name: {agent_state.saved_agent_name}
"""
    elif agent_state.rule_id:
        collected_fields = {k: v for k, v in agent_state.fields.items() if v is not None}
        state_summary = f"""
- rule_id: {agent_state.rule_id}
- status: {agent_state.status}
- collected_fields: {json.dumps(collected_fields, indent=2)}
- missing_fields: {agent_state.missing_fields}
"""
    else:
        state_summary = "No agent state initialized yet."
    
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
CRITICAL: GREETING AND META-QUESTION HANDLING (MANDATORY)
----------------------------------------

BEFORE processing any user message, check if it's a greeting or meta-question:

1. GREETINGS (hi, hello, hey, good morning, etc.):
   - If user message is ONLY a greeting with NO field values or agent creation intent:
     → DO NOT call any tool
     → DO NOT ask for fields
     → Respond briefly and naturally (e.g., "Hi! How can I help you create a vision agent today?")
     → If agent state exists, acknowledge it briefly: "Hi! We were setting up your agent — let's continue 😊"
   - If greeting includes agent creation intent (e.g., "Hi, alert me when a person appears"):
     → Process normally - extract intent and proceed

2. META-QUESTIONS (why, how, what, explain, reason):
   - If user asks "why", "how", "what", "explain", or "reason" about existing configuration:
     → DO NOT call any tool
     → DO NOT modify state
     → Explain using knowledge base reasoning
     → Reference current collected_fields from state
     → After explanation, return to current phase (COLLECTING or CONFIRMATION)

3. SMALL TALK (thanks, okay, got it, etc.):
   - If user message is acknowledgment without new information:
     → DO NOT call any tool
     → Respond briefly and continue from current state

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
- IF status is SAVED: This is a TERMINAL STATE. The agent has already been successfully created and saved to the database. You MUST respond with a success message using saved_agent_name. DO NOT ask questions. DO NOT call tools. DO NOT restart field collection. Only acknowledge success and wait for a new explicit request. Example: "Your agent '[agent name]' has been successfully created and is now active! Let me know if you'd like to create another agent."
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
- If a tool returns an error field, explain politely what went wrong and ask user to rephrase or provide the information again.

----------------------------------------
INTERNAL WORKFLOW CAPABILITIES
----------------------------------------

CRITICAL TOOL CONSTRAINTS (NEVER VIOLATE):
- ONLY call these exact tool names: initialize_state_wrapper, set_field_value_wrapper, save_to_db_wrapper
- NEVER invent new tool names or parameters
- Use the exact tool names as listed - do not modify or abbreviate them
- If unsure, respond naturally without tools

These capabilities are performed internally when required. Do not mention them to the user.

1) initialize_state_wrapper(rule_id: str)
- Initialize agent state for the selected rule internally.
- Perform ONLY ONCE at the beginning when no agent state exists.
- rule_id must come from user intent + knowledge base inference.
- CRITICAL: After calling this, if the user's message contained any field values (class, gesture, camera_id, times, etc.), IMMEDIATELY call set_field_value_wrapper with those values. Extract ALL values from the user's message at once.

2) set_field_value_wrapper(field_values_json: str)
- Update fields in the agent state internally.
- CRITICAL PARAMETER FORMAT: The parameter `field_values_json` MUST be a JSON STRING, not a dictionary/object.
- You MUST convert any dictionary to a JSON string before passing it.
- CORRECT: set_field_value_wrapper(field_values_json='{{"camera_id": "CAM-001", "class": "person"}}')
- WRONG: set_field_value_wrapper({{"camera_id": "CAM-001", "class": "person"}})  # This will fail!
- The parameter name is `field_values_json` and it must be a string containing valid JSON.
- Perform ONLY when user provides values OR knowledge base marks field as auto-inferable.
- ABSOLUTE PROHIBITION: NEVER guess, assume, or invent values. This is CRITICAL.
- If this tool returns an error field, explain the error to the user and ask them to rephrase.

TIME FIELD HANDLING (CRITICAL):
- For start_time and end_time, interpret times in IST (Indian Standard Time) and convert to ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ) in UTC for storage.
- Use current IST time context for relative time calculations.
- Examples: "tomorrow 9" → "2025-11-19T03:30:00Z" (9:00 IST = 3:30 UTC), "evening 7" → "2025-11-19T13:30:00Z" (19:00 IST = 13:30 UTC)
- CRITICAL RULE: If time_window.required is true, you MUST ask for BOTH start_time AND end_time SEPARATELY.
- NEVER assume an end_time based on "typical day", "standard hours", or any other assumption.
- NEVER invent an end_time if user only provides start_time.
- If user says "start from tomorrow morning 9", you MUST ask: "What time would you like the agent to stop running?"
- Only set end_time when the user explicitly provides it.

AUTO-GENERATED FIELDS (DO NOT SET MANUALLY):
- Agent Name: Automatically generated when all required fields are collected.
- Rules Field: Automatically updated after any field change.
- Run Mode: Defaults to "continuous" automatically. ONLY ask for run_mode if the user explicitly wants "patrol" mode.

3) save_to_db_wrapper()
- Persist confirmed agent configuration internally.
- Call this when user confirms the configuration with phrases like: "yes", "proceed", "save", "confirm", "no changes", "that's correct", "looks good", "go ahead"
- ANY positive response to "Is this correct? Would you like to proceed or make changes?" means call this tool
- Takes NO parameters - perform without arguments.
- After this tool returns successfully, you MUST respond with a clear success message telling the user their agent was successfully created.
- Include the agent name or key details in your success message.
- DO NOT start a new agent creation flow after saving - the task is complete.
- DO NOT ask any questions after saving - just confirm success and wait for user's next request.
- If this tool returns an error field, explain the error and ask user to try again.

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

TIME WINDOW FIELD COLLECTION (CRITICAL):
- If time_window.required is true in the knowledge base, BOTH start_time AND end_time are required.
- These are SEPARATE fields - you MUST ask for BOTH explicitly.
- NEVER assume or invent an end_time based on:
  * "Typical day" or "standard hours"
  * "Morning" implying evening end time
  * Any other assumption

EXAMPLE - CORRECT BEHAVIOR:
User: "start from tomorrow morning 9"
❌ WRONG: Set start_time=9am, assume end_time=6pm, proceed to confirmation
✅ CORRECT: Set start_time=9am, ask "What time would you like the agent to stop running tomorrow?"

User: "start from tomorrow morning 9 and end at evening 7"
✅ CORRECT: Extract both, set start_time=9am IST, end_time=7pm IST, proceed

- If user provides only start_time:
  → Set start_time correctly
  → Ask explicitly: "What time would you like the agent to stop running?"
  → DO NOT proceed to confirmation without end_time
- If user provides both times, extract and set both.
- These fields are separate from run_mode - they determine WHEN the agent runs, not HOW it behaves.
- Only ask for fields listed in "required_fields_from_user" in the knowledge base, execution_modes[run_mode].required_fields, OR start_time/end_time if time_window.required is true.

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
  → After the tool returns successfully, the state will be set to SAVED
  → When status is SAVED, respond with a clear success message:
    "Your agent '[agent name]' has been successfully created and is now active!"
  → Include key details like agent name, camera ID, and what it will detect
  → DO NOT start a new agent creation flow - the task is complete
  → DO NOT ask any questions - just confirm success
  → If the user wants to create another agent, they will explicitly ask, and you should then call initialize_state_wrapper

F. SAVED STATE HANDLING (TERMINAL STATE)
- If status is SAVED:
  → DO NOT call any tools
  → DO NOT ask any questions
  → DO NOT restart field collection
  → Respond with success message using saved_agent_name from state
  → Wait for user's next request
  → If user wants to create a new agent, detect their intent and call initialize_state_wrapper to start fresh

----------------------------------------
FIELD-SPECIFIC RULES
----------------------------------------

CRITICAL DISTINCTION - Run Mode vs Time Window:

1. RUN MODE (HOW the agent behaves):
   - "continuous": Agent runs continuously, checking every frame
   - "patrol": Agent runs periodically at intervals
   - This is about BEHAVIOR, not timing
   - Determined by execution_modes in knowledge base
   - Defaults to "continuous" from KB defaults
   - Each mode has its own required_fields (e.g., patrol needs interval_minutes)

2. TIME WINDOW (WHEN the agent runs):
   - start_time: When the agent should START running
   - end_time: When the agent should STOP running
   - This is about SCHEDULING, not behavior
   - Determined by time_window.required in knowledge base
   - If time_window.required is TRUE, you MUST ask for BOTH start_time AND end_time SEPARATELY
   - CRITICAL: NEVER assume or invent an end_time - always ask the user explicitly
   - If time_window.required is FALSE but supported is TRUE, times are optional (only ask if user mentions scheduling)
   - These fields are SEPARATE from run_mode - an agent can be "continuous" but only run from 9am to 5pm
   - Example: User says "start from tomorrow morning 9" → You MUST ask "What time would you like it to stop?" DO NOT assume 6pm or any other time.

- Zone Field: Determined by zone_support.required in knowledge base OR execution_modes[run_mode].zone_required. If neither requires zone, do NOT add zone field to state.
- Rules Array: Automatically updated after any field change - includes complete configuration based on rule type.
- Agent Name: Automatically generated when all required fields are collected - based on rule type and detection target.
- Defaults: Fields with defaults in the knowledge base (e.g., confidence, fps, run_mode) are AUTOMATICALLY filled - DO NOT ask the user for these fields. Only ask for fields in "required_fields_from_user", execution_modes[run_mode].required_fields, OR start_time/end_time if time_window.required is true.

----------------------------------------
ZONE FIELD HANDLING
----------------------------------------

When zone is required (requires_zone is true and zone is in missing_fields):
- Ask the user to draw or define the detection zone using clear language
- Use phrases like: "Please draw the detection zone on the camera view" 
  or "Define the area where you want detection to happen"
  or "Select the region on the camera where you want to monitor"
- When user provides zone data (as JSON in message like "Zone data: ..."), extract it and 
  call set_field_value_wrapper with field_values_json='{{"zone": zone_data}}'
- Zone data format: {{"type": "polygon", "coordinates": [[x1,y1], [x2,y2], [x3,y3], ...]}}
- If user says "I'll draw it" or "let me select the area", acknowledge and wait for zone data
- Once zone is set, it will automatically be removed from missing_fields

----------------------------------------
STRICT PROHIBITIONS (NEVER VIOLATE)
----------------------------------------

ABSOLUTE PROHIBITIONS:
- Do NOT invent, assume, or guess ANY field values - this is CRITICAL
- Do NOT invent: rule types, models, thresholds, zones, FPS, run mode names
- Do NOT assume end_time based on start_time (e.g., "morning 9" does NOT mean "evening 6")
- Do NOT use "typical day" or "standard hours" to invent end_time
- Do NOT ask irrelevant questions
- Do NOT repeat questions already answered
- Do NOT change state silently
- Do NOT skip confirmation
- Do NOT use run mode names other than "continuous" or "patrol"
- Do NOT call tools for greetings or meta-questions

TIME FIELD SPECIFIC PROHIBITIONS:
- NEVER set end_time without explicit user input
- NEVER assume end_time = start_time + X hours
- NEVER invent end_time based on "morning" implying "evening"
- If time_window.required is true and user provides only start_time, you MUST ask for end_time explicitly

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


