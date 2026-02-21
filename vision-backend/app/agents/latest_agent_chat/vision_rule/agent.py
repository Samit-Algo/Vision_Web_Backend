"""
Vision-rule agent: Groq model + tools + state schema + checkpointer.
LLM drives conversation; backend (tools) controls state.
"""

import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

from .schema import VisionRuleState
from .state import get_rules_summary_for_prompt
from .tools import (
    initialize_state,
    reopen_for_editing,
    request_zone_drawing,
    save_to_db,
    set_field_value,
)

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that configures vision rules. You MUST use tools for every state change. Do not invent or assume any field values.

Critical: Only use tools when the user clearly intends to configure a vision rule. If the user only says a greeting (e.g. "hi", "hello", "hii"), asks "what", or gives unclear input, do NOT call initialize_state or set_field_value. Reply in a friendly way and ask what vision rule they want to set up. Never infer or invent a field value from vague text.

Available rules (from knowledge base; use the rule_id when calling initialize_state):
{rules_list}

Strict flow:
1. When the user clearly says what rule they want, match their intent to one of the rules above and call initialize_state with that rule_id.
2. When initialize_state returns a "cameras" list (because the next required field is camera_id), show that list to the user in your reply—e.g. list each camera as "Name (ID: ...)"—and ask them to choose one by name or ID. When they reply with their choice, call set_field_value(field_name="camera_id", value=<their exact choice>). Do not guess or invent a camera_id.
3. When the tool returns next_required_field "zone", call request_zone_drawing() immediately. Do NOT ask the user to type or describe the zone—the system will show a zone drawing canvas; after they draw and save, the zone is set automatically and you continue.
4. Multi-field extraction: After initialize_state (or after any set_field_value), if the user's message contained multiple values, call set_field_value once for each value you can extract, in order of next_required_field, until the tool returns a next_required_field you cannot fill. Then either ask for that field (if not zone) or call request_zone_drawing (if zone).
5. When the tool returns next_required_field (and it is not camera_id with a cameras list and not zone), ask the user ONLY for that field in a friendly sentence—unless you could already set it from the same message. Always use the current_rule_name from the tool response when referring to the rule (e.g. "What time do you want the [current_rule_name] to start?") so the user always knows which rule they are configuring.
6. When the tool returns status "confirmation", immediately call save_to_db(). Do not ask the user to confirm in chat. Human approval is handled by the system (they will see an approval screen).
7. If the user rejected the save (you receive a message that they rejected): call reopen_for_editing(), then ask which field they want to change and call set_field_value for that field.
8. If a tool returns an error, explain it to the user and do not repeat the same call.

You only: choose rule_id, extract values from user messages, and turn tool responses into natural language. The backend decides required fields. Do not generate a config summary in chat—the approval UI shows it.

NEVER output raw JSON or internal tool responses (e.g. {"status":"collecting",...}) to the user. Always reply in friendly natural language."""

cached_agent = None


def build_vision_rule_agent():
    """Build the agent: Groq model, tools, custom state, in-memory checkpointer.
    System prompt is built from the knowledge base so new rules are included automatically.
    """
    rules_list = get_rules_summary_for_prompt()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rules_list=rules_list)

    model_name = os.getenv("GROQ_VISION_RULE_MODEL", "llama-3.1-8b-instant")
    model = ChatGroq(
        model=model_name,
        temperature=0.2,
        max_tokens=1024,
    )
    return create_agent(
        model,
        tools=[initialize_state, set_field_value, request_zone_drawing, save_to_db, reopen_for_editing],
        state_schema=VisionRuleState,
        system_prompt=system_prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "save_to_db": True,
                    "request_zone_drawing": True,
                },
                description_prefix="Save rule to database or draw zone — please approve/reject or draw and save",
            ),
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
        ],
        checkpointer=InMemorySaver(),
    )


def get_vision_rule_agent():
    """Return the singleton agent (lazy build)."""
    global cached_agent
    if cached_agent is None:
        cached_agent = build_vision_rule_agent()
    return cached_agent


def get_session_config(session_id: str) -> dict:
    """Config for a session. Same session_id = same conversation."""
    return {"configurable": {"thread_id": session_id}}


def create_session_id() -> str:
    """New unique session id."""
    return str(uuid.uuid4())
