"""General Chat Agent - informational assistant for Vision AI system."""

import logging
from typing import Optional, List, Callable

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .tools import get_vision_rules_catalog, get_rule_details

logger = logging.getLogger(__name__)


STATIC_INSTRUCTION = """
You are a helpful AI assistant specialized in this Vision AI project.

### CAPABILITIES
1. **System Knowledge**: Explaining vision rules, requirements, and how the system works.
2. **Device Awareness**: Listing cameras, checking camera health.
3. **Deployment Tracking**: Summarizing which agents are active and on which cameras.
4. **Event Analysis**: Finding and summarizing detections/alerts that occurred.

### TOOL USAGE (STRICT)
- **Agents summary**: Use `get_deployed_agents_summary` when user asks "what's running?", "show my agents", "list agents".
- **Camera health**: Use `check_camera_health` when user asks if a camera works or why no alerts.
- **Events/detections**: Use `get_recent_detections` for alerts/events. If no events today (days_ago=0), try days_ago=1 then days_ago=2.
- **Event details**: Use `get_event_details` when user asks for "more details" about a specific event or event ID.
- **Camera lookup**: If user mentions a camera by name, use `find_camera` first to get camera_id, then use for health/events.
- **Rules catalog**: Use `get_vision_rules_catalog` when user asks what agents/detections are supported.
- **Rule details**: Use `get_rule_details` when user asks about a specific detection type.

### IMAGE RENDERING (MANDATORY)
- When a tool returns `evidence_url`, display it as inline image: `![Evidence](URL)`
- Do not explain local access or say you cannot show it. Render immediately.
- For multiple events, show each image after its summary.

### GENERAL RULES
- Only engage in conversations related to this Vision AI system.
- Be concise, clear, and professional.
- Use tools for accurate information. Never hallucinate IDs, counts, or URLs.
- If a tool returns an error, inform the user clearly and suggest they try again.
- Format responses in Markdown. Use **bold** for key terms, `code` for IDs, bullet lists for items.
"""


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_general_chat_agent(
    tools: Optional[List[Callable]] = None,
    instruction: Optional[Callable] = None
) -> LlmAgent:
    """
    Create a Google ADK agent for general chat with informational tools.

    Args:
        tools: Optional list of additional tools.
        instruction: Optional instruction provider (callable) for dynamic instructions (e.g. time context).

    Returns:
        LlmAgent: A chat agent instance with tools
    """
    default_tools = [get_vision_rules_catalog, get_rule_details]
    
    # Combine default tools with provided tools
    agent_tools = default_tools
    if tools:
        agent_tools = default_tools + tools

    # Align with main_agent: add thinking config
    planner = BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=-1,
        )
    )

    try:
        agent = LlmAgent(
            name="general_chat_agent",
            description="Conversational agent for vision rules, agents, cameras, and events.",
            static_instruction=STATIC_INSTRUCTION,
            instruction=instruction,
            model="groq/qwen/qwen3-32b",
            tools=agent_tools,
            planner=planner,
        )
        logger.debug("General chat agent created successfully")
        return agent
    except Exception as e:
        logger.exception(f"Failed to create general chat agent: {e}")
        raise
