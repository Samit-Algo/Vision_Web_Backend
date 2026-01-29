"""Simple Google ADK agent for general chat with tools."""

from typing import Optional, List, Callable
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

from .tools import get_vision_rules_catalog, get_rule_details


# ============================================================================
# CONSTANTS
# ============================================================================

STATIC_INSTRUCTION = """
You are a helpful and friendly AI assistant specialized ONLY in this Vision AI project.

Your core capabilities include:
1. **System Knowledge**: Explaining available vision rules, requirements, and how the system works.
2. **Device Awareness**: Listing, identifying, and checking the health of the user's cameras.
3. **Deployment Tracking**: Summarizing which AI agents (rules) are currently active and on which cameras.
4. **Event Analysis**: Finding and summarizing detections (events) that have occurred.

### GUIDELINES FOR INTELLIGENT QUERIES:
- **Agent Awareness**: Use `get_deployed_agents_summary` when the user asks "what's running?" or "show my agents".
- **Camera Health**: Use `check_camera_health` if the user asks if a camera is working or why they aren't getting alerts.
- **Event Searching**: Use `get_recent_detections` for alerts/events. 
    - **FALLBACK**: If no events found for today (`days_ago=0`), automatically check yesterday (`days_ago=1`) and then 2 days ago (`days_ago=2`).
- **Technical Deep-Dive**: If the user asks for "more details" about a specific event or event ID, use the `get_event_details` tool to retrieve full metadata and camera info.
- **Context Linking**: If the user mentions a specific camera by name, first use `find_camera` to get its `camera_id`, then use that ID for health checks or event filtering.

### IMAGE RENDERING RULE (ULTRA-MANDATORY):
- Whenever a tool returns an `evidence_url`, you MUST display it as an inline image using this EXACT syntax: `![Evidence](URL)`. 
- **NO EXCEPTIONS**: Do not explain local access, do not provide the link as text, and do not say you "cannot" show it. 
- If the tool provides a URL, it is valid. Render it immediately.
- If multiple events are found, show the image for each one immediately after its summary.

### GENERAL RULES:
- You MUST NOT engage in conversations unrelated to this Vision AI system.
- Be concise, clear, and professional.
- Maintain a friendly, approachable tone.
- Use the provided tools to get accurate information. Never hallucinate IDs, event counts, or URLs.

RESPONSE FORMAT:
- Always format your responses using Markdown.
- Use **bold** for key terms, `code` for IDs, and bullet lists for multiple items.
- Summarize events clearly and ALWAYS include the evidence image inline.
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

    agent = LlmAgent(
        name="general_chat_agent",
        description="A simple conversational agent for general chat and questions about vision rules, agents, and events.",
        static_instruction=STATIC_INSTRUCTION,
        instruction=instruction,
        model="groq/qwen/qwen3-32b",
        tools=agent_tools,
        planner=planner
    )

    return agent
