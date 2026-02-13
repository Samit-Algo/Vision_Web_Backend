"""Groq reasoning agent with RAG + reanalyze tools for video QA."""

import json
from pathlib import Path

from groq import Groq

from ..core.config import get_settings
from .gemini_client import ask_video
from .vector_store import has_video, search, store_analysis

AGENT_MODEL = "qwen/qwen3-32b"
MAX_AGENT_TURNS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_video_knowledge",
            "description": "Search the stored video analysis for relevant information. Use this FIRST before reanalyzing. Returns relevant text chunks from the video's analyzed content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query - rephrase the user's question for semantic search"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reanalyze_video",
            "description": "Re-analyze the video with Gemini using a specific question. Use ONLY when search_video_knowledge returns insufficient or irrelevant information to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The specific question to ask about the video"},
                },
                "required": ["question"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an intelligent video QA assistant. You have access to:
1. search_video_knowledge - Search stored analysis (use FIRST, it's fast)
2. reanalyze_video - Re-analyze video with a specific question (use only when search results are insufficient)

Workflow:
- First call search_video_knowledge with the user's question (or a rephrased query)
- If the search results contain enough information to answer, synthesize a clear answer
- If search results are empty, irrelevant, or insufficient, call reanalyze_video with the user's question
Give a reasoned answer based on what is visible. Avoid saying "Not observable" unless truly nothing relevant is shown.

Answering rules:
- Give ONLY the direct answer. Do NOT include meta-commentary such as "No further analysis is needed", "The search results provided sufficient detail", "The analysis does not mention...", or similar.
- Format responses in markdown when helpful: use **bold** for emphasis, lists for multiple items, etc."""

def run_agent(video_id: str, question: str, video_path: str | Path) -> str:
    """Run Groq agent with tool calling. Tools need access to video_id and video_path."""
    settings = get_settings()
    api_key = settings.groq_api_key
    if not api_key:
        raise RuntimeError("GROQ_API_KEY required")

    client = Groq(api_key=api_key)
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_id}")

    def search_video_knowledge(query: str) -> str:
        results = search(video_id, query, top_k=5)
        if not results:
            return "No relevant information found in stored analysis."
        return "\n\n---\n\n".join(results)

    def reanalyze_video(question: str) -> str:
        return ask_video(path, question)

    tools_map = {
        "search_video_knowledge": search_video_knowledge,
        "reanalyze_video": reanalyze_video,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Video ID: {video_id}\n\nUser question: {question}"},
    ]

    model = getattr(settings, "static_video_agent_model", None) or AGENT_MODEL
    temperature = getattr(settings, "llm_temperature", 0.3)

    for _ in range(MAX_AGENT_TURNS):
        create_kwargs = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": 1024,
        }
        # qwen3-32b: reasoning_format must be parsed or hidden for tool use
        if "qwen" in model.lower():
            create_kwargs["reasoning_format"] = "hidden"
        response = client.chat.completions.create(**create_kwargs)

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return (msg.content or "").strip()

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn = tools_map.get(fn_name)
            if not fn:
                result = f"Unknown tool: {fn_name}"
            else:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = fn(**args)
                except Exception as e:
                    result = f"Error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": str(result),
            })

    return "Unable to generate answer after multiple attempts."
