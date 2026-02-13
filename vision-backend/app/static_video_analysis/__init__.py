"""
Static Video Analysis — Simple video Q&A using Gemini + RAG.

Flow:
  1. Upload video (with optional question) → Gemini analyzes → store in ChromaDB
  2. Ask question → RAG search first, reanalyze with Gemini only when needed
"""

from .agent import run_agent
from .gemini_client import analyze_video, analyze_video_full, ask_video
from .vector_store import has_video, search, store_analysis
from .registry import register_video, get_video_path

__all__ = [
    "run_agent",
    "analyze_video",
    "analyze_video_full",
    "ask_video",
    "has_video",
    "search",
    "store_analysis",
    "register_video",
    "get_video_path",
]
