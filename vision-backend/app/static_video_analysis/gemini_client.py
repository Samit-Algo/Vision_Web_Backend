"""Google Gemini: native video analysis — upload video, ask questions, get answers."""

import os
import time
from pathlib import Path

from ..core.config import get_settings

MODEL_ID = "gemini-3-flash-preview"


def _wait_for_file_active(client, file_obj, max_wait_sec: int = 300, poll_interval: int = 5) -> None:
    """Poll until file is ACTIVE. Videos can take 30–60+ seconds to process."""
    from google.genai import types

    start = time.time()
    while time.time() - start < max_wait_sec:
        file_obj = client.files.get(name=file_obj.name)
        state = getattr(file_obj, "state", None)
        state_str = str(state).upper() if state is not None else ""
        if state == types.FileState.ACTIVE or "ACTIVE" in state_str:
            return
        if state == types.FileState.FAILED or "FAILED" in state_str:
            raise RuntimeError(f"File processing failed: {file_obj.name}")
        time.sleep(poll_interval)
    raise TimeoutError(f"File still processing after {max_wait_sec}s")


def ask_video(video_path: str | Path, question: str) -> str:
    """Upload video to Gemini, ask question, return answer."""
    from google import genai

    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    api_key = (
        get_settings().gemini_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or gemini_api_key in config required")

    client = genai.Client(api_key=api_key)
    video_file = client.files.upload(file=str(video_path))
    _wait_for_file_active(client, video_file)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            video_file,
            question.strip() or GENERAL_PROMPT,
        ],
    )
    return (response.text or "").strip()


GENERAL_PROMPT = (
    "You are a security and incident analyst. Analyze this video for SECURITY and ACCIDENT-related details. "
    "Be extremely thorough and factual. Include:\n\n"
    "**WHAT HAPPENED?**\n"
    "- Describe every significant event, action, and incident in chronological order.\n"
    "- Note any accidents, collisions, falls, near-misses, or unsafe behavior.\n"
    "- Note any suspicious activity, unauthorized entry, or security breaches.\n"
    "- Include timestamps (e.g. 0:00–0:30) when events occur.\n\n"
    "**WHO ARE INVOLVED?**\n"
    "- Count and describe all people: number, approximate age, gender, clothing, position, role.\n"
    "- Describe vehicles (type, color, plate if visible, direction).\n"
    "- Note any objects or equipment involved.\n\n"
    "**HOW DID IT HAPPEN?**\n"
    "- Explain the sequence of events and cause.\n"
    "- Note conditions: lighting, weather, obstacles.\n"
    "- Describe movements: how people/vehicles entered, exited, approached.\n\n"
    "**WHEN & WHERE?**\n"
    "- Entry/exit times and locations.\n"
    "- How they came (e.g. from left, right, doorway, vehicle).\n"
    "- Key locations and camera coverage area.\n\n"
    "**AUDIO & VISIBLE TEXT**\n"
    "- Transcribe or summarize speech, dialogue, alarms, sirens.\n"
    "- Note any text on screen (signs, labels, plates).\n\n"
    "Be thorough so any security or accident-related question can be answered from your description."
)


def analyze_video_full(video_path: str | Path) -> str:
    """Full comprehensive analysis for indexing. No specific question."""
    return ask_video(video_path, GENERAL_PROMPT)


def analyze_video(video_path: str | Path, question: str | None = None) -> str:
    """Analyze video: if question provided, use it; else use general prompt."""
    prompt = question.strip() if question and question.strip() else GENERAL_PROMPT
    return ask_video(video_path, prompt)
