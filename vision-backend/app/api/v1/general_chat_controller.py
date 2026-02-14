"""
General chat API: text message, stream, voice message (STT → LLM → TTS).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import json
import re
from io import BytesIO
from typing import Optional

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, File, Form, HTTPException, status, UploadFile
from fastapi.responses import StreamingResponse

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.general_chat_use_case import GeneralChatUseCase
from ...application.use_cases.chat.voice_chat_use_case import VoiceChatUseCase
from ...di.container import get_container

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(tags=["general-chat"])


def encode_sse(event_name: str, data: dict) -> bytes:
    """Encode event as Server-Sent Events format."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n".encode("utf-8")


def sanitize_header_value(value: str, max_length: int = 200) -> str:
    """Make string safe for HTTP headers: strip control chars, limit length."""
    if not value:
        return ""
    sanitized = re.sub(r"[\r\n\x00-\x1f\x7f-\x9f]", " ", str(value))
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 3] + "..."
    return sanitized


def sanitize_error_message(msg: str) -> str:
    """Sanitize error message for responses; handle rate-limit wording."""
    if not msg:
        return "An error occurred"
    if "rate limit" in msg.lower() or "429" in msg or "rpm" in msg.lower():
        return "API rate limit exceeded. Please try again in a few seconds."
    sanitized = msg.replace("`", "'").replace("\n", " ").replace("\r", " ")
    sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", sanitized)
    if len(sanitized) > 500:
        sanitized = sanitized[:500] + "..."
    return sanitized.strip()


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post("/message", response_model=ChatMessageResponse)
async def general_chat_message(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> ChatMessageResponse:
    """Send a text message to the general chat agent."""
    use_case = GeneralChatUseCase()
    try:
        return await use_case.execute(request=request, user_id=current_user.id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error_message(str(e)),
        )


@router.post("/message/stream")
async def general_chat_message_stream(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """Stream general chat response as Server-Sent Events."""
    use_case = GeneralChatUseCase()

    async def event_generator():
        try:
            async for item in use_case.stream_execute(request=request, user_id=current_user.id):
                ev = item.get("event") or "message"
                data = item.get("data") or {}
                yield encode_sse(str(ev), data if isinstance(data, dict) else {"data": data})
        except Exception as e:
            yield encode_sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/voice-message/stream")
async def voice_chat_message_stream(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream voice chat: audio → STT → LLM → TTS chunks via SSE.

    Events: stt_start, stt_result, llm_start, llm_token, llm_done, tts_start, tts_chunk, tts_done, done, error.
    """
    container = get_container()
    use_case = container.get(VoiceChatUseCase)

    async def event_generator():
        try:
            audio_bytes = await audio_file.read()
            if not audio_bytes:
                yield encode_sse("error", {"message": "Audio file is empty"})
                return
            async for event in use_case.stream_execute(
                audio_bytes=audio_bytes,
                filename=audio_file.filename or "audio.wav",
                session_id=session_id,
                user_id=current_user.id,
            ):
                ev = event.get("event") or "message"
                data = event.get("data") or {}
                yield encode_sse(str(ev), data if isinstance(data, dict) else {"data": data})
        except ValueError as e:
            yield encode_sse("error", {"message": sanitize_error_message(str(e))})
        except Exception as e:
            yield encode_sse("error", {"message": sanitize_error_message(str(e))})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/voice-message")
async def voice_chat_message(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """Process voice message and return audio response (WAV). Optional text/session in headers."""
    container = get_container()
    use_case = container.get(VoiceChatUseCase)

    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is empty",
            )
        audio_response_bytes, text_response, final_session_id = await use_case.execute(
            audio_bytes=audio_bytes,
            filename=audio_file.filename or "audio.wav",
            session_id=session_id,
            user_id=current_user.id,
        )
        sanitized_text = sanitize_header_value(text_response)
        sanitized_session_id = sanitize_header_value(str(final_session_id) if final_session_id else "")
        return StreamingResponse(
            BytesIO(audio_response_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Text-Response": sanitized_text,
                "X-Session-Id": sanitized_session_id,
            },
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_error_message(str(e)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=sanitize_error_message(str(e)),
        )
