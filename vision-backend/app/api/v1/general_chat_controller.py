"""Controller for general chat API endpoints."""
from typing import Optional
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import json

from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.general_chat_use_case import GeneralChatUseCase
from ...application.use_cases.chat.voice_chat_use_case import VoiceChatUseCase
from .dependencies import get_current_user


router = APIRouter(tags=["general-chat"])


@router.post("/message", response_model=ChatMessageResponse)
async def general_chat_message(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> ChatMessageResponse:
    """
    Send a message to the general chat agent
    
    Args:
        request: Chat message request with message and optional session_id
        current_user: Current authenticated user (from dependency)
        
    Returns:
        ChatMessageResponse with agent's response and session_id
    """
    general_chat_use_case = GeneralChatUseCase()
    
    try:
        response = await general_chat_use_case.execute(
            request=request,
            user_id=current_user.id
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"General chat error: {str(e)}"
        )


@router.post("/message/stream")
async def general_chat_message_stream(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream a message to the general chat agent (token streaming via SSE).
    """
    general_chat_use_case = GeneralChatUseCase()

    def _encode_sse(event_name: str, data: dict) -> bytes:
        payload = json.dumps(data, ensure_ascii=False)
        return (f"event: {event_name}\n" f"data: {payload}\n\n").encode("utf-8")

    async def event_generator():
        try:
            async for item in general_chat_use_case.stream_execute(request=request, user_id=current_user.id):
                ev = item.get("event") or "message"
                data = item.get("data") or {}
                yield _encode_sse(str(ev), data if isinstance(data, dict) else {"data": data})
        except Exception as e:
            yield _encode_sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/voice-message")
async def voice_chat_message(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """
    Send an audio message to the general chat agent and receive audio response.
    
    This endpoint:
    1. Converts audio to text using Groq STT
    2. Sends text to LLM for response
    3. Converts LLM response to audio using Groq TTS
    4. Returns audio file (WAV format)
    
    Args:
        audio_file: Audio file (supports: FLAC, MP3, WAV, MP4, MPEG, MPGA, M4A, OGG, WEBM)
        session_id: Optional session ID for conversation continuity
        current_user: Current authenticated user (from dependency)
        
    Returns:
        StreamingResponse with audio file (WAV format) and optional text in headers
        
    Raises:
        HTTPException: If audio processing fails
    """
    voice_chat_use_case = VoiceChatUseCase()
    
    try:
        # Read audio file bytes
        audio_bytes = await audio_file.read()
        
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is empty"
            )
        
        # Process voice chat: audio → text → LLM → audio
        audio_response_bytes, text_response, final_session_id = await voice_chat_use_case.execute(
            audio_bytes=audio_bytes,
            filename=audio_file.filename or "audio.wav",
            session_id=session_id,
            user_id=current_user.id
        )
        
        # Return audio as streaming response
        return StreamingResponse(
            BytesIO(audio_response_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Text-Response": text_response,  # Optional: include text in header for debugging
                "X-Session-Id": final_session_id   # Include session ID in header
            }
        )
        
    except ValueError as e:
        # Client errors (invalid format, empty text, etc.)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Voice chat error: {str(e)}"
        )
    except Exception as e:
        # Server errors (API failures, etc.)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice chat error: {str(e)}"
        )

