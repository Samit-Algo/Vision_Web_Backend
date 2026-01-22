"""Controller for general chat API endpoints."""
from typing import Optional
from io import BytesIO
import re
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import json

from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.general_chat_use_case import GeneralChatUseCase
from ...application.use_cases.chat.voice_chat_use_case import VoiceChatUseCase
from ...di.container import get_container
from .dependencies import get_current_user


router = APIRouter(tags=["general-chat"])


def sanitize_header_value(value: str, max_length: int = 200) -> str:
    """
    Sanitize a string to be safe for use in HTTP headers.
    Removes newlines, control characters, and limits length.
    """
    if not value:
        return ""
    # Remove newlines, carriage returns, and other control characters
    sanitized = re.sub(r'[\r\n\x00-\x1f\x7f-\x9f]', ' ', str(value))
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 3] + "..."
    return sanitized


def sanitize_error_message(msg: str) -> str:
    """
    Sanitize error message for use in HTTP responses.
    Removes problematic characters and extracts key information.
    """
    if not msg:
        return "An error occurred"
    
    # Check for rate limit errors and provide user-friendly message
    if "rate limit" in msg.lower() or "429" in msg or "rpm" in msg.lower():
        return "API rate limit exceeded. Please try again in a few seconds."
    
    # Remove backticks and other problematic characters
    sanitized = msg.replace('`', "'").replace('\n', ' ').replace('\r', ' ')
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', sanitized)
    # Limit length
    if len(sanitized) > 500:
        sanitized = sanitized[:500] + "..."
    
    return sanitized.strip()


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
            detail=sanitize_error_message(str(e))
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


@router.post("/voice-message/stream")
async def voice_chat_message_stream(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream voice chat processing with real-time events via SSE.
    
    This endpoint:
    1. Converts audio to text using Groq STT (emits stt_result event)
    2. Sends text to LLM for response (emits llm_token events)
    3. Converts LLM response to audio using Groq TTS (emits tts_chunk events)
    4. Returns events via Server-Sent Events (SSE)
    
    Event types:
    - stt_start: STT processing begins
    - stt_result: Transcribed text available (contains "text" field)
    - llm_start: LLM processing begins
    - llm_token: Streaming token (contains "delta" field)
    - llm_done: LLM response complete (contains "text" field)
    - tts_start: TTS processing begins
    - tts_chunk: Audio chunk available (contains "audio" base64 field)
    - tts_done: TTS processing complete
    - done: All processing complete (contains "session_id" field)
    - error: Error occurred (contains "message" field)
    
    Args:
        audio_file: Audio file (supports: FLAC, MP3, WAV, MP4, MPEG, MPGA, M4A, OGG, WEBM)
        session_id: Optional session ID for conversation continuity
        current_user: Current authenticated user (from dependency)
        
    Returns:
        StreamingResponse with SSE events
        
    Raises:
        HTTPException: If audio processing fails
    """
    container = get_container()
    voice_chat_use_case = container.get(VoiceChatUseCase)
    
    def _encode_sse(event_name: str, data: dict) -> bytes:
        """Encode event as SSE format."""
        payload = json.dumps(data, ensure_ascii=False)
        return (f"event: {event_name}\n" f"data: {payload}\n\n").encode("utf-8")
    
    async def event_generator():
        try:
            # Read audio file bytes
            audio_bytes = await audio_file.read()
            
            if not audio_bytes:
                yield _encode_sse("error", {"message": "Audio file is empty"})
                return
            
            # Stream voice chat events
            async for event in voice_chat_use_case.stream_execute(
                audio_bytes=audio_bytes,
                filename=audio_file.filename or "audio.wav",
                session_id=session_id,
                user_id=current_user.id
            ):
                ev = event.get("event") or "message"
                data = event.get("data") or {}
                yield _encode_sse(str(ev), data if isinstance(data, dict) else {"data": data})
                
        except ValueError as e:
            # Client errors (invalid format, empty text, etc.)
            yield _encode_sse("error", {"message": sanitize_error_message(str(e))})
        except Exception as e:
            # Server errors (API failures, etc.)
            error_msg = sanitize_error_message(str(e))
            yield _encode_sse("error", {"message": error_msg})
    
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
    container = get_container()
    voice_chat_use_case = container.get(VoiceChatUseCase)
    
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
        
        # Sanitize header values to prevent invalid HTTP header errors
        sanitized_text = sanitize_header_value(text_response)
        sanitized_session_id = sanitize_header_value(str(final_session_id) if final_session_id else "")
        
        # Return audio as streaming response
        return StreamingResponse(
            BytesIO(audio_response_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Text-Response": sanitized_text,  # Sanitized header value
                "X-Session-Id": sanitized_session_id   # Sanitized session ID
            }
        )
        
    except ValueError as e:
        # Client errors (invalid format, empty text, etc.)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=sanitize_error_message(str(e))
        )
    except Exception as e:
        # Server errors (API failures, etc.)
        error_msg = sanitize_error_message(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

