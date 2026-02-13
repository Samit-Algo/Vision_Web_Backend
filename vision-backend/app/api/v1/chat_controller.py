# External package imports
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import json

# Local application imports
from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.chat_with_agent import ChatWithAgentUseCase
from ...di.container import get_container
from ...agents.exceptions import ValidationError
from .dependencies import get_current_user


router = APIRouter(tags=["chat"])


@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> ChatMessageResponse:
    """
    Send a message to the agent chatbot
    
    Args:
        request: Chat message request with message and optional session_id
        current_user: Current authenticated user (from dependency)
        
    Returns:
        ChatMessageResponse with agent's response and session_id
    """
    container = get_container()
    chat_use_case = container.get(ChatWithAgentUseCase)
    
    try:
        response = await chat_use_case.execute(
            request=request,
            user_id=current_user.id
        )
        return response
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.user_message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}"
        )


@router.post("/message/stream")
async def chat_message_stream(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream a message to the agent chatbot (token streaming via SSE).

    Response is Server-Sent Events (SSE):
    - event: meta  (includes session_id)
    - event: token (delta text)
    - event: done  (final ChatMessageResponse JSON)
    - event: error (error message)
    """
    container = get_container()
    chat_use_case = container.get(ChatWithAgentUseCase)

    def _encode_sse(event_name: str, data: dict) -> bytes:
        payload = json.dumps(data, ensure_ascii=False)
        return (f"event: {event_name}\n" f"data: {payload}\n\n").encode("utf-8")

    async def event_generator():
        try:
            async for item in chat_use_case.stream_execute(request=request, user_id=current_user.id):
                ev = item.get("event") or "message"
                data = item.get("data") or {}
                yield _encode_sse(str(ev), data if isinstance(data, dict) else {"data": data})
        except ValidationError as e:
            fallback_session_id = request.session_id or "validation-error"
            yield _encode_sse("error", {"message": e.user_message})
            yield _encode_sse("done", ChatMessageResponse(
                response=e.user_message,
                session_id=fallback_session_id,
                status="error",
            ).model_dump())
        except Exception as e:
            # Last-resort safety: always emit an error event if something goes wrong in streaming layer
            yield _encode_sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
