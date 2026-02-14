"""
Agent chat API: send message and stream response (SSE).
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import json

# -----------------------------------------------------------------------------
# Third-party
# -----------------------------------------------------------------------------
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------
from ...agents.exceptions import ValidationError
from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.chat_with_agent import ChatWithAgentUseCase
from ...di.container import get_container

from .dependencies import get_current_user

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(tags=["chat"])


def encode_sse(event_name: str, data: dict) -> bytes:
    """Encode event name and data as Server-Sent Events format."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n".encode("utf-8")


@router.post("/message", response_model=ChatMessageResponse)
async def chat_message(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> ChatMessageResponse:
    """Send a message to the agent chatbot; returns full response."""
    container = get_container()
    use_case = container.get(ChatWithAgentUseCase)
    try:
        return await use_case.execute(request=request, user_id=current_user.id)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.user_message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat error",
        )


@router.post("/message/stream")
async def chat_message_stream(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """Stream agent response as Server-Sent Events (event: meta, token, done, error)."""
    container = get_container()
    use_case = container.get(ChatWithAgentUseCase)

    async def event_generator():
        try:
            async for item in use_case.stream_execute(request=request, user_id=current_user.id):
                ev = item.get("event") or "message"
                data = item.get("data") or {}
                yield encode_sse(str(ev), data if isinstance(data, dict) else {"data": data})
        except ValidationError as e:
            yield encode_sse("error", {"message": e.user_message})
            sid = request.session_id or "validation-error"
            yield encode_sse("done", ChatMessageResponse(
                response=e.user_message,
                session_id=sid,
                status="error",
            ).model_dump())
        except Exception as e:
            yield encode_sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
