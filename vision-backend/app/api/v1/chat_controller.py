"""
Chat API: non-streaming message and SSE stream endpoints.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ...agents.exceptions import ValidationError, VisionAgentError
from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.chat_with_agent import ChatWithAgentUseCase
from ...di.container import get_container

from .dependencies import get_current_user

router = APIRouter(tags=["chat"])


def _sse_encode(event: str, data: dict) -> bytes:
    """Encode one SSE event (event + data)."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


# Never show technical errors to the user.
GENERIC_MESSAGE = "Please try again."


def _error_done_response(message: str, session_id: str | None, default_sid: str = "error") -> dict:
    return ChatMessageResponse(
        response=message,
        session_id=session_id or default_sid,
        status="error",
    ).model_dump()


@router.post("/message", response_model=ChatMessageResponse)
async def post_message(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> ChatMessageResponse:
    """Send a message; return full response."""
    use_case = get_container().get(ChatWithAgentUseCase)
    try:
        return await use_case.execute(request=request, user_id=current_user.id)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.user_message)
    except (VisionAgentError, Exception):
        return ChatMessageResponse(
            response=GENERIC_MESSAGE,
            session_id=request.session_id or "",
            status="error",
        )


@router.post("/message/stream")
async def post_message_stream(
    request: ChatMessageRequest,
    current_user: UserResponse = Depends(get_current_user),
) -> StreamingResponse:
    """Stream response as SSE (meta, token, done, error)."""
    use_case = get_container().get(ChatWithAgentUseCase)

    async def generate():
        try:
            async for item in use_case.stream_execute(request=request, user_id=current_user.id):
                event = item.get("event") or "message"
                data = item.get("data") or {}
                yield _sse_encode(str(event), data if isinstance(data, dict) else {"data": data})
        except ValidationError as e:
            yield _sse_encode("done", _error_done_response(e.user_message, request.session_id, "validation-error"))
        except (VisionAgentError, Exception):
            yield _sse_encode("done", _error_done_response(GENERIC_MESSAGE, request.session_id))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
