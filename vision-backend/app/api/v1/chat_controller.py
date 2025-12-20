# External package imports
from fastapi import APIRouter, Depends, HTTPException, status

# Local application imports
from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.chat_with_agent import ChatWithAgentUseCase
from ...di.container import get_container
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}"
        )
