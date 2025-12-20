"""Controller for general chat API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status

from ...application.dto.chat_dto import ChatMessageRequest, ChatMessageResponse
from ...application.dto.user_dto import UserResponse
from ...application.use_cases.chat.general_chat_use_case import GeneralChatUseCase
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

