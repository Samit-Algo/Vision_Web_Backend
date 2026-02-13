"""DTOs for static video analysis."""

from pydantic import BaseModel


class AskResponse(BaseModel):
    """Response for ask endpoint - just the answer."""

    answer: str
