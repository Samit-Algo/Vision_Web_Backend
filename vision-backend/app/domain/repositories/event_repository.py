from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple, List

from ..models.event import Event


class EventRepository(ABC):
    """Repository interface - defines contract for event data access"""

    @abstractmethod
    async def create(self, event: Event) -> str:
        """Create event and return event_id"""
        pass

    @abstractmethod
    async def get_by_id(self, owner_user_id: str, event_id: str) -> Optional[Event]:
        """Get event by ID (scoped to user)"""
        pass

    @abstractmethod
    async def list(
        self,
        owner_user_id: Optional[str],
        start_utc: Optional[datetime],
        end_utc: Optional[datetime],
        limit: int,
        skip: int,
    ) -> Tuple[int, List[Event]]:
        """List events for a user within time range, returning (total, items)"""
        pass
