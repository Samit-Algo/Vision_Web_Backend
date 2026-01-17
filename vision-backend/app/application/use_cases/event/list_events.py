from datetime import datetime
from typing import Optional

from ....domain.repositories.event_repository import EventRepository
from ....application.dto.event_dto import EventListResponse, EventListItemResponse


class ListEventsUseCase:
    def __init__(self, event_repository: EventRepository) -> None:
        self._event_repository = event_repository

    async def execute(
        self,
        owner_user_id: Optional[str],
        start_utc: Optional[datetime],
        end_utc: Optional[datetime],
        limit: int,
        skip: int,
    ) -> EventListResponse:
        total, items = await self._event_repository.list(
            owner_user_id=owner_user_id,
            start_utc=start_utc,
            end_utc=end_utc,
            limit=limit,
            skip=skip,
        )
        return EventListResponse(
            total=total,
            items=[
                EventListItemResponse(
                    id=e.id or "",
                    session_id=e.session_id,
                    label=e.label,
                    severity=e.severity,
                    rule_index=e.rule_index,
                    camera_id=e.camera_id,
                    agent_id=e.agent_id,
                    agent_name=e.agent_name,
                    device_id=e.device_id,
                    event_ts=e.event_ts,
                    received_at=e.received_at,
                    has_image=bool(e.image_path),
                )
                for e in items
            ],
        )
