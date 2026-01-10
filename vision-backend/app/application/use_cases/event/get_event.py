from ....domain.repositories.event_repository import EventRepository
from ....application.dto.event_dto import EventDetailResponse


class GetEventUseCase:
    def __init__(self, event_repository: EventRepository) -> None:
        self._event_repository = event_repository

    async def execute(self, owner_user_id: str, event_id: str) -> EventDetailResponse:
        ev = await self._event_repository.get_by_id(owner_user_id=owner_user_id, event_id=event_id)
        if not ev:
            raise ValueError("Event not found")

        return EventDetailResponse(
            id=ev.id or "",
            owner_user_id=ev.owner_user_id,
            session_id=ev.session_id,
            label=ev.label,
            severity=ev.severity,
            rule_index=ev.rule_index,
            camera_id=ev.camera_id,
            agent_id=ev.agent_id,
            agent_name=ev.agent_name,
            device_id=ev.device_id,
            event_ts=ev.event_ts,
            received_at=ev.received_at,
            has_image=bool(ev.image_path),
            has_json=bool(ev.json_path),
            metadata=ev.metadata or {},
        )
