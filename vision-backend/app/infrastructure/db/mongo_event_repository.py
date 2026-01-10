# Standard library imports
from datetime import datetime
from typing import Optional, Tuple, List

# External package imports
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from bson.errors import InvalidId

# Local application imports
from ...domain.repositories.event_repository import EventRepository
from ...domain.models.event import Event
from ...domain.constants import EventFields
from .mongo_connection import get_event_collection


class MongoEventRepository(EventRepository):
    """MongoDB implementation of EventRepository"""

    def __init__(self, event_collection: Optional[AsyncIOMotorCollection] = None) -> None:
        self.event_collection = event_collection if event_collection is not None else get_event_collection()

    async def create(self, event: Event) -> str:
        if not event:
            raise ValueError("Event cannot be None")

        doc = {
            EventFields.OWNER_USER_ID: event.owner_user_id,
            EventFields.SESSION_ID: event.session_id,
            EventFields.LABEL: event.label,
            EventFields.SEVERITY: event.severity,
            EventFields.RULE_INDEX: event.rule_index,
            EventFields.CAMERA_ID: event.camera_id,
            EventFields.AGENT_ID: event.agent_id,
            EventFields.AGENT_NAME: event.agent_name,
            EventFields.DEVICE_ID: event.device_id,
            EventFields.EVENT_TS: event.event_ts,
            EventFields.RECEIVED_AT: event.received_at,
            EventFields.IMAGE_PATH: event.image_path,
            EventFields.JSON_PATH: event.json_path,
            EventFields.METADATA: event.metadata or {},
        }

        result = await self.event_collection.insert_one(doc)
        return str(result.inserted_id)

    async def get_by_id(self, owner_user_id: str, event_id: str) -> Optional[Event]:
        if not event_id:
            return None
        try:
            object_id = ObjectId(event_id)
        except (InvalidId, ValueError, TypeError):
            return None

        query = {EventFields.MONGO_ID: object_id}
        # Dev-friendly fallback: if owner_user_id is missing in payload and stored as None,
        # allow access for authenticated user (matches WS broadcast fallback).
        if owner_user_id:
            query = {
                EventFields.MONGO_ID: object_id,
                "$or": [
                    {EventFields.OWNER_USER_ID: owner_user_id},
                    {EventFields.OWNER_USER_ID: None},
                    {EventFields.OWNER_USER_ID: {"$exists": False}},
                ],
            }

        doc = await self.event_collection.find_one(query)
        if not doc:
            return None
        return self._document_to_event(doc)

    async def list(
        self,
        owner_user_id: Optional[str],
        start_utc: Optional[datetime],
        end_utc: Optional[datetime],
        limit: int,
        skip: int,
    ) -> Tuple[int, List[Event]]:
        query = {}
        if owner_user_id:
            query["$or"] = [
                {EventFields.OWNER_USER_ID: owner_user_id},
                {EventFields.OWNER_USER_ID: None},
                {EventFields.OWNER_USER_ID: {"$exists": False}},
            ]

        if start_utc or end_utc:
            ts_query = {}
            if start_utc:
                ts_query["$gte"] = start_utc
            if end_utc:
                ts_query["$lt"] = end_utc
            query[EventFields.EVENT_TS] = ts_query

        total = await self.event_collection.count_documents(query)
        cursor = (
            self.event_collection.find(query)
            .sort([(EventFields.EVENT_TS, -1), (EventFields.RECEIVED_AT, -1)])
            .skip(max(0, int(skip)))
            .limit(max(1, int(limit)))
        )

        items: List[Event] = []
        async for doc in cursor:
            items.append(self._document_to_event(doc))
        return total, items

    def _document_to_event(self, doc: dict) -> Event:
        return Event(
            id=str(doc.get(EventFields.MONGO_ID)),
            owner_user_id=doc.get(EventFields.OWNER_USER_ID),
            session_id=doc.get(EventFields.SESSION_ID) or "",
            label=doc.get(EventFields.LABEL) or "Event",
            severity=doc.get(EventFields.SEVERITY) or "info",
            rule_index=doc.get(EventFields.RULE_INDEX),
            camera_id=doc.get(EventFields.CAMERA_ID),
            agent_id=doc.get(EventFields.AGENT_ID),
            agent_name=doc.get(EventFields.AGENT_NAME),
            device_id=doc.get(EventFields.DEVICE_ID),
            event_ts=doc.get(EventFields.EVENT_TS),
            received_at=doc.get(EventFields.RECEIVED_AT) or datetime.utcnow(),
            image_path=doc.get(EventFields.IMAGE_PATH),
            json_path=doc.get(EventFields.JSON_PATH),
            metadata=doc.get(EventFields.METADATA) or {},
        )
