from typing import TYPE_CHECKING

from ...domain.repositories.event_repository import EventRepository
from ...application.use_cases.event.list_events import ListEventsUseCase
from ...application.use_cases.event.get_event import GetEventUseCase

if TYPE_CHECKING:
    from ..base_container import BaseContainer


class EventsProvider:
    """Events use case provider - registers all event-related use cases"""

    @staticmethod
    def register(container: "BaseContainer") -> None:
        container.register_factory(
            ListEventsUseCase,
            lambda: ListEventsUseCase(event_repository=container.get(EventRepository)),
        )

        container.register_factory(
            GetEventUseCase,
            lambda: GetEventUseCase(event_repository=container.get(EventRepository)),
        )
