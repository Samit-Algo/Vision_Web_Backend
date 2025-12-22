"""Use case for listing agents by camera ID."""
from typing import List, Optional

from ....domain.repositories.agent_repository import AgentRepository
from ...dto.agent_dto import AgentResponse
from ....domain.models.agent import Agent


class ListAgentsByCameraUseCase:
    """Use case for listing agents for a specific camera"""
    
    def __init__(self, agent_repository: AgentRepository) -> None:
        self.agent_repository = agent_repository
    
    async def execute(self, camera_id: str, user_id: Optional[str] = None) -> List[AgentResponse]:
        """
        List all agents for a specific camera
        
        Args:
            camera_id: ID of the camera
            user_id: Optional user ID to filter agents by owner
            
        Returns:
            List of AgentResponse objects
        """
        agents = await self.agent_repository.list_by_camera_id(camera_id, user_id)
        
        return [
            AgentResponse(
                id=agent.id,
                name=agent.name,
                camera_id=agent.camera_id,
                model=agent.model,
                fps=agent.fps,
                rules=agent.rules,
                run_mode=agent.run_mode,
                interval_minutes=agent.interval_minutes,
                check_duration_seconds=agent.check_duration_seconds,
                start_time=agent.start_time,
                end_time=agent.end_time,
                zone=agent.zone,
                requires_zone=agent.requires_zone,
                status=agent.status,
                created_at=agent.created_at,
                owner_user_id=agent.owner_user_id,
                stream_config=agent.stream_config,
            )
            for agent in agents
        ]

