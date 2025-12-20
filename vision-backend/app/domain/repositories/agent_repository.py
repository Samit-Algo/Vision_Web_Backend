from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.agent import Agent


class AgentRepository(ABC):
    """Repository interface - defines contract for agent data access"""
    
    @abstractmethod
    async def save(self, agent: Agent) -> Agent:
        """Save agent (create or update)"""
        pass
    
    @abstractmethod
    async def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID"""
        pass
    
    @abstractmethod
    async def list_by_user(self, user_id: str) -> List[Agent]:
        """List all agents for a user"""
        pass
    
    @abstractmethod
    async def list_by_camera_id(self, camera_id: str, user_id: Optional[str] = None) -> List[Agent]:
        """List all agents for a specific camera
        
        Args:
            camera_id: ID of the camera
            user_id: Optional user ID to filter agents by owner
            
        Returns:
            List of agents for the camera
        """
        pass