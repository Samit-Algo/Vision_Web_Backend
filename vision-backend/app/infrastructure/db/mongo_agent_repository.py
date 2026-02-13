from typing import Optional, List
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from bson.errors import InvalidId
from ...domain.repositories.agent_repository import AgentRepository
from ...domain.models.agent import Agent
from ...domain.constants import AgentFields
from .mongo_connection import get_agent_collection


class MongoAgentRepository(AgentRepository):
    """MongoDB implementation of AgentRepository"""
    
    def __init__(self, agent_collection: Optional[AsyncIOMotorCollection] = None) -> None:
        self.agent_collection = agent_collection if agent_collection is not None else get_agent_collection()
    
    async def find_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID"""
        if not agent_id:
            return None
        
        try:
            object_id = ObjectId(agent_id)
        except (InvalidId, ValueError, TypeError):
            return None
        
        try:
            document = await self.agent_collection.find_one({AgentFields.MONGO_ID: object_id})
            if document is None:
                return None
            return self._document_to_agent(document)
        except Exception as e:
            # Log error in production (you might want to use a logger here)
            raise RuntimeError(f"Error finding agent by ID: {str(e)}")
    
    async def list_by_user(self, user_id: str) -> List[Agent]:
        """List all agents for a user"""
        if not user_id:
            return []
        
        try:
            cursor = self.agent_collection.find({AgentFields.OWNER_USER_ID: user_id})
            agents = []
            async for document in cursor:
                agents.append(self._document_to_agent(document))
            return agents
        except Exception as e:
            # Log error in production (you might want to use a logger here)
            raise RuntimeError(f"Error listing agents for user: {str(e)}")
    
    async def list_by_camera_id(self, camera_id: str, user_id: Optional[str] = None) -> List[Agent]:
        """List all agents for a specific camera"""
        if not camera_id:
            return []
        
        try:
            # Build query filter
            query_filter = {AgentFields.CAMERA_ID: camera_id}
            
            # Optionally filter by user_id if provided
            if user_id:
                query_filter[AgentFields.OWNER_USER_ID] = user_id
            
            cursor = self.agent_collection.find(query_filter)
            agents = []
            async for document in cursor:
                agents.append(self._document_to_agent(document))
            return agents
        except Exception as e:
            # Log error in production (you might want to use a logger here)
            raise RuntimeError(f"Error listing agents for camera: {str(e)}")
    
    async def save(self, agent: Agent) -> Agent:
        """Save agent (create new or update existing)"""
        if not agent:
            raise ValueError("Agent cannot be None")
        
        try:
            agent_dict = self._agent_to_dict(agent)
            
            if agent.id:
                # Update existing agent
                try:
                    object_id = ObjectId(agent.id)
                except (InvalidId, ValueError, TypeError):
                    raise ValueError(f"Invalid agent ID format: {agent.id}")
                
                # Update the document
                update_result = await self.agent_collection.update_one(
                    {AgentFields.MONGO_ID: object_id},
                    {"$set": {k: v for k, v in agent_dict.items() if k != AgentFields.MONGO_ID}}
                )
                
                # Check if the document was actually updated
                if update_result.matched_count == 0:
                    raise ValueError(f"Agent with ID {agent.id} not found")
                
                # Fetch and return the updated document from database
                updated_document = await self.agent_collection.find_one({AgentFields.MONGO_ID: object_id})
                if updated_document is None:
                    raise RuntimeError(f"Agent {agent.id} was updated but could not be retrieved")
                
                return self._document_to_agent(updated_document)
            else:
                # Create new agent
                # Remove id from dict if it's None or invalid
                if AgentFields.MONGO_ID in agent_dict:
                    del agent_dict[AgentFields.MONGO_ID]
                
                # Insert the new document
                result = await self.agent_collection.insert_one(agent_dict)
                
                # Fetch and return the newly created document from database
                new_document = await self.agent_collection.find_one({AgentFields.MONGO_ID: result.inserted_id})
                if new_document is None:
                    raise RuntimeError("Agent was created but could not be retrieved")
                
                return self._document_to_agent(new_document)
        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Log error in production (you might want to use a logger here)
            raise RuntimeError(f"Error saving agent: {str(e)}")
    
    def _document_to_agent(self, document: dict) -> Agent:
        """Convert MongoDB document to Agent domain model"""
        if not document or AgentFields.MONGO_ID not in document:
            raise ValueError("Invalid document: missing _id field")
        
        # Convert start_time and end_time to datetime if they are strings (for backward compatibility)
        start_time = document.get(AgentFields.START_TIME)
        end_time = document.get(AgentFields.END_TIME)
        
        if start_time and isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        
        if end_time and isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        video_path = document.get(AgentFields.VIDEO_PATH, "") or ""
        source_type = document.get(AgentFields.SOURCE_TYPE, "rtsp") or "rtsp"
        try:
            return Agent(
                id=str(document[AgentFields.MONGO_ID]),
                name=document.get(AgentFields.NAME, ""),
                camera_id=document.get(AgentFields.CAMERA_ID, ""),
                model=document.get(AgentFields.MODEL, ""),
                fps=document.get(AgentFields.FPS),
                rules=document.get(AgentFields.RULES, []),
                run_mode=document.get(AgentFields.RUN_MODE),
                interval_minutes=document.get(AgentFields.INTERVAL_MINUTES),
                check_duration_seconds=document.get(AgentFields.CHECK_DURATION_SECONDS),
                start_time=start_time,
                end_time=end_time,
                zone=document.get(AgentFields.ZONE),
                requires_zone=document.get(AgentFields.REQUIRES_ZONE, False),
                status=document.get(AgentFields.STATUS, "ACTIVE"),
                created_at=document.get(AgentFields.CREATED_AT),
                owner_user_id=document.get(AgentFields.OWNER_USER_ID),
                stream_config=document.get(AgentFields.STREAM_CONFIG),
                video_path=video_path,
                source_type=source_type,
            )
        except Exception as e:
            raise ValueError(f"Error converting document to Agent: {str(e)}")
    
    def _agent_to_dict(self, agent: Agent) -> dict:
        """Convert Agent domain model to MongoDB document"""
        if not agent:
            raise ValueError("Agent cannot be None")
        
        agent_dict = {
            AgentFields.NAME: agent.name,
            AgentFields.CAMERA_ID: getattr(agent, "camera_id", "") or "",
            AgentFields.MODEL: agent.model,
            AgentFields.FPS: agent.fps,
            AgentFields.RULES: agent.rules,
            AgentFields.RUN_MODE: agent.run_mode,
            AgentFields.INTERVAL_MINUTES: agent.interval_minutes,
            AgentFields.CHECK_DURATION_SECONDS: agent.check_duration_seconds,
            AgentFields.START_TIME: agent.start_time,
            AgentFields.END_TIME: agent.end_time,
            AgentFields.ZONE: agent.zone,
            AgentFields.REQUIRES_ZONE: agent.requires_zone,
            AgentFields.STATUS: agent.status,
            AgentFields.CREATED_AT: agent.created_at,
            AgentFields.OWNER_USER_ID: agent.owner_user_id,
            AgentFields.STREAM_CONFIG: agent.stream_config,
            AgentFields.VIDEO_PATH: getattr(agent, "video_path", "") or "",
            AgentFields.SOURCE_TYPE: getattr(agent, "source_type", "rtsp") or "rtsp",
        }
        
        # Only include _id if agent.id is valid
        if agent.id:
            try:
                agent_dict[AgentFields.MONGO_ID] = ObjectId(agent.id)
            except (InvalidId, ValueError, TypeError):
                # If ID is invalid, don't include it (will create new document)
                pass
        
        return agent_dict
