from typing import Optional, List
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from bson.errors import InvalidId
from ...domain.repositories.agent_repository import AgentRepository
from ...domain.models.agent import Agent
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
            document = await self.agent_collection.find_one({"_id": object_id})
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
            cursor = self.agent_collection.find({"owner_user_id": user_id})
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
            query_filter = {"camera_id": camera_id}
            
            # Optionally filter by user_id if provided
            if user_id:
                query_filter["owner_user_id"] = user_id
            
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
                    {"_id": object_id},
                    {"$set": {k: v for k, v in agent_dict.items() if k != "_id"}}
                )
                
                # Check if the document was actually updated
                if update_result.matched_count == 0:
                    raise ValueError(f"Agent with ID {agent.id} not found")
                
                # Fetch and return the updated document from database
                updated_document = await self.agent_collection.find_one({"_id": object_id})
                if updated_document is None:
                    raise RuntimeError(f"Agent {agent.id} was updated but could not be retrieved")
                
                return self._document_to_agent(updated_document)
            else:
                # Create new agent
                # Remove id from dict if it's None or invalid
                if "_id" in agent_dict:
                    del agent_dict["_id"]
                
                # Insert the new document
                result = await self.agent_collection.insert_one(agent_dict)
                
                # Fetch and return the newly created document from database
                new_document = await self.agent_collection.find_one({"_id": result.inserted_id})
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
        if not document or "_id" not in document:
            raise ValueError("Invalid document: missing _id field")
        
        # Convert start_time and end_time to datetime if they are strings (for backward compatibility)
        start_time = document.get("start_time")
        end_time = document.get("end_time")
        
        if start_time and isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        
        if end_time and isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        try:
            return Agent(
                id=str(document["_id"]),
                name=document.get("name", ""),
                camera_id=document.get("camera_id", ""),
                model=document.get("model", ""),
                fps=document.get("fps"),
                rules=document.get("rules", []),
                run_mode=document.get("run_mode"),
                interval_minutes=document.get("interval_minutes"),
                check_duration_seconds=document.get("check_duration_seconds"),
                start_time=start_time,
                end_time=end_time,
                zone=document.get("zone"),
                requires_zone=document.get("requires_zone", False),
                status=document.get("status", "ACTIVE"),
                created_at=document.get("created_at"),
                owner_user_id=document.get("owner_user_id"),
                stream_config=document.get("stream_config"),
            )
        except Exception as e:
            raise ValueError(f"Error converting document to Agent: {str(e)}")
    
    def _agent_to_dict(self, agent: Agent) -> dict:
        """Convert Agent domain model to MongoDB document"""
        if not agent:
            raise ValueError("Agent cannot be None")
        
        agent_dict = {
            "name": agent.name,
            "camera_id": agent.camera_id,
            "model": agent.model,
            "fps": agent.fps,
            "rules": agent.rules,
            "run_mode": agent.run_mode,
            "interval_minutes": agent.interval_minutes,
            "check_duration_seconds": agent.check_duration_seconds,
            "start_time": agent.start_time,
            "end_time": agent.end_time,
            "zone": agent.zone,
            "requires_zone": agent.requires_zone,
            "status": agent.status,
            "created_at": agent.created_at,
            "owner_user_id": agent.owner_user_id,
            "stream_config": agent.stream_config,
        }
        
        # Only include _id if agent.id is valid
        if agent.id:
            try:
                agent_dict["_id"] = ObjectId(agent.id)
            except (InvalidId, ValueError, TypeError):
                # If ID is invalid, don't include it (will create new document)
                pass
        
        return agent_dict
