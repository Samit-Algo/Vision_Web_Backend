"""
Tools for summarizing deployed agents and their status.
"""

import asyncio
from typing import List, Dict, Any

def get_deployed_agents_summary(user_id: str) -> Dict[str, Any]:
    """
    Get a summary of all vision agents currently deployed by the user.
    """
    from ....di.container import get_container
    from ....domain.repositories.agent_repository import AgentRepository

    container = get_container()
    agent_repo = container.get(AgentRepository)
    
    try:
        async def _fetch():
            return await agent_repo.list_by_user(user_id=user_id)
            
        agents = asyncio.run(_fetch())
        
        if not agents:
            return {
                "status": "no_agents",
                "message": "You haven't created any vision agents yet."
            }
            
        summary = []
        for agent in agents:
            summary.append({
                "name": agent.name,
                "status": agent.status,
                "rules": [r.get("rule_id") for r in agent.rules],
                "camera_id": agent.camera_id
            })
            
        return {
            "status": "success",
            "agents": summary
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
