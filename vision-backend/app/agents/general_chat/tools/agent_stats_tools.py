"""
Tools for summarizing deployed agents and their status.
"""

import asyncio
from typing import List, Dict, Any


def _run_sync(coro):
    """Run coroutine from sync code. Works inside or outside a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


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
            
        agents = _run_sync(_fetch())
        
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
