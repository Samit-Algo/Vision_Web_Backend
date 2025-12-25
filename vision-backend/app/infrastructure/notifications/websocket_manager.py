"""WebSocket Manager for managing client connections and broadcasting notifications"""

import asyncio
import logging
from typing import Dict, Set, Optional
from threading import Lock
import json

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts notifications to connected clients.
    
    Thread-safe connection management with support for user-based filtering.
    """
    
    def __init__(self):
        """Initialize WebSocket manager"""
        # Map user_id -> Set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = Lock()  # Thread safety for connection management
        logger.info("WebSocketManager initialized")
    
    async def add_connection(self, user_id: str, websocket: WebSocket) -> None:
        """
        Add a WebSocket connection for a user.
        
        Args:
            user_id: User ID who owns this connection
            websocket: WebSocket connection instance
        """
        with self._lock:
            if user_id not in self._connections:
                self._connections[user_id] = set()
            self._connections[user_id].add(websocket)
        
        logger.info(f"Added WebSocket connection for user {user_id}. Total connections: {self.get_total_connections()}")
    
    async def remove_connection(self, user_id: str, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection for a user.
        
        Args:
            user_id: User ID who owns this connection
            websocket: WebSocket connection instance
        """
        with self._lock:
            if user_id in self._connections:
                self._connections[user_id].discard(websocket)
                
                # Clean up empty sets
                if not self._connections[user_id]:
                    del self._connections[user_id]
        
        logger.info(f"Removed WebSocket connection for user {user_id}. Total connections: {self.get_total_connections()}")
    
    async def send_to_user(self, user_id: str, message: dict) -> int:
        """
        Send a message to all WebSocket connections for a specific user.
        
        Args:
            user_id: User ID to send message to
            message: Message dictionary (will be JSON serialized)
            
        Returns:
            Number of connections the message was successfully sent to
        """
        # Get connections for this user (thread-safe copy)
        with self._lock:
            connections = self._connections.get(user_id, set()).copy()
        
        if not connections:
            logger.debug(f"No connections found for user {user_id}")
            return 0
        
        # Serialize message to JSON
        try:
            message_json = json.dumps(message)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize message to JSON: {e}")
            return 0
        
        # Send to all connections for this user
        sent_count = 0
        disconnected_connections = []
        
        for websocket in connections:
            try:
                await websocket.send_text(message_json)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send message to connection for user {user_id}: {e}")
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        if disconnected_connections:
            with self._lock:
                if user_id in self._connections:
                    for ws in disconnected_connections:
                        self._connections[user_id].discard(ws)
                    if not self._connections[user_id]:
                        del self._connections[user_id]
        
        logger.debug(f"Sent notification to {sent_count}/{len(connections)} connections for user {user_id}")
        return sent_count
    
    async def broadcast_to_all(self, message: dict) -> int:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message dictionary (will be JSON serialized)
            
        Returns:
            Total number of connections the message was successfully sent to
        """
        # Get all user IDs (thread-safe copy)
        with self._lock:
            user_ids = list(self._connections.keys())
        
        total_sent = 0
        for user_id in user_ids:
            sent_count = await self.send_to_user(user_id, message)
            total_sent += sent_count
        
        logger.debug(f"Broadcast message sent to {total_sent} total connections")
        return total_sent
    
    def get_connected_users(self) -> list[str]:
        """
        Get list of user IDs that have active connections.
        
        Returns:
            List of user IDs with active connections
        """
        with self._lock:
            return list(self._connections.keys())
    
    def get_total_connections(self) -> int:
        """
        Get total number of active WebSocket connections.
        
        Returns:
            Total number of connections across all users
        """
        with self._lock:
            return sum(len(connections) for connections in self._connections.values())
    
    def has_connections(self, user_id: str) -> bool:
        """
        Check if a user has any active connections.
        
        Args:
            user_id: User ID to check
            
        Returns:
            True if user has at least one active connection
        """
        with self._lock:
            return user_id in self._connections and len(self._connections[user_id]) > 0

