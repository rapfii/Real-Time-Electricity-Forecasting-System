"""
WebSocket streaming handler.

Pushes real-time forecast events and periodic metrics updates
to connected dashboard clients.

Reference: SYSTEM_DESIGN.md Section 14.2
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections for broadcast."""

    def __init__(self) -> None:
        self._active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self._active_connections.append(websocket)
        logger.info(
            "WebSocket client connected. Active connections: %d",
            len(self._active_connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        if websocket in self._active_connections:
            self._active_connections.remove(websocket)
        logger.info(
            "WebSocket client disconnected. Active connections: %d",
            len(self._active_connections),
        )

    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients.

        Disconnected clients are silently removed.
        """
        disconnected: list[WebSocket] = []
        for connection in self._active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        for ws in disconnected:
            self.disconnect(ws)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time forecast streaming.

    Sends per-event forecast messages and periodic metrics updates
    to the connected client. The client keeps the connection open
    and receives messages as they are produced by the pipeline.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; receive any client messages (ping/pong)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_text('{"type": "ping"}')
                except Exception:
                    break
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
