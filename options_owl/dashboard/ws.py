"""WebSocket manager: broadcasts trade updates to connected dashboard clients."""

from __future__ import annotations

import asyncio
import json
import os

from loguru import logger
from starlette.websockets import WebSocket

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore


class ConnectionManager:
    """Track active WebSocket connections per agent_id."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, agent_id: str):
        await websocket.accept()
        if agent_id not in self._connections:
            self._connections[agent_id] = []
        self._connections[agent_id].append(websocket)

    def disconnect(self, websocket: WebSocket, agent_id: str):
        if agent_id in self._connections:
            self._connections[agent_id] = [
                ws for ws in self._connections[agent_id] if ws is not websocket
            ]

    async def broadcast(self, agent_id: str, message: dict):
        if agent_id not in self._connections:
            return
        dead = []
        data = json.dumps(message, default=str)
        for ws in self._connections[agent_id]:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, agent_id)

    @property
    def active_count(self) -> int:
        return sum(len(v) for v in self._connections.values())


manager = ConnectionManager()


async def redis_subscriber(pool=None):
    """Subscribe to Redis trade update channels and broadcast to WebSocket clients.

    Runs as a background task. Listens on owl:trade_update:* channels.
    """
    if aioredis is None:
        logger.warning("Dashboard WS: redis package not installed, no live updates")
        return

    url = os.getenv("REDIS_URL", "redis://redis:6379/0")

    while True:
        try:
            r = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=5)
            pubsub = r.pubsub()
            await pubsub.psubscribe("owl:trade_update:*", "owl:dashboard:*")
            logger.info("Dashboard WS: subscribed to Redis trade updates")

            async for msg in pubsub.listen():
                if msg["type"] not in ("pmessage",):
                    continue

                channel = msg.get("channel", "")
                try:
                    data = json.loads(msg["data"])
                except (json.JSONDecodeError, TypeError):
                    continue

                # Extract agent_id from channel: owl:trade_update:owlet_kody
                parts = channel.split(":")
                if len(parts) >= 3:
                    agent_id = parts[-1]
                    await manager.broadcast(agent_id, data)

        except Exception as exc:
            logger.warning(f"Dashboard WS: Redis subscriber error: {exc}")
            await asyncio.sleep(5)
