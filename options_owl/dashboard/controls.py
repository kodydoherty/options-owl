"""Agent controls: paper mode toggle, kill switch, restart via Redis."""

from __future__ import annotations

import os

from loguru import logger

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore


_CONTROL_PREFIX = "owl:control:"


async def _get_redis():
    """Get a Redis connection for control operations."""
    if aioredis is None:
        return None
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        r = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=5)
        await r.ping()
        return r
    except Exception as exc:
        logger.warning(f"Dashboard Redis connect failed: {exc}")
        return None


async def get_paper_mode(agent_id: str) -> bool | None:
    """Read current paper mode from Redis. None if unknown."""
    r = await _get_redis()
    if not r:
        return None
    try:
        val = await r.get(f"{_CONTROL_PREFIX}{agent_id}:paper_mode")
        return val == "true" if val is not None else None
    finally:
        await r.aclose()


async def set_paper_mode(agent_id: str, enabled: bool) -> bool:
    r = await _get_redis()
    if not r:
        return False
    try:
        await r.set(f"{_CONTROL_PREFIX}{agent_id}:paper_mode", "true" if enabled else "false")
        logger.info(f"Dashboard: set paper_mode={enabled} for {agent_id}")
        return True
    finally:
        await r.aclose()


async def get_kill_switch(agent_id: str) -> bool | None:
    r = await _get_redis()
    if not r:
        return None
    try:
        val = await r.get(f"{_CONTROL_PREFIX}{agent_id}:kill_switch")
        return val == "true" if val is not None else None
    finally:
        await r.aclose()


async def set_kill_switch(agent_id: str, enabled: bool) -> bool:
    r = await _get_redis()
    if not r:
        return False
    try:
        await r.set(f"{_CONTROL_PREFIX}{agent_id}:kill_switch", "true" if enabled else "false")
        logger.info(f"Dashboard: set kill_switch={enabled} for {agent_id}")
        return True
    finally:
        await r.aclose()


async def get_agent_heartbeat(agent_id: str) -> str | None:
    """Read last heartbeat from Redis."""
    r = await _get_redis()
    if not r:
        return None
    try:
        val = await r.get(f"owl:heartbeat:{agent_id}")
        return val
    finally:
        await r.aclose()
