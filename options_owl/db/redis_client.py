"""Async Redis client for cross-agent coordination.

Provides fire-and-forget safe operations for:
- Regime score sharing
- Signal deduplication (prevent 4 bots entering same signal)
- Cross-agent position tracking
- Webull API rate limiting
- Daily loss tracking (cross-agent circuit breaker)

All operations gracefully degrade when Redis is unavailable — they log
warnings and return safe defaults so trading is never blocked by Redis.
"""

from __future__ import annotations

import time

from loguru import logger

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_redis: aioredis.Redis | None = None  # type: ignore[name-defined]

# Key prefixes
_PFX_REGIME = "owl:regime:"
_PFX_SIGNAL = "owl:signal:"
_PFX_POSITIONS = "owl:positions:"
_PFX_RATE = "owl:rate:"
_PFX_LOSS = "owl:loss:"

# TTLs (seconds)
_TTL_DAILY = 86400  # 24 hours
_TTL_SIGNAL = 300  # 5 minutes
_TTL_POSITION = 3600  # 1 hour


# ── Connection management ──────────────────────────────────────────────────


async def init_redis(url: str = "redis://redis:6379/0") -> None:
    """Initialise the module-level Redis connection."""
    global _redis

    if aioredis is None:
        logger.warning("redis package not installed — Redis features disabled")
        return

    try:
        _redis = aioredis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        # Verify connectivity
        await _redis.ping()
        logger.info(f"Redis connected: {url}")
    except Exception as exc:
        logger.warning(f"Redis connection failed ({exc}) — operating without Redis")
        _redis = None


def is_connected() -> bool:
    """Return True if Redis client is initialised (may still lose connection later)."""
    return _redis is not None


async def close() -> None:
    """Gracefully close the Redis connection."""
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception as exc:
            logger.warning(f"Redis close error: {exc}")
        finally:
            _redis = None
            logger.info("Redis connection closed")


# ── Cross-agent regime sharing ─────────────────────────────────────────────


async def set_regime_score(date_str: str, score: float, skip: bool) -> None:
    """Store today's regime score so all agents share the same decision."""
    if _redis is None:
        return
    try:
        key = f"{_PFX_REGIME}{date_str}"
        await _redis.hset(key, mapping={"score": str(score), "skip": str(int(skip))})
        await _redis.expire(key, _TTL_DAILY)
    except Exception as exc:
        logger.warning(f"Redis set_regime_score failed: {exc}")


async def get_regime_decision(date_str: str) -> dict | None:
    """Retrieve the regime decision for a date. Returns {score, skip} or None."""
    if _redis is None:
        return None
    try:
        key = f"{_PFX_REGIME}{date_str}"
        data = await _redis.hgetall(key)
        if not data:
            return None
        return {
            "score": float(data["score"]),
            "skip": bool(int(data["skip"])),
        }
    except Exception as exc:
        logger.warning(f"Redis get_regime_decision failed: {exc}")
        return None


# ── Signal deduplication ───────────────────────────────────────────────────


async def try_claim_signal(signal_id: str, agent_id: str, ttl: int = 300) -> bool:
    """Atomically claim a signal for an agent (SET NX). Returns True if claimed."""
    if _redis is None:
        return True  # no Redis = every agent trades independently
    try:
        key = f"{_PFX_SIGNAL}{signal_id}"
        result = await _redis.set(key, agent_id, nx=True, ex=ttl)
        return result is not None
    except Exception as exc:
        logger.warning(f"Redis try_claim_signal failed: {exc}")
        return True  # fail-open: allow trade if Redis is down


# ── Cross-agent position tracking ─────────────────────────────────────────


async def get_total_open_positions() -> int:
    """Return the total number of open positions across all agents."""
    if _redis is None:
        return 0
    try:
        keys = []
        async for key in _redis.scan_iter(match=f"{_PFX_POSITIONS}*"):
            keys.append(key)
        return len(keys)
    except Exception as exc:
        logger.warning(f"Redis get_total_open_positions failed: {exc}")
        return 0


async def register_position(agent_id: str, ticker: str, direction: str) -> None:
    """Register an open position so other agents can see it."""
    if _redis is None:
        return
    try:
        key = f"{_PFX_POSITIONS}{agent_id}:{ticker}"
        await _redis.set(key, direction, ex=_TTL_POSITION)
    except Exception as exc:
        logger.warning(f"Redis register_position failed: {exc}")


async def unregister_position(agent_id: str, ticker: str) -> None:
    """Remove a closed position from the shared tracker."""
    if _redis is None:
        return
    try:
        key = f"{_PFX_POSITIONS}{agent_id}:{ticker}"
        await _redis.delete(key)
    except Exception as exc:
        logger.warning(f"Redis unregister_position failed: {exc}")


async def get_index_positions() -> list[str]:
    """Return list of index tickers currently held across all agents."""
    if _redis is None:
        return []
    try:
        index_tickers = {"SPY", "QQQ", "IWM", "DIA", "XLF", "XLK"}
        held: list[str] = []
        async for key in _redis.scan_iter(match=f"{_PFX_POSITIONS}*"):
            # key format: owl:positions:{agent_id}:{ticker}
            parts = key.split(":")
            if len(parts) >= 4:
                ticker = parts[3]
                if ticker in index_tickers and ticker not in held:
                    held.append(ticker)
        return held
    except Exception as exc:
        logger.warning(f"Redis get_index_positions failed: {exc}")
        return []


# ── Rate limiting (Webull API) ─────────────────────────────────────────────


async def check_rate_limit(agent_id: str, max_per_minute: int = 10) -> bool:
    """Return True if agent is within rate limit, False if should back off."""
    if _redis is None:
        return True  # no Redis = no rate limiting
    try:
        key = f"{_PFX_RATE}{agent_id}"
        count = await _redis.get(key)
        if count is None:
            return True
        return int(count) < max_per_minute
    except Exception as exc:
        logger.warning(f"Redis check_rate_limit failed: {exc}")
        return True  # fail-open


async def increment_api_call(agent_id: str) -> None:
    """Record an API call for rate limiting. Counter expires after 60s."""
    if _redis is None:
        return
    try:
        key = f"{_PFX_RATE}{agent_id}"
        pipe = _redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)  # 1 minute sliding window
        await pipe.execute()
    except Exception as exc:
        logger.warning(f"Redis increment_api_call failed: {exc}")


# ── Daily loss tracking (cross-agent circuit breaker) ──────────────────────


async def add_daily_loss(agent_id: str, amount: float) -> float:
    """Add a loss amount for today. Returns the total daily loss across all agents."""
    if _redis is None:
        return 0.0
    try:
        today = time.strftime("%Y-%m-%d")
        key = f"{_PFX_LOSS}{today}"
        field = agent_id
        # Increment agent's loss contribution
        pipe = _redis.pipeline()
        pipe.hincrbyfloat(key, field, amount)
        pipe.expire(key, _TTL_DAILY)
        results = await pipe.execute()
        # Return total across all agents
        return await get_total_daily_loss()
    except Exception as exc:
        logger.warning(f"Redis add_daily_loss failed: {exc}")
        return 0.0


async def get_total_daily_loss() -> float:
    """Get the total daily loss across all agents for today."""
    if _redis is None:
        return 0.0
    try:
        today = time.strftime("%Y-%m-%d")
        key = f"{_PFX_LOSS}{today}"
        data = await _redis.hgetall(key)
        if not data:
            return 0.0
        return sum(float(v) for v in data.values())
    except Exception as exc:
        logger.warning(f"Redis get_total_daily_loss failed: {exc}")
        return 0.0


async def reset_daily_losses() -> None:
    """Reset today's daily loss counters (e.g., manual override)."""
    if _redis is None:
        return
    try:
        today = time.strftime("%Y-%m-%d")
        key = f"{_PFX_LOSS}{today}"
        await _redis.delete(key)
    except Exception as exc:
        logger.warning(f"Redis reset_daily_losses failed: {exc}")
