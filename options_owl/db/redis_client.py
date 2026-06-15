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

import json
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
        await pipe.execute()
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


# ── Real-time flow data sharing ─────────────────────────────────────────────

_PFX_FLOW = "owl:flow:"
_PFX_PRICE = "owl:price:"
_PFX_OPTION = "owl:option:"
_PFX_SNAPSHOT = "owl:snapshot:"
_PFX_SIGNAL_DATA = "owl:ml_signal:"
_PFX_CANDLE = "owl:candle:"
_FLOW_CHANNEL = "owl:flow:bars"
_SIGNAL_CHANNEL = "owl:signals"


async def publish_signal(signal_dict: dict) -> None:
    """Publish a sourcing signal to all trading bots via Redis pub/sub."""
    if _redis is None:
        return
    try:

        # Pub/sub for instant notification
        await _redis.publish(_SIGNAL_CHANNEL, json.dumps(signal_dict))
        # Also cache as latest signal for the ticker
        ticker = signal_dict.get("ticker", "")
        if ticker:
            key = f"{_PFX_SIGNAL_DATA}{ticker}"
            await _redis.set(key, json.dumps(signal_dict), ex=300)  # 5 min TTL
    except Exception as exc:
        logger.debug(f"Redis publish_signal failed: {exc}")


async def get_latest_signal(ticker: str) -> dict | None:
    """Read the latest sourcing signal for a ticker."""
    if _redis is None:
        return None
    try:

        key = f"{_PFX_SIGNAL_DATA}{ticker}"
        data = await _redis.get(key)
        return json.loads(data) if data else None
    except Exception:
        return None


async def publish_price(ticker: str, price: float) -> None:
    """Publish a stock price update. All agents can read via get_price()."""
    if _redis is None:
        return
    try:
        key = f"{_PFX_PRICE}{ticker}"
        await _redis.set(
            key,
            json.dumps({"price": price, "t": time.time()}),
            ex=120,  # 2 min TTL
        )
    except Exception as exc:
        logger.debug(f"Redis publish_price failed: {exc}")


async def get_price(ticker: str, max_age: float = 120) -> tuple[float, float] | None:
    """Read latest stock price from Redis (set by harvester).

    Returns (price, age_seconds) or None if missing/expired.
    """
    if _redis is None:
        return None
    try:
        key = f"{_PFX_PRICE}{ticker}"
        val = await _redis.get(key)
        if not val:
            return None
        data = json.loads(val)
        age = time.time() - data.get("t", 0)
        if age > max_age:
            return None
        return (data["price"], age)
    except Exception:
        return None


async def publish_spy_change(change_pct: float, open_price: float, last_price: float) -> None:
    """Publish SPY change-from-open so all bots share the same value."""
    if _redis is None:
        return
    try:
        await _redis.set(
            "owl:spy_change",
            json.dumps({
                "change_pct": change_pct,
                "open": open_price,
                "last": last_price,
                "t": time.time(),
            }),
            ex=120,
        )
    except Exception as exc:
        logger.debug(f"Redis publish_spy_change failed: {exc}")


async def get_spy_change(max_age: float = 120) -> dict | None:
    """Read SPY change-from-open published by the harvester.

    Returns {"change_pct": float, "open": float, "last": float} or None.
    """
    if _redis is None:
        return None
    try:
        val = await _redis.get("owl:spy_change")
        if not val:
            return None
        data = json.loads(val)
        age = time.time() - data.get("t", 0)
        if age > max_age:
            return None
        return data
    except Exception:
        return None


async def publish_candle_bars(ticker: str, timeframe: str, bars_json: list[dict]) -> None:
    """Publish candle bars from harvester so all bots share the same data.

    bars_json: list of {"t": timestamp_ms, "o": open, "h": high, "l": low, "c": close, "v": volume, "vw": vwap}
    """
    if _redis is None:
        return
    try:
        key = f"{_PFX_CANDLE}{ticker}:{timeframe}"
        await _redis.set(
            key,
            json.dumps({"bars": bars_json, "t": time.time()}),
            ex=300,  # 5 min TTL
        )
    except Exception as exc:
        logger.debug(f"Redis publish_candle_bars failed: {exc}")


async def get_candle_bars(ticker: str, timeframe: str, max_age: float = 300) -> list[dict] | None:
    """Read candle bars published by the harvester. Returns list of bar dicts or None."""
    if _redis is None:
        return None
    try:
        key = f"{_PFX_CANDLE}{ticker}:{timeframe}"
        val = await _redis.get(key)
        if not val:
            return None
        data = json.loads(val)
        age = time.time() - data.get("t", 0)
        if age > max_age:
            return None
        return data.get("bars")
    except Exception:
        return None


def _normalize_contract_key(contract_key: str) -> str:
    """Normalize contract key to lowercase option_type for consistent lookups."""
    parts = contract_key.split(":")
    if len(parts) >= 2:
        parts[1] = parts[1].lower()
    return ":".join(parts)


async def publish_option_premium(
    contract_key: str, bid: float, ask: float, mid: float,
) -> None:
    """Publish an option premium update. contract_key = ticker:type:strike:expiry."""
    if _redis is None:
        return
    try:
        contract_key = _normalize_contract_key(contract_key)
        key = f"{_PFX_OPTION}{contract_key}"
        await _redis.set(
            key,
            json.dumps({"bid": bid, "ask": ask, "mid": mid, "t": time.time()}),
            ex=300,  # 5 min — survives 2-3 harvester poll failures
        )
    except Exception as exc:
        logger.debug(f"Redis publish_option_premium failed: {exc}")


async def publish_option_snapshot(
    contract_key: str, snapshot: dict,
) -> None:
    """Publish a FULL option snapshot (bid/ask/greeks/volume/underlying) for ML models.

    contract_key = ticker:type:strike:expiry (e.g. NVDA:call:130:2026-05-27)
    snapshot keys: bid, ask, mid, iv, delta, gamma, theta, vega, volume,
                   open_interest, underlying_price, bid_size, ask_size
    """
    if _redis is None:
        return
    try:
        contract_key = _normalize_contract_key(contract_key)
        key = f"{_PFX_SNAPSHOT}{contract_key}"
        snapshot["t"] = time.time()
        await _redis.set(key, json.dumps(snapshot), ex=120)
    except Exception as exc:
        logger.debug(f"Redis publish_option_snapshot failed: {exc}")


async def get_option_snapshot(contract_key: str) -> dict | None:
    """Read full option snapshot from Redis (set by harvester).

    Returns dict with bid, ask, mid, iv, delta, theta, vega, volume,
    underlying_price, bid_size, ask_size, t — or None.
    """
    if _redis is None:
        return None
    try:
        contract_key = _normalize_contract_key(contract_key)
        key = f"{_PFX_SNAPSHOT}{contract_key}"
        val = await _redis.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


async def get_option_snapshots_for_ticker(
    ticker: str, expiry: str | None = None,
) -> list[dict]:
    """Read all option snapshots for a ticker (optionally filtered by expiry).

    Returns list of snapshot dicts, each including 'contract_key' for identification.
    """
    if _redis is None:
        return []
    try:
        pattern = f"{_PFX_SNAPSHOT}{ticker.upper()}:*"
        results = []
        async for key in _redis.scan_iter(match=pattern):
            if expiry:
                # key format: owl:snapshot:TICKER:type:strike:expiry
                parts = key.split(":")
                if len(parts) >= 6 and parts[5] != expiry:
                    continue
            val = await _redis.get(key)
            if val:
                snap = json.loads(val)
                # Extract contract info from key
                contract_key = key.replace(_PFX_SNAPSHOT, "")
                snap["contract_key"] = contract_key
                results.append(snap)
        return results
    except Exception as exc:
        logger.debug(f"Redis get_option_snapshots_for_ticker failed: {exc}")
        return []


async def get_option_premium(contract_key: str) -> dict | None:
    """Read latest option premium from Redis (set by harvester).

    Returns {"bid": float, "ask": float, "mid": float, "t": float} or None.
    """
    if _redis is None:
        return None
    try:
        contract_key = _normalize_contract_key(contract_key)
        key = f"{_PFX_OPTION}{contract_key}"
        val = await _redis.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


async def publish_flow_bar(bar_dict: dict) -> None:
    """Publish a 5-minute flow bar to all agents via Redis pub/sub.

    Fire-and-forget — never blocks trading if Redis is down.
    """
    if _redis is None:
        return
    try:

        await _redis.publish(_FLOW_CHANNEL, json.dumps(bar_dict))
    except Exception as exc:
        logger.debug(f"Redis publish_flow_bar failed: {exc}")


async def set_latest_flow(ticker: str, bar_dict: dict) -> None:
    """Cache the latest flow bar per ticker so new agents can read immediately."""
    if _redis is None:
        return
    try:

        key = f"{_PFX_FLOW}{ticker}"
        await _redis.set(key, json.dumps(bar_dict), ex=600)  # 10 min TTL
    except Exception as exc:
        logger.debug(f"Redis set_latest_flow failed: {exc}")


async def get_latest_flow(ticker: str) -> dict | None:
    """Read the latest flow bar for a ticker from Redis cache."""
    if _redis is None:
        return None
    try:

        key = f"{_PFX_FLOW}{ticker}"
        data = await _redis.get(key)
        return json.loads(data) if data else None
    except Exception as exc:
        logger.debug(f"Redis get_latest_flow failed: {exc}")
        return None


async def get_all_latest_flow() -> dict[str, dict]:
    """Read latest flow bars for all tickers from Redis cache."""
    if _redis is None:
        return {}
    try:

        result = {}
        async for key in _redis.scan_iter(match=f"{_PFX_FLOW}*"):
            ticker = key.replace(_PFX_FLOW, "")
            data = await _redis.get(key)
            if data:
                result[ticker] = json.loads(data)
        return result
    except Exception as exc:
        logger.debug(f"Redis get_all_latest_flow failed: {exc}")
        return {}


# ── WS Health status (published by harvester watchdog) ────────────────────


async def get_ws_health() -> dict | None:
    """Read WS health status published by the harvester watchdog.

    Returns dict with: healthy, issues, candle_ws, flow_ws, t — or None.
    """
    if _redis is None:
        return None
    try:
        val = await _redis.get("owl:ws_health")
        return json.loads(val) if val else None
    except Exception:
        return None


# ── Dashboard controls (paper mode, kill switch) ─────────────────────────

_PFX_CONTROL = "owl:control:"


async def get_control(agent_id: str, key: str) -> bool | None:
    """Read a dashboard control value. Returns True/False or None if not set."""
    if _redis is None:
        return None
    try:
        val = await _redis.get(f"{_PFX_CONTROL}{agent_id}:{key}")
        if val is None:
            return None
        return val == "true"
    except Exception:
        return None


async def set_control(agent_id: str, key: str, value: bool) -> None:
    """Write a dashboard control value."""
    if _redis is None:
        return
    try:
        await _redis.set(
            f"{_PFX_CONTROL}{agent_id}:{key}",
            "true" if value else "false",
        )
    except Exception as exc:
        logger.debug(f"Redis set_control failed: {exc}")


async def get_paper_mode(agent_id: str) -> bool | None:
    """Check if paper mode is overridden via dashboard. None = use env."""
    return await get_control(agent_id, "paper_mode")


async def get_kill_switch(agent_id: str) -> bool | None:
    """Check if kill switch is overridden via dashboard. None = use env."""
    return await get_control(agent_id, "kill_switch")
