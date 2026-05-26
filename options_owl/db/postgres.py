"""Shared PostgreSQL connection pool and schema for all trading agents.

Phase 1: Dual-write — all agents write to both SQLite (primary) and Postgres.
Phase 2: Switch reads to Postgres. Phase 3: Drop SQLite writes.

The pool is created once per process and shared. Each agent connects with
its AGENT_ID so we can track which bot placed which trade.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from loguru import logger

_pool: asyncpg.Pool | None = None

# ---------------------------------------------------------------------------
# Schema — mirrors SQLite paper_trades + trade_events
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Core trade table (mirrors paper_trades from SQLite)
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    sqlite_id INTEGER,                          -- original SQLite ID for reconciliation
    agent_id TEXT NOT NULL,                      -- which bot placed this trade
    signal_id INTEGER,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    sentiment TEXT NOT NULL DEFAULT '',
    score INTEGER NOT NULL DEFAULT 0,
    strength TEXT NOT NULL DEFAULT '',
    bot_source TEXT NOT NULL DEFAULT '',

    -- Entry
    entry_price REAL NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    premium_per_contract REAL NOT NULL,
    total_cost REAL NOT NULL,

    -- Targets
    target_1 REAL,
    target_2 REAL,
    target_3 REAL,
    target_4 REAL,
    target_5 REAL,
    stop_price REAL,
    exit_by TEXT,
    expiry_date TEXT,

    -- Signal vs actual
    signal_premium REAL,
    entry_slippage REAL,
    exit_slippage REAL,

    -- Status
    status TEXT NOT NULL DEFAULT 'open',         -- open, closed, expired
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,

    -- Exit
    exit_premium REAL,
    exit_price REAL,
    exit_reason TEXT,
    exit_source TEXT DEFAULT 'ai',               -- ai, manual
    pnl_dollars REAL,
    pnl_pct REAL,
    hold_minutes REAL,

    -- Peak tracking
    peak_premium REAL,
    peak_gain_pct REAL,
    min_premium REAL,
    max_adverse_pct REAL,

    -- Webull
    webull_order_id TEXT,
    webull_client_order_id TEXT,
    webull_entry_fill_price REAL,
    webull_exit_fill_price REAL,

    -- DCA
    parent_trade_id INTEGER,                     -- for scaleout children
    dca_count INTEGER DEFAULT 0,
    original_contracts INTEGER,
    original_premium REAL,

    -- ML signal fields
    ml_confidence REAL,
    ml_threshold REAL,
    ml_model_source TEXT,
    ml_runner_score REAL,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trade events audit log (mirrors trade_events from SQLite)
CREATE TABLE IF NOT EXISTS trade_events (
    id BIGSERIAL PRIMARY KEY,
    sqlite_id INTEGER,
    agent_id TEXT NOT NULL,
    trade_id INTEGER,                            -- references trades.sqlite_id (not PG id)
    event_type TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent account state (portfolio balance, positions)
CREATE TABLE IF NOT EXISTS agent_state (
    agent_id TEXT PRIMARY KEY,
    portfolio_size REAL NOT NULL,
    open_trade_count INTEGER NOT NULL DEFAULT 0,
    daily_pnl REAL NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0,
    win_count INTEGER NOT NULL DEFAULT 0,
    loss_count INTEGER NOT NULL DEFAULT 0,
    last_heartbeat TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Signals from sourcing scanner (consumed by trading bots)
CREATE TABLE IF NOT EXISTS ml_signals (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    score INTEGER NOT NULL,
    ml_confidence REAL,
    ml_threshold REAL,
    ml_model_source TEXT,
    ml_runner_score REAL,
    premium REAL,
    strike REAL,
    expiry_date TEXT,
    indicators JSONB,
    score_breakdown JSONB,
    emitted_at TIMESTAMPTZ NOT NULL,
    consumed_by TEXT[] DEFAULT '{}',              -- agent_ids that picked this up
    status TEXT NOT NULL DEFAULT 'pending',       -- pending, consumed, expired
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_sqlite ON trades(agent_id, sqlite_id);
CREATE INDEX IF NOT EXISTS idx_events_agent ON trade_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_trade ON trade_events(trade_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON trade_events(event_type);
CREATE INDEX IF NOT EXISTS idx_signals_status ON ml_signals(status, emitted_at);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON ml_signals(ticker);

-- Market data: stock price ticks (for backtesting, ML training, replay)
CREATE TABLE IF NOT EXISTS stock_ticks (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    price REAL NOT NULL,
    bid REAL,
    ask REAL,
    volume BIGINT,
    vwap REAL,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Market data: option contract snapshots (premium, greeks, volume)
CREATE TABLE IF NOT EXISTS option_ticks (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    option_type TEXT NOT NULL,             -- 'call' or 'put'
    strike REAL NOT NULL,
    expiry_date TEXT NOT NULL,
    bid REAL,
    ask REAL,
    mid REAL,
    last REAL,
    volume INTEGER,
    open_interest INTEGER,
    iv REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    underlying_price REAL,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Market data: stock OHLCV candles (5m aggregates for analysis)
CREATE TABLE IF NOT EXISTS stock_candles (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    timeframe TEXT NOT NULL,               -- '1m', '5m', '15m'
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume BIGINT NOT NULL,
    vwap REAL,
    bar_time TIMESTAMPTZ NOT NULL,
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (ticker, timeframe, bar_time)
);

CREATE INDEX IF NOT EXISTS idx_stock_ticks_time ON stock_ticks(ticker, captured_at);
CREATE INDEX IF NOT EXISTS idx_option_ticks_time ON option_ticks(ticker, captured_at);
CREATE INDEX IF NOT EXISTS idx_option_ticks_contract ON option_ticks(ticker, option_type, strike, expiry_date, captured_at);
CREATE INDEX IF NOT EXISTS idx_stock_candles_time ON stock_candles(ticker, timeframe, bar_time);
"""


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------


async def init_pool(dsn: str | None = None) -> asyncpg.Pool:
    """Create the connection pool and ensure schema exists.

    Call once at startup. Subsequent calls return the existing pool.
    """
    global _pool
    if _pool is not None:
        return _pool

    if dsn is None:
        dsn = os.getenv(
            "DATABASE_URL",
            "postgresql://owl:owl_dev_2026@localhost:5432/options_owl",
        )

    logger.info(f"PG: Connecting to {dsn.split('@')[-1]}")
    _pool = await asyncio.wait_for(
        asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10),
        timeout=15,
    )

    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    logger.info("PG: Pool initialized, schema ready")
    return _pool


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("PG pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PG: Pool closed")


def is_connected() -> bool:
    return _pool is not None


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def execute(query: str, *args) -> str:
    async with acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args) -> list[asyncpg.Record]:
    async with acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args) -> asyncpg.Record | None:
    async with acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args):
    async with acquire() as conn:
        return await conn.fetchval(query, *args)


# ---------------------------------------------------------------------------
# Trade write helpers (Phase 1: dual-write from paper_trader.py)
# ---------------------------------------------------------------------------


async def write_trade_open(agent_id: str, sqlite_id: int, trade_data: dict) -> int | None:
    """Write a new trade to Postgres. Returns PG trade ID or None on failure."""
    if not is_connected():
        return None

    try:
        pg_id = await fetchval(
            """
            INSERT INTO trades (
                sqlite_id, agent_id, signal_id, ticker, direction, sentiment,
                score, strength, bot_source,
                entry_price, strike, option_type, contracts,
                premium_per_contract, total_cost,
                target_1, target_2, target_3, target_4, target_5,
                stop_price, exit_by, expiry_date,
                signal_premium, status, opened_at,
                webull_order_id, webull_client_order_id, webull_entry_fill_price,
                ml_confidence, ml_threshold, ml_model_source, ml_runner_score
            ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9,
                $10, $11, $12, $13,
                $14, $15,
                $16, $17, $18, $19, $20,
                $21, $22, $23,
                $24, 'open', $25,
                $26, $27, $28,
                $29, $30, $31, $32
            )
            RETURNING id
            """,
            sqlite_id,
            agent_id,
            trade_data.get("signal_id"),
            trade_data["ticker"],
            trade_data["direction"],
            trade_data.get("sentiment", ""),
            trade_data.get("score", 0),
            trade_data.get("strength", ""),
            trade_data.get("bot_source", ""),
            trade_data["entry_price"],
            trade_data["strike"],
            trade_data["option_type"],
            trade_data["contracts"],
            trade_data["premium_per_contract"],
            trade_data["total_cost"],
            trade_data.get("target_1"),
            trade_data.get("target_2"),
            trade_data.get("target_3"),
            trade_data.get("target_4"),
            trade_data.get("target_5"),
            trade_data.get("stop_price"),
            trade_data.get("exit_by"),
            trade_data.get("expiry_date"),
            trade_data.get("signal_premium"),
            trade_data.get("opened_at"),
            trade_data.get("webull_order_id"),
            trade_data.get("webull_client_order_id"),
            trade_data.get("webull_entry_fill_price"),
            trade_data.get("ml_confidence"),
            trade_data.get("ml_threshold"),
            trade_data.get("ml_model_source"),
            trade_data.get("ml_runner_score"),
        )
        return pg_id
    except Exception as exc:
        logger.warning(f"PG write_trade_open failed: {exc}")
        return None


async def write_trade_close(
    agent_id: str,
    sqlite_id: int,
    exit_premium: float,
    exit_reason: str,
    pnl_dollars: float,
    pnl_pct: float,
    hold_minutes: float,
    exit_source: str = "ai",
    peak_premium: float | None = None,
    peak_gain_pct: float | None = None,
    webull_exit_fill_price: float | None = None,
) -> bool:
    """Update trade to closed status. Returns True on success."""
    if not is_connected():
        return False

    try:
        await execute(
            """
            UPDATE trades SET
                status = 'closed',
                closed_at = NOW(),
                exit_premium = $3,
                exit_reason = $4,
                pnl_dollars = $5,
                pnl_pct = $6,
                hold_minutes = $7,
                exit_source = $8,
                peak_premium = $9,
                peak_gain_pct = $10,
                webull_exit_fill_price = $11,
                updated_at = NOW()
            WHERE agent_id = $1 AND sqlite_id = $2
            """,
            agent_id,
            sqlite_id,
            exit_premium,
            exit_reason,
            pnl_dollars,
            pnl_pct,
            hold_minutes,
            exit_source,
            peak_premium,
            peak_gain_pct,
            webull_exit_fill_price,
        )
        return True
    except Exception as exc:
        logger.warning(f"PG write_trade_close failed: {exc}")
        return False


async def write_trade_event(
    agent_id: str, trade_id: int | None, event_type: str, details: dict
) -> int | None:
    """Write a trade event to Postgres. Returns PG event ID or None."""
    if not is_connected():
        return None

    try:
        import json
        return await fetchval(
            """
            INSERT INTO trade_events (agent_id, trade_id, event_type, details)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id
            """,
            agent_id,
            trade_id,
            event_type,
            json.dumps(details, default=str),
        )
    except Exception as exc:
        logger.warning(f"PG write_trade_event failed: {exc}")
        return None


async def update_agent_state(
    agent_id: str,
    portfolio_size: float,
    open_trade_count: int = 0,
    daily_pnl: float = 0,
    total_pnl: float = 0,
    win_count: int = 0,
    loss_count: int = 0,
) -> None:
    """Upsert agent state (portfolio balance, stats)."""
    if not is_connected():
        return

    try:
        await execute(
            """
            INSERT INTO agent_state (
                agent_id, portfolio_size, open_trade_count,
                daily_pnl, total_pnl, win_count, loss_count,
                last_heartbeat, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
            ON CONFLICT (agent_id) DO UPDATE SET
                portfolio_size = $2,
                open_trade_count = $3,
                daily_pnl = $4,
                total_pnl = $5,
                win_count = $6,
                loss_count = $7,
                last_heartbeat = NOW(),
                updated_at = NOW()
            """,
            agent_id,
            portfolio_size,
            open_trade_count,
            daily_pnl,
            total_pnl,
            win_count,
            loss_count,
        )
    except Exception as exc:
        logger.warning(f"PG update_agent_state failed: {exc}")


# ---------------------------------------------------------------------------
# Signal consumption (sourcing → trading bridge)
# ---------------------------------------------------------------------------


async def get_pending_signals(
    agent_id: str, max_age_minutes: int = 10
) -> list[dict]:
    """Fetch pending ML signals not yet consumed by this agent.

    Returns signals emitted within max_age_minutes that this agent hasn't
    picked up yet.
    """
    if not is_connected():
        return []

    try:
        rows = await fetch(
            """
            SELECT * FROM ml_signals
            WHERE status = 'pending'
              AND emitted_at > NOW() - ($2 || ' minutes')::interval
              AND NOT ($1 = ANY(consumed_by))
            ORDER BY emitted_at ASC
            LIMIT 5
            """,
            agent_id,
            str(max_age_minutes),
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning(f"PG get_pending_signals failed: {exc}")
        return []


async def mark_signal_consumed(signal_id: int, agent_id: str) -> None:
    """Mark a signal as consumed by this agent."""
    if not is_connected():
        return

    try:
        await execute(
            """
            UPDATE ml_signals
            SET consumed_by = array_append(consumed_by, $2),
                status = CASE
                    WHEN array_length(consumed_by, 1) >= 3 THEN 'consumed'
                    ELSE status
                END
            WHERE id = $1
            """,
            signal_id,
            agent_id,
        )
    except Exception as exc:
        logger.warning(f"PG mark_signal_consumed failed: {exc}")


async def emit_ml_signal(signal_data: dict) -> int | None:
    """Write an ML signal from the sourcing scanner. Returns signal ID."""
    if not is_connected():
        return None

    try:
        import json
        return await fetchval(
            """
            INSERT INTO ml_signals (
                ticker, direction, score, ml_confidence, ml_threshold,
                ml_model_source, ml_runner_score, premium, strike,
                expiry_date, indicators, score_breakdown, emitted_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb, $13
            )
            RETURNING id
            """,
            signal_data["ticker"],
            signal_data["direction"],
            signal_data.get("score", 0),
            signal_data.get("ml_confidence"),
            signal_data.get("ml_threshold"),
            signal_data.get("ml_model_source"),
            signal_data.get("ml_runner_score"),
            signal_data.get("premium"),
            signal_data.get("strike"),
            signal_data.get("expiry_date"),
            json.dumps(signal_data.get("indicators", {}), default=str),
            json.dumps(signal_data.get("score_breakdown", {}), default=str),
            signal_data.get("emitted_at"),
        )
    except Exception as exc:
        logger.warning(f"PG emit_ml_signal failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Market data capture (stock ticks, option ticks, candles)
# ---------------------------------------------------------------------------


async def write_stock_tick(
    ticker: str, price: float, bid: float = 0, ask: float = 0,
    volume: int = 0, vwap: float = 0,
) -> None:
    """Write a stock price tick for future backtesting/replay."""
    if not is_connected():
        return
    try:
        await execute(
            """
            INSERT INTO stock_ticks (ticker, price, bid, ask, volume, vwap)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            ticker, price, bid or None, ask or None, volume or None, vwap or None,
        )
    except Exception as exc:
        logger.debug(f"PG write_stock_tick failed: {exc}")


async def write_option_tick(
    ticker: str, option_type: str, strike: float, expiry_date: str,
    bid: float = 0, ask: float = 0, mid: float = 0, last: float = 0,
    volume: int = 0, open_interest: int = 0, iv: float = 0,
    delta: float = 0, gamma: float = 0, theta: float = 0, vega: float = 0,
    underlying_price: float = 0,
) -> None:
    """Write an option contract snapshot for future backtesting."""
    if not is_connected():
        return
    try:
        await execute(
            """
            INSERT INTO option_ticks (
                ticker, option_type, strike, expiry_date,
                bid, ask, mid, last, volume, open_interest,
                iv, delta, gamma, theta, vega, underlying_price
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            """,
            ticker, option_type, strike, expiry_date,
            bid or None, ask or None, mid or None, last or None,
            volume or None, open_interest or None,
            iv or None, delta or None, gamma or None, theta or None, vega or None,
            underlying_price or None,
        )
    except Exception as exc:
        logger.debug(f"PG write_option_tick failed: {exc}")


async def write_stock_candle(
    ticker: str, timeframe: str, open_: float, high: float, low: float,
    close: float, volume: int, vwap: float, bar_time,
) -> None:
    """Write a stock OHLCV candle (upsert on conflict)."""
    if not is_connected():
        return
    try:
        await execute(
            """
            INSERT INTO stock_candles (ticker, timeframe, open, high, low, close, volume, vwap, bar_time)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (ticker, timeframe, bar_time) DO UPDATE SET
                high = GREATEST(stock_candles.high, EXCLUDED.high),
                low = LEAST(stock_candles.low, EXCLUDED.low),
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                vwap = EXCLUDED.vwap
            """,
            ticker, timeframe, open_, high, low, close, volume, vwap or None, bar_time,
        )
    except Exception as exc:
        logger.debug(f"PG write_stock_candle failed: {exc}")


async def write_stock_ticks_batch(ticks: list[dict]) -> None:
    """Batch write stock ticks (more efficient for high-frequency data)."""
    if not is_connected() or not ticks:
        return
    try:
        pool = _pool
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO stock_ticks (ticker, price, bid, ask, volume, vwap)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [
                    (t["ticker"], t["price"], t.get("bid"), t.get("ask"),
                     t.get("volume"), t.get("vwap"))
                    for t in ticks
                ],
            )
    except Exception as exc:
        logger.debug(f"PG write_stock_ticks_batch failed: {exc}")


async def write_option_ticks_batch(ticks: list[dict]) -> None:
    """Batch write option ticks (more efficient for snapshots)."""
    if not is_connected() or not ticks:
        return
    try:
        pool = _pool
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO option_ticks (
                    ticker, option_type, strike, expiry_date,
                    bid, ask, mid, last, volume, open_interest,
                    iv, delta, gamma, theta, vega, underlying_price
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                """,
                [
                    (t["ticker"], t["option_type"], t["strike"], t["expiry_date"],
                     t.get("bid"), t.get("ask"), t.get("mid"), t.get("last"),
                     t.get("volume"), t.get("open_interest"),
                     t.get("iv"), t.get("delta"), t.get("gamma"),
                     t.get("theta"), t.get("vega"), t.get("underlying_price"))
                    for t in ticks
                ],
            )
    except Exception as exc:
        logger.debug(f"PG write_option_ticks_batch failed: {exc}")
