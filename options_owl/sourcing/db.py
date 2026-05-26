"""PostgreSQL connection pool and schema management for owlet-sourcing.

Uses asyncpg for high-performance async access. Connection pool is
created once at startup and shared across the entire scan loop.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from loguru import logger

_pool: asyncpg.Pool | None = None

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Scoring audit: every ticker evaluation, pass or fail
CREATE TABLE IF NOT EXISTS scoring_audit (
    id BIGSERIAL PRIMARY KEY,
    scan_time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT,
    score_total INTEGER,
    score_direction INTEGER,
    score_timing INTEGER,
    score_amplifiers INTEGER,
    score_adjustments INTEGER,
    score_calibration INTEGER,
    reasons JSONB,
    filter_result TEXT,
    filter_reason TEXT,
    options_strike NUMERIC(10,2),
    options_premium NUMERIC(10,4),
    options_spread_pct NUMERIC(5,2),
    scan_duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Signals: emitted signals for trading bots to consume (Phase 2)
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    score INTEGER NOT NULL,
    score_breakdown JSONB NOT NULL,
    strike NUMERIC(10,2),
    expiry DATE,
    premium NUMERIC(10,4),
    spread_pct NUMERIC(5,2),
    indicators JSONB,
    alpha_sources JSONB,
    emitted_at TIMESTAMPTZ NOT NULL,
    consumed_by JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Cooldowns: per-ticker, per-direction cooldown tracking
CREATE TABLE IF NOT EXISTS cooldowns (
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    last_alert_at TIMESTAMPTZ NOT NULL,
    cooldown_minutes INTEGER NOT NULL DEFAULT 90,
    PRIMARY KEY (ticker, direction)
);

-- Alert counters: daily emission counts
CREATE TABLE IF NOT EXISTS alert_counters (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, ticker, direction)
);

-- Circuit breaker
CREATE TABLE IF NOT EXISTS circuit_breaker (
    id INTEGER PRIMARY KEY DEFAULT 1,
    is_tripped BOOLEAN NOT NULL DEFAULT FALSE,
    trip_count INTEGER NOT NULL DEFAULT 0,
    tripped_at TIMESTAMPTZ,
    reset_at TIMESTAMPTZ,
    reason TEXT
);

-- Create indexes (IF NOT EXISTS is implied by CREATE INDEX ... ON ...)
CREATE INDEX IF NOT EXISTS idx_audit_scan_time ON scoring_audit(scan_time);
CREATE INDEX IF NOT EXISTS idx_audit_ticker ON scoring_audit(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_emitted ON signals(emitted_at);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
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
        dsn = os.getenv("DATABASE_URL", "postgresql://owl:owl_dev_2026@localhost:5432/options_owl")

    logger.info(f"Connecting to PostgreSQL: {dsn.split('@')[-1]}")
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)

    # Run schema migrations
    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    logger.info("PostgreSQL pool initialized, schema ready")
    return _pool


async def get_pool() -> asyncpg.Pool:
    """Get the existing pool. Raises if init_pool() hasn't been called."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    """Gracefully close the pool on shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


async def execute(query: str, *args) -> str:
    """Execute a query (INSERT, UPDATE, DELETE)."""
    async with acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args) -> list[asyncpg.Record]:
    """Fetch multiple rows."""
    async with acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args) -> asyncpg.Record | None:
    """Fetch a single row."""
    async with acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args):
    """Fetch a single value."""
    async with acquire() as conn:
        return await conn.fetchval(query, *args)
