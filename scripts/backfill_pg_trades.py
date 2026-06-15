#!/usr/bin/env python3
"""Backfill PostgreSQL with recent trade data from all 4 bots' SQLite databases.

Run on the droplet where both SQLite files and PG are accessible:

    python scripts/backfill_pg_trades.py              # last 2 weeks
    python scripts/backfill_pg_trades.py --days 30    # last 30 days
    python scripts/backfill_pg_trades.py --dry-run    # preview without writing
    python scripts/backfill_pg_trades.py --bot kody   # single bot only

Expects:
    - SQLite DBs at journal/owlet-{name}/raw_messages.db
    - PostgreSQL reachable (default: postgresql://owl:owl_dev_2026@postgres:5432/options_owl)
    - Override PG DSN via DATABASE_URL env var
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import asyncpg

# ---------------------------------------------------------------------------
# Schema (imported inline to avoid needing the full options_owl package)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    sqlite_id INTEGER,
    agent_id TEXT NOT NULL,
    signal_id INTEGER,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    sentiment TEXT NOT NULL DEFAULT '',
    score INTEGER NOT NULL DEFAULT 0,
    strength TEXT NOT NULL DEFAULT '',
    bot_source TEXT NOT NULL DEFAULT '',
    entry_price REAL NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    premium_per_contract REAL NOT NULL,
    total_cost REAL NOT NULL,
    target_1 REAL,
    target_2 REAL,
    target_3 REAL,
    target_4 REAL,
    target_5 REAL,
    stop_price REAL,
    exit_by TEXT,
    expiry_date TEXT,
    signal_premium REAL,
    entry_slippage REAL,
    exit_slippage REAL,
    status TEXT NOT NULL DEFAULT 'open',
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    exit_premium REAL,
    exit_price REAL,
    exit_reason TEXT,
    exit_source TEXT DEFAULT 'ai',
    pnl_dollars REAL,
    pnl_pct REAL,
    hold_minutes REAL,
    peak_premium REAL,
    peak_gain_pct REAL,
    min_premium REAL,
    max_adverse_pct REAL,
    webull_order_id TEXT,
    webull_client_order_id TEXT,
    webull_entry_fill_price REAL,
    webull_exit_fill_price REAL,
    parent_trade_id INTEGER,
    dca_count INTEGER DEFAULT 0,
    original_contracts INTEGER,
    original_premium REAL,
    ml_confidence REAL,
    ml_threshold REAL,
    ml_model_source TEXT,
    ml_runner_score REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trade_events (
    id BIGSERIAL PRIMARY KEY,
    sqlite_id INTEGER,
    agent_id TEXT NOT NULL,
    trade_id INTEGER,
    event_type TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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
    consumed_by TEXT[] DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_agent ON trades(agent_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_opened ON trades(opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_sqlite ON trades(agent_id, sqlite_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_upsert ON trades(agent_id, sqlite_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_upsert ON trade_events(agent_id, sqlite_id);
CREATE INDEX IF NOT EXISTS idx_events_agent ON trade_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_events_trade ON trade_events(trade_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON trade_events(event_type);
CREATE INDEX IF NOT EXISTS idx_signals_status ON ml_signals(status, emitted_at);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON ml_signals(ticker);
"""

BOTS = ["kody", "adam", "vinny", "yank"]

DEFAULT_DSN = "postgresql://owl:owl_dev_2026@postgres:5432/options_owl"


def _parse_sqlite_ts(ts_str: str | None) -> datetime | None:
    """Parse a SQLite ISO timestamp string to a timezone-aware UTC datetime."""
    if not ts_str:
        return None
    # SQLite stores naive ISO strings — they are UTC per CLAUDE.md
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Last resort — try fromisoformat (Python 3.11+)
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create tables and indexes if they don't exist."""
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
    print("[schema] PostgreSQL schema ensured")


async def backfill_trades(
    pool: asyncpg.Pool,
    agent_id: str,
    db_path: str,
    cutoff: str,
    dry_run: bool,
) -> int:
    """Read trades from SQLite and upsert into PG. Returns count of rows synced."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM paper_trades WHERE opened_at >= ? ORDER BY id",
            (cutoff,),
        )
        rows = await cursor.fetchall()

    if not rows:
        return 0

    count = 0
    async with pool.acquire() as conn:
        for row in rows:
            r = dict(row)
            sqlite_id = r["id"]

            # Map SQLite columns to PG columns
            # SQLite mfe_premium/mae_premium -> PG peak_premium/min_premium
            # SQLite mfe_pnl_pct/mae_pnl_pct -> PG peak_gain_pct/max_adverse_pct
            # SQLite duration_minutes -> PG hold_minutes
            opened_at = _parse_sqlite_ts(r.get("opened_at"))
            closed_at = _parse_sqlite_ts(r.get("closed_at"))

            if opened_at is None:
                print(f"  [skip] trade #{sqlite_id}: unparseable opened_at={r.get('opened_at')!r}")
                continue

            if dry_run:
                count += 1
                continue

            await conn.execute(
                """
                INSERT INTO trades (
                    sqlite_id, agent_id, signal_id, ticker, direction, sentiment,
                    score, strength, bot_source,
                    entry_price, strike, option_type, contracts,
                    premium_per_contract, total_cost,
                    target_1, target_2, target_3, target_4, target_5,
                    stop_price, exit_by, expiry_date,
                    signal_premium, entry_slippage, exit_slippage,
                    status, opened_at, closed_at,
                    exit_premium, exit_price, exit_reason, exit_source,
                    pnl_dollars, pnl_pct, hold_minutes,
                    peak_premium, peak_gain_pct, min_premium, max_adverse_pct,
                    webull_order_id, webull_client_order_id,
                    webull_entry_fill_price, webull_exit_fill_price,
                    parent_trade_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9,
                    $10, $11, $12, $13,
                    $14, $15,
                    $16, $17, $18, $19, $20,
                    $21, $22, $23,
                    $24, $25, $26,
                    $27, $28, $29,
                    $30, $31, $32, $33,
                    $34, $35, $36,
                    $37, $38, $39, $40,
                    $41, $42,
                    $43, $44,
                    $45
                )
                ON CONFLICT (agent_id, sqlite_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    closed_at = EXCLUDED.closed_at,
                    exit_premium = EXCLUDED.exit_premium,
                    exit_price = EXCLUDED.exit_price,
                    exit_reason = EXCLUDED.exit_reason,
                    exit_source = EXCLUDED.exit_source,
                    pnl_dollars = EXCLUDED.pnl_dollars,
                    pnl_pct = EXCLUDED.pnl_pct,
                    hold_minutes = EXCLUDED.hold_minutes,
                    peak_premium = EXCLUDED.peak_premium,
                    peak_gain_pct = EXCLUDED.peak_gain_pct,
                    min_premium = EXCLUDED.min_premium,
                    max_adverse_pct = EXCLUDED.max_adverse_pct,
                    webull_order_id = EXCLUDED.webull_order_id,
                    webull_client_order_id = EXCLUDED.webull_client_order_id,
                    webull_entry_fill_price = EXCLUDED.webull_entry_fill_price,
                    webull_exit_fill_price = EXCLUDED.webull_exit_fill_price,
                    contracts = EXCLUDED.contracts,
                    premium_per_contract = EXCLUDED.premium_per_contract,
                    total_cost = EXCLUDED.total_cost,
                    updated_at = NOW()
                """,
                sqlite_id,                                          # $1
                agent_id,                                           # $2
                r.get("signal_id"),                                 # $3
                r["ticker"],                                        # $4
                r["direction"],                                     # $5
                r.get("sentiment", ""),                             # $6
                r.get("score", 0),                                  # $7
                r.get("strength", ""),                              # $8
                r.get("bot_source", ""),                             # $9
                r["entry_price"],                                   # $10
                r["strike"],                                        # $11
                r["option_type"],                                   # $12
                r["contracts"],                                     # $13
                r["premium_per_contract"],                          # $14
                r["total_cost"],                                    # $15
                r.get("target_1"),                                  # $16
                r.get("target_2"),                                  # $17
                r.get("target_3"),                                  # $18
                r.get("target_4"),                                  # $19
                r.get("target_5"),                                  # $20
                r.get("stop_price"),                                # $21
                r.get("exit_by"),                                   # $22
                r.get("expiry_date"),                               # $23
                r.get("signal_premium"),                            # $24
                r.get("entry_slippage"),                            # $25
                r.get("exit_slippage"),                              # $26
                r.get("status", "open"),                            # $27
                opened_at,                                          # $28
                closed_at,                                          # $29
                r.get("exit_premium"),                              # $30
                r.get("exit_price"),                                # $31
                r.get("exit_reason"),                               # $32
                r.get("exit_source", "ai"),                         # $33
                r.get("pnl_dollars"),                               # $34
                r.get("pnl_pct"),                                   # $35
                r.get("duration_minutes"),                          # $36 -> hold_minutes
                r.get("mfe_premium"),                               # $37 -> peak_premium
                r.get("mfe_pnl_pct"),                               # $38 -> peak_gain_pct
                r.get("mae_premium"),                               # $39 -> min_premium
                r.get("mae_pnl_pct"),                               # $40 -> max_adverse_pct
                r.get("webull_order_id"),                           # $41
                r.get("webull_client_order_id"),                    # $42
                r.get("webull_entry_fill_price"),                   # $43
                r.get("webull_exit_fill_price"),                    # $44
                r.get("parent_trade_id"),                           # $45
            )
            count += 1

    return count


async def backfill_events(
    pool: asyncpg.Pool,
    agent_id: str,
    db_path: str,
    cutoff: str,
    dry_run: bool,
) -> int:
    """Read trade_events from SQLite and insert into PG. Returns count."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM trade_events WHERE created_at >= ? ORDER BY id",
            (cutoff,),
        )
        rows = await cursor.fetchall()

    if not rows:
        return 0

    if dry_run:
        return len(rows)

    count = 0
    async with pool.acquire() as conn:
        # Batch insert for performance — build list of tuples
        records = []
        for row in rows:
            r = dict(row)
            created_at = _parse_sqlite_ts(r.get("created_at"))
            if created_at is None:
                continue

            # SQLite trade_events has: id, trade_id, ticker, event_type, detail, created_at
            # PG trade_events has: id, sqlite_id, agent_id, trade_id, event_type, details (JSONB), created_at
            # Pack ticker + detail into JSONB details
            detail_str = r.get("detail", "")
            try:
                detail_obj = json.loads(detail_str) if detail_str else {}
            except (json.JSONDecodeError, TypeError):
                detail_obj = {"raw": detail_str}

            # Always include ticker in details
            detail_obj["ticker"] = r.get("ticker", "")

            records.append((
                r["id"],            # sqlite_id
                agent_id,           # agent_id
                r.get("trade_id"),  # trade_id
                r["event_type"],    # event_type
                json.dumps(detail_obj, default=str),  # details
                created_at,         # created_at
            ))

        # Use executemany via a prepared statement approach — batch in chunks
        for i in range(0, len(records), 500):
            batch = records[i:i + 500]
            await conn.executemany(
                """
                INSERT INTO trade_events (sqlite_id, agent_id, trade_id, event_type, details, created_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                ON CONFLICT (agent_id, sqlite_id) DO NOTHING
                """,
                batch,
            )
            count += len(batch)

    return count


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill PG from SQLite trade data")
    parser.add_argument("--days", type=int, default=14, help="How many days back to backfill (default: 14)")
    parser.add_argument("--dry-run", action="store_true", help="Preview counts without writing to PG")
    parser.add_argument("--bot", type=str, default=None, help="Single bot name (kody/adam/vinny/yank)")
    parser.add_argument("--dsn", type=str, default=None, help="PostgreSQL DSN (overrides DATABASE_URL)")
    args = parser.parse_args()

    dsn = args.dsn or os.getenv("DATABASE_URL", DEFAULT_DSN)
    bots = [args.bot] if args.bot else BOTS
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=args.days)
    cutoff = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Resolve project root (script lives in scripts/)
    project_root = Path(__file__).resolve().parent.parent

    print(f"Backfill: last {args.days} days (since {cutoff})")
    print(f"PG DSN: {dsn.split('@')[-1]}")
    print(f"Bots: {', '.join(bots)}")
    if args.dry_run:
        print("[DRY RUN] No writes will be performed\n")
    else:
        print()

    # Verify SQLite files exist
    db_paths: dict[str, str] = {}
    for bot in bots:
        db_file = project_root / "journal" / f"owlet-{bot}" / "raw_messages.db"
        if not db_file.exists():
            print(f"  [warn] {db_file} does not exist, skipping {bot}")
            continue
        db_paths[bot] = str(db_file)

    if not db_paths:
        print("No SQLite databases found. Are you running on the droplet?")
        sys.exit(1)

    # Connect to PG
    try:
        pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
    except Exception as exc:
        print(f"Failed to connect to PostgreSQL: {exc}")
        sys.exit(1)

    try:
        # Ensure schema
        if not args.dry_run:
            await ensure_schema(pool)
        else:
            print("[schema] Skipped (dry run)\n")

        # Backfill each bot
        total_trades = 0
        total_events = 0

        for bot, db_path in db_paths.items():
            agent_id = f"owlet-{bot}"
            print(f"--- {agent_id} ({db_path}) ---")

            trade_count = await backfill_trades(pool, agent_id, db_path, cutoff, args.dry_run)
            event_count = await backfill_events(pool, agent_id, db_path, cutoff, args.dry_run)

            action = "would sync" if args.dry_run else "synced"
            print(f"  trades: {action} {trade_count}")
            print(f"  events: {action} {event_count}")
            print()

            total_trades += trade_count
            total_events += event_count

        print("=" * 50)
        label = "Would sync" if args.dry_run else "Synced"
        print(f"{label} {total_trades} trades, {total_events} events across {len(db_paths)} bots")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
