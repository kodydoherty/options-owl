#!/usr/bin/env python3
"""Reconcile the Postgres mirror from each bot's SQLite source-of-truth.

The bots dual-write trades to their own SQLite (raw_messages.db, authoritative) AND to shared
Postgres (the dashboard/analytics mirror). The PG writes were fire-and-forget + UPDATE-only on
close, so any dropped write left the mirror wrong: phantom 'open' rows, missing trades, and no
agent_state. This script makes PG match SQLite exactly and is safe to re-run (idempotent upsert).

Run inside a bot container (has DATABASE_URL + the journal mount):
  docker compose exec owlet-kody python scripts/reconcile_pg_from_sqlite.py
  docker compose exec owlet-kody python scripts/reconcile_pg_from_sqlite.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone

import asyncpg

# SQLite paper_trades expr  ->  PG trades column. Only columns the dashboard/analytics use.
COLMAP = [
    ("id", "sqlite_id"), ("signal_id", "signal_id"), ("ticker", "ticker"),
    ("direction", "direction"), ("sentiment", "sentiment"), ("score", "score"),
    ("strength", "strength"), ("bot_source", "bot_source"), ("entry_price", "entry_price"),
    ("strike", "strike"), ("option_type", "option_type"), ("contracts", "contracts"),
    ("premium_per_contract", "premium_per_contract"), ("total_cost", "total_cost"),
    ("stop_price", "stop_price"), ("exit_by", "exit_by"), ("expiry_date", "expiry_date"),
    ("signal_premium", "signal_premium"), ("entry_slippage", "entry_slippage"),
    ("exit_slippage", "exit_slippage"), ("status", "status"), ("opened_at", "opened_at"),
    ("closed_at", "closed_at"), ("exit_premium", "exit_premium"), ("exit_price", "exit_price"),
    ("exit_reason", "exit_reason"), ("exit_source", "exit_source"),
    ("pnl_dollars", "pnl_dollars"), ("pnl_pct", "pnl_pct"),
    ("duration_minutes", "hold_minutes"), ("mfe_premium", "peak_premium"),
    ("mfe_pnl_pct", "peak_gain_pct"), ("webull_order_id", "webull_order_id"),
    ("webull_client_order_id", "webull_client_order_id"),
    ("webull_entry_fill_price", "webull_entry_fill_price"),
    ("webull_exit_fill_price", "webull_exit_fill_price"),
    ("parent_trade_id", "parent_trade_id"),
]
SRC = [s for s, _ in COLMAP]
PG = [p for _, p in COLMAP]
TS_COLS = {"opened_at", "closed_at"}  # need ::timestamptz cast


def _dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        f"postgresql://owl:{os.getenv('POSTGRES_PASSWORD', 'owl_dev_2026')}@postgres:5432/options_owl",
    )


def _parse_ts(v):
    """Parse an ISO string to a tz-aware UTC datetime (all DB timestamps are UTC). asyncpg
    requires a datetime object for timestamptz params — never a string."""
    if not v:
        return None
    if isinstance(v, datetime):
        dt = v
    else:
        try:
            dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _agent_id(bot_dir: str) -> str:
    # journal/owlet-kody -> owlet_kody
    return os.path.basename(bot_dir.rstrip("/")).replace("-", "_")


def _open_snapshot(db_path: str):
    """Open a read-only snapshot. WAL DBs can't open over a :ro mount (SQLite needs to write
    the -shm sidecar), so copy the (small) raw_messages.db + its -wal/-shm to a writable temp
    and read the copy. The source is never touched; the copy replays the WAL so data is current.
    Returns (connection, tempdir_to_cleanup)."""
    tmp = tempfile.mkdtemp(prefix="recon_")
    base = os.path.basename(db_path)
    dst = os.path.join(tmp, base)
    for suf in ("", "-wal", "-shm"):
        src = db_path + suf
        if os.path.exists(src):
            shutil.copy2(src, dst + suf)
    conn = sqlite3.connect(dst)
    conn.row_factory = sqlite3.Row
    return conn, tmp


async def reconcile_bot(conn, db_path: str, dry_run: bool) -> dict:
    agent_id = _agent_id(os.path.dirname(db_path))
    sq, tmp = _open_snapshot(db_path)
    has = sq.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_trades'"
    ).fetchone()
    if not has:  # e.g. the harvester DB — not a trading bot
        sq.close()
        shutil.rmtree(tmp, ignore_errors=True)
        return {"agent_id": agent_id, "skipped": True}
    rows = sq.execute(f"SELECT {', '.join(SRC)} FROM paper_trades").fetchall()

    # Build the upsert once.
    cols = ["agent_id", *PG]
    ph = []
    i = 1
    for c in cols:
        ph.append(f"${i}::timestamptz" if c in TS_COLS else f"${i}")
        i += 1
    updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in PG) + ", updated_at=NOW()"
    upsert = (
        f"INSERT INTO trades ({', '.join(cols)}) VALUES ({', '.join(ph)}) "
        f"ON CONFLICT (agent_id, sqlite_id) DO UPDATE SET {updates}"
    )

    n = 0
    for r in rows:
        vals = [agent_id]
        for src, pgc in COLMAP:
            v = r[src]
            vals.append(_parse_ts(v) if pgc in TS_COLS else v)
        if not dry_run:
            await conn.execute(upsert, *vals)
        n += 1

    # agent_state from paper_portfolio + live SQLite aggregates
    port = sq.execute(
        "SELECT current_balance FROM paper_portfolio ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    agg = sq.execute(
        "SELECT COUNT(*) FILTER (WHERE status='open') open_n, "
        "COUNT(*) FILTER (WHERE status='closed' AND pnl_dollars>0) wins, "
        "COUNT(*) FILTER (WHERE status='closed' AND pnl_dollars<=0) losses, "
        "COALESCE(SUM(pnl_dollars) FILTER (WHERE status='closed'),0) total_pnl, "
        "COALESCE(SUM(pnl_dollars) FILTER (WHERE status='closed' AND date(closed_at)=date('now')),0) daily "
        "FROM paper_trades"
    ).fetchone()
    bal = float(port["current_balance"]) if port else 0.0
    if not dry_run:
        await conn.execute(
            """INSERT INTO agent_state
                 (agent_id, portfolio_size, open_trade_count, daily_pnl, total_pnl,
                  win_count, loss_count, last_heartbeat, updated_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,NOW(),NOW())
               ON CONFLICT (agent_id) DO UPDATE SET
                 portfolio_size=$2, open_trade_count=$3, daily_pnl=$4, total_pnl=$5,
                 win_count=$6, loss_count=$7, updated_at=NOW()""",
            agent_id, bal, agg["open_n"], float(agg["daily"]), float(agg["total_pnl"]),
            agg["wins"], agg["losses"],
        )
    sq.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"agent_id": agent_id, "trades": n, "balance": bal,
            "open": agg["open_n"], "total_pnl": float(agg["total_pnl"])}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--journal", default="journal")
    args = ap.parse_args()

    conn = await asyncpg.connect(_dsn())
    try:
        # Enable the upsert key (safe — verified no dupes). Idempotent.
        if not args.dry_run:
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_trades_agent_sqlite "
                "ON trades(agent_id, sqlite_id)"
            )
        dbs = sorted(glob.glob(os.path.join(args.journal, "owlet-*", "raw_messages.db")))
        print(f"{'DRY-RUN ' if args.dry_run else ''}reconciling {len(dbs)} bot DB(s)\n")
        for db in dbs:
            try:
                res = await reconcile_bot(conn, db, args.dry_run)
                if res.get("skipped"):
                    print(f"  {res['agent_id']:14s} (skipped — no paper_trades)")
                    continue
                print(f"  {res['agent_id']:14s} trades={res['trades']:4d} "
                      f"open={res['open']:2d} total_pnl=${res['total_pnl']:>10,.2f} "
                      f"bal=${res['balance']:>10,.2f}")
            except Exception as exc:
                print(f"  {db}: ERROR {exc}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
