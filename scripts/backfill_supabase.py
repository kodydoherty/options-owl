"""Backfill historical trades into Supabase shared brain.

Uses Vince's RPC function to match (ticker, direction, fill_time) → alert_id,
then writes fills, closes, and execution_decisions for all historical Webull trades.

Backfill rows are tagged with source_system='main_scanner_backfill' so Vince's
ML can distinguish them from live data.

Usage:
    # Dry run (default) — shows what would be sent, no writes
    python scripts/backfill_supabase.py

    # Actually write to Supabase
    python scripts/backfill_supabase.py --live

    # Single bot only
    python scripts/backfill_supabase.py --live --bot kody
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_WEBULL_JWT = os.getenv("SUPABASE_WEBULL_JWT", "")

# RPC function name Vince created for alert matching
RPC_MATCH_ALERT = "match_alert_for_fill"

BOTS = ["kody", "adam", "vinny", "yank"]
BOT_AGENT_IDS = {
    "kody": "owlet_kody",
    "adam": "owlet_adam",
    "vinny": "owlet_vinny",
    "yank": "owlet_yank",
}
JOURNAL_BASE = Path("journal")

# Direction mapping (our DB stores "call"/"put", Vince expects "bullish"/"bearish")
DIRECTION_MAP = {"call": "bullish", "put": "bearish"}

# Exit reason mapping (same as supabase_brain.py)
EXIT_REASON_MAP = {
    "eod_cutoff": "eod",
    "eod_expiry": "eod",
    "hard_stop": "stop_loss",
    "confirmed_stop": "stop_loss",
    "graduated_stop": "stop_loss",
    "checkpoint_cut": "stop_loss",
    "backstop": "stop_loss",
    "breakeven_ratchet": "stop_loss",
    "max_trade_loss": "stop_loss",
    "profit_target": "target_hit",
    "adaptive_trail": "target_hit",
    "soft_trail": "target_hit",
    "scalp_trail": "target_hit",
    "scaleout": "partial_50",
    "theta_exit": "time_stop",
    "theta_bleed": "time_stop",
    "theta_timer": "time_stop",
    "bid_disappearance": "momentum_fade",
    "sideways_scalp": "momentum_fade",
    "velocity_exit": "target_hit",
    "dollar_trail": "target_hit",
    "max_loss_cap": "stop_loss",
    "scaleout_20": "partial_50",
    "stop_loss": "stop_loss",
    "manual": "manual",
    "signal_flip": "manual",
    "expired": "expired",
}

READ_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_WEBULL_JWT}",
    "Content-Type": "application/json",
}

WRITE_HEADERS = {
    **READ_HEADERS,
    "Prefer": "return=minimal",
}


# ---------------------------------------------------------------------------
# Data extraction from local DB
# ---------------------------------------------------------------------------

def get_trades(bot: str) -> list[dict]:
    """Read all closed Webull trades from a bot's DB."""
    db_path = JOURNAL_BASE / f"owlet-{bot}" / "raw_messages.db"
    if not db_path.exists():
        print(f"  [SKIP] {db_path} not found")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Check which columns exist (exit_source may not exist in older DBs)
    col_info = conn.execute("PRAGMA table_info(paper_trades)").fetchall()
    col_names = {row[1] for row in col_info}
    exit_source_col = "exit_source" if "exit_source" in col_names else "'ai' as exit_source"

    rows = conn.execute(
        f"SELECT id, ticker, direction, option_type, strike, "
        f"premium_per_contract, contracts, total_cost, "
        f"webull_order_id, webull_entry_fill_price, signal_premium, "
        f"opened_at, closed_at, exit_reason, {exit_source_col}, "
        f"pnl_dollars, pnl_pct, mfe_premium, score "
        f"FROM paper_trades "
        f"WHERE webull_order_id IS NOT NULL AND status = 'closed' "
        f"ORDER BY id"
    ).fetchall()
    conn.close()

    trades = [dict(r) for r in rows]
    print(f"  owlet-{bot}: {len(trades)} closed Webull trades")
    return [(bot, t) for t in trades]


# ---------------------------------------------------------------------------
# RPC call to match alert_id
# ---------------------------------------------------------------------------

async def _rpc_match(
    client: httpx.AsyncClient,
    ticker: str,
    direction: str,
    fill_time: str,
    window_min: int,
    rpc_name: str,
) -> str | None:
    """Single RPC call attempt with a given time window."""
    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/{rpc_name}",
        json={
            "p_ticker": ticker,
            "p_direction": direction,
            "p_fill_time": fill_time,
            "p_window_min": window_min,
        },
        headers=READ_HEADERS,
    )
    if resp.status_code == 200:
        result = resp.json()
        if result:
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("alert_id") or result[0].get("matched_alert_id")
            elif isinstance(result, dict):
                return result.get("alert_id") or result.get("matched_alert_id")
            elif isinstance(result, str):
                return result
        return None
    else:
        print(f"    RPC match failed: HTTP {resp.status_code} — {resp.text[:100]}")
        return None


async def match_alert(
    client: httpx.AsyncClient,
    ticker: str,
    direction: str,
    fill_time: str,
    rpc_name: str = RPC_MATCH_ALERT,
) -> str | None:
    """Match a historical trade to an alert_id. Tries 5min window, then 10min."""
    try:
        # Try narrow window first (5 minutes)
        alert_id = await _rpc_match(client, ticker, direction, fill_time, 5, rpc_name)
        if alert_id:
            return alert_id
        # Retry with wider window (10 minutes)
        return await _rpc_match(client, ticker, direction, fill_time, 10, rpc_name)
    except Exception as exc:
        print(f"    RPC match error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Write backfill data to Supabase
# ---------------------------------------------------------------------------

async def write_fill(client: httpx.AsyncClient, alert_id: str, trade: dict, bot: str) -> bool:
    """Write a fill record for a historical trade."""
    fill_price = trade["webull_entry_fill_price"] or trade["premium_per_contract"]
    signal_prem = trade["signal_premium"] or trade["premium_per_contract"]
    slippage = ((fill_price - signal_prem) / signal_prem * 100) if signal_prem > 0 else None

    agent_id = BOT_AGENT_IDS[bot]
    payload = {
        "alert_id": alert_id,
        "agent_id": agent_id,
        "broker_order_id": trade["webull_order_id"],
        "fill_time": trade["opened_at"],
        "fill_price": fill_price,
        "fill_quantity": trade["contracts"],
        "strike_filled": trade["strike"],
    }
    if slippage is not None:
        payload["slippage_pct"] = round(slippage, 2)

    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/fills",
        json=payload,
        headers=WRITE_HEADERS,
    )
    if resp.status_code in (200, 201, 409):
        return True
    print(f"    fill FAILED: HTTP {resp.status_code} — {resp.text[:100]}")
    return False


async def write_close(client: httpx.AsyncClient, alert_id: str, trade: dict, bot: str) -> bool:
    """Write a close record for a historical trade."""
    exit_reason = trade["exit_reason"] or "manual"
    mapped_reason = EXIT_REASON_MAP.get(exit_reason, "manual")

    # Calculate hold_minutes
    hold_minutes = None
    if trade["opened_at"] and trade["closed_at"]:
        try:
            opened = datetime.fromisoformat(trade["opened_at"])
            closed = datetime.fromisoformat(trade["closed_at"])
            hold_minutes = round((closed - opened).total_seconds() / 60, 1)
        except Exception:
            pass

    exit_premium = None
    if trade["pnl_dollars"] is not None and trade["contracts"] and trade["contracts"] > 0:
        entry = trade["webull_entry_fill_price"] or trade["premium_per_contract"]
        exit_premium = entry + (trade["pnl_dollars"] / (trade["contracts"] * 100))

    agent_id = BOT_AGENT_IDS[bot]
    payload = {
        "alert_id": alert_id,
        "agent_id": agent_id,
        "close_time": trade["closed_at"],
        "close_price": round(exit_premium, 4) if exit_premium else 0,
        "close_reason": mapped_reason,
    }
    if trade["pnl_pct"] is not None:
        payload["real_pnl_pct"] = round(trade["pnl_pct"], 2)
    if trade["pnl_dollars"] is not None:
        payload["real_pnl_usd"] = round(trade["pnl_dollars"], 2)
    if hold_minutes is not None:
        payload["hold_minutes"] = hold_minutes
    if trade["mfe_premium"]:
        payload["peak_premium"] = round(trade["mfe_premium"], 4)

    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/closes",
        json=payload,
        headers=WRITE_HEADERS,
    )
    if resp.status_code in (200, 201, 409):
        return True
    print(f"    close FAILED: HTTP {resp.status_code} — {resp.text[:100]}")
    return False


async def write_decision(client: httpx.AsyncClient, alert_id: str, trade: dict, bot: str) -> bool:
    """Write an execution_decision record for a historical trade."""
    agent_id = BOT_AGENT_IDS[bot]
    payload = {
        "alert_id": alert_id,
        "agent_id": agent_id,
        "decision": "executed",
        "reason": "executed_normal",
        "actual_contracts": trade["contracts"],
        "actual_strike": trade["strike"],
        "notes": f"backfill from owlet-{bot} trade#{trade['id']}",
    }
    if trade["score"]:
        payload["conviction_score"] = max(0, min(100, int(trade["score"])))

    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/execution_decisions",
        json=payload,
        headers=WRITE_HEADERS,
    )
    if resp.status_code in (200, 201, 409):
        return True
    print(f"    decision FAILED: HTTP {resp.status_code} — {resp.text[:100]}")
    return False


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

async def backfill(bots: list[str], live: bool, rpc_name: str = RPC_MATCH_ALERT) -> None:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY or not SUPABASE_WEBULL_JWT:
        print("ERROR: Missing Supabase credentials in .env")
        sys.exit(1)

    # Gather all trades
    all_trades = []
    print("Reading trades from local DBs...")
    for bot in bots:
        all_trades.extend(get_trades(bot))

    print(f"\nTotal: {len(all_trades)} trades to backfill")

    if not all_trades:
        print("Nothing to backfill.")
        return

    if not live:
        print("\n--- DRY RUN (add --live to actually write) ---\n")
        # Show sample of what would be sent
        for bot, trade in all_trades[:5]:
            direction = DIRECTION_MAP.get(trade["option_type"], "bullish")
            print(
                f"  [{bot}] #{trade['id']} {trade['ticker']} {direction} "
                f"${trade['strike']} x{trade['contracts']} @ ${trade['premium_per_contract']:.2f} "
                f"→ {trade['exit_reason']} PnL ${trade['pnl_dollars']:+.2f}"
            )
        if len(all_trades) > 5:
            print(f"  ... and {len(all_trades) - 5} more")
        print("\nRe-run with --live to write to Supabase.")
        return

    # Live mode
    print("\nStarting backfill (LIVE)...\n")
    async with httpx.AsyncClient(timeout=15.0) as client:
        matched = 0
        unmatched = 0
        fills_ok = 0
        closes_ok = 0
        decisions_ok = 0
        errors = 0

        for i, (bot, trade) in enumerate(all_trades):
            ticker = trade["ticker"]
            direction = DIRECTION_MAP.get(trade["option_type"], "bullish")
            fill_time = trade["opened_at"]

            # Step 1: Match alert_id via RPC
            alert_id = await match_alert(client, ticker, direction, fill_time, rpc_name)

            if not alert_id:
                unmatched += 1
                print(f"  [{i+1}/{len(all_trades)}] [{bot}] #{trade['id']} {ticker} — NO MATCH")
                continue

            matched += 1

            # Step 2: Write fill
            if await write_fill(client, alert_id, trade, bot):
                fills_ok += 1
            else:
                errors += 1

            # Step 3: Write close
            if await write_close(client, alert_id, trade, bot):
                closes_ok += 1
            else:
                errors += 1

            # Step 4: Write execution decision
            if await write_decision(client, alert_id, trade, bot):
                decisions_ok += 1
            else:
                errors += 1

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(all_trades)} processed...")

            # Rate limit: ~2 req/sec to be safe
            await asyncio.sleep(0.5)

        print(f"\n{'='*50}")
        print(f"BACKFILL COMPLETE")
        print(f"{'='*50}")
        print(f"  Total trades:    {len(all_trades)}")
        print(f"  Matched:         {matched}")
        print(f"  Unmatched:       {unmatched}")
        print(f"  Fills written:   {fills_ok}")
        print(f"  Closes written:  {closes_ok}")
        print(f"  Decisions written: {decisions_ok}")
        print(f"  Errors:          {errors}")


def main():
    parser = argparse.ArgumentParser(description="Backfill historical trades into Supabase")
    parser.add_argument("--live", action="store_true", help="Actually write to Supabase (default: dry run)")
    parser.add_argument("--bot", choices=BOTS, help="Backfill a single bot only")
    parser.add_argument("--rpc-function", default=RPC_MATCH_ALERT, help="Name of Vince's RPC matcher function")
    args = parser.parse_args()

    rpc_name = args.rpc_function

    bots = [args.bot] if args.bot else BOTS
    asyncio.run(backfill(bots, args.live, rpc_name))


if __name__ == "__main__":
    main()
