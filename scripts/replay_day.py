#!/usr/bin/env python3
"""Replay a day's raw Discord messages through the fixed parser + paper trader.

Usage:
    python scripts/replay_day.py                    # replay today
    python scripts/replay_day.py --date 2026-04-09  # replay specific date
    python scripts/replay_day.py --date 2026-04-08  # yesterday

Reads raw_messages from the live DB, re-parses each through the current
(fixed) parser, and runs them through evaluate_and_trade() with a fresh
temporary paper_trades table. Prints a side-by-side comparison of old
(recorded) trades vs what the fixed code would have produced.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiosqlite
from loguru import logger

from options_owl.collectors.discord_collector import parse_trade_signal
from options_owl.config.settings import Settings
from options_owl.execution.paper_trader import PaperTrader, _select_trade_premium


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def get_raw_messages(db_path: str, date: str) -> list[dict]:
    """Fetch all raw messages for a given date."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        rows = await conn.execute_fetchall(
            "SELECT id, author_name, content, timestamp "
            "FROM raw_messages WHERE date(timestamp) = ? ORDER BY id",
            (date,),
        )
        return [dict(r) for r in rows]


async def get_old_trades(db_path: str, date: str) -> list[dict]:
    """Fetch paper trades that were opened on the given date."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        rows = await conn.execute_fetchall(
            "SELECT id, signal_id, ticker, direction, score, contracts, "
            "premium_per_contract, total_cost, strike, status, exit_reason, "
            "pnl_dollars, pnl_pct, strategy "
            "FROM paper_trades WHERE date(opened_at) = ? ORDER BY id",
            (date,),
        )
        return [dict(r) for r in rows]


def bot_name(author: str) -> str:
    """Extract short bot name from Discord author."""
    name = author.split("#")[0].strip()
    # Remove emoji
    for ch in "💸🛎🗡💤":
        name = name.replace(ch, "")
    return name.strip()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

async def replay(date: str) -> None:
    settings = Settings()
    live_db = settings.DB_PATH

    # Fetch raw messages and old trades from live DB
    messages = await get_raw_messages(live_db, date)
    old_trades = await get_old_trades(live_db, date)

    signal_messages = []
    for msg in messages:
        sig = parse_trade_signal(
            msg["content"],
            message_id=msg["id"],
            channel="replay",
            author=msg["author_name"],
        )
        if sig is not None:
            signal_messages.append((msg, sig))

    print(f"\n{'='*80}")
    print(f"  REPLAY: {date} — {len(messages)} messages, {len(signal_messages)} signals parsed")
    print(f"{'='*80}\n")

    # --- Part 1: Show parser results for each signal ---
    print("PARSED SIGNALS (fixed parser):")
    print(f"{'#':<4} {'Time':<8} {'Bot':<16} {'Ticker':<6} {'Dir':<5} {'Score':<6} "
          f"{'Strike':<8} {'ATM$':<8} {'OTM$':<8} {'ATM_K':<8} {'OTM_K':<8}")
    print("-" * 95)

    for msg, sig in signal_messages:
        ts = msg["timestamp"]
        try:
            t = datetime.fromisoformat(ts).strftime("%H:%M")
        except (ValueError, TypeError):
            t = "??:??"

        adjusted = _select_trade_premium(sig)

        print(f"{msg['id']:<4} {t:<8} {bot_name(msg['author_name']):<16} "
              f"{sig.ticker:<6} {sig.direction.value:<5} {sig.score:<6} "
              f"${sig.strike:<7.1f} ${sig.atm_premium or 0:<7.2f} "
              f"${sig.otm_premium or 0:<7.2f} "
              f"${sig.atm_strike or 0:<7.1f} ${sig.otm_strike or 0:<7.1f}")

    # --- Part 2: Run through paper trader with fresh DB ---
    print(f"\n\nSIMULATED TRADES (fixed parser + paper trader):")
    print("-" * 95)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_db = tmp.name

    replay_settings = Settings(
        DB_PATH=tmp_db,
        PAPER_TRADE=True,
        ANTI_CHASE_MAX_MOVE_PCT=99.0,  # don't reject stale prices in replay
        ENABLE_SMART_ENTRY=False,       # no live quotes in replay
    )
    trader = PaperTrader(replay_settings)
    await trader.init()

    new_trades = []
    skipped = []
    no_premium = []

    for msg, sig in signal_messages:
        # Skip signals with no premium data — can't trade without a price
        has_premium = (sig.atm_premium and sig.atm_premium > 0) or (
            sig.otm_premium and sig.otm_premium > 0
        )
        if not has_premium:
            no_premium.append(sig.ticker)
            continue

        result = await trader.evaluate_and_trade(sig, signal_id=msg["id"])
        if result is not None:
            new_trades.append(result)
        else:
            skipped.append(f"{sig.ticker}(#{msg['id']})")

    print(f"{'#':<4} {'Ticker':<6} {'Type':<5} {'Contracts':<10} "
          f"{'Premium':<10} {'Cost':<10} {'Strike':<8} {'Strategy'}")
    print("-" * 80)

    for t in new_trades:
        print(f"{t['trade_id']:<4} {t['ticker']:<6} {t.get('option_type', '?'):<5} "
              f"{t['contracts']:<10} "
              f"${t['premium']:<9.2f} ${t['total_cost']:<9.2f} "
              f"${t['strike']:<7.1f} {t.get('strategy', '?')}")

    if no_premium:
        print(f"\nNo premium in message ({len(no_premium)}): {', '.join(no_premium)}")
    if skipped:
        print(f"Rejected by pipeline ({len(skipped)}): {', '.join(skipped)}")

    # --- Part 3: Compare old vs new ---
    print(f"\n\n{'='*80}")
    print("  COMPARISON: Old (recorded) vs New (fixed)")
    print(f"{'='*80}\n")

    # Group old trades by ticker (some have multiple partials)
    old_by_ticker: dict[str, list[dict]] = {}
    for t in old_trades:
        old_by_ticker.setdefault(t["ticker"], []).append(t)

    new_by_ticker: dict[str, dict] = {}
    for t in new_trades:
        new_by_ticker[t["ticker"]] = t

    all_tickers = sorted(set(list(old_by_ticker.keys()) + list(new_by_ticker.keys())))

    print(f"{'Ticker':<6} {'Old Cts':<8} {'Old Prem':<10} {'Old Cost':<10} "
          f"{'New Cts':<8} {'New Prem':<10} {'New Cost':<10} {'Diff'}")
    print("-" * 85)

    for ticker in all_tickers:
        old_list = old_by_ticker.get(ticker, [])
        new = new_by_ticker.get(ticker)

        if old_list:
            # Use first trade's premium (partials share same entry)
            old_cts = old_list[0]["contracts"]
            old_prem = old_list[0]["premium_per_contract"]
            old_cost = old_list[0]["total_cost"]
        else:
            old_cts = old_prem = old_cost = 0

        if new:
            new_cts = new["contracts"]
            new_prem = new["premium"]
            new_cost = new["total_cost"]
        else:
            new_cts = new_prem = new_cost = 0

        diff = ""
        if old_prem and new_prem and abs(old_prem - new_prem) > 0.01:
            diff = f"PREMIUM CHANGED ${old_prem:.2f} → ${new_prem:.2f}"
        elif old_cts != new_cts:
            diff = f"CONTRACTS {old_cts} → {new_cts}"
        elif not old_list and new:
            diff = "NEW (wasn't traded before)"
        elif old_list and not new:
            diff = "DROPPED (rejected by pipeline)"
        else:
            diff = "OK"

        print(f"{ticker:<6} {old_cts:<8} ${old_prem:<9.2f} ${old_cost:<9.2f} "
              f"{new_cts:<8} ${new_prem:<9.2f} ${new_cost:<9.2f} {diff}")

    # Old P&L summary
    old_total_pnl = sum(t["pnl_dollars"] or 0 for t in old_trades)
    print(f"\nOld recorded P&L total: ${old_total_pnl:,.2f}")
    print(f"(New P&L requires market data for exits — run position_monitor for that)")

    # Cleanup
    Path(tmp_db).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay a day's signals through fixed code")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to replay (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # quiet mode — only show errors

    asyncio.run(replay(args.date))


if __name__ == "__main__":
    main()
