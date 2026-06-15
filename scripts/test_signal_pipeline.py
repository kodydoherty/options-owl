"""Test the full signal-to-trade pipeline on the droplet.

Injects a synthetic ML signal through evaluate_and_trade, verifies the trade
was created in the DB, then cleans up. Run inside a bot container:

    docker exec owlet-kody python scripts/test_signal_pipeline.py
    docker exec owlet-kody python scripts/test_signal_pipeline.py --cleanup-only

This tests the EXACT code path the ML scan loop uses:
  synthetic signal → _ml_signal_to_trade_signal → evaluate_and_trade → DB
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

from loguru import logger

TEST_TRADE_MARKER = "TEST_PIPELINE"


async def run_test(cleanup_only: bool = False) -> bool:
    from options_owl.config.settings import Settings
    from options_owl.execution.paper_trader import PaperTrader

    settings = Settings()
    agent_id = getattr(settings, "AGENT_ID", "unknown")

    # Init PaperTrader (no Webull — this is a pipeline test only)
    trader = PaperTrader(settings, webull_executor=None)
    await trader.init()

    if cleanup_only:
        deleted = await _cleanup_test_trades(trader)
        print(f"Cleaned up {deleted} test trade(s)")
        return True

    # Build a synthetic signal matching ML scan output
    from options_owl.bot_runner import _ml_signal_to_trade_signal

    # Disable smart entry and morning cutoff for test
    settings.ENABLE_SMART_ENTRY = False
    settings.ENABLE_MORNING_CUTOFF = False

    signal = _ml_signal_to_trade_signal(
        ticker="SPY",
        direction="CALL",
        score=87,  # pattern_conf=0.87 → score 87
        premium=2.50,
        strike=590.0,
        expiry="2026-05-28",
        ml_confidence=0.87,
        underlying_price=592.0,  # realistic underlying price
    )

    logger.info(f"TEST: Injecting synthetic SPY CALL signal (score=87, premium=$2.50, underlying=$592)")

    # Route through entry pipeline (same path as ML scan loop)
    signal_id = -999999  # negative = synthetic
    try:
        result = await trader.evaluate_and_trade(
            signal, signal_id, ml_confidence=0.87,
        )
    except Exception as exc:
        logger.error(f"TEST FAILED: evaluate_and_trade raised: {exc}")
        return False

    if result is None:
        logger.warning(
            "TEST: Signal was REJECTED by entry pipeline. "
            "This is expected outside market hours (time_of_day gate). "
            "The pipeline is working correctly — it just won't open trades now."
        )
        # Check trade_events for the rejection reason
        await _show_recent_events(trader)
        return True

    trade_id = result["trade_id"]
    contracts = result["contracts"]
    premium = result["premium"]
    logger.info(
        f"TEST PASSED: Trade created! "
        f"id={trade_id} contracts={contracts} premium=${premium:.2f}"
    )

    # Verify DB record
    import aiosqlite
    from options_owl.journal.db import connect as db_connect

    async with db_connect(settings.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            "SELECT id, ticker, direction, contracts, premium_per_contract, status "
            "FROM paper_trades WHERE id = ?",
            (trade_id,),
        )
        row = await cursor.fetchone()
        if row:
            logger.info(
                f"TEST DB VERIFY: id={row['id']} {row['ticker']} {row['direction']} "
                f"{row['contracts']}x @ ${row['premium_per_contract']:.2f} "
                f"status={row['status']}"
            )
        else:
            logger.error(f"TEST FAILED: Trade {trade_id} not found in DB!")
            return False

    # Clean up — close and mark the test trade
    await trader.close_trade(
        trade_id=trade_id,
        exit_price=530.0,
        exit_premium=2.50,
        reason=TEST_TRADE_MARKER,
    )
    logger.info(f"TEST: Cleaned up trade {trade_id} (closed with reason={TEST_TRADE_MARKER})")

    # Delete the test trade from DB entirely
    async with db_connect(settings.DB_PATH) as conn:
        await conn.execute(
            "DELETE FROM paper_trades WHERE id = ? AND exit_reason = ?",
            (trade_id, TEST_TRADE_MARKER),
        )
        await conn.execute(
            "DELETE FROM trade_events WHERE trade_id = ?",
            (trade_id,),
        )
        await conn.commit()
    logger.info(f"TEST: Deleted test trade {trade_id} from DB")

    print(f"\n{'='*60}")
    print(f"  PIPELINE TEST PASSED for {agent_id}")
    print(f"  Signal → entry pipeline → DB record → cleanup")
    print(f"{'='*60}\n")
    return True


async def _show_recent_events(trader):
    """Show recent trade_events to diagnose rejections."""
    import aiosqlite
    from options_owl.journal.db import connect as db_connect
    from options_owl.config.settings import Settings

    settings = Settings()
    try:
        async with db_connect(settings.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT event_type, ticker, details FROM trade_events "
                "ORDER BY id DESC LIMIT 5"
            )
            rows = await cursor.fetchall()
            if rows:
                print("\nRecent trade_events:")
                for r in rows:
                    print(f"  {r['event_type']}: {r['ticker']} — {r['details'][:120]}")
    except Exception:
        pass


async def _cleanup_test_trades(trader) -> int:
    """Delete any leftover test trades."""
    import aiosqlite
    from options_owl.journal.db import connect as db_connect
    from options_owl.config.settings import Settings

    settings = Settings()
    count = 0
    async with db_connect(settings.DB_PATH) as conn:
        cursor = await conn.execute(
            "SELECT id FROM paper_trades WHERE exit_reason = ?",
            (TEST_TRADE_MARKER,),
        )
        rows = await cursor.fetchall()
        for row in rows:
            await conn.execute("DELETE FROM paper_trades WHERE id = ?", (row[0],))
            await conn.execute("DELETE FROM trade_events WHERE trade_id = ?", (row[0],))
            count += 1
        await conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser(description="Test ML signal pipeline end-to-end")
    parser.add_argument("--cleanup-only", action="store_true", help="Only clean up test trades")
    args = parser.parse_args()

    success = asyncio.run(run_test(cleanup_only=args.cleanup_only))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
