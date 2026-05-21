#!/usr/bin/env python3
"""Backfill missing Webull fill prices for historical trades.

Runs inside a bot container on the droplet. Two strategies:
1. Entry fills: look up via client_order_id (stored in paper_trades)
2. Exit fills: pull order history day-by-day and match SELL orders by
   ticker + date + contracts (since exit order IDs weren't stored historically)

Usage (on droplet, exec into container):
    docker compose exec owlet-kody python scripts/backfill_fill_prices.py
    docker compose exec owlet-adam python scripts/backfill_fill_prices.py

Or with --dry-run to preview without writing:
    docker compose exec owlet-kody python scripts/backfill_fill_prices.py --dry-run
"""
import asyncio
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from options_owl.execution.webull_executor import WebullExecutor
from options_owl.config.settings import Settings


async def backfill(dry_run: bool = False):
    settings = Settings()

    # Find the DB
    db_path = None
    for candidate in [
        Path("journal") / "raw_messages.db",
        Path("/app/journal/raw_messages.db"),
    ]:
        if candidate.exists():
            db_path = str(candidate)
            break

    if not db_path:
        print("ERROR: Cannot find raw_messages.db")
        return

    print(f"DB: {db_path}")
    if dry_run:
        print("DRY RUN — no changes will be written\n")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find trades missing fill prices
    rows = conn.execute("""
        SELECT id, ticker, strike, option_type, contracts,
               opened_at, closed_at, premium_per_contract, exit_premium,
               pnl_dollars, pnl_pct,
               webull_order_id, webull_client_order_id,
               webull_exit_order_id,
               webull_entry_fill_price, webull_exit_fill_price,
               expiry_date
        FROM paper_trades
        WHERE webull_order_id IS NOT NULL
          AND status = 'closed'
          AND (webull_entry_fill_price IS NULL OR webull_entry_fill_price = 0
               OR webull_exit_fill_price IS NULL OR webull_exit_fill_price = 0)
        ORDER BY id
    """).fetchall()

    if not rows:
        print("No trades need backfilling!")
        conn.close()
        return

    print(f"Found {len(rows)} trades needing fill price backfill\n")

    # Init Webull executor
    executor = WebullExecutor(settings)
    try:
        executor._ensure_clients()
        print("Webull connection OK\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to Webull: {e}")
        conn.close()
        return

    # ================================================================
    # PHASE 1: Entry fill prices via client_order_id lookup
    # ================================================================
    print("=" * 60)
    print("PHASE 1: Entry fill prices via client_order_id lookup")
    print("=" * 60)

    entry_updated = 0
    entry_fills = {}  # trade_id -> fill_price

    for row in rows:
        trade_id = row["id"]
        ticker = row["ticker"]
        entry_fill = row["webull_entry_fill_price"] or 0
        client_order_id = row["webull_client_order_id"]

        if entry_fill > 0:
            entry_fills[trade_id] = entry_fill
            continue
        if not client_order_id:
            print(f"  #{trade_id} {ticker}: no client_order_id, skipping")
            continue

        try:
            price = await executor.get_fill_price(client_order_id, retries=2)
            if price and price > 0:
                print(f"  #{trade_id} {ticker}: entry_fill=${price:.2f}")
                entry_fills[trade_id] = price
                if not dry_run:
                    conn.execute(
                        "UPDATE paper_trades SET webull_entry_fill_price = ? WHERE id = ?",
                        (price, trade_id),
                    )
                entry_updated += 1
            else:
                print(f"  #{trade_id} {ticker}: no entry fill (oid={client_order_id[:12]}...)")
        except Exception as e:
            print(f"  #{trade_id} {ticker}: error — {e}")

        await asyncio.sleep(1.0)  # generous rate limit

    if not dry_run and entry_updated:
        conn.commit()
    print(f"\nEntry fills found: {entry_updated}\n")

    # ================================================================
    # PHASE 2: Exit fills via order history matching
    # ================================================================
    # Group trades by close date so we can pull history day-by-day
    trades_by_close_date = defaultdict(list)
    for row in rows:
        exit_fill = row["webull_exit_fill_price"] or 0
        if exit_fill > 0:
            continue  # already have exit fill
        close_date = (row["closed_at"] or "")[:10]
        if close_date:
            trades_by_close_date[close_date].append(dict(row))

    if not trades_by_close_date:
        print("All exit fills already present!")
    else:
        print("=" * 60)
        print(f"PHASE 2: Exit fills via order history ({len(trades_by_close_date)} days)")
        print("=" * 60)

        exit_updated = 0
        pnl_fixed = 0

        for date_str in sorted(trades_by_close_date.keys()):
            trades_for_day = trades_by_close_date[date_str]
            print(f"\n--- {date_str} ({len(trades_for_day)} trades) ---")

            # Pull order history for this day
            await asyncio.sleep(3.0)  # wait before each API call to avoid 429
            try:
                history = await executor.get_order_history(date_str, date_str, page_size=100)
                print(f"  Got {len(history)} orders from Webull")
            except Exception as e:
                print(f"  ERROR getting history: {e}")
                continue

            if not history:
                # Try broader range (day before + day after)
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    start = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
                    end = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
                    await asyncio.sleep(3.0)
                    history = await executor.get_order_history(start, end, page_size=100)
                    print(f"  Retried with range {start}→{end}: got {len(history)} orders")
                except Exception as e:
                    print(f"  Retry failed: {e}")
                    continue

            if not history:
                print(f"  No order history for {date_str}")
                continue

            # Filter to SELL orders and extract fill prices
            sell_orders = []
            for order in history:
                if not isinstance(order, dict):
                    continue
                # Check if it's a SELL order
                side = order.get("side", "")
                orders_list = order.get("orders", [])

                if side == "SELL":
                    sell_orders.append(order)
                elif orders_list:
                    for sub in orders_list:
                        if isinstance(sub, dict) and sub.get("side") == "SELL":
                            sell_orders.append(sub)

            print(f"  Found {len(sell_orders)} SELL orders")

            # Debug: show first sell order structure
            if sell_orders and date_str == sorted(trades_by_close_date.keys())[0]:
                sample = sell_orders[0]
                print(f"  Sample SELL order keys: {list(sample.keys())}")
                # Look for symbol info
                for key in ("symbol", "ticker", "instrument"):
                    if key in sample:
                        print(f"    {key}: {sample[key]}")
                legs = sample.get("legs", sample.get("option_legs", []))
                if legs:
                    print(f"    legs[0] keys: {list(legs[0].keys()) if isinstance(legs[0], dict) else 'N/A'}")

            # Match sell orders to trades by ticker + strike + contracts
            for trade in trades_for_day:
                trade_id = trade["id"]
                ticker = trade["ticker"]
                strike = trade["strike"]
                contracts = trade["contracts"]
                option_type = (trade["option_type"] or "").upper()
                expiry = trade.get("expiry_date", "")
                old_pnl = trade["pnl_dollars"]

                matched = False
                for sell in sell_orders:
                    # Try to match by ticker + strike + option type
                    sell_symbol = sell.get("symbol", "")
                    sell_legs = sell.get("legs", sell.get("option_legs", []))

                    # Check top-level symbol match
                    if sell_symbol and sell_symbol != ticker:
                        continue

                    # Check legs for more specific matching
                    for leg in sell_legs:
                        if not isinstance(leg, dict):
                            continue
                        leg_symbol = leg.get("symbol", "")
                        leg_strike = float(leg.get("strike_price", 0) or 0)
                        leg_type = (leg.get("option_type", "") or "").upper()
                        leg_qty = int(float(leg.get("quantity", 0) or 0))
                        leg_expiry = leg.get("option_expire_date", "")

                        if (leg_symbol == ticker and
                            abs(leg_strike - strike) < 0.01 and
                            leg_type == option_type):

                            price = WebullExecutor._extract_fill_price(sell)
                            if not price:
                                # Try extracting from leg
                                for pkey in ("avg_filled_price", "avgFilledPrice", "filled_price"):
                                    p = leg.get(pkey)
                                    if p:
                                        price = float(p)
                                        break

                            if price and price > 0:
                                entry_fill = entry_fills.get(trade_id, 0)
                                print(f"  #{trade_id} {ticker} ${strike} {option_type}: "
                                      f"exit_fill=${price:.2f}", end="")

                                if entry_fill > 0:
                                    real_pnl = (price - entry_fill) * contracts * 100
                                    real_pnl_pct = (price - entry_fill) / entry_fill * 100
                                    print(f" → P&L ${old_pnl:.2f} → ${real_pnl:.2f} "
                                          f"({real_pnl_pct:+.1f}%)")

                                    if not dry_run:
                                        conn.execute(
                                            """UPDATE paper_trades
                                               SET webull_exit_fill_price = ?,
                                                   pnl_dollars = ?,
                                                   pnl_pct = ?
                                               WHERE id = ?""",
                                            (price, real_pnl, real_pnl_pct, trade_id),
                                        )
                                    pnl_fixed += 1
                                else:
                                    print(f" (no entry fill for P&L recompute)")
                                    if not dry_run:
                                        conn.execute(
                                            "UPDATE paper_trades SET webull_exit_fill_price = ? WHERE id = ?",
                                            (price, trade_id),
                                        )

                                exit_updated += 1
                                matched = True
                                # Remove from sell_orders to avoid double-matching
                                sell_orders.remove(sell)
                                break

                    if matched:
                        break

                if not matched:
                    print(f"  #{trade_id} {ticker} ${strike} {option_type}: "
                          f"no matching SELL order found")

        if not dry_run:
            conn.commit()

        print(f"\nExit fills updated: {exit_updated}")
        print(f"P&L recomputed: {pnl_fixed}")

    # Summary
    remaining = conn.execute("""
        SELECT count(*) as cnt,
               sum(CASE WHEN webull_entry_fill_price IS NULL OR webull_entry_fill_price = 0
                        THEN 1 ELSE 0 END) as missing_entry,
               sum(CASE WHEN webull_exit_fill_price IS NULL OR webull_exit_fill_price = 0
                        THEN 1 ELSE 0 END) as missing_exit
        FROM paper_trades
        WHERE webull_order_id IS NOT NULL AND status = 'closed'
          AND (webull_entry_fill_price IS NULL OR webull_entry_fill_price = 0
               OR webull_exit_fill_price IS NULL OR webull_exit_fill_price = 0)
    """).fetchone()

    print(f"\nRemaining gaps: {remaining['cnt']} trades "
          f"({remaining['missing_entry']} missing entry, "
          f"{remaining['missing_exit']} missing exit)")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    asyncio.run(backfill(dry_run))
