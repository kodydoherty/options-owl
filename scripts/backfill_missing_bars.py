"""Backfill missing option bars for tickers that have trading_days rows but 0 bars.

The download_historical_0dte.py script skips days already in trading_days,
but many had failed downloads (0 option bars). This script re-downloads those.

Usage:
    python scripts/backfill_missing_bars.py                    # all tickers
    python scripts/backfill_missing_bars.py --ticker TSLA      # one ticker
    python scripts/backfill_missing_bars.py --ticker TSLA --max 50  # limit days
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_historical_0dte import (
    DB_PATH, api_get, build_option_ticker, download_bars, find_atm_strike, init_db,
)

PRIORITY_TICKERS = [
    "TSLA", "NVDA", "AAPL", "AMZN", "META", "GOOGL", "MSFT",
    "AMD", "PLTR", "MSTR", "AVGO",
]


def backfill(ticker: str | None = None, max_days: int = 0):
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=10000")

    # Find days with underlying bars but 0 option bars
    query = """
        SELECT date, ticker, close_price, atm_strike
        FROM trading_days
        WHERE call_bars < 10 AND underlying_bars > 60
    """
    params = []
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    else:
        # Priority tickers first
        placeholders = ",".join("?" * len(PRIORITY_TICKERS))
        query += f" AND ticker IN ({placeholders})"
        params.extend(PRIORITY_TICKERS)
    query += " ORDER BY ticker, date"

    rows = conn.execute(query, params).fetchall()
    if max_days > 0:
        rows = rows[:max_days]

    print(f"Found {len(rows)} days with missing option bars")
    if not rows:
        return

    total_bars = 0
    errors = 0
    start_time = time.time()

    for i, (date_str, tkr, close_price, atm_strike) in enumerate(rows):
        if not atm_strike or atm_strike <= 0:
            tick_size = 1.0 if tkr in ("SPY", "QQQ", "IWM") else 0.50
            atm_strike = find_atm_strike(close_price, tick_size)

        call_ticker = build_option_ticker(tkr, date_str, "call", atm_strike)
        put_ticker = build_option_ticker(tkr, date_str, "put", atm_strike)

        elapsed = time.time() - start_time
        rate = (i / elapsed * 3600) if elapsed > 0 and i > 0 else 0
        eta = ((len(rows) - i) / rate) if rate > 0 else 0

        print(f"[{i+1}/{len(rows)}] {tkr} {date_str} ATM ${atm_strike:.0f} | ETA {eta:.1f}h")

        call_count = 0
        put_count = 0

        try:
            call_results = download_bars(call_ticker, date_str)
            if call_results:
                conn.executemany(
                    "INSERT OR IGNORE INTO option_bars VALUES (?,?,?,?,?,?,?,?,?)",
                    [(call_ticker, b["t"], b["o"], b["h"], b["l"], b["c"],
                      b.get("v", 0), b.get("vw", 0), b.get("n", 0)) for b in call_results]
                )
                call_count = len(call_results)
                total_bars += call_count
        except Exception as e:
            print(f"  ERROR call: {e}")
            errors += 1

        try:
            put_results = download_bars(put_ticker, date_str)
            if put_results:
                conn.executemany(
                    "INSERT OR IGNORE INTO option_bars VALUES (?,?,?,?,?,?,?,?,?)",
                    [(put_ticker, b["t"], b["o"], b["h"], b["l"], b["c"],
                      b.get("v", 0), b.get("vw", 0), b.get("n", 0)) for b in put_results]
                )
                put_count = len(put_results)
                total_bars += put_count
        except Exception as e:
            print(f"  ERROR put: {e}")
            errors += 1

        # Update trading_days with new bar counts
        conn.execute(
            "UPDATE trading_days SET call_bars = ?, put_bars = ?, "
            "atm_call_ticker = ?, atm_put_ticker = ? "
            "WHERE date = ? AND ticker = ?",
            (call_count, put_count, call_ticker, put_ticker, date_str, tkr)
        )
        conn.commit()
        print(f"  call={call_count} put={put_count}")

    conn.close()
    elapsed = time.time() - start_time
    print(f"\nBackfill complete: {total_bars:,} bars, {errors} errors, {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()
    backfill(ticker=args.ticker, max_days=args.max)
