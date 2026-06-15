"""Backfill real 1-min stock OHLC from historical_0dte.db (Polygon) into thetadata_options.db.

The thetadata_options.db stock_ohlc table currently has fake data (O=H=L=C, volume=0)
extracted from option_greeks underlying_price. This script replaces it with real
1-min bars from Polygon's underlying_bars table in historical_0dte.db.

Source: journal/historical_0dte.db → underlying_bars (epoch ms timestamps, 8.8M rows)
Target: journal/thetadata_options.db → stock_ohlc (ISO timestamps with timezone)

Usage:
    python scripts/backfill_real_stock_ohlc.py                # all tickers
    python scripts/backfill_real_stock_ohlc.py --ticker SPY   # single ticker
    python scripts/backfill_real_stock_ohlc.py --dry-run      # count only, no write
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).resolve().parent.parent
THETA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
POLYGON_DB = str(PROJECT_DIR / "journal" / "historical_0dte.db")

ET = ZoneInfo("America/New_York")

# Tickers present in both DBs
TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]


def epoch_ms_to_iso_et(epoch_ms: int) -> str:
    """Convert epoch milliseconds to ISO string with ET timezone offset."""
    dt_utc = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    dt_et = dt_utc.astimezone(ET)
    # Format: 2026-05-19 09:30:00-04:00
    return dt_et.strftime("%Y-%m-%d %H:%M:%S%z")


def format_offset(s: str) -> str:
    """Ensure timezone offset has colon: -0400 -> -04:00"""
    if len(s) >= 5 and s[-5] in ('+', '-') and ':' not in s[-5:]:
        return s[:-2] + ':' + s[-2:]
    return s


def backfill(tickers: list[str], dry_run: bool = False) -> None:
    poly_conn = sqlite3.connect(POLYGON_DB)
    poly_conn.execute("PRAGMA journal_mode = WAL")
    poly_conn.execute("PRAGMA busy_timeout = 5000")

    theta_conn = sqlite3.connect(THETA_DB)
    theta_conn.execute("PRAGMA journal_mode = WAL")
    theta_conn.execute("PRAGMA busy_timeout = 5000")

    # Check which tickers exist in Polygon DB
    poly_tickers = {
        row[0] for row in
        poly_conn.execute("SELECT DISTINCT ticker FROM underlying_bars").fetchall()
    }

    total_replaced = 0
    total_inserted = 0

    for ticker in tickers:
        if ticker not in poly_tickers:
            print(f"  {ticker}: not in Polygon DB, skipping")
            continue

        # Count existing fake rows for this ticker
        fake_count = theta_conn.execute(
            "SELECT COUNT(*) FROM stock_ohlc WHERE ticker = ? AND volume = 0",
            (ticker,),
        ).fetchone()[0]

        # Count real rows available from Polygon
        real_count = poly_conn.execute(
            "SELECT COUNT(*) FROM underlying_bars WHERE ticker = ?",
            (ticker,),
        ).fetchone()[0]

        print(f"  {ticker}: {fake_count:,} fake rows in theta DB, {real_count:,} real rows in Polygon")

        if dry_run:
            total_replaced += fake_count
            total_inserted += real_count
            continue

        # Delete fake rows for dates where we have real data
        # Get date range from Polygon
        date_range = poly_conn.execute(
            "SELECT MIN(date), MAX(date) FROM underlying_bars WHERE ticker = ?",
            (ticker,),
        ).fetchone()

        if not date_range or not date_range[0]:
            continue

        min_date, max_date = date_range
        print(f"    Polygon date range: {min_date} to {max_date}")

        # Delete fake rows in this date range
        deleted = theta_conn.execute(
            "DELETE FROM stock_ohlc WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?",
            (ticker, f"{min_date} 00:00:00", f"{max_date} 23:59:59"),
        ).rowcount
        print(f"    Deleted {deleted:,} fake/old rows")

        # Read real bars from Polygon in chunks
        cursor = poly_conn.execute(
            "SELECT timestamp, open, high, low, close, volume, vwap "
            "FROM underlying_bars WHERE ticker = ? ORDER BY timestamp",
            (ticker,),
        )

        batch = []
        batch_size = 10000
        inserted = 0

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for epoch_ms, o, h, l, c, vol, vwap in rows:
                iso_ts = format_offset(epoch_ms_to_iso_et(epoch_ms))
                batch.append((ticker, iso_ts, o, h, l, c, vol or 0, vwap))

            theta_conn.executemany(
                "INSERT OR IGNORE INTO stock_ohlc "
                "(ticker, timestamp, open, high, low, close, volume, vwap) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            inserted += len(batch)
            batch.clear()

        theta_conn.commit()
        print(f"    Inserted {inserted:,} real bars")
        total_replaced += deleted
        total_inserted += inserted

    poly_conn.close()
    theta_conn.close()

    action = "Would replace" if dry_run else "Replaced"
    print(f"\n{action} {total_replaced:,} fake rows with {total_inserted:,} real bars")


def main():
    parser = argparse.ArgumentParser(description="Backfill real stock OHLC from Polygon into ThetaData DB")
    parser.add_argument("--ticker", type=str, help="Single ticker (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no writes")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    print(f"Backfilling real 1-min stock OHLC")
    print(f"  Source: {POLYGON_DB}")
    print(f"  Target: {THETA_DB}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    backfill(tickers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
