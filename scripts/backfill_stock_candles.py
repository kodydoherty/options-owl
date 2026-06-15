"""Backfill historical 5m stock candles from Polygon REST API into harvester DB.

Downloads 5-minute OHLCV bars for all tracked tickers and inserts them into
the stock_candles table (same schema the harvester's candle_collector uses).

Usage:
    python scripts/backfill_stock_candles.py                    # backfill Mar 27 - May 20
    python scripts/backfill_stock_candles.py --from 2026-03-01  # custom start
    python scripts/backfill_stock_candles.py --ticker SPY       # single ticker
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

HARVESTER_DB = str(PROJECT_DIR / "journal" / "owlet-harvester" / "options_data.db")

TICKERS = [
    "SPY", "QQQ", "NVDA", "TSLA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
    "COIN", "SMCI", "BA", "NFLX", "MU", "JPM",
    "DIA", "XLF", "XLK", "GLD", "SLV", "TLT",
]

CANDLE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS stock_candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    bar_start_ts INTEGER NOT NULL,
    bar_start TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL DEFAULT 0,
    vwap REAL DEFAULT 0,
    source TEXT DEFAULT 'polygon_backfill'
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON stock_candles(ticker, timeframe, bar_start_ts);
"""


def init_db(conn):
    """Ensure stock_candles table exists."""
    conn.executescript(CANDLE_SCHEMA)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.commit()


def get_existing_dates(conn, ticker: str) -> set[str]:
    """Get dates that already have candle data for a ticker."""
    rows = conn.execute("""
        SELECT DISTINCT date(bar_start) FROM stock_candles
        WHERE ticker = ? AND timeframe = '5m'
    """, (ticker,)).fetchall()
    return {r[0] for r in rows}


def fetch_polygon_candles(api_key: str, ticker: str, date_str: str) -> list[dict]:
    """Fetch 5m candles for a single day from Polygon REST API."""
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute"
        f"/{date_str}/{date_str}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
    )

    resp = requests.get(url, timeout=30)
    if resp.status_code == 429:
        # Rate limited — wait and retry
        time.sleep(12)
        resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        return []

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return []

    candles = []
    for bar in results:
        ts_ms = bar["t"]  # Unix ms
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
        dt_et = dt_utc.astimezone(ET)

        # Filter to market hours (9:30 - 16:00 ET)
        if dt_et.hour < 9 or (dt_et.hour == 9 and dt_et.minute < 30):
            continue
        if dt_et.hour >= 16:
            continue

        candles.append({
            "ticker": ticker,
            "timeframe": "5m",
            "bar_start_ts": ts_ms,
            "bar_start": dt_et.strftime("%Y-%m-%d %H:%M:%S"),
            "open": bar["o"],
            "high": bar["h"],
            "low": bar["l"],
            "close": bar["c"],
            "volume": bar.get("v", 0),
            "vwap": bar.get("vw", 0),
            "source": "polygon_backfill",
        })

    return candles


def insert_candles(conn, candles: list[dict]):
    """Insert candles into DB, skipping duplicates."""
    conn.executemany("""
        INSERT OR IGNORE INTO stock_candles
        (ticker, timeframe, bar_start_ts, bar_start, open, high, low, close, volume, vwap, source)
        VALUES (:ticker, :timeframe, :bar_start_ts, :bar_start, :open, :high, :low, :close, :volume, :vwap, :source)
    """, candles)
    conn.commit()


def get_trading_days(start: str, end: str) -> list[str]:
    """Generate list of weekday dates between start and end."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    days = []
    current = s
    while current <= e:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", default="2026-03-27",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", default="2026-05-20",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--ticker", help="Single ticker to backfill")
    args = parser.parse_args()

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set in .env")
        sys.exit(1)

    tickers = [args.ticker] if args.ticker else TICKERS
    trading_days = get_trading_days(args.from_date, args.to_date)

    print(f"Backfilling {len(tickers)} tickers x {len(trading_days)} days")
    print(f"Period: {args.from_date} to {args.to_date}")
    print(f"DB: {HARVESTER_DB}")

    conn = sqlite3.connect(HARVESTER_DB)
    init_db(conn)

    total_inserted = 0
    total_skipped = 0

    for ti, ticker in enumerate(tickers):
        existing = get_existing_dates(conn, ticker)
        days_needed = [d for d in trading_days if d not in existing]

        if not days_needed:
            print(f"[{ti+1}/{len(tickers)}] {ticker}: all {len(trading_days)} days already exist, skipping")
            total_skipped += len(trading_days)
            continue

        print(f"[{ti+1}/{len(tickers)}] {ticker}: {len(days_needed)} days to fetch "
              f"({len(existing)} already exist)", end="", flush=True)

        ticker_bars = 0
        for di, day in enumerate(days_needed):
            candles = fetch_polygon_candles(api_key, ticker, day)
            if candles:
                insert_candles(conn, candles)
                ticker_bars += len(candles)
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)

            # Polygon free tier: 5 calls/min. Paid tier much higher.
            # Be conservative — 250ms between calls
            time.sleep(0.25)

        total_inserted += ticker_bars
        print(f" {ticker_bars} bars")

    conn.close()
    print(f"\nDone. Inserted {total_inserted:,} bars, skipped {total_skipped} ticker-days.")


if __name__ == "__main__":
    main()
