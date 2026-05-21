"""Download historical 0DTE options data from Polygon.io.

Downloads 1-minute bars for SPY/QQQ ATM call and put 0DTE options.
Free tier: ~2 years of data (April 2024+), 5 req/min rate limit.

Per trading day: 3 API calls (underlying bars + ATM call bars + ATM put bars)
At 12s/request = ~36s per day. ~500 days = ~5 hours total.

Resumable: skips days already in the database.

Usage:
    python scripts/download_historical_0dte.py
    python scripts/download_historical_0dte.py --ticker SPY --start 2024-04-01
    python scripts/download_historical_0dte.py --ticker QQQ --start 2024-06-01
"""

import argparse
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading

import httpx

API_KEY = os.getenv("POLYGON_API_KEY", "Zi2nVXh9YJdPtfmuQRScmecxj3IlSpET")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
BASE_URL = "https://api.polygon.io"
REQUEST_DELAY = float(os.getenv("DOWNLOAD_DELAY", "0.5"))  # paid plan: unlimited calls

_thread_local = threading.local()


def _get_client() -> httpx.Client:
    """Thread-local HTTP client for safe parallel downloads."""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = httpx.Client(timeout=30)
    return _thread_local.client


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # allow concurrent readers/writers
    conn.execute("PRAGMA busy_timeout=10000")  # wait up to 10s on lock
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trading_days (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open_price REAL,
            close_price REAL,
            high_price REAL,
            low_price REAL,
            atm_call_ticker TEXT,
            atm_put_ticker TEXT,
            atm_strike REAL,
            call_bars INTEGER DEFAULT 0,
            put_bars INTEGER DEFAULT 0,
            underlying_bars INTEGER DEFAULT 0,
            downloaded_at TEXT,
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS option_bars (
            contract_ticker TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER, vwap REAL, num_trades INTEGER,
            PRIMARY KEY (contract_ticker, timestamp)
        );

        CREATE TABLE IF NOT EXISTS underlying_bars (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER, vwap REAL,
            PRIMARY KEY (ticker, timestamp)
        );

        CREATE INDEX IF NOT EXISTS idx_ob_contract ON option_bars(contract_ticker);
        CREATE INDEX IF NOT EXISTS idx_ub_date ON underlying_bars(ticker, date);
    """)
    conn.close()


def api_get(url):
    """Make an API call with rate limiting."""
    time.sleep(REQUEST_DELAY)
    client = _get_client()
    r = client.get(url)
    if r.status_code == 429:
        print("    Rate limited — waiting 60s...")
        time.sleep(60)
        r = client.get(url)
    return r


def get_trading_days(ticker, start_date, end_date):
    """Get daily OHLC for all trading days (single API call, no rate limit needed for daily)."""
    url = (f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000&apiKey={API_KEY}")
    time.sleep(REQUEST_DELAY)
    r = _get_client().get(url)
    data = r.json()
    days = []
    for bar in data.get("results", []):
        dt = datetime.utcfromtimestamp(bar["t"] / 1000)
        days.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": bar["o"], "close": bar["c"],
            "high": bar["h"], "low": bar["l"],
        })
    return days


def build_option_ticker(underlying, date_str, option_type, strike):
    """Build Polygon options ticker symbol.

    Format: O:SPY240603C00526000
    = O:{TICKER}{YYMMDD}{C/P}{strike*1000 zero-padded to 8 digits}
    """
    yy = date_str[2:4]
    mm = date_str[5:7]
    dd = date_str[8:10]
    cp = "C" if option_type == "call" else "P"
    # Strike in thousands, 8 digits zero-padded
    strike_int = int(round(strike * 1000))
    strike_str = f"{strike_int:08d}"
    return f"O:{underlying}{yy}{mm}{dd}{cp}{strike_str}"


def find_atm_strike(close_price, tick_size=1.0):
    """Find the nearest ATM strike price.

    SPY strikes come in $1 increments. Round to nearest.
    """
    return round(close_price / tick_size) * tick_size


def download_bars(ticker_sym, date_str):
    """Download 1-minute bars for any ticker on a given date."""
    url = (f"{BASE_URL}/v2/aggs/ticker/{ticker_sym}/range/1/minute/"
           f"{date_str}/{date_str}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}")
    r = api_get(url)
    data = r.json()
    if data.get("status") == "NOT_AUTHORIZED":
        return None  # date too old for our tier
    return data.get("results", [])


def run_download(ticker="SPY", start_date="2024-04-01", end_date=None):
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA busy_timeout=10000")

    print(f"Downloading 0DTE data: {ticker} from {start_date} to {end_date}")

    # Get all trading days
    trading_days = get_trading_days(ticker, start_date, end_date)
    print(f"Found {len(trading_days)} trading days")

    # Filter out already-downloaded days
    downloaded = set()
    for row in conn.execute("SELECT date FROM trading_days WHERE ticker = ?", (ticker,)):
        downloaded.add(row[0])

    remaining = [d for d in trading_days if d["date"] not in downloaded]
    print(f"{len(remaining)} days to download ({len(downloaded)} already done)")

    # Tick size: SPY/QQQ use $1 strikes, others may use $0.50
    tick_size = 1.0 if ticker in ("SPY", "QQQ", "IWM") else 0.50

    total_option_bars = 0
    total_und_bars = 0
    errors = 0
    start_time = time.time()

    for i, day in enumerate(remaining):
        date_str = day["date"]
        close = day["close"]
        atm_strike = find_atm_strike(close, tick_size)

        call_ticker = build_option_ticker(ticker, date_str, "call", atm_strike)
        put_ticker = build_option_ticker(ticker, date_str, "put", atm_strike)

        elapsed_total = time.time() - start_time
        rate = (i / elapsed_total * 3600) if elapsed_total > 0 and i > 0 else 0
        eta_hours = ((len(remaining) - i) / rate) if rate > 0 else 0

        print(f"[{i+1}/{len(remaining)}] {date_str} | {ticker} ${close:.2f} | "
              f"ATM strike ${atm_strike:.0f} | ETA {eta_hours:.1f}h")

        # 1. Download underlying 1-min bars
        try:
            und_results = download_bars(ticker, date_str)
            if und_results is None:
                print(f"  NOT_AUTHORIZED — skipping (date too old for free tier)")
                conn.execute(
                    "INSERT OR REPLACE INTO trading_days VALUES (?,?,?,?,?,?,?,?,?,0,0,0,?)",
                    (date_str, ticker, day["open"], close, day["high"], day["low"],
                     call_ticker, put_ticker, atm_strike, datetime.now().isoformat())
                )
                conn.commit()
                continue

            und_count = 0
            if und_results:
                conn.executemany(
                    "INSERT OR IGNORE INTO underlying_bars VALUES (?,?,?,?,?,?,?,?,?)",
                    [(ticker, date_str, b["t"], b["o"], b["h"], b["l"], b["c"],
                      b.get("v", 0), b.get("vw", 0)) for b in und_results]
                )
                und_count = len(und_results)
                total_und_bars += und_count
        except Exception as e:
            print(f"  ERROR underlying: {e}")
            errors += 1
            und_count = 0

        # 2. Download ATM call bars
        call_count = 0
        try:
            call_results = download_bars(call_ticker, date_str)
            if call_results:
                conn.executemany(
                    "INSERT OR IGNORE INTO option_bars VALUES (?,?,?,?,?,?,?,?,?)",
                    [(call_ticker, b["t"], b["o"], b["h"], b["l"], b["c"],
                      b.get("v", 0), b.get("vw", 0), b.get("n", 0)) for b in call_results]
                )
                call_count = len(call_results)
                total_option_bars += call_count
        except Exception as e:
            print(f"  ERROR call: {e}")
            errors += 1

        # 3. Download ATM put bars
        put_count = 0
        try:
            put_results = download_bars(put_ticker, date_str)
            if put_results:
                conn.executemany(
                    "INSERT OR IGNORE INTO option_bars VALUES (?,?,?,?,?,?,?,?,?)",
                    [(put_ticker, b["t"], b["o"], b["h"], b["l"], b["c"],
                      b.get("v", 0), b.get("vw", 0), b.get("n", 0)) for b in put_results]
                )
                put_count = len(put_results)
                total_option_bars += put_count
        except Exception as e:
            print(f"  ERROR put: {e}")
            errors += 1

        # Save trading day record
        conn.execute(
            "INSERT OR REPLACE INTO trading_days VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (date_str, ticker, day["open"], close, day["high"], day["low"],
             call_ticker, put_ticker, atm_strike,
             call_count, put_count, und_count, datetime.now().isoformat())
        )
        conn.commit()

        print(f"  call={call_count} put={put_count} underlying={und_count} bars")

    conn.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE — {ticker}")
    print(f"{'='*60}")
    print(f"  Period:        {start_date} to {end_date}")
    print(f"  Days:          {len(remaining)} downloaded")
    print(f"  Option bars:   {total_option_bars:,}")
    print(f"  Underlying:    {total_und_bars:,} bars")
    print(f"  Errors:        {errors}")
    print(f"  Time:          {elapsed/3600:.1f} hours")
    print(f"  Database:      {DB_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2024-04-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    run_download(ticker=args.ticker, start_date=args.start, end_date=args.end)
