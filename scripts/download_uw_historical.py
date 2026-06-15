#!/usr/bin/env python3
"""
Download historical data from Unusual Whales API for ML signal features.

Our subscription gives:
- 30 trading days of intraday data (net-prem-ticks, spot-exposures, full-tape)
- ~1 year of daily data (greek-exposure, options-volume)
- Unlimited: insider, congress, institutional, correlations

Stores everything in journal/uw_historical.db (SQLite WAL mode).

Usage:
    python scripts/download_uw_historical.py                     # all tickers, all endpoints
    python scripts/download_uw_historical.py --tickers SPY TSLA  # specific tickers
    python scripts/download_uw_historical.py --days 10           # last N trading days only
    python scripts/download_uw_historical.py --daily-only        # skip intraday (faster)
"""

import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta

import requests

UW_KEY = "0294df1c-4517-4c0a-bae9-f037a39aa5ef"
BASE_URL = "https://api.unusualwhales.com/api"
HEADERS = {"Authorization": f"Bearer {UW_KEY}", "Accept": "application/json"}

TICKERS = [
    "SPY", "QQQ", "TSLA", "NVDA", "META", "AAPL", "AMZN",
    "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "IWM",
]

DB_PATH = "journal/uw_historical.db"

# Rate limit: be conservative — 1 req/sec
RATE_LIMIT_DELAY = 1.0
MAX_RETRIES = 3


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    # Daily greek exposure (GEX) — ~1yr history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS greek_exposure (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            call_gamma REAL, put_gamma REAL,
            call_delta REAL, put_delta REAL,
            call_charm REAL, put_charm REAL,
            call_vanna REAL, put_vanna REAL,
            PRIMARY KEY (ticker, date)
        )
    """)

    # Daily options volume/premium — ~1yr history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS options_volume (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            call_volume INTEGER, put_volume INTEGER,
            call_volume_ask_side INTEGER, call_volume_bid_side INTEGER,
            put_volume_ask_side INTEGER, put_volume_bid_side INTEGER,
            net_call_premium REAL, net_put_premium REAL,
            call_premium REAL, put_premium REAL,
            bearish_premium REAL, bullish_premium REAL,
            put_open_interest INTEGER, call_open_interest INTEGER,
            avg_30_day_call_volume REAL, avg_30_day_put_volume REAL,
            PRIMARY KEY (ticker, date)
        )
    """)

    # Intraday net premium ticks — 30 trading days
    conn.execute("""
        CREATE TABLE IF NOT EXISTS net_prem_ticks (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            tape_time TEXT NOT NULL,
            call_volume INTEGER, put_volume INTEGER,
            call_volume_ask_side INTEGER, call_volume_bid_side INTEGER,
            put_volume_ask_side INTEGER, put_volume_bid_side INTEGER,
            net_call_premium REAL, net_put_premium REAL,
            net_call_volume INTEGER, net_put_volume INTEGER,
            net_delta REAL,
            PRIMARY KEY (ticker, tape_time)
        )
    """)

    # Intraday spot GEX — 30 trading days
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spot_gex (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            price REAL,
            gamma_per_pct_oi REAL,
            charm_per_pct_oi REAL,
            vanna_per_pct_oi REAL,
            gamma_per_pct_vol REAL,
            charm_per_pct_vol REAL,
            vanna_per_pct_vol REAL,
            PRIMARY KEY (ticker, time)
        )
    """)

    # Flow alerts (unusual activity detections) — 30 trading days
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flow_alerts (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            created_at TEXT NOT NULL,
            type TEXT,
            strike REAL,
            expiry TEXT,
            price REAL,
            volume INTEGER,
            open_interest INTEGER,
            total_premium REAL,
            underlying_price REAL,
            trade_count INTEGER,
            iv_start REAL, iv_end REAL,
            volume_oi_ratio REAL,
            has_sweep INTEGER,
            has_floor INTEGER,
            has_multileg INTEGER,
            all_opening_trades INTEGER,
            alert_rule TEXT,
            total_bid_side_prem REAL,
            total_ask_side_prem REAL
        )
    """)

    # Dark pool trades — 30 trading days
    conn.execute("""
        CREATE TABLE IF NOT EXISTS darkpool (
            ticker TEXT NOT NULL,
            tracking_id TEXT NOT NULL,
            executed_at TEXT NOT NULL,
            size INTEGER,
            price REAL,
            premium REAL,
            nbbo_bid REAL,
            nbbo_ask REAL,
            volume INTEGER,
            market_center TEXT,
            PRIMARY KEY (ticker, tracking_id)
        )
    """)

    # Max pain per expiry — daily
    conn.execute("""
        CREATE TABLE IF NOT EXISTS max_pain (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            expiry TEXT NOT NULL,
            max_pain REAL,
            open_price REAL,
            close_price REAL,
            PRIMARY KEY (ticker, date, expiry)
        )
    """)

    # Congress trades
    conn.execute("""
        CREATE TABLE IF NOT EXISTS congress_trades (
            politician_id TEXT NOT NULL,
            ticker TEXT,
            transaction_date TEXT NOT NULL,
            name TEXT,
            txn_type TEXT,
            amounts TEXT,
            notes TEXT,
            filed_at_date TEXT,
            member_type TEXT,
            PRIMARY KEY (politician_id, transaction_date, ticker)
        )
    """)

    # Download log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS download_log (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            rows INTEGER,
            downloaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date, endpoint)
        )
    """)

    conn.commit()
    return conn


def api_call(endpoint: str, params: dict | None = None, retries: int = MAX_RETRIES) -> dict | list | None:
    """Make an API call with retry logic."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", data) if isinstance(data, dict) else data
            elif resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            elif resp.status_code == 404:
                return None
            else:
                msg = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                print(f"  HTTP {resp.status_code}: {msg}", flush=True)
                if "historic_data_access_missing" in str(msg):
                    return None  # Date too old for our subscription
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(3)
        except Exception as e:
            print(f"  Error: {e} (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(2)
    return None


def is_already_downloaded(conn: sqlite3.Connection, ticker: str, date: str, endpoint: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM download_log WHERE ticker=? AND date=? AND endpoint=?",
        (ticker, date, endpoint),
    ).fetchone()
    return row is not None


def log_download(conn: sqlite3.Connection, ticker: str, date: str, endpoint: str, rows: int):
    conn.execute(
        "INSERT OR REPLACE INTO download_log (ticker, date, endpoint, rows) VALUES (?,?,?,?)",
        (ticker, date, endpoint, rows),
    )
    conn.commit()


def download_greek_exposure(conn: sqlite3.Connection, ticker: str):
    """Download daily GEX data (~1yr history available)."""
    if is_already_downloaded(conn, ticker, "all", "greek_exposure"):
        print(f"  {ticker} greek_exposure: already downloaded", flush=True)
        return

    data = api_call(f"stock/{ticker}/greek-exposure")
    if not data:
        print(f"  {ticker} greek_exposure: no data", flush=True)
        return

    rows = 0
    for row in data:
        conn.execute(
            """INSERT OR REPLACE INTO greek_exposure
            (ticker, date, call_gamma, put_gamma, call_delta, put_delta,
             call_charm, put_charm, call_vanna, put_vanna)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ticker, row["date"],
             float(row.get("call_gamma") or 0), float(row.get("put_gamma") or 0),
             float(row.get("call_delta") or 0), float(row.get("put_delta") or 0),
             float(row.get("call_charm") or 0), float(row.get("put_charm") or 0),
             float(row.get("call_vanna") or 0), float(row.get("put_vanna") or 0)),
        )
        rows += 1

    conn.commit()
    log_download(conn, ticker, "all", "greek_exposure", rows)
    print(f"  {ticker} greek_exposure: {rows} days", flush=True)
    time.sleep(RATE_LIMIT_DELAY)


def download_options_volume(conn: sqlite3.Connection, ticker: str, date: str):
    """Download daily options volume for a specific date."""
    if is_already_downloaded(conn, ticker, date, "options_volume"):
        return 0

    data = api_call(f"stock/{ticker}/options-volume", {"date": date})
    if not data:
        return 0

    rows = 0
    items = data if isinstance(data, list) else [data]
    for row in items:
        d = row.get("date", date)
        conn.execute(
            """INSERT OR REPLACE INTO options_volume
            (ticker, date, call_volume, put_volume,
             call_volume_ask_side, call_volume_bid_side,
             put_volume_ask_side, put_volume_bid_side,
             net_call_premium, net_put_premium, call_premium, put_premium,
             bearish_premium, bullish_premium,
             put_open_interest, call_open_interest,
             avg_30_day_call_volume, avg_30_day_put_volume)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker, d,
             int(row.get("call_volume") or 0), int(row.get("put_volume") or 0),
             int(row.get("call_volume_ask_side") or 0), int(row.get("call_volume_bid_side") or 0),
             int(row.get("put_volume_ask_side") or 0), int(row.get("put_volume_bid_side") or 0),
             float(row.get("net_call_premium") or 0), float(row.get("net_put_premium") or 0),
             float(row.get("call_premium") or 0), float(row.get("put_premium") or 0),
             float(row.get("bearish_premium") or 0), float(row.get("bullish_premium") or 0),
             int(row.get("put_open_interest") or 0), int(row.get("call_open_interest") or 0),
             float(row.get("avg_30_day_call_volume") or 0), float(row.get("avg_30_day_put_volume") or 0)),
        )
        rows += 1

    conn.commit()
    if rows:
        log_download(conn, ticker, date, "options_volume", rows)
    return rows


def download_net_prem_ticks(conn: sqlite3.Connection, ticker: str, date: str):
    """Download intraday net premium ticks (1-min resolution, 30 day history)."""
    if is_already_downloaded(conn, ticker, date, "net_prem_ticks"):
        return 0

    data = api_call(f"stock/{ticker}/net-prem-ticks", {"date": date})
    if not data:
        return 0

    rows = 0
    for row in data:
        conn.execute(
            """INSERT OR REPLACE INTO net_prem_ticks
            (ticker, date, tape_time, call_volume, put_volume,
             call_volume_ask_side, call_volume_bid_side,
             put_volume_ask_side, put_volume_bid_side,
             net_call_premium, net_put_premium,
             net_call_volume, net_put_volume, net_delta)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker, date, row.get("tape_time", ""),
             int(row.get("call_volume") or 0), int(row.get("put_volume") or 0),
             int(row.get("call_volume_ask_side") or 0), int(row.get("call_volume_bid_side") or 0),
             int(row.get("put_volume_ask_side") or 0), int(row.get("put_volume_bid_side") or 0),
             float(row.get("net_call_premium") or 0), float(row.get("net_put_premium") or 0),
             int(row.get("net_call_volume") or 0), int(row.get("net_put_volume") or 0),
             float(row.get("net_delta") or 0)),
        )
        rows += 1

    conn.commit()
    if rows:
        log_download(conn, ticker, date, "net_prem_ticks", rows)
    return rows


def download_spot_gex(conn: sqlite3.Connection, ticker: str, date: str):
    """Download intraday spot GEX (per-minute, 30 day history)."""
    if is_already_downloaded(conn, ticker, date, "spot_gex"):
        return 0

    data = api_call(f"stock/{ticker}/spot-exposures", {"date": date})
    if not data:
        return 0

    rows = 0
    for row in data:
        conn.execute(
            """INSERT OR REPLACE INTO spot_gex
            (ticker, date, time, price,
             gamma_per_pct_oi, charm_per_pct_oi, vanna_per_pct_oi,
             gamma_per_pct_vol, charm_per_pct_vol, vanna_per_pct_vol)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ticker, date, row.get("time", ""),
             float(row.get("price") or 0),
             float(row.get("gamma_per_one_percent_move_oi") or 0),
             float(row.get("charm_per_one_percent_move_oi") or 0),
             float(row.get("vanna_per_one_percent_move_oi") or 0),
             float(row.get("gamma_per_one_percent_move_vol") or 0),
             float(row.get("charm_per_one_percent_move_vol") or 0),
             float(row.get("vanna_per_one_percent_move_vol") or 0)),
        )
        rows += 1

    conn.commit()
    if rows:
        log_download(conn, ticker, date, "spot_gex", rows)
    return rows


def download_flow_alerts(conn: sqlite3.Connection, ticker: str, date: str):
    """Download unusual flow alerts for a ticker on a date."""
    if is_already_downloaded(conn, ticker, date, "flow_alerts"):
        return 0

    data = api_call(f"stock/{ticker}/flow-alerts", {"date": date, "limit": 200})
    if not data:
        log_download(conn, ticker, date, "flow_alerts", 0)
        return 0

    rows = 0
    for row in data:
        alert_id = row.get("id") or row.get("option_chain", "") + "_" + row.get("created_at", "")
        conn.execute(
            """INSERT OR REPLACE INTO flow_alerts
            (id, ticker, created_at, type, strike, expiry, price,
             volume, open_interest, total_premium, underlying_price,
             trade_count, iv_start, iv_end, volume_oi_ratio,
             has_sweep, has_floor, has_multileg, all_opening_trades,
             alert_rule, total_bid_side_prem, total_ask_side_prem)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (alert_id, ticker, row.get("created_at", ""),
             row.get("type"), float(row.get("strike") or 0), row.get("expiry"),
             float(row.get("price") or 0),
             int(row.get("volume") or 0), int(row.get("open_interest") or 0),
             float(row.get("total_premium") or 0), float(row.get("underlying_price") or 0),
             int(row.get("trade_count") or 0),
             float(row.get("iv_start") or 0), float(row.get("iv_end") or 0),
             float(row.get("volume_oi_ratio") or 0),
             int(row.get("has_sweep", False)), int(row.get("has_floor", False)),
             int(row.get("has_multileg", False)), int(row.get("all_opening_trades", False)),
             row.get("alert_rule"), float(row.get("total_bid_side_prem") or 0),
             float(row.get("total_ask_side_prem") or 0)),
        )
        rows += 1

    conn.commit()
    log_download(conn, ticker, date, "flow_alerts", rows)
    return rows


def download_darkpool(conn: sqlite3.Connection, ticker: str, date: str):
    """Download dark pool trades for a ticker on a date."""
    if is_already_downloaded(conn, ticker, date, "darkpool"):
        return 0

    data = api_call(f"darkpool/{ticker}", {"date": date, "limit": 500})
    if not data:
        log_download(conn, ticker, date, "darkpool", 0)
        return 0

    rows = 0
    for row in data:
        tid = str(row.get("tracking_id", ""))
        if not tid:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO darkpool
            (ticker, tracking_id, executed_at, size, price, premium,
             nbbo_bid, nbbo_ask, volume, market_center)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ticker, tid, row.get("executed_at", ""),
             int(row.get("size") or 0), float(row.get("price") or 0),
             float(row.get("premium") or 0),
             float(row.get("nbbo_bid") or 0), float(row.get("nbbo_ask") or 0),
             int(row.get("volume") or 0), row.get("market_center")),
        )
        rows += 1

    conn.commit()
    log_download(conn, ticker, date, "darkpool", rows)
    return rows


def download_max_pain(conn: sqlite3.Connection, ticker: str, date: str):
    """Download max pain data for a ticker."""
    if is_already_downloaded(conn, ticker, date, "max_pain"):
        return 0

    data = api_call(f"stock/{ticker}/max-pain", {"date": date})
    if not data:
        return 0

    rows = 0
    for row in data:
        conn.execute(
            """INSERT OR REPLACE INTO max_pain
            (ticker, date, expiry, max_pain, open_price, close_price)
            VALUES (?,?,?,?,?,?)""",
            (ticker, date, row.get("expiry", ""),
             float(row.get("max_pain") or 0),
             float(row.get("open") or 0), float(row.get("close") or 0)),
        )
        rows += 1

    conn.commit()
    if rows:
        log_download(conn, ticker, date, "max_pain", rows)
    return rows


def download_congress_trades(conn: sqlite3.Connection):
    """Download recent congress trades (not ticker-specific)."""
    print("\nDownloading congress trades...", flush=True)
    data = api_call("congress/recent-trades", {"limit": 200})
    if not data:
        print("  No congress trade data", flush=True)
        return

    rows = 0
    for row in data:
        ticker = row.get("ticker")
        pid = row.get("politician_id", "")
        txn_date = row.get("transaction_date", "")
        if not pid or not txn_date:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO congress_trades
            (politician_id, ticker, transaction_date, name, txn_type,
             amounts, notes, filed_at_date, member_type)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (pid, ticker, txn_date,
             row.get("name"), row.get("txn_type"),
             row.get("amounts"), row.get("notes"),
             row.get("filed_at_date"), row.get("member_type")),
        )
        rows += 1

    conn.commit()
    print(f"  Congress trades: {rows} records", flush=True)
    time.sleep(RATE_LIMIT_DELAY)


def get_trading_days(n_days: int) -> list[str]:
    """Get last N trading days (weekdays only)."""
    days = []
    d = datetime.now().date()
    while len(days) < n_days:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(days))


def main():
    parser = argparse.ArgumentParser(description="Download Unusual Whales historical data")
    parser.add_argument("--tickers", nargs="+", default=TICKERS, help="Tickers to download")
    parser.add_argument("--days", type=int, default=30, help="Number of trading days to download")
    parser.add_argument("--daily-only", action="store_true", help="Skip intraday endpoints (faster)")
    parser.add_argument("--db", default=DB_PATH, help="Database path")
    args = parser.parse_args()

    conn = init_db(args.db)
    trading_days = get_trading_days(args.days)

    print("=" * 70, flush=True)
    print("UNUSUAL WHALES HISTORICAL DOWNLOAD", flush=True)
    print("=" * 70, flush=True)
    print(f"Tickers:    {', '.join(args.tickers)}", flush=True)
    print(f"Days:       {len(trading_days)} ({trading_days[0]} to {trading_days[-1]})", flush=True)
    print(f"DB:         {args.db}", flush=True)
    print(f"Intraday:   {'skip' if args.daily_only else 'include'}", flush=True)
    print("=" * 70, flush=True)

    # 1. Daily greek exposure (full history, one call per ticker)
    print("\n--- DAILY GREEK EXPOSURE (~1yr) ---", flush=True)
    for ticker in args.tickers:
        download_greek_exposure(conn, ticker)

    # 2. Congress trades (not ticker-specific)
    download_congress_trades(conn)

    # 3. Per-ticker, per-day endpoints
    total_tickers = len(args.tickers)
    total_days = len(trading_days)

    for di, date in enumerate(trading_days):
        print(f"\n[Day {di + 1}/{total_days}] {date}:", flush=True)
        day_rows = 0

        for ti, ticker in enumerate(args.tickers):
            ticker_rows = 0

            # Daily endpoints (should work for all 30 days)
            r = download_options_volume(conn, ticker, date)
            ticker_rows += r
            time.sleep(RATE_LIMIT_DELAY)

            r = download_max_pain(conn, ticker, date)
            ticker_rows += r
            time.sleep(RATE_LIMIT_DELAY)

            r = download_flow_alerts(conn, ticker, date)
            ticker_rows += r
            time.sleep(RATE_LIMIT_DELAY)

            r = download_darkpool(conn, ticker, date)
            ticker_rows += r
            time.sleep(RATE_LIMIT_DELAY)

            if not args.daily_only:
                # Intraday endpoints (30 trading day limit)
                r = download_net_prem_ticks(conn, ticker, date)
                ticker_rows += r
                time.sleep(RATE_LIMIT_DELAY)

                r = download_spot_gex(conn, ticker, date)
                ticker_rows += r
                time.sleep(RATE_LIMIT_DELAY)

            if ticker_rows > 0:
                print(f"  {ticker}: {ticker_rows} rows", flush=True)
            day_rows += ticker_rows

        print(f"  Day total: {day_rows} rows", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("DOWNLOAD COMPLETE", flush=True)
    print("=" * 70, flush=True)
    for table in ["greek_exposure", "options_volume", "net_prem_ticks", "spot_gex",
                   "flow_alerts", "darkpool", "max_pain", "congress_trades"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
