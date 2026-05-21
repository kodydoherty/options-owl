"""Download missing harvester data for 10 signals from Polygon snapshots API.

Fetches intraday option quotes for specific contracts that the harvester missed.
Stores them in the same harvest_snapshots table so backtests can use them.
"""
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import json
import urllib.request

API_KEY = os.getenv("POLYGON_API_KEY", "Zi2nVXh9YJdPtfmuQRScmecxj3IlSpET")
HARVESTER_DB = "journal/owlet-harvester/options_data.db"
BASE_URL = "https://api.polygon.io"
DELAY = 0.5  # seconds between requests

# Contracts to download: (ticker, strike, opt_type, signal_date)
# These have no harvester data under any nearby expiry
MISSING = [
    ("MSTR", 171.0, "C", "2026-04-21"),
    ("AVGO", 402.5, "C", "2026-04-21"),
    ("AVGO", 412.5, "C", "2026-04-22"),
    ("AVGO", 420.0, "C", "2026-04-22"),
    ("AVGO", 422.5, "C", "2026-04-23"),
    ("AVGO", 430.0, "C", "2026-04-23"),
    ("AVGO", 427.5, "C", "2026-04-23"),
    ("MSTR", 177.0, "P", "2026-04-23"),
    ("AVGO", 417.5, "P", "2026-04-24"),
]


def build_contract_ticker(ticker, strike, opt_type, expiry_date):
    dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    strike_int = int(strike * 1000)
    return "O:{t}{d}{o}{s:08d}".format(t=ticker, d=date_str, o=opt_type, s=strike_int)


def _get_json(url):
    """Fetch JSON from URL using stdlib."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        return {}, e.code
    except Exception:
        return {}, 0


def find_valid_expiry(ticker, strike, opt_type, signal_date):
    """Try signal date and next 4 business days to find valid contract."""
    base = datetime.strptime(signal_date, "%Y-%m-%d").date()
    for delta in range(0, 5):
        try_date = base + timedelta(days=delta)
        if try_date.weekday() >= 5:
            continue
        ct = build_contract_ticker(ticker, strike, opt_type, try_date.strftime("%Y-%m-%d"))
        # Check via aggregates (snapshot needs higher tier)
        url = "{}/v2/aggs/ticker/{}/range/1/minute/{}/{}?adjusted=true&sort=asc&limit=1&apiKey={}".format(
            BASE_URL, ct, signal_date, signal_date, API_KEY
        )
        time.sleep(DELAY)
        data, status = _get_json(url)
        if status == 200 and data.get("resultsCount", 0) > 0:
            return ct, try_date.strftime("%Y-%m-%d")
    return None, None


def fetch_trades(contract_ticker, trade_date):
    """Fetch intraday 1-min bars for a contract on a specific date."""
    url = "{}/v2/aggs/ticker/{}/range/1/minute/{}/{}?adjusted=true&sort=asc&limit=50000&apiKey={}".format(
        BASE_URL, contract_ticker, trade_date, trade_date, API_KEY
    )
    time.sleep(DELAY)
    data, status = _get_json(url)
    if status != 200:
        print("    HTTP {} for {}".format(status, contract_ticker))
        return []
    return data.get("results", [])


def fetch_underlying_price(ticker, trade_date):
    """Fetch underlying stock 1-min bars for the day."""
    url = "{}/v2/aggs/ticker/{}/range/1/minute/{}/{}?adjusted=true&sort=asc&limit=50000&apiKey={}".format(
        BASE_URL, ticker, trade_date, trade_date, API_KEY
    )
    time.sleep(DELAY)
    data, status = _get_json(url)
    if status != 200:
        return {}
    bars = data.get("results", [])
    result = {}
    for bar in bars:
        ts = bar["t"] // 1000
        result[ts // 60] = bar["c"]
    return result


def main():
    conn = sqlite3.connect(HARVESTER_DB)

    total_inserted = 0

    for ticker, strike, opt_type, signal_date in MISSING:
        print("Processing {} ${} {} date={}...".format(ticker, strike, opt_type, signal_date))

        # Try to find valid contract (may be different expiry)
        ct, real_expiry = find_valid_expiry(ticker, strike, opt_type, signal_date)
        if not ct:
            print("  No valid contract found, skipping")
            continue

        print("  Contract: {} (expiry={})".format(ct, real_expiry))

        # Ensure contract row exists
        conn.execute("""
            INSERT OR IGNORE INTO harvest_contracts
            (contract_ticker, underlying, strike, expiry_date, option_type, first_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ct, ticker, strike, real_expiry,
              "call" if opt_type == "C" else "put", signal_date + "T00:00:00"))

        # Check if already in DB
        existing = conn.execute(
            "SELECT COUNT(*) FROM harvest_snapshots WHERE contract_ticker = ?", (ct,)
        ).fetchone()[0]
        if existing > 0:
            print("  Already have {} snapshots, skipping".format(existing))
            continue

        # Fetch option bars for signal date
        bars = fetch_trades(ct, signal_date)
        if not bars:
            print("  No bars found for {}".format(signal_date))
            continue

        # Fetch underlying prices
        underlying_prices = fetch_underlying_price(ticker, signal_date)

        # Insert as harvest_snapshots rows
        inserted = 0
        for bar in bars:
            ts_ms = bar["t"]
            ts_s = ts_ms // 1000
            captured = datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

            mid = (bar.get("o", 0) + bar.get("c", 0)) / 2
            ul_price = underlying_prices.get(ts_s // 60)

            conn.execute("""
                INSERT INTO harvest_snapshots
                (contract_ticker, captured_at, underlying_price,
                 bid, ask, midpoint, day_open, day_high, day_low, day_close,
                 day_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ct, captured, ul_price,
                bar.get("l"), bar.get("h"), mid,
                bar.get("o"), bar.get("h"), bar.get("l"), bar.get("c"),
                bar.get("v"),
            ))
            inserted += 1

        conn.commit()
        total_inserted += inserted
        print("  Inserted {} snapshots".format(inserted))

    conn.close()
    print("\nDone. Total inserted: {}".format(total_inserted))


if __name__ == "__main__":
    main()
