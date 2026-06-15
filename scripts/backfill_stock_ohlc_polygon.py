"""Backfill 1-min stock OHLC into thetadata_options.db.stock_ohlc from Polygon (any ticker).

Needed because download_thetadata.py --ohlc-only SKIPS greeks, and stock_ohlc is normally derived
from greeks (extract_stock_from_greeks). The flow discovery reads stock_ohlc.close, so the new
flow tickers had no underlying series. This pulls real 1-min bars from Polygon directly.

Usage:
    python scripts/backfill_stock_ohlc_polygon.py --tickers MU,SMH,MRVL,TSM,INTC,ORCL,QCOM \
        --start 2026-03-01 --end 2026-06-12
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "journal" / "thetadata_options.db")
ET = ZoneInfo("America/New_York")
KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
            if ln.startswith("POLYGON_API_KEY=")), "")


def fetch_day(ticker: str, day: str) -> list[tuple]:
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{day}/{day}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={KEY}")
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(3 * (attempt + 1)); continue
        if r.status_code == 429:
            time.sleep(12); continue
        if r.status_code != 200:
            return []
        rows = []
        for bar in r.json().get("results", []) or []:
            dt_et = datetime.fromtimestamp(bar["t"] / 1000, tz=UTC).astimezone(ET)
            if dt_et.hour < 9 or (dt_et.hour == 9 and dt_et.minute < 30) or dt_et.hour >= 16:
                continue
            ts = dt_et.strftime("%Y-%m-%d %H:%M:%S%z")
            ts = ts[:-2] + ":" + ts[-2:]  # +0500 -> +05:00 style to match existing rows
            rows.append((ticker, ts, bar["o"], bar["h"], bar["l"], bar["c"],
                         int(bar.get("v", 0)), bar.get("vw", 0.0)))
        return rows
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    d0 = datetime.strptime(args.start, "%Y-%m-%d").date()
    d1 = datetime.strptime(args.end, "%Y-%m-%d").date()

    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    for tk in tickers:
        total = 0
        d = d0
        while d <= d1:
            if d.weekday() < 5:  # skip weekends
                rows = fetch_day(tk, d.isoformat())
                if rows:
                    con.executemany(
                        "INSERT OR IGNORE INTO stock_ohlc "
                        "(ticker,timestamp,open,high,low,close,volume,vwap) VALUES (?,?,?,?,?,?,?,?)",
                        rows)
                    con.commit()
                    total += len(rows)
            d += timedelta(days=1)
        print(f"  {tk}: {total} 1-min stock bars inserted", flush=True)
    con.close()
    print("stock_ohlc backfill complete")


if __name__ == "__main__":
    main()
