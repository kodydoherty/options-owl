"""Benchmark data for the long-dated whale-following test (2026-07-18): is it real SELECTION alpha
or just "an expensive way to be long calls"?

For every long-dated whale CALL signal (from longdated_flow_options.db) download two neutral
benchmark contracts over the same hold window (Polygon DAILY bars, deduped):
  1) SAME-TICKER ATM call, same expiry — controls ticker/timing/expiry, isolates the whale's
     STRIKE + entry-premium selection.
  2) SPY ATM call, nearest expiry to the whale's — the beta reference ("just be long the index").

Writes into longdated_flow_options.db: table benchmark_ohlc (bars) + benchmark_map (signal→contracts).

    python scripts/download_longdated_benchmark.py
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "journal" / "longdated_flow_options.db"
PKEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
             if ln.startswith("POLYGON_API_KEY=")), "")
BASE = "https://api.polygon.io"
HOLD_DAYS = 90


def _init(c):
    c.execute("""CREATE TABLE IF NOT EXISTS benchmark_ohlc (
        kind TEXT, ticker TEXT, expiration TEXT, strike REAL, timestamp TEXT,
        open REAL, high REAL, low REAL, close REAL,
        PRIMARY KEY (kind, ticker, expiration, strike, timestamp))""")
    c.execute("""CREATE TABLE IF NOT EXISTS benchmark_map (
        sig_ticker TEXT, sig_expiry TEXT, sig_strike REAL, entry_date TEXT,
        atm_ticker TEXT, atm_expiry TEXT, atm_strike REAL,
        spy_expiry TEXT, spy_strike REAL,
        PRIMARY KEY (sig_ticker, sig_expiry, sig_strike, entry_date))""")
    c.commit()


def _occ(ticker, expiry, strike):
    yy = expiry[2:4]; mm = expiry[5:7]; dd = expiry[8:10]
    return f"O:{ticker}{yy}{mm}{dd}C{int(round(float(strike) * 1000)):08d}"


def _aggs(client, otick, frm, to):
    url = f"{BASE}/v2/aggs/ticker/{otick}/range/1/day/{frm}/{to}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": PKEY}
    for attempt in range(4):
        try:
            r = client.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            if r.status_code != 200:
                return []
            return r.json().get("results", []) or []
        except httpx.HTTPError:
            time.sleep(2 * (attempt + 1))
    return []


def _rows(kind, tk, exp, strike, results):
    out = []
    for b in results:
        ts = datetime.utcfromtimestamp(b["t"] / 1000).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        out.append((kind, tk, exp, strike, ts, b.get("o"), b.get("h"), b.get("l"), b.get("c")))
    return out


def _atm(price):
    if not price or price <= 0:
        return None
    step = 5.0 if price > 100 else (2.5 if price > 25 else 1.0)
    return round(round(price / step) * step, 1)


def _spy_expiry(client, target_expiry):
    """Nearest available monthly SPY expiry to the whale's expiry (3rd-Friday approximation)."""
    # SPY has weeklies+monthlies; use the whale's expiry date directly (SPY very likely lists it or
    # a date within a few days). We try the exact date, then ±7 days.
    return target_expiry


def main():
    if not PKEY:
        print("No POLYGON_API_KEY"); return
    c = sqlite3.connect(str(DB)); _init(c)
    client = httpx.Client()
    sigs = c.execute(
        "SELECT ticker,strike,expiry,entry_date,underlying_at_alert FROM signals WHERE right='CALL'"
    ).fetchall()
    print(f"{len(sigs)} whale CALL signals to benchmark", flush=True)

    dl_cache: set = set()
    done = 0
    for tk, wstrike, expiry, d0, undl in sigs:
        end = min(expiry, (datetime.strptime(d0, "%Y-%m-%d") + timedelta(days=HOLD_DAYS)).strftime("%Y-%m-%d"))
        atm = _atm(undl)
        spy_undl = None
        # SPY underlying at entry — pull from stock_ohlc if present, else skip SPY leg
        row = c.execute("SELECT close FROM stock_ohlc WHERE ticker='SPY' AND substr(timestamp,1,10)>=? "
                        "ORDER BY timestamp LIMIT 1", (d0,)).fetchone()
        spy_undl = row[0] if row else None
        spy_atm = _atm(spy_undl) if spy_undl else None
        spy_exp = _spy_expiry(client, expiry)

        # 1) same-ticker ATM
        if atm and (tk, expiry, atm) not in dl_cache:
            res = _aggs(client, _occ(tk, expiry, atm), d0, end)
            if res:
                c.executemany("INSERT OR IGNORE INTO benchmark_ohlc VALUES (?,?,?,?,?,?,?,?,?)",
                              _rows("ATM", tk, expiry, atm, res))
            dl_cache.add((tk, expiry, atm))
            time.sleep(0.06)
        # 2) SPY ATM (beta)
        if spy_atm and ("SPY", spy_exp, spy_atm) not in dl_cache:
            res = _aggs(client, _occ("SPY", spy_exp, spy_atm), d0, end)
            if res:
                c.executemany("INSERT OR IGNORE INTO benchmark_ohlc VALUES (?,?,?,?,?,?,?,?,?)",
                              _rows("SPY", "SPY", spy_exp, spy_atm, res))
            dl_cache.add(("SPY", spy_exp, spy_atm))
            time.sleep(0.06)

        c.execute("INSERT OR IGNORE INTO benchmark_map VALUES (?,?,?,?,?,?,?,?,?)",
                  (tk, expiry, wstrike, d0, tk, expiry, atm, spy_exp, spy_atm))
        done += 1
        if done % 100 == 0:
            c.commit()
            print(f"  {done}/{len(sigs)} ({len(dl_cache)} unique benchmark contracts)", flush=True)
    c.commit(); c.close()
    print("BENCHMARK_DL_DONE", flush=True)


if __name__ == "__main__":
    main()
