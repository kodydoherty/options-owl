"""Download the whale's ACTUAL long-dated (>30 DTE) contracts for the 15-20% "follow-the-whale
LEAP-ish sleeve" test (2026-07-18, Kody's idea).

Signals come from our LOCAL flow history (journal/uw_historical.db `flow_alerts`) — no UW API
dependency. Filter = our live whale filter (ask-side sweep, >$250k premium) BUT long-dated
(DTE > MIN_DTE). For each whale contract we pull DAILY option + underlying bars (a weeks-long hold
doesn't need minute resolution → far lighter) over [entry, min(expiry, entry+HOLD_DAYS)] from
Polygon aggregates, into journal/longdated_flow_options.db. Resumable (INSERT OR IGNORE), Polygon
(NOT ThetaData) so it runs parallel to the ticker-expansion Theta downloads with no contention.

    python scripts/download_longdated_flow.py
"""
from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
FLOW_DB = ROOT / "journal" / "uw_historical.db"
OUT_DB = ROOT / "journal" / "longdated_flow_options.db"
PKEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
             if ln.startswith("POLYGON_API_KEY=")), "")
BASE = "https://api.polygon.io"

MIN_DTE = 30            # long-dated only (the untested regime; <30 DTE ≈ the refuted multi-day)
MIN_PREMIUM = 250_000   # our live whale filter
HOLD_DAYS = 90          # cap the download/hold window at 90 calendar days (or expiry, whichever first)


def _init():
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(OUT_DB))
    c.execute("""CREATE TABLE IF NOT EXISTS option_ohlc (
        ticker TEXT, expiration TEXT, strike REAL, right TEXT, timestamp TEXT,
        open REAL, high REAL, low REAL, close REAL, volume INTEGER, vwap REAL,
        PRIMARY KEY (ticker, expiration, strike, right, timestamp))""")
    c.execute("""CREATE TABLE IF NOT EXISTS stock_ohlc (
        ticker TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL,
        volume INTEGER, vwap REAL, PRIMARY KEY (ticker, timestamp))""")
    # Persist the signal set so the backtest reads exactly what we downloaded.
    c.execute("""CREATE TABLE IF NOT EXISTS signals (
        ticker TEXT, right TEXT, strike REAL, expiry TEXT, entry_date TEXT, entry_time TEXT,
        entry_premium REAL, total_premium REAL, dte INTEGER, underlying_at_alert REAL,
        PRIMARY KEY (ticker, right, strike, expiry, entry_date))""")
    c.commit()
    return c


def _occ(ticker, expiry, strike, is_put):
    yy = expiry[2:4]; mm = expiry[5:7]; dd = expiry[8:10]
    cp = "P" if is_put else "C"
    return f"O:{ticker}{yy}{mm}{dd}{cp}{int(round(float(strike) * 1000)):08d}"


def _aggs(client, otick, frm, to, span="day"):
    url = f"{BASE}/v2/aggs/ticker/{otick}/range/1/{span}/{frm}/{to}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": PKEY}
    for attempt in range(5):
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


def _rows(results):
    out = []
    for b in results:
        ts = datetime.utcfromtimestamp(b["t"] / 1000).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        out.append((ts, b.get("o"), b.get("h"), b.get("l"), b.get("c"), b.get("v"), b.get("vw")))
    return out


def _gather_signals():
    """Long-dated whale sweeps from the local flow history."""
    fc = sqlite3.connect(str(FLOW_DB))
    rows = fc.execute(
        """SELECT ticker, type, strike, expiry, created_at, price, total_premium,
                  underlying_price,
                  CAST(julianday(expiry)-julianday(date(created_at)) AS INT) AS dte,
                  total_ask_side_prem, total_bid_side_prem
           FROM flow_alerts
           WHERE has_sweep=1 AND total_premium > ?
             AND julianday(expiry)-julianday(date(created_at)) > ?
             AND expiry IS NOT NULL AND strike IS NOT NULL
             AND COALESCE(total_ask_side_prem,0) >= COALESCE(total_bid_side_prem,0)
           """,
        (MIN_PREMIUM, MIN_DTE),
    ).fetchall()
    fc.close()
    # de-dup to the EARLIEST alert per contract (first time the whale showed up)
    best: dict[tuple, tuple] = {}
    for r in rows:
        tk, typ, strike, expiry, created, price, tot, undl, dte, ask, bid = r
        right = "PUT" if str(typ).lower().startswith("p") else "CALL"
        key = (tk, right, float(strike), str(expiry)[:10])
        d0 = str(created)[:10]
        if key not in best or d0 < best[key][4]:
            best[key] = (tk, right, float(strike), str(expiry)[:10], d0, str(created)[:19],
                         price, tot, int(dte), undl)
    return list(best.values())


def main():
    if not PKEY:
        print("No POLYGON_API_KEY in .env"); return
    con = _init()
    client = httpx.Client()

    sigs = _gather_signals()
    print(f"{len(sigs)} unique long-dated whale contracts (DTE>{MIN_DTE}, >${MIN_PREMIUM/1e3:.0f}k, "
          f"ask-side sweep)", flush=True)
    con.executemany(
        "INSERT OR IGNORE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?)", sigs)
    con.commit()

    # underlying daily bars per ticker over the union window
    stock_win: dict[str, list] = {}
    done = 0
    for (tk, right, strike, expiry, d0, created, price, tot, dte, undl) in sigs:
        is_put = right == "PUT"
        end = min(expiry, (datetime.strptime(d0, "%Y-%m-%d")
                           + timedelta(days=HOLD_DAYS)).strftime("%Y-%m-%d"))
        otick = _occ(tk, expiry, strike, is_put)
        res = _aggs(client, otick, d0, end, span="day")
        if res:
            con.executemany(
                "INSERT OR IGNORE INTO option_ohlc VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(tk, expiry, strike, right, ts, o, h, l, c, v, vw)
                 for (ts, o, h, l, c, v, vw) in _rows(res)])
        stock_win.setdefault(tk, [d0, end])
        stock_win[tk][0] = min(stock_win[tk][0], d0)
        stock_win[tk][1] = max(stock_win[tk][1], end)
        done += 1
        if done % 25 == 0:
            con.commit()
            print(f"  {done}/{len(sigs)} contracts ({tk} {expiry} ${strike:g}{right[0]}: "
                  f"{len(res)} daily bars)", flush=True)
        time.sleep(0.08)
    con.commit()
    print(f"option bars done ({done} contracts)", flush=True)

    # underlying daily bars
    for i, (tk, (lo, hi)) in enumerate(stock_win.items(), 1):
        res = _aggs(client, tk, lo, hi, span="day")
        if res:
            con.executemany(
                "INSERT OR IGNORE INTO stock_ohlc VALUES (?,?,?,?,?,?,?,?)",
                [(tk, ts, o, h, l, c, v, vw) for (ts, o, h, l, c, v, vw) in _rows(res)])
        if i % 10 == 0:
            con.commit()
            print(f"  underlying {i}/{len(stock_win)}", flush=True)
        time.sleep(0.08)
    con.commit()
    con.close()
    print("LONGDATED_DL_DONE", flush=True)


if __name__ == "__main__":
    main()
