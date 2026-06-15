"""Download the WHALE'S ACTUAL contracts (multi-day expiries) for historical flow alerts.

The 0DTE thetadata DB only has 0-4 DTE, so multi-day flow (the whale's real far-dated expiry)
was never testable. This pulls, for each historical whale ask-side sweep on the flow whitelist,
the exact contract the whale bought (ticker/expiry/strike from the OCC option_chain) — option +
underlying MINUTE bars over a bounded holding window — into journal/multiday_flow_options.db
(same schema as thetadata_options.db, so scripts/backtest_multiday_flow.py reuses the V7 sim).

Polygon option-aggregates. Resumable (INSERT OR IGNORE). Run in background:
    python scripts/download_multiday_flow.py
"""
from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "journal" / "multiday_flow_options.db"
PKEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
             if ln.startswith("POLYGON_API_KEY=")), "")
PUT_UNIV = D.CUR_PUT | {"SPY"}
CALL_UNIV = D.CUR_CALL
HOLD_DAYS = 12          # calendar days of bars per contract (caps download; V7 exits fire inside)
BASE = "https://api.polygon.io"


def _init():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS option_ohlc (
        ticker TEXT, expiration TEXT, strike REAL, right TEXT, timestamp TEXT,
        open REAL, high REAL, low REAL, close REAL, volume INTEGER, vwap REAL,
        PRIMARY KEY (ticker, expiration, strike, right, timestamp))""")
    c.execute("""CREATE TABLE IF NOT EXISTS stock_ohlc (
        ticker TEXT, timestamp TEXT, open REAL, high REAL, low REAL, close REAL,
        volume INTEGER, vwap REAL, PRIMARY KEY (ticker, timestamp))""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_opt ON option_ohlc(ticker, expiration)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_stk ON stock_ohlc(ticker, timestamp)")
    c.commit()
    return c


def _occ(ticker, expiry, strike, is_put):
    yy = expiry[2:4]; mm = expiry[5:7]; dd = expiry[8:10]
    cp = "P" if is_put else "C"
    return f"O:{ticker}{yy}{mm}{dd}{cp}{int(round(float(strike) * 1000)):08d}"


def _aggs(client, otick, frm, to):
    url = f"{BASE}/v2/aggs/ticker/{otick}/range/1/minute/{frm}/{to}"
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


def _rows_to_iso(results):
    out = []
    for b in results:
        ts = datetime.utcfromtimestamp(b["t"] / 1000).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        out.append((ts, b.get("o"), b.get("h"), b.get("l"), b.get("c"), b.get("v"), b.get("vw")))
    return out


def main():
    if not PKEY:
        print("No POLYGON_API_KEY in .env"); return
    con = _init()
    client = httpx.Client()

    # 1) Gather whale contracts from flow alerts (both sides, whitelist)
    contracts: dict[tuple, dict] = {}
    stock_window: dict[str, list] = {}
    for is_put, wl in ((True, PUT_UNIV), (False, CALL_UNIV)):
        D.UNIVERSE = list(wl)
        sig = D.fetch_sweeps(is_put)
        if sig.empty:
            continue
        sig = sig[sig["ticker"].isin(wl)]
        for _, ev in sig.iterrows():
            tk = ev["ticker"]; exp = str(ev["expiry"]); strike = float(ev["strike"])
            key = (tk, exp, strike, is_put)
            d0 = ev["date"]
            if key not in contracts or d0 < contracts[key]["d0"]:
                contracts.setdefault(key, {"d0": d0})
                contracts[key]["d0"] = min(d0, contracts[key]["d0"])
            stock_window.setdefault(tk, [d0, d0])
            stock_window[tk][0] = min(stock_window[tk][0], d0)
            stock_window[tk][1] = max(stock_window[tk][1], exp[:10])
    print(f"{len(contracts)} unique whale contracts across {len(stock_window)} tickers", flush=True)

    # 2) Download each whale contract's option bars over [d0, min(expiry, d0+HOLD_DAYS)]
    done = 0
    for (tk, exp, strike, is_put), meta in contracts.items():
        d0 = meta["d0"]
        end = min(exp[:10], (datetime.strptime(d0, "%Y-%m-%d") + timedelta(days=HOLD_DAYS)).strftime("%Y-%m-%d"))
        otick = _occ(tk, exp, strike, is_put)
        right = "PUT" if is_put else "CALL"
        res = _aggs(client, otick, d0, end)
        if res:
            con.executemany(
                "INSERT OR IGNORE INTO option_ohlc VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(tk, exp, strike, right, ts, o, h, l, c, v, vw)
                 for (ts, o, h, l, c, v, vw) in _rows_to_iso(res)])
        done += 1
        if done % 50 == 0:
            con.commit()
            print(f"  {done}/{len(contracts)} contracts ({tk} {exp} ${strike:g}{right[0]}: {len(res)} bars)", flush=True)
        time.sleep(0.08)
    con.commit()
    print(f"option bars done ({done} contracts)", flush=True)

    # 3) Download underlying minute bars per ticker over the union window (chunked by month)
    for tk, (lo, hi) in stock_window.items():
        cur = datetime.strptime(lo, "%Y-%m-%d")
        hi_d = datetime.strptime(hi, "%Y-%m-%d")
        total = 0
        while cur <= hi_d:
            chunk_end = min(cur + timedelta(days=30), hi_d)
            url = f"{BASE}/v2/aggs/ticker/{tk}/range/1/minute/{cur.strftime('%Y-%m-%d')}/{chunk_end.strftime('%Y-%m-%d')}"
            for attempt in range(5):
                try:
                    r = client.get(url, params={"adjusted": "true", "sort": "asc",
                                                "limit": 50000, "apiKey": PKEY}, timeout=30)
                    if r.status_code == 429:
                        time.sleep(2 * (attempt + 1)); continue
                    res = r.json().get("results", []) if r.status_code == 200 else []
                    break
                except httpx.HTTPError:
                    time.sleep(2 * (attempt + 1)); res = []
            if res:
                con.executemany(
                    "INSERT OR IGNORE INTO stock_ohlc VALUES (?,?,?,?,?,?,?,?)",
                    [(tk, ts, o, h, l, c, v, vw) for (ts, o, h, l, c, v, vw) in _rows_to_iso(res)])
                total += len(res)
            cur = chunk_end + timedelta(days=1)
            time.sleep(0.1)
        con.commit()
        print(f"  stock {tk}: {total} bars", flush=True)

    no = con.execute("SELECT COUNT(*) FROM option_ohlc").fetchone()[0]
    ns = con.execute("SELECT COUNT(*) FROM stock_ohlc").fetchone()[0]
    print(f"DONE → {DB}  ({no:,} option bars, {ns:,} stock bars)", flush=True)
    con.close()


if __name__ == "__main__":
    main()
