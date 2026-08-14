"""Download ThetaData option history via the v3 REST API.

WHY THIS REPLACES download_thetadata.py
---------------------------------------
The old script used `thetadata` SDK v1.0.4, which speaks gRPC to a ThetaTerminal v1.
That stack is three versions behind: the account now issues `td1_prod_*` API keys, the
terminal is v3, the REST API is v3, and v3 renamed `root` to `symbol`.

The failure was SILENT. The SDK authenticated ("Connected!"), every request returned
nothing, and the script reported "Batch done: 0 rows" and exited 0. thetadata therefore
stopped at 2026-07-15 while appearing healthy, so every backtest run afterwards was
scored on data that predates the code being evaluated -- which is why the harness and
live results diverged so badly and could not be reconciled.

Two guards exist because of that:
  * a terminal preflight, so a missing/old terminal fails LOUDLY instead of returning 0 rows
  * a post-run assertion that MAX(date) actually advanced; a "successful" run that adds
    nothing now exits non-zero

ENDPOINT CHOICE
---------------
`/v3/option/history/greeks/first_order` returns bid, ask, delta, theta, vega,
implied_vol AND underlying_price in ONE call, and is included in the Standard tier.
`greeks/all` needs Professional and we do not use second/third-order greeks. The old
script made three round trips (ohlc + quote + greeks) to assemble the same row.

Usage:
  java -jar ThetaTerminalv3.jar --creds-file creds.txt &     # must be running
  python scripts/download_thetadata_v3.py --start 2026-07-16 --end 2026-08-13
"""

from __future__ import annotations

import argparse
import csv
import io
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests

BASE = "http://127.0.0.1:25503/v3"
DB_PATH = "journal/thetadata_options.db"
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "META", "AAPL", "AMZN",
                   "GOOGL", "AMD", "MSTR", "PLTR", "NFLX", "SMCI", "BA", "JPM"]
STRIKE_WINDOW = 4          # strikes above/below ATM, matching the old script's OTM depth
REQUEST_PAUSE = 0.05       # terminal allows 4 concurrent; stay well under


def preflight() -> None:
    """Fail LOUDLY if the terminal is absent. A silent 0-row run is the bug we are fixing."""
    try:
        r = requests.get(f"{BASE}/option/list/expirations", params={"symbol": "SPY"}, timeout=15)
    except requests.RequestException as exc:
        sys.exit(
            f"FATAL: no ThetaTerminal at {BASE} ({exc}).\n"
            "Start it first:  java -jar ThetaTerminalv3.jar --creds-file creds.txt &\n"
            "Without it every request returns empty and the download silently does nothing."
        )
    if r.status_code != 200 or "expiration" not in r.text[:200]:
        sys.exit(f"FATAL: terminal responded but not as expected:\n{r.text[:300]}")
    print(f"  terminal OK at {BASE}")


def get_csv(path: str, params: dict) -> list[dict]:
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=60)
        except requests.RequestException:
            time.sleep(1 + attempt)
            continue
        if r.status_code != 200:
            return []
        txt = r.text
        if not txt or "No data found" in txt[:60] or "subscription" in txt[:120]:
            return []
        return list(csv.DictReader(io.StringIO(txt)))
    return []


def trading_days(start: date, end: date) -> list[date]:
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def nearest_expirations(sym: str, day: date, max_dte: int) -> list[str]:
    rows = get_csv("option/list/expirations", {"symbol": sym})
    exps = sorted({r["expiration"].strip('"') for r in rows if r.get("expiration")})
    out = []
    for e in exps:
        try:
            ed = datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (ed - day).days
        if 0 <= dte <= max_dte:
            out.append(e)
    return out


def atm_strikes(sym: str, exp: str, day: date) -> list[float]:
    rows = get_csv("option/list/strikes", {"symbol": sym, "expiration": exp})
    ks = sorted({float(r["strike"]) for r in rows if r.get("strike")})
    if not ks:
        return []
    # underlying reference from any contract's greeks row
    probe = get_csv("option/history/greeks/first_order", {
        "symbol": sym, "expiration": exp, "strike": ks[len(ks) // 2], "right": "CALL",
        "start_date": day.isoformat(), "end_date": day.isoformat(), "interval": "1m"})
    spot = 0.0
    for r in probe:
        try:
            spot = float(r.get("underlying_price") or 0)
            if spot:
                break
        except ValueError:
            pass
    if not spot:
        return []
    ks.sort(key=lambda k: abs(k - spot))
    return sorted(ks[: STRIKE_WINDOW * 2 + 1])


def store(con: sqlite3.Connection, sym: str, rows: list[dict]) -> int:
    g, o = [], []
    for r in rows:
        try:
            exp = r["expiration"].strip('"')
            k = float(r["strike"])
            right = r["right"].strip('"')[0].upper()  # CALL -> C, matching legacy rows
            ts = r["timestamp"].strip('"')
            g.append((sym, exp, k, right, ts,
                      _f(r.get("bid")), _f(r.get("ask")), _f(r.get("delta")),
                      _f(r.get("theta")), _f(r.get("vega")), _f(r.get("implied_vol")),
                      _f(r.get("underlying_price"))))
            mid = (_f(r.get("bid")) + _f(r.get("ask"))) / 2
            o.append((sym, exp, k, right, ts, mid, mid, mid, mid, 0, mid))
        except (KeyError, ValueError, TypeError):
            continue
    if g:
        con.executemany(
            "INSERT OR REPLACE INTO option_greeks (ticker,expiration,strike,right,timestamp,"
            "bid,ask,delta,theta,vega,implied_vol,underlying_price) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", g)
        con.executemany(
            "INSERT OR REPLACE INTO option_ohlc (ticker,expiration,strike,right,timestamp,"
            "open,high,low,close,volume,vwap) VALUES (?,?,?,?,?,?,?,?,?,?,?)", o)
    return len(g)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--ticker", default=",".join(DEFAULT_TICKERS))
    ap.add_argument("--max-dte", type=int, default=4)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    print("=" * 66)
    print("THETADATA v3 DOWNLOAD")
    print("=" * 66)
    preflight()

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    days = trading_days(start, end)
    print(f"  {len(tickers)} tickers x {len(days)} trading days, DTE<={args.max_dte}\n")

    con = sqlite3.connect(args.db)
    before = con.execute("SELECT COALESCE(MAX(date),'') FROM download_log").fetchone()[0]
    total = 0
    try:
        for day in days:
            dstr = day.isoformat()
            day_rows = 0
            for sym in tickers:
                done = con.execute(
                    "SELECT rows_downloaded FROM download_log WHERE ticker=? AND date=? AND data_type=?",
                    (sym, dstr, "greeks_v3")).fetchone()
                if done and done[0]:
                    continue
                n = 0
                for exp in nearest_expirations(sym, day, args.max_dte):
                    for k in atm_strikes(sym, exp, day):
                        for right in ("CALL", "PUT"):
                            rows = get_csv("option/history/greeks/first_order", {
                                "symbol": sym, "expiration": exp, "strike": k, "right": right,
                                "start_date": dstr, "end_date": dstr, "interval": "1m"})
                            n += store(con, sym, rows)
                            time.sleep(REQUEST_PAUSE)
                con.execute(
                    "INSERT OR REPLACE INTO download_log (ticker,date,data_type,rows_downloaded,downloaded_at)"
                    " VALUES (?,?,?,?,?)",
                    (sym, dstr, "greeks_v3", n, datetime.now(timezone.utc).isoformat()))
                con.commit()
                day_rows += n
            total += day_rows
            print(f"  {dstr}: {day_rows:>7,} rows   (running {total:,})", flush=True)
    finally:
        con.commit()
        after = con.execute("SELECT COALESCE(MAX(date),'') FROM download_log").fetchone()[0]
        con.close()

    print(f"\n  total rows: {total:,}   MAX(date) {before} -> {after}")
    # The assertion that would have caught the silent failure a month ago.
    if total == 0:
        sys.exit("FATAL: downloaded 0 rows. The terminal answered but returned no data — "
                 "check the subscription tier and that the date range is in the past.")
    print("  OK")


if __name__ == "__main__":
    main()
