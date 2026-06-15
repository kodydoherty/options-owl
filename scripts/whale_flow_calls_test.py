"""Symmetric test: do whale ask-side CALL sweeps precede UP moves? (mirror of the PUT test)

If yes, the UW flow logic improves CALLs too (bullish conviction → buy calls), not just PUTs.
Same method: ask-side (bought) + sweep, single-name, forward 30m move. CALL wins when the
name goes UP (positive forward return). Read-only; 90d UW flow + thetadata SPY/name 1m.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "journal" / "thetadata_options.db")
ET = ZoneInfo("America/New_York")
BASE = "https://api.unusualwhales.com/api/option-trades/flow-alerts"
KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
            if ln.startswith("UNUSUAL_WHALES_API_KEY=")), "")
NAMES = ["NVDA", "TSLA", "META", "AAPL", "AMZN", "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "MU"]
MIN_PREM = 250_000


def fetch_call_sweeps():
    hdr = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
    rows, older = [], None
    for _ in range(220):
        p = {"limit": 200, "is_put": "false", "min_premium": MIN_PREM}
        if older:
            p["older_than"] = older
        r = requests.get(BASE, headers=hdr, params=p, timeout=20)
        if r.status_code != 200:
            break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < "2026-03-14":
            break
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    df = df[(df["type"] == "call") & df["ticker"].isin(NAMES)].copy()
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["total_premium"].astype(float).clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = ((ts.dt.hour - 9) * 60 + ts.dt.minute - 30)
    return df[df["mi"].between(0, 360)]


def main():
    print("Fetching whale CALL sweeps (ask-side, >= $250k)...", flush=True)
    df = fetch_call_sweeps()
    print(f"Ask-side call sweeps on names: {len(df)}\n")
    con = sqlite3.connect(DB)
    rows = []
    print("=== whale ask-side CALL sweep -> name's forward 30m move (UP = call wins) ===")
    for tk in NAMES:
        s = pd.read_sql_query("SELECT timestamp, close FROM stock_ohlc WHERE ticker=?", con, params=(tk,))
        if s.empty:
            continue
        sts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
        s["date"] = sts.dt.strftime("%Y-%m-%d")
        s["mi"] = (sts.dt.hour - 9) * 60 + sts.dt.minute - 30
        px = {(d, int(m)): c for d, m, c in zip(s["date"], s["mi"], s["close"])}
        for d, m in zip(df[df["ticker"] == tk]["date"], df[df["ticker"] == tk]["mi"]):
            a = px.get((d, int(m) // 5 * 5)); b = px.get((d, int(m) // 5 * 5 + 30))
            if a and b:
                rows.append({"tk": tk, "f30": (b - a) / a * 100})
    con.close()
    nm = pd.DataFrame(rows)
    if not len(nm):
        print("no matched events"); return
    print(f"  total events: {len(nm)} | mean fwd30 {nm['f30'].mean():+.3f}%  %UP {np.mean(nm['f30']>0)*100:.0f}%")
    print("  by ticker (mean fwd 30m, %UP, n):")
    for tk, g in nm.groupby("tk"):
        if len(g) >= 5:
            mark = "  <- CALL signal" if g["f30"].mean() > 0.05 and np.mean(g["f30"] > 0) > 0.55 else ""
            print(f"    {tk:<6} {g['f30'].mean():+6.3f}%  %UP {np.mean(g['f30']>0)*100:3.0f}%  (n={len(g)}){mark}")


if __name__ == "__main__":
    main()
