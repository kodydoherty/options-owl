"""Phase 4a: does WHALE PUT FLOW precede the down move? (signal predictive value)

Before a full premium backtest, validate the signal: when big put-flow surges on the
index (SPY/SPX/SPXW), does SPY drop more over the next 30/60 min than from a random
minute? If not, the flow signal is no better than the price classifier and we stop.

Pulls 90-day put-flow alerts from Unusual Whales (API Advanced), maps index alerts to
SPY 1m forward returns (thetadata, overlap ~2026-03..06-09). Read-only.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "journal" / "thetadata_options.db")
ET = ZoneInfo("America/New_York")
BASE = "https://api.unusualwhales.com/api/option-trades/flow-alerts"
KEY = next((line.split("=", 1)[1].strip() for line in (ROOT / ".env").read_text().splitlines()
            if line.startswith("UNUSUAL_WHALES_API_KEY=")), "")
INDEX = {"SPY", "SPX", "SPXW"}
MIN_PREM = 250_000      # whale floor
MAX_PAGES = 60


def fetch_put_flow():
    hdr = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
    rows, older = [], None
    for _ in range(MAX_PAGES):
        params = {"limit": 200, "is_put": "true", "min_premium": MIN_PREM}
        if older:
            params["older_than"] = older
        r = requests.get(BASE, headers=hdr, params=params, timeout=20)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} — stop"); break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < "2026-03-14":
            break
        time.sleep(0.6)   # 120/min limit
    return rows


def main():
    print("Fetching 90d whale put-flow (>= $250k)...", flush=True)
    raw = fetch_put_flow()
    df = pd.DataFrame(raw)
    df = df[df["type"] == "put"].copy()
    df["prem"] = df["total_premium"].astype(float)
    df["voi"] = df["volume_oi_ratio"].astype(float)
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    print(f"Put alerts >= $250k: {len(df):,} | tickers: {df['ticker'].nunique()}")
    print("Top tickers by put-flow $:")
    print(df.groupby("ticker")["prem"].sum().sort_values(ascending=False).head(8).to_string())

    # ASK-SIDE fraction = puts BOUGHT (bearish conviction) vs sold/hedged
    df["ask_prem"] = df["total_ask_side_prem"].astype(float)
    df["ask_frac"] = df["ask_prem"] / df["prem"].clip(lower=1)
    df["sweep"] = df["has_sweep"].astype(bool)

    idx_all = df[df["ticker"].isin(INDEX) & (df["mi"].between(0, 360))].copy()
    # REFINED: aggressively-bought puts only (ask-side dominant AND a sweep)
    idx = idx_all[(idx_all["ask_frac"] >= 0.6) & idx_all["sweep"]].copy()
    idx["mb"] = (idx["mi"] // 5) * 5
    surge = idx.groupby(["date", "mb"]).agg(prem=("prem", "sum"), voi=("voi", "max")).reset_index()
    print(f"\nIndex put alerts: {len(idx_all)} | REFINED (ask-side >=60% + sweep): {len(idx)} "
          f"-> {len(surge)} surge events")

    con = sqlite3.connect(DB)
    spy = pd.read_sql_query("SELECT timestamp, close FROM stock_ohlc WHERE ticker='SPY'", con)
    con.close()
    sts = pd.to_datetime(spy["timestamp"], utc=True).dt.tz_convert(ET)
    spy["date"] = sts.dt.strftime("%Y-%m-%d")
    spy["mi"] = (sts.dt.hour - 9) * 60 + sts.dt.minute - 30
    sidx = {(d, int(m)): c for d, m, c in zip(spy["date"], spy["mi"], spy["close"])}

    def fwd(d, m, n):
        a, b = sidx.get((d, m)), sidx.get((d, m + n))
        return (b - a) / a * 100 if a and b else np.nan

    surge["f30"] = [fwd(d, int(m), 30) for d, m in zip(surge["date"], surge["mb"])]
    surge["f60"] = [fwd(d, int(m), 60) for d, m in zip(surge["date"], surge["mb"])]
    surge = surge.dropna(subset=["f30"])

    # baseline: same days, random minutes
    rng = np.random.RandomState(0)
    base = []
    for d in surge["date"].unique():
        for _ in range(5):
            m = int(rng.randint(0, 330))
            base.append(fwd(d, m, 30))
    base = pd.Series([b for b in base if not np.isnan(b)])

    print("\n=== Does whale PUT flow precede a DROP? (SPY forward return; more negative = PUT wins) ===")
    print(f"  surge events matched: {len(surge)}")
    print(f"  SPY fwd 30m after surge:  mean {surge['f30'].mean():+.3f}%   median {surge['f30'].median():+.3f}%   %down {np.mean(surge['f30']<0)*100:.0f}%")
    print(f"  SPY fwd 60m after surge:  mean {surge['f60'].mean():+.3f}%   median {surge['f60'].median():+.3f}%")
    print(f"  BASELINE fwd 30m (random):mean {base.mean():+.3f}%   median {base.median():+.3f}%   %down {np.mean(base<0)*100:.0f}%")
    edge = surge['f30'].mean() - base.mean()
    print(f"\n  EDGE (surge - baseline, 30m): {edge:+.3f}%  -> {'PUT-flow PRECEDES drops (signal has value)' if edge < -0.02 else 'NO meaningful edge'}")
    big = surge[surge["prem"] >= surge["prem"].quantile(0.8)]
    print(f"  Top-quintile-$ surges fwd 30m: mean {big['f30'].mean():+.3f}%  %down {np.mean(big['f30']<0)*100:.0f}%  (n={len(big)})")
    surge.to_csv(ROOT / "journal" / "v3_eval_results" / "whale_put_flow_surges.csv", index=False)

    # ── SINGLE-NAME flow (directional, not hedging) — the real opportunity ──
    names = ["NVDA", "TSLA", "META", "AAPL", "AMZN", "GOOGL", "MSFT", "AMD", "MSTR", "PLTR", "AVGO", "MU"]
    con = sqlite3.connect(DB)
    nm_rows = []
    print("\n=== SINGLE-NAME ask-side put sweeps -> that NAME's forward move ===")
    for tk in names:
        s = pd.read_sql_query("SELECT timestamp, close FROM stock_ohlc WHERE ticker=?", con, params=(tk,))
        if s.empty:
            continue
        sts = pd.to_datetime(s["timestamp"], utc=True).dt.tz_convert(ET)
        s["date"] = sts.dt.strftime("%Y-%m-%d")
        s["mi"] = (sts.dt.hour - 9) * 60 + sts.dt.minute - 30
        px = {(d, int(m)): c for d, m, c in zip(s["date"], s["mi"], s["close"])}
        fl = df[(df["ticker"] == tk) & (df["ask_frac"] >= 0.6) & df["sweep"] & df["mi"].between(0, 360)]
        for d, m in zip(fl["date"], fl["mi"]):
            a = px.get((d, int(m) // 5 * 5)); b = px.get((d, int(m) // 5 * 5 + 30))
            if a and b:
                nm_rows.append({"tk": tk, "f30": (b - a) / a * 100})
    con.close()
    nm = pd.DataFrame(nm_rows)
    if len(nm):
        print(f"  single-name put-sweep events: {len(nm)}")
        print(f"  NAME fwd 30m: mean {nm['f30'].mean():+.3f}%  median {nm['f30'].median():+.3f}%  %down {np.mean(nm['f30']<0)*100:.0f}%")
        print("  by ticker (mean fwd 30m, n):")
        for tk, g in nm.groupby("tk"):
            if len(g) >= 5:
                print(f"    {tk:<6} {g['f30'].mean():+.3f}%  %down {np.mean(g['f30']<0)*100:3.0f}%  (n={len(g)})")


if __name__ == "__main__":
    main()
