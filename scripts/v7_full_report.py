"""Unified end-to-end V7 report: V7 core (ML-scan, optimized 0.62 gates) + UW flow sleeve,
per-day P&L with the two sleeves SEPARATE, big down-days flagged, and a capture check that
verifies we took advantage of every large down day in the window.

Inputs (produced by the two backtests):
  journal/v3_eval_results/v7_core_trades.csv   (day,ticker,direction,pnl,reason,...)
  journal/v3_eval_results/uw_flow_trades.csv   (date,tk,ret,pnl)   [$750/trade sleeve]
SPY daily open->close from thetadata. Read-only.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = str(ROOT / "journal" / "thetadata_options.db")
RES = ROOT / "journal" / "v3_eval_results"
ET = ZoneInfo("America/New_York")
BIG_DOWN = -1.0   # SPY open->close <= -1% = a "large down day"


def spy_daily():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT timestamp, open, close FROM stock_ohlc WHERE ticker='SPY' ORDER BY timestamp", con)
    con.close()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d"); df["hhmm"] = ts.dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= "09:30") & (df["hhmm"] <= "16:00")]
    out = {}
    for d, g in df.groupby("date"):
        g = g.sort_values("hhmm"); op = float(g.iloc[0]["open"]) or float(g.iloc[0]["close"])
        if op > 0:
            out[d] = (float(g.iloc[-1]["close"]) - op) / op * 100
    return out


def main():
    core = pd.read_csv(RES / "v7_core_trades.csv") if (RES / "v7_core_trades.csv").exists() else pd.DataFrame()
    flow = pd.read_csv(RES / "uw_flow_trades.csv") if (RES / "uw_flow_trades.csv").exists() else pd.DataFrame()
    spy = spy_daily()

    core_day = defaultdict(float); core_n = defaultdict(int)
    for _, t in core.iterrows():
        core_day[t["day"]] += t["pnl"]; core_n[t["day"]] += 1
    flow_day = defaultdict(float); flow_n = defaultdict(int)
    for _, t in flow.iterrows():
        flow_day[t["date"]] += t["pnl"]; flow_n[t["date"]] += 1

    days = sorted(set(core_day) | set(flow_day) | {d for d in spy if d >= "2026-03-16"})
    print(f"{'date':<12}{'SPY%':>7}{'':3}{'core$':>9}{'core#':>6}{'flow$':>9}{'flow#':>6}{'combined$':>11}")
    ct = ft = 0.0
    for d in days:
        s = spy.get(d, 0.0); flag = " *DOWN*" if s <= BIG_DOWN else ""
        c = core_day.get(d, 0.0); f = flow_day.get(d, 0.0); ct += c; ft += f
        if core_n.get(d) or flow_n.get(d) or flag:
            print(f"{d:<12}{s:>6.2f}%{flag:>9}{c:>9.0f}{core_n.get(d,0):>6}{f:>9.0f}{flow_n.get(d,0):>6}{c+f:>11.0f}")
    print(f"\n{'TOTAL':<12}{'':>9}{ct:>12.0f}{'':>6}{ft:>9.0f}{'':>6}{ct+ft:>11.0f}")

    # === DOWN-DAY CAPTURE CHECK ===
    print("\n=== LARGE DOWN DAYS (SPY o->c <= -1%) — did we capitalize? ===")
    big = sorted([d for d in spy if spy[d] <= BIG_DOWN and d >= "2026-03-16"])
    print(f"{'date':<12}{'SPY%':>7}{'core$':>9}{'flow$':>9}{'combined$':>11}  verdict")
    captured = miss = 0
    for d in big:
        c = core_day.get(d, 0.0); f = flow_day.get(d, 0.0)
        ok = (c + f) > 0
        captured += ok; miss += (not ok)
        print(f"{d:<12}{spy[d]:>6.2f}%{c:>9.0f}{f:>9.0f}{c+f:>11.0f}  {'CAPTURED' if ok else 'MISSED'}")
    print(f"\n  {captured}/{len(big)} large down days profitable  ({miss} missed)")

    # === exit reasons (V7 core) ===
    if not core.empty:
        print("\n=== V7 core exit reasons ===")
        er = core.groupby("reason")["pnl"].agg(["count", "sum"]).sort_values("count", ascending=False)
        for r, row in er.iterrows():
            print(f"  {r:<22} n={int(row['count']):<4} ${row['sum']:+,.0f}")


if __name__ == "__main__":
    main()
