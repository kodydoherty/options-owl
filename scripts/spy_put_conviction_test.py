"""Is SPY-PUT salvageable by conviction gating? Aggregate SPY put flow is breakeven (PF~1.02),
but crash-day convexity may hide inside it. Bucket SPY put sweeps by cluster size / premium /
ask_frac and check whether high-conviction SPY puts are net winners. Also list the biggest
SPY down days and whether flow fired. Read-only; reuses uw_ticker_discovery internals.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402

CLUSTER_WIN = 30


def fetch_spy_puts():
    hdr = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json"}
    rows, older = [], None
    for _ in range(260):
        p = {"limit": 200, "is_put": "true", "min_premium": D.MIN_PREM}
        if older:
            p["older_than"] = older
        r = None
        for a in range(5):
            try:
                r = requests.get(D.BASE, headers=hdr, params=p, timeout=30); break
            except requests.exceptions.RequestException:
                time.sleep(2 * (a + 1))
        if r is None or r.status_code != 200:
            break
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        older = min(x["created_at"] for x in data)
        if older < D.START:
            break
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    df = df[(df["type"] == "put") & (df["ticker"] == "SPY")].copy()
    df["prem"] = df["total_premium"].astype(float)
    df["ask_frac"] = df["total_ask_side_prem"].astype(float) / df["prem"].clip(lower=1)
    df = df[(df["ask_frac"] >= 0.6) & df["has_sweep"].astype(bool)]
    ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(D.ET)
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["mi"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    df = df[df["mi"].between(0, 375)]
    return df.sort_values("mi")


def main():
    raw = fetch_spy_puts()
    stock, opts = D._stock("SPY"), D._opts("SPY", "PUT")
    cfg = D.apply_v7_wide_trail_exits(
        D.get_ticker_config("SPY", use_per_ticker=True, option_type="put"), is_put=True)
    rows = []
    for d, g in raw.groupby("date"):
        mis = g["mi"].to_numpy()
        seen = set()
        for _, ev in g.iterrows():
            mb = (int(ev["mi"]) // 5) * 5
            if mb in seen or d not in stock or mb not in stock[d]:
                continue
            seen.add(mb)
            csize = int(np.sum(np.abs(mis - ev["mi"]) <= CLUSTER_WIN))
            spot = stock[d][mb]
            oday = opts[(opts["date"] == d) & (opts["mi"] == mb)]
            if oday.empty:
                continue
            dte0 = oday["dte"].min()
            av = oday[oday["dte"] == dte0].assign(dist=(oday["strike"] - spot).abs()).sort_values("dist")
            strike = av.iloc[0]["strike"]
            ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
            ch = ch[ch["mi"] >= mb].sort_values("mi")
            if len(ch) < 5:
                continue
            pp = ch["close"].values.astype(float)
            mp = ch["mi"].values.astype(int)
            up = [stock[d].get(int(m), spot) for m in mp]
            if np.isnan(pp[0]) or pp[0] <= 0:
                continue
            ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=mb)
            rows.append({"date": d, "csize": csize, "prem": ev["prem"], "ask_frac": ev["ask_frac"],
                         "ret": D._sim(pp, mp, up, pp[0], ets, cfg, int(dte0), "put")})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no SPY put trades"); return

    def stat(s):
        s = np.array(s)
        pf = s[s > 0].sum() / max(1e-9, -s[s < 0].sum())
        return f"n={len(s):<4} mean={s.mean():+6.1f}%  win={np.mean(s>0)*100:3.0f}%  runner={np.mean(s>=100)*100:4.1f}%  PF={pf:.2f}  total={s.sum():+.0f}%"

    print(f"=== SPY PUT — ALL: {stat(df['ret'])} ===")
    print("\n  by cluster size:")
    print(f"    single(1):    {stat(df[df.csize == 1]['ret'])}")
    print(f"    clustered>=2: {stat(df[df.csize >= 2]['ret'])}")
    print(f"    heavy>=4:     {stat(df[df.csize >= 4]['ret'])}")
    print("\n  by premium:")
    print(f"    $250-500k: {stat(df[df.prem < 5e5]['ret'])}")
    print(f"    $500k-1M:  {stat(df[(df.prem >= 5e5) & (df.prem < 1e6)]['ret'])}")
    print(f"    $1M+:      {stat(df[df.prem >= 1e6]['ret'])}")
    print("\n  by ask_frac:")
    print(f"    0.60-0.85: {stat(df[df.ask_frac < 0.85]['ret'])}")
    print(f"    0.85+:     {stat(df[df.ask_frac >= 0.85]['ret'])}")
    print("\n  COMBINED high-conviction (>=2 cluster AND $1M+):")
    hc = df[(df.csize >= 2) & (df.prem >= 1e6)]
    print(f"    {stat(hc['ret']) if len(hc) else 'n=0'}")
    print("\n  best SPY put DAYS (sum of that day's flow-put returns):")
    byday = df.groupby("date")["ret"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    for d, r in byday.head(5).iterrows():
        print(f"    {d}: {r['sum']:+.0f}% over {int(r['count'])} entries")
    print("  worst:")
    for d, r in byday.tail(3).iterrows():
        print(f"    {d}: {r['sum']:+.0f}% over {int(r['count'])} entries")


if __name__ == "__main__":
    main()
