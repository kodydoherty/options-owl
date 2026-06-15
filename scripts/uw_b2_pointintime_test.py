"""B2 POINT-IN-TIME re-validation (no lookahead). For each flow sweep, classify aligned/misaligned
using the market tide AS OF the entry minute (intraday cumulative net premium up to that tick) — NOT
the end-of-day tide. This is what the live gate can actually see. If aligned still >> misaligned,
B2 is real and wireable. Reuses uw_ticker_discovery fetch+sim. Read-only.
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

TIDE = "https://api.unusualwhales.com/api/market/market-tide"
H = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json"}
SLEEVE = 750.0


def tide_timeline(day: str):
    """Return list of (minute_since_open, bias) for the day, oldest→newest (cumulative net prem)."""
    try:
        r = requests.get(TIDE, headers=H, params={"date": day}, timeout=15)
        ticks = r.json().get("data", []) if r.status_code == 200 else []
    except requests.exceptions.RequestException:
        return []
    out = []
    for t in ticks:
        ts = str(t.get("timestamp", ""))[11:16]
        if len(ts) == 5:
            hh, mm = int(ts[:2]), int(ts[3:5])
            mi = (hh - 9) * 60 + mm - 30
            bias = float(t.get("net_call_premium") or 0) - float(t.get("net_put_premium") or 0)
            out.append((mi, bias))
    return sorted(out)


def bias_at(timeline, entry_min):
    """Tide bias AS OF entry_min (last tick at/before entry) — no lookahead."""
    b = None
    for mi, bias in timeline:
        if mi <= entry_min:
            b = bias
        else:
            break
    return b


def run_side(is_put):
    otype = "put" if is_put else "call"
    right = "PUT" if is_put else "CALL"
    sig = D.fetch_sweeps(is_put)  # has ticker,date,mb (entry 5m bar)
    if sig.empty:
        return []
    tl_cache = {}
    out = []
    for tk in sorted(sig["ticker"].unique()):
        stock, opts = D._stock(tk), D._opts(tk, right)
        cfg = D.apply_v7_wide_trail_exits(
            D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
        for _, ev in sig[sig["ticker"] == tk].iterrows():
            d, em = ev["date"], int(ev["mb"])
            if d not in tl_cache:
                tl_cache[d] = tide_timeline(d); time.sleep(0.35)
            bias = bias_at(tl_cache[d], em)
            if bias is None:
                continue
            if d not in stock or em not in stock[d]:
                continue
            spot = stock[d][em]
            oday = opts[(opts["date"] == d) & (opts["mi"] == em)]
            if oday.empty:
                continue
            dte0 = oday["dte"].min()
            av = oday[oday["dte"] == dte0].assign(dist=(oday["strike"] - spot).abs()).sort_values("dist")
            strike = av.iloc[0]["strike"]
            ch = opts[(opts["date"] == d) & (opts["strike"] == strike) & (opts["dte"] == dte0)]
            ch = ch[ch["mi"] >= em].sort_values("mi")
            if len(ch) < 5:
                continue
            pp = ch["close"].values.astype(float); mp = ch["mi"].values.astype(int)
            up = [stock[d].get(int(m), spot) for m in mp]
            if np.isnan(pp[0]) or pp[0] <= 0:
                continue
            ets = datetime(*map(int, d.split("-")), 9, 30, tzinfo=D.ET) + timedelta(minutes=em)
            ret = D._sim(pp, mp, up, pp[0], ets, cfg, int(dte0), otype)
            aligned = (not is_put and bias > 0) or (is_put and bias < 0)
            out.append({"side": otype, "aligned": aligned, "ret": ret})
    return out


def _stat(s):
    s = np.array(s)
    if len(s) == 0:
        return "n=0"
    g = s[s > 0].sum(); l = -s[s < 0].sum()
    pf = g / l if l > 0 else float("inf")
    return f"n={len(s):<4} mean={s.mean():+6.1f}%  win={np.mean(s>0)*100:3.0f}%  PF={pf:.2f}  total=${s.sum()/100*SLEEVE:+,.0f}"


def main():
    rows = run_side(True) + run_side(False)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no data"); return
    print("=== B2 POINT-IN-TIME (no lookahead) — tide as of entry minute ===")
    print(f"  ALIGNED   : {_stat(df[df.aligned]['ret'])}")
    print(f"  MISALIGNED: {_stat(df[~df.aligned]['ret'])}")
    for side in ("call", "put"):
        s = df[df.side == side]
        print(f"    {side} aligned   : {_stat(s[s.aligned]['ret'])}")
        print(f"    {side} misaligned: {_stat(s[~s.aligned]['ret'])}")
    print("\nIf ALIGNED still >> MISALIGNED → B2 is real (no lookahead) → wire it. Else it was the bias.")


if __name__ == "__main__":
    main()
