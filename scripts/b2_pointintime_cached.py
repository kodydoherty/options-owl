"""B2 point-in-time tide gate (NO lookahead, fast). Uses the cached UW sweeps + caches market-tide.
For each flow trade, classify aligned/misaligned by the tide AS OF the entry minute (cumulative net
call-put premium up to that tick). If aligned still >> misaligned, the gate is real + wireable.
"""
from __future__ import annotations

import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from options_owl.bot_runner import select_flow_strike  # noqa: E402

H = {"Authorization": f"Bearer {D.KEY}", "Accept": "application/json"}
TIDE = "https://api.unusualwhales.com/api/market/market-tide"
PUT_UNIV, CALL_UNIV = D.CUR_PUT | {"SPY"}, D.CUR_CALL
OTM_CALL, OTM_PUT, OTM_TARGET = {"AMD", "INTC", "META", "SPY"}, {"TSLA"}, 2.0
CACHE_TIDE = Path("/tmp/b2_tide.pkl")


def tide_timelines(days):
    if CACHE_TIDE.exists():
        return pickle.loads(CACHE_TIDE.read_bytes())
    tl = {}
    for d in days:
        try:
            r = requests.get(TIDE, headers=H, params={"date": d}, timeout=15)
            ticks = r.json().get("data", []) if r.status_code == 200 else []
        except requests.exceptions.RequestException:
            ticks = []
        out = []
        for t in ticks:
            ts = str(t.get("timestamp", ""))[11:16]
            try:
                mi = (int(ts[:2]) - 9) * 60 + int(ts[3:5]) - 30
                out.append((mi, float(t.get("net_call_premium") or 0) - float(t.get("net_put_premium") or 0)))
            except (ValueError, IndexError):
                pass
        tl[d] = sorted(out)
        time.sleep(0.35)
    CACHE_TIDE.write_bytes(pickle.dumps(tl))
    return tl


def bias_at(timeline, entry_min):
    b = None
    for mi, bias in timeline:
        if mi <= entry_min:
            b = bias
        else:
            break
    return b


def flow_trades():
    sweeps = pickle.loads(Path("/tmp/flow_otm_sweeps.pkl").read_bytes())
    out = []
    for is_put, wl in ((True, PUT_UNIV), (False, CALL_UNIV)):
        sig = sweeps[is_put]
        sig = sig[sig["ticker"].isin(wl)]
        otype, right = ("put", "PUT") if is_put else ("call", "CALL")
        for tk in sorted(sig["ticker"].unique()):
            stock, opts = D._stock(tk), D._opts(tk, right)
            cfg = D.apply_v7_wide_trail_exits(
                D.get_ticker_config(tk, use_per_ticker=True, option_type=otype), is_put=is_put)
            for _, ev in sig[sig["ticker"] == tk].iterrows():
                d, mb = ev["date"], (int(ev["mi"]) // 5) * 5
                if d not in stock or mb not in stock[d]:
                    continue
                spot = stock[d][mb]
                oday = opts[(opts["date"] == d) & (opts["mi"] == mb)]
                if oday.empty:
                    continue
                dte0 = oday["dte"].min()
                same = oday[oday["dte"] == dte0]
                pchain = [{"strike": float(r.strike), "mid": float(r.close)} for r in same.itertuples()]
                strike, _ = select_flow_strike(pchain, spot, is_put, tk in (OTM_PUT if is_put else OTM_CALL), OTM_TARGET)
                if not strike:
                    continue
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
                ret = D._sim(pp, list(mp), list(up), pp[0], ets, cfg, int(dte0), otype)
                out.append({"date": d, "side": otype, "mb": mb, "ret": ret})
    return pd.DataFrame(out)


def _stat(s):
    s = np.array(s)
    if len(s) == 0:
        return "n=0"
    g = s[s > 0].sum(); l = -s[s < 0].sum()
    pf = g / l if l > 0 else float("inf")
    return f"n={len(s):<4} mean={s.mean():+6.1f}%  win={np.mean(s>0)*100:3.0f}%  PF={pf:.2f}  total={s.sum():+.0f}%"


def main():
    print("building flow trades (cached) + tide timelines...", flush=True)
    df = flow_trades()
    tl = tide_timelines(sorted(df["date"].unique()))
    df["bias"] = df.apply(lambda r: bias_at(tl.get(r["date"], []), r["mb"]), axis=1)
    df = df.dropna(subset=["bias"])
    df["aligned"] = ((df.side == "call") & (df.bias > 0)) | ((df.side == "put") & (df.bias < 0))
    print(f"\n=== B2 POINT-IN-TIME (tide as of entry minute, NO lookahead) — {len(df)} trades ===")
    print(f"  ALIGNED   : {_stat(df[df.aligned]['ret'])}")
    print(f"  MISALIGNED: {_stat(df[~df.aligned]['ret'])}")
    print("  by side:")
    for side in ("call", "put"):
        s = df[df.side == side]
        print(f"    {side} aligned   : {_stat(s[s.aligned]['ret'])}")
        print(f"    {side} misaligned: {_stat(s[~s.aligned]['ret'])}")
    print("\nIf ALIGNED still >> MISALIGNED (esp. gating out misaligned puts) → the tide gate is real + wireable.")


if __name__ == "__main__":
    main()
