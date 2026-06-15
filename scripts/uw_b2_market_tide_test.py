"""B2: does the market-wide whale TIDE improve the flow book? Pull daily UW market-tide
(net_call_premium vs net_put_premium) over the flow window; classify each flow trade as ALIGNED
(call on a net-call-bullish day / put on a net-put-bearish day) vs MISALIGNED; compare PF/return.
If aligned >> misaligned, a tide gate/size-tilt adds value (Track E). Read-only.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
KEY = next((l.split("=", 1)[1].strip() for l in (ROOT / ".env").read_text().splitlines()
            if l.startswith("UNUSUAL_WHALES_API_KEY=")), "")
H = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
TIDE = "https://api.unusualwhales.com/api/market/market-tide"
FLOW = ROOT / "journal" / "v3_eval_results" / "flow_gold_standard_trades.csv"
SLEEVE = 750.0


def daily_bias():
    flow = pd.read_csv(FLOW)
    days = sorted(flow["date"].astype(str).unique())
    bias = {}
    for d in days:
        try:
            r = requests.get(TIDE, headers=H, params={"date": d}, timeout=15)
            ticks = r.json().get("data", []) if r.status_code == 200 else []
        except requests.exceptions.RequestException:
            ticks = []
        if ticks:
            last = ticks[-1]
            nc = float(last.get("net_call_premium") or 0)
            npm = float(last.get("net_put_premium") or 0)
            # bullish tide if net call premium dominates net put premium
            bias[d] = nc - npm
        time.sleep(0.35)
    return flow, bias


def _stat(s):
    s = np.array(s)
    if len(s) == 0:
        return "n=0"
    g = s[s > 0].sum(); l = -s[s < 0].sum()
    pf = g / l if l > 0 else float("inf")
    return f"n={len(s):<4} mean={s.mean():+6.1f}%  win={np.mean(s>0)*100:3.0f}%  PF={pf:.2f}  total=${s.sum()/100*SLEEVE:+,.0f}"


def main():
    flow, bias = daily_bias()
    print(f"market-tide pulled for {len(bias)}/{flow['date'].nunique()} days\n")
    flow["bias"] = flow["date"].astype(str).map(bias)
    flow = flow.dropna(subset=["bias"])
    # aligned: call on bullish-tide day (bias>0) OR put on bearish-tide day (bias<0)
    flow["aligned"] = ((flow["side"] == "call") & (flow["bias"] > 0)) | \
                      ((flow["side"] == "put") & (flow["bias"] < 0))
    print("=== B2: flow trades by market-tide alignment ===")
    print(f"  ALIGNED   : {_stat(flow[flow.aligned]['ret_pct'])}")
    print(f"  MISALIGNED: {_stat(flow[~flow.aligned]['ret_pct'])}")
    print("\n  by side:")
    for side in ("call", "put"):
        s = flow[flow.side == side]
        print(f"    {side} aligned   : {_stat(s[s.aligned]['ret_pct'])}")
        print(f"    {side} misaligned: {_stat(s[~s.aligned]['ret_pct'])}")
    print("\nIf ALIGNED >> MISALIGNED on PF → gate/size flow by the market tide (Track E lever).")


if __name__ == "__main__":
    main()
