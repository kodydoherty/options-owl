"""Test: does the GEX gamma regime distinguish SUSTAINED crashes from REBOUND traps?

Hypothesis (from GEX theory): negative net gamma = trending/expansive (dealers amplify
drops → sustained crash); positive net gamma = pinning/mean-reverting (drop snaps back →
rebound trap, the Trump V). If true, net-gamma sign is the down-day filter we couldn't get
from price (phases 1-2). Uses UW daily greek-exposure + down_days.csv labels. Read-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "journal" / "v3_eval_results"
KEY = next((ln.split("=", 1)[1].strip() for ln in (ROOT / ".env").read_text().splitlines()
            if ln.startswith("UNUSUAL_WHALES_API_KEY=")), "")


def fetch_gex(ticker="SPY"):
    r = requests.get(f"https://api.unusualwhales.com/api/stock/{ticker}/greek-exposure",
                     headers={"Authorization": f"Bearer {KEY}"}, params={"limit": 500}, timeout=20)
    d = pd.DataFrame(r.json()["data"])
    for c in ("call_gamma", "put_gamma"):
        d[c] = d[c].astype(float)
    d["net_gamma"] = (d["call_gamma"] + d["put_gamma"]) / 1e9  # $bn/1% notional
    return d[["date", "net_gamma"]]


def main():
    gex = fetch_gex("SPY")
    dd = pd.read_csv(RES / "down_days.csv")[["date", "cls", "max_dd_pct", "oc_ret_pct"]]
    m = dd.merge(gex, on="date", how="inner")
    print(f"Days with both GEX + classification: {len(m)}\n")

    print("=== net gamma ($bn) by day class ===")
    for c in ["SUSTAINED", "REBOUND", "PARTIAL", "up_or_flat"]:
        g = m[m["cls"] == c]
        if len(g):
            print(f"  {c:<11} n={len(g):<4} mean net_gamma={g['net_gamma'].mean():+.2f}  "
                  f"median={g['net_gamma'].median():+.2f}  %negative={np.mean(g['net_gamma']<0)*100:.0f}%")

    # The key test: among DOWN days (price dropped), does negative gamma => SUSTAINED?
    down = m[m["cls"].isin(["SUSTAINED", "REBOUND"])].copy()
    if len(down):
        print("\n=== among DOWN days: split by gamma sign — does negative gamma = sustained? ===")
        for label, sub in [("negative gamma", down[down["net_gamma"] < 0]),
                           ("positive gamma", down[down["net_gamma"] >= 0])]:
            if len(sub):
                sus = np.mean(sub["cls"] == "SUSTAINED") * 100
                print(f"  {label:<16} n={len(sub):<4} -> {sus:.0f}% were SUSTAINED crashes "
                      f"(avg o->c {sub['oc_ret_pct'].mean():+.2f}%)")
        # base rate
        print(f"  [base rate: {np.mean(down['cls']=='SUSTAINED')*100:.0f}% of down days are sustained]")
    m.to_csv(RES / "gex_regime_test.csv", index=False)
    print(f"\nSaved -> {RES / 'gex_regime_test.csv'}")


if __name__ == "__main__":
    main()
