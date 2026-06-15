"""Combined end-to-end: V7 ML-sourced gold standard + UW flow book.

HONEST apples-to-apples: both books at a FIXED $750 sleeve (no compounding artifact — letting
flow size off a compounding balance with %-caps inflates to fantasy numbers no liquidity could fill).
ML returns come from v7_core_trades.csv (pnl_pct), flow from flow_gold_standard_trades.csv (ret_pct).
Also prints the COMPOUNDING ML number for reference (the ~$200k you remember) with a clear caveat
that it's not additive with the fixed-sleeve flow.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ML_CSV = ROOT / "journal" / "v3_eval_results" / "v7_core_trades.csv"
FLOW_CSV = ROOT / "journal" / "v3_eval_results" / "flow_gold_standard_trades.csv"
PORTFOLIO_START = 20000.0
SLEEVE = 750.0


def _pf(p):
    g = p[p > 0].sum(); l = -p[p < 0].sum()
    return g / l if l > 0 else float("inf")


def _dd(daily):
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


def main():
    ml = pd.read_csv(ML_CSV); flow = pd.read_csv(FLOW_CSV)
    ml["date"] = ml["day"].astype(str); flow["date"] = flow["date"].astype(str)
    days = sorted(set(ml["date"]) | set(flow["date"]))

    ml["p750"] = ml["pnl_pct"] / 100.0 * SLEEVE
    cm = flow["conv_mult"] / flow["conv_mult"].mean()   # conviction reallocation, same avg capital
    flow["p750_flat"] = flow["ret_pct"] / 100.0 * SLEEVE
    flow["p750_conv"] = flow["ret_pct"] / 100.0 * SLEEVE * cm
    ml_t = float(ml["p750"].sum()); ff = float(flow["p750_flat"].sum()); fc = float(flow["p750_conv"].sum())

    def dly(df, c):
        return df.groupby("date")[c].sum().to_dict()
    comb = {d: dly(ml, "p750").get(d, 0) + dly(flow, "p750_conv").get(d, 0) for d in days}

    print(f"=== COMBINED END-TO-END — FIXED ${SLEEVE:.0f}/trade (honest, no compounding) ===")
    print(f"  window: {days[0]} → {days[-1]} ({len(days)} trading days)\n")
    print(f"  {'book':<32}{'P&L':>12}{'trades':>8}")
    print(f"  {'V7 ML-sourced':<32}{ml_t:>+12,.0f}{len(ml):>8}")
    print(f"  {'UW flow (flat)':<32}{ff:>+12,.0f}{len(flow):>8}")
    print(f"  {'UW flow (conviction-sized)':<32}{fc:>+12,.0f}{len(flow):>8}")
    print(f"  {'-'*52}")
    print(f"  {'COMBINED (ML + flow conviction)':<32}{ml_t+fc:>+12,.0f}{len(ml)+len(flow):>8}")
    cp = pd.concat([ml["p750"], flow["p750_conv"]])
    print(f"\n  combined PF {_pf(cp):.2f} | WR {(cp>0).mean()*100:.0f}% | maxDD ${_dd(comb):+,.0f}")
    print(f"  conviction sizing adds ${fc-ff:+,.0f} to the flow leg (equal capital)")
    print("\n  --- reference (NOT additive — different sizing model) ---")
    print(f"  V7 ML COMPOUNDING off ${PORTFOLIO_START:,.0f}: ${float(ml['pnl'].sum()):+,.0f}  "
          "(the ~$200k figure — aggressive %-compounding, not liquidity-capped)")


if __name__ == "__main__":
    main()
