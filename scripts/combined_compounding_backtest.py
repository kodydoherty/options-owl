"""Combined COMPOUNDING portfolio — V7 ML + UW flow on ONE balance, sized realistically.

The fix that makes compounding+flow believable: a per-trade LIQUIDITY CAP. Both books are sized
off the running (compounding) balance — per_slot = bal*risk/max_concurrent, scaled by each trade's
conviction multiplier (ML size_mult, flow conviction_mult), then capped by BOTH the position-% cap
AND an absolute liquidity cap (how much a single 0DTE option can actually absorb). Without the
liquidity cap, %-sizing on a ballooning balance inflates to numbers no fill could support.

Shows the curve across liquidity caps so you can see compounding upside vs realism.
Inputs: v7_core_trades.csv (ML: pnl_pct, size_mult) + flow_gold_standard_trades.csv (ret_pct, conv_mult).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ML_CSV = ROOT / "journal" / "v3_eval_results" / "v7_core_trades.csv"
FLOW_CSV = ROOT / "journal" / "v3_eval_results" / "flow_gold_standard_trades.csv"
PORTFOLIO_START = 20000.0
RISK_PCT = 0.75
MAX_CONCURRENT = 8
POS_CAP_PCT = 0.15
PUT_BUDGET = 0.50


def load():
    ml = pd.read_csv(ML_CSV); flow = pd.read_csv(FLOW_CSV)
    ev = []
    for _, t in ml.iterrows():
        ev.append({"date": str(t["day"]), "src": "ml", "ret": float(t["pnl_pct"]),
                   "mult": float(t.get("size_mult", 1.0) or 1.0),
                   "is_put": str(t["direction"]).lower() == "put"})
    for _, t in flow.iterrows():
        ev.append({"date": str(t["date"]), "src": "flow", "ret": float(t["ret_pct"]),
                   "mult": float(t["conv_mult"]), "is_put": t["side"] == "put"})
    df = pd.DataFrame(ev).sort_values("date")
    return df


def run(df, liq_cap):
    bal = PORTFOLIO_START
    ml_pnl = flow_pnl = 0.0
    daily = {}
    for d, g in df.groupby("date", sort=True):
        per_slot = bal * RISK_PCT / MAX_CONCURRENT
        pos_cap = bal * POS_CAP_PCT
        day = 0.0
        for _, t in g.iterrows():
            dirmult = PUT_BUDGET if t["is_put"] else 1.0
            size = per_slot * t["mult"] * dirmult
            size = min(size, pos_cap)
            if liq_cap:
                size = min(size, liq_cap)
            pnl = size * t["ret"] / 100.0
            day += pnl
            if t["src"] == "ml":
                ml_pnl += pnl
            else:
                flow_pnl += pnl
        daily[d] = day
        bal += day
    # max drawdown on the equity curve
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return bal, ml_pnl, flow_pnl, dd


def main():
    df = load()
    days = df["date"].nunique()
    print(f"=== COMBINED COMPOUNDING — V7 ML + UW flow, one balance (${PORTFOLIO_START:,.0f} start, "
          f"{days} days, {len(df)} trades) ===")
    print(f"  sizing: bal*{RISK_PCT:.0%}/{MAX_CONCURRENT} * conviction, capped by {POS_CAP_PCT:.0%} pos AND liq cap\n")
    print(f"  {'liquidity cap':<16}{'end balance':>14}{'total P&L':>13}{'ML':>12}{'flow':>13}{'maxDD':>12}")
    for cap in (None, 200_000, 100_000, 50_000, 25_000, 10_000):
        bal, ml, fl, dd = run(df, cap)
        label = "none (∞)" if cap is None else f"${cap/1e3:.0f}k"
        print(f"  {label:<16}{bal:>+14,.0f}{bal-PORTFOLIO_START:>+13,.0f}{ml:>+12,.0f}{fl:>+13,.0f}{dd:>+12,.0f}")
    print("\n  Read: 'none' = the fantasy explosion (no fill could absorb it). Lower caps = realistic.")
    print("  A $25k-50k/trade cap on liquid 0DTE keeps most of the compounding while staying fillable.")


if __name__ == "__main__":
    main()
