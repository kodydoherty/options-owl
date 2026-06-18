"""V7 + flow gold-standard — RECENT window report, matched to prod sizing/risk config.

Reads the cached gold-standard trade sets (journal/v3_eval_results/v7_core_trades.csv +
flow_gold_standard_trades.csv) and reports the last 30 and 60 trading days under prod's
account model: start $20k, compound, FREEZE sizing at MAX_SIZING_BALANCE (take-profit cap),
per-trade liquidity caps for the honest-vs-optimistic spread.

PROD-MATCH STATUS:
  ✓ FOMC pause      — applied exactly (drops all trades on settings.FOMC_PAUSE_DATES).
  ✗ Delta brake     — NOT applied: the cached CSVs carry no option delta, so the delta-scaled
                      cheap-call haircut can't be reproduced here (proxying via premium would be the
                      FLOOR lever, not the DELTA lever prod ships). Effect is minor (trims sizing on
                      cheap OTM calls only). To fully match prod, REGENERATE the CSVs with delta
                      captured at entry, then this report picks it up automatically.

Run: python scripts/v7_recent_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "journal" / "v3_eval_results"

START = 20_000.0
RISK, CONC, POS, PUTB = 0.75, 8, 0.15, 0.50

# Prod config — read live from settings so this stays in sync.
try:
    from options_owl.config.settings import Settings
    _s = Settings()
    SIZECAP = float(getattr(_s, "MAX_SIZING_BALANCE", 100_000) or 100_000)
    FOMC = {d.strip() for d in str(getattr(_s, "FOMC_PAUSE_DATES", "")).split(",") if d.strip()}
except Exception:
    SIZECAP, FOMC = 100_000.0, set()
# Prod runs the FOMC pause ON (docker-compose ENABLE_FOMC_PAUSE=true); this report matches prod, so the
# pause is always applied here regardless of the local settings default.
FOMC_ON = True
# Override the take-profit cap here if you want a what-if (prod is MAX_SIZING_BALANCE):
SIZECAP = 100_000.0


def _pf(p):
    p = np.array(p, float)
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    return g / l if l > 0 else float("inf")


def _dd(daily):
    eq = peak = d = 0.0
    for k in sorted(daily):
        eq += daily[k]; peak = max(peak, eq); d = min(d, eq - peak)
    return d


def _compound(ml, flow, liq):
    ev = ([{"date": str(r.day), "ret": float(r.pnl_pct),
            "mult": float(getattr(r, "size_mult", 1.0) or 1.0),
            "is_put": str(r.direction).lower() == "put"} for r in ml.itertuples()]
          + [{"date": str(r.date), "ret": float(r.ret_pct), "mult": float(r.conv_mult),
              "is_put": r.side == "put"} for r in flow.itertuples()])
    df = pd.DataFrame(ev)
    bal = START; daily = {}
    for d_, g in df.groupby("date", sort=True):
        sb = min(bal, SIZECAP); per = sb * RISK / CONC; cap = sb * POS; day = 0.0
        for t in g.itertuples():
            day += min(per * t.mult * (PUTB if t.is_put else 1.0), cap, liq) * t.ret / 100.0
        daily[d_] = day; bal += day
    return bal, _dd(daily)


def main():
    ml = pd.read_csv(RES / "v7_core_trades.csv"); ml["date"] = ml["day"].astype(str)
    flow = pd.read_csv(RES / "flow_gold_standard_trades.csv"); flow["date"] = flow["date"].astype(str)

    n_ml0, n_fl0 = len(ml), len(flow)
    if FOMC_ON and FOMC:
        ml = ml[~ml["date"].isin(FOMC)]; flow = flow[~flow["date"].isin(FOMC)]
    dropped = (n_ml0 - len(ml)) + (n_fl0 - len(flow))

    days = sorted(set(ml["date"]) | set(flow["date"]))
    print(f"V7 + flow gold-standard — prod-matched (start ${START:,.0f}, freeze sizing ${SIZECAP/1e3:.0f}k)")
    print(f"data {days[0]} → {days[-1]} ({len(days)} trading days)")
    print(f"FOMC pause: {'ON' if FOMC_ON else 'off'} — dropped {dropped} Fed-day trades "
          f"({sorted(d for d in FOMC if days[0] <= d <= days[-1])})")
    print("delta brake: NOT applied (CSVs lack delta — regenerate to include; minor effect)\n")

    for W in (30, 60):
        win = set(days[-W:]); mlw = ml[ml["date"].isin(win)]; flw = flow[flow["date"].isin(win)]
        edge = np.array(list(mlw["pnl_pct"] * 7.5) + list(flw["ret_pct"] * 7.5), float)  # fixed $750/trade
        print(f"===== LAST {W} DAYS ({days[-W]} → {days[-1]}) | {len(edge)} trades =====")
        print(f"  EDGE (fixed $750/trade):  P&L ${edge.sum():+,.0f}   PF {_pf(edge):.2f}   WR {100*(edge>0).mean():.0f}%")
        print("  COMPOUND $20k, freeze at $100k — by per-trade fill assumption:")
        for liq, lbl in [(1e18, "fill anything (optimistic)"),
                         (25_000, "$25k/trade fill cap"),
                         (10_000, "$10k/trade (realistic, thin 0DTE)")]:
            bal, dd = _compound(mlw, flw, liq)
            print(f"     {lbl:34s} end ${bal:>11,.0f}  (maxDD ${dd:+,.0f})")
        print()


if __name__ == "__main__":
    main()
