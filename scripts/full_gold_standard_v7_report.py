"""FULL V7 gold-standard report — 60 days, EVERYTHING: V7 ML-sourced (0.62 gate, V7 wide-trail
exits) + UW flow book (calls+puts, new tickers, conviction sizing) on the combined picture.

Two honest framings:
  (A) FIXED $750/trade  — the EDGE (no compounding), apples-to-apples, additive across books.
  (B) COMPOUNDING       — one balance, conviction-sized, per-trade liquidity-capped (the upside shape).

Reads v7_core_trades.csv (ML) + flow_gold_standard_trades.csv (flow). Writes markdown + CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "journal" / "v3_eval_results"
ML_CSV = RES / "v7_core_trades.csv"
FLOW_CSV = RES / "flow_gold_standard_trades.csv"
OUT_MD = RES / "full_gold_standard_v7_report.md"
SLEEVE = 750.0
START_BAL = 20000.0
RISK_PCT, MAX_CONC, POS_CAP, PUT_BUDGET = 0.75, 8, 0.15, 0.50


def _pf(p):
    g = p[p > 0].sum(); l = -p[p < 0].sum()
    return g / l if l > 0 else float("inf")


def _dd(daily):
    eq = peak = dd = 0.0
    for d in sorted(daily):
        eq += daily[d]; peak = max(peak, eq); dd = min(dd, eq - peak)
    return dd


def compounding(ml, flow, liq_cap):
    ev = ([{"date": str(r.day), "src": "ML", "ret": float(r.pnl_pct),
            "mult": float(getattr(r, "size_mult", 1.0) or 1.0),
            "is_put": str(r.direction).lower() == "put"} for r in ml.itertuples()]
          + [{"date": str(r.date), "src": "flow", "ret": float(r.ret_pct),
              "mult": float(r.conv_mult), "is_put": r.side == "put"} for r in flow.itertuples()])
    df = pd.DataFrame(ev)
    bal = START_BAL; ml_p = fl_p = 0.0; daily = {}
    for d, g in df.groupby("date", sort=True):
        per_slot = bal * RISK_PCT / MAX_CONC; cap = bal * POS_CAP; day = 0.0
        for t in g.itertuples():
            size = min(per_slot * t.mult * (PUT_BUDGET if t.is_put else 1.0), cap, liq_cap or 1e18)
            pnl = size * t.ret / 100.0; day += pnl
            if t.src == "ML": ml_p += pnl
            else: fl_p += pnl
        daily[d] = day; bal += day
    return bal, ml_p, fl_p, _dd(daily)


def main():
    ml = pd.read_csv(ML_CSV); flow = pd.read_csv(FLOW_CSV)
    ml["date"] = ml["day"].astype(str); flow["date"] = flow["date"].astype(str)
    days = sorted(set(ml["date"]) | set(flow["date"]))
    ml["p750"] = ml["pnl_pct"] / 100 * SLEEVE
    cm = flow["conv_mult"] / flow["conv_mult"].mean()
    flow["p750"] = flow["ret_pct"] / 100 * SLEEVE * cm
    L = []
    L.append(f"# FULL V7 Gold Standard + UW Flow — {days[0]} → {days[-1]} ({len(days)} trading days)\n")
    L.append("Everything: V7 0.62 gate + V7 wide-trail exits (ML-sourced) **and** UW flow book "
             "(calls+puts, new tickers MU/ORCL/INTC/ARM/GOOG/LRCX, conviction sizing, $50k liquidity cap).\n")

    L.append("## A) Fixed $750/trade — the EDGE (apples-to-apples, no compounding)")
    L.append("| book | trades | P&L | PF | WR |")
    L.append("|---|---|---|---|---|")
    for name, p in [("V7 ML-sourced", ml["p750"]), ("UW flow (conviction)", flow["p750"])]:
        L.append(f"| {name} | {len(p)} | ${p.sum():+,.0f} | {_pf(p):.2f} | {(p>0).mean()*100:.0f}% |")
    comb = pd.concat([ml["p750"], flow["p750"]])
    cdaily = {d: ml[ml.date == d]["p750"].sum() + flow[flow.date == d]["p750"].sum() for d in days}
    L.append(f"| **COMBINED** | {len(comb)} | **${comb.sum():+,.0f}** | **{_pf(comb):.2f}** | {(comb>0).mean()*100:.0f}% |")
    L.append(f"\nCombined maxDD: ${_dd(cdaily):+,.0f}\n")

    L.append("## B) Compounding off $20k — the UPSIDE (one balance, conviction-sized, liquidity-capped)")
    L.append("| per-trade liq cap | end balance | total P&L | ML | flow | maxDD |")
    L.append("|---|---|---|---|---|---|")
    for cap in (None, 100_000, 50_000, 25_000):
        bal, mlp, flp, dd = compounding(ml, flow, cap)
        lbl = "none (∞, unfillable)" if cap is None else f"${cap/1e3:.0f}k"
        L.append(f"| {lbl} | ${bal:,.0f} | ${bal-START_BAL:+,.0f} | ${mlp:+,.0f} | ${flp:+,.0f} | ${dd:+,.0f} |")
    L.append("\n_ML-only compounding (no flow) = $234,746 — the historical headline; flow ~2× ML at every realistic cap._\n")

    L.append("## Per-source")
    L.append(f"- **V7 ML-sourced:** {len(ml)} trades, fixed-sleeve ${ml['p750'].sum():+,.0f} (PF {_pf(ml['p750']):.2f}), "
             f"compounding $234,746.")
    L.append(f"- **UW flow:** {len(flow)} trades, fixed-sleeve ${flow['p750'].sum():+,.0f} (PF {_pf(flow['p750']):.2f}); "
             f"new tickers (MU/ORCL/INTC/ARM/GOOG/LRCX) ${flow[flow.ticker.isin(['MU','ORCL','INTC','ARM','GOOG','LRCX'])]['p750'].sum():+,.0f}.")

    L.append("\n## Per-ticker (fixed-sleeve, both books)")
    L.append("| ticker | source | trades | P&L | PF |")
    L.append("|---|---|---|---|---|")
    rows = ([("ML", t, g["p750"]) for t, g in ml.groupby("ticker")]
            + [("flow", f"{t}", g["p750"]) for t, g in flow.groupby("ticker")])
    for src, tk, p in sorted(rows, key=lambda x: -x[2].sum()):
        L.append(f"| {tk} | {src} | {len(p)} | ${p.sum():+,.0f} | {_pf(p):.2f} |")

    L.append("\n## Exit reasons (combined, fixed-sleeve)")
    ml_ex = ml.assign(reason=ml["reason"], p=ml["p750"])[["reason", "p"]]
    fl_ex = flow.assign(reason=flow["exit_reason"], p=flow["p750"])[["reason", "p"]]
    allex = pd.concat([ml_ex, fl_ex])
    L.append("| reason | trades | P&L |")
    L.append("|---|---|---|")
    for reason, g in sorted(allex.groupby("reason"), key=lambda x: -x[1]["p"].sum()):
        L.append(f"| {reason} | {len(g)} | ${g['p'].sum():+,.0f} |")

    L.append("\n## Per-day combined P&L (fixed-sleeve)")
    L.append("| date | trades | day P&L | cum |")
    L.append("|---|---|---|---|")
    cum = 0.0
    for d in days:
        n = len(ml[ml.date == d]) + len(flow[flow.date == d]); dp = cdaily[d]; cum += dp
        L.append(f"| {d} | {n} | ${dp:+,.0f} | ${cum:+,.0f} |")

    OUT_MD.write_text("\n".join(L))
    print("\n".join(L[:34]))
    print(f"\nFull report -> {OUT_MD}")


if __name__ == "__main__":
    main()
