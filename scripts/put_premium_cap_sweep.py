"""Flow PUT premium-cap sweep — mirror of the validated CALL cap methodology (flat $750/trade
edge measure). A premium cap is a pure ENTRY filter (block puts whose per-contract option
premium > cap); it doesn't change exits, so we sweep it directly over the flow trade list.

Motivation: 2026-06-26 MU flow PUT @ $9.83 → −$303, the single biggest afternoon loss. The
$9 flow cap is CALLS-ONLY. Question: does an analogous PUT cap add value over the flow window?

Reads journal/v3_eval_results/flow_gold_standard_trades.csv. Read-only.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "journal" / "v3_eval_results" / "flow_gold_standard_trades.csv"
SLEEVE = 750.0  # flat per-trade edge measure (matches the call-cap validation)
CAPS = [None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]


def _load(side):
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["side"] != side:
                continue
            try:
                rows.append({
                    "ticker": r["ticker"],
                    "opt_premium": float(r["opt_premium"]),
                    "pnl": float(r["ret_pct"]) / 100.0 * SLEEVE,
                    "ret_pct": float(r["ret_pct"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _pf(pnls):
    g = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    return (g / losses) if losses > 0 else float("inf")


def _sweep(rows, label):
    print(f"\n=== {label} PUT premium cap sweep ({len(rows)} flow {label.lower()} trades, flat ${SLEEVE:.0f}/trade) ===")
    print(f"  {'cap':<8}{'kept':>6}{'blocked':>9}{'P&L':>11}{'PF':>7}{'WR%':>7}{'blocked$':>11}")
    print("  " + "-" * 60)
    base = None
    best = None
    for cap in CAPS:
        kept = [r for r in rows if cap is None or r["opt_premium"] <= cap]
        blocked = [r for r in rows if cap is not None and r["opt_premium"] > cap]
        pnls = [r["pnl"] for r in kept]
        pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        bpnl = sum(r["pnl"] for r in blocked)
        pf = _pf(pnls)
        if cap is None:
            base = pnl
        delta = "" if cap is None else f"  ({pnl - base:+,.0f})"
        pfs = "inf" if pf == float("inf") else f"{pf:.2f}"
        capl = "off" if cap is None else f"${cap:.0f}"
        print(f"  {capl:<8}{len(kept):>6}{len(blocked):>9}{'$'+format(pnl,'+,.0f'):>11}"
              f"{pfs:>7}{wr:>6.0f}%{'$'+format(bpnl,'+,.0f'):>11}{delta}")
        if cap is not None and (best is None or pnl > best[1]):
            best = (cap, pnl)
    if best:
        print(f"  → best cap: ${best[0]:.0f} (P&L ${best[1]:+,.0f}, "
              f"{best[1]-base:+,.0f} vs off)")

    # expensive-put diagnosis: are puts above each level net losers?
    print(f"\n  Expensive-{label.lower()} diagnosis (net P&L of trades ABOVE each premium):")
    for thr in (6, 7, 8, 9, 10):
        ab = [r for r in rows if r["opt_premium"] > thr]
        if ab:
            print(f"    > ${thr}: {len(ab):>3} trades, net ${sum(r['pnl'] for r in ab):+,.0f} "
                  f"(avg ${sum(r['pnl'] for r in ab)/len(ab):+,.0f})")


def main():
    if not CSV.exists():
        print(f"missing {CSV} — run flow_gold_standard_report.py first")
        return
    puts = _load("put")
    calls = _load("call")
    _sweep(puts, "PUT")
    # reference: show the call side too (we already cap calls at $9)
    print("\n" + "=" * 62)
    _sweep(calls, "CALL")


if __name__ == "__main__":
    main()
