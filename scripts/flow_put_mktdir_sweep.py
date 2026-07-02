"""Sweep a MARKET-DIRECTION filter on flow PUTs (2026-07-01).

Does skipping flow puts bought while the underlying is RALLYING (mkt_chg > X% from the day
open at entry) help or hurt the flow-put edge? Today's SPY-put -54% was a put bought into a
+0.86% SPY rally — flow bypasses put_market_direction. Uses the prod-faithful flow harness
(flow_gold_standard_report.collect) — one API fetch, many thresholds swept in-memory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_gold_standard_report as F  # noqa: E402

BASE = 750.0  # flat $/trade edge measure (matches the flow report's FLAT column)


def _pnl(recs, conv=True):
    return sum(BASE * (r["conv_mult"] if conv else 1.0) * r["ret_pct"] / 100.0 for r in recs)


def _pf(recs, conv=True):
    g = l = 0.0
    for r in recs:
        v = BASE * (r["conv_mult"] if conv else 1.0) * r["ret_pct"] / 100.0
        if v > 0:
            g += v
        else:
            l -= v
    return g / l if l > 0 else float("inf")


def _wr(recs):
    return (sum(1 for r in recs if r["ret_pct"] > 0) / len(recs) * 100) if recs else 0.0


def main():
    print("Fetching flow PUTs (prod-faithful harness)...")
    puts = F.collect(True, F.PUT_UNIV)
    print(f"{len(puts)} flow-put trades\n")

    print("MARKET-DIRECTION FILTER — skip puts where underlying is up > X% from day open at entry")
    print(f"  {'threshold':<12}{'kept':>6}{'skip':>6}{'skip_avg%':>11}"
          f"{'PUTS P&L':>12}{'PF':>7}{'WR':>6}{'vs none':>10}")
    print("  " + "-" * 70)
    base_pnl = None
    for thr in (None, 1.5, 1.0, 0.8, 0.5, 0.3, 0.0):
        if thr is None:
            kept, skipped = puts, []
        else:
            kept = [r for r in puts if r["mkt_chg"] <= thr]
            skipped = [r for r in puts if r["mkt_chg"] > thr]
        p = _pnl(kept)
        if thr is None:
            base_pnl = p
        skip_avg = (sum(r["ret_pct"] for r in skipped) / len(skipped)) if skipped else 0.0
        lbl = "none" if thr is None else f">+{thr}%"
        dv = "" if thr is None else f"${p - base_pnl:+,.0f}"
        print(f"  {lbl:<12}{len(kept):>6}{len(skipped):>6}{skip_avg:>10.0f}%"
              f"{('$' + format(p, '+,.0f')):>12}{_pf(kept):>7.2f}{_wr(kept):>5.0f}%{dv:>10}")

    # SPY-only view (the actual culprit today) — SPY puts bought into a green SPY tape
    spy = [r for r in puts if r["ticker"] == "SPY"]
    if spy:
        print(f"\n  SPY puts only ({len(spy)}):")
        base = _pnl(spy)
        for thr in (None, 0.5, 0.3, 0.0):
            kept = spy if thr is None else [r for r in spy if r["mkt_chg"] <= thr]
            lbl = "none" if thr is None else f">+{thr}%"
            dv = "" if thr is None else f"${_pnl(kept) - base:+,.0f}"
            print(f"    {lbl:<10}{len(kept):>4} kept  ${_pnl(kept):+,.0f}  PF {_pf(kept):.2f}{dv:>10}")


if __name__ == "__main__":
    main()
