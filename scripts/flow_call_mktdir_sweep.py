"""Sweep a MARKET-DIRECTION filter on flow CALLs (2026-07-08).

Mirror of the deployed flow-PUT market-direction filter (flow_put_mktdir_sweep.py, live
2026-07-01). Question: does skipping flow CALLs bought while the tape is FALLING help or
hurt the flow-call edge? Motivated by 2026-07-08: TSLA/NVDA flow+ML CALLs bled to the -25%
stop while SPY was below VWAP / regime=bearish — flow CALLs bypass directional_regime.

Two gates swept in-memory over one API fetch (prod-faithful flow harness):
  1. OWN-underlying: skip calls where the call's own underlying is down > X% from day-open
     at entry (direct mirror of the put filter, which used the underlying's mkt_chg).
  2. SPY-broad:      skip calls where SPY is down > X% from its day-open at entry (targets
     today's "buying calls into a red market" leak).

Also splits INDEX (SPY) vs single-name, since the put filter's edge was index-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_gold_standard_report as F  # noqa: E402

BASE = 750.0  # flat $/trade edge measure (matches the flow report's FLAT column)


def _pnl(recs, conv=False):
    return sum(BASE * (r["conv_mult"] if conv else 1.0) * r["ret_pct"] / 100.0 for r in recs)


def _pf(recs, conv=False):
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


def _sweep(calls, field, label):
    """Skip calls where <field> < -thr (tape falling more than thr% at entry)."""
    print(f"\n{label} — skip CALLs where {field} < -X% (tape falling at entry)")
    print(f"  {'threshold':<12}{'kept':>6}{'skip':>6}{'skip_avg%':>11}"
          f"{'CALLS P&L':>12}{'PF':>7}{'WR':>6}{'vs none':>10}")
    print("  " + "-" * 72)
    base_pnl = None
    for thr in (None, 1.5, 1.0, 0.8, 0.5, 0.3, 0.0):
        if thr is None:
            kept, skipped = calls, []
        else:
            kept = [r for r in calls if r[field] >= -thr]
            skipped = [r for r in calls if r[field] < -thr]
        p = _pnl(kept)
        if thr is None:
            base_pnl = p
        skip_avg = (sum(r["ret_pct"] for r in skipped) / len(skipped)) if skipped else 0.0
        lbl = "none" if thr is None else f"<-{thr}%"
        dv = "" if thr is None else f"${p - base_pnl:+,.0f}"
        print(f"  {lbl:<12}{len(kept):>6}{len(skipped):>6}{skip_avg:>10.0f}%"
              f"{('$' + format(p, '+,.0f')):>12}{_pf(kept):>7.2f}{_wr(kept):>5.0f}%{dv:>10}")


def main():
    print("Fetching flow CALLs (prod-faithful harness)...")
    calls = F.collect(False, F.CALL_UNIV)
    dates = sorted({r["date"] for r in calls})
    print(f"{len(calls)} flow-call trades over {len(dates)} days "
          f"({dates[0]}..{dates[-1]}) — universe {sorted(F.CALL_UNIV)}")
    print(f"BASELINE (no filter): P&L ${_pnl(calls):+,.0f} flat | "
          f"PF {_pf(calls):.2f} | WR {_wr(calls):.0f}% | n={len(calls)}")

    _sweep(calls, "mkt_chg", "GATE 1: OWN-UNDERLYING (mirror of the put filter)")
    _sweep(calls, "spy_chg", "GATE 2: SPY-BROAD (today's leak)")

    # Index (SPY) vs single-name — the put filter's edge was index-only
    idx = [r for r in calls if r["ticker"] in F.INDEX_TICKERS]
    nono = [r for r in calls if r["ticker"] not in F.INDEX_TICKERS]
    print(f"\nINDEX (SPY) calls: n={len(idx)}  P&L ${_pnl(idx):+,.0f}  PF {_pf(idx):.2f}  WR {_wr(idx):.0f}%")
    if idx:
        _sweep(idx, "spy_chg", "  GATE 2 on INDEX-ONLY (SPY calls, SPY-broad)")
    print(f"\nSINGLE-NAME calls: n={len(nono)}  P&L ${_pnl(nono):+,.0f}  PF {_pf(nono):.2f}  WR {_wr(nono):.0f}%")
    if nono:
        _sweep(nono, "spy_chg", "  GATE 2 on SINGLE-NAME (SPY-broad)")

    # Sanity: how do the reddest-tape calls do? (deep counter-trend)
    deep = [r for r in calls if r["spy_chg"] < -0.5]
    print(f"\nCounter-trend deep (SPY < -0.5% at entry): n={len(deep)} "
          f"avg ret {(sum(r['ret_pct'] for r in deep)/len(deep) if deep else 0):.0f}% "
          f"P&L ${_pnl(deep):+,.0f}")


if __name__ == "__main__":
    main()
