"""Underlying-reversal EXIT overlay test (2026-07-13).

Chop-day pain has one shape: a trade goes +30%, then the underlying reverses and it round-trips through
breakeven to the -25% hardstop. Our trail is PREMIUM-based (reacts after the option has already bled).
This tests a faster, UNDERLYING-based exit: once in profit, track the underlying's favorable extreme; if
the underlying retraces R% against the position, get out NOW instead of waiting for the premium trail.

This is the classic tension: a tighter exit that rescues chop days ALSO clips runners on trend days. So we
measure BOTH: rescue on the 10 worst long-book days AND cost on the 10 best (runner) days. Ships only if
net-positive with acceptable runner cost.

No lookahead: the overlay only uses underlying data up to the current minute (a running favorable extreme).
Baseline = the prod FSM (sim_trade, incl -25% hardstop + profit-lock). Flow book (per-minute up[] paths).

Usage: python scripts/candle_exit_test.py [--since 2026-03-01]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from exit_risk_sweep import CACHE, BASE, HC, _mk_settings  # noqa: E402
from side_halt_backtest import sim_trade  # noqa: E402

WORST = {"2026-06-15", "2026-06-10", "2026-04-02", "2026-05-14", "2026-05-11",
         "2026-04-15", "2026-03-31", "2026-04-06", "2026-05-07", "2026-05-15"}
BEST = {"2026-05-26", "2026-05-08", "2026-06-25", "2026-05-06", "2026-04-21",
        "2026-03-26", "2026-04-23", "2026-05-28", "2026-06-09", "2026-06-11"}


def overlay_ret(t, base_ret, base_exit_mi, arm_pct, retrace_pct):
    """Return ret_pct with the underlying-reversal exit applied (or base_ret if it never fires)."""
    pp, mp, up, ep = t["pp"], t["mp"], t["up"], t["ep"]
    is_call = t["otype"] == "call"
    fav_ext = up[0]  # favorable extreme of underlying since entry
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        if int(mp[k]) >= base_exit_mi:
            break  # baseline already closed here
        u = up[k]
        gain = (prem / ep - 1) * 100
        if is_call:
            fav_ext = max(fav_ext, u)
            retrace = (fav_ext - u) / fav_ext * 100 if fav_ext > 0 else 0
        else:
            fav_ext = min(fav_ext, u)
            retrace = (u - fav_ext) / fav_ext * 100 if fav_ext > 0 else 0
        if gain >= arm_pct and retrace >= retrace_pct:
            return (prem * (1 - HC) - ep) / ep * 100, int(mp[k])
    return base_ret, base_exit_mi


def bucket_pnl(rows, days):
    return sum(BASE * r / 100 for d, r in rows if d in days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-01")
    args = ap.parse_args()
    trades = [t for t in pickle.load(open(CACHE, "rb")) if t["date"] >= args.since]
    S = _mk_settings()

    # baseline once
    base = []  # (date, ret, exit_mi, trade)
    for t in trades:
        r, xm = sim_trade(t, S)
        base.append((t["date"], r, xm, t))
    base_rows = [(d, r) for d, r, _, _ in base]
    b_all = sum(BASE * r / 100 for _, r in base_rows)
    b_worst = bucket_pnl(base_rows, WORST)
    b_best = bucket_pnl(base_rows, BEST)
    print(f"\nCandle/underlying-reversal EXIT overlay — {len(trades)} flow trades since {args.since}")
    print(f"BASELINE (prod FSM):  all ${b_all:+,.0f}   worst-10-days ${b_worst:+,.0f}   best-10-days ${b_best:+,.0f}\n")

    print(f"  {'arm%':>5}{'retrace%':>9}{'all P&L':>11}{'Δ all':>9}{'worst-days':>12}{'Δ worst':>9}"
          f"{'best-days':>11}{'Δ best':>9}{'fired':>7}")
    grid = [(a, r) for a in (10, 20) for r in (0.15, 0.25, 0.4, 0.6, 1.0)]
    for arm, rr in grid:
        rows, fired = [], 0
        for d, br, bx, t in base:
            r, xm = overlay_ret(t, br, bx, arm, rr)
            if xm != bx:
                fired += 1
            rows.append((d, r))
        a_all = sum(BASE * x / 100 for _, x in rows)
        a_w = bucket_pnl(rows, WORST)
        a_b = bucket_pnl(rows, BEST)
        print(f"  {arm:>5}{rr:>9.2f}{f'${a_all:+,.0f}':>11}{f'{a_all-b_all:+,.0f}':>9}"
              f"{f'${a_w:+,.0f}':>12}{f'{a_w-b_worst:+,.0f}':>9}"
              f"{f'${a_b:+,.0f}':>11}{f'{a_b-b_best:+,.0f}':>9}{fired:>7}")

    print("\nREAD: Δ worst > 0 = rescues chop days; Δ best < 0 = clips runners (the cost). Ships only if")
    print("Δ all > 0 with the runner clip acceptable. If Δ all <= 0, the premium trail already handles it.")


if __name__ == "__main__":
    main()
