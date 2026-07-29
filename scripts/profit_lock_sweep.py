"""Profit-lock vs wide-trail sweep (2026-07-14) — the "let moonshots run" prize (#6).

Audit finding: profit-lock (keep-0.8/arm-25, FSM gate 3.6) fires BEFORE the adaptive wide-trail (gate 8),
so a big runner closes on a 20%-of-peak dip and never reaches the 52-71% wide-trail tier it was tuned for.
This sweeps, through the REAL FSM (sim_trade):
  1. keep_frac  — 0.5 (loose, ride) .. 0.9 (tight, bank) at arm 25
  2. activate_pct — when the lock arms
  3. peak_exempt — a leg peaking past X% SKIPS the lock and rides the wide trail (V7_PROFIT_LOCK_PEAK_EXEMPT_PCT)
Reports total P&L, WR, and the MOONSHOT subset (peak>100%) separately — that's where the exemption should
show up. Split call/put (moonshots are mostly calls; puts ride slow crashes).

Baseline = prod (keep 0.8, arm 25, no exemption). Flow book (per-minute paths). Cheap — reuses the cache.

Usage: python scripts/profit_lock_sweep.py [--since 2026-03-01]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from exit_risk_sweep import BASE, CACHE, _mk_settings  # noqa: E402
from side_halt_backtest import sim_trade  # noqa: E402


def peak_gain(t):
    pp, ep = t["pp"], t["ep"]
    pk = 0.0
    for prem in pp[1:]:
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        pk = max(pk, (prem / ep - 1) * 100)
    return pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-01")
    args = ap.parse_args()
    trades = [t for t in pickle.load(open(CACHE, "rb")) if t["date"] >= args.since]
    peaks = {id(t): peak_gain(t) for t in trades}

    # --- characterize the peak distribution (do moonshots even exist in the flow book?) ---
    buckets = [("<25%", 0, 25), ("25-50", 25, 50), ("50-100", 50, 100),
               ("100-200", 100, 200), ("200%+", 200, 1e9)]
    print(f"\nProfit-lock sweep — {len(trades)} flow trades since {args.since}")
    print("\n  === peak-gain distribution (where the runners are) ===")
    for lab, lo, hi in buckets:
        sub = [t for t in trades if lo <= peaks[id(t)] < hi]
        calls = sum(1 for t in sub if t["otype"] == "call")
        print(f"  peak {lab:<8} {len(sub):>4} trades  ({calls} call / {len(sub)-calls} put)")

    def run(**over):
        S = _mk_settings(**over)
        tot = moon = mc = 0.0
        w = mn = 0
        for t in trades:
            r, _ = sim_trade(t, S)
            d = BASE * r / 100
            tot += d
            if r > 0:
                w += 1
            if peaks[id(t)] > 100:  # moonshot subset
                moon += d
                mn += 1
                if t["otype"] == "call":
                    mc += d
        return tot, w, moon, mn, mc

    b_tot, b_w, b_moon, b_mn, b_mc = run()  # baseline = prod defaults
    print(f"\n  BASELINE (keep 0.8, arm 25, no exempt): ${b_tot:+,.0f}  WR {100*b_w/len(trades):.0f}%"
          f"  | moonshots(peak>100%, n={b_mn}): ${b_moon:+,.0f} (calls ${b_mc:+,.0f})")

    print("\n  === 1) keep_frac sweep (arm 25) — loose=ride, tight=bank ===")
    print(f"  {'keep':>6}{'P&L':>11}{'Δ':>9}{'WR':>6}{'moonshot$':>12}{'Δmoon':>9}")
    for k in (0.5, 0.6, 0.7, 0.8, 0.9):
        tot, w, moon, mn, mc = run(V7_PROFIT_LOCK_KEEP_FRAC=k)
        print(f"  {k:>6.1f}{f'${tot:+,.0f}':>11}{f'{tot-b_tot:+,.0f}':>9}{f'{100*w/len(trades):.0f}%':>6}"
              f"{f'${moon:+,.0f}':>12}{f'{moon-b_moon:+,.0f}':>9}")

    print("\n  === 2) moonshot exemption (keep 0.8, arm 25) — peak>X rides the wide trail ===")
    print(f"  {'exempt':>7}{'P&L':>11}{'Δ':>9}{'WR':>6}{'moonshot$':>12}{'Δmoon':>9}")
    for x in (0, 75, 100, 150, 200, 300):
        tot, w, moon, mn, mc = run(V7_PROFIT_LOCK_PEAK_EXEMPT_PCT=float(x))
        tag = "off" if x == 0 else f">{x}%"
        print(f"  {tag:>7}{f'${tot:+,.0f}':>11}{f'{tot-b_tot:+,.0f}':>9}{f'{100*w/len(trades):.0f}%':>6}"
              f"{f'${moon:+,.0f}':>12}{f'{moon-b_moon:+,.0f}':>9}")

    print("\n  === 3) best keep x exempt combos ===")
    print(f"  {'keep':>6}{'exempt':>8}{'P&L':>11}{'Δ':>9}{'moonshot$':>12}")
    best = (b_tot, "baseline")
    for k in (0.6, 0.7, 0.8):
        for x in (0, 100, 150, 200):
            tot, w, moon, mn, mc = run(V7_PROFIT_LOCK_KEEP_FRAC=k, V7_PROFIT_LOCK_PEAK_EXEMPT_PCT=float(x))
            if tot > best[0]:
                best = (tot, f"keep {k} / exempt>{x}%")
            print(f"  {k:>6.1f}{('off' if x==0 else f'>{x}%'):>8}{f'${tot:+,.0f}':>11}"
                  f"{f'{tot-b_tot:+,.0f}':>9}{f'${moon:+,.0f}':>12}")
    print(f"\n  BEST: {best[1]} → ${best[0]:+,.0f} (baseline ${b_tot:+,.0f}, Δ${best[0]-b_tot:+,.0f})")
    print("\nNOTE: flow book skews short-DTE — if few trades peak >100%, the exemption is moot HERE and the")
    print("real moonshot prize is on the ML/call book (needs the gold-standard harness). keep_frac is the")
    print("lever that moves the common winners either way.")


if __name__ == "__main__":
    main()
