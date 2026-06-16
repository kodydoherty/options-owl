"""Validation for the two deploy candidates, with PER-MONTH consistency (not just aggregate):

  #2 Multi-level CALL adds 30/80/150 under M1 own-trail (each add its own fresh V7 FSM).
  #3 CALL-ONLY profit-lock layered on the V7 downside stops (NOT a pure rule): run the V7 FSM AND
     an additive "exit if gain < keep*peak_gain once peak>=act" check each minute, first to fire wins.
     This is exactly how it will ship — V7 keeps protecting the downside, the lock just tightens the
     give-back on the upside. Puts are untouched (V7 wide trail; the ratchet hurts puts).

Reports per-month so we can see the most recent weeks, not just the blended total.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
import backtest_add_handling as B  # noqa: E402
from options_owl.risk.exit_v5.fsm import ExitFSM, TradeState  # noqa: E402

H = B.HAIRCUT
KEEP, ACT = 0.60, 30.0          # #3 profit-lock params (call-only)
CALL_LEVELS = [30, 80, 150]


def sim_v7_plus_lock(t, keep=KEEP, act=ACT, lock=True, start_idx=0):
    """V7 FSM (downside + stops) + optional additive profit-lock upside gate, from start_idx.
    Returns ret% relative to the entry premium at start_idx."""
    pp, mp, up, ets, cfg, dte, otype = t["pp"], t["mp"], t["up"], t["ets"], t["cfg"], t["dte"], t["otype"]
    ep = pp[start_idx]
    fsm = ExitFSM(cfg, settings=D._S())
    st = TradeState(trade_id=1, ticker="X", option_type=otype, entry_premium=ep,
                    entry_time=ets + timedelta(minutes=int(mp[start_idx] - mp[0])),
                    contracts=1, peak_premium=ep, entry_underlying_price=up[start_idx], dte=dte,
                    expiry_date=ets.strftime("%Y-%m-%d"))
    last, peak_g = ep, 0.0
    for k in range(start_idx + 1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last = prem
        g = (prem - ep) / ep * 100
        peak_g = max(peak_g, g)
        now = ets + timedelta(minutes=int(mp[k] - mp[0]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        if lock and peak_g >= act and g < keep * peak_g:          # additive profit-lock
            return (prem * (1 - H) - ep) / ep * 100
        a = fsm.evaluate(st, prem, prem * (1 - H), prem, now,
                         current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if a.should_exit:                                          # V7 downside/stops
            return (prem * (1 - H) - ep) / ep * 100
    return (last * (1 - H) - ep) / ep * 100


def add_legs_m1(t, levels, lock=False):
    """M1 own-trail add legs for one trade: each crossed level -> a fresh ret% (optionally locked)."""
    out = []
    for L in levels:
        ai = B.add_index(t["pp"], t["mp"], t["pp"][0], L)
        if ai is None:
            continue
        out.append(sim_v7_plus_lock(t, lock=lock, start_idx=ai))
    return out


def row(name, rets):
    s = B.stats(rets)
    return f"{name:<22}{s[0]:>6}{s[1]:>+9.1f}{s[2]:>8.2f}{s[4]:>7.0f}%{s[3]:>+11.0f}"


def per_month(label, trades, fn):
    """fn(trade)->list of leg rets. Print per-month + total."""
    by = defaultdict(list)
    for t in trades:
        by[t["month"]].extend(fn(t))
    print(f"  {label}")
    print(f"  {'month':<22}{'legs':>6}{'mean%':>9}{'PF':>8}{'win%':>8}{'tot%':>11}")
    allr = []
    for m in sorted(by):
        allr.extend(by[m])
        print("  " + row(m, by[m]))
    print("  " + row("TOTAL", allr) + "\n")
    return allr


def main():
    print("loading flow paths...", flush=True)
    trades = B.load_paths()
    calls = [t for t in trades if t["otype"] == "call"]
    print(f"flow trades: {len(trades)} ({len(calls)} call)\n")

    print("=" * 70)
    print("#3 CALL profit-lock (keep60@+30, layered on V7 stops) — per month")
    print("=" * 70)
    per_month("V7 wide (now):", calls, lambda t: [sim_v7_plus_lock(t, lock=False)])
    per_month("V7 + profit-lock:", calls, lambda t: [sim_v7_plus_lock(t, lock=True)])

    print("=" * 70)
    print("#2 Multi-level CALL adds 30/80/150 (M1 own-trail) — per month, BOOK = base+adds")
    print("=" * 70)
    per_month("base only (V7):", calls, lambda t: [sim_v7_plus_lock(t, lock=False)])
    per_month("base + adds 30/80/150:", calls,
              lambda t: [sim_v7_plus_lock(t, lock=False)] + add_legs_m1(t, CALL_LEVELS))

    print("=" * 70)
    print("#2 + #3 COMBINED: profit-lock base + adds 30/80/150 (adds also locked) — per month")
    print("=" * 70)
    per_month("lock base + locked adds:", calls,
              lambda t: [sim_v7_plus_lock(t, lock=True)] + add_legs_m1(t, CALL_LEVELS, lock=True))


if __name__ == "__main__":
    main()
