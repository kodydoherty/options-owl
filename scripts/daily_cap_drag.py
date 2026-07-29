"""Daily-loss-cap DRAG backtest (2026-07-13).

The live DailyLossGate halts NEW entries once the day is down 10% of portfolio. Kody wants to know the
cost of TIGHTENING it (6-7%) and/or arming the 5% emergency close-all. Core question, same shape as the
side-halt test: after a day is down by threshold, are the REMAINING trades net winners (cap costs you)
or net losers (cap saves you)?

No lookahead: per day, entries in minute order; the running loss at a candidate's entry counts only
trades that have CLOSED by then (exit_mi <= entry_mi). Once cumulative realized loss <= -threshold, all
later NEW entries that day are skipped. Flow book, flat-$750, prod FSM. The threshold is in flow-book $
(this book is ~a slice of the real portfolio, so read the DIRECTION + the skipped W/L mix, not the exact
$ level). Reports how often it trips and whether the skipped bucket was winners or losers.

Usage: python scripts/daily_cap_drag.py [--since 2026-05-20]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from side_halt_backtest import sim_trade  # noqa: E402
from exit_risk_sweep import CACHE, _mk_settings, BASE  # noqa: E402


def pnl(rets):
    return sum(BASE * r / 100 for r in rets)


def run(sims, cap_dollars):
    """Halt NEW entries for the rest of a day once realized loss <= -cap_dollars."""
    byday = defaultdict(list)
    for s in sims:
        byday[s["date"]].append(s)
    kept, skipped, tripped_days = [], [], 0
    for day, rows in byday.items():
        rows.sort(key=lambda r: r["entry_mi"])
        tripped = False
        for cand in rows:
            if cap_dollars > 0:
                realized = pnl([o["ret"] for o in rows if o["exit_mi"] <= cand["entry_mi"]])
                if realized <= -cap_dollars:
                    tripped = True
                    skipped.append(cand)
                    continue
            kept.append(cand)
        if tripped:
            tripped_days += 1
    return kept, skipped, tripped_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-05-20")
    args = ap.parse_args()

    trades = [t for t in pickle.load(open(CACHE, "rb")) if t["date"] >= args.since]
    S = _mk_settings()
    sims = []
    for t in trades:
        ret, exit_mi = sim_trade(t, S)
        sims.append({"date": t["date"], "side": t["otype"], "entry_mi": t["mi"],
                     "exit_mi": exit_mi, "ret": ret})
    days = sorted({s["date"] for s in sims})
    base = pnl([s["ret"] for s in sims])
    # per-day P&L distribution to calibrate what a "cap" would even catch
    dpnl = defaultdict(list)
    for s in sims:
        dpnl[s["date"]].append(s["ret"])
    day_totals = sorted(pnl(v) for v in dpnl.values())
    print(f"\nDaily-cap drag — {len(sims)} flow trades, {len(days)} days ({days[0]}..{days[-1]})")
    print(f"BASELINE ${base:+,.0f}. Per-day P&L: worst ${day_totals[0]:+,.0f}, "
          f"median ${day_totals[len(day_totals)//2]:+,.0f}, best ${day_totals[-1]:+,.0f}\n")

    print(f"  {'halt new entries after day down':<34}{'kept':>5}{'skip':>6}{'days':>6}{'P&L':>10}{'Δ':>9}"
          f"{'skipped:W/L':>14}")
    for cap in (750, 1500, 2250, 3000, 4500):
        kept, skipped, tdays = run(sims, cap)
        kp = pnl([s["ret"] for s in kept])
        sk_w = sum(1 for s in skipped if s["ret"] > 0)
        sk_l = sum(1 for s in skipped if s["ret"] <= 0)
        sk_pnl = pnl([s["ret"] for s in skipped])
        print(f"  -${cap:<32,}{len(kept):>5}{len(skipped):>6}{tdays:>6}   ${kp:>+7,.0f}"
              f"{kp-base:>+9,.0f}   {sk_w}W/{sk_l}L (${sk_pnl:+,.0f})")

    print("\nVERDICT: tightening HELPS only if Δ is positive AND the skipped bucket is net-negative")
    print("(mostly losers). If skipped is net-positive, the cap is clipping the recovery — a pure drag.")
    print("Flow-only; scale the $ threshold mentally to your real portfolio. The direction is the signal.")


if __name__ == "__main__":
    main()
