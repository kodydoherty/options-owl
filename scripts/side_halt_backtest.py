"""Asymmetric CALL/PUT intraday halt backtest (2026-07-13).

Kody's idea after a -$1,347 chop day: if one SIDE isn't working today (a string of call losses on a
down tape, or puts on a rip), CUT that side for the rest of the day to stop the bleed — while leaving
the working side on. This tests whether that early-halt actually saves money over the last ~6 choppy
weeks, or just clips winners (the failure mode that killed the old portfolio-wide consecutive-loss
breaker).

No lookahead: entries are processed in entry-minute order; the halt for side S at minute m counts only
trades of side S that have ALREADY CLOSED (exit_mi <= m) as losers — exactly the info a live bot has.
After N closed losers of a side, new entries of THAT side are skipped for the rest of the day; the other
side keeps trading. Baseline = take everything. Flat-$750, same FSM exits (prod-faithful, incl -25%
hardstop + profit-lock). Flow book only (ML book would need the gold-standard harness).

Usage: python scripts/side_halt_backtest.py [--since 2026-05-20]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import flow_gold_standard_report as R  # noqa: E402
from exit_risk_sweep import CACHE, _base_cfg, _mk_settings, HC, BASE  # noqa: E402


def sim_trade(t, settings):
    """Return (ret_pct, exit_mi) for one trade under the prod FSM."""
    cfg = _base_cfg(t["tk"], t["otype"])
    pp, mp, up, ep = t["pp"], t["mp"], t["up"], t["ep"]
    fsm = R.ExitFSM(cfg, settings=settings)
    st = R.TradeState(trade_id=1, ticker=t["tk"], option_type=t["otype"], entry_premium=ep,
                      entry_time=t["ets"], contracts=1, peak_premium=ep,
                      entry_underlying_price=up[0], dte=t["dte0"],
                      expiry_date=t["ets"].strftime("%Y-%m-%d"))
    last, last_mi = ep, int(mp[-1])
    for k in range(1, len(pp)):
        prem = pp[k]
        if prem is None or np.isnan(prem) or prem <= 0:
            continue
        last, last_mi = prem, int(mp[k])
        now = t["ets"] + timedelta(minutes=int(mp[k] - mp[0]))
        mtc = max(0, 960 - (now.hour * 60 + now.minute))
        act = fsm.evaluate(st, prem, prem * (1 - HC), prem, now,
                           current_underlying=up[k], minutes_to_close=mtc, candle_data={})
        if act.should_exit:
            return (prem * (1 - HC) - ep) / ep * 100, int(mp[k])
    return (last * (1 - HC) - ep) / ep * 100, last_mi


def pnl(rets):
    return sum(BASE * r / 100 for r in rets)


def run(sims, halt_n):
    """sims: list of dicts {date, side, entry_mi, exit_mi, ret}. Return (kept_pnl, skipped)."""
    byday = defaultdict(list)
    for s in sims:
        byday[s["date"]].append(s)
    kept, skipped = [], []
    for day, rows in byday.items():
        rows.sort(key=lambda r: r["entry_mi"])
        for cand in rows:
            if halt_n > 0:
                closed_losers = sum(
                    1 for o in rows
                    if o["side"] == cand["side"] and o["exit_mi"] <= cand["entry_mi"] and o["ret"] < 0
                )
                if closed_losers >= halt_n:
                    skipped.append(cand)
                    continue
            kept.append(cand)
    return kept, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-05-20", help="only days >= this (the choppy window)")
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
    print(f"\nSide-halt backtest — {len(sims)} flow trades, {len(days)} days "
          f"({days[0]}..{days[-1]}), flat ${BASE:.0f}\n")
    print(f"  BASELINE (take all)                 {len(sims):>4} tr   ${base:>+8,.0f}\n")

    print(f"  {'halt side after N closed losers':<34}{'kept':>5}{'skip':>6}{'P&L':>10}{'Δ':>9}"
          f"{'skipped:W/L':>13}")
    for n in (2, 3, 4, 5):
        kept, skipped = run(sims, n)
        kp = pnl([s["ret"] for s in kept])
        sk_w = sum(1 for s in skipped if s["ret"] > 0)
        sk_l = sum(1 for s in skipped if s["ret"] <= 0)
        sk_pnl = pnl([s["ret"] for s in skipped])
        print(f"  N={n} (both sides)                  {len(kept):>5}{len(skipped):>6}"
              f"   ${kp:>+7,.0f}{kp-base:>+9,.0f}   {sk_w}W/{sk_l}L (${sk_pnl:+,.0f})")

    # side-specific: does halting CALLS help but PUTS hurt (or vice versa)?
    print("\n  === which side benefits? (halt only that side, N=3) ===")
    for side in ("call", "put"):
        byday = defaultdict(list)
        for s in sims:
            byday[s["date"]].append(s)
        kept = []
        skipped = []
        for day, rows in byday.items():
            rows.sort(key=lambda r: r["entry_mi"])
            for cand in rows:
                if cand["side"] == side:
                    cl = sum(1 for o in rows if o["side"] == side
                             and o["exit_mi"] <= cand["entry_mi"] and o["ret"] < 0)
                    if cl >= 3:
                        skipped.append(cand); continue
                kept.append(cand)
        kp = pnl([s["ret"] for s in kept])
        sk_w = sum(1 for s in skipped if s["ret"] > 0)
        sk_l = sum(1 for s in skipped if s["ret"] <= 0)
        print(f"    halt {side.upper()}S only after 3 losers   ${kp:>+8,.0f}  (Δ${kp-base:+,.0f})  "
              f"skipped {len(skipped)} = {sk_w}W/{sk_l}L")

    print("\nVERDICT: a halt only helps if Δ is POSITIVE and the skipped bucket is mostly LOSERS (L>>W).")
    print("If it skips as many winners as losers, it's clipping the recovery — the old breaker's failure.")
    print("Flow-only + no-lookahead; the ML book (today's call pain) needs a separate run.")


if __name__ == "__main__":
    main()
