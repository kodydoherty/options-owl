"""Chop-day detector test (2026-07-13).

Kody's -$1,780 whipsaw day: both calls AND puts got stopped out (chop, not a trend). Can we RECOGNIZE a
chop day early from the TAPE and sit out (or size down) the rest of it? This is distinct from the refuted
loss-count halt (which used our own P&L) — here we measure SPY's regime directly.

Metric: Kaufman EFFICIENCY RATIO over the first M minutes of SPY = |net move| / (sum of |bar-to-bar
moves|). ~1 = clean trend, ~0 = choppy/directionless. No lookahead: ER is computed from minutes [0, M];
it can only gate entries made AFTER minute M.

The REAL test isn't "does filtering help on average" — it's does early-ER actually PREDICT the rest of
the day? So we bucket days by early-ER and show the rest-of-day book P&L per bucket. If choppy-morning
days (low ER) have negative rest-of-day P&L and trendy ones positive, the signal is real and a filter
works. If there's no separation, chop isn't predictable early (same wall as regime-timing before).

Usage: python scripts/chop_day_test.py [--window 60] [--since 2026-05-20]
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402
from side_halt_backtest import sim_trade  # noqa: E402
from exit_risk_sweep import CACHE, _mk_settings, BASE  # noqa: E402


def efficiency_ratio(spy_day, window_min):
    """ER over minutes [0, window_min] from SPY 5-min closes. None if insufficient data."""
    mins = sorted(m for m in spy_day if 0 <= m <= window_min)
    if len(mins) < 4:
        return None
    px = [spy_day[m] for m in mins]
    net = abs(px[-1] - px[0])
    path = sum(abs(px[i] - px[i - 1]) for i in range(1, len(px)))
    if path <= 0:
        return None
    return net / path


def pnl(rets):
    return sum(BASE * r / 100 for r in rets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60, help="minutes after open to measure ER")
    ap.add_argument("--since", default="2026-05-20")
    args = ap.parse_args()
    M = args.window

    trades = [t for t in pickle.load(open(CACHE, "rb")) if t["date"] >= args.since]
    spy = D._stock("SPY")
    S = _mk_settings()

    # per-day ER + rest-of-day (entry after M) trade rets
    day_er, day_rest, day_all = {}, defaultdict(list), defaultdict(list)
    for t in trades:
        ret, _ = sim_trade(t, S)
        day_all[t["date"]].append(ret)
        if t["mi"] > M:
            day_rest[t["date"]].append(ret)
    for d in day_all:
        er = efficiency_ratio(spy.get(d, {}), M) if d in spy else None
        if er is not None:
            day_er[d] = er

    days = sorted(day_er)
    print(f"\nChop-day test — ER over first {M} min of SPY, {len(days)} days "
          f"({days[0]}..{days[-1]}), {len(trades)} trades\n")

    # THE diagnostic: does early-ER predict rest-of-day P&L?
    print("  === does a choppy morning (low ER) predict a losing rest-of-day? ===")
    quart = sorted(day_er.values())
    q1, q2, q3 = quart[len(quart)//4], quart[len(quart)//2], quart[3*len(quart)//4]
    print(f"  ER quartile cuts: q1={q1:.2f} q2={q2:.2f} q3={q3:.2f}  (low ER = choppy)\n")
    buckets = [("choppiest (ER<q1)", lambda e: e < q1),
               ("q1-q2", lambda e: q1 <= e < q2),
               ("q2-q3", lambda e: q2 <= e < q3),
               ("trendiest (ER>=q3)", lambda e: e >= q3)]
    print(f"  {'bucket':<22}{'days':>5}{'rest-of-day P&L':>18}{'full-day P&L':>16}")
    for label, cond in buckets:
        bd = [d for d in days if cond(day_er[d])]
        rest = pnl([r for d in bd for r in day_rest[d]])
        full = pnl([r for d in bd for r in day_all[d]])
        print(f"  {label:<22}{len(bd):>5}{f'${rest:+,.0f}':>18}{f'${full:+,.0f}':>16}")

    # filter sim: skip entries after M on days flagged choppy (ER < threshold)
    print(f"\n  === filter: skip post-{M}min entries when morning ER < threshold ===")
    base = pnl([r for d in days for r in day_all[d]])
    print(f"  {'ER threshold':<16}{'days cut':>9}{'P&L':>11}{'Δ':>10}{'skipped W/L':>16}")
    for thr in (0.15, 0.20, 0.25, 0.30, 0.40):
        kept, skip = [], []
        cut_days = 0
        for d in days:
            choppy = day_er[d] < thr
            if choppy and day_rest[d]:
                cut_days += 1
            for t in trades:
                if t["date"] != d:
                    continue
                ret, _ = sim_trade(t, S)
                if choppy and t["mi"] > M:
                    skip.append(ret)
                else:
                    kept.append(ret)
        kp = pnl(kept)
        sk_w = sum(1 for r in skip if r > 0); sk_l = sum(1 for r in skip if r <= 0)
        print(f"  ER < {thr:<11.2f}{cut_days:>9}{f'${kp:+,.0f}':>11}{f'{kp-base:+,.0f}':>10}"
              f"{f'{sk_w}W/{sk_l}L':>16}")

    print("\nVERDICT: the BUCKET table is the truth. If choppiest-morning days DON'T have a clearly worse")
    print("rest-of-day than trendy ones, chop isn't predictable early → can't sit it out (recompute the")
    print("filter Δ won't be robust). If low-ER days ARE clearly negative, it's a real, deployable signal.")


if __name__ == "__main__":
    main()
