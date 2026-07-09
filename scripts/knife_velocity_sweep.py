"""Sweep a VELOCITY / falling-knife entry filter on ML CALL trades (2026-07-09).

Reads a gold-standard trade dump (backtest_gold_standard.py --dump-trades) that now stamps each
call trade with the underlying's velocity at the DECISION minute (u_chg_5m, u_chg_15m). Question:
does skipping ML calls bought while the underlying is DIVING (velocity < -X% over 5m or 15m) help,
and what's the optimal threshold — globally, per-ticker, and does it hold in the recent regime?

Held to the user's standard: RECOVER losing days, break ZERO winning days. Flat-$ edge measure
(BASE/trade) so the filter effect isn't conflated with position sizing.

Usage: python scripts/knife_velocity_sweep.py <trades.json> [--recent-start 2026-05-01]
"""
import argparse
import json
from collections import defaultdict

BASE = 750.0
RECENT_START = "2026-05-01"


def pnl(recs):
    return sum(BASE * r["ret_pct"] / 100.0 for r in recs)


def pf(recs):
    g = l = 0.0
    for r in recs:
        v = BASE * r["ret_pct"] / 100.0
        if v > 0:
            g += v
        else:
            l -= v
    return g / l if l > 0 else float("inf")


def wr(recs):
    return (sum(1 for r in recs if r["ret_pct"] > 0) / len(recs) * 100) if recs else 0.0


def byday(recs):
    d = defaultdict(float)
    for r in recs:
        d[r["day"]] += BASE * r["ret_pct"] / 100.0
    return d


def sweep(calls, field, label, thresholds=(None, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5)):
    print(f"\n{label} — skip calls where {field} < -X% (diving at entry)")
    print(f"  {'thr':<9}{'kept':>6}{'skip':>6}{'skip_avg%':>11}{'P&L':>11}{'PF':>7}{'WR':>6}{'vs none':>10}")
    print("  " + "-" * 66)
    base = None
    for thr in thresholds:
        if thr is None:
            kept, sk = calls, []
        else:
            kept = [r for r in calls if r.get(field, 0.0) >= -thr]
            sk = [r for r in calls if r.get(field, 0.0) < -thr]
        p = pnl(kept)
        if thr is None:
            base = p
        savg = (sum(r["ret_pct"] for r in sk) / len(sk)) if sk else 0.0
        lbl = "none" if thr is None else f"<-{thr}%"
        dv = "" if thr is None else f"${p - base:+,.0f}"
        print(f"  {lbl:<9}{len(kept):>6}{len(sk):>6}{savg:>10.0f}%${p:>+9,.0f}{pf(kept):>7.2f}{wr(kept):>5.0f}%{dv:>10}")


def winday(calls, field, thr, label):
    base_day = byday(calls)
    kept = [r for r in calls if r.get(field, 0.0) >= -thr]
    filt_day = byday(kept)
    win = {d: p for d, p in base_day.items() if p > 0}
    los = {d: p for d, p in base_day.items() if p <= 0}
    wb, wf = sum(win.values()), sum(filt_day.get(d, 0.0) for d in win)
    lb, lf = sum(los.values()), sum(filt_day.get(d, 0.0) for d in los)
    broke = sum(1 for d in win if filt_day.get(d, 0.0) < 0)
    print(f"  {label} (skip {field} < -{thr}%):  "
          f"WIN days {len(win)}: ${wb:+,.0f}->${wf:+,.0f} (kept {wf/wb*100 if wb else 0:.0f}%, {broke} broke) | "
          f"LOSE days {len(los)}: ${lb:+,.0f}->${lf:+,.0f} (rec ${lf-lb:+,.0f}) | NET ${(wf+lf)-(wb+lb):+,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--recent-start", default=RECENT_START)
    args = ap.parse_args()

    trades = json.load(open(args.dump))
    calls = [t for t in trades if t.get("direction") == "call"]
    days = sorted({t["day"] for t in calls})
    recent = [t for t in calls if t["day"] >= args.recent_start]
    print(f"{len(calls)} call trades over {len(days)} days ({days[0]}..{days[-1]}); "
          f"{len(recent)} in recent regime (>= {args.recent_start})")

    # winner vs loser velocity distribution — is 'diving at entry' actually predictive?
    W = [t for t in calls if t["ret_pct"] > 0]
    L = [t for t in calls if t["ret_pct"] <= 0]
    for fld in ("u_chg_5m", "u_chg_15m"):
        aw = sum(t.get(fld, 0) for t in W) / len(W) if W else 0
        al = sum(t.get(fld, 0) for t in L) / len(L) if L else 0
        print(f"  avg {fld}:  winners {aw:+.2f}%   losers {al:+.2f}%   (gap {aw-al:+.2f})")

    for period, recs in (("FULL 6-MONTH", calls), (f"RECENT ({args.recent_start}+)", recent)):
        print("\n" + "=" * 70 + f"\n{period}: baseline ${pnl(recs):+,.0f}  PF {pf(recs):.2f}  "
              f"WR {wr(recs):.0f}%  n={len(recs)}\n" + "=" * 70)
        sweep(recs, "u_chg_5m", "5-min velocity")
        sweep(recs, "u_chg_15m", "15-min velocity")

    print("\n" + "=" * 70 + "\nWINNING-DAY PRESERVATION (full period) — recover losers, break 0 winners\n" + "=" * 70)
    for f, t in (("u_chg_5m", 0.5), ("u_chg_5m", 1.0), ("u_chg_5m", 1.5),
                 ("u_chg_15m", 0.8), ("u_chg_15m", 1.5), ("u_chg_15m", 2.0)):
        winday(calls, f, t, f"{f}")

    # Per-ticker optimum: for each ticker, the 5m/15m threshold maximizing net vs baseline
    print("\n" + "=" * 70 + "\nPER-TICKER OPTIMUM (5m & 15m velocity, full period)\n" + "=" * 70)
    print(f"  {'ticker':<7}{'n':>4}{'base$':>9}  best-gate            {'gated$':>9}{'gain':>9}{'brokeWin':>9}")
    byT = defaultdict(list)
    for t in calls:
        byT[t["ticker"]].append(t)
    for tk in sorted(byT, key=lambda k: -len(byT[k])):
        recs = byT[tk]
        if len(recs) < 15:
            continue
        b = pnl(recs)
        best = (0.0, "none", b, None, 0)  # gain, label, gated_pnl, (f,t)
        for f in ("u_chg_5m", "u_chg_15m"):
            for thr in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5):
                kept = [r for r in recs if r.get(f, 0.0) >= -thr]
                g = pnl(kept) - b
                if g > best[0]:
                    bd = byday(recs); fd = byday(kept)
                    broke = sum(1 for d in bd if bd[d] > 0 and fd.get(d, 0) < 0)
                    best = (g, f"{f}<-{thr}%", pnl(kept), (f, thr), broke)
        print(f"  {tk:<7}{len(recs):>4}${b:>+7,.0f}  {best[1]:<20}${best[2]:>+7,.0f}${best[0]:>+7,.0f}{best[4]:>9}")


if __name__ == "__main__":
    main()
