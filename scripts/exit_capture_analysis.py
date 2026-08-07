"""Why does the exit stack convert so little of the available upside?

Diagnostic, not a sweep. Runs on the REAL live trades + REAL recorded quote paths produced by
scripts/extract_live_exit_paths.py, and is deliberately calibration-independent: every number
below comes from the recorded path and the ACTUAL realised exit, never from a re-simulation.
(The replay only matches live exit reasons 35% of the time, so it cannot be trusted for this.)

The motivating fact: 85.5% of real CALL entries eventually reach +8%, 53% reach +50%,
36.5% reach +100% — and the book still lost money.

Three questions:
  Q1 CAPTURE   — of the peak gain available, how much did we actually realise?
  Q2 TIMING    — when does the peak arrive vs when do we exit? (structural mismatch?)
  Q3 PATH      — do winners DIP before they run? If so, early cuts kill them by construction,
                 and the fix is entry-relative dip tolerance, not threshold tuning.

Usage:
  python scripts/exit_capture_analysis.py --paths <pkl> [--otype call] [--min-mfe 50]
"""

from __future__ import annotations

import argparse
import pickle
import statistics as stats
from collections import defaultdict


def _bid(p: dict) -> float | None:
    b = p.get("bid")
    if b is None:
        b = p.get("mid")
    return b if (b is not None and b > 0) else None


def analyse(t: dict) -> dict | None:
    """Per-trade excursion profile from the real recorded path."""
    entry = t["entry"]
    path = t["path"]
    if entry <= 0 or len(path) < 5:
        return None
    t0 = path[0]["ts"]

    mfe_pct, mfe_i, mfe_min = -1e9, 0, 0.0
    for i, p in enumerate(path):
        b = _bid(p)
        if b is None:
            continue
        g = (b - entry) / entry * 100.0
        if g > mfe_pct:
            mfe_pct, mfe_i = g, i
            mfe_min = (p["ts"] - t0).total_seconds() / 60.0

    # Worst drawdown BEFORE the peak — what a trade had to survive to get there.
    pre_dd = 0.0
    for p in path[: mfe_i + 1]:
        b = _bid(p)
        if b is None:
            continue
        pre_dd = min(pre_dd, (b - entry) / entry * 100.0)

    # What we actually realised (real fill, not a re-sim).
    exit_prem = t.get("actual_exit_premium")
    realised = ((exit_prem - entry) / entry * 100.0) if exit_prem else None

    held = None
    if t.get("closed_at") and t.get("opened_at"):
        try:
            from datetime import datetime

            def _p(s):
                return datetime.fromisoformat(s.replace(" ", "T", 1))

            held = (_p(t["closed_at"]) - _p(t["opened_at"])).total_seconds() / 60.0
        except Exception:
            held = None

    return {
        "tk": t["tk"], "bot": t["bot"], "month": t["opened_at"][:7],
        "mfe": mfe_pct, "mfe_min": mfe_min, "pre_dd": pre_dd,
        "realised": realised, "held": held,
        "reason": (t.get("actual_exit_reason") or "none").split(":")[0],
        "contracts": t["contracts"], "pnl": t.get("actual_pnl") or 0.0,
        "entry": entry,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--otype", default="call")
    ap.add_argument("--live-only", action="store_true", default=True)
    ap.add_argument("--min-mfe", type=float, default=50.0,
                    help="Threshold for the 'big mover' cohort")
    args = ap.parse_args()

    trades = pickle.load(open(args.paths, "rb"))
    if args.live_only:
        trades = [t for t in trades if t["was_webull"]]
    if args.otype:
        trades = [t for t in trades if t["otype"] == args.otype]

    rows = [r for r in (analyse(t) for t in trades) if r]
    rows = [r for r in rows if r["realised"] is not None]
    print(f"{len(rows)} real-fill {args.otype.upper()} trades with a realised exit\n")

    # ── Q1 CAPTURE ────────────────────────────────────────────────────
    print("=" * 74)
    print("Q1  CAPTURE — realised vs peak available, by how big the move got")
    print("=" * 74)
    buckets = [(-1e9, 0), (0, 8), (8, 25), (25, 50), (50, 100), (100, 1e9)]
    names = ["never green", "0 to +8%", "+8 to +25%", "+25 to +50%",
             "+50 to +100%", "+100%+"]
    print(f"{'peak reached':<15} {'n':>4} {'avg peak':>9} {'avg realised':>13} "
          f"{'capture':>8} {'total $':>11}")
    print("-" * 74)
    for (lo, hi), nm in zip(buckets, names):
        sub = [r for r in rows if lo < r["mfe"] <= hi]
        if not sub:
            continue
        ap_ = stats.mean(r["mfe"] for r in sub)
        ar = stats.mean(r["realised"] for r in sub)
        cap = (ar / ap_ * 100) if ap_ > 0 else float("nan")
        tot = sum(r["pnl"] for r in sub)
        capf = f"{cap:>7.0f}%" if ap_ > 0 else "     n/a"
        print(f"{nm:<15} {len(sub):>4} {ap_:>8.0f}% {ar:>12.0f}% {capf} {tot:>11,.0f}")

    # ── Q2 TIMING ─────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("Q2  TIMING — when the peak arrives vs when we actually exit")
    print("=" * 74)
    big = [r for r in rows if r["mfe"] >= args.min_mfe]
    small = [r for r in rows if r["mfe"] < args.min_mfe]
    for nm, sub in (("peak >= +%.0f%%" % args.min_mfe, big),
                    ("peak <  +%.0f%%" % args.min_mfe, small)):
        if not sub:
            continue
        held = [r["held"] for r in sub if r["held"] is not None]
        print(f"  {nm:<18} n={len(sub):>4}  "
              f"median time-to-peak {stats.median(r['mfe_min'] for r in sub):>6.1f}m  "
              f"median hold {stats.median(held) if held else float('nan'):>6.1f}m")
    exited_early = [r for r in big if r["held"] is not None and r["held"] < r["mfe_min"]]
    if big:
        print(f"\n  Of the {len(big)} big movers, {len(exited_early)} "
              f"({100*len(exited_early)/len(big):.0f}%) were EXITED BEFORE their peak.")
        miss = sum((r["mfe"] - r["realised"]) / 100.0 * r["entry"] * r["contracts"] * 100
                   for r in exited_early)
        print(f"  Unrealised on those (peak vs realised, upper bound): ${miss:,.0f}")

    # ── Q3 PATH SHAPE ─────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("Q3  PATH — did the big movers DIP before they ran?")
    print("=" * 74)
    if big:
        dd = [r["pre_dd"] for r in big]
        print(f"  Drawdown BEFORE the peak, for trades that reached >= +{args.min_mfe:.0f}%:")
        print(f"    median {stats.median(dd):>6.1f}%   mean {stats.mean(dd):>6.1f}%   "
              f"worst {min(dd):>6.1f}%")
        for thr in (-5, -8, -12, -15, -20, -25):
            n = sum(1 for d in dd if d <= thr)
            print(f"    dipped past {thr:>4}% before running: {n:>3}/{len(dd)} "
                  f"({100*n/len(dd):>4.1f}%)   <- an {abs(thr)}% cut kills these")

    # ── exit-reason capture ───────────────────────────────────────────
    print("\n" + "=" * 74)
    print("Capture by ACTUAL exit gate (what each gate leaves behind)")
    print("=" * 74)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["reason"]].append(r)
    print(f"{'gate':<24} {'n':>4} {'avg peak':>9} {'avg realised':>13} {'total $':>11}")
    print("-" * 74)
    for g, sub in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(sub) < 3:
            continue
        print(f"{g:<24} {len(sub):>4} {stats.mean(r['mfe'] for r in sub):>8.0f}% "
              f"{stats.mean(r['realised'] for r in sub):>12.0f}% "
              f"{sum(r['pnl'] for r in sub):>11,.0f}")


if __name__ == "__main__":
    main()
