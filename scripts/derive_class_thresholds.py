"""Derive volatility-normalised exit thresholds per ticker group.

Consumes scripts/measure_move_profile.py output.

THE PROBLEM IT SOLVES
---------------------
Exit thresholds are absolute percentages tuned on the tech/index book. The expansion
tickers move about half as far (live fills: 13.6% avg MFE vs 23.4%), so a +25%
profit-lock arm is unreachable for them — their live exit-reason mix contains ONLY loss
gates. They cannot win by construction, regardless of signal quality.

METHOD — PERCENTILE MATCHING, NOT A RATIO
-----------------------------------------
A naive fix is to scale every threshold by 13.6/23.4. That assumes the two distributions
have the same SHAPE and differ only in scale, which is exactly the kind of assumption
that has failed repeatedly this session. Instead:

  1. Find where each live threshold sits in the BASELINE (tech) distribution — e.g. "the
     +25% profit-lock arm is at the 71st percentile of tech MFE".
  2. Read the value at that SAME percentile off each other group's own distribution.

If the distributions really are just scaled, this agrees with the ratio. Where they are
not, percentile matching is right and the ratio is wrong. Both are printed so the
difference is visible rather than hidden.

READ THIS BEFORE USING THE NUMBERS
----------------------------------
* The reference entry is a fixed clock time, not a signal, so these are the ticker's move
  ENVELOPE — not what a real trade earns. Use them for RELATIVE calibration only.
* ~10 weeks of option_ticks (starts 2026-05-26). One regime, not a cycle.
* This says nothing about whether a ticker is worth trading. The class re-run says the
  expansion book is not: leveraged PF 0.65, crypto PF 0.74, commodity 0 trades, and only
  HOOD is positive — and even that is inside the harness's ~$84/trade optimism.
  Right-sized thresholds on a -EV book still lose; this removes a structural handicap,
  it does not create an edge.
"""

from __future__ import annotations

import argparse
import pickle
import statistics as st

# Live values this is calibrating against (settings.py / exit_v5 config).
LIVE = {
    "profit_lock_arm": 25.0,      # V7_PROFIT_LOCK_ACTIVATE_PCT
    "scalp_target": 35.0,         # SCALP_TARGET_PCT
    "breakeven_trigger": 20.0,    # V6_BREAKEVEN_TRIGGER_PCT
    "scaleout_gain": 20.0,        # V6_SCALEOUT_GAIN_PCT
}
LIVE_DOWN = {
    "nevergreen_loss": -8.0,      # NEVERGREEN_CUT_LOSS_PCT
    "premium_hardstop": -25.0,    # PREMIUM_HARDSTOP_0DTE_PCT
}


def pct_rank(sorted_vals: list[float], x: float) -> float:
    """Percentile of x within sorted_vals (0-100)."""
    if not sorted_vals:
        return float("nan")
    n = sum(1 for v in sorted_vals if v <= x)
    return 100.0 * n / len(sorted_vals)


def at_pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--baseline", default="tech")
    args = ap.parse_args()

    blob = pickle.load(open(args.profile, "rb"))
    groups, prof = blob["groups"], blob["profile"]

    gm: dict[str, dict] = {}
    for g, tickers in groups.items():
        mfe = sorted(s["mfe"] for t in tickers for s in prof.get(t, []))
        mae = sorted(s["mae"] for t in tickers for s in prof.get(t, []))
        if mfe:
            gm[g] = {"mfe": mfe, "mae": mae, "n": len(mfe)}

    print("MOVE PROFILE (option_ticks, trade-independent)\n")
    print(f"{'group':<11} {'n':>5} {'MFE p50':>9} {'MFE p75':>9} {'MFE p90':>9} {'MAE p50':>9} {'MAE p25':>9}")
    print("-" * 68)
    for g, d in gm.items():
        print(f"{g:<11} {d['n']:>5} {st.median(d['mfe']):>8.1f}% {at_pct(d['mfe'],75):>8.1f}% "
              f"{at_pct(d['mfe'],90):>8.1f}% {st.median(d['mae']):>8.1f}% {at_pct(d['mae'],25):>8.1f}%")

    base = gm.get(args.baseline)
    if not base:
        print(f"\nbaseline group '{args.baseline}' has no samples — cannot derive")
        return

    print(f"\n\nDERIVED THRESHOLDS (percentile-matched to '{args.baseline}')")
    print("ratio = naive scale by median MFE, shown so the difference is visible\n")
    for name, live in LIVE.items():
        p = pct_rank(base["mfe"], live)
        print(f"  {name}  (live +{live:.0f}%, = p{p:.0f} of {args.baseline} MFE)")
        for g, d in gm.items():
            matched = at_pct(d["mfe"], p)
            ratio = live * (st.median(d["mfe"]) / max(1e-9, st.median(base["mfe"])))
            flag = "" if g == args.baseline else ("   <- differs from ratio"
                                                  if abs(matched - ratio) > 0.15 * max(1.0, abs(ratio)) else "")
            print(f"      {g:<11} percentile-matched {matched:>6.1f}%   ratio {ratio:>6.1f}%{flag}")
        print()
    for name, live in LIVE_DOWN.items():
        p = pct_rank(base["mae"], live)
        print(f"  {name}  (live {live:.0f}%, = p{p:.0f} of {args.baseline} MAE)")
        for g, d in gm.items():
            matched = at_pct(d["mae"], p)
            ratio = live * (abs(st.median(d["mae"])) / max(1e-9, abs(st.median(base["mae"]))))
            print(f"      {g:<11} percentile-matched {matched:>6.1f}%   ratio {ratio:>6.1f}%")
        print()

    print("REACHABILITY — share of day-samples that ever reach the LIVE +25% arm:")
    for g, d in gm.items():
        share = 100.0 * sum(1 for v in d["mfe"] if v >= LIVE["profit_lock_arm"]) / len(d["mfe"])
        print(f"  {g:<11} {share:>5.1f}%")
    print("\n  A group far below the baseline here cannot reach the only exit gate with")
    print("  positive capture, and can therefore only ever exit through loss gates.")


if __name__ == "__main__":
    main()
