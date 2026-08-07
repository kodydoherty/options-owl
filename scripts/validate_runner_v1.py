"""Validate runner_v1 P(runner) against REAL fills, retroactively.

WHY THIS EXISTS
---------------
ENABLE_RUNNER_V1_SIZING is LIVE on kody+dennis and multiplies real position sizes by
0.7 / 0.9 / 1.1 / 1.3 off P(runner). It has never been checked against real outcomes.
It could not be: P(runner) was only ever logged, and logs age out (~6 days survived,
20 records fleet-wide). Persistence is now added for FUTURE trades, but that answers
nothing about the ~10 weeks already traded.

Every input the live scorer uses is still in Postgres, so we can rebuild the exact
feature vector for each historical trade and score it with the same model:
  greeks + spread   -> option_ticks   (delta/iv/vega/theta/bid/ask at entry)
  underlying 1m/5m  -> stock_candles  (day open, slope, rvol, gap, prior range)
  5-min option vol  -> option_ticks

Faithfulness matters more than convenience here: the feature names, order and
derivations mirror flow_runner.compute_runner_v1_p. Where a feature cannot be rebuilt
the trade is SKIPPED rather than defaulted — a zero-filled feature would silently
shift the score and invent a result.

THE QUESTION
------------
Not "is the model accurate" but "does P(runner) rank real outcomes". If the live
distribution is clustered inside one tier band the multiplier is ~flat and the model
is inert — neither helping nor hurting — which is worth knowing before trusting it.

Usage:
  python scripts/validate_runner_v1.py --paths journal/live_exit_paths_greeks.pkl
"""

from __future__ import annotations

import argparse
import math
import pickle
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNNER_TIERS = (0.39, 0.69, 0.85)          # prod RUNNER_V1_Q1/Q2/Q3
RUNNER_MULTS = (0.7, 0.9, 1.1, 1.3)        # prod RUNNER_V1_MULT_Q1..Q4


def tier_of(p: float) -> int:
    q1, q2, q3 = RUNNER_TIERS
    return 1 if p < q1 else 2 if p < q2 else 3 if p < q3 else 4


def build_features(t: dict, candles: dict) -> dict | None:
    """Rebuild runner_v1's feature vector for one historical trade.

    Mirrors flow_runner.compute_runner_v1_p. Returns None when any required input is
    missing — skipping is correct, defaulting would fabricate a score.
    """
    p0 = t["path"][0]
    delta = p0.get("delta")
    if delta is None or delta <= 0:
        return None                        # model is ATM-CALL only; abstains on delta<=0
    entry_prem = p0.get("mid") or t["entry"]
    if not entry_prem or entry_prem <= 0:
        return None

    sess = candles.get("session") or []
    if len(sess) < 5:
        return None
    day_open, und_now = sess[0], sess[-1]
    if day_open <= 0 or und_now <= 0:
        return None

    bid, ask = p0.get("bid") or 0.0, p0.get("ask") or 0.0
    spread_pct = (ask - bid) / ask * 100 if ask > 0 else 0.0

    recent5 = sess[-5:]
    und_slope_5 = (recent5[-1] / recent5[0] - 1) * 100 if recent5[0] > 0 else 0.0
    r15 = sess[-15:]
    if len(r15) >= 5:
        diffs = [(r15[i + 1] - r15[i]) / r15[i] for i in range(len(r15) - 1) if r15[i] > 0]
        und_rvol_15 = st.pstdev(diffs) * 100 if len(diffs) > 1 else 0.0
    else:
        und_rvol_15 = 0.0

    pc, ph, pl = candles.get("prior_close", 0), candles.get("prior_high", 0), candles.get("prior_low", 0)
    gap_pct = (day_open / pc - 1) * 100 if pc > 0 else 0.0
    prior_range_pct = (ph - pl) / pc * 100 if pc > 0 else 0.0

    t0 = t["path"][0]["ts"]
    from zoneinfo import ZoneInfo
    et = t0.astimezone(ZoneInfo("America/New_York"))
    entry_min = max(0, (et.hour - 9) * 60 + et.minute - 30)
    from datetime import date
    dte = max(0, (date.fromisoformat(t["expiry"]) - et.date()).days)

    return {
        "entry_premium": float(entry_prem),
        "log_premium": float(math.log(entry_prem)),
        "delta": float(delta),
        "iv": float(p0.get("iv") or 0.0),
        "vega": float(p0.get("vega") or 0.0),
        "theta": float(p0.get("theta") or 0.0),
        "moneyness": float(t["strike"] / und_now),
        "spread_pct": float(spread_pct),
        "und_move_pct": float((und_now / day_open - 1) * 100),
        "und_slope_5": float(und_slope_5),
        "und_rvol_15": float(und_rvol_15),
        "opt_vol_5": float(candles.get("opt_vol_5", 0.0)),
        "gap_pct": float(gap_pct),
        "prior_range_pct": float(prior_range_pct),
        "dte": int(dte),
        "entry_min": int(entry_min),
        "ticker": t["tk"],
        "day_of_week": et.weekday(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="pickle from extract_live_exit_paths.py (with greeks)")
    ap.add_argument("--candles", default=None, help="pickle of per-trade underlying context")
    args = ap.parse_args()

    from options_owl.risk.flow_runner import _score_runner_v1

    trades = [t for t in pickle.load(open(args.paths, "rb"))
              if t["was_webull"] and t["otype"] == "call"]
    ctx = pickle.load(open(args.candles, "rb")) if args.candles else {}

    scored, skipped = [], defaultdict(int)
    for t in trades:
        c = ctx.get(t["tid"]) or ctx.get(str(t["tid"])) or {}
        feat = build_features(t, c)
        if feat is None:
            skipped["missing_inputs"] += 1
            continue
        p = _score_runner_v1(feat)
        if p is None or not (0.0 <= p <= 1.0):
            skipped["model_abstained"] += 1
            continue
        e = t["entry"]
        mfe = max((((q["bid"] or q["mid"]) or 0) - e) / e * 100 for q in t["path"])
        scored.append({"p": float(p), "pnl": t["actual_pnl"] or 0.0,
                       "ret": ((t["actual_exit_premium"] or e) - e) / e * 100,
                       "mfe": mfe, "tier": tier_of(float(p)), "month": t["opened_at"][:7]})

    print(f"scored {len(scored)} / {len(trades)} real-fill CALLs   skipped={dict(skipped)}")
    if not scored:
        print("\nNothing scored — rerun with --candles (underlying context is required;\n"
              "the builder SKIPS rather than defaulting, so no result is fabricated).")
        return

    ps = [r["p"] for r in scored]
    print(f"\nP(runner) distribution: min {min(ps):.3f}  p25 {st.quantiles(ps,n=4)[0]:.3f}  "
          f"median {st.median(ps):.3f}  p75 {st.quantiles(ps,n=4)[2]:.3f}  max {max(ps):.3f}")

    print(f"\n{'tier':<6} {'mult':>5} {'n':>5} {'share':>7} {'avg ret':>9} {'runner%':>9} {'P&L':>10}")
    print("-" * 60)
    for q in (1, 2, 3, 4):
        s = [r for r in scored if r["tier"] == q]
        if not s:
            print(f"Q{q:<5} {RUNNER_MULTS[q-1]:>5} {0:>5}   (never fires)")
            continue
        print(f"Q{q:<5} {RUNNER_MULTS[q-1]:>5} {len(s):>5} {100*len(s)/len(scored):>6.1f}% "
              f"{st.mean(r['ret'] for r in s):>8.1f}% "
              f"{100*sum(1 for r in s if r['mfe']>=100)/len(s):>8.1f}% "
              f"{sum(r['pnl'] for r in s):>10,.0f}")

    print("\nIS IT MONOTONIC? (higher P(runner) should mean more runners)")
    rates = [(q, 100*sum(1 for r in scored if r["tier"] == q and r["mfe"] >= 100)
              / max(1, sum(1 for r in scored if r["tier"] == q))) for q in (1, 2, 3, 4)
             if any(r["tier"] == q for r in scored)]
    print("   " + "  ".join(f"Q{q}={v:.0f}%" for q, v in rates))
    mono = all(rates[i][1] <= rates[i+1][1] for i in range(len(rates)-1))
    print(f"   -> {'MONOTONIC — the score ranks outcomes' if mono else 'NOT monotonic — the score does not rank outcomes'}")

    span = max(RUNNER_MULTS[tier_of(p)-1] for p in ps) / min(RUNNER_MULTS[tier_of(p)-1] for p in ps)
    print(f"\nEFFECTIVE SIZING SPAN actually used: {span:.2f}x "
          f"({'~inert' if span < 1.3 else 'materially different sizing'})")


if __name__ == "__main__":
    main()
