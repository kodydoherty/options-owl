"""Sweep handling strategies for super-cheap "moonshot" 0DTE contracts (2026-07-10).

Trigger: SMCI 0DTE call at $0.26 premium sized to 120 contracts (budget / cost_per_contract
=> huge count on a cheap contract). Kody: "that seems risky / find a better way to handle these."

This reads a gold-standard trade dump (backtest_gold_standard.py --dump-trades, which stamps
`effective_entry` = per-contract premium, `effective_contracts`, `pnl` = actual sized $ P&L, and
`ret_pct` = size-neutral % return). It answers, in order:

  0. Do cheap-premium contracts actually UNDERPERFORM? (bucket by entry premium)
  A. PREMIUM FLOOR   — skip trades below $X premium. Do the cheap ones bleed, or are they the tail?
  B. CONTRACT CAP    — cap the lot at N contracts (pnl scales linearly with count).
  C. BUDGET HAIRCUT  — allocate fewer $ to cheap contracts (half-size the moonshots).

All three are linear rescalings of the dump's actual `pnl`, so they're measured directly — no
re-sim. Held to Kody's bar: recover risk WITHOUT cutting the moonshot right tail that pays for it.
(D = smaller-bet-then-double-down needs re-simulation via the add-handling harness — scoped separate.)

Usage: python scripts/moonshot_sizing_sweep.py <fullbook_6mo.json>
"""
import json
import sys
from collections import defaultdict

BASE = 750.0
CHEAP = 0.50  # "cheap moonshot" threshold ($/contract)


def wr(recs):
    return (sum(1 for r in recs if r["pnl"] > 0) / len(recs) * 100) if recs else 0.0


def pf(recs):
    g = sum(r["pnl"] for r in recs if r["pnl"] > 0)
    l = -sum(r["pnl"] for r in recs if r["pnl"] < 0)
    return g / l if l > 0 else float("inf")


def avg_ret(recs):
    return (sum(r["ret_pct"] for r in recs) / len(recs)) if recs else 0.0


def summ(lbl, recs):
    tot = sum(r["pnl"] for r in recs)
    ac = (sum(r["effective_contracts"] for r in recs) / len(recs)) if recs else 0
    print(f"  {lbl:<22}{len(recs):>5} tr  ${tot:>+10,.0f}  PF {pf(recs):>5.2f}  WR {wr(recs):>3.0f}%"
          f"  avgRet {avg_ret(recs):>+6.0f}%  avgLot {ac:>4.0f}")


def main():
    trades = json.load(open(sys.argv[1]))
    calls = [t for t in trades if t.get("direction") == "call"]
    days = sorted({t["day"] for t in calls})
    print(f"{len(calls)} CALL trades over {len(days)} days ({days[0]}..{days[-1]}); "
          f"total ${sum(t['pnl'] for t in calls):+,.0f}\n")

    # ---- 0. Do cheap contracts underperform? Bucket by entry premium ----
    print("=== 0. CALL P&L by entry-premium bucket (is 'cheap' actually worse?) ===")
    buckets = [(0, 0.25), (0.25, 0.50), (0.50, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9)]
    for lo, hi in buckets:
        b = [t for t in calls if lo <= t["effective_entry"] < hi]
        hedge = f"${lo:.2f}-{hi:.2f}" if hi < 1e9 else f"${lo:.2f}+"
        summ(hedge, b)
    cheap = [t for t in calls if t["effective_entry"] < CHEAP]
    print(f"\n  'cheap moonshots' (< ${CHEAP:.2f}): {len(cheap)} trades, "
          f"${sum(t['pnl'] for t in cheap):+,.0f} "
          f"({sum(t['pnl'] for t in cheap)/sum(t['pnl'] for t in calls)*100:.0f}% of call P&L), "
          f"tail: best +{max((t['ret_pct'] for t in cheap), default=0):.0f}% / "
          f"worst {min((t['ret_pct'] for t in cheap), default=0):.0f}%")

    daypnl = defaultdict(float)
    for t in calls:
        daypnl[t["day"]] += t["pnl"]
    base_win = {d: p for d, p in daypnl.items() if p > 0}

    def winday_check(scaled_by):
        """scaled_by(trade)->float multiplier on pnl; report win-day preservation + net delta."""
        nd = defaultdict(float)
        for t in calls:
            nd[t["day"]] += t["pnl"] * scaled_by(t)
        broke = sum(1 for d in base_win if nd[d] < 0)
        net = sum(nd.values()) - sum(daypnl.values())
        return broke, net

    # ---- A. PREMIUM FLOOR ----
    print("\n=== A. PREMIUM FLOOR — skip calls below $X (does cutting cheap help?) ===")
    print(f"  {'floor':<8}{'skipped':>8}{'skip P&L':>11}{'kept P&L':>11}{'netΔ':>10}{'brokeWin':>9}")
    full = sum(t["pnl"] for t in calls)
    for fl in (0.0, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0):
        sk = [t for t in calls if t["effective_entry"] < fl]
        kept = full - sum(t["pnl"] for t in sk)
        broke, _ = winday_check(lambda t, fl=fl: 0.0 if t["effective_entry"] < fl else 1.0)
        print(f"  <${fl:<6.2f}{len(sk):>8}${sum(t['pnl'] for t in sk):>+9,.0f}"
              f"${kept:>+9,.0f}${kept-full:>+8,.0f}{broke:>9}")

    # ---- B. CONTRACT-COUNT CAP (pnl scales linearly with count) ----
    print("\n=== B. CONTRACT CAP — cap lot at N (pnl *= min(N,lot)/lot) ===")
    print(f"  {'cap':<8}{'affected':>9}{'capped P&L':>12}{'netΔ':>10}{'brokeWin':>9}")
    for cap in (25, 40, 50, 75, 100, 150, None):
        def scale(t, cap=cap):
            if cap is None or t["effective_contracts"] <= cap:
                return 1.0
            return cap / t["effective_contracts"]
        aff = sum(1 for t in calls if cap is not None and t["effective_contracts"] > cap)
        tot = sum(t["pnl"] * scale(t) for t in calls)
        broke, _ = winday_check(scale)
        print(f"  {str(cap):<8}{aff:>9}${tot:>+10,.0f}${tot-full:>+8,.0f}{broke:>9}")

    # ---- C. BUDGET HAIRCUT on cheap contracts only ----
    print(f"\n=== C. BUDGET HAIRCUT — scale $ on cheap (< ${CHEAP:.2f}) calls only ===")
    print(f"  {'haircut':<9}{'cheap P&L':>12}{'book P&L':>11}{'netΔ':>10}{'brokeWin':>9}")
    for h in (1.0, 0.75, 0.50, 0.35, 0.25):
        def scale(t, h=h):
            return h if t["effective_entry"] < CHEAP else 1.0
        cheap_tot = sum(t["pnl"] * h for t in cheap)
        tot = sum(t["pnl"] * scale(t) for t in calls)
        broke, _ = winday_check(scale)
        print(f"  x{h:<8.2f}${cheap_tot:>+10,.0f}${tot:>+9,.0f}${tot-full:>+8,.0f}{broke:>9}")

    print("\nNOTE: A/B/C are linear rescalings of realized pnl (exits already applied). Positive netΔ")
    print("with 0 broken win-days = a real improvement. Watch: cutting cheap also cuts the moonshot tail.")


if __name__ == "__main__":
    main()
