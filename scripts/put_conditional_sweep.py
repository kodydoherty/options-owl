"""Test CONDITIONAL relaxation of the PUT exclusion for banned high-beta names (2026-07-09).

PLTR/AMD/MSTR/AVGO (+AMZN/GOOGL) are banned from puts — they dip intraday then bounce, so puts on
them lose over the full sample. Hypothesis (Kody): on a CONFIRMED sustained down-day (name down hard
from open AND SPY bearish) their puts DO win, and we're missing those. Test: run the harness with the
ban LIFTED (PUT_INCLUDE_BLACKLIST=1, --dump-trades), then in-memory:
  - UNCONDITIONAL: take all banned-name puts (the ban's justification — should be a net loser).
  - CONDITIONAL: take them only when put_und_move < -X% AND is_bear_mode (SPY down) — does THAT subset win?

Held to: the conditional subset must be clearly +EV AND not just an in-sample fluke. Flat-$ edge (BASE/trade).

Usage: python scripts/put_conditional_sweep.py <put_6mo.json>
"""
import json
import sys
from collections import defaultdict

BASE = 750.0
BANNED = {"PLTR", "AMD", "MSTR", "AVGO", "AMZN", "GOOGL"}


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


def line(lbl, recs):
    print(f"  {lbl:<34}{len(recs):>5} trades   ${pnl(recs):>+9,.0f}   PF {pf(recs):>5.2f}   WR {wr(recs):>3.0f}%")


def main():
    trades = json.load(open(sys.argv[1]))
    puts = [t for t in trades if t.get("direction") == "put"]
    banned = [t for t in puts if t["ticker"] in BANNED]
    allowed = [t for t in puts if t["ticker"] not in BANNED]
    days = sorted({t["day"] for t in puts})
    print(f"{len(puts)} puts over {len(days)} days ({days[0]}..{days[-1]}); "
          f"{len(banned)} on BANNED names, {len(allowed)} on allowed names\n")

    print("=== BASELINE put books ===")
    line("ALLOWED-name puts (prod takes)", allowed)
    line("BANNED-name puts (prod SKIPS all)", banned)
    print("  ^ if banned total is negative/marginal, the blanket ban is justified on average.\n")

    print("=== CONDITIONAL relaxation: take BANNED puts only on a sustained down-day ===")
    print("  gate = name down > X% from open AT ENTRY  AND (optionally) SPY bear mode")
    for bear_only in (False, True):
        tag = "name-down + SPY-bear" if bear_only else "name-down only"
        print(f"\n  -- {tag} --")
        print(f"     {'thr (name down)':<18}{'take':>6}{'skip':>6}{'TAKEN P&L':>12}{'PF':>7}{'WR':>6}")
        for thr in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            sub = [r for r in banned if r.get("put_und_move", 0) < -thr
                   and (r.get("is_bear_mode", False) if bear_only else True)]
            print(f"     <-{thr:<15}%{len(sub):>6}{len(banned) - len(sub):>6}"
                  f"${pnl(sub):>+10,.0f}{pf(sub):>7.2f}{wr(sub):>5.0f}%")

    # Per-banned-ticker: does any single name carry it? (guard against one lucky name)
    print("\n=== PER-BANNED-NAME (all banned puts, and the name-down<-1.5% + bear subset) ===")
    print(f"  {'ticker':<7}{'allN':>5}{'all$':>9}{'condN':>7}{'cond$':>9}{'condWR':>8}")
    byT = defaultdict(list)
    for t in banned:
        byT[t["ticker"]].append(t)
    for tk in sorted(byT, key=lambda k: -len(byT[k])):
        recs = byT[tk]
        cond = [r for r in recs if r.get("put_und_move", 0) < -1.5 and r.get("is_bear_mode", False)]
        print(f"  {tk:<7}{len(recs):>5}${pnl(recs):>+7,.0f}{len(cond):>7}${pnl(cond):>+7,.0f}{wr(cond):>7.0f}%")

    # Sanity: how do ALLOWED-name puts do under the SAME sustained-down gate? (does the condition
    # itself carry the edge, independent of the banned names?)
    print("\n=== CONTROL: ALLOWED-name puts under the same name-down<-1.5% + bear gate ===")
    ctrl = [r for r in allowed if r.get("put_und_move", 0) < -1.5 and r.get("is_bear_mode", False)]
    line("allowed puts, sustained-down subset", ctrl)
    line("allowed puts, all", allowed)


if __name__ == "__main__":
    main()
