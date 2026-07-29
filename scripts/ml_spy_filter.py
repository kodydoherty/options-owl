"""ML CALL SPY-direction filter test (2026-07-13).

Today's -$1,347 pain was mostly ML CALLs stopping out on a red tape. Flow already blocks counter-trend
calls (SPY down >0.5%) but ML has NO such filter. This tests it: take the prod-faithful ML call book
(gold-standard --dump-trades) and drop calls entered while SPY was down more than a threshold from its
open — the "don't buy calls into a falling market" rule. Uses each trade's REAL sized P&L from the dump.

A filter ships only if it ADDS book P&L and the dropped bucket was net-negative (mostly losers) — not
if it clips as many winners as losers (the refuted ML velocity-gate's failure mode).

Usage: python scripts/ml_spy_filter.py <dump.json> [--since 2026-05-20]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uw_ticker_discovery as D  # noqa: E402


def main():
    dump = sys.argv[1]
    since = "2026-05-20"
    if "--since" in sys.argv:
        since = sys.argv[sys.argv.index("--since") + 1]
    trades = json.load(open(dump))
    spy = D._stock("SPY")

    calls = []
    for t in trades:
        if str(t.get("direction", "")).lower() not in ("call", "bull", "bullish"):
            continue
        day = t["day"]
        if day < since or day not in spy:
            continue
        mb = (int(t["minute"]) // 5) * 5
        s = spy[day]
        if not s:
            continue
        opx = s.get(min(s))
        npx = s.get(mb) or s.get(min(s, key=lambda m: abs(m - mb)))
        if not opx or not npx:
            continue
        t["spy_chg"] = (npx - opx) / opx * 100
        calls.append(t)

    if not calls:
        print("No ML calls with SPY data in window."); return
    days = sorted({t["day"] for t in calls})
    base = sum(t["pnl"] for t in calls)
    print(f"\nML CALL SPY-direction filter — {len(calls)} calls, {len(days)} days "
          f"({days[0]}..{days[-1]})")
    print(f"BASELINE (take all) ${base:+,.0f}\n")

    def wl(rs):
        return f"{sum(1 for r in rs if r['pnl']>0)}W/{sum(1 for r in rs if r['pnl']<=0)}L"

    print(f"  {'drop CALLs when SPY down > X% from open':<40}{'kept':>5}{'skip':>6}{'P&L':>11}{'Δ':>9}"
          f"{'skipped W/L ($)':>20}")
    for thr in (-0.25, -0.5, -0.75, -1.0, -1.5):
        kept = [t for t in calls if t["spy_chg"] >= thr]
        skip = [t for t in calls if t["spy_chg"] < thr]
        kp = sum(t["pnl"] for t in kept)
        sp = sum(t["pnl"] for t in skip)
        print(f"  SPY < {thr:>+4.2f}%                              {len(kept):>5}{len(skip):>6}"
              f"   ${kp:>+8,.0f}{kp-base:>+9,.0f}   {wl(skip)} (${sp:+,.0f})")

    # what fraction of the book is entered into a red SPY at all?
    red = [t for t in calls if t["spy_chg"] < 0]
    print(f"\n  context: {len(red)}/{len(calls)} ML calls entered while SPY was RED "
          f"(net ${sum(t['pnl'] for t in red):+,.0f}); "
          f"green {len(calls)-len(red)} (net ${sum(t['pnl'] for t in calls if t['spy_chg']>=0):+,.0f})")
    print("\nVERDICT: ships only if Δ positive AND skipped bucket net-negative. If red-SPY calls are")
    print("net-positive, ML calls DON'T need a tape filter (unlike flow) — the pain was variance.")


if __name__ == "__main__":
    main()
