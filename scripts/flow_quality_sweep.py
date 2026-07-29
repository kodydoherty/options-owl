"""UW flow-QUALITY filter test (2026-07-10) — UW data opportunity #1.

We ingest UW flow sweeps but throw away the alert-quality tags UW attaches. This tests whether
filtering the flow book by those tags improves it:
  - has_multileg: the sweep is one leg of a SPREAD → NOT a directional bet (should underperform)
  - all_opening: a new (opening) position → real conviction vs closing/unwind (should outperform)
  - has_floor: a floor trade
  - alert_rule: which UW rule fired (best/worst rules)

Uses the prod-faithful flow backtest (flow_gold_standard_report.collect → V7 exits via thetadata),
now carrying the tags on each trade. Flat-$ edge (BASE/trade). Held to the standard bar: a filter
must ADD P&L without cutting a chunk of the book that was actually winning.

Usage: python scripts/flow_quality_sweep.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow_gold_standard_report as R  # noqa: E402

BASE = 750.0


def pnl(recs):
    return sum(BASE * r["ret_pct"] / 100.0 for r in recs)


def pf(recs):
    g = sum(BASE * r["ret_pct"] / 100 for r in recs if r["ret_pct"] > 0)
    l = -sum(BASE * r["ret_pct"] / 100 for r in recs if r["ret_pct"] < 0)
    return g / l if l > 0 else float("inf")


def wr(recs):
    return (sum(1 for r in recs if r["ret_pct"] > 0) / len(recs) * 100) if recs else 0.0


def line(lbl, recs, base_total=None):
    d = f"  Δ${pnl(recs) - base_total:>+8,.0f}" if base_total is not None else ""
    print(f"  {lbl:<34}{len(recs):>5} tr   ${pnl(recs):>+9,.0f}   PF {pf(recs):>5.2f}   WR {wr(recs):>3.0f}%{d}")


def main():
    print("Collecting flow trades with quality tags (calls + puts, prod-faithful V7 exits)…")
    trades = R.collect(True, R.PUT_UNIV) + R.collect(False, R.CALL_UNIV)
    if not trades:
        print("No flow trades collected (API/thetadata window empty).")
        return
    days = sorted({t["date"] for t in trades})
    print(f"\n{len(trades)} flow trades over {len(days)} days ({days[0]}..{days[-1]}); "
          f"total ${pnl(trades):+,.0f}\n")

    base = pnl(trades)
    print("=== SPLIT by each quality tag ===")
    for tag, label in (("has_multileg", "multileg (spread leg)"),
                       ("all_opening", "opening position"),
                       ("has_floor", "floor trade")):
        yes = [t for t in trades if t.get(tag)]
        no = [t for t in trades if not t.get(tag)]
        print(f"\n  -- {label} --")
        line(f"{tag}=True", yes)
        line(f"{tag}=False", no)

    print("\n=== FILTER simulations (book P&L if we DROP a bucket) ===")
    print(f"  {'baseline (take all)':<34}{len(trades):>5} tr   ${base:>+9,.0f}")
    line("drop multileg", [t for t in trades if not t.get("has_multileg")], base)
    line("keep opening-only", [t for t in trades if t.get("all_opening")], base)
    line("drop multileg + keep opening", [t for t in trades
         if not t.get("has_multileg") and t.get("all_opening")], base)
    line("drop floor", [t for t in trades if not t.get("has_floor")], base)

    # By-rule: which UW alert rules carry the edge?
    print("\n=== BY alert_rule (n>=5) ===")
    byr = defaultdict(list)
    for t in trades:
        byr[t.get("alert_rule") or "(none)"].append(t)
    print(f"  {'rule':<28}{'n':>4}{'P&L':>10}{'PF':>7}{'WR':>6}")
    for rule in sorted(byr, key=lambda k: -pnl(byr[k])):
        r = byr[rule]
        if len(r) < 5:
            continue
        print(f"  {rule[:27]:<28}{len(r):>4}${pnl(r):>+8,.0f}{pf(r):>7.2f}{wr(r):>5.0f}%")

    print("\nNOTE: flat-$ edge. A tag is a real filter only if dropping it ADDS book P&L (positive Δ)")
    print("and the dropped bucket was genuinely a net loser — not just variance on a thin slice.")


if __name__ == "__main__":
    main()
