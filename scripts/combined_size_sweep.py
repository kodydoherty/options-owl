"""Combined FULL-STACK portfolio-size sweep (2026-07-15) — ML + UW flow, calls+puts, INTEGER sizing.

Answers: on the book kody ACTUALLY trades (ML pattern + UW flow, both sides), what starting balance is
the sweet spot — and does the pricier flow contract ($9 cap) price a small account out (the affordability
caveat the ML-only sweep couldn't test, since ML premiums avg only ~$2)?

Unlike scripts/backtest_full_stack.py (continuous-dollar, can't skip a trade), this sizes EACH trade with
the REAL prod score_to_contracts → integer contracts, and SKIPS a trade when one contract > 15% of balance
(the "priced out" mechanic). Sizing is prod-faithful:
  ML   → ml_confidence=pattern_conf, conf_linear 0.4-1.8, conviction_mult 1.0
  FLOW → ml_confidence=None (→0.85 flat), conviction_mult=flow conv_mult
Both: is_put→0.5 put budget, 15% position cap, 75% risk / 8 concurrent, per-5min-bucket concurrency cap.

⚠️ Model caveats (stated): serialized-immediate-pnl compounding (like full_stack — no capital tied in open
positions), so $ is OPTIMISTIC and compounds faster than reality. That means the affordability constraint is
if anything UNDERSTATED here (the account grows past the price-out threshold quicker than live). Read priced-out
counts as a FLOOR and the cross-size RANKING as the signal. Window = the flow data window (~2026-03-14 on).

Inputs (generate first):
  ML dump : backtest_gold_standard.py --start 2026-03-14 --end 2026-07-01 --pattern-threshold 0.62
            --no-entry-filter --puts --dump-trades <ml.json>   (kody-faithful env: CONF_LINEAR_CURRENT=1 etc.)
  Flow    : flow_gold_standard_report.collect(...) cached to <flow.json>

Usage: python scripts/combined_size_sweep.py --ml-dump ml.json --flow-cache flow.json --sizes 5000,10000,15000,20000
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from options_owl.risk.vinny_strategy import score_to_contracts  # noqa: E402

MAX_CONCURRENT = 8
MAX_POSITION_PCT = 15.0
MAX_RISK_PCT = 75.0
SCORE = 90  # both books already passed entry gates; score only gates the floor, doesn't drive magnitude


def _is_put(side: str) -> bool:
    return str(side).lower() in ("put", "bearish", "short")


def load_ml(path):
    out = []
    for t in json.loads(Path(path).read_text()):
        prem = t.get("effective_entry") or t.get("entry") or 0.0
        if prem <= 0:
            continue
        out.append({
            "book": "ML", "date": t["day"], "mi": int(t.get("signal_minute", t.get("minute", 0)) or 0),
            "ticker": t["ticker"], "side": t.get("direction", "call"),
            "cost": prem * 100.0, "ret_pct": float(t.get("ret_pct", 0.0)),
            "conf": t.get("pattern_conf"), "conv": 1.0,
        })
    return out


def load_flow(path, window_dates):
    out = []
    for t in json.loads(Path(path).read_text()):
        if window_dates and t["date"] not in window_dates:
            continue
        prem = t.get("entry_prem") or 0.0
        if prem <= 0:
            continue
        out.append({
            "book": "FLOW", "date": t["date"], "mi": int(t.get("mi", 200) or 200),
            "ticker": t["ticker"], "side": t["side"],
            "cost": prem * 100.0, "ret_pct": float(t.get("ret_pct", 0.0)),
            "conf": None, "conv": float(t.get("conv_mult", 1.0)),
        })
    return out


def contracts_for(tr, balance):
    """Prod-faithful integer sizing. Returns (contracts, afford_max) — afford_max=0 means priced out."""
    is_put = _is_put(tr["side"])
    afford_max = int((balance * MAX_POSITION_PCT / 100.0) / tr["cost"]) if tr["cost"] > 0 else 0
    n = score_to_contracts(
        SCORE, cost_per_contract=tr["cost"], balance=balance,
        max_position_pct=MAX_POSITION_PCT, max_concurrent=MAX_CONCURRENT,
        max_portfolio_risk_pct=MAX_RISK_PCT,
        ml_confidence=tr["conf"], conviction_mult=tr["conv"],
        is_put=is_put, put_budget_multiplier=0.50,
        conf_linear=(tr["conf"] is not None), conf_budget_min=0.4, conf_budget_max=1.8,
    )
    return n, afford_max


def simulate(trades, start):
    """Serialized compounding with integer contracts + affordability telemetry."""
    trades = sorted(trades, key=lambda x: (x["date"], x["mi"]))
    bal = float(start)
    peak = bal
    max_dd = 0.0
    wins = losses = 0
    gw = gl = 0.0
    per_book = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    stats = {"taken": 0, "skip_unafford": 0, "cap_bound": 0,
             "contracts_sum": 0, "taken_prem_sum": 0.0, "unafford_prem_sum": 0.0}
    bucket = defaultdict(int)
    for t in trades:
        key = (t["date"], (t["mi"] // 5) * 5)
        if bucket[key] >= MAX_CONCURRENT:
            continue
        n, afford_max = contracts_for(t, bal)
        prem = t["cost"] / 100.0
        if afford_max == 0:
            stats["skip_unafford"] += 1
            stats["unafford_prem_sum"] += prem
            continue
        if n <= 0:
            continue
        bucket[key] += 1
        stats["taken"] += 1
        stats["contracts_sum"] += n
        stats["taken_prem_sum"] += prem
        if n >= afford_max:
            stats["cap_bound"] += 1
        pnl = n * t["cost"] * t["ret_pct"] / 100.0
        bal += pnl
        b = per_book[t["book"]]
        b["n"] += 1
        b["pnl"] += pnl
        if pnl > 0:
            wins += 1
            gw += pnl
        elif pnl < 0:
            losses += 1
            gl += -pnl
        peak = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak * 100.0)
    nt = wins + losses
    return {
        "final": bal, "return_pct": (bal - start) / start * 100.0,
        "trades": nt, "win_rate": (wins / nt * 100.0) if nt else 0.0,
        "pf": (gw / gl) if gl > 0 else float("inf"), "max_dd": max_dd,
        "per_book": dict(per_book), "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-dump", required=True)
    ap.add_argument("--flow-cache", required=True)
    ap.add_argument("--sizes", default="5000,10000,15000,20000")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    ml = load_ml(args.ml_dump)
    # window = the overlapping DATE RANGE (not per-day intersection — a day where only one book
    # traded still belongs on the shared account). flow is the limiter (back to ~2026-03-14).
    flow_all = json.loads(Path(args.flow_cache).read_text())
    flow_dates = {t["date"] for t in flow_all}
    ml_dates = {t["date"] for t in ml}
    lo = max(min(flow_dates), min(ml_dates)) if ml_dates else min(flow_dates)
    hi = min(max(flow_dates), max(ml_dates)) if ml_dates else max(flow_dates)
    window = {d for d in (flow_dates | ml_dates) if lo <= d <= hi}
    ml = [t for t in ml if t["date"] in window]
    flow = load_flow(args.flow_cache, window)
    combined = ml + flow
    days = sorted({t["date"] for t in combined})

    print("=" * 96)
    print("COMBINED FULL-STACK PORTFOLIO-SIZE SWEEP — ML + UW flow, calls+puts, integer sizing")
    print(f"Window: {days[0]} → {days[-1]}  ({len(days)} trading days)  |  "
          f"ML {len(ml)} + FLOW {len(flow)} = {len(combined)} raw signals")
    print("=" * 96)

    rows = []
    for size in sizes:
        r = simulate(combined, size)
        rows.append((size, r))

    print(f"\n{'Start':>8} {'FinalEq':>12} {'Return%':>9} {'PF':>6} {'WR%':>6} {'Taken':>7} "
          f"{'MaxDD%':>8} {'AvgContr':>9} {'FlowP&L':>11} {'MLP&L':>11}")
    print("-" * 96)
    for size, r in rows:
        st = r["stats"]
        avg_c = st["contracts_sum"] / max(1, st["taken"])
        fpnl = r["per_book"].get("FLOW", {}).get("pnl", 0.0)
        mpnl = r["per_book"].get("ML", {}).get("pnl", 0.0)
        print(f"${size:>7,} ${r['final']:>11,.0f} {r['return_pct']:>8.1f}% {r['pf']:>6.2f} "
              f"{r['win_rate']:>6.1f} {st['taken']:>7} {r['max_dd']:>7.1f}% "
              f"{avg_c:>9.2f} ${fpnl:>10,.0f} ${mpnl:>10,.0f}")

    print("\n" + "=" * 96)
    print("AFFORDABILITY — does the pricier FLOW contract price a small account out? (the ML-only caveat)")
    print("=" * 96)
    print(f"{'Start':>8} {'Taken':>7} {'PricedOut':>10} {'Afford%':>8} {'CapClipped':>11} "
          f"{'AvgPremTaken':>13} {'AvgPremPricedOut':>17}")
    print("-" * 96)
    for size, r in rows:
        st = r["stats"]
        taken, skip = st["taken"], st["skip_unafford"]
        elig = taken + skip
        afford = 100.0 * taken / max(1, elig)
        apt = st["taken_prem_sum"] / max(1, taken)
        aps = st["unafford_prem_sum"] / max(1, skip)
        print(f"${size:>7,} {taken:>7} {skip:>10} {afford:>7.1f}% {st['cap_bound']:>11} "
              f"${apt:>11.2f} ${aps:>15.2f}")
    print("\nPricedOut = passed gates but 1 contract > 15% of balance → skipped (mostly pricey flow $6-9).")
    print("NOTE: serialized-optimistic compounding grows the account fast, so PricedOut is a FLOOR (live would")
    print("price out more early on). Read cross-size RANKING as the signal, absolute $ as optimistic.")


if __name__ == "__main__":
    main()
