"""Put-improvement sweep (2026-07-20): test the research-backed levers for making PUTs profitable,
puts-only, over a given window. Baseline = prod-faithful put config; each lever changes ONE knob;
COMBINED stacks the ones pointing the right way. Models load once; each config re-runs the sim.

Levers (from the online best-practices research + our own chop-bleed data):
  L1 morning-only : PUT_SCAN_END_MIN 360→120 (theta slower AM; first 90min best follow-through)
  L2 tight-trigger: PUT_DIRECTION_TRIGGER_PCT -0.15→-0.5 (require a FAST/decisive down-move to beat theta)
  L2b tighter     : -0.15→-1.0
  L3 fast-stop    : THETA_MIN_OVERRIDE None(999)→60 (cut dead puts fast instead of bleeding)
  L3b faster      : →30
  L4 higher-delta : DELTA_MIN 0.15→0.40 (ITM tracks the move, less theta drag)
  L4b deeper-ITM  : →0.55

    SIZING_MODE=current CONF_LINEAR_CURRENT=1 CONF_BUDGET_MIN=0.4 CONF_BUDGET_MAX=1.8 \
    python scripts/put_improvement_sweep.py --days 30      # 6 weeks
    ... --days 126   # 6 months
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_gold_standard as gs  # noqa: E402

# Baseline (prod-faithful put) values — restored between runs
BASE = dict(PUT_SCAN_END_MIN=360, PUT_DIRECTION_TRIGGER_PCT=-0.15, THETA_MIN_OVERRIDE=None, DELTA_MIN=0.15)

LEVERS = [
    ("baseline",       {}),
    ("L1 morning-only",  dict(PUT_SCAN_END_MIN=120)),
    ("L2 trig -0.5",     dict(PUT_DIRECTION_TRIGGER_PCT=-0.5)),
    ("L2b trig -1.0",    dict(PUT_DIRECTION_TRIGGER_PCT=-1.0)),
    ("L3 fast-stop 60",  dict(THETA_MIN_OVERRIDE=60)),
    ("L3b fast-stop 30", dict(THETA_MIN_OVERRIDE=30)),
    ("L4 delta>=0.40",   dict(DELTA_MIN=0.40)),
    ("L4b delta>=0.55",  dict(DELTA_MIN=0.55)),
    ("COMBINED",         dict(PUT_SCAN_END_MIN=120, PUT_DIRECTION_TRIGGER_PCT=-0.5,
                              THETA_MIN_OVERRIDE=60, DELTA_MIN=0.40)),
]


def _apply(overrides):
    for k, v in BASE.items():
        setattr(gs, k, v)          # reset to baseline first
    for k, v in overrides.items():
        setattr(gs, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--pattern-threshold", type=float, default=0.62)
    ap.add_argument("--regime-threshold", type=float, default=0.02)
    args = ap.parse_args()

    # prod-faithful global setup + PUTS-ONLY
    gs.ENABLE_ANTI_CHASE = True; gs.ENABLE_MOMENTUM_CONFIRM = True
    gs.ENABLE_DIRECTIONAL_REGIME = True; gs.ENABLE_CONSECUTIVE_LOSER = True
    gs.ENABLE_CORRELATION_CAP = True; gs.ENABLE_PUT_BEARISH_CONFIRM = True
    gs.ENABLE_PUTS = True; gs.PUTS_ONLY = True
    gs.PORTFOLIO_START = 23000

    conn = sqlite3.connect(gs.THETADATA_DB)
    all_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT substr(timestamp,1,10) FROM option_ohlc WHERE ticker='SPY' ORDER BY 1 DESC").fetchall()]
    conn.close()
    end_date = all_dates[0]
    start_date = all_dates[min(args.days - 1, len(all_dates) - 1)]
    tickers = [t for t in gs.TICKERS if t not in gs.EXCLUDED_TICKERS]

    print("=" * 90)
    print(f"PUT-IMPROVEMENT SWEEP (puts-only) | {start_date} → {end_date} ({args.days}d) | {len(tickers)} tickers")
    print("=" * 90)
    print("Loading models once...")
    m = gs.load_models(use_entry_filter=False, use_regime=True)

    rows = []
    for name, ov in LEVERS:
        _apply(ov)
        print(f"\n{'#'*60}\n### {name}  {ov or '(prod-faithful)'}\n{'#'*60}", flush=True)
        r = gs.run_backtest(m[0], m[1], m[2], m[3], args.pattern_threshold, 0.80, tickers,
                            start_date, end_date, m[4], m[5], args.regime_threshold, m[6],
                            m[7], m[8], m[9], m[10], m[11], "none")
        rows.append((name, r))

    print("\n\n" + "=" * 90)
    print(f"PUT-IMPROVEMENT RESULTS — {args.days}d ({start_date}→{end_date})")
    print("=" * 90)
    print(f"{'lever':<20}{'trades':>7}{'WR%':>6}{'P&L':>10}{'PF':>7}{'DD%':>7}")
    print("-" * 90)
    base_pnl = None
    for name, r in rows:
        if name == "baseline":
            base_pnl = r["total_pnl"]
        delta = f"  ({r['total_pnl']-base_pnl:+.0f} vs base)" if base_pnl is not None and name != "baseline" else ""
        print(f"{name:<20}{r['trades']:>7}{r['win_rate']:>6.0f}{r['total_pnl']:>+10.0f}"
              f"{r['profit_factor']:>7.2f}{r['max_drawdown_pct']:>7.1f}{delta}")
    print("=" * 90)


if __name__ == "__main__":
    main()
