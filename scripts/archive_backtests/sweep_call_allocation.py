"""Sweep CALL portfolio allocation parameters using the gold standard backtest.

Tests whether higher per-trade allocation improves CALL-only P&L.
Uses the full ML pipeline (pattern + entry + regime + V5 FSM).

Sweeps: MAX_RISK_PCT, MAX_POSITION_PCT, MAX_CONCURRENT, MAX_POSITION_DOLLARS

Usage:
    python scripts/sweep_call_allocation.py
    python scripts/sweep_call_allocation.py --days 60
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import scripts.backtest_gold_standard as gs
from scripts.sweep_isolated_tests import BASELINE, apply_overrides, reset_to_baseline


def main():
    parser = argparse.ArgumentParser(description="Sweep CALL allocation using gold standard backtest")
    parser.add_argument("--days", type=int, default=90, help="Trading days (default: 90)")
    args = parser.parse_args()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=int(args.days * 1.45))).strftime("%Y-%m-%d")

    print("=" * 110, flush=True)
    print("CALL ALLOCATION SWEEP (Gold Standard Backtest — full ML pipeline)", flush=True)
    print(f"Period: ~{args.days} trading days ending {end_date}", flush=True)
    print(f"Portfolio: ${gs.PORTFOLIO_START:,}", flush=True)
    print("=" * 110, flush=True)

    # Load models once (all enabled for gold standard)
    models = gs.load_models(use_entry_filter=True, use_regime=True)
    pattern_model, pattern_meta, entry_model, entry_features, stop_model, regime_model, signal_model = models

    tickers = [t for t in gs.TICKERS if t not in gs.EXCLUDED_TICKERS]

    # All configs disable the hard dollar cap (set to $999k) so only % matters
    NO_CAP = 999_000

    configs = [
        # (label, overrides)
        # --- Baseline WITH dollar cap (reference) ---
        ("BASELINE (w/ $5k cap)",        {}),

        # --- Baseline WITHOUT dollar cap ---
        ("NoCap 75%/15%/8slot",          {"MAX_POSITION_DOLLARS": NO_CAP}),

        # --- Sweep MAX_RISK_PCT (deployable %) ---
        ("Risk 80%  / 15% / 8slot",     {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.80}),
        ("Risk 85%  / 15% / 8slot",     {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.85}),
        ("Risk 90%  / 15% / 8slot",     {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.90}),
        ("Risk 95%  / 15% / 8slot",     {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.95}),

        # --- Sweep MAX_POSITION_PCT (per-trade cap) ---
        ("Risk 75% / 20% / 8slot",      {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_POSITION_PCT": 0.20}),
        ("Risk 75% / 25% / 8slot",      {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_POSITION_PCT": 0.25}),
        ("Risk 75% / 30% / 8slot",      {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_POSITION_PCT": 0.30}),
        ("Risk 75% / 40% / 8slot",      {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_POSITION_PCT": 0.40}),

        # --- Sweep MAX_CONCURRENT (slot count) ---
        ("Risk 75% / 15% / 4slot",      {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_CONCURRENT": 4, "MAX_SAME_DIRECTION": 4}),
        ("Risk 75% / 15% / 5slot",      {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_CONCURRENT": 5, "MAX_SAME_DIRECTION": 5}),
        ("Risk 75% / 15% / 6slot",      {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_CONCURRENT": 6, "MAX_SAME_DIRECTION": 6}),

        # --- Best % combos (bigger per-trade via %) ---
        ("85% / 20% / 6slot",           {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.85,
                                          "MAX_POSITION_PCT": 0.20, "MAX_CONCURRENT": 6,
                                          "MAX_SAME_DIRECTION": 6}),
        ("90% / 25% / 5slot",           {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.90,
                                          "MAX_POSITION_PCT": 0.25, "MAX_CONCURRENT": 5,
                                          "MAX_SAME_DIRECTION": 5}),
        ("95% / 30% / 4slot",           {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.95,
                                          "MAX_POSITION_PCT": 0.30, "MAX_CONCURRENT": 4,
                                          "MAX_SAME_DIRECTION": 4}),
        ("90% / 20% / 4slot",           {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.90,
                                          "MAX_POSITION_PCT": 0.20, "MAX_CONCURRENT": 4,
                                          "MAX_SAME_DIRECTION": 4}),
        ("85% / 25% / 5slot",           {"MAX_POSITION_DOLLARS": NO_CAP, "MAX_RISK_PCT": 0.85,
                                          "MAX_POSITION_PCT": 0.25, "MAX_CONCURRENT": 5,
                                          "MAX_SAME_DIRECTION": 5}),
    ]

    print(f"\n{'Config':<32} {'Trades':>6} {'WR%':>5} {'P&L':>12} {'PF':>5} "
          f"{'MaxDD':>6} {'AvgWin':>8} {'AvgLoss':>8} {'Time':>5}")
    print("-" * 100)

    for label, overrides in configs:
        reset_to_baseline()
        apply_overrides(overrides)

        t0 = time.time()
        # Suppress per-trade output by redirecting stdout
        import io, contextlib
        f_buf = io.StringIO()
        with contextlib.redirect_stdout(f_buf):
            result = gs.run_backtest(
                pattern_model=pattern_model,
                pattern_meta=pattern_meta,
                entry_model=entry_model,
                entry_features=entry_features,
                pattern_threshold=BASELINE["PATTERN_THRESHOLD"],
                entry_threshold=BASELINE["ENTRY_THRESHOLD"],
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                stop_model=stop_model,
                regime_model=regime_model,
                regime_threshold=BASELINE["REGIME_THRESHOLD"],
                signal_model=signal_model,
            )
        elapsed = time.time() - t0

        trades = result.get("trades", 0)
        wr = result.get("win_rate", 0)
        pnl = result.get("total_pnl", 0)
        pf = result.get("profit_factor", 0)
        mdd = result.get("max_drawdown_pct", 0)
        avg_win = result.get("avg_win", 0)
        avg_loss = result.get("avg_loss", 0)

        print(f"{label:<32} {trades:>6} {wr:>4.0f}% "
              f"${pnl:>10,.0f} {pf:>4.2f} "
              f"{mdd:>5.1f}% ${avg_win:>7,.0f} "
              f"${avg_loss:>7,.0f} {elapsed:>4.0f}s", flush=True)

    reset_to_baseline()
    print("\nDone.")


if __name__ == "__main__":
    main()
