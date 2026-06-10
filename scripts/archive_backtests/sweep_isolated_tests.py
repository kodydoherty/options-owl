"""Isolated Variable Sweep — tests ONE parameter change at a time from gold standard baseline.

Based on specs/volume-profitability-research.html test plan.
Each test overrides a single constant in backtest_gold_standard.py and runs the full 60-day backtest.
Results are compared to the CTRL baseline.

Usage:
    python scripts/sweep_isolated_tests.py                  # run all Tier 1 tests
    python scripts/sweep_isolated_tests.py --test E1        # run single test
    python scripts/sweep_isolated_tests.py --test E1,E3,X1  # run specific tests
    python scripts/sweep_isolated_tests.py --days 90        # override period
    python scripts/sweep_isolated_tests.py --tier2          # include Tier 2 code-change tests
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import the gold standard backtest module
import scripts.backtest_gold_standard as gs

THETADATA_DB = str(PROJECT_DIR / "journal" / "thetadata_options.db")
RESULTS_DIR = PROJECT_DIR / "journal" / "sweep_results"

# ── Baseline values (snapshot of gold standard defaults) ──────────────────

BASELINE = {
    # Entry pipeline
    "SCAN_START_MIN": 5,
    "SCAN_END_MIN": 90,
    "SCAN_INTERVAL": 1,
    "PATTERN_THRESHOLD": 0.85,
    "ENTRY_THRESHOLD": 0.70,
    "REGIME_THRESHOLD": 0.19,
    "use_entry_filter": True,
    "use_regime": True,
    "PREMIUM_CAP": 6.0,
    "MIN_PREMIUM_FLOOR": 0.20,
    "SPREAD_GATE_PCT": 15.0,
    "MAX_CONCURRENT": 4,
    "MAX_SAME_DIRECTION": 2,
    "MAX_INDEX_CONCURRENT": 1,
    "MIN_ENTRY_SPACING_MIN": 5,
    "MAX_POSITION_DOLLARS": 5_000,
    "MAX_CONTRACTS": 200,
    "MAX_POSITION_PCT": 0.15,
    "MAX_RISK_PCT": 0.75,
    "GFV_BUFFER_PCT": 15.0,
    "DAILY_LOSS_CB_PCT": 15.0,
    "BAD_DAY_THRESHOLD": 0.90,
    "LOSS_COUNT_THRESHOLD": 2,
    # Production pipeline gates (all disabled by default — ablation showed zero/negative value)
    "ENABLE_ANTI_CHASE": False,
    "ENABLE_MOMENTUM_CONFIRM": False,
    "ENABLE_DIRECTIONAL_REGIME": False,
    "ENABLE_CONSECUTIVE_LOSER": False,
    "ENABLE_CORRELATION_CAP": False,
    "ANTI_CHASE_MAX_MOVE_PCT": 0.3,
    "CONSECUTIVE_LOSER_MAX": 2,
    "CONSECUTIVE_LOSER_PAUSE_MIN": 15,
    "CORRELATION_CAP_MAX_PER_GROUP": 3,
    # Exit / V6 settings
    "SCALP_TARGET_PCT": 25.0,
    "SCALP_RUNNER_CONFIRM_PCT": 40.0,
    "V6_SCALEOUT_GAIN_PCT": 20.0,
    "V6_BREAKEVEN_TRIGGER_PCT": 20.0,
    "ENABLE_V6_2PM_TIGHTEN": True,
    "ENABLE_V6_SCALEOUT": True,
    "profit_target_general_pct": 0,  # disabled for CALLs in baseline
}

# ── Test Definitions ──────────────────────────────────────────────────────
# Each test: {override_key: override_value, ...}
# Tests override exactly ONE concept (may touch 1-2 related constants).

TIER1_TESTS = {
    # ── Entry Pipeline ──
    "E1":  ("Scan window → 2:00 PM", {"SCAN_END_MIN": 270}),
    "E2":  ("Scan window → 3:45 PM (full day)", {"SCAN_END_MIN": 375}),
    "E3":  ("Pattern threshold → 0.80", {"PATTERN_THRESHOLD": 0.80}),
    "E4":  ("Pattern threshold → 0.75", {"PATTERN_THRESHOLD": 0.75}),
    "E5":  ("Entry timing threshold → 0.60", {"ENTRY_THRESHOLD": 0.60}),
    "E6":  ("Remove entry timing filter", {"use_entry_filter": False}),
    "E7":  ("Premium cap → $10", {"PREMIUM_CAP": 10.0}),
    "E8":  ("Premium cap removed ($999)", {"PREMIUM_CAP": 999.0}),
    "E9":  ("Premium floor → $0.05", {"MIN_PREMIUM_FLOOR": 0.05}),
    "E10": ("Spread gate → 25%", {"SPREAD_GATE_PCT": 25.0}),
    "E11": ("Max concurrent → 6", {"MAX_CONCURRENT": 6}),
    "E12": ("Max same direction → 3", {"MAX_SAME_DIRECTION": 3}),
    "E13": ("Entry spacing → 2 min", {"MIN_ENTRY_SPACING_MIN": 2}),
    "E14": ("Remove regime filter", {"use_regime": False}),

    # ── Exit Engine ──
    "X1":  ("Scalp target → 35%", {"SCALP_TARGET_PCT": 35.0}),
    "X2":  ("Scalp target → 50%", {"SCALP_TARGET_PCT": 50.0}),
    "X3":  ("Scaleout trigger → 35%", {"V6_SCALEOUT_GAIN_PCT": 35.0}),
    "X4":  ("Breakeven ratchet → 30%", {"V6_BREAKEVEN_TRIGGER_PCT": 30.0}),
    "X5":  ("Disable 2PM trail tighten", {"ENABLE_V6_2PM_TIGHTEN": False}),
    "X6":  ("Disable scaleout", {"ENABLE_V6_SCALEOUT": False}),
    "X7":  ("Profit target 50% (all CALLs)", {"profit_target_general_pct": 50.0}),
    "X8":  ("Profit target 100% (all CALLs)", {"profit_target_general_pct": 100.0}),

    # ── Premium Bands ──
    "P1":  ("Pennies only ($0.05-$0.50)", {"MIN_PREMIUM_FLOOR": 0.05, "PREMIUM_CAP": 0.50}),
    "P2":  ("Sweet spot ($0.50-$1.50)", {"MIN_PREMIUM_FLOOR": 0.50, "PREMIUM_CAP": 1.50}),
    "P3":  ("Mid-range ($1.50-$3.00)", {"MIN_PREMIUM_FLOOR": 1.50, "PREMIUM_CAP": 3.00}),
    "P4":  ("Expensive+quickTP ($3-$10, TP40%)", {"MIN_PREMIUM_FLOOR": 3.00, "PREMIUM_CAP": 10.00,
                                                    "profit_target_general_pct": 40.0}),

    # ── Time Windows ──
    "T1":  ("Tighter morning (9:35-10:30)", {"SCAN_END_MIN": 60}),
    "T2":  ("Power hour only (2:30-3:45)", {"SCAN_START_MIN": 300, "SCAN_END_MIN": 375}),
}

# ── Combo Tests: combine the winners from Tier 1 sweep ─────────────────

COMBO_TESTS = {
    # Volume combos — stacking trade frequency levers
    "C1":  ("Pattern 0.80 + 2min spacing",
            {"PATTERN_THRESHOLD": 0.80, "MIN_ENTRY_SPACING_MIN": 2}),
    "C2":  ("Pattern 0.80 + no regime + 2min spacing",
            {"PATTERN_THRESHOLD": 0.80, "use_regime": False, "MIN_ENTRY_SPACING_MIN": 2}),
    "C3":  ("Pattern 0.80 + no regime + same-dir 3",
            {"PATTERN_THRESHOLD": 0.80, "use_regime": False, "MAX_SAME_DIRECTION": 3}),
    "C4":  ("Pattern 0.80 + no regime + 2min + same-dir 3",
            {"PATTERN_THRESHOLD": 0.80, "use_regime": False,
             "MIN_ENTRY_SPACING_MIN": 2, "MAX_SAME_DIRECTION": 3}),

    # Volume + quality exits — more trades but with profit targets
    "C5":  ("Pattern 0.80 + 2min + scalp 35%",
            {"PATTERN_THRESHOLD": 0.80, "MIN_ENTRY_SPACING_MIN": 2,
             "SCALP_TARGET_PCT": 35.0}),
    "C6":  ("Pattern 0.80 + no regime + TP 50%",
            {"PATTERN_THRESHOLD": 0.80, "use_regime": False,
             "profit_target_general_pct": 50.0}),
    "C7":  ("Pattern 0.80 + no regime + 2min + TP 50%",
            {"PATTERN_THRESHOLD": 0.80, "use_regime": False,
             "MIN_ENTRY_SPACING_MIN": 2, "profit_target_general_pct": 50.0}),

    # Full volume stack (all volume levers + best exit)
    "C8":  ("FULL STACK: p0.80 + noReg + 2min + sd3 + scalp35",
            {"PATTERN_THRESHOLD": 0.80, "use_regime": False,
             "MIN_ENTRY_SPACING_MIN": 2, "MAX_SAME_DIRECTION": 3,
             "SCALP_TARGET_PCT": 35.0}),
    "C9":  ("FULL STACK: p0.80 + noReg + 2min + sd3 + TP50",
            {"PATTERN_THRESHOLD": 0.80, "use_regime": False,
             "MIN_ENTRY_SPACING_MIN": 2, "MAX_SAME_DIRECTION": 3,
             "profit_target_general_pct": 50.0}),

    # E4 was best P&L — test it with safety rails
    "C10": ("Pattern 0.75 + TP 50% (cap runners)",
            {"PATTERN_THRESHOLD": 0.75, "profit_target_general_pct": 50.0}),
    "C11": ("Pattern 0.75 + TP 50% + 2min + no regime",
            {"PATTERN_THRESHOLD": 0.75, "profit_target_general_pct": 50.0,
             "MIN_ENTRY_SPACING_MIN": 2, "use_regime": False}),

    # Sizing + volume — E11 was best PF, combine with volume
    "C12": ("Max concurrent 6 + pattern 0.80 + 2min",
            {"MAX_CONCURRENT": 6, "PATTERN_THRESHOLD": 0.80,
             "MIN_ENTRY_SPACING_MIN": 2}),
    "C13": ("Max concurrent 6 + pattern 0.80 + no regime",
            {"MAX_CONCURRENT": 6, "PATTERN_THRESHOLD": 0.80,
             "use_regime": False}),

    # Premium band + volume combos
    "C14": ("Sweet spot $0.50-$1.50 + pattern 0.80 + no regime",
            {"MIN_PREMIUM_FLOOR": 0.50, "PREMIUM_CAP": 1.50,
             "PATTERN_THRESHOLD": 0.80, "use_regime": False}),

    # Gate relaxation tests — what if we loosen the filters?
    "G1":  ("No entry filter + no regime (both gates off)",
            {"use_entry_filter": False, "use_regime": False}),
    "G2":  ("No entry filter + no regime + 2min spacing",
            {"use_entry_filter": False, "use_regime": False,
             "MIN_ENTRY_SPACING_MIN": 2}),
    "G3":  ("No entry filter + no regime + pattern 0.80",
            {"use_entry_filter": False, "use_regime": False,
             "PATTERN_THRESHOLD": 0.80}),
    "G4":  ("Spread gate 50% (very loose)",
            {"SPREAD_GATE_PCT": 50.0}),
    "G5":  ("Spread gate removed + premium floor $0.05",
            {"SPREAD_GATE_PCT": 999.0, "MIN_PREMIUM_FLOOR": 0.05}),
    "G6":  ("All gates loose: p0.75 noEntry noRegime sd3 2min",
            {"PATTERN_THRESHOLD": 0.75, "use_entry_filter": False,
             "use_regime": False, "MAX_SAME_DIRECTION": 3,
             "MIN_ENTRY_SPACING_MIN": 2}),
    "G7":  ("Same-dir 4 (essentially uncapped for CALL-only)",
            {"MAX_SAME_DIRECTION": 4}),
    "G8":  ("Max concurrent 8 + same-dir 4",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4}),

    # Production gate tests — enable individually to test value
    "G9":  ("Enable anti-chase (0.3%)",
            {"ENABLE_ANTI_CHASE": True, "ANTI_CHASE_MAX_MOVE_PCT": 0.3}),
    "G10": ("Enable anti-chase (0.5% looser)",
            {"ENABLE_ANTI_CHASE": True, "ANTI_CHASE_MAX_MOVE_PCT": 0.5}),
    "G11": ("Enable directional regime",
            {"ENABLE_DIRECTIONAL_REGIME": True}),
    "G12": ("Enable momentum confirm",
            {"ENABLE_MOMENTUM_CONFIRM": True}),
    "G13": ("Enable all prod gates",
            {"ENABLE_ANTI_CHASE": True, "ENABLE_MOMENTUM_CONFIRM": True,
             "ENABLE_DIRECTIONAL_REGIME": True, "ENABLE_CONSECUTIVE_LOSER": True,
             "ENABLE_CORRELATION_CAP": True}),
    "G14": ("Enable anti-chase + dir regime (top 2 blockers)",
            {"ENABLE_ANTI_CHASE": True, "ENABLE_DIRECTIONAL_REGIME": True}),

    # ── G8 Enhancement Combos (deployed config: MAX_CONCURRENT=8) ──
    "H1":  ("G8 + anti-chase 0.5%",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "ENABLE_ANTI_CHASE": True, "ANTI_CHASE_MAX_MOVE_PCT": 0.5}),
    "H2":  ("G8 + pattern 0.80",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "PATTERN_THRESHOLD": 0.80}),
    "H3":  ("G8 + pattern 0.80 + anti-chase 0.5%",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "PATTERN_THRESHOLD": 0.80,
             "ENABLE_ANTI_CHASE": True, "ANTI_CHASE_MAX_MOVE_PCT": 0.5}),
    "H4":  ("G8 + pattern 0.80 + scalp 35%",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "PATTERN_THRESHOLD": 0.80, "SCALP_TARGET_PCT": 35.0}),
    "H5":  ("G8 + pattern 0.80 + no regime",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "PATTERN_THRESHOLD": 0.80, "use_regime": False}),
    "H6":  ("G8 + pattern 0.80 + no regime + anti-chase 0.5%",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "PATTERN_THRESHOLD": 0.80, "use_regime": False,
             "ENABLE_ANTI_CHASE": True, "ANTI_CHASE_MAX_MOVE_PCT": 0.5}),
    "H7":  ("G8 + scan to 2PM (more hours)",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "SCAN_END_MIN": 270}),
    "H8":  ("G8 + scan to 2PM + anti-chase 0.5%",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "SCAN_END_MIN": 270,
             "ENABLE_ANTI_CHASE": True, "ANTI_CHASE_MAX_MOVE_PCT": 0.5}),
    "H9":  ("G8 + 2min spacing + scalp 35%",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "MIN_ENTRY_SPACING_MIN": 2, "SCALP_TARGET_PCT": 35.0}),
    "H10": ("G8 + no regime + 2min + scalp 35%",
            {"MAX_CONCURRENT": 8, "MAX_SAME_DIRECTION": 4,
             "use_regime": False, "MIN_ENTRY_SPACING_MIN": 2,
             "SCALP_TARGET_PCT": 35.0}),
}


def apply_overrides(overrides: dict):
    """Monkey-patch module-level constants in backtest_gold_standard."""

    # Entry pipeline constants
    entry_map = {
        "SCAN_START_MIN": "SCAN_START_MIN",
        "SCAN_END_MIN": "SCAN_END_MIN",
        "SCAN_INTERVAL": "SCAN_INTERVAL",
        "PREMIUM_CAP": "PREMIUM_CAP",
        "MIN_PREMIUM_FLOOR": "MIN_PREMIUM_FLOOR",
        "SPREAD_GATE_PCT": "SPREAD_GATE_PCT",
        "MAX_CONCURRENT": "MAX_CONCURRENT",
        "MAX_SAME_DIRECTION": "MAX_SAME_DIRECTION",
        "MAX_INDEX_CONCURRENT": "MAX_INDEX_CONCURRENT",
        "MIN_ENTRY_SPACING_MIN": "MIN_ENTRY_SPACING_MIN",
        "MAX_POSITION_DOLLARS": "MAX_POSITION_DOLLARS",
        "MAX_CONTRACTS": "MAX_CONTRACTS",
        "MAX_POSITION_PCT": "MAX_POSITION_PCT",
        "MAX_RISK_PCT": "MAX_RISK_PCT",
        "GFV_BUFFER_PCT": "GFV_BUFFER_PCT",
        "DAILY_LOSS_CB_PCT": "DAILY_LOSS_CB_PCT",
        "BAD_DAY_THRESHOLD": "BAD_DAY_THRESHOLD",
        "LOSS_COUNT_THRESHOLD": "LOSS_COUNT_THRESHOLD",
        "ENABLE_ANTI_CHASE": "ENABLE_ANTI_CHASE",
        "ENABLE_MOMENTUM_CONFIRM": "ENABLE_MOMENTUM_CONFIRM",
        "ENABLE_DIRECTIONAL_REGIME": "ENABLE_DIRECTIONAL_REGIME",
        "ENABLE_CONSECUTIVE_LOSER": "ENABLE_CONSECUTIVE_LOSER",
        "ENABLE_CORRELATION_CAP": "ENABLE_CORRELATION_CAP",
        "ANTI_CHASE_MAX_MOVE_PCT": "ANTI_CHASE_MAX_MOVE_PCT",
        "CONSECUTIVE_LOSER_MAX": "CONSECUTIVE_LOSER_MAX",
        "CONSECUTIVE_LOSER_PAUSE_MIN": "CONSECUTIVE_LOSER_PAUSE_MIN",
        "CORRELATION_CAP_MAX_PER_GROUP": "CORRELATION_CAP_MAX_PER_GROUP",
    }

    for key, attr in entry_map.items():
        if key in overrides:
            setattr(gs, attr, overrides[key])
        elif key in BASELINE:
            setattr(gs, attr, BASELINE[key])

    # V6 settings (modify the _V6_SETTINGS namespace)
    v6_map = {
        "SCALP_TARGET_PCT": "SCALP_TARGET_PCT",
        "SCALP_RUNNER_CONFIRM_PCT": "SCALP_RUNNER_CONFIRM_PCT",
        "V6_SCALEOUT_GAIN_PCT": "V6_SCALEOUT_GAIN_PCT",
        "V6_BREAKEVEN_TRIGGER_PCT": "V6_BREAKEVEN_TRIGGER_PCT",
        "ENABLE_V6_2PM_TIGHTEN": "ENABLE_V6_2PM_TIGHTEN",
        "ENABLE_V6_SCALEOUT": "ENABLE_V6_SCALEOUT",
    }

    for key, attr in v6_map.items():
        if key in overrides:
            setattr(gs._V6_SETTINGS, attr, overrides[key])
        elif key in BASELINE:
            setattr(gs._V6_SETTINGS, attr, BASELINE[key])

    # Premium cap in V6 settings too
    if "PREMIUM_CAP" in overrides:
        gs._V6_SETTINGS.V6_PREMIUM_CAP = overrides["PREMIUM_CAP"]
        if overrides["PREMIUM_CAP"] >= 999:
            gs._V6_SETTINGS.ENABLE_V6_PREMIUM_CAP = False
        else:
            gs._V6_SETTINGS.ENABLE_V6_PREMIUM_CAP = True
            gs._V6_SETTINGS.V6_PREMIUM_CAP_MID = overrides["PREMIUM_CAP"] + 1.0
            gs._V6_SETTINGS.V6_PREMIUM_CAP_HIGH = overrides["PREMIUM_CAP"] + 3.0
    else:
        gs._V6_SETTINGS.ENABLE_V6_PREMIUM_CAP = True
        gs._V6_SETTINGS.V6_PREMIUM_CAP = BASELINE["PREMIUM_CAP"]
        gs._V6_SETTINGS.V6_PREMIUM_CAP_MID = 7.0
        gs._V6_SETTINGS.V6_PREMIUM_CAP_HIGH = 9.0

    # Spread gate in V6
    if "SPREAD_GATE_PCT" in overrides:
        gs._V6_SETTINGS.V6_MAX_SPREAD_PCT = overrides["SPREAD_GATE_PCT"]
    else:
        gs._V6_SETTINGS.V6_MAX_SPREAD_PCT = BASELINE["SPREAD_GATE_PCT"]

    # General profit target — monkey-patch get_ticker_config to inject override
    pt = overrides.get("profit_target_general_pct", BASELINE["profit_target_general_pct"])
    if pt > 0:
        from dataclasses import replace as dc_replace
        from options_owl.risk.exit_v5.config import get_ticker_config as _orig_gtc

        def _patched_gtc(ticker, **kwargs):
            cfg = _orig_gtc(ticker, **kwargs)
            return dc_replace(cfg, profit_target_general_pct=pt)

        gs.get_ticker_config = _patched_gtc
    else:
        # Restore original
        from options_owl.risk.exit_v5.config import get_ticker_config as _orig_gtc
        gs.get_ticker_config = _orig_gtc


def reset_to_baseline():
    """Reset all constants to baseline values."""
    apply_overrides({})


def run_single_test(test_id: str, description: str, overrides: dict,
                    models: tuple, tickers: list, start_date: str, end_date: str) -> dict:
    """Run one isolated test and return results."""
    print(f"\n{'=' * 70}")
    print(f"TEST {test_id}: {description}")
    print(f"  Overrides: {overrides}")
    print(f"{'=' * 70}")

    # Apply overrides
    apply_overrides(overrides)

    # Determine model usage from overrides
    use_entry = overrides.get("use_entry_filter", BASELINE["use_entry_filter"])
    use_regime = overrides.get("use_regime", BASELINE["use_regime"])

    pattern_model, pattern_meta, entry_model, entry_features, stop_model, regime_model, signal_model = models

    # Determine thresholds
    pattern_threshold = overrides.get("PATTERN_THRESHOLD", BASELINE["PATTERN_THRESHOLD"])
    entry_threshold = overrides.get("ENTRY_THRESHOLD", BASELINE["ENTRY_THRESHOLD"])
    regime_threshold = overrides.get("REGIME_THRESHOLD", BASELINE["REGIME_THRESHOLD"])

    t0 = time.time()
    result = gs.run_backtest(
        pattern_model=pattern_model,
        pattern_meta=pattern_meta,
        entry_model=entry_model if use_entry else None,
        entry_features=entry_features if use_entry else None,
        pattern_threshold=pattern_threshold,
        entry_threshold=entry_threshold,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        stop_model=stop_model,
        regime_model=regime_model if use_regime else None,
        regime_threshold=regime_threshold if use_regime else 0.0,
        signal_model=signal_model,
    )
    elapsed = time.time() - t0

    # Reset to baseline
    reset_to_baseline()

    n = result["trades"]
    days = result["trading_days"]
    tpd = n / days if days > 0 else 0

    summary = {
        "test_id": test_id,
        "description": description,
        "overrides": json.dumps(overrides),
        "trades": n,
        "trades_per_day": round(tpd, 2),
        "win_rate": result["win_rate"],
        "total_pnl": result["total_pnl"],
        "profit_factor": result["profit_factor"],
        "sharpe": result["sharpe"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "avg_win": result["avg_win"],
        "avg_loss": result["avg_loss"],
        "avg_pnl_per_trade": result["avg_pnl_per_trade"],
        "regime_skipped_days": result["regime_skipped_days"],
        "elapsed_sec": round(elapsed, 1),
    }

    print(f"\n  RESULT: {n} trades ({tpd:.2f}/day), {result['win_rate']:.1f}% WR, "
          f"${result['total_pnl']:+,.0f}, PF={result['profit_factor']:.2f}, "
          f"Sharpe={result['sharpe']:.2f}, MaxDD={result['max_drawdown_pct']:.1f}%"
          f" [{elapsed:.0f}s]")

    return summary


def print_results_table(results: list[dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 120)
    print("ISOLATED VARIABLE SWEEP — RESULTS")
    print("=" * 120)

    header = (f"{'ID':<6} {'Description':<38} {'Trades':>6} {'T/Day':>6} "
              f"{'WR%':>6} {'P&L':>11} {'PF':>7} {'Sharpe':>7} {'MaxDD':>7}")
    print(header)
    print("-" * 120)

    for r in results:
        line = (f"{r['test_id']:<6} {r['description'][:37]:<38} "
                f"{r['trades']:>6} {r['trades_per_day']:>6.2f} "
                f"{r['win_rate']:>5.1f}% ${r['total_pnl']:>+10,.0f} "
                f"{r['profit_factor']:>7.2f} {r['sharpe']:>7.2f} "
                f"{r['max_drawdown_pct']:>6.1f}%")

        # Color coding: green if better than CTRL, red if worse
        if r["test_id"] == "CTRL":
            print(f"\033[1m{line}\033[0m")  # bold for baseline
        elif r["total_pnl"] > results[0]["total_pnl"] and r["win_rate"] >= results[0]["win_rate"] - 10:
            print(f"\033[92m{line}\033[0m")  # green
        elif r["total_pnl"] < 0 or r["win_rate"] < 50:
            print(f"\033[91m{line}\033[0m")  # red
        else:
            print(line)

    print("-" * 120)

    # Summary: best by each metric
    if len(results) > 1:
        non_ctrl = [r for r in results if r["test_id"] != "CTRL"]
        if non_ctrl:
            best_pnl = max(non_ctrl, key=lambda r: r["total_pnl"])
            best_tpd = max(non_ctrl, key=lambda r: r["trades_per_day"])
            best_wr = max(non_ctrl, key=lambda r: r["win_rate"])
            best_pf = max(non_ctrl, key=lambda r: r["profit_factor"])
            best_sharpe = max(non_ctrl, key=lambda r: r["sharpe"])

            print(f"\n  Best P&L:       {best_pnl['test_id']} — ${best_pnl['total_pnl']:+,.0f}")
            print(f"  Best Trades/Day: {best_tpd['test_id']} — {best_tpd['trades_per_day']:.2f}")
            print(f"  Best Win Rate:  {best_wr['test_id']} — {best_wr['win_rate']:.1f}%")
            print(f"  Best PF:        {best_pf['test_id']} — {best_pf['profit_factor']:.2f}")
            print(f"  Best Sharpe:    {best_sharpe['test_id']} — {best_sharpe['sharpe']:.2f}")

            # Volume winners (trades/day > baseline AND profitable)
            ctrl_tpd = results[0]["trades_per_day"]
            volume_winners = [r for r in non_ctrl
                              if r["trades_per_day"] > ctrl_tpd + 0.3 and r["total_pnl"] > 0]
            if volume_winners:
                print(f"\n  VOLUME WINNERS (more trades + still profitable):")
                for r in sorted(volume_winners, key=lambda x: -x["trades_per_day"]):
                    print(f"    {r['test_id']}: {r['trades_per_day']:.2f} T/day, "
                          f"${r['total_pnl']:+,.0f}, {r['win_rate']:.1f}% WR")


def save_results(results: list[dict], output_dir: Path):
    """Save results as CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = output_dir / f"sweep_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV saved: {csv_path}")

    # JSON (full detail)
    json_path = output_dir / f"sweep_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Isolated Variable Sweep")
    parser.add_argument("--days", type=int, default=60, help="Backtest period (trading days)")
    parser.add_argument("--test", type=str, help="Run specific test(s), comma-separated (e.g. E1,E3,X1)")
    parser.add_argument("--tier2", action="store_true", help="Include Tier 2 tests (require code changes)")
    parser.add_argument("--combos", action="store_true", help="Run combo + gate tests instead of isolated")
    parser.add_argument("--all", action="store_true", help="Run ALL tests (isolated + combos)")
    parser.add_argument("--start", type=str, help="Override start date")
    parser.add_argument("--end", type=str, help="Override end date")
    args = parser.parse_args()

    print("=" * 70)
    print("ISOLATED VARIABLE SWEEP")
    print("One change at a time from Gold Standard baseline")
    print("=" * 70)

    # Determine date range
    conn = sqlite3.connect(THETADATA_DB)
    all_dates = [r[0] for r in conn.execute("""
        SELECT DISTINCT substr(timestamp, 1, 10) FROM option_ohlc
        WHERE ticker = 'SPY' ORDER BY 1 DESC
    """).fetchall()]
    conn.close()

    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    elif args.end:
        end_date = args.end
        target_dates = [d for d in all_dates if d <= end_date]
        start_date = target_dates[min(args.days - 1, len(target_dates) - 1)]
    else:
        end_date = all_dates[0]
        start_date = all_dates[min(args.days - 1, len(all_dates) - 1)]

    tickers = [t for t in gs.TICKERS if t not in gs.EXCLUDED_TICKERS]

    print(f"\n  Period: {start_date} to {end_date}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Portfolio: ${gs.PORTFOLIO_START:,}")

    # Load models ONCE (shared across all tests)
    print("\nLoading models (shared across all tests)...")
    all_models = gs.load_models(use_entry_filter=True, use_regime=True)
    print("  All models loaded.\n")

    # Determine which tests to run
    if args.combos:
        tests_to_run = dict(COMBO_TESTS)
    elif args.all:
        tests_to_run = {**TIER1_TESTS, **COMBO_TESTS}
    else:
        tests_to_run = dict(TIER1_TESTS)

    if args.test:
        selected = [t.strip().upper() for t in args.test.split(",")]
        all_available = {**TIER1_TESTS, **COMBO_TESTS}
        tests_to_run = {k: v for k, v in all_available.items() if k in selected}
        if not tests_to_run:
            print(f"ERROR: No matching tests found for: {args.test}")
            print(f"Available: {', '.join(all_available.keys())}")
            sys.exit(1)

    print(f"Running {len(tests_to_run) + 1} tests (CTRL + {len(tests_to_run)} isolated)...")
    total_est = (len(tests_to_run) + 1) * 5  # ~5 min each
    print(f"Estimated time: ~{total_est} minutes ({total_est/60:.1f} hours)\n")

    results = []

    # ── CTRL: baseline ──
    ctrl = run_single_test(
        "CTRL", "Gold Standard Baseline",
        {},  # no overrides
        all_models, tickers, start_date, end_date,
    )
    results.append(ctrl)

    # ── Run each isolated test ──
    for test_id, (description, overrides) in tests_to_run.items():
        try:
            r = run_single_test(
                test_id, description, overrides,
                all_models, tickers, start_date, end_date,
            )
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR in {test_id}: {e}")
            results.append({
                "test_id": test_id,
                "description": description,
                "overrides": json.dumps(overrides),
                "trades": 0, "trades_per_day": 0, "win_rate": 0,
                "total_pnl": 0, "profit_factor": 0, "sharpe": 0,
                "max_drawdown_pct": 0, "avg_win": 0, "avg_loss": 0,
                "avg_pnl_per_trade": 0, "regime_skipped_days": 0,
                "elapsed_sec": 0, "error": str(e),
            })

    # Print comparison table
    print_results_table(results)

    # Save results
    save_results(results, RESULTS_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
