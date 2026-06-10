"""Parameter sweep framework for ML E2E backtest.

Systematically varies ONE parameter at a time to find optimal values.
Uses backtest_ml_e2e.run_backtest() as the core engine.

Usage:
    python scripts/sweep_params.py                          # run all sweeps
    python scripts/sweep_params.py --sweep score_threshold  # single sweep
    python scripts/sweep_params.py --sweep scalp_target_pct # single sweep
    python scripts/sweep_params.py --list                   # list available sweeps
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.backtest_ml_e2e import (
    HARVESTER_DB,
    THETADATA_DB,
    BacktestConfig,
    BacktestResult,
    run_backtest,
)


# ── Sweep definitions ────────────────────────────────────────────────────────

@dataclass
class SweepDef:
    name: str
    param: str          # BacktestConfig field name
    values: list        # values to test
    description: str


SWEEPS: dict[str, SweepDef] = {
    "score_threshold": SweepDef(
        name="Score Threshold",
        param="score_threshold",
        values=[0, 30, 40, 45, 50, 55, 60, 65, 70],
        description="Entry scoring threshold (0=ML only, 60=current prod)",
    ),
    "score_threshold_fine": SweepDef(
        name="Score Threshold (Fine)",
        param="score_threshold",
        values=[48, 50, 52, 53, 54, 55, 56, 57, 58, 60],
        description="Fine-grained score threshold around the sweet spot",
    ),
    "scalp_target_pct": SweepDef(
        name="Scalp Target %",
        param="scalp_target_pct",
        values=[15, 20, 25, 30, 35, 40, 50],
        description="Take profit at this % gain (unless confirmed runner)",
    ),
    "scalp_runner_confirm_pct": SweepDef(
        name="Scalp Runner Confirm %",
        param="scalp_runner_confirm_pct",
        values=[25, 30, 35, 40, 50, 60],
        description="Peak % needed to confirm a runner (skip scalp target)",
    ),
    "premium_cap": SweepDef(
        name="Premium Cap ($)",
        param="premium_cap",
        values=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        description="Max entry premium (blocks expensive contracts)",
    ),
    "spread_gate_pct": SweepDef(
        name="Spread Gate %",
        param="spread_gate_pct",
        values=[5, 8, 10, 12, 15, 20, 25, 30],
        description="Max bid-ask spread % to enter",
    ),
    "max_concurrent": SweepDef(
        name="Max Concurrent",
        param="max_concurrent",
        values=[2, 3, 4, 5, 6, 8],
        description="Max simultaneous open positions",
    ),
    "max_position_pct": SweepDef(
        name="Max Position %",
        param="max_position_pct",
        values=[0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
        description="Max % of portfolio per trade",
    ),
    "max_same_direction": SweepDef(
        name="Max Same Direction",
        param="max_same_direction",
        values=[1, 2, 3, 4, 5, 14],
        description="Max positions in same direction (correlation guard)",
    ),
    "daily_loss_cb_pct": SweepDef(
        name="Daily Loss Circuit Breaker %",
        param="daily_loss_cb_pct",
        values=[5, 8, 10, 12, 15, 20, 25, 50],
        description="Stop trading after this % daily loss",
    ),
    "scalp_only": SweepDef(
        name="Scalp-Only Mode (no scalp target = runner mode)",
        param="enable_scalp_target",
        values=[True, False],
        description="Enable scalp target (+25%) vs let runners ride",
    ),
    "morning_cutoff": SweepDef(
        name="Morning Cutoff Hour",
        param="morning_cutoff_hour",
        values=[0, 10, 11, 12, 13, 14, 15],
        description="Stop new entries after this hour (0=no cutoff)",
    ),
    "after_hour_scalp_only": SweepDef(
        name="After-Hour Scalp-Only",
        param="after_hour_scalp_only",
        values=[None, 10, 11, 12, 13],
        description="After this hour, only take scalp exits (no runners)",
    ),
}


# ── Runner ────────────────────────────────────────────────────────────────────


def run_sweep(sweep: SweepDef, conn, baseline_cfg: BacktestConfig) -> list[tuple]:
    """Run a single parameter sweep, returning list of (value, BacktestResult)."""
    results = []

    for val in sweep.values:
        # Handle special cases
        cfg = replace(baseline_cfg, quiet=True)
        if sweep.param == "morning_cutoff_hour":
            if val == 0:
                cfg = replace(cfg, enable_morning_cutoff=False)
            else:
                cfg = replace(cfg, enable_morning_cutoff=True, morning_cutoff_hour=val)
        else:
            cfg = replace(cfg, **{sweep.param: val})

        t0 = time.time()
        res = run_backtest(cfg, conn=conn)
        elapsed = time.time() - t0

        results.append((val, res))
        pnl_str = f"${res.total_pnl:>+10,.0f}"
        print(f"  {sweep.param}={str(val):>8}  →  {pnl_str}  "
              f"{res.trades:>3} trades  {res.win_rate:>5.1f}% WR  "
              f"PF={res.profit_factor:>5.2f}  DD={res.max_drawdown:>5.1f}%  "
              f"Sharpe={res.sharpe:>5.2f}  [{elapsed:.1f}s]")

    return results


def print_sweep_summary(sweep: SweepDef, results: list[tuple]):
    """Print comparison table for a sweep."""
    print(f"\n{'=' * 100}")
    print(f"SWEEP: {sweep.name} — {sweep.description}")
    print(f"{'=' * 100}")
    print(f"{'Value':>10} {'P&L':>12} {'Trades':>7} {'WinRate':>8} {'AvgWin':>9} {'AvgLoss':>9} "
          f"{'MaxDD':>7} {'PF':>6} {'Sharpe':>7} {'Final$':>12}")
    print("-" * 100)

    best_pnl = max(r[1].total_pnl for r in results)
    traded = [r for r in results if r[1].trades > 0]
    best_sharpe = max(r[1].sharpe for r in traded) if traded else 0

    for val, res in results:
        is_best_pnl = "***" if res.total_pnl == best_pnl else "   "
        is_best_sharpe = "S" if res.trades > 0 and traded and res.sharpe == best_sharpe else " "
        print(f"{str(val):>10} ${res.total_pnl:>10,.0f} {res.trades:>7} "
              f"{res.win_rate:>7.1f}% ${res.avg_win:>7,.0f} ${res.avg_loss:>7,.0f} "
              f"{res.max_drawdown:>6.1f}% {res.profit_factor:>5.2f} {res.sharpe:>6.2f} "
              f"${res.final_portfolio:>10,.0f} {is_best_pnl}{is_best_sharpe}")

    # Highlight optimal
    best_by_pnl = max(results, key=lambda r: r[1].total_pnl)
    best_by_sharpe = max(results, key=lambda r: r[1].sharpe if r[1].trades > 0 else -999)
    best_by_pf = max(results, key=lambda r: r[1].profit_factor if r[1].trades > 5 else -999)

    print(f"\n  Best by P&L:    {sweep.param}={best_by_pnl[0]} → ${best_by_pnl[1].total_pnl:,.0f}")
    print(f"  Best by Sharpe: {sweep.param}={best_by_sharpe[0]} → {best_by_sharpe[1].sharpe:.2f}")
    print(f"  Best by PF:     {sweep.param}={best_by_pf[0]} → {best_by_pf[1].profit_factor:.2f}")


def run_per_ticker_sweep(sweep: SweepDef, conn, baseline_cfg: BacktestConfig,
                          tickers: list[str]) -> dict[str, list[tuple]]:
    """Run a sweep for each ticker individually. Returns {ticker: [(val, result), ...]}."""
    all_ticker_results = {}
    for ticker in tickers:
        print(f"\n  [{ticker}]")
        ticker_cfg = replace(baseline_cfg, tickers=[ticker])
        results = run_sweep(sweep, conn, ticker_cfg)
        all_ticker_results[ticker] = results
    return all_ticker_results


def print_per_ticker_summary(sweep: SweepDef, all_ticker_results: dict[str, list[tuple]]):
    """Print per-ticker optimal values."""
    print(f"\n{'=' * 100}")
    print(f"PER-TICKER SWEEP: {sweep.name} — {sweep.description}")
    print(f"{'=' * 100}")
    print(f"{'Ticker':<8} {'Best Value':>12} {'P&L':>12} {'Trades':>7} {'WR':>8} {'PF':>6} {'Sharpe':>7}")
    print("-" * 65)
    for ticker, results in sorted(all_ticker_results.items()):
        # Skip tickers with no trades at any value
        if all(r[1].trades == 0 for r in results):
            print(f"{ticker:<8} {'N/A':>12} {'$0':>12} {'0':>7} {'N/A':>8} {'N/A':>6} {'N/A':>7}")
            continue
        best = max(results, key=lambda r: r[1].total_pnl)
        print(f"{ticker:<8} {str(best[0]):>12} ${best[1].total_pnl:>10,.0f} "
              f"{best[1].trades:>7} {best[1].win_rate:>7.1f}% "
              f"{best[1].profit_factor:>5.2f} {best[1].sharpe:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for ML E2E backtest")
    parser.add_argument("--sweep", type=str, help="Run specific sweep (or 'all')")
    parser.add_argument("--list", action="store_true", help="List available sweeps")
    parser.add_argument("--portfolio", type=int, default=20_000)
    parser.add_argument("--score-threshold", type=int, default=60,
                        help="Baseline score threshold (default 60)")
    parser.add_argument("--ticker", type=str,
                        help="Run sweep for specific ticker(s), comma-separated. "
                             "Use 'all' for per-ticker sweep of every ticker.")
    parser.add_argument("--data-source", choices=["harvester", "thetadata"], default="harvester")
    parser.add_argument("--theta-start", type=str, default="2026-03-27")
    parser.add_argument("--theta-end", type=str, default="2026-05-21")
    args = parser.parse_args()

    if args.list:
        print("Available sweeps:")
        for name, s in SWEEPS.items():
            print(f"  {name:30s} {s.description}")
            print(f"    values: {s.values}")
        return

    sweeps_to_run = []
    if args.sweep and args.sweep != "all":
        names = args.sweep.split(",")
        for n in names:
            n = n.strip()
            if n not in SWEEPS:
                print(f"Unknown sweep: {n}")
                print(f"Available: {', '.join(SWEEPS.keys())}")
                sys.exit(1)
            sweeps_to_run.append(SWEEPS[n])
    else:
        # Default: run the most impactful sweeps
        sweeps_to_run = [
            SWEEPS["score_threshold"],
            SWEEPS["scalp_target_pct"],
            SWEEPS["premium_cap"],
            SWEEPS["max_concurrent"],
            SWEEPS["max_same_direction"],
        ]

    # Determine tickers for per-ticker mode
    per_ticker_mode = False
    ticker_filter = None
    if args.ticker:
        if args.ticker.lower() == "all":
            per_ticker_mode = True
            from scripts.backtest_ml_e2e import TICKERS
            ticker_list = TICKERS
        else:
            ticker_list = [t.strip().upper() for t in args.ticker.split(",")]
            if len(ticker_list) == 1:
                ticker_filter = ticker_list
            else:
                per_ticker_mode = True

    baseline = BacktestConfig(
        portfolio_start=args.portfolio,
        score_threshold=args.score_threshold,
        quiet=True,
        data_source=args.data_source,
        theta_start=args.theta_start,
        theta_end=args.theta_end,
        tickers=ticker_filter,
    )

    db_path = THETADATA_DB if args.data_source == "thetadata" else HARVESTER_DB
    conn = sqlite3.connect(db_path)
    print(f"Portfolio: ${args.portfolio:,} | Data: {args.data_source}")
    if ticker_filter:
        print(f"Ticker filter: {', '.join(ticker_filter)}")
    print(f"Running {len(sweeps_to_run)} sweep(s)...\n")

    all_results = {}
    total_start = time.time()

    for sweep in sweeps_to_run:
        print(f"\n--- {sweep.name} ({len(sweep.values)} values) ---")
        if per_ticker_mode:
            ticker_results = run_per_ticker_sweep(sweep, conn, baseline, ticker_list)
            print_per_ticker_summary(sweep, ticker_results)
            all_results[sweep.name] = (sweep, ticker_results)
        else:
            results = run_sweep(sweep, conn, baseline)
            all_results[sweep.name] = (sweep, results)
            print_sweep_summary(sweep, results)

    conn.close()

    # Final summary
    elapsed = time.time() - total_start
    print(f"\n{'=' * 100}")
    print(f"SWEEP COMPLETE — {len(sweeps_to_run)} params × "
          f"{sum(len(s.values) for s in sweeps_to_run)} runs in {elapsed:.0f}s")
    print(f"{'=' * 100}")

    if not per_ticker_mode:
        print(f"\n{'Parameter':>30} {'Best Value':>12} {'P&L':>12} {'Trades':>7} {'WR':>6} {'Sharpe':>7}")
        print("-" * 80)
        for name, (sweep, results) in all_results.items():
            if isinstance(results, dict):
                continue  # per-ticker results already printed
            best = max(results, key=lambda r: r[1].total_pnl)
            print(f"{name:>30} {str(best[0]):>12} ${best[1].total_pnl:>10,.0f} "
                  f"{best[1].trades:>7} {best[1].win_rate:>5.1f}% {best[1].sharpe:>6.2f}")


if __name__ == "__main__":
    main()
