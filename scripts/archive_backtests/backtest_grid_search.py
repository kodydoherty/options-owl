"""Grid search over Vinny exit gate parameters to find optimal settings.

Tests combinations of the key parameters that affect early selling:
- setup_failed: minutes + min gain %
- premium stop %
- trailing stop activation + drop %
- profit lock tiers
- time decay hold minutes
- theta bleed settings

Runs on the top tickers by data volume for speed, then validates on all.

Usage:
    python scripts/backtest_grid_search.py
    python scripts/backtest_grid_search.py --quick   # fewer combos, faster
    python scripts/backtest_grid_search.py --ticker SPY
"""

import argparse
import itertools
import os
import sqlite3
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.backtest_vinny import (
    DB_PATH,
    STARTING_BALANCE,
    TradeResult,
    VinnyParams,
    load_option_bars,
    load_trading_days,
    run_backtest,
)


@dataclass
class GridResult:
    """Summary of a single parameter combination."""
    params: dict
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    num_trades: int
    avg_win: float
    avg_loss: float
    avg_duration: float
    reasons: dict
    # Captures how much money was left on the table
    avg_mfe_gap: float  # avg(peak_pnl - actual_pnl) for winners


def run_grid_backtest(tickers, params_dict, base_params=None):
    """Run backtest on multiple tickers with given params, return GridResult."""
    if base_params is None:
        base_params = VinnyParams()

    # Apply overrides
    p = VinnyParams(**{**base_params.__dict__, **params_dict})

    combined_pnl = 0.0
    combined_trades = 0
    combined_wins = 0
    combined_losses = 0
    combined_reasons = {}
    max_dd = 0.0
    all_win_pcts = []
    all_loss_pcts = []
    all_durations = []
    mfe_gaps = []  # how much profit was left on the table

    for ticker in tickers:
        result = run_backtest(ticker, p, verbose=False)
        if result is None:
            continue

        combined_pnl += result["total_pnl"]
        combined_trades += result["num_trades"]
        combined_wins += result["wins"]
        combined_losses += result["losses"]
        if result["max_drawdown"] > max_dd:
            max_dd = result["max_drawdown"]

        for reason, count in result["reasons"].items():
            combined_reasons[reason] = combined_reasons.get(reason, 0) + count

        # Collect per-trade stats
        for t in result["trades"]:
            all_durations.append(t.duration_min)
            peak_pnl = (t.peak_premium - t.entry_premium) / t.entry_premium * 100 if t.entry_premium > 0 else 0
            if t.pnl_pct >= 0:
                all_win_pcts.append(t.pnl_pct)
                mfe_gaps.append(peak_pnl - t.pnl_pct)
            else:
                all_loss_pcts.append(t.pnl_pct)

    if combined_trades == 0:
        return None

    wr = combined_wins / combined_trades * 100
    gross_wins = sum(all_win_pcts) if all_win_pcts else 0
    gross_losses = abs(sum(all_loss_pcts)) if all_loss_pcts else 0
    pf_val = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0

    return GridResult(
        params=params_dict,
        total_pnl=combined_pnl,
        total_pnl_pct=combined_pnl / STARTING_BALANCE * 100,
        win_rate=wr,
        profit_factor=pf_val,
        max_drawdown=max_dd,
        num_trades=combined_trades,
        avg_win=sum(all_win_pcts) / len(all_win_pcts) if all_win_pcts else 0,
        avg_loss=sum(all_loss_pcts) / len(all_loss_pcts) if all_loss_pcts else 0,
        avg_duration=sum(all_durations) / len(all_durations) if all_durations else 0,
        reasons=combined_reasons,
        avg_mfe_gap=sum(mfe_gaps) / len(mfe_gaps) if mfe_gaps else 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer combinations for faster results")
    parser.add_argument("--ticker", default=None, help="Test on specific ticker only")
    parser.add_argument("--top", type=int, default=5, help="Use top N tickers by data volume")
    args = parser.parse_args()

    # Determine which tickers to test
    if args.ticker:
        tickers = [args.ticker]
    else:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT ticker, COUNT(*) as days
            FROM trading_days WHERE call_bars > 0
            GROUP BY ticker ORDER BY days DESC LIMIT ?
        """, (args.top,)).fetchall()
        conn.close()
        tickers = [r[0] for r in rows]

    print(f"Grid search on tickers: {', '.join(tickers)}")
    print()

    # ============================================================
    # Define parameter grid
    # Focus on the gates that caused early exits:
    # 1. setup_failed (was killing winners that hadn't gained 10% by min 10)
    # 2. premium_stop (hard stop)
    # 3. trailing stop (phase trail)
    # 4. time_decay (stale premium exit)
    # 5. theta_bleed
    # ============================================================

    if args.quick:
        grid = {
            "setup_failed_min": [10, 20],
            "setup_failed_gain_pct": [5, 15],
            "premium_stop_pct": [50, 70],
            "grace_period_min": [5],
            "time_decay_hold_min": [45, 90],
            "time_decay_stale_min": [5, 10],
            "theta_bleed_hold_min": [45, 90],
            "theta_bleed_max_loss_pct": [30, 50],
            "time_stop_min": [30, 60],
            "time_stop_gain_pct": [3, 5],
        }
    else:
        grid = {
            "setup_failed_min": [10, 15, 20, 30],
            "setup_failed_gain_pct": [3, 5, 10, 15],
            "premium_stop_pct": [40, 50, 60, 70, 80],
            "grace_period_min": [3, 5, 8],
            "time_decay_hold_min": [30, 45, 60, 90],
            "time_decay_stale_min": [5, 8, 10, 15],
            "theta_bleed_hold_min": [30, 45, 60, 90],
            "theta_bleed_max_loss_pct": [20, 30, 40, 50],
            "time_stop_min": [20, 30, 45, 60],
            "time_stop_gain_pct": [3, 5, 8],
        }

    # Generate all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    print(f"Total parameter combinations: {len(combos):,}")

    # That's a LOT of combos for full grid. Let's do a smarter approach:
    # Phase 1: Test each parameter independently to find best values
    # Phase 2: Combine the best values and do a focused grid

    # ============================================================
    # PHASE 1: Independent parameter sweeps
    # ============================================================
    print("\n" + "=" * 100)
    print("  PHASE 1: Independent parameter sweeps")
    print("=" * 100)

    baseline_params = VinnyParams()
    baseline = run_grid_backtest(tickers, {})
    print(f"\n  BASELINE: PnL=${baseline.total_pnl:+,.2f} ({baseline.total_pnl_pct:+.1f}%) | "
          f"WR={baseline.win_rate:.0f}% | PF={baseline.profit_factor:.2f} | "
          f"Trades={baseline.num_trades} | MaxDD={baseline.max_drawdown:.1f}% | "
          f"AvgWin={baseline.avg_win:+.1f}% | AvgLoss={baseline.avg_loss:+.1f}% | "
          f"MFE_gap={baseline.avg_mfe_gap:.1f}%")
    print(f"  Exit reasons: {baseline.reasons}")
    print()

    best_per_param = {}
    for param_name in keys:
        print(f"  --- Sweeping {param_name} ---")
        param_results = []
        for val in grid[param_name]:
            r = run_grid_backtest(tickers, {param_name: val})
            if r is None:
                continue
            param_results.append((val, r))
            marker = " <<<" if r.total_pnl > baseline.total_pnl else ""
            print(f"    {param_name}={val:>6}: PnL=${r.total_pnl:+,.2f} WR={r.win_rate:.0f}% "
                  f"PF={r.profit_factor:.2f} Trades={r.num_trades} "
                  f"AvgWin={r.avg_win:+.1f}% AvgLoss={r.avg_loss:+.1f}% "
                  f"MFE_gap={r.avg_mfe_gap:.1f}%{marker}")

        # Best = highest PnL with reasonable win rate
        if param_results:
            best_val, best_r = max(param_results, key=lambda x: x[1].total_pnl)
            best_per_param[param_name] = best_val
            print(f"    → Best: {param_name}={best_val}")
        print()

    # ============================================================
    # PHASE 2: Combined best values + focused grid around them
    # ============================================================
    print("=" * 100)
    print("  PHASE 2: Combined best values")
    print("=" * 100)

    combined_best = run_grid_backtest(tickers, best_per_param)
    print(f"\n  COMBINED BEST: PnL=${combined_best.total_pnl:+,.2f} ({combined_best.total_pnl_pct:+.1f}%) | "
          f"WR={combined_best.win_rate:.0f}% | PF={combined_best.profit_factor:.2f} | "
          f"Trades={combined_best.num_trades} | MaxDD={combined_best.max_drawdown:.1f}% | "
          f"AvgWin={combined_best.avg_win:+.1f}% | AvgLoss={combined_best.avg_loss:+.1f}% | "
          f"MFE_gap={combined_best.avg_mfe_gap:.1f}%")
    print(f"  Params: {best_per_param}")
    print(f"  Exit reasons: {combined_best.reasons}")

    # ============================================================
    # PHASE 3: Focused grid around the best combinations
    # Vary the top 3-4 most impactful params near their best values
    # ============================================================
    print()
    print("=" * 100)
    print("  PHASE 3: Focused grid search around best values")
    print("=" * 100)

    # Build a focused grid: for each best value, try ±1 step
    focused_grid = {}
    for param_name, best_val in best_per_param.items():
        original_values = grid[param_name]
        idx = original_values.index(best_val) if best_val in original_values else 0
        # Include best and neighbors
        neighbors = set()
        for offset in [-1, 0, 1]:
            ni = idx + offset
            if 0 <= ni < len(original_values):
                neighbors.add(original_values[ni])
        focused_grid[param_name] = sorted(neighbors)

    focused_keys = list(focused_grid.keys())
    focused_values = list(focused_grid.values())
    focused_combos = list(itertools.product(*focused_values))
    print(f"\n  Focused combinations: {len(focused_combos):,}")

    all_results = []
    start_time = time.time()
    for i, combo in enumerate(focused_combos):
        params_dict = dict(zip(focused_keys, combo))
        r = run_grid_backtest(tickers, params_dict)
        if r is None:
            continue
        all_results.append(r)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(focused_combos) - i - 1) / rate
            print(f"  [{i+1}/{len(focused_combos)}] {rate:.1f} combos/sec, ETA {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n  Completed {len(focused_combos)} combos in {elapsed:.0f}s")

    # ============================================================
    # RESULTS: Top 20 parameter combinations
    # ============================================================
    print()
    print("=" * 100)
    print("  TOP 20 PARAMETER COMBINATIONS (sorted by PnL)")
    print("=" * 100)

    # Sort by total PnL
    all_results.sort(key=lambda r: r.total_pnl, reverse=True)

    for rank, r in enumerate(all_results[:20], 1):
        print(f"\n  #{rank}: PnL=${r.total_pnl:+,.2f} ({r.total_pnl_pct:+.1f}%) | "
              f"WR={r.win_rate:.0f}% | PF={r.profit_factor:.2f} | "
              f"Trades={r.num_trades} | MaxDD={r.max_drawdown:.1f}% | "
              f"AvgWin={r.avg_win:+.1f}% | AvgLoss={r.avg_loss:+.1f}% | "
              f"MFE_gap={r.avg_mfe_gap:.1f}%")
        print(f"       Params: {r.params}")
        print(f"       Exits:  {r.reasons}")

    # ============================================================
    # Compare baseline vs best
    # ============================================================
    best = all_results[0] if all_results else combined_best
    print()
    print("=" * 100)
    print("  BASELINE vs BEST")
    print("=" * 100)
    print(f"                    {'Baseline':>12}  {'Best':>12}  {'Delta':>12}")
    print(f"  PnL $             ${baseline.total_pnl:>+10,.2f}  ${best.total_pnl:>+10,.2f}  ${best.total_pnl - baseline.total_pnl:>+10,.2f}")
    print(f"  PnL %             {baseline.total_pnl_pct:>+11.1f}%  {best.total_pnl_pct:>+11.1f}%  {best.total_pnl_pct - baseline.total_pnl_pct:>+11.1f}%")
    print(f"  Win Rate           {baseline.win_rate:>10.0f}%   {best.win_rate:>10.0f}%   {best.win_rate - baseline.win_rate:>+10.0f}%")
    print(f"  Profit Factor     {baseline.profit_factor:>12.2f}  {best.profit_factor:>12.2f}  {best.profit_factor - baseline.profit_factor:>+12.2f}")
    print(f"  Max Drawdown      {baseline.max_drawdown:>11.1f}%  {best.max_drawdown:>11.1f}%  {best.max_drawdown - baseline.max_drawdown:>+11.1f}%")
    print(f"  Trades            {baseline.num_trades:>12}  {best.num_trades:>12}  {best.num_trades - baseline.num_trades:>+12}")
    print(f"  Avg Win           {baseline.avg_win:>+11.1f}%  {best.avg_win:>+11.1f}%  {best.avg_win - baseline.avg_win:>+11.1f}%")
    print(f"  Avg Loss          {baseline.avg_loss:>+11.1f}%  {best.avg_loss:>+11.1f}%  {best.avg_loss - baseline.avg_loss:>+11.1f}%")
    print(f"  MFE Gap           {baseline.avg_mfe_gap:>11.1f}%  {best.avg_mfe_gap:>11.1f}%  {best.avg_mfe_gap - baseline.avg_mfe_gap:>+11.1f}%")
    print()
    print(f"  RECOMMENDED SETTINGS:")
    for k, v in best.params.items():
        print(f"    {k} = {v}")
    print()

    # ============================================================
    # Also show top by risk-adjusted (PnL / MaxDD)
    # ============================================================
    risk_adjusted = sorted(all_results, key=lambda r: r.total_pnl / max(r.max_drawdown, 0.1), reverse=True)
    print("=" * 100)
    print("  TOP 5 RISK-ADJUSTED (PnL / MaxDD)")
    print("=" * 100)
    for rank, r in enumerate(risk_adjusted[:5], 1):
        ra = r.total_pnl / max(r.max_drawdown, 0.1)
        print(f"  #{rank}: RA={ra:.1f} | PnL=${r.total_pnl:+,.2f} | "
              f"WR={r.win_rate:.0f}% | MaxDD={r.max_drawdown:.1f}% | "
              f"Params: {r.params}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
