"""Mass strategy parameter sweep using real Polygon 1-minute options data.

Tests 300+ combinations of stop width, min hold time, exit mode, and trail %
across all 13 tickers and 500+ days of real data. Finds the sweet spot for
stop timing and width.

Usage:
    python scripts/strategy_sweep.py
    python scripts/strategy_sweep.py --ticker SPY
    python scripts/strategy_sweep.py --top 30
"""

import argparse
import itertools
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
STARTING_BALANCE = 10_000.0


@dataclass
class SweepParams:
    """Tunable parameters for the sweep."""
    # --- Core question: how wide should stops be? ---
    stop_pct: float = 50.0           # premium drop % from entry to exit
    min_hold_min: int = 0            # don't check stops for first N minutes

    # --- Exit mode ---
    # "targets"       = sell partial at T1/T2 gains, then trail
    # "trail"         = pure trailing stop from peak (no targets)
    # "trail_after_t1"= targets + switch to trail after T1
    # "eod"           = hold until deadline (no early exit except stop)
    exit_mode: str = "targets"

    # --- Trail settings ---
    trail_activation_pct: float = 30.0  # % gain from entry to activate trail
    trail_drop_pct: float = 25.0        # % drop from peak to exit

    # --- Target settings (for "targets" and "trail_after_t1" modes) ---
    t1_gain_pct: float = 50.0
    t1_sell_pct: float = 50.0       # % of position to sell at T1
    t2_gain_pct: float = 200.0
    t2_sell_pct: float = 25.0       # % of remaining at T2

    # --- Fixed settings ---
    max_pos_pct: float = 5.0
    min_premium: float = 0.30
    grace_period_min: int = 2       # absolute min before any exit
    deadline_hour: int = 15
    deadline_minute: int = 30
    entry_hour: int = 10
    entry_minute: int = 0
    direction: str = "both"
    slippage_bps: float = 50.0


@dataclass
class TradeResult:
    date: str
    ticker: str
    direction: str
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    pnl_dollars: float
    exit_reason: str
    duration_min: int
    mfe_pct: float = 0.0  # max favorable excursion


def load_trading_days(ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT date, atm_call_ticker, atm_put_ticker, atm_strike
        FROM trading_days
        WHERE ticker = ? AND call_bars > 0 AND put_bars > 0
        ORDER BY date
    """, (ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_option_bars(contract_ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM option_bars
        WHERE contract_ticker = ?
        ORDER BY timestamp
    """, (contract_ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ts_to_et(timestamp_ms):
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)


def find_entry_bar(bars, entry_hour, entry_minute):
    target_minutes = entry_hour * 60 + entry_minute
    best_idx = None
    best_diff = float("inf")
    for i, bar in enumerate(bars):
        et = ts_to_et(bar["timestamp"])
        bar_minutes = et.hour * 60 + et.minute
        diff = abs(bar_minutes - target_minutes)
        if bar_minutes >= target_minutes and diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def simulate_trade(bars, params: SweepParams, entry_idx, direction):
    if entry_idx >= len(bars) - 5:
        return None

    entry_bar = bars[entry_idx]
    raw_entry = entry_bar["vwap"] if entry_bar.get("vwap", 0) > 0 else entry_bar["close"]
    if raw_entry <= 0 or raw_entry < params.min_premium:
        return None

    entry_premium = raw_entry * (1 + params.slippage_bps / 10000)
    cost_per_contract = entry_premium * 100
    max_spend = STARTING_BALANCE * (params.max_pos_pct / 100)
    contracts = max(1, int(max_spend / cost_per_contract))
    total_cost = contracts * cost_per_contract

    peak_premium = entry_premium
    t1_hit = False
    t2_hit = False
    remaining_pct = 100.0
    weighted_pnl = 0.0
    exit_reason = None
    exit_premium = entry_premium
    exit_bar_idx = entry_idx
    trail_active = params.exit_mode == "trail"  # pure trail starts immediately

    for bar_idx in range(entry_idx + 1, len(bars)):
        bar = bars[bar_idx]
        minutes_elapsed = bar_idx - entry_idx
        current = bar["close"]
        if current <= 0:
            continue

        if current > peak_premium:
            peak_premium = current

        bar_et = ts_to_et(bar["timestamp"])
        gain_pct = (current - entry_premium) / entry_premium * 100

        # ---- Deadline ----
        if (bar_et.hour > params.deadline_hour or
            (bar_et.hour == params.deadline_hour and bar_et.minute >= params.deadline_minute)):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "deadline"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # ---- Grace period ----
        if minutes_elapsed < params.grace_period_min:
            continue

        # ---- Targets (if applicable) ----
        if params.exit_mode in ("targets", "trail_after_t1"):
            if not t1_hit and gain_pct >= params.t1_gain_pct and remaining_pct > 0:
                sell_pct = remaining_pct * (params.t1_sell_pct / 100)
                weighted_pnl += sell_pct * gain_pct
                remaining_pct -= sell_pct
                t1_hit = True
                if params.exit_mode == "trail_after_t1":
                    trail_active = True

            if not t2_hit and gain_pct >= params.t2_gain_pct and remaining_pct > 0:
                sell_pct = remaining_pct * (params.t2_sell_pct / 100)
                weighted_pnl += sell_pct * gain_pct
                remaining_pct -= sell_pct
                t2_hit = True

        # ---- Min hold before stops ----
        if minutes_elapsed < params.min_hold_min:
            continue

        # ---- Stop loss (EOD mode skips this) ----
        if params.exit_mode != "eod" and remaining_pct > 0:
            drop_from_entry = (entry_premium - current) / entry_premium * 100
            if drop_from_entry >= params.stop_pct:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
                exit_reason = "stop"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # ---- Trailing stop ----
        if trail_active and remaining_pct > 0 and peak_premium > entry_premium:
            peak_gain = (peak_premium - entry_premium) / entry_premium * 100
            if peak_gain >= params.trail_activation_pct:
                drop_from_peak = (peak_premium - current) / peak_premium * 100
                if drop_from_peak >= params.trail_drop_pct:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trail"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break

    # If never exited, close at last bar
    if exit_reason is None:
        last_bar = bars[-1]
        exit_premium = last_bar["close"]
        if exit_premium > 0:
            gain_pct = (exit_premium - entry_premium) / entry_premium * 100
        else:
            gain_pct = -100
        if remaining_pct > 0:
            weighted_pnl += remaining_pct * gain_pct
            remaining_pct = 0
        exit_reason = "eod"
        exit_bar_idx = len(bars) - 1

    slippage_cost = params.slippage_bps / 100
    total_pnl_pct = weighted_pnl / 100 - slippage_cost
    pnl_dollars = total_cost * (total_pnl_pct / 100)
    mfe_pct = (peak_premium - entry_premium) / entry_premium * 100

    return TradeResult(
        date="",
        ticker="",
        direction=direction,
        entry_premium=entry_premium,
        exit_premium=exit_premium,
        peak_premium=peak_premium,
        pnl_pct=total_pnl_pct,
        pnl_dollars=pnl_dollars,
        exit_reason=exit_reason,
        duration_min=exit_bar_idx - entry_idx,
        mfe_pct=mfe_pct,
    )


def run_sweep_single(ticker, params: SweepParams):
    """Run one parameter set across all days for a ticker. Returns summary dict."""
    trading_days = load_trading_days(ticker)
    if not trading_days:
        return None

    trades = []
    for day in trading_days:
        directions = []
        if params.direction in ("call", "both"):
            directions.append(("call", day["atm_call_ticker"]))
        if params.direction in ("put", "both"):
            directions.append(("put", day["atm_put_ticker"]))

        for direction, contract_ticker in directions:
            bars = load_option_bars(contract_ticker)
            if not bars or len(bars) < 20:
                continue

            entry_idx = find_entry_bar(bars, params.entry_hour, params.entry_minute)
            if entry_idx is None or entry_idx >= len(bars) - 5:
                continue

            result = simulate_trade(bars, params, entry_idx, direction)
            if result is None:
                continue
            result.date = day["date"]
            result.ticker = ticker
            trades.append(result)

    if not trades:
        return None

    wins = [t for t in trades if t.pnl_dollars >= 0]
    losses = [t for t in trades if t.pnl_dollars < 0]
    total_pnl = sum(t.pnl_dollars for t in trades)
    gross_wins = sum(t.pnl_dollars for t in wins)
    gross_losses = abs(sum(t.pnl_dollars for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else 999.0

    # Track balance for drawdown
    balance = STARTING_BALANCE
    peak_bal = STARTING_BALANCE
    max_dd = 0.0
    for t in trades:
        balance += t.pnl_dollars
        if balance > peak_bal:
            peak_bal = balance
        dd = (peak_bal - balance) / peak_bal * 100 if peak_bal > 0 else 0
        if dd > max_dd:
            max_dd = dd

    avg_mfe = sum(t.mfe_pct for t in trades) / len(trades) if trades else 0
    stopped = [t for t in trades if t.exit_reason == "stop"]
    stopped_mfe = sum(t.mfe_pct for t in stopped) / len(stopped) if stopped else 0

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "total_pnl": total_pnl,
        "pnl_pct": total_pnl / STARTING_BALANCE * 100,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "avg_win": sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
        "avg_mfe": avg_mfe,
        "stopped_trades": len(stopped),
        "stopped_avg_mfe": stopped_mfe,
        "avg_duration": sum(t.duration_min for t in trades) / len(trades),
    }


def build_param_grid():
    """Build the parameter combinations to test."""
    grid = []

    # --- Stop width × min hold × exit mode ---
    stop_pcts = [25, 35, 50, 65, 80, 100]          # 6 values
    min_holds = [0, 5, 10, 15, 20, 30]              # 6 values
    exit_modes = ["targets", "trail", "trail_after_t1", "eod"]  # 4 values
    trail_drops = [15, 25, 35, 50]                   # 4 values
    trail_activations = [20, 30, 50]                  # 3 values

    for stop in stop_pcts:
        for hold in min_holds:
            for mode in exit_modes:
                if mode in ("trail", "trail_after_t1"):
                    for ta in trail_activations:
                        for td in trail_drops:
                            p = SweepParams(
                                stop_pct=stop,
                                min_hold_min=hold,
                                exit_mode=mode,
                                trail_activation_pct=ta,
                                trail_drop_pct=td,
                            )
                            grid.append(p)
                else:
                    # targets/eod: trail settings don't matter
                    p = SweepParams(
                        stop_pct=stop,
                        min_hold_min=hold,
                        exit_mode=mode,
                    )
                    grid.append(p)

    return grid


def run_sweep(tickers=None, top_n=30):
    """Run the full parameter sweep."""
    if tickers is None:
        # Use most common 0DTE tickers
        tickers = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "META", "AMD", "AMZN", "GOOGL"]

    grid = build_param_grid()
    print(f"Testing {len(grid)} parameter combinations across {len(tickers)} tickers")
    print(f"({len(grid)} combos × ~500 days × ~2 dirs = ~{len(grid)*500*2:,} simulated trades)")
    print()

    results = []
    for i, params in enumerate(grid):
        combo_pnl = 0
        combo_trades = 0
        combo_wins = 0
        combo_losses = 0
        combo_stopped = 0
        combo_stopped_mfe = 0
        max_dd = 0

        for ticker in tickers:
            r = run_sweep_single(ticker, params)
            if r is None:
                continue
            combo_pnl += r["total_pnl"]
            combo_trades += r["trades"]
            combo_wins += r["wins"]
            combo_losses += r["losses"]
            combo_stopped += r["stopped_trades"]
            combo_stopped_mfe += r["stopped_avg_mfe"] * r["stopped_trades"]
            if r["max_drawdown"] > max_dd:
                max_dd = r["max_drawdown"]

        if combo_trades == 0:
            continue

        gross_win_rate = combo_wins / combo_trades * 100

        results.append({
            "stop_pct": params.stop_pct,
            "min_hold": params.min_hold_min,
            "exit_mode": params.exit_mode,
            "trail_act": params.trail_activation_pct,
            "trail_drop": params.trail_drop_pct,
            "total_pnl": combo_pnl,
            "pnl_pct": combo_pnl / (STARTING_BALANCE * len(tickers)) * 100,
            "trades": combo_trades,
            "wins": combo_wins,
            "losses": combo_losses,
            "win_rate": gross_win_rate,
            "max_dd": max_dd,
            "stopped": combo_stopped,
            "stopped_mfe": combo_stopped_mfe / combo_stopped if combo_stopped > 0 else 0,
        })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(grid)} combos tested...")

    if not results:
        print("No results!")
        return

    # Sort by total PnL
    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    # Print top N
    print()
    print("=" * 140)
    print(f"  TOP {top_n} STRATEGY CONFIGURATIONS (by total PnL across {len(tickers)} tickers, {STARTING_BALANCE:,.0f} starting balance each)")
    print("=" * 140)
    print()
    print(f"  {'#':>3} {'Stop%':>6} {'Hold':>5} {'Mode':>14} {'Trail':>10} "
          f"{'Trades':>6} {'WR%':>5} {'PnL$':>12} {'PnL%':>7} {'MaxDD':>6} "
          f"{'Stops':>6} {'StopMFE':>8}")
    print("  " + "-" * 130)

    for i, r in enumerate(results[:top_n], 1):
        trail_str = f"{r['trail_act']:.0f}/{r['trail_drop']:.0f}" if r['exit_mode'] in ('trail', 'trail_after_t1') else "-"
        print(f"  {i:3d} {r['stop_pct']:>5.0f}% {r['min_hold']:>4d}m {r['exit_mode']:>14} {trail_str:>10} "
              f"{r['trades']:>6} {r['win_rate']:>4.0f}% ${r['total_pnl']:>+11,.0f} "
              f"{r['pnl_pct']:>+6.1f}% {r['max_dd']:>5.1f}% "
              f"{r['stopped']:>6} {r['stopped_mfe']:>+7.0f}%")

    # Print worst for contrast
    print()
    print("  WORST 5:")
    print("  " + "-" * 130)
    for i, r in enumerate(results[-5:], 1):
        trail_str = f"{r['trail_act']:.0f}/{r['trail_drop']:.0f}" if r['exit_mode'] in ('trail', 'trail_after_t1') else "-"
        print(f"  {i:3d} {r['stop_pct']:>5.0f}% {r['min_hold']:>4d}m {r['exit_mode']:>14} {trail_str:>10} "
              f"{r['trades']:>6} {r['win_rate']:>4.0f}% ${r['total_pnl']:>+11,.0f} "
              f"{r['pnl_pct']:>+6.1f}% {r['max_dd']:>5.1f}% "
              f"{r['stopped']:>6} {r['stopped_mfe']:>+7.0f}%")

    # Key insights
    print()
    print("=" * 140)
    print("  KEY INSIGHTS")
    print("=" * 140)

    # Best by exit mode
    for mode in ["targets", "trail", "trail_after_t1", "eod"]:
        mode_results = [r for r in results if r["exit_mode"] == mode]
        if mode_results:
            best = mode_results[0]
            worst = mode_results[-1]
            trail_str = f" trail={best['trail_act']:.0f}/{best['trail_drop']:.0f}" if mode in ('trail', 'trail_after_t1') else ""
            print(f"  Best '{mode}': stop={best['stop_pct']:.0f}% hold={best['min_hold']}m{trail_str}"
                  f" → ${best['total_pnl']:+,.0f} ({best['win_rate']:.0f}% WR)")
            print(f"  Worst '{mode}': stop={worst['stop_pct']:.0f}% hold={worst['min_hold']}m"
                  f" → ${worst['total_pnl']:+,.0f}")

    # Average PnL by stop width
    print()
    print("  Average PnL by stop width:")
    for stop in sorted(set(r["stop_pct"] for r in results)):
        subset = [r for r in results if r["stop_pct"] == stop]
        avg = sum(r["total_pnl"] for r in subset) / len(subset)
        avg_wr = sum(r["win_rate"] for r in subset) / len(subset)
        print(f"    Stop {stop:>3.0f}%: avg PnL ${avg:>+10,.0f}  avg WR {avg_wr:.0f}%  (n={len(subset)})")

    # Average PnL by min hold
    print()
    print("  Average PnL by min hold time:")
    for hold in sorted(set(r["min_hold"] for r in results)):
        subset = [r for r in results if r["min_hold"] == hold]
        avg = sum(r["total_pnl"] for r in subset) / len(subset)
        avg_stopped = sum(r["stopped"] for r in subset) / len(subset)
        print(f"    Hold {hold:>2d}m: avg PnL ${avg:>+10,.0f}  avg stops {avg_stopped:.0f}")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "..", "journal", "strategy_sweep_results.csv")
    with open(csv_path, "w") as f:
        f.write("rank,stop_pct,min_hold,exit_mode,trail_activation,trail_drop,"
                "trades,wins,losses,win_rate,total_pnl,pnl_pct,max_dd,"
                "stopped,stopped_avg_mfe\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i},{r['stop_pct']},{r['min_hold']},{r['exit_mode']},"
                    f"{r['trail_act']},{r['trail_drop']},"
                    f"{r['trades']},{r['wins']},{r['losses']},{r['win_rate']:.1f},"
                    f"{r['total_pnl']:.2f},{r['pnl_pct']:.2f},{r['max_dd']:.1f},"
                    f"{r['stopped']},{r['stopped_mfe']:.1f}\n")
    print(f"\n  Full results saved to {csv_path}")

    return results


def run_single_ticker_sweep(ticker, top_n=20):
    """Run sweep on a single ticker for detailed analysis."""
    grid = build_param_grid()
    print(f"Testing {len(grid)} combinations on {ticker}")

    results = []
    for i, params in enumerate(grid):
        r = run_sweep_single(ticker, params)
        if r is None:
            continue

        results.append({
            "stop_pct": params.stop_pct,
            "min_hold": params.min_hold_min,
            "exit_mode": params.exit_mode,
            "trail_act": params.trail_activation_pct,
            "trail_drop": params.trail_drop_pct,
            **r,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(grid)} tested...")

    if not results:
        print("No results!")
        return

    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    print()
    print("=" * 130)
    print(f"  TOP {top_n} FOR {ticker} ({results[0]['trades']//2 if results else 0}+ days)")
    print("=" * 130)
    print()
    print(f"  {'#':>3} {'Stop%':>6} {'Hold':>5} {'Mode':>14} {'Trail':>10} "
          f"{'Trades':>6} {'WR%':>5} {'PnL$':>11} {'PnL%':>7} {'DD%':>5} "
          f"{'AvgWin':>7} {'AvgLoss':>8} {'PF':>5}")
    print("  " + "-" * 120)

    for i, r in enumerate(results[:top_n], 1):
        trail_str = f"{r['trail_act']:.0f}/{r['trail_drop']:.0f}" if r['exit_mode'] in ('trail', 'trail_after_t1') else "-"
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"  {i:3d} {r['stop_pct']:>5.0f}% {r['min_hold']:>4d}m {r['exit_mode']:>14} {trail_str:>10} "
              f"{r['trades']:>6} {r['win_rate']:>4.0f}% ${r['total_pnl']:>+10,.0f} "
              f"{r['pnl_pct']:>+6.1f}% {r['max_drawdown']:>4.1f}% "
              f"{r['avg_win']:>+6.0f}% {r['avg_loss']:>+7.0f}% {pf:>5}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass strategy parameter sweep")
    parser.add_argument("--ticker", default=None, help="Single ticker (default: all 9)")
    parser.add_argument("--top", type=int, default=30, help="Show top N results")
    args = parser.parse_args()

    if args.ticker:
        run_single_ticker_sweep(args.ticker, top_n=args.top)
    else:
        run_sweep(top_n=args.top)
