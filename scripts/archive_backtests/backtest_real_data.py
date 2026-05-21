"""Backtest V3: Uses REAL Polygon options price data (no premium estimation).

This is the definitive backtest — every premium value comes from actual
1-minute options bars, not a delta/theta model.

For each trading day in the historical DB:
1. Generate a synthetic signal at a configurable entry time
2. Look up the real ATM call and put premium from option_bars
3. Walk bar-by-bar through the day applying exit rules
4. Track PnL using actual option prices

This simulates: "If we got a signal at 10:00 AM for SPY ATM call,
what would have happened with our strategy?"

Usage:
    python scripts/backtest_real_data.py
    python scripts/backtest_real_data.py --ticker SPY --entry-hour 10
"""

import argparse
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
STARTING_BALANCE = 3000.0


@dataclass
class StrategyParams:
    """All tunable strategy parameters."""
    # Position sizing
    max_pos_pct: float = 2.0          # % of portfolio per trade
    min_premium: float = 0.50         # skip options cheaper than this

    # Stop loss
    premium_stop_pct: float = 50.0    # exit if premium drops this % from entry
    grace_period_min: int = 5         # no stops for first N minutes

    # Premium-based profit targets (sell partial)
    t1_gain_pct: float = 50.0         # sell at +50% premium gain
    t1_sell_pct: float = 50.0         # sell 50% of position at T1
    t2_gain_pct: float = 200.0        # sell at +200% premium gain
    t2_sell_pct: float = 25.0         # sell 25% of remaining at T2

    # Trailing stop
    trail_activation_pct: float = 50.0  # activate when premium up 50%
    trail_drop_pct: float = 50.0        # exit if drops 50% from peak

    # Time stop (no momentum)
    time_stop_min: int = 45           # cut after N minutes if no gain
    time_stop_gain_pct: float = 5.0   # must be up this % or get cut

    # Exit deadline
    deadline_hour: int = 15           # 3:00 PM ET
    deadline_minute: int = 0

    # Entry timing
    entry_hour: int = 10              # default entry at 10:00 AM ET
    entry_minute: int = 0

    # DCA
    enable_dca: bool = True
    dca_tranches: int = 3             # 40/30/30 split
    dca_first_pct: float = 40.0
    dca_dip_pct: float = 10.0         # add next tranche on 10% dip
    dca_window_min: int = 45

    # Slippage (applied to entry/exit premium)
    entry_slippage_bps: float = 50.0  # 0.5%
    exit_slippage_bps: float = 50.0

    # Daily limits
    daily_loss_limit_pct: float = 5.0
    max_trades_per_day: int = 3

    # Direction: "call", "put", or "both" (take both directions each day)
    direction: str = "both"


@dataclass
class TradeResult:
    """Result of a single simulated trade."""
    date: str
    ticker: str
    direction: str
    strike: float
    entry_premium: float
    exit_premium: float
    peak_premium: float
    pnl_pct: float
    pnl_dollars: float
    contracts: int
    total_cost: float
    exit_reason: str
    duration_min: int
    t1_hit: bool = False
    t2_hit: bool = False
    balance_after: float = 0.0
    entry_time: str = ""
    exit_time: str = ""


def load_trading_days(ticker):
    """Load all downloaded trading days for a ticker."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT date, open_price, close_price, high_price, low_price,
               atm_call_ticker, atm_put_ticker, atm_strike,
               call_bars, put_bars, underlying_bars
        FROM trading_days
        WHERE ticker = ? AND call_bars > 0 AND put_bars > 0
        ORDER BY date
    """, (ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_option_bars(contract_ticker):
    """Load all 1-minute bars for an options contract, sorted by time."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM option_bars
        WHERE contract_ticker = ?
        ORDER BY timestamp
    """, (contract_ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_underlying_bars(ticker, date):
    """Load 1-min underlying bars for a given date."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM underlying_bars
        WHERE ticker = ? AND date = ?
        ORDER BY timestamp
    """, (ticker, date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ts_to_et(timestamp_ms):
    """Convert unix ms timestamp to ET datetime (approximate: UTC-4 for EDT)."""
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)  # EDT


def et_time_str(timestamp_ms):
    """Format timestamp as HH:MM ET string."""
    et = ts_to_et(timestamp_ms)
    return et.strftime("%H:%M")


def simulate_trade(option_bars, params, entry_idx, direction, strike, balance):
    """Simulate a single trade using real options bar data.

    Args:
        option_bars: list of 1-min option bar dicts (real prices!)
        params: StrategyParams
        entry_idx: index into option_bars where we enter
        direction: "call" or "put"
        strike: strike price
        balance: current portfolio balance

    Returns:
        TradeResult or None if can't enter
    """
    if entry_idx >= len(option_bars):
        return None

    entry_bar = option_bars[entry_idx]

    # Use VWAP as entry price (more realistic than close)
    raw_entry = entry_bar["vwap"] if entry_bar["vwap"] > 0 else entry_bar["close"]
    if raw_entry <= 0 or raw_entry < params.min_premium:
        return None

    # Apply entry slippage (pay more)
    entry_premium = raw_entry * (1 + params.entry_slippage_bps / 10000)

    # Position sizing
    max_position = balance * (params.max_pos_pct / 100)
    cost_per_contract = entry_premium * 100
    if cost_per_contract <= 0:
        return None

    contracts = max(1, int(max_position / cost_per_contract))
    total_cost = contracts * cost_per_contract

    if total_cost > balance:
        contracts = max(1, int(balance / cost_per_contract))
        total_cost = contracts * cost_per_contract
    if total_cost > balance:
        return None

    # DCA: if enabled, only buy first tranche now
    if params.enable_dca and contracts >= params.dca_tranches:
        first_tranche = max(1, int(contracts * params.dca_first_pct / 100))
    else:
        first_tranche = contracts

    # Walk through bars
    peak_premium = entry_premium
    t1_hit = False
    t2_hit = False
    remaining_pct = 100.0  # % of position still open
    weighted_pnl = 0.0     # accumulated weighted PnL from partial exits
    exit_reason = None
    exit_premium = entry_premium
    exit_bar_idx = entry_idx
    dca_done = 1  # tranches filled so far
    dca_contracts = first_tranche
    dca_total_cost = first_tranche * cost_per_contract
    # For DCA: track average entry premium
    avg_entry = entry_premium

    deadline_ts = None  # will compute from first bar

    for bar_idx in range(entry_idx + 1, len(option_bars)):
        bar = option_bars[bar_idx]
        minutes_elapsed = bar_idx - entry_idx
        current = bar["close"]

        if current <= 0:
            continue  # skip bars with no trades

        # Track peak
        if current > peak_premium:
            peak_premium = current

        # Compute ET time for deadline check
        bar_et = ts_to_et(bar["timestamp"])

        # ---- DCA: add tranches on dips ----
        if (params.enable_dca and dca_done < params.dca_tranches
                and minutes_elapsed <= params.dca_window_min
                and contracts > dca_contracts):
            dip_pct = (avg_entry - current) / avg_entry * 100 if avg_entry > 0 else 0
            if dip_pct >= params.dca_dip_pct:
                # Add next tranche
                remaining_contracts = contracts - dca_contracts
                next_tranche = max(1, int(remaining_contracts / (params.dca_tranches - dca_done)))
                dca_contracts += next_tranche
                # Update average entry (weighted)
                new_cost = next_tranche * current * 100
                dca_total_cost += new_cost
                avg_entry = dca_total_cost / (dca_contracts * 100)
                dca_done += 1

        # Use avg_entry for PnL calculations
        gain_pct = (current - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0

        # ---- Exit deadline ----
        if (bar_et.hour > params.deadline_hour or
            (bar_et.hour == params.deadline_hour and bar_et.minute >= params.deadline_minute)):
            if remaining_pct > 0:
                trade_pnl = gain_pct
                weighted_pnl += remaining_pct * trade_pnl
                remaining_pct = 0
            exit_reason = "exit_deadline"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # ---- Premium targets (partial sells) ----
        if not t1_hit and gain_pct >= params.t1_gain_pct and remaining_pct > 0:
            sell_pct = remaining_pct * (params.t1_sell_pct / 100)
            weighted_pnl += sell_pct * gain_pct
            remaining_pct -= sell_pct
            t1_hit = True

        if not t2_hit and gain_pct >= params.t2_gain_pct and remaining_pct > 0:
            sell_pct = remaining_pct * (params.t2_sell_pct / 100)
            weighted_pnl += sell_pct * gain_pct
            remaining_pct -= sell_pct
            t2_hit = True

        # ---- Grace period ----
        if minutes_elapsed < params.grace_period_min:
            continue

        # ---- Premium stop ----
        if avg_entry > 0:
            drop_pct = (avg_entry - current) / avg_entry * 100
            if drop_pct >= params.premium_stop_pct:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "premium_stop"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # ---- Trailing stop ----
        if remaining_pct > 0 and peak_premium > avg_entry:
            peak_gain = (peak_premium - avg_entry) / avg_entry * 100
            if peak_gain >= params.trail_activation_pct:
                drop_from_peak = (peak_premium - current) / peak_premium * 100
                if drop_from_peak >= params.trail_drop_pct:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trailing_stop"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break

        # ---- Time stop (no momentum) ----
        if minutes_elapsed >= params.time_stop_min and not t1_hit:
            if gain_pct < params.time_stop_gain_pct:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "time_stop"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

    # If never exited, close at last bar
    if exit_reason is None:
        last_bar = option_bars[-1]
        exit_premium = last_bar["close"]
        gain_pct = (exit_premium - avg_entry) / avg_entry * 100 if avg_entry > 0 else 0
        if remaining_pct > 0:
            weighted_pnl += remaining_pct * gain_pct
            remaining_pct = 0
        exit_reason = "eod_close"
        exit_bar_idx = len(option_bars) - 1

    # Apply exit slippage
    slippage_pct = params.exit_slippage_bps / 100
    total_pnl_pct = weighted_pnl / 100 - slippage_pct

    pnl_dollars = total_cost * (total_pnl_pct / 100)

    return TradeResult(
        date=et_time_str(option_bars[entry_idx]["timestamp"])[:5],
        ticker="",  # filled by caller
        direction=direction,
        strike=strike,
        entry_premium=entry_premium,
        exit_premium=exit_premium,
        peak_premium=peak_premium,
        pnl_pct=total_pnl_pct,
        pnl_dollars=pnl_dollars,
        contracts=contracts,
        total_cost=total_cost,
        exit_reason=exit_reason,
        duration_min=exit_bar_idx - entry_idx,
        t1_hit=t1_hit,
        t2_hit=t2_hit,
        entry_time=et_time_str(option_bars[entry_idx]["timestamp"]),
        exit_time=et_time_str(option_bars[exit_bar_idx]["timestamp"]) if exit_bar_idx < len(option_bars) else "",
    )


def find_entry_bar(bars, entry_hour, entry_minute):
    """Find the bar index closest to the desired entry time."""
    target_minutes = entry_hour * 60 + entry_minute

    best_idx = None
    best_diff = float("inf")

    for i, bar in enumerate(bars):
        et = ts_to_et(bar["timestamp"])
        bar_minutes = et.hour * 60 + et.minute
        diff = abs(bar_minutes - target_minutes)
        # Only consider bars at or after the target time
        if bar_minutes >= target_minutes and diff < best_diff:
            best_diff = diff
            best_idx = i

    return best_idx


def run_backtest(ticker="SPY", params=None, verbose=True):
    """Run backtest using real Polygon options data."""
    if params is None:
        params = StrategyParams()

    trading_days = load_trading_days(ticker)
    if not trading_days:
        print(f"No data for {ticker}. Run download_all_tickers.py first.")
        return None

    if verbose:
        print(f"Backtesting {ticker}: {len(trading_days)} days with real options data")
        print(f"Strategy: {params.max_pos_pct}% pos, ${params.min_premium}+ prem, "
              f"{params.premium_stop_pct}% stop, T1@+{params.t1_gain_pct}%/{params.t1_sell_pct}% sell, "
              f"trail {params.trail_activation_pct}/{params.trail_drop_pct}%, "
              f"{params.time_stop_min}m time stop, {params.deadline_hour}:{params.deadline_minute:02d} deadline")
        print()

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []
    daily_pnl = {}

    for day in trading_days:
        date_str = day["date"]
        strike = day["atm_strike"]

        # Determine which contracts to trade
        directions = []
        if params.direction in ("call", "both"):
            directions.append(("call", day["atm_call_ticker"]))
        if params.direction in ("put", "both"):
            directions.append(("put", day["atm_put_ticker"]))

        day_loss = daily_pnl.get(date_str, 0)
        daily_limit = STARTING_BALANCE * (params.daily_loss_limit_pct / 100)

        for direction, contract_ticker in directions:
            # Daily loss limit check
            if day_loss <= -daily_limit:
                break

            # Max trades per day
            day_trades = sum(1 for t in trades if t.date == date_str)
            if day_trades >= params.max_trades_per_day:
                break

            # Load real option bars
            bars = load_option_bars(contract_ticker)
            if not bars or len(bars) < 30:  # need reasonable data
                continue

            # Find entry bar
            entry_idx = find_entry_bar(bars, params.entry_hour, params.entry_minute)
            if entry_idx is None or entry_idx >= len(bars) - 10:
                continue

            # Simulate the trade
            result = simulate_trade(bars, params, entry_idx, direction, strike, balance)
            if result is None:
                continue

            result.date = date_str
            result.ticker = ticker

            # Update balance
            balance += result.pnl_dollars
            result.balance_after = balance
            trades.append(result)

            # Track daily PnL
            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + result.pnl_dollars

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

    # ---- Report ----
    if not trades:
        print("No trades executed!")
        return None

    wins = [t for t in trades if t.pnl_dollars >= 0]
    losses = [t for t in trades if t.pnl_dollars < 0]
    total_pnl = balance - STARTING_BALANCE
    win_rate = len(wins) / len(trades) * 100
    gross_wins = sum(t.pnl_dollars for t in wins)
    gross_losses = abs(sum(t.pnl_dollars for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    if verbose:
        print("=" * 80)
        print(f"  {ticker} BACKTEST — REAL OPTIONS DATA ({len(trading_days)} days)")
        print("=" * 80)
        print()
        print(f"  Starting Balance:    ${STARTING_BALANCE:,.2f}")
        print(f"  Final Balance:       ${balance:,.2f}")
        print(f"  Total PnL:           ${total_pnl:+,.2f} ({total_pnl/STARTING_BALANCE*100:+.1f}%)")
        print()
        print(f"  Total Trades:        {len(trades)}")
        print(f"  Wins / Losses:       {len(wins)} / {len(losses)}")
        print(f"  Win Rate:            {win_rate:.1f}%")
        print(f"  Avg Win:             {avg_win:+.1f}%")
        print(f"  Avg Loss:            {avg_loss:+.1f}%")
        print(f"  Profit Factor:       {pf:.2f}")
        print(f"  Max Drawdown:        {max_drawdown:.1f}%")
        print()

        print("  Exit Reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pnl_for_reason = sum(t.pnl_dollars for t in trades if t.exit_reason == reason)
            print(f"    {reason:20s} {count:3d} trades  ${pnl_for_reason:+,.2f}")

        t1_trades = [t for t in trades if t.t1_hit]
        print(f"\n  T1 hits: {len(t1_trades)}/{len(trades)} ({len(t1_trades)/len(trades)*100:.0f}%)")

        # Trade details (first 30)
        print()
        show = trades[:30] if len(trades) > 30 else trades
        print(f"  {'Date':>10} {'Dir':>4} {'Strike':>7} {'Entry':>6} {'Exit':>6} {'Peak':>6} "
              f"{'PnL%':>7} {'PnL$':>9} {'Exit':>15} {'T1':>3} {'Min':>4}")
        print("  " + "-" * 95)
        for t in show:
            t1 = "Y" if t.t1_hit else "-"
            print(f"  {t.date:>10} {t.direction:>4} ${t.strike:>6.0f} "
                  f"${t.entry_premium:>5.2f} ${t.exit_premium:>5.2f} ${t.peak_premium:>5.2f} "
                  f"{t.pnl_pct:>+6.1f}% ${t.pnl_dollars:>+8.2f} {t.exit_reason:>15} "
                  f"{t1:>3} {t.duration_min:>4}")
        if len(trades) > 30:
            print(f"  ... and {len(trades) - 30} more trades")

        print()
        print("=" * 80)

    return {
        "trades": trades,
        "balance": balance,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / STARTING_BALANCE * 100,
        "num_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_drawdown": max_drawdown,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "reasons": reasons,
    }


def run_optimizer(ticker="SPY"):
    """Test multiple parameter combinations and rank them."""
    import itertools

    trading_days = load_trading_days(ticker)
    if not trading_days:
        print(f"No data for {ticker}.")
        return

    print(f"Optimizing strategy on {ticker}: {len(trading_days)} days of real options data")
    print()

    # Parameter grid
    grid = {
        "max_pos_pct":       [2.0, 5.0],
        "premium_stop_pct":  [40.0, 50.0],
        "t1_gain_pct":       [30.0, 50.0, 100.0],
        "t1_sell_pct":       [50.0, 75.0, 100.0],  # 100% = sell all at T1
        "trail_activation_pct": [30.0, 50.0],
        "trail_drop_pct":    [40.0, 50.0],
        "time_stop_min":     [30, 45],
        "deadline_hour":     [14, 15],  # 2 PM vs 3 PM
        "entry_hour":        [10],      # fixed for now
        "direction":         ["call", "put", "both"],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Testing {len(combos)} combinations...")

    results = []
    for i, values in enumerate(combos):
        params = StrategyParams()
        for k, v in zip(keys, values):
            setattr(params, k, v)

        result = run_backtest(ticker, params, verbose=False)
        if result is None:
            continue

        combo_str = " | ".join(f"{k}={v}" for k, v in zip(keys, values))
        result["combo"] = combo_str
        result["params"] = {k: v for k, v in zip(keys, values)}
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)} tested...")

    if not results:
        print("No results!")
        return

    # Sort by PnL
    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    print()
    print("=" * 120)
    print(f"  TOP 20 STRATEGIES — {ticker} ({len(trading_days)} days real data)")
    print("=" * 120)
    print()
    print(f"  {'Rank':>4} {'Trades':>6} {'W':>3} {'L':>3} {'WR%':>5} "
          f"{'PnL$':>10} {'PnL%':>7} {'PF':>5} {'MaxDD':>6} {'AvgW':>6} {'AvgL':>6}")
    print("  " + "-" * 70)

    for i, r in enumerate(results[:20], 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"  {i:4d} {r['num_trades']:>6} {r['wins']:>3} {r['losses']:>3} "
              f"{r['win_rate']:>4.0f}% ${r['total_pnl']:>+9.2f} "
              f"{r['total_pnl_pct']:>+6.1f}% {pf_str:>5} {r['max_drawdown']:>5.1f}% "
              f"{r['avg_win']:>+5.0f}% {r['avg_loss']:>+5.0f}%")

    # Print params for top 5
    print()
    for i, r in enumerate(results[:5], 1):
        p = r["params"]
        print(f"  #{i}: pos={p['max_pos_pct']}% stop={p['premium_stop_pct']}% "
              f"T1=+{p['t1_gain_pct']}%/sell{p['t1_sell_pct']}% "
              f"trail={p['trail_activation_pct']}/{p['trail_drop_pct']}% "
              f"timestop={p['time_stop_min']}m deadline={p['deadline_hour']}:00 "
              f"dir={p['direction']} | PnL: ${r['total_pnl']:+.2f} ({r['total_pnl_pct']:+.1f}%)")

    # Print worst combo for contrast
    worst = results[-1]
    wp = worst["params"]
    print(f"\n  WORST: pos={wp['max_pos_pct']}% stop={wp['premium_stop_pct']}% "
          f"T1=+{wp['t1_gain_pct']}%/sell{wp['t1_sell_pct']}% "
          f"dir={wp['direction']} | PnL: ${worst['total_pnl']:+.2f} ({worst['total_pnl_pct']:+.1f}%)")

    # Detailed output for #1
    print()
    print("=" * 120)
    print("  BEST STRATEGY — DETAILED RESULTS")
    print("=" * 120)
    best_params = StrategyParams()
    bp = results[0]["params"]
    for k, v in bp.items():
        setattr(best_params, k, v)
    run_backtest(ticker, best_params, verbose=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--optimize", action="store_true", help="Run parameter optimizer")
    parser.add_argument("--entry-hour", type=int, default=10)
    parser.add_argument("--direction", default="both", choices=["call", "put", "both"])
    args = parser.parse_args()

    if args.optimize:
        run_optimizer(ticker=args.ticker)
    else:
        params = StrategyParams(entry_hour=args.entry_hour, direction=args.direction)
        run_backtest(ticker=args.ticker, params=params)
