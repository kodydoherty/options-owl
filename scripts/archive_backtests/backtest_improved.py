"""Backtest the improved strategy against all 28 signals using 1-minute intraday data.

Strategy features:
- Premium-only stops (60% drop from entry, no underlying price stops)
- 5-minute grace period (no stop checks in first 5 min)
- Trailing stop (activate at +30% from entry, exit if drops 40% from peak)
- No-momentum exit (cut after 30 min if premium hasn't gained 5%)
- Graduated scale-out (20%/25%/33%/50% at T1/T2/T3/T4, 100% at T5)
- EOD exit at 15:45 ET

Usage:
    python scripts/backtest_improved.py
"""

import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "raw_messages.db")

# ---- Strategy Parameters (matching .env / settings.py) ----
STARTING_BALANCE = 3000.0
MAX_POSITION_PCT = 20.0
MAX_CONCURRENT = 3
MIN_SCORE = 75

# Premium stop
PREMIUM_STOP_PCT = 60.0
GRACE_PERIOD_MINUTES = 5

# Trailing stop
TRAILING_ACTIVATION_PCT = 30.0
TRAILING_DROP_PCT = 40.0

# No-momentum exit
NO_MOMENTUM_MINUTES = 30
NO_MOMENTUM_MIN_GAIN_PCT = 5.0

# Scale-out percentages (% of remaining contracts to sell at each target)
SCALE_OUT = {
    "t1": 20.0,
    "t2": 25.0,
    "t3": 33.0,
    "t4": 50.0,
    "t5": 100.0,
}

# Slippage (basis points)
ENTRY_SLIPPAGE_BPS = 50.0
EXIT_SLIPPAGE_BPS = 50.0

# EOD cutoff
EOD_CUTOFF_HOUR = 15
EOD_CUTOFF_MINUTE = 45


def estimate_premium(entry_premium, entry_price, current_price, direction, strike, minutes_elapsed, total_minutes=390):
    """Estimate option premium from underlying price using dollar-delta model.

    For 0DTE options, premium change is driven by:
    1. Delta * (underlying price change in dollars) — the dominant factor
    2. Theta decay — loses time value over the day

    Key insight: cheap options have enormous leverage because delta * $move
    can easily exceed the entire premium.
    """
    if entry_price == 0 or entry_premium == 0:
        return entry_premium

    # Time decay
    time_remaining_pct = max(0.01, 1 - minutes_elapsed / total_minutes)

    # Delta: starts ~0.50 for ATM, decays toward expiry, drops for OTM
    moneyness = abs(current_price - strike) / strike if strike > 0 else 0
    base_delta = 0.50 * math.exp(-moneyness * 15)
    # Delta decays with time (gamma effect for 0DTE)
    delta = base_delta * (0.5 + 0.5 * math.sqrt(time_remaining_pct))

    # Dollar change from delta
    underlying_change = current_price - entry_price
    if direction == "put":
        underlying_change = -underlying_change
    delta_dollars = delta * underlying_change

    # Theta: for 0DTE, extrinsic value decays roughly as sqrt of time remaining
    # Total theta loss over the day = ~70% of initial extrinsic value
    extrinsic = max(entry_premium - max(0, underlying_change if direction == "call" else -underlying_change), 0)
    theta_dollars = entry_premium * (1 - math.sqrt(time_remaining_pct)) * 0.5

    # Net premium
    estimated = entry_premium + delta_dollars - theta_dollars

    # Intrinsic floor: option is worth at least its intrinsic value
    if direction == "call":
        intrinsic = max(0, current_price - strike)
    else:
        intrinsic = max(0, strike - current_price)

    estimated = max(estimated, intrinsic)
    # Absolute floor: bid is at least $0.01
    return max(0.01, estimated)


def load_signals():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, ticker, direction, strike, expiry, atm_premium, score,
               created_at, entry_price, target_1, target_2, stop_price,
               bot_source, target_1_pct, target_2_pct
        FROM trade_signals
        ORDER BY created_at
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def parse_signal_time(created_at: str) -> datetime:
    """Parse signal created_at (stored as UTC) to datetime."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(created_at, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse: {created_at}")


def utc_to_et_offset(utc_dt: datetime) -> datetime:
    """Convert UTC to ET (approximate: -4 for EDT, -5 for EST)."""
    # March-November is EDT (-4), otherwise EST (-5)
    month = utc_dt.month
    if 3 <= month <= 11:
        return utc_dt - timedelta(hours=4)
    return utc_dt - timedelta(hours=5)


def fetch_intraday(ticker: str, date_str: str):
    """Fetch 1-minute bars for a ticker on a given date using yfinance."""
    t = yf.Ticker(ticker)
    # yfinance needs start/end for intraday
    start = f"{date_str}"
    end_date = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
    end = end_date.strftime("%Y-%m-%d")

    df = t.history(start=start, end=end, interval="1m")
    if df.empty:
        return None
    return df


def simulate_trade(signal, bars_df, signal_et_time):
    """Simulate a single trade through 1-minute bars with the improved strategy.

    Returns dict with trade result or None if couldn't enter.
    """
    ticker = signal["ticker"]
    direction = signal["direction"]
    entry_price = signal["entry_price"]
    strike = signal["strike"]
    entry_premium = signal["atm_premium"]

    if not entry_premium or entry_premium <= 0:
        return None

    # Find the bar closest to signal time
    signal_hour = signal_et_time.hour
    signal_minute = signal_et_time.minute

    start_idx = None
    for i, (ts, row) in enumerate(bars_df.iterrows()):
        bar_time = ts.to_pydatetime()
        # bars_df index is in ET from yfinance
        if hasattr(bar_time, 'hour'):
            if (bar_time.hour > signal_hour or
                (bar_time.hour == signal_hour and bar_time.minute >= signal_minute)):
                start_idx = i
                break

    if start_idx is None:
        return None

    # Apply entry slippage
    slippage_mult = 1 + ENTRY_SLIPPAGE_BPS / 10000
    adjusted_entry_premium = entry_premium * slippage_mult

    # Total minutes in trading day (9:30 to 16:00 = 390 min)
    market_open_minutes = (signal_hour - 9) * 60 + (signal_minute - 30)
    total_remaining = 390 - market_open_minutes

    # Targets from signal
    targets = {}
    for t_num in range(1, 6):
        t_key = f"target_{t_num}"
        if signal.get(t_key):
            targets[t_num] = signal[t_key]

    # State tracking
    peak_premium = adjusted_entry_premium
    minutes_elapsed = 0
    exit_reason = None
    exit_premium = adjusted_entry_premium
    exit_bar = 0
    targets_hit = set()
    scale_out_log = []  # (target, pct_sold, premium_at_exit)

    bars_list = list(bars_df.iterrows())

    for bar_idx in range(start_idx, len(bars_list)):
        ts, row = bars_list[bar_idx]
        bar_time = ts.to_pydatetime()
        current_price = row["Close"]
        minutes_elapsed = bar_idx - start_idx

        # Estimate current premium
        current_premium = estimate_premium(
            adjusted_entry_premium, entry_price, current_price,
            direction, strike, minutes_elapsed, total_remaining
        )

        # Track peak premium (for trailing stop)
        if current_premium > peak_premium:
            peak_premium = current_premium

        # ---- EOD cutoff ----
        if hasattr(bar_time, 'hour'):
            if (bar_time.hour > EOD_CUTOFF_HOUR or
                (bar_time.hour == EOD_CUTOFF_HOUR and bar_time.minute >= EOD_CUTOFF_MINUTE)):
                exit_reason = "eod_cutoff"
                exit_premium = current_premium
                exit_bar = bar_idx - start_idx
                break

        # ---- Grace period: no stops for first N minutes ----
        if minutes_elapsed < GRACE_PERIOD_MINUTES:
            # Still check targets during grace period
            pass
        else:
            # ---- Premium stop ----
            if adjusted_entry_premium > 0:
                drop_pct = (adjusted_entry_premium - current_premium) / adjusted_entry_premium * 100
                if drop_pct >= PREMIUM_STOP_PCT:
                    exit_reason = "premium_stop"
                    exit_premium = current_premium
                    exit_bar = bar_idx - start_idx
                    break

            # ---- Trailing stop ----
            if adjusted_entry_premium > 0 and peak_premium > adjusted_entry_premium:
                gain_from_entry = (peak_premium - adjusted_entry_premium) / adjusted_entry_premium * 100
                if gain_from_entry >= TRAILING_ACTIVATION_PCT:
                    drop_from_peak = (peak_premium - current_premium) / peak_premium * 100
                    if drop_from_peak >= TRAILING_DROP_PCT:
                        exit_reason = "trailing_stop"
                        exit_premium = current_premium
                        exit_bar = bar_idx - start_idx
                        break

            # ---- No-momentum exit ----
            if minutes_elapsed >= NO_MOMENTUM_MINUTES:
                gain_pct = (current_premium - adjusted_entry_premium) / adjusted_entry_premium * 100
                if gain_pct < NO_MOMENTUM_MIN_GAIN_PCT and not targets_hit:
                    exit_reason = "no_momentum"
                    exit_premium = current_premium
                    exit_bar = bar_idx - start_idx
                    break

        # ---- Check target hits (for scale-out tracking) ----
        for t_num in sorted(targets.keys()):
            if t_num in targets_hit:
                continue
            target_price = targets[t_num]
            hit = False
            if direction == "call":
                hit = current_price >= target_price
            else:
                hit = current_price <= target_price

            if hit:
                targets_hit.add(t_num)
                t_key = f"t{t_num}"
                pct_to_sell = SCALE_OUT.get(t_key, 0)
                scale_out_log.append((t_num, pct_to_sell, current_premium))

    # If we never exited, EOD close
    if exit_reason is None:
        exit_reason = "eod_close"
        if bars_list:
            last_ts, last_row = bars_list[-1]
            exit_premium = estimate_premium(
                adjusted_entry_premium, entry_price, last_row["Close"],
                direction, strike, len(bars_list) - start_idx, total_remaining
            )
        exit_bar = len(bars_list) - start_idx

    # Apply exit slippage
    slippage_mult = 1 - EXIT_SLIPPAGE_BPS / 10000
    exit_premium *= slippage_mult

    # ---- Calculate PnL with scale-out ----
    if scale_out_log:
        # Account for partial profits taken at targets + remaining position at final exit
        total_pnl_pct = _calc_scaleout_pnl(
            adjusted_entry_premium, exit_premium, scale_out_log, exit_reason
        )
    else:
        # Full position exit (no targets were hit)
        if adjusted_entry_premium > 0:
            total_pnl_pct = (exit_premium - adjusted_entry_premium) / adjusted_entry_premium * 100
        else:
            total_pnl_pct = 0.0

    return {
        "signal_id": signal["id"],
        "ticker": ticker,
        "direction": direction,
        "entry_premium": adjusted_entry_premium,
        "exit_premium": exit_premium,
        "peak_premium": peak_premium,
        "pnl_pct": total_pnl_pct,
        "exit_reason": exit_reason,
        "duration_bars": exit_bar,
        "targets_hit": sorted(targets_hit),
        "scale_outs": len(scale_out_log),
        "score": signal["score"],
        "bot_source": signal["bot_source"],
    }


def _calc_scaleout_pnl(entry_prem, final_exit_prem, scale_log, exit_reason):
    """Calculate blended PnL% accounting for scale-out exits at different premiums."""
    remaining_pct = 100.0  # % of position still open
    weighted_pnl = 0.0

    for t_num, sell_pct, exit_at_premium in scale_log:
        # sell_pct is % of REMAINING, convert to % of TOTAL
        actual_pct = remaining_pct * (sell_pct / 100)
        trade_pnl = (exit_at_premium - entry_prem) / entry_prem * 100
        weighted_pnl += actual_pct * trade_pnl
        remaining_pct -= actual_pct

    # Remaining position exits at final premium
    if remaining_pct > 0:
        final_pnl = (final_exit_prem - entry_prem) / entry_prem * 100
        weighted_pnl += remaining_pct * final_pnl

    return weighted_pnl / 100  # normalize back to %


def prefetch_bars(signals):
    """Fetch all 1-min bars upfront and cache them."""
    date_ticker_map = {}
    for sig in signals:
        dt = parse_signal_time(sig["created_at"])
        et_time = utc_to_et_offset(dt)
        date_str = et_time.strftime("%Y-%m-%d")
        key = (date_str, sig["ticker"])
        if key not in date_ticker_map:
            date_ticker_map[key] = []
        date_ticker_map[key].append((sig, et_time))

    bars_cache = {}
    unique_fetches = set(date_ticker_map.keys())
    print(f"Fetching 1-min data for {len(unique_fetches)} ticker-days...")
    for date_str, ticker in sorted(unique_fetches):
        cache_key = (date_str, ticker)
        if cache_key not in bars_cache:
            df = fetch_intraday(ticker, date_str)
            bars_cache[cache_key] = df
            status = f"{len(df)} bars" if df is not None else "NO DATA"
            print(f"  {ticker} {date_str}: {status}")
    print()
    return bars_cache


def run_single_backtest(signals, bars_cache, *, min_premium=0.0, max_pos_pct=20.0,
                        time_cutoff_hour=16, time_cutoff_minute=0, min_premium_filter=0.0):
    """Run one backtest with given filter parameters.

    Filters:
      min_premium: skip signals with atm_premium < this (0 = no filter)
      max_pos_pct: max position as % of balance
      time_cutoff_hour/minute: skip signals after this ET time
      min_premium_filter: alias for min_premium (uses max of both)
    """
    effective_min_prem = max(min_premium, min_premium_filter)

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []
    skipped_reasons = {"low_score": 0, "cheap": 0, "late": 0, "no_data": 0, "no_balance": 0}

    for sig in signals:
        if sig["score"] < MIN_SCORE:
            skipped_reasons["low_score"] += 1
            continue

        entry_premium = sig["atm_premium"]
        if not entry_premium or entry_premium <= 0:
            continue

        # Filter 1/4: minimum premium
        if effective_min_prem > 0 and entry_premium < effective_min_prem:
            skipped_reasons["cheap"] += 1
            continue

        dt = parse_signal_time(sig["created_at"])
        et_time = utc_to_et_offset(dt)

        # Filter 3: time cutoff
        if (et_time.hour > time_cutoff_hour or
            (et_time.hour == time_cutoff_hour and et_time.minute > time_cutoff_minute)):
            skipped_reasons["late"] += 1
            continue

        date_str = et_time.strftime("%Y-%m-%d")
        bars_df = bars_cache.get((date_str, sig["ticker"]))
        if bars_df is None or bars_df.empty:
            skipped_reasons["no_data"] += 1
            continue

        # Filter 2: position sizing
        max_position = balance * (max_pos_pct / 100)
        cost_per_contract = entry_premium * 100
        contracts = max(1, int(max_position / cost_per_contract))
        total_cost = contracts * cost_per_contract

        if total_cost > balance:
            contracts = max(1, int(balance / cost_per_contract))
            total_cost = contracts * cost_per_contract

        if total_cost > balance:
            skipped_reasons["no_balance"] += 1
            continue

        result = simulate_trade(sig, bars_df, et_time)
        if result is None:
            continue

        pnl_pct = result["pnl_pct"]
        pnl_dollars = total_cost * (pnl_pct / 100)
        balance += pnl_dollars

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        result["contracts"] = contracts
        result["total_cost"] = total_cost
        result["pnl_dollars"] = pnl_dollars
        result["balance_after"] = balance
        trades.append(result)

    total_pnl = balance - STARTING_BALANCE
    wins = [t for t in trades if t["pnl_dollars"] >= 0]
    losses = [t for t in trades if t["pnl_dollars"] < 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    gross_wins = sum(t["pnl_dollars"] for t in wins)
    gross_losses = abs(sum(t["pnl_dollars"] for t in losses))
    pf = (gross_wins / gross_losses) if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0)

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
        "skipped": skipped_reasons,
    }


def run_combo_backtest():
    """Run all 16 filter combinations and rank them."""
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")
    print(f"Starting balance: ${STARTING_BALANCE:,.2f}")
    print()

    bars_cache = prefetch_bars(signals)

    # Define the 4 filters (on/off)
    filters = {
        "A": ("Min prem $0.25", {"min_premium": 0.25}),
        "B": ("Pos size 10%",   {"max_pos_pct": 10.0}),
        "C": ("No after 14:30", {"time_cutoff_hour": 14, "time_cutoff_minute": 30}),
        "D": ("Min prem $0.50", {"min_premium_filter": 0.50}),
    }

    # Generate all 16 combinations
    import itertools
    filter_keys = list(filters.keys())
    results = []

    for n in range(len(filter_keys) + 1):
        for combo in itertools.combinations(filter_keys, n):
            # Merge filter kwargs
            kwargs = {}
            for key in combo:
                kwargs.update(filters[key][1])

            # If both A ($0.25) and D ($0.50) are active, D dominates
            label = " + ".join(filters[k][0] for k in combo) if combo else "BASELINE (no filters)"
            combo_str = "".join(combo) if combo else "-"

            result = run_single_backtest(signals, bars_cache, **kwargs)
            result["label"] = label
            result["combo"] = combo_str
            results.append(result)

    # Sort by total PnL (best first)
    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    # Print summary table
    print("=" * 110)
    print("  ALL 16 FILTER COMBINATIONS — RANKED BY TOTAL PnL")
    print("=" * 110)
    print()
    print(f"  {'Rank':>4} {'Combo':>5} {'Trades':>6} {'Wins':>4} {'WR%':>5} "
          f"{'PnL$':>10} {'PnL%':>7} {'PF':>5} {'MaxDD':>6}  Filters")
    print("  " + "-" * 100)

    for i, r in enumerate(results, 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"  {i:4d} {r['combo']:>5} {r['num_trades']:>6} {r['wins']:>4} "
              f"{r['win_rate']:>4.0f}% ${r['total_pnl']:>+9.2f} "
              f"{r['total_pnl_pct']:>+6.1f}% {pf_str:>5} {r['max_drawdown']:>5.1f}%  {r['label']}")

    # Print detailed breakdown for top 3
    print()
    print("=" * 110)
    print("  TOP 3 COMBINATIONS — DETAILED BREAKDOWN")
    print("=" * 110)

    for rank, r in enumerate(results[:3], 1):
        print(f"\n  #{rank}: {r['label']}")
        print(f"  Balance: ${STARTING_BALANCE:,.2f} -> ${r['balance']:,.2f} "
              f"({r['total_pnl_pct']:+.1f}%)")
        print(f"  Trades: {r['num_trades']} | Wins: {r['wins']} | "
              f"Win Rate: {r['win_rate']:.0f}% | PF: {r['profit_factor']:.2f} | "
              f"Max DD: {r['max_drawdown']:.1f}%")

        if r["trades"]:
            print(f"  {'#':>5} {'Ticker':>6} {'Dir':>4} {'Prem':>6} {'PnL%':>7} "
                  f"{'PnL$':>9} {'Exit':>16} {'Targets':>8}")
            print("  " + "-" * 72)
            for j, t in enumerate(r["trades"], 1):
                targets_str = ",".join(f"T{x}" for x in t["targets_hit"]) or "-"
                print(f"  {j:5d} {t['ticker']:>6} {t['direction']:>4} "
                      f"${t['entry_premium']:>5.2f} {t['pnl_pct']:>+6.1f}% "
                      f"${t['pnl_dollars']:>+8.2f} {t['exit_reason']:>16} {targets_str:>8}")

    # Print the worst combo too for contrast
    worst = results[-1]
    print(f"\n  WORST: {worst['label']}")
    print(f"  Balance: ${STARTING_BALANCE:,.2f} -> ${worst['balance']:,.2f} "
          f"({worst['total_pnl_pct']:+.1f}%)")
    print(f"  Trades: {worst['num_trades']} | Wins: {worst['wins']} | "
          f"Win Rate: {worst['win_rate']:.0f}%")

    print()
    print("=" * 110)

    return results


if __name__ == "__main__":
    run_combo_backtest()
