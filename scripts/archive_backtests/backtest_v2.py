"""Backtest V2: Research-based 0DTE strategy.

Key changes from V1:
- Premium-based exit targets (not underlying price targets)
  - Sell 50% at +100% premium gain
  - Sell 25% at +200% premium gain
  - Trail last 25% with 50% trailing stop from peak
- 1-2% position size (not 20%)
- $0.50 minimum premium filter
- 50% premium stop (tighter than 60%)
- 45-minute time stop (no momentum)
- 3:00 PM ET exit deadline (not 3:45)
- Entry time filter: only 9:45-10:30 AM or 1:00-2:00 PM ET
- DCA: 40/30/30 tranches, max 45 min window
- Daily loss cap: 5% of portfolio

Tests all combinations of filters to find optimal setup.
"""

import math
import os
import sqlite3
import sys
import itertools
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yfinance as yf

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "raw_messages.db")

# ---- Base Strategy Parameters ----
STARTING_BALANCE = 3000.0
MIN_SCORE = 75

# V2 defaults (research-based)
V2_DEFAULTS = {
    "max_pos_pct": 2.0,         # 1-2% per trade
    "min_premium": 0.50,         # skip cheap options
    "premium_stop_pct": 50.0,    # tighter stop
    "time_stop_minutes": 45,     # no-momentum cutoff
    "time_stop_min_gain": 5.0,   # must be up 5% or exit
    "exit_deadline_hour": 15,    # 3:00 PM ET
    "exit_deadline_minute": 0,
    "grace_period": 5,           # 5 min no stops
    # Premium-based exit targets
    "t1_premium_pct": 100.0,     # sell 50% at +100%
    "t1_sell_pct": 50.0,
    "t2_premium_pct": 200.0,     # sell 25% at +200%
    "t2_sell_pct": 25.0,
    # Trailing stop on remainder
    "trail_activation_pct": 50.0,  # activate trailing at +50%
    "trail_drop_pct": 50.0,        # exit if drops 50% from peak
    # Entry timing
    "entry_start_1": (9, 45),    # first window: 9:45-10:30
    "entry_end_1": (10, 30),
    "entry_start_2": (13, 0),    # second window: 1:00-2:00
    "entry_end_2": (14, 0),
    "use_time_windows": False,   # default off (our signals don't fit these windows)
    # DCA
    "dca_tranches": 3,
    "dca_first_pct": 40.0,
    "dca_window_minutes": 45,
    # Daily limits
    "daily_loss_limit_pct": 5.0,
    "max_concurrent": 3,
    # Slippage
    "entry_slippage_bps": 50.0,
    "exit_slippage_bps": 50.0,
}

# Entry/exit slippage
ENTRY_SLIPPAGE_BPS = 50.0
EXIT_SLIPPAGE_BPS = 50.0


def estimate_premium(entry_premium, entry_price, current_price, direction, strike,
                     minutes_elapsed, total_minutes=390):
    """Estimate option premium using dollar-delta model."""
    if entry_price == 0 or entry_premium == 0:
        return entry_premium

    time_remaining_pct = max(0.01, 1 - minutes_elapsed / total_minutes)

    # Delta: ~0.50 ATM, drops for OTM, decays with time
    moneyness = abs(current_price - strike) / strike if strike > 0 else 0
    base_delta = 0.50 * math.exp(-moneyness * 15)
    delta = base_delta * (0.5 + 0.5 * math.sqrt(time_remaining_pct))

    # Dollar change from delta
    underlying_change = current_price - entry_price
    if direction == "put":
        underlying_change = -underlying_change
    delta_dollars = delta * underlying_change

    # Theta decay (non-linear, sqrt of time)
    theta_dollars = entry_premium * (1 - math.sqrt(time_remaining_pct)) * 0.5

    estimated = entry_premium + delta_dollars - theta_dollars

    # Intrinsic floor
    if direction == "call":
        intrinsic = max(0, current_price - strike)
    else:
        intrinsic = max(0, strike - current_price)

    estimated = max(estimated, intrinsic)
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


def parse_signal_time(created_at):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(created_at, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse: {created_at}")


def utc_to_et_offset(utc_dt):
    month = utc_dt.month
    if 3 <= month <= 11:
        return utc_dt - timedelta(hours=4)
    return utc_dt - timedelta(hours=5)


def fetch_intraday(ticker, date_str):
    t = yf.Ticker(ticker)
    start = date_str
    end_date = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
    end = end_date.strftime("%Y-%m-%d")
    df = t.history(start=start, end=end, interval="1m")
    if df.empty:
        return None
    return df


def prefetch_bars(signals):
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


def simulate_trade_v2(signal, bars_df, signal_et_time, params):
    """Simulate a trade with V2 premium-based exit targets.

    Exit priority:
    1. Premium stop (50% loss from entry) — after grace period
    2. Premium target T1 (+100%): sell 50% of position
    3. Premium target T2 (+200%): sell 25% of position
    4. Trailing stop on remainder (50% from peak after +50% activation)
    5. Time stop (45 min no movement)
    6. Exit deadline (3:00 PM ET)
    """
    direction = signal["direction"]
    entry_price = signal["entry_price"]
    strike = signal["strike"]
    entry_premium = signal["atm_premium"]

    if not entry_premium or entry_premium <= 0:
        return None

    # Find starting bar
    signal_hour = signal_et_time.hour
    signal_minute = signal_et_time.minute

    start_idx = None
    for i, (ts, row) in enumerate(bars_df.iterrows()):
        bar_time = ts.to_pydatetime()
        if hasattr(bar_time, 'hour'):
            if (bar_time.hour > signal_hour or
                (bar_time.hour == signal_hour and bar_time.minute >= signal_minute)):
                start_idx = i
                break

    if start_idx is None:
        return None

    # Apply entry slippage
    slippage_mult = 1 + params["entry_slippage_bps"] / 10000
    adj_entry = entry_premium * slippage_mult

    # Calculate total remaining minutes
    market_open_minutes = (signal_hour - 9) * 60 + (signal_minute - 30)
    total_remaining = max(1, 390 - market_open_minutes)

    # Extract params
    premium_stop_pct = params["premium_stop_pct"]
    grace_minutes = params["grace_period"]
    t1_prem_pct = params["t1_premium_pct"]
    t1_sell_pct = params["t1_sell_pct"]
    t2_prem_pct = params["t2_premium_pct"]
    t2_sell_pct = params["t2_sell_pct"]
    trail_activation = params["trail_activation_pct"]
    trail_drop = params["trail_drop_pct"]
    time_stop_min = params["time_stop_minutes"]
    time_stop_gain = params["time_stop_min_gain"]
    deadline_h = params["exit_deadline_hour"]
    deadline_m = params["exit_deadline_minute"]

    # State
    peak_premium = adj_entry
    exit_reason = None
    exit_premium = adj_entry
    exit_bar = 0
    t1_hit = False
    t2_hit = False
    remaining_pct = 100.0  # % of position still open
    weighted_pnl = 0.0     # accumulated from partial exits

    bars_list = list(bars_df.iterrows())

    for bar_idx in range(start_idx, len(bars_list)):
        ts, row = bars_list[bar_idx]
        bar_time = ts.to_pydatetime()
        current_price = row["Close"]
        minutes_elapsed = bar_idx - start_idx

        # Estimate current premium
        current_premium = estimate_premium(
            adj_entry, entry_price, current_price,
            direction, strike, minutes_elapsed, total_remaining
        )

        # Track peak
        if current_premium > peak_premium:
            peak_premium = current_premium

        # Premium gain from entry
        gain_pct = (current_premium - adj_entry) / adj_entry * 100 if adj_entry > 0 else 0

        # ---- Exit deadline ----
        if hasattr(bar_time, 'hour'):
            if (bar_time.hour > deadline_h or
                (bar_time.hour == deadline_h and bar_time.minute >= deadline_m)):
                # Close everything remaining
                if remaining_pct > 0:
                    trade_pnl = (current_premium - adj_entry) / adj_entry * 100
                    weighted_pnl += remaining_pct * trade_pnl
                    remaining_pct = 0
                exit_reason = "exit_deadline"
                exit_premium = current_premium
                exit_bar = minutes_elapsed
                break

        # ---- Premium-based targets (sell partial) ----
        if not t1_hit and gain_pct >= t1_prem_pct and remaining_pct > 0:
            # T1: sell t1_sell_pct of remaining
            sell_pct = remaining_pct * (t1_sell_pct / 100)
            trade_pnl = (current_premium - adj_entry) / adj_entry * 100
            weighted_pnl += sell_pct * trade_pnl
            remaining_pct -= sell_pct
            t1_hit = True

        if not t2_hit and gain_pct >= t2_prem_pct and remaining_pct > 0:
            # T2: sell t2_sell_pct of remaining
            sell_pct = remaining_pct * (t2_sell_pct / 100)
            trade_pnl = (current_premium - adj_entry) / adj_entry * 100
            weighted_pnl += sell_pct * trade_pnl
            remaining_pct -= sell_pct
            t2_hit = True

        # ---- Grace period: no stops for first N minutes ----
        if minutes_elapsed < grace_minutes:
            continue

        # ---- Premium stop (hard stop) ----
        if adj_entry > 0:
            drop_pct = (adj_entry - current_premium) / adj_entry * 100
            if drop_pct >= premium_stop_pct:
                if remaining_pct > 0:
                    trade_pnl = (current_premium - adj_entry) / adj_entry * 100
                    weighted_pnl += remaining_pct * trade_pnl
                    remaining_pct = 0
                exit_reason = "premium_stop"
                exit_premium = current_premium
                exit_bar = minutes_elapsed
                break

        # ---- Trailing stop on remaining position ----
        if remaining_pct > 0 and adj_entry > 0 and peak_premium > adj_entry:
            peak_gain = (peak_premium - adj_entry) / adj_entry * 100
            if peak_gain >= trail_activation:
                drop_from_peak = (peak_premium - current_premium) / peak_premium * 100
                if drop_from_peak >= trail_drop:
                    trade_pnl = (current_premium - adj_entry) / adj_entry * 100
                    weighted_pnl += remaining_pct * trade_pnl
                    remaining_pct = 0
                    exit_reason = "trailing_stop"
                    exit_premium = current_premium
                    exit_bar = minutes_elapsed
                    break

        # ---- Time stop (no momentum) ----
        if minutes_elapsed >= time_stop_min and not t1_hit:
            if gain_pct < time_stop_gain:
                if remaining_pct > 0:
                    trade_pnl = (current_premium - adj_entry) / adj_entry * 100
                    weighted_pnl += remaining_pct * trade_pnl
                    remaining_pct = 0
                exit_reason = "time_stop"
                exit_premium = current_premium
                exit_bar = minutes_elapsed
                break

    # If never exited, close at EOD
    if exit_reason is None:
        if bars_list and remaining_pct > 0:
            last_ts, last_row = bars_list[-1]
            exit_premium = estimate_premium(
                adj_entry, entry_price, last_row["Close"],
                direction, strike, len(bars_list) - start_idx, total_remaining
            )
            trade_pnl = (exit_premium - adj_entry) / adj_entry * 100
            weighted_pnl += remaining_pct * trade_pnl
            remaining_pct = 0
        exit_reason = "eod_close"
        exit_bar = len(bars_list) - start_idx

    # Apply exit slippage to the blended PnL
    # (approximate: reduce gains, increase losses by slippage)
    slippage_pct = params["exit_slippage_bps"] / 100  # bps to %
    total_pnl_pct = weighted_pnl / 100 - slippage_pct  # weighted_pnl is sum of (pct * pnl_pct)

    targets_hit = []
    if t1_hit:
        targets_hit.append("T1")
    if t2_hit:
        targets_hit.append("T2")

    return {
        "signal_id": signal["id"],
        "ticker": signal["ticker"],
        "direction": direction,
        "entry_premium": adj_entry,
        "exit_premium": exit_premium,
        "peak_premium": peak_premium,
        "pnl_pct": total_pnl_pct,
        "exit_reason": exit_reason,
        "duration_bars": exit_bar,
        "targets_hit": targets_hit,
        "score": signal["score"],
        "bot_source": signal["bot_source"],
    }


def in_time_window(et_time, params):
    """Check if signal time is within allowed entry windows."""
    h, m = et_time.hour, et_time.minute
    t = h * 60 + m

    s1_h, s1_m = params["entry_start_1"]
    e1_h, e1_m = params["entry_end_1"]
    s2_h, s2_m = params["entry_start_2"]
    e2_h, e2_m = params["entry_end_2"]

    w1_start = s1_h * 60 + s1_m
    w1_end = e1_h * 60 + e1_m
    w2_start = s2_h * 60 + s2_m
    w2_end = e2_h * 60 + e2_m

    return (w1_start <= t <= w1_end) or (w2_start <= t <= w2_end)


def run_single(signals, bars_cache, params):
    """Run one backtest with given parameters."""
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []
    daily_pnl = {}  # date -> cumulative loss for daily limit
    skipped = {"score": 0, "cheap": 0, "late": 0, "no_data": 0,
               "no_balance": 0, "daily_limit": 0, "time_window": 0}

    for sig in signals:
        if sig["score"] < MIN_SCORE:
            skipped["score"] += 1
            continue

        entry_premium = sig["atm_premium"]
        if not entry_premium or entry_premium <= 0:
            continue

        # Min premium filter
        if entry_premium < params["min_premium"]:
            skipped["cheap"] += 1
            continue

        dt = parse_signal_time(sig["created_at"])
        et_time = utc_to_et_offset(dt)
        date_str = et_time.strftime("%Y-%m-%d")

        # Time window filter
        if params["use_time_windows"] and not in_time_window(et_time, params):
            skipped["time_window"] += 1
            continue

        # Daily loss limit check
        day_loss = daily_pnl.get(date_str, 0)
        daily_limit = STARTING_BALANCE * (params["daily_loss_limit_pct"] / 100)
        if day_loss <= -daily_limit:
            skipped["daily_limit"] += 1
            continue

        bars_df = bars_cache.get((date_str, sig["ticker"]))
        if bars_df is None or bars_df.empty:
            skipped["no_data"] += 1
            continue

        # Position sizing
        max_position = balance * (params["max_pos_pct"] / 100)
        cost_per_contract = entry_premium * 100
        contracts = max(1, int(max_position / cost_per_contract))
        total_cost = contracts * cost_per_contract

        if total_cost > balance:
            contracts = max(1, int(balance / cost_per_contract))
            total_cost = contracts * cost_per_contract
        if total_cost > balance:
            skipped["no_balance"] += 1
            continue

        result = simulate_trade_v2(sig, bars_df, et_time, params)
        if result is None:
            continue

        pnl_pct = result["pnl_pct"]
        pnl_dollars = total_cost * (pnl_pct / 100)
        balance += pnl_dollars

        # Track daily PnL
        daily_pnl[date_str] = daily_pnl.get(date_str, 0) + pnl_dollars

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
    avg_win = (sum(t["pnl_pct"] for t in wins) / len(wins)) if wins else 0
    avg_loss = (sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0

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
        "skipped": skipped,
    }


def run_all_combos():
    """Test all meaningful combinations of strategy parameters."""
    signals = load_signals()
    print(f"Loaded {len(signals)} signals")
    print(f"Starting balance: ${STARTING_BALANCE:,.2f}")
    print()
    bars_cache = prefetch_bars(signals)

    # Define parameter variations to test
    # Each "filter" is a dict of param overrides
    filters = {
        "A": ("Pos 2%",          {"max_pos_pct": 2.0}),
        "B": ("Pos 5%",          {"max_pos_pct": 5.0}),
        "C": ("Prem $0.50+",     {"min_premium": 0.50}),
        "D": ("Prem $0.25+",     {"min_premium": 0.25}),
        "E": ("Stop 50%",        {"premium_stop_pct": 50.0}),
        "F": ("Stop 40%",        {"premium_stop_pct": 40.0}),
        "G": ("Deadline 3PM",    {"exit_deadline_hour": 15, "exit_deadline_minute": 0}),
        "H": ("Deadline 2:30",   {"exit_deadline_hour": 14, "exit_deadline_minute": 30}),
        "I": ("T1@+50%/sell50",  {"t1_premium_pct": 50.0, "t1_sell_pct": 50.0}),
        "J": ("T1@+100%/sell50", {"t1_premium_pct": 100.0, "t1_sell_pct": 50.0}),
        "K": ("TimeStop 30m",    {"time_stop_minutes": 30}),
        "L": ("TimeStop 45m",    {"time_stop_minutes": 45}),
        "M": ("DailyLim 5%",     {"daily_loss_limit_pct": 5.0}),
        "N": ("DailyLim 3%",     {"daily_loss_limit_pct": 3.0}),
    }

    # Build meaningful combos: pick one from each category
    # Categories: position size (A/B), premium filter (C/D), stop (E/F),
    #             deadline (G/H), target (I/J), time stop (K/L), daily limit (M/N)
    categories = [
        ["A", "B"],   # pos size
        ["C", "D"],   # premium filter
        ["E", "F"],   # stop %
        ["G", "H"],   # exit deadline
        ["I", "J"],   # T1 target
        ["K", "L"],   # time stop
        ["M", "N"],   # daily limit
    ]

    # Generate all combos: 2^7 = 128
    results = []
    combos = list(itertools.product(*categories))
    print(f"Testing {len(combos)} parameter combinations...")
    print()

    for combo in combos:
        # Start from V2 defaults, override with combo selections
        params = dict(V2_DEFAULTS)
        labels = []
        for key in combo:
            label, overrides = filters[key]
            params.update(overrides)
            labels.append(label)

        combo_str = "".join(combo)
        label_str = " | ".join(labels)

        result = run_single(signals, bars_cache, params)
        result["combo"] = combo_str
        result["label"] = label_str
        result["params"] = params.copy()
        results.append(result)

    # Also run baseline (no filters, old strategy params)
    baseline_params = dict(V2_DEFAULTS)
    baseline_params.update({
        "max_pos_pct": 20.0,
        "min_premium": 0.0,
        "premium_stop_pct": 60.0,
        "exit_deadline_hour": 15, "exit_deadline_minute": 45,
        "t1_premium_pct": 100.0, "t1_sell_pct": 50.0,
        "time_stop_minutes": 30,
        "daily_loss_limit_pct": 10.0,
    })
    baseline = run_single(signals, bars_cache, baseline_params)
    baseline["combo"] = "OLD"
    baseline["label"] = "OLD STRATEGY (V1 baseline)"
    baseline["params"] = baseline_params
    results.append(baseline)

    # Sort by total PnL
    results.sort(key=lambda r: r["total_pnl"], reverse=True)

    # Print top 20 and bottom 5
    print("=" * 120)
    print("  0DTE STRATEGY OPTIMIZATION — TOP 20 COMBINATIONS (of 128 + baseline)")
    print("=" * 120)
    print()
    print(f"  {'Rank':>4} {'Combo':>8} {'Trades':>6} {'W':>3} {'L':>3} {'WR%':>5} "
          f"{'PnL$':>10} {'PnL%':>7} {'PF':>5} {'MaxDD':>6} {'AvgW':>6} {'AvgL':>6}  Config")
    print("  " + "-" * 112)

    for i, r in enumerate(results[:20], 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"  {i:4d} {r['combo']:>8} {r['num_trades']:>6} {r['wins']:>3} {r['losses']:>3} "
              f"{r['win_rate']:>4.0f}% ${r['total_pnl']:>+9.2f} "
              f"{r['total_pnl_pct']:>+6.1f}% {pf_str:>5} {r['max_drawdown']:>5.1f}% "
              f"{r['avg_win']:>+5.0f}% {r['avg_loss']:>+5.0f}%  {r['label']}")

    # Find where baseline ranks
    for i, r in enumerate(results, 1):
        if r["combo"] == "OLD":
            print(f"\n  Old strategy ranks #{i} of {len(results)}")
            break

    # Detailed breakdown of top 5
    print()
    print("=" * 120)
    print("  TOP 5 — TRADE-BY-TRADE DETAIL")
    print("=" * 120)

    for rank, r in enumerate(results[:5], 1):
        print(f"\n  #{rank}: [{r['combo']}] {r['label']}")
        print(f"  ${STARTING_BALANCE:,.0f} -> ${r['balance']:,.2f} ({r['total_pnl_pct']:+.1f}%) | "
              f"{r['wins']}W/{r['losses']}L ({r['win_rate']:.0f}%) | "
              f"PF {r['profit_factor']:.2f} | DD {r['max_drawdown']:.1f}%")

        if r["trades"]:
            print(f"    {'#':>3} {'Ticker':>6} {'Dir':>4} {'Prem':>6} {'PnL%':>7} "
                  f"{'PnL$':>9} {'Exit':>15} {'Tgts':>5} {'Bars':>4}")
            for j, t in enumerate(r["trades"], 1):
                tgts = ",".join(t["targets_hit"]) or "-"
                print(f"    {j:3d} {t['ticker']:>6} {t['direction']:>4} "
                      f"${t['entry_premium']:>5.2f} {t['pnl_pct']:>+6.1f}% "
                      f"${t['pnl_dollars']:>+8.2f} {t['exit_reason']:>15} {tgts:>5} {t['duration_bars']:>4}")

    # Summary recommendation
    best = results[0]
    print()
    print("=" * 120)
    print("  RECOMMENDATION")
    print("=" * 120)
    p = best["params"]
    print(f"  Best combo: [{best['combo']}]")
    print(f"  Position size:    {p['max_pos_pct']}% per trade")
    print(f"  Min premium:      ${p['min_premium']:.2f}")
    print(f"  Premium stop:     {p['premium_stop_pct']}% loss")
    print(f"  T1 exit:          sell {p['t1_sell_pct']:.0f}% at +{p['t1_premium_pct']:.0f}% premium")
    print(f"  T2 exit:          sell {p['t2_sell_pct']:.0f}% at +{p['t2_premium_pct']:.0f}% premium")
    print(f"  Trailing stop:    activate +{p['trail_activation_pct']:.0f}%, drop {p['trail_drop_pct']:.0f}%")
    print(f"  Time stop:        {p['time_stop_minutes']} min")
    print(f"  Exit deadline:    {p['exit_deadline_hour']}:{p['exit_deadline_minute']:02d} ET")
    print(f"  Daily loss limit: {p['daily_loss_limit_pct']}%")
    print("=" * 120)

    return results


if __name__ == "__main__":
    run_all_combos()
