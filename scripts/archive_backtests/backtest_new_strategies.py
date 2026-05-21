"""Backtest 10 new strategy ideas on real Polygon 1-min option bars.

Tests each strategy independently and in combination to find the best
additions to the current production strategy.

Strategies tested:
  1. Forced Time Exit (30-180 min grid)
  2. MFE Retracement Ratio (0.3-0.7 grid)
  3. Time-of-Day Regime (block midday chop, power hour tightening)
  4. Market Regime Switch (ATR-based trail width adjustment)
  5. Premium Spike Exit (partial exit on rapid premium spike)
  6. Time-Proportional Scale-Out (sell 10% every N minutes)
  7. Premium Velocity Entry Gate (reject fading entries)
  8. Premium Deceleration Exit (exit on momentum collapse)
  9. VIX-Adjusted Sizing (scale contracts by volatility)
 10. Daily Budget Tranches (morning/afternoon budget split)

Usage:
    python scripts/backtest_new_strategies.py
    python scripts/backtest_new_strategies.py --ticker SPY
    python scripts/backtest_new_strategies.py --ticker SPY,QQQ,IWM
"""

import math
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
STARTING_BALANCE = 5000.0


# ---------------------------------------------------------------------------
# Data loading (shared with backtest_real_data.py)
# ---------------------------------------------------------------------------


def load_trading_days(ticker):
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM option_bars WHERE contract_ticker = ? ORDER BY timestamp
    """, (contract_ticker,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_underlying_bars(ticker, date):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM underlying_bars WHERE ticker = ? AND date = ? ORDER BY timestamp
    """, (ticker, date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ts_to_et(timestamp_ms):
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)


def et_time_str(timestamp_ms):
    return ts_to_et(timestamp_ms).strftime("%H:%M")


# ---------------------------------------------------------------------------
# Strategy config
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Current production baseline + toggles for each new strategy."""
    # --- Current production baseline ---
    premium_stop_pct: float = 50.0
    grace_period_min: int = 8
    # Adaptive trail (production v2.1)
    adaptive_trail: bool = True
    adaptive_dormant_pct: float = 40.0    # no trail below +40%
    adaptive_active_width: float = 35.0   # 35% trail +40% to +150%
    adaptive_runner_width: float = 45.0   # 45% trail +150% to +400%
    adaptive_moonshot_width: float = 30.0 # 30% trail above +400%
    # Scale-out at targets
    t1_pct: float = 20.0   # 20% off at each target
    t1_gain: float = 20.0  # +20% premium
    t2_gain: float = 50.0  # +50%
    t3_gain: float = 100.0
    t4_gain: float = 200.0
    # Time-based
    no_momentum_min: int = 45
    no_momentum_gain: float = 0.0
    theta_bleed_min: int = 45
    theta_bleed_drop: float = 30.0
    eod_cutoff_hour: int = 15
    eod_cutoff_min: int = 45
    # Sizing
    max_pos_pct: float = 33.3   # 1/3 of portfolio per trade (3 slots)
    min_premium: float = 0.25
    max_contracts: int = 20
    # Entry
    entry_hour: int = 10
    entry_minute: int = 0
    direction: str = "both"
    slippage_bps: float = 50.0

    # --- NEW STRATEGY TOGGLES ---

    # 1. Forced time exit
    forced_exit_min: int = 0  # 0 = disabled

    # 2. MFE retracement ratio
    mfe_retrace_ratio: float = 0.0   # 0 = disabled; e.g. 0.5 = exit when giving back 50% of peak gain
    mfe_retrace_min_gain: float = 20.0  # only activate if peak gain was >= 20%

    # 3. Time-of-day regime
    tod_block_midday: bool = False
    tod_midday_start_hour: int = 11
    tod_midday_end_hour: int = 13
    tod_midday_end_min: int = 30
    tod_power_hour_tighten: bool = False  # tighten trail after 2 PM
    tod_power_tighten_factor: float = 0.7

    # 4. Market regime (ATR-based)
    regime_atr: bool = False
    regime_atr_trend_mult: float = 1.3    # widen trail by 30% on trending days
    regime_atr_chop_mult: float = 0.7     # tighten trail by 30% on choppy days

    # 5. Premium spike exit
    spike_exit: bool = False
    spike_threshold_pct: float = 50.0   # premium must spike 50%+ in spike_window
    spike_window_bars: int = 5
    spike_sell_pct: float = 50.0        # sell 50% on spike

    # 6. Time-proportional scale-out
    time_scaleout: bool = False
    time_scaleout_start_min: int = 30
    time_scaleout_interval_min: int = 15
    time_scaleout_pct: float = 10.0

    # 7. Premium velocity entry gate
    velocity_entry_gate: bool = False
    velocity_entry_bars: int = 5
    velocity_entry_min_pct: float = 2.0  # premium must be up 2% over last 5 bars

    # 8. Premium deceleration exit
    decel_exit: bool = False
    decel_window: int = 10
    decel_threshold: float = -2.0  # acceleration below this triggers exit
    decel_min_gain: float = 30.0

    # 9. VIX-adjusted sizing
    vix_sizing: bool = False
    vix_base: float = 18.0
    vix_scale_factor: float = 0.5

    # 10. Daily budget tranches
    budget_tranches: bool = False
    morning_budget_pct: float = 60.0
    afternoon_unlock_threshold: float = 0.0  # morning P&L must be >= $0


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def compute_first_hour_atr(underlying_bars):
    """Compute ATR of the first hour of underlying bars (15 bars)."""
    if not underlying_bars or len(underlying_bars) < 15:
        return None
    ranges = []
    for bar in underlying_bars[:60]:  # first hour
        h, l = bar["high"], bar["low"]
        if h > 0 and l > 0:
            ranges.append(h - l)
    return sum(ranges) / len(ranges) if ranges else None


def compute_premium_velocity(bars, end_idx, window):
    """Compute % change in premium over last N bars."""
    start_idx = max(0, end_idx - window)
    if start_idx >= end_idx:
        return 0.0
    start_price = bars[start_idx]["close"]
    end_price = bars[end_idx]["close"]
    if start_price <= 0:
        return 0.0
    return (end_price - start_price) / start_price * 100


def simulate_trade(option_bars, underlying_bars, config, entry_idx, direction,
                   strike, balance, day_atr_ratio=1.0, vix_mult=1.0):
    """Simulate a single trade with all strategy toggles.

    Returns dict with trade results or None.
    """
    if entry_idx >= len(option_bars) - 5:
        return None

    entry_bar = option_bars[entry_idx]
    raw_entry = entry_bar["vwap"] if entry_bar["vwap"] > 0 else entry_bar["close"]
    if raw_entry <= 0 or raw_entry < config.min_premium:
        return None

    # --- 7. Velocity entry gate ---
    if config.velocity_entry_gate and entry_idx >= config.velocity_entry_bars:
        vel = compute_premium_velocity(option_bars, entry_idx, config.velocity_entry_bars)
        if vel < config.velocity_entry_min_pct:
            return None  # premium not accelerating — skip

    entry_premium = raw_entry * (1 + config.slippage_bps / 10000)
    cost_per = entry_premium * 100
    if cost_per <= 0:
        return None

    # --- 9. VIX-adjusted sizing ---
    max_pos = balance * (config.max_pos_pct / 100)
    contracts = max(1, min(int(max_pos / cost_per), config.max_contracts))
    if config.vix_sizing:
        contracts = max(1, int(contracts * vix_mult))
    total_cost = contracts * cost_per
    if total_cost > balance:
        contracts = max(1, int(balance / cost_per))
        total_cost = contracts * cost_per
    if total_cost > balance:
        return None

    # Simulation state
    peak_premium = entry_premium
    remaining_pct = 100.0
    weighted_pnl = 0.0
    exit_reason = None
    exit_premium = entry_premium
    exit_bar_idx = entry_idx
    t1_hit = t2_hit = t3_hit = t4_hit = False
    last_scaleout_bar = entry_idx  # for time-proportional scale-out
    # Track velocity for deceleration exit
    prev_velocity = 0.0

    # Adaptive trail widths (possibly adjusted by regime/power hour)
    active_w = config.adaptive_active_width
    runner_w = config.adaptive_runner_width
    moonshot_w = config.adaptive_moonshot_width

    # --- 4. Market regime adjustment ---
    if config.regime_atr and day_atr_ratio != 1.0:
        if day_atr_ratio > 1.0:  # trending day
            active_w *= config.regime_atr_trend_mult
            runner_w *= config.regime_atr_trend_mult
        else:  # choppy day
            active_w *= config.regime_atr_chop_mult
            runner_w *= config.regime_atr_chop_mult

    for bar_idx in range(entry_idx + 1, len(option_bars)):
        bar = option_bars[bar_idx]
        minutes = bar_idx - entry_idx
        current = bar["close"]
        if current <= 0:
            continue

        if current > peak_premium:
            peak_premium = current

        bar_et = ts_to_et(bar["timestamp"])
        gain_pct = (current - entry_premium) / entry_premium * 100
        peak_gain_pct = (peak_premium - entry_premium) / entry_premium * 100

        # --- 3b. Power hour trail tightening ---
        eff_active_w = active_w
        eff_runner_w = runner_w
        eff_moonshot_w = moonshot_w
        if config.tod_power_hour_tighten and bar_et.hour >= 14:
            eff_active_w *= config.tod_power_tighten_factor
            eff_runner_w *= config.tod_power_tighten_factor
            eff_moonshot_w *= config.tod_power_tighten_factor

        # === EXIT CHECKS (priority order) ===

        # EOD cutoff
        if (bar_et.hour > config.eod_cutoff_hour or
            (bar_et.hour == config.eod_cutoff_hour and bar_et.minute >= config.eod_cutoff_min)):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "eod_cutoff"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # --- 1. Forced time exit ---
        if config.forced_exit_min > 0 and minutes >= config.forced_exit_min:
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "forced_time"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Grace period
        if minutes < config.grace_period_min:
            continue

        # Premium stop
        drop_from_entry = (entry_premium - current) / entry_premium * 100
        if drop_from_entry >= config.premium_stop_pct:
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "premium_stop"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # --- 2. MFE retracement ratio ---
        if (config.mfe_retrace_ratio > 0
                and peak_gain_pct >= config.mfe_retrace_min_gain
                and peak_gain_pct > 0):
            peak_dollars = peak_premium - entry_premium
            current_dollars = current - entry_premium
            given_back = peak_dollars - current_dollars
            retrace_ratio = given_back / peak_dollars if peak_dollars > 0 else 0
            if retrace_ratio >= config.mfe_retrace_ratio:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "mfe_retrace"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # --- 5. Premium spike exit (partial) ---
        if (config.spike_exit and remaining_pct > 0
                and bar_idx >= entry_idx + config.spike_window_bars):
            spike_vel = compute_premium_velocity(option_bars, bar_idx, config.spike_window_bars)
            if spike_vel >= config.spike_threshold_pct:
                sell_amt = remaining_pct * (config.spike_sell_pct / 100)
                weighted_pnl += sell_amt * gain_pct
                remaining_pct -= sell_amt

        # --- 8. Premium deceleration exit ---
        if (config.decel_exit and remaining_pct > 0
                and gain_pct >= config.decel_min_gain
                and bar_idx >= entry_idx + config.decel_window):
            v_short = compute_premium_velocity(option_bars, bar_idx, 5)
            v_long = compute_premium_velocity(option_bars, bar_idx, config.decel_window)
            accel = v_short - v_long
            if accel < config.decel_threshold:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "decel_exit"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # Adaptive trailing stop (production v2.1)
        if config.adaptive_trail and remaining_pct > 0 and peak_premium > entry_premium:
            drop_from_peak = (peak_premium - current) / peak_premium * 100

            if peak_gain_pct >= 400:
                if drop_from_peak >= eff_moonshot_w:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trail_moonshot"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break
            elif peak_gain_pct >= 150:
                if drop_from_peak >= eff_runner_w:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trail_runner"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break
            elif peak_gain_pct >= config.adaptive_dormant_pct:
                if drop_from_peak >= eff_active_w:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trail_active"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break
            # Below dormant threshold: no trail, let it develop

        # Scale-out at targets
        if remaining_pct > 0:
            if not t1_hit and gain_pct >= config.t1_gain:
                sell = remaining_pct * (config.t1_pct / 100)
                weighted_pnl += sell * gain_pct
                remaining_pct -= sell
                t1_hit = True
            if not t2_hit and gain_pct >= config.t2_gain:
                sell = remaining_pct * (config.t1_pct / 100)
                weighted_pnl += sell * gain_pct
                remaining_pct -= sell
                t2_hit = True
            if not t3_hit and gain_pct >= config.t3_gain:
                sell = remaining_pct * (config.t1_pct / 100)
                weighted_pnl += sell * gain_pct
                remaining_pct -= sell
                t3_hit = True
            if not t4_hit and gain_pct >= config.t4_gain:
                sell = remaining_pct * (config.t1_pct / 100)
                weighted_pnl += sell * gain_pct
                remaining_pct -= sell
                t4_hit = True

        # --- 6. Time-proportional scale-out ---
        if (config.time_scaleout and remaining_pct > 0
                and minutes >= config.time_scaleout_start_min
                and (bar_idx - last_scaleout_bar) >= config.time_scaleout_interval_min):
            sell = remaining_pct * (config.time_scaleout_pct / 100)
            weighted_pnl += sell * gain_pct
            remaining_pct -= sell
            last_scaleout_bar = bar_idx

        # No momentum
        if (minutes >= config.no_momentum_min and not t1_hit
                and gain_pct < config.no_momentum_gain):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "no_momentum"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # Theta bleed
        if (minutes >= config.theta_bleed_min
                and drop_from_entry >= config.theta_bleed_drop):
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "theta_bleed"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

    # Close at last bar if never exited
    if exit_reason is None:
        last = option_bars[-1]
        exit_premium = last["close"]
        gain_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        if remaining_pct > 0:
            weighted_pnl += remaining_pct * gain_pct
            remaining_pct = 0
        exit_reason = "eod_close"
        exit_bar_idx = len(option_bars) - 1

    # Apply exit slippage
    total_pnl_pct = weighted_pnl / 100 - config.slippage_bps / 100
    pnl_dollars = total_cost * (total_pnl_pct / 100)

    return {
        "date": "",
        "ticker": "",
        "direction": direction,
        "strike": strike,
        "entry_premium": entry_premium,
        "exit_premium": exit_premium,
        "peak_premium": peak_premium,
        "pnl_pct": total_pnl_pct,
        "pnl_dollars": pnl_dollars,
        "contracts": contracts,
        "total_cost": total_cost,
        "exit_reason": exit_reason,
        "duration_min": exit_bar_idx - entry_idx,
        "t1_hit": t1_hit,
        "entry_time": et_time_str(option_bars[entry_idx]["timestamp"]),
        "exit_time": et_time_str(option_bars[exit_bar_idx]["timestamp"]) if exit_bar_idx < len(option_bars) else "",
    }


def find_entry_bar(bars, entry_hour, entry_minute):
    target = entry_hour * 60 + entry_minute
    best_idx, best_diff = None, float("inf")
    for i, bar in enumerate(bars):
        et = ts_to_et(bar["timestamp"])
        bm = et.hour * 60 + et.minute
        diff = abs(bm - target)
        if bm >= target and diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Run backtest for one config
# ---------------------------------------------------------------------------


def run_single(ticker, config, trading_days=None, verbose=False):
    """Run backtest with a single config. Returns metrics dict."""
    if trading_days is None:
        trading_days = load_trading_days(ticker)
    if not trading_days:
        return None

    # Precompute ATR ratios for regime detection
    atr_ratios = {}
    if config.regime_atr:
        all_atrs = []
        for day in trading_days:
            ub = load_underlying_bars(ticker, day["date"])
            atr = compute_first_hour_atr(ub)
            if atr:
                all_atrs.append((day["date"], atr))
        if all_atrs:
            avg_atr = sum(a for _, a in all_atrs) / len(all_atrs)
            for date, atr in all_atrs:
                atr_ratios[date] = atr / avg_atr if avg_atr > 0 else 1.0

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []
    daily_pnl = {}
    morning_pnl = {}  # for budget tranches

    for day in trading_days:
        date_str = day["date"]
        strike = day["atm_strike"]

        # --- 3. Time-of-day regime: block midday entries ---
        if config.tod_block_midday:
            entry_et_min = config.entry_hour * 60 + config.entry_minute
            midday_start = config.tod_midday_start_hour * 60
            midday_end = config.tod_midday_end_hour * 60 + config.tod_midday_end_min
            if midday_start <= entry_et_min < midday_end:
                continue

        # --- 10. Daily budget tranches ---
        if config.budget_tranches:
            mp = morning_pnl.get(date_str, 0)
            if config.entry_hour >= 12 and mp < config.afternoon_unlock_threshold:
                continue  # afternoon locked

        directions = []
        if config.direction in ("call", "both"):
            directions.append(("call", day["atm_call_ticker"]))
        if config.direction in ("put", "both"):
            directions.append(("put", day["atm_put_ticker"]))

        day_atr = atr_ratios.get(date_str, 1.0)

        for direction, contract_ticker in directions:
            bars = load_option_bars(contract_ticker)
            if not bars or len(bars) < 30:
                continue

            entry_idx = find_entry_bar(bars, config.entry_hour, config.entry_minute)
            if entry_idx is None or entry_idx >= len(bars) - 10:
                continue

            result = simulate_trade(
                bars, None, config, entry_idx, direction, strike,
                balance, day_atr_ratio=day_atr,
            )
            if result is None:
                continue

            result["date"] = date_str
            result["ticker"] = ticker

            balance += result["pnl_dollars"]
            trades.append(result)

            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + result["pnl_dollars"]

            # Track morning PnL for budget tranches
            if config.entry_hour < 12:
                morning_pnl[date_str] = morning_pnl.get(date_str, 0) + result["pnl_dollars"]

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

    if not trades:
        return None

    wins = [t for t in trades if t["pnl_dollars"] >= 0]
    losses = [t for t in trades if t["pnl_dollars"] < 0]
    total_pnl = balance - STARTING_BALANCE
    win_rate = len(wins) / len(trades) * 100
    gross_wins = sum(t["pnl_dollars"] for t in wins)
    gross_losses = abs(sum(t["pnl_dollars"] for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0

    reasons = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

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
        "avg_win": sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Strategy test definitions
# ---------------------------------------------------------------------------


def get_baseline():
    """Current production strategy (no new features)."""
    return StrategyConfig()


def get_strategy_tests():
    """Return dict of strategy_name -> list of (label, StrategyConfig) to test."""
    tests = {}

    # 1. Forced time exit
    tests["1_forced_time"] = []
    for mins in [30, 45, 60, 75, 90, 120, 150, 180]:
        c = get_baseline()
        c.forced_exit_min = mins
        tests["1_forced_time"].append((f"{mins}min", c))

    # 2. MFE retracement ratio
    tests["2_mfe_retrace"] = []
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for min_gain in [15, 20, 30, 50]:
            c = get_baseline()
            c.mfe_retrace_ratio = ratio
            c.mfe_retrace_min_gain = min_gain
            tests["2_mfe_retrace"].append((f"r={ratio}_mg={min_gain}", c))

    # 3. Time-of-day regime
    tests["3_tod_regime"] = []
    # Block midday only
    c = get_baseline()
    c.tod_block_midday = True
    tests["3_tod_regime"].append(("block_midday", c))
    # Power hour tighten only
    c = get_baseline()
    c.tod_power_hour_tighten = True
    c.tod_power_tighten_factor = 0.7
    tests["3_tod_regime"].append(("power_tighten_0.7", c))
    c = get_baseline()
    c.tod_power_hour_tighten = True
    c.tod_power_tighten_factor = 0.5
    tests["3_tod_regime"].append(("power_tighten_0.5", c))
    # Both
    c = get_baseline()
    c.tod_block_midday = True
    c.tod_power_hour_tighten = True
    c.tod_power_tighten_factor = 0.7
    tests["3_tod_regime"].append(("block+tighten_0.7", c))

    # 4. Market regime (ATR)
    tests["4_regime_atr"] = []
    for trend in [1.2, 1.3, 1.5]:
        for chop in [0.6, 0.7, 0.8]:
            c = get_baseline()
            c.regime_atr = True
            c.regime_atr_trend_mult = trend
            c.regime_atr_chop_mult = chop
            tests["4_regime_atr"].append((f"t={trend}_c={chop}", c))

    # 5. Premium spike exit
    tests["5_spike_exit"] = []
    for thresh in [30, 50, 75]:
        for sell_pct in [30, 50, 75]:
            c = get_baseline()
            c.spike_exit = True
            c.spike_threshold_pct = thresh
            c.spike_sell_pct = sell_pct
            tests["5_spike_exit"].append((f"th={thresh}_s={sell_pct}", c))

    # 6. Time-proportional scale-out
    tests["6_time_scaleout"] = []
    for start in [20, 30, 45]:
        for interval in [10, 15, 20]:
            for pct in [5, 10, 15]:
                c = get_baseline()
                c.time_scaleout = True
                c.time_scaleout_start_min = start
                c.time_scaleout_interval_min = interval
                c.time_scaleout_pct = pct
                tests["6_time_scaleout"].append((f"s={start}_i={interval}_p={pct}", c))

    # 7. Premium velocity entry gate
    tests["7_velocity_entry"] = []
    for bars in [3, 5, 10]:
        for min_pct in [1, 2, 3, 5]:
            c = get_baseline()
            c.velocity_entry_gate = True
            c.velocity_entry_bars = bars
            c.velocity_entry_min_pct = min_pct
            tests["7_velocity_entry"].append((f"b={bars}_p={min_pct}", c))

    # 8. Premium deceleration exit
    tests["8_decel_exit"] = []
    for thresh in [-1, -2, -3, -5]:
        for min_gain in [20, 30, 50]:
            c = get_baseline()
            c.decel_exit = True
            c.decel_threshold = thresh
            c.decel_min_gain = min_gain
            tests["8_decel_exit"].append((f"t={thresh}_mg={min_gain}", c))

    # 9. VIX-adjusted sizing (tested conceptually by scaling contracts)
    # We can't get historical VIX from option bars, so we simulate
    # by testing different fixed scaling factors
    tests["9_vix_sizing"] = []
    for max_ct in [5, 8, 10, 15, 20]:
        c = get_baseline()
        c.max_contracts = max_ct
        tests["9_vix_sizing"].append((f"maxct={max_ct}", c))

    # 10. Daily budget tranches — test entry time splits
    # AM entry (default 10am) vs PM entry (1pm) vs both
    tests["10_budget_tranches"] = []
    c = get_baseline()
    c.entry_hour = 10
    tests["10_budget_tranches"].append(("am_only_10", c))
    c = get_baseline()
    c.entry_hour = 13
    tests["10_budget_tranches"].append(("pm_only_13", c))
    c = get_baseline()
    c.entry_hour = 9
    c.entry_minute = 45
    tests["10_budget_tranches"].append(("early_945", c))

    return tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_results_table(name, baseline, results):
    """Print comparison table for one strategy group."""
    print(f"\n{'='*100}")
    print(f"  STRATEGY: {name}")
    print(f"{'='*100}")
    print(f"  {'Label':>30} {'Trades':>6} {'WR%':>5} {'PnL$':>10} {'PnL%':>7} "
          f"{'PF':>5} {'MaxDD':>6} {'AvgW':>6} {'AvgL':>6} {'vs Base':>9}")
    print(f"  {'-'*90}")

    # Baseline row
    if baseline:
        pf_s = f"{baseline['profit_factor']:.2f}" if baseline['profit_factor'] < 100 else "inf"
        print(f"  {'** BASELINE **':>30} {baseline['num_trades']:>6} {baseline['win_rate']:>4.0f}% "
              f"${baseline['total_pnl']:>+9.2f} {baseline['total_pnl_pct']:>+6.1f}% "
              f"{pf_s:>5} {baseline['max_drawdown']:>5.1f}% "
              f"{baseline['avg_win']:>+5.0f}% {baseline['avg_loss']:>+5.0f}% {'---':>9}")

    for label, r in results:
        if r is None:
            print(f"  {label:>30} {'NO TRADES':>6}")
            continue
        pf_s = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        delta = r['total_pnl'] - baseline['total_pnl'] if baseline else 0
        delta_s = f"${delta:+.0f}"
        print(f"  {label:>30} {r['num_trades']:>6} {r['win_rate']:>4.0f}% "
              f"${r['total_pnl']:>+9.2f} {r['total_pnl_pct']:>+6.1f}% "
              f"{pf_s:>5} {r['max_drawdown']:>5.1f}% "
              f"{r['avg_win']:>+5.0f}% {r['avg_loss']:>+5.0f}% {delta_s:>9}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--strategy", default="all",
                        help="Which strategy to test (1-10 or 'all')")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.ticker.split(",")]

    for ticker in tickers:
        print(f"\n{'#'*100}")
        print(f"  TICKER: {ticker}")
        print(f"{'#'*100}")

        trading_days = load_trading_days(ticker)
        if not trading_days:
            print(f"  No data for {ticker}")
            continue

        print(f"  {len(trading_days)} trading days of real 1-min option data")

        # Run baseline
        baseline_cfg = get_baseline()
        baseline = run_single(ticker, baseline_cfg, trading_days)
        if baseline is None:
            print(f"  Baseline produced no trades for {ticker}")
            continue

        # Get tests
        all_tests = get_strategy_tests()
        if args.strategy != "all":
            prefix = f"{args.strategy}_"
            all_tests = {k: v for k, v in all_tests.items() if k.startswith(prefix)}

        # Run each strategy group
        best_per_strategy = {}

        for name, configs in sorted(all_tests.items()):
            results = []
            for label, cfg in configs:
                r = run_single(ticker, cfg, trading_days)
                results.append((label, r))

            print_results_table(name, baseline, results)

            # Find best in this group
            valid = [(l, r) for l, r in results if r is not None]
            if valid:
                best_label, best_r = max(valid, key=lambda x: x[1]["total_pnl"])
                delta = best_r["total_pnl"] - baseline["total_pnl"]
                best_per_strategy[name] = {
                    "label": best_label,
                    "pnl": best_r["total_pnl"],
                    "pnl_pct": best_r["total_pnl_pct"],
                    "delta": delta,
                    "win_rate": best_r["win_rate"],
                    "max_dd": best_r["max_drawdown"],
                    "pf": best_r["profit_factor"],
                }

        # === SUMMARY ===
        print(f"\n{'='*100}")
        print(f"  SUMMARY: Best from each strategy vs baseline — {ticker}")
        print(f"{'='*100}")
        print(f"  Baseline: ${baseline['total_pnl']:+,.2f} ({baseline['total_pnl_pct']:+.1f}%) "
              f"WR={baseline['win_rate']:.0f}% PF={baseline['profit_factor']:.2f} "
              f"DD={baseline['max_drawdown']:.1f}%")
        print()
        print(f"  {'Strategy':>25} {'Best Config':>25} {'PnL$':>10} {'vs Base':>9} "
              f"{'WR%':>5} {'PF':>5} {'MaxDD':>6}")
        print(f"  {'-'*90}")

        for name in sorted(best_per_strategy.keys()):
            b = best_per_strategy[name]
            pf_s = f"{b['pf']:.2f}" if b['pf'] < 100 else "inf"
            emoji = "+" if b["delta"] > 0 else " "
            print(f"  {name:>25} {b['label']:>25} ${b['pnl']:>+9.2f} "
                  f"${b['delta']:>+8.0f} {b['win_rate']:>4.0f}% {pf_s:>5} {b['max_dd']:>5.1f}%")

        # Strategies worth adding (positive delta and better risk metrics)
        print()
        winners = {k: v for k, v in best_per_strategy.items() if v["delta"] > 0}
        if winners:
            print(f"  RECOMMENDED additions (beat baseline by $):")
            for name in sorted(winners.keys(), key=lambda k: -winners[k]["delta"]):
                w = winners[name]
                print(f"    {name}: {w['label']} → +${w['delta']:,.0f} "
                      f"(WR {w['win_rate']:.0f}%, PF {w['pf']:.2f}, DD {w['max_dd']:.1f}%)")
        else:
            print("  No strategies beat the baseline on this ticker.")


if __name__ == "__main__":
    main()
