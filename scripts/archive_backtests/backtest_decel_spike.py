"""Backtest: Deceleration Exit + Spike Partial Exit on historical 1-min bars.

Tests each strategy independently and in combination with the profit retrace
gate (already implemented) to find the best additions.

Strategies:
  1. Premium Deceleration Exit — exit when short-term velocity drops well below
     long-term velocity (momentum collapse). Catches trades bleeding theta.
  2. Premium Spike Exit (partial) — sell a portion when premium spikes rapidly,
     let the rest ride. Locks in windfall gains on explosive moves.

Also tests combos:
  - Decel only
  - Spike only
  - Decel + Spike
  - Decel + Profit Retrace (already live)
  - Spike + Profit Retrace
  - All three together

Usage:
    python scripts/backtest_decel_spike.py
    python scripts/backtest_decel_spike.py --ticker SPY,QQQ,IWM
"""

import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
STARTING_BALANCE = 5000.0


def load_trading_days(ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT date, open_price, close_price, high_price, low_price,
               atm_call_ticker, atm_put_ticker, atm_strike,
               call_bars, put_bars
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


def ts_to_et(timestamp_ms):
    dt = datetime.utcfromtimestamp(timestamp_ms / 1000)
    return dt - timedelta(hours=4)


def et_time_str(timestamp_ms):
    return ts_to_et(timestamp_ms).strftime("%H:%M")


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


def compute_velocity(bars, end_idx, window):
    start_idx = max(0, end_idx - window)
    if start_idx >= end_idx:
        return 0.0
    start_price = bars[start_idx]["close"]
    end_price = bars[end_idx]["close"]
    if start_price <= 0:
        return 0.0
    return (end_price - start_price) / start_price * 100


@dataclass
class Config:
    name: str
    # Adaptive trail (production v2.1)
    adaptive_dormant_pct: float = 40.0
    adaptive_active_width: float = 35.0
    adaptive_runner_width: float = 45.0
    adaptive_moonshot_width: float = 30.0
    # Profit retrace (just deployed)
    profit_retrace: bool = True
    profit_retrace_pct: float = 35.0
    profit_retrace_min_gain: float = 10.0
    # Deceleration exit
    decel_exit: bool = False
    decel_short_window: int = 5       # short-term velocity window (bars)
    decel_long_window: int = 15       # long-term velocity window (bars)
    decel_threshold: float = -3.0     # short - long must be below this
    decel_min_gain: float = 5.0       # only after trade was up at least this %
    decel_min_hold: int = 10          # minimum bars held before decel can fire
    # Spike exit (partial)
    spike_exit: bool = False
    spike_threshold_pct: float = 30.0   # premium spike % in window
    spike_window_bars: int = 3          # how many bars for the spike
    spike_sell_pct: float = 50.0        # sell this % of remaining position
    spike_min_gain: float = 20.0        # only spike-exit if up at least this %
    # Common
    premium_stop_pct: float = 50.0
    grace_period_min: int = 8
    slippage_bps: float = 50.0
    max_pos_pct: float = 33.3
    min_premium: float = 0.25
    max_contracts: int = 20
    t1_pct: float = 20.0
    t1_gain: float = 20.0
    t2_gain: float = 50.0
    t3_gain: float = 100.0
    t4_gain: float = 200.0
    no_momentum_min: int = 45
    theta_bleed_min: int = 45
    theta_bleed_drop: float = 30.0
    eod_cutoff_hour: int = 15
    eod_cutoff_min: int = 45


def simulate_trade(option_bars, config, entry_idx, direction, strike, balance):
    if entry_idx >= len(option_bars) - 5:
        return None

    entry_bar = option_bars[entry_idx]
    raw_entry = entry_bar["vwap"] if entry_bar["vwap"] > 0 else entry_bar["close"]
    if raw_entry <= 0 or raw_entry < config.min_premium:
        return None

    entry_premium = raw_entry * (1 + config.slippage_bps / 10000)
    cost_per = entry_premium * 100
    if cost_per <= 0:
        return None

    max_pos = balance * (config.max_pos_pct / 100)
    contracts = max(1, min(int(max_pos / cost_per), config.max_contracts))
    total_cost = contracts * cost_per
    if total_cost > balance:
        contracts = max(1, int(balance / cost_per))
        total_cost = contracts * cost_per
    if total_cost > balance:
        return None

    peak_premium = entry_premium
    remaining_pct = 100.0
    weighted_pnl = 0.0
    exit_reason = None
    exit_premium = entry_premium
    exit_bar_idx = entry_idx
    t1_hit = t2_hit = t3_hit = t4_hit = False
    spike_fired = False

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

        # === PROFIT RETRACE (deployed) ===
        if (config.profit_retrace and remaining_pct > 0
                and peak_gain_pct >= config.profit_retrace_min_gain
                and peak_gain_pct < config.adaptive_dormant_pct
                and peak_premium > entry_premium):
            profit_at_peak = peak_premium - entry_premium
            profit_given_back = profit_at_peak - (current - entry_premium)
            retrace_ratio = (profit_given_back / profit_at_peak * 100) if profit_at_peak > 0 else 0
            if retrace_ratio >= config.profit_retrace_pct:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
                exit_reason = "profit_retrace"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # === SPIKE EXIT (partial) ===
        if (config.spike_exit and remaining_pct > 0 and not spike_fired
                and bar_idx >= entry_idx + config.spike_window_bars
                and gain_pct >= config.spike_min_gain):
            spike_vel = compute_velocity(option_bars, bar_idx, config.spike_window_bars)
            if spike_vel >= config.spike_threshold_pct:
                sell_amt = remaining_pct * (config.spike_sell_pct / 100)
                weighted_pnl += sell_amt * gain_pct
                remaining_pct -= sell_amt
                spike_fired = True
                # Don't break — let the rest ride

        # === DECELERATION EXIT ===
        if (config.decel_exit and remaining_pct > 0
                and minutes >= config.decel_min_hold
                and bar_idx >= entry_idx + config.decel_long_window
                and gain_pct >= config.decel_min_gain):
            v_short = compute_velocity(option_bars, bar_idx, config.decel_short_window)
            v_long = compute_velocity(option_bars, bar_idx, config.decel_long_window)
            accel = v_short - v_long
            if accel < config.decel_threshold:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
                exit_reason = "decel_exit"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # Adaptive trailing stop
        if remaining_pct > 0 and peak_premium > entry_premium:
            drop_from_peak = (peak_premium - current) / peak_premium * 100

            if peak_gain_pct >= 400:
                if drop_from_peak >= config.adaptive_moonshot_width:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trail_moonshot"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break
            elif peak_gain_pct >= 150:
                if drop_from_peak >= config.adaptive_runner_width:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trail_runner"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break
            elif peak_gain_pct >= config.adaptive_dormant_pct:
                if drop_from_peak >= config.adaptive_active_width:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                    exit_reason = "trail_active"
                    exit_premium = current
                    exit_bar_idx = bar_idx
                    break

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

        # No momentum
        if (minutes >= config.no_momentum_min and not t1_hit and gain_pct < 0):
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

    if exit_reason is None:
        last = option_bars[-1]
        exit_premium = last["close"]
        gain_pct = (exit_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0
        if remaining_pct > 0:
            weighted_pnl += remaining_pct * gain_pct
            remaining_pct = 0
        exit_reason = "eod_close"
        exit_bar_idx = len(option_bars) - 1

    total_pnl_pct = weighted_pnl / 100 - config.slippage_bps / 100
    pnl_dollars = total_cost * (total_pnl_pct / 100)

    return {
        "date": "", "ticker": "", "direction": direction, "strike": strike,
        "entry_premium": entry_premium, "exit_premium": exit_premium,
        "peak_premium": peak_premium, "pnl_pct": total_pnl_pct,
        "pnl_dollars": pnl_dollars, "contracts": contracts,
        "total_cost": total_cost, "exit_reason": exit_reason,
        "duration_min": exit_bar_idx - entry_idx,
        "spike_fired": spike_fired,
        "peak_gain_pct": (peak_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0,
    }


def run_backtest(ticker, config, trading_days=None):
    if trading_days is None:
        trading_days = load_trading_days(ticker)
    if not trading_days:
        return None

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []

    for day in trading_days:
        for direction, ct in [("call", day["atm_call_ticker"]),
                               ("put", day["atm_put_ticker"])]:
            bars = load_option_bars(ct)
            if not bars or len(bars) < 30:
                continue
            entry_idx = find_entry_bar(bars, 10, 0)
            if entry_idx is None or entry_idx >= len(bars) - 10:
                continue
            result = simulate_trade(bars, config, entry_idx, direction, day["atm_strike"], balance)
            if result is None:
                continue
            result["date"] = day["date"]
            result["ticker"] = ticker
            balance += result["pnl_dollars"]
            trades.append(result)
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
    avg_dur = sum(t["duration_min"] for t in trades) / len(trades)

    reasons = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

    spike_count = sum(1 for t in trades if t.get("spike_fired"))

    return {
        "trades": trades, "balance": balance, "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl / STARTING_BALANCE * 100,
        "num_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate, "profit_factor": pf, "max_drawdown": max_drawdown,
        "avg_win": sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0,
        "avg_loss": sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0,
        "avg_duration": avg_dur, "reasons": reasons, "spike_count": spike_count,
    }


def main():
    tickers = ["SPY"]
    if len(sys.argv) > 1 and sys.argv[1] == "--ticker":
        tickers = sys.argv[2].split(",")
    elif len(sys.argv) > 1:
        tickers = sys.argv[1].split(",")

    print("=" * 120)
    print("BACKTEST: Deceleration Exit + Spike Partial Exit (with Profit Retrace baseline)")
    print("=" * 120)

    for ticker in tickers:
        trading_days = load_trading_days(ticker)
        if not trading_days:
            print(f"\n  No data for {ticker}")
            continue

        print(f"\n{'='*120}")
        print(f"  {ticker} — {len(trading_days)} trading days")
        print(f"{'='*120}")

        # === BASELINE: Current production (profit retrace + adaptive trail) ===
        baseline = Config(name="BASELINE (retrace+adaptive)")
        baseline_r = run_backtest(ticker, baseline, trading_days)
        if not baseline_r:
            continue

        # === INDIVIDUAL STRATEGY TESTS ===
        configs = []

        # --- Decel exit parameter sweep ---
        for short_w in [3, 5, 8]:
            for long_w in [10, 15, 20]:
                for thresh in [-2.0, -3.0, -5.0]:
                    for min_gain in [5, 10, 20]:
                        for min_hold in [8, 15]:
                            c = Config(
                                name=f"DECEL s={short_w} l={long_w} t={thresh} mg={min_gain} h={min_hold}",
                                decel_exit=True,
                                decel_short_window=short_w,
                                decel_long_window=long_w,
                                decel_threshold=thresh,
                                decel_min_gain=min_gain,
                                decel_min_hold=min_hold,
                            )
                            configs.append(c)

        # --- Spike exit parameter sweep ---
        for spike_thresh in [20, 30, 50]:
            for spike_win in [3, 5]:
                for sell_pct in [30, 50, 75]:
                    for min_gain in [15, 20, 30]:
                        c = Config(
                            name=f"SPIKE th={spike_thresh} w={spike_win} s={sell_pct} mg={min_gain}",
                            spike_exit=True,
                            spike_threshold_pct=spike_thresh,
                            spike_window_bars=spike_win,
                            spike_sell_pct=sell_pct,
                            spike_min_gain=min_gain,
                        )
                        configs.append(c)

        # Run all
        results = []
        for c in configs:
            r = run_backtest(ticker, c, trading_days)
            if r:
                r["config"] = c
                results.append(r)

        # Sort by total P&L
        results.sort(key=lambda x: x["total_pnl"], reverse=True)

        # Split into decel and spike results
        decel_results = [r for r in results if r["config"].decel_exit]
        spike_results = [r for r in results if r["config"].spike_exit]

        b = baseline_r
        header = (f"  {'Strategy':<60} {'PnL':>10} {'WR%':>7} {'PF':>6} "
                  f"{'Trades':>7} {'AvgDur':>7} {'MaxDD':>7} {'Delta':>10}")
        sep = f"  {'-'*60} {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*10}"

        def print_row(r, is_baseline=False):
            delta = r["total_pnl"] - b["total_pnl"]
            name = r["config"].name if not is_baseline else ">>> BASELINE (retrace+adaptive) <<<"
            spike_info = f" [{r['spike_count']}sp]" if r.get("spike_count", 0) > 0 else ""
            print(f"  {name:<60} "
                  f"${r['total_pnl']:>+9,.0f} "
                  f"{r['win_rate']:>6.1f}% "
                  f"{r['profit_factor']:>5.2f} "
                  f"{r['num_trades']:>6} "
                  f"{r['avg_duration']:>6.0f}m "
                  f"{r['max_drawdown']:>6.1f}% "
                  f"${delta:>+9,.0f}{spike_info}")

        # --- TOP DECEL CONFIGS ---
        print(f"\n  --- Top 15 Deceleration Exit Configs ---")
        print(header)
        print(sep)
        baseline_r["config"] = baseline
        print_row(baseline_r, is_baseline=True)
        print()
        for r in decel_results[:15]:
            print_row(r)

        # --- TOP SPIKE CONFIGS ---
        print(f"\n  --- Top 15 Spike Partial Exit Configs ---")
        print(header)
        print(sep)
        print_row(baseline_r, is_baseline=True)
        print()
        for r in spike_results[:15]:
            print_row(r)

        # --- COMBOS: best decel + best spike + both ---
        best_decel = decel_results[0]["config"] if decel_results else None
        best_spike = spike_results[0]["config"] if spike_results else None

        if best_decel and best_spike:
            print(f"\n  --- Combination Tests ---")
            print(header)
            print(sep)
            print_row(baseline_r, is_baseline=True)
            print()

            combos = []

            # Best decel alone (already have it)
            combos.append(("Best DECEL alone", decel_results[0]))

            # Best spike alone (already have it)
            combos.append(("Best SPIKE alone", spike_results[0]))

            # Best decel + best spike
            combo = Config(
                name=f"DECEL+SPIKE best combo",
                decel_exit=True,
                decel_short_window=best_decel.decel_short_window,
                decel_long_window=best_decel.decel_long_window,
                decel_threshold=best_decel.decel_threshold,
                decel_min_gain=best_decel.decel_min_gain,
                decel_min_hold=best_decel.decel_min_hold,
                spike_exit=True,
                spike_threshold_pct=best_spike.spike_threshold_pct,
                spike_window_bars=best_spike.spike_window_bars,
                spike_sell_pct=best_spike.spike_sell_pct,
                spike_min_gain=best_spike.spike_min_gain,
            )
            r = run_backtest(ticker, combo, trading_days)
            if r:
                r["config"] = combo
                combos.append(("DECEL + SPIKE", r))

            # Without profit retrace (to see if retrace is helping or hurting)
            combo_no_retrace = Config(
                name=f"DECEL+SPIKE (no retrace)",
                profit_retrace=False,
                decel_exit=True,
                decel_short_window=best_decel.decel_short_window,
                decel_long_window=best_decel.decel_long_window,
                decel_threshold=best_decel.decel_threshold,
                decel_min_gain=best_decel.decel_min_gain,
                decel_min_hold=best_decel.decel_min_hold,
                spike_exit=True,
                spike_threshold_pct=best_spike.spike_threshold_pct,
                spike_window_bars=best_spike.spike_window_bars,
                spike_sell_pct=best_spike.spike_sell_pct,
                spike_min_gain=best_spike.spike_min_gain,
            )
            r = run_backtest(ticker, combo_no_retrace, trading_days)
            if r:
                r["config"] = combo_no_retrace
                combos.append(("DECEL+SPIKE (no retrace)", r))

            for label, r in combos:
                if isinstance(r, dict) and "total_pnl" in r:
                    print_row(r)

            # Exit reason breakdown for best combo
            if len(combos) >= 3 and isinstance(combos[2][1], dict):
                print(f"\n  Exit reasons:")
                print(f"    Baseline: {baseline_r['reasons']}")
                print(f"    Best decel: {decel_results[0]['reasons']}")
                print(f"    Best spike: {spike_results[0]['reasons']}")
                combo_r = combos[2][1]
                if "reasons" in combo_r:
                    print(f"    DECEL+SPIKE: {combo_r['reasons']}")

        # Show best params
        if best_decel:
            print(f"\n  Best DECEL params: short_window={best_decel.decel_short_window}, "
                  f"long_window={best_decel.decel_long_window}, "
                  f"threshold={best_decel.decel_threshold}, "
                  f"min_gain={best_decel.decel_min_gain}%, "
                  f"min_hold={best_decel.decel_min_hold}")
        if best_spike:
            print(f"  Best SPIKE params: threshold={best_spike.spike_threshold_pct}%, "
                  f"window={best_spike.spike_window_bars}, "
                  f"sell={best_spike.spike_sell_pct}%, "
                  f"min_gain={best_spike.spike_min_gain}%")

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    main()
