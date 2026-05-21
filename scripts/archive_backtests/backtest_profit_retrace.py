"""Backtest: Profit-Based Retracement vs Current System.

Compares the current adaptive trail (which measures drop from PEAK PRICE)
against a profit-based retracement (which measures % of PROFIT given back).

Current system (adaptive trail):
  - Buy at $1.00, peak $1.50, 50% drop from peak → exit at $0.75
  - Gives back ALL profit AND loses $0.25 of capital

Proposed (profit-based retracement):
  - Buy at $1.00, peak $1.50, 35% of profit retraced → exit at $1.325
  - Locks in $0.325 profit (65% of the move)

Tests multiple retracement percentages: 25%, 30%, 35%, 40%, 50%, 60%
Also tests minimum gain thresholds: 10%, 15%, 20%, 30%

Usage:
    python scripts/backtest_profit_retrace.py
    python scripts/backtest_profit_retrace.py --ticker SPY
"""

import os
import sqlite3
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "journal", "historical_0dte.db")
STARTING_BALANCE = 5000.0


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class TrailConfig:
    """Trail configuration for comparison."""
    name: str
    # Adaptive trail (current production)
    adaptive_trail: bool = True
    adaptive_dormant_pct: float = 40.0
    adaptive_active_width: float = 35.0
    adaptive_runner_width: float = 45.0
    adaptive_moonshot_width: float = 30.0
    # Profit-based retracement (NEW)
    profit_retrace: bool = False
    profit_retrace_pct: float = 35.0    # exit when this % of profit is given back
    profit_retrace_min_gain: float = 10.0  # only activate after this % gain from entry
    # Common
    premium_stop_pct: float = 50.0
    grace_period_min: int = 8
    slippage_bps: float = 50.0
    max_pos_pct: float = 33.3
    min_premium: float = 0.25
    max_contracts: int = 20
    # Scale-out
    t1_pct: float = 20.0
    t1_gain: float = 20.0
    t2_gain: float = 50.0
    t3_gain: float = 100.0
    t4_gain: float = 200.0
    # Time-based
    no_momentum_min: int = 45
    theta_bleed_min: int = 45
    theta_bleed_drop: float = 30.0
    eod_cutoff_hour: int = 15
    eod_cutoff_min: int = 45


def simulate_trade(option_bars, config, entry_idx, direction, strike, balance):
    """Simulate a single trade."""
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

        # Premium stop (hard stop from entry)
        drop_from_entry = (entry_premium - current) / entry_premium * 100
        if drop_from_entry >= config.premium_stop_pct:
            if remaining_pct > 0:
                weighted_pnl += remaining_pct * gain_pct
                remaining_pct = 0
            exit_reason = "premium_stop"
            exit_premium = current
            exit_bar_idx = bar_idx
            break

        # === PROFIT-BASED RETRACEMENT (the new gate) ===
        if (config.profit_retrace and remaining_pct > 0
                and peak_gain_pct >= config.profit_retrace_min_gain
                and peak_premium > entry_premium):
            profit_at_peak = peak_premium - entry_premium
            profit_now = current - entry_premium
            profit_given_back = profit_at_peak - profit_now
            retrace_pct = (profit_given_back / profit_at_peak * 100) if profit_at_peak > 0 else 0

            if retrace_pct >= config.profit_retrace_pct:
                if remaining_pct > 0:
                    weighted_pnl += remaining_pct * gain_pct
                    remaining_pct = 0
                exit_reason = "profit_retrace"
                exit_premium = current
                exit_bar_idx = bar_idx
                break

        # === ADAPTIVE TRAILING STOP (current production) ===
        if config.adaptive_trail and remaining_pct > 0 and peak_premium > entry_premium:
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
        "peak_gain_pct": (peak_premium - entry_premium) / entry_premium * 100 if entry_premium > 0 else 0,
        "entry_time": et_time_str(option_bars[entry_idx]["timestamp"]),
        "exit_time": et_time_str(option_bars[exit_bar_idx]["timestamp"]) if exit_bar_idx < len(option_bars) else "",
    }


def run_backtest(ticker, config, trading_days=None):
    """Run backtest with given config."""
    if trading_days is None:
        trading_days = load_trading_days(ticker)
    if not trading_days:
        return None

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    max_drawdown = 0.0
    trades = []

    for day in trading_days:
        date_str = day["date"]
        strike = day["atm_strike"]

        for direction, contract_ticker in [("call", day["atm_call_ticker"]),
                                            ("put", day["atm_put_ticker"])]:
            bars = load_option_bars(contract_ticker)
            if not bars or len(bars) < 30:
                continue

            entry_idx = find_entry_bar(bars, 10, 0)
            if entry_idx is None or entry_idx >= len(bars) - 10:
                continue

            result = simulate_trade(bars, config, entry_idx, direction, strike, balance)
            if result is None:
                continue

            result["date"] = date_str
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

    reasons = {}
    for t in trades:
        reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

    avg_dur = sum(t["duration_min"] for t in trades) / len(trades)

    # How much MFE is captured (profit_captured / max_possible)
    mfe_capture_ratios = []
    for t in trades:
        if t["peak_gain_pct"] > 0:
            captured = t["pnl_pct"]
            possible = t["peak_gain_pct"]
            mfe_capture_ratios.append(captured / possible * 100 if possible > 0 else 0)
    avg_mfe_capture = sum(mfe_capture_ratios) / len(mfe_capture_ratios) if mfe_capture_ratios else 0

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
        "avg_duration": avg_dur,
        "avg_mfe_capture": avg_mfe_capture,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tickers = ["SPY"]
    if len(sys.argv) > 1 and sys.argv[1] == "--ticker":
        tickers = sys.argv[2].split(",")
    elif len(sys.argv) > 1:
        tickers = sys.argv[1].split(",")

    print("=" * 100)
    print("BACKTEST: Profit-Based Retracement vs Current Adaptive Trail")
    print("=" * 100)
    print()
    print("Current system: adaptive trail measures % drop from PEAK PRICE")
    print("  → Buy $1.00, peak $1.50, 35% drop from peak → exit at $0.975 (lose $0.025)")
    print()
    print("Proposed: profit-based retracement measures % of PROFIT given back")
    print("  → Buy $1.00, peak $1.50, 35% of $0.50 profit retraced → exit at $1.325 (keep $0.325)")
    print()

    for ticker in tickers:
        trading_days = load_trading_days(ticker)
        if not trading_days:
            print(f"\n  No data for {ticker}")
            continue

        print(f"\n{'='*100}")
        print(f"  {ticker} — {len(trading_days)} trading days")
        print(f"{'='*100}")

        # 1. BASELINE: Current production (adaptive trail only)
        baseline_cfg = TrailConfig(name="BASELINE (adaptive trail)")
        baseline = run_backtest(ticker, baseline_cfg, trading_days)
        if not baseline:
            print(f"  No trades for {ticker}")
            continue

        # 2. PROFIT RETRACE ONLY (no adaptive trail) — test various thresholds
        retrace_configs = []
        for retrace_pct in [25, 30, 35, 40, 50, 60]:
            for min_gain in [10, 15, 20, 30]:
                cfg = TrailConfig(
                    name=f"Profit retrace {retrace_pct}% (min +{min_gain}%)",
                    adaptive_trail=False,  # disable adaptive trail
                    profit_retrace=True,
                    profit_retrace_pct=retrace_pct,
                    profit_retrace_min_gain=min_gain,
                )
                retrace_configs.append(cfg)

        # 3. HYBRID: Profit retrace + adaptive trail together
        hybrid_configs = []
        for retrace_pct in [25, 30, 35, 40, 50]:
            for min_gain in [10, 15, 20]:
                cfg = TrailConfig(
                    name=f"HYBRID retrace {retrace_pct}% (min +{min_gain}%) + adaptive",
                    adaptive_trail=True,  # keep adaptive trail
                    profit_retrace=True,
                    profit_retrace_pct=retrace_pct,
                    profit_retrace_min_gain=min_gain,
                )
                hybrid_configs.append(cfg)

        all_configs = retrace_configs + hybrid_configs
        results = []
        for cfg in all_configs:
            r = run_backtest(ticker, cfg, trading_days)
            if r:
                r["config"] = cfg
                results.append(r)

        # Sort by total P&L
        results.sort(key=lambda x: x["total_pnl"], reverse=True)

        # Print comparison table
        print(f"\n  {'Strategy':<55} {'PnL':>10} {'WR%':>7} {'PF':>6} {'Trades':>7} {'AvgDur':>7} {'MFE%':>7} {'MaxDD':>7}")
        print(f"  {'-'*55} {'-'*10} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

        # Print baseline first
        b = baseline
        print(f"  {'>>> BASELINE (adaptive trail) <<<':<55} "
              f"${b['total_pnl']:>+9,.0f} "
              f"{b['win_rate']:>6.1f}% "
              f"{b['profit_factor']:>5.2f} "
              f"{b['num_trades']:>6} "
              f"{b['avg_duration']:>6.0f}m "
              f"{b['avg_mfe_capture']:>6.1f}% "
              f"{b['max_drawdown']:>6.1f}%")
        print()

        # Print top 20 results
        printed = 0
        for r in results:
            cfg = r["config"]
            delta = r["total_pnl"] - b["total_pnl"]
            marker = " *** BEST" if r == results[0] else ""
            is_hybrid = "HYBRID" in cfg.name

            print(f"  {cfg.name:<55} "
                  f"${r['total_pnl']:>+9,.0f} "
                  f"{r['win_rate']:>6.1f}% "
                  f"{r['profit_factor']:>5.2f} "
                  f"{r['num_trades']:>6} "
                  f"{r['avg_duration']:>6.0f}m "
                  f"{r['avg_mfe_capture']:>6.1f}% "
                  f"{r['max_drawdown']:>6.1f}%"
                  f"  ({delta:>+,.0f} vs base){marker}")

            printed += 1
            if printed >= 25:
                break

        # Show exit reason breakdown for top 3
        print(f"\n  --- Exit Reason Breakdown (top 3 vs baseline) ---")
        print(f"\n  BASELINE: {baseline['reasons']}")
        for r in results[:3]:
            print(f"  {r['config'].name}: {r['reasons']}")

        # Show the user's specific ask: 35% profit retrace
        print(f"\n  --- Your Proposed Config: 35% Profit Retrace ---")
        for r in results:
            cfg = r["config"]
            if cfg.profit_retrace_pct == 35 and cfg.profit_retrace_min_gain == 10:
                delta = r["total_pnl"] - b["total_pnl"]
                hybrid_str = "(with adaptive trail)" if cfg.adaptive_trail else "(standalone)"
                print(f"  {hybrid_str}: PnL=${r['total_pnl']:>+,.0f} ({delta:>+,.0f} vs base), "
                      f"WR={r['win_rate']:.1f}%, PF={r['profit_factor']:.2f}, "
                      f"MFE capture={r['avg_mfe_capture']:.1f}%, "
                      f"MaxDD={r['max_drawdown']:.1f}%")
                print(f"    Exit reasons: {r['reasons']}")

        # Detailed trade comparison for 35% retrace vs baseline
        print(f"\n  --- Trade-by-Trade Comparison: 35% Profit Retrace vs Baseline ---")
        target_r = None
        for r in results:
            cfg = r["config"]
            if cfg.profit_retrace_pct == 35 and cfg.profit_retrace_min_gain == 10 and cfg.adaptive_trail:
                target_r = r
                break
        if not target_r:
            for r in results:
                cfg = r["config"]
                if cfg.profit_retrace_pct == 35 and cfg.profit_retrace_min_gain == 10:
                    target_r = r
                    break

        if target_r and baseline:
            # Compare trades where they diverge
            bt = baseline["trades"]
            tt = target_r["trades"]
            better = worse = same = 0
            big_diffs = []
            for i in range(min(len(bt), len(tt))):
                diff = tt[i]["pnl_dollars"] - bt[i]["pnl_dollars"]
                if abs(diff) < 1:
                    same += 1
                elif diff > 0:
                    better += 1
                else:
                    worse += 1
                if abs(diff) > 20:
                    big_diffs.append((bt[i]["date"], bt[i]["direction"],
                                     bt[i]["pnl_dollars"], tt[i]["pnl_dollars"],
                                     diff, bt[i]["exit_reason"], tt[i]["exit_reason"],
                                     bt[i]["peak_gain_pct"]))

            print(f"  Trades improved: {better}, worsened: {worse}, same: {same}")
            if big_diffs:
                big_diffs.sort(key=lambda x: x[4], reverse=True)
                print(f"\n  Top differences (>$20):")
                print(f"  {'Date':<12} {'Dir':<5} {'Base$':>8} {'New$':>8} {'Diff$':>8} {'Base Exit':<16} {'New Exit':<16} {'Peak%':>6}")
                for d in big_diffs[:20]:
                    print(f"  {d[0]:<12} {d[1]:<5} ${d[2]:>+7,.0f} ${d[3]:>+7,.0f} ${d[4]:>+7,.0f} "
                          f"{d[5]:<16} {d[6]:<16} +{d[7]:.0f}%")

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    main()
